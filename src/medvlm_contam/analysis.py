"""Post-hoc analysis of audit JSONL artifacts.

Consumes the per-example JSONL files written by :func:`medvlm_contam.audit.run_audit`
across (benchmark, model) cells and produces:

- :func:`load_audit_jsonl`: parse one JSONL into a list of records.
- :func:`build_heatmap`: ``(benchmark, model) -> exchangeability p-value``
  table built from the ``.summary.json`` sidecars produced by the driver.
- :func:`flag_examples`: per-example flagging given detector thresholds.
- :func:`aggregate_flags`: union flags across detectors for the
  clean-benchmark step.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


@dataclass(frozen=True)
class AuditRecord:
    benchmark: str
    model: str
    example_id: str
    n_answer_tokens: int
    sum_logprob: float
    mean_logprob: float
    mink_pp_score: Optional[float]
    metadata: dict


def load_audit_jsonl(path: Path) -> list[AuditRecord]:
    """Parse one ``run_audit`` JSONL into :class:`AuditRecord` objects."""
    path = Path(path)
    out: list[AuditRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out.append(
                AuditRecord(
                    benchmark=r["benchmark"],
                    model=r["model"],
                    example_id=r["example_id"],
                    n_answer_tokens=int(r["n_answer_tokens"]),
                    sum_logprob=float(r["sum_logprob"]),
                    mean_logprob=float(r["mean_logprob"]),
                    mink_pp_score=(
                        float(r["mink_pp_score"])
                        if r.get("mink_pp_score") is not None
                        else None
                    ),
                    metadata=r.get("metadata", {}) or {},
                )
            )
    return out


def build_heatmap(summary_paths: Iterable[Path]) -> dict:
    """Aggregate ``.summary.json`` sidecars into a heatmap structure.

    Returns
    -------
    dict
        ``{
            "models": [model_a, model_b, ...],
            "benchmarks": [bench_x, bench_y, ...],
            "p_value": np.ndarray of shape (n_bench, n_model),  # NaN if missing
            "significant_at_0_01": np.ndarray of shape (n_bench, n_model, dtype=bool),
            "n_examples": np.ndarray of shape (n_bench, n_model),
        }``
    """
    rows: list[dict] = []
    for p in summary_paths:
        rows.append(json.loads(Path(p).read_text(encoding="utf-8")))

    benchmarks = sorted({r["benchmark"] for r in rows})
    models = sorted({r["model"] for r in rows})
    b_idx = {b: i for i, b in enumerate(benchmarks)}
    m_idx = {m: j for j, m in enumerate(models)}

    pmat = np.full((len(benchmarks), len(models)), np.nan)
    smat = np.zeros((len(benchmarks), len(models)), dtype=bool)
    nmat = np.zeros((len(benchmarks), len(models)), dtype=int)

    for r in rows:
        i, j = b_idx[r["benchmark"]], m_idx[r["model"]]
        nmat[i, j] = int(r.get("n_examples", 0))
        exch = r.get("exchangeability")
        if exch is not None:
            pmat[i, j] = float(exch["p_value"])
            smat[i, j] = bool(exch.get("significant_at_0_01", False))

    return {
        "models": models,
        "benchmarks": benchmarks,
        "p_value": pmat,
        "significant_at_0_01": smat,
        "n_examples": nmat,
    }


def flag_examples(
    records: list[AuditRecord],
    *,
    mink_pp_threshold: Optional[float] = None,
    mink_pp_quantile: Optional[float] = None,
) -> list[str]:
    """Flag examples whose Min-K%++ score exceeds a threshold.

    Either pass an absolute ``mink_pp_threshold`` or a top-quantile (e.g.
    ``0.05`` for the top 5%%) computed from the records themselves.
    """
    if (mink_pp_threshold is None) == (mink_pp_quantile is None):
        raise ValueError("pass exactly one of mink_pp_threshold or mink_pp_quantile")

    scores = np.array(
        [r.mink_pp_score for r in records if r.mink_pp_score is not None],
        dtype=np.float64,
    )
    if scores.size == 0:
        return []

    if mink_pp_quantile is not None:
        if not (0.0 < mink_pp_quantile < 1.0):
            raise ValueError("mink_pp_quantile must be in (0, 1)")
        thresh = float(np.quantile(scores, 1.0 - mink_pp_quantile))
    else:
        thresh = float(mink_pp_threshold)

    return [
        r.example_id
        for r in records
        if r.mink_pp_score is not None and r.mink_pp_score >= thresh
    ]


def flag_examples_loss_attack(
    records: list[AuditRecord],
    *,
    quantile: Optional[float] = None,
    threshold: Optional[float] = None,
) -> list[str]:
    """Flag examples whose loss-attack score is in the top quantile.

    Loss-attack score = ``-mean_logprob`` (Yeom et al. CSF'18; Carlini
    et al. ICLR'23). Lower loss \u2192 higher score \u2192 more memorization
    evidence. This detector is statistically distinct from Min-K%%++
    because it uses only the chosen-token loss rather than per-position
    moments over the vocabulary.
    """
    if (threshold is None) == (quantile is None):
        raise ValueError("pass exactly one of threshold or quantile")

    scores = np.array([-r.mean_logprob for r in records], dtype=np.float64)
    if scores.size == 0:
        return []

    if quantile is not None:
        if not (0.0 < quantile < 1.0):
            raise ValueError("quantile must be in (0, 1)")
        thresh = float(np.quantile(scores, 1.0 - quantile))
    else:
        thresh = float(threshold)

    return [r.example_id for r, s in zip(records, scores) if s >= thresh]


def aggregate_flags(detector_to_ids: dict[str, Iterable[str]]) -> dict:
    """Build a flag table summarising per-detector and union flags."""
    by_id: dict[str, set[str]] = {}
    for det, ids in detector_to_ids.items():
        for eid in ids:
            by_id.setdefault(eid, set()).add(det)
    return {
        "n_flagged_total": len(by_id),
        "per_detector_counts": {d: len(list(ids)) for d, ids in detector_to_ids.items()},
        "n_flagged_by_multiple": sum(1 for v in by_id.values() if len(v) >= 2),
        "flag_table": {eid: sorted(dets) for eid, dets in by_id.items()},
    }
