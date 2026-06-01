"""Clean-benchmark builder + leaderboard rank-churn analysis (E3).

Given a benchmark and a set of per-(example, detector) flags, derive a
``clean`` subset (full minus union-of-flagged) and quantify the impact on
the public leaderboard:

- Per-model accuracy on **full** vs **clean**.
- Headline-accuracy deltas.
- Kendall's tau between the full and clean leaderboards.

This module is loader/detector/scorer-agnostic — it consumes per-example
correctness booleans and a flag table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Flag combination
# ---------------------------------------------------------------------------


def union_flagged_ids(detector_flags: Mapping[str, Iterable[str]]) -> set[str]:
    """Union of flagged example IDs across an arbitrary set of detectors.

    ``detector_flags`` maps detector name → iterable of flagged example IDs.
    """
    out: set[str] = set()
    for ids in detector_flags.values():
        out.update(ids)
    return out


def clean_subset_ids(
    all_example_ids: Iterable[str], flagged_ids: Iterable[str]
) -> list[str]:
    """All IDs minus the union of flags, preserving the input order."""
    flag_set = set(flagged_ids)
    return [eid for eid in all_example_ids if eid not in flag_set]


# ---------------------------------------------------------------------------
# Leaderboard rank churn
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelAccuracy:
    model: str
    full_accuracy: float
    clean_accuracy: float
    full_n: int
    clean_n: int

    @property
    def delta(self) -> float:
        return self.full_accuracy - self.clean_accuracy


@dataclass(frozen=True)
class RankChurnResult:
    per_model: list[ModelAccuracy]
    kendall_tau: float
    spearman_rho: float
    max_abs_delta: float
    max_delta_model: str
    n_clean: int
    n_full: int


def _kendall_tau(x: np.ndarray, y: np.ndarray) -> float:
    n = x.size
    if n < 2:
        return 1.0
    concordant = 0
    discordant = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            sgn = dx * dy
            if sgn > 0:
                concordant += 1
            elif sgn < 0:
                discordant += 1
    denom = concordant + discordant
    return float((concordant - discordant) / denom) if denom else 1.0


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    from .detectors.exchangeability import _spearman as _impl  # reuse

    return _impl(x.astype(np.float64), y.astype(np.float64))


def rank_churn(
    model_correctness: Mapping[str, Mapping[str, bool]],
    *,
    flagged_ids: Iterable[str],
) -> RankChurnResult:
    """Compute leaderboard rank churn between full and clean subsets.

    Parameters
    ----------
    model_correctness:
        ``{model_name: {example_id: is_correct_bool}}``. Every model is
        assumed to have evaluated the same example universe; missing IDs
        for any model raise.
    flagged_ids:
        Example IDs to remove when forming the ``clean`` subset.

    Returns
    -------
    RankChurnResult
        With per-model full/clean accuracy, Kendall's tau between the
        full-leaderboard rank and the clean-leaderboard rank, and the
        worst-case headline-accuracy delta.
    """
    flagged = set(flagged_ids)

    # Use the example universe of the first model as canonical and verify
    # all other models agree.
    if not model_correctness:
        raise ValueError("model_correctness must be non-empty")
    first_model = next(iter(model_correctness))
    universe = list(model_correctness[first_model].keys())
    for m, table in model_correctness.items():
        missing = set(universe) - set(table.keys())
        extra = set(table.keys()) - set(universe)
        if missing or extra:
            raise ValueError(
                f"model {m!r} has mismatched example universe vs {first_model!r}: "
                f"missing={len(missing)} extra={len(extra)}"
            )

    clean_ids = [eid for eid in universe if eid not in flagged]
    full_ids = universe

    per_model: list[ModelAccuracy] = []
    for model, table in model_correctness.items():
        full_acc = float(np.mean([table[i] for i in full_ids])) if full_ids else 0.0
        clean_acc = float(np.mean([table[i] for i in clean_ids])) if clean_ids else 0.0
        per_model.append(
            ModelAccuracy(
                model=model,
                full_accuracy=full_acc,
                clean_accuracy=clean_acc,
                full_n=len(full_ids),
                clean_n=len(clean_ids),
            )
        )

    full_scores = np.array([m.full_accuracy for m in per_model])
    clean_scores = np.array([m.clean_accuracy for m in per_model])
    tau = _kendall_tau(full_scores, clean_scores)
    rho = _spearman(full_scores, clean_scores)

    deltas = np.array([m.delta for m in per_model])
    abs_deltas = np.abs(deltas)
    worst = int(np.argmax(abs_deltas)) if abs_deltas.size else 0
    return RankChurnResult(
        per_model=per_model,
        kendall_tau=tau,
        spearman_rho=rho,
        max_abs_delta=float(abs_deltas.max()) if abs_deltas.size else 0.0,
        max_delta_model=per_model[worst].model if per_model else "",
        n_clean=len(clean_ids),
        n_full=len(full_ids),
    )
