"""End-to-end audit simulation on synthetic benchmarks.

Runs the full audit + analysis pipeline using the deterministic
:class:`MockVLMScorer` so the artifact layout can be validated without
GPUs or model downloads. Drop-in template for the real run: swap
``MockVLMScorer`` for ``HFVisionScorer`` and the loaders for the
SLAKE / PathVQA loaders, keep the rest.

Usage::

    python scripts/simulate_audit.py --output-dir outputs/sim

Produces, per (benchmark, model) cell:

    outputs/sim/<bench>__<model>.jsonl                (per-example records)
    outputs/sim/<bench>__<model>.summary.json         (driver sidecar)

And, aggregated across cells:

    outputs/sim/heatmap.json                          (model × benchmark)
    outputs/sim/flag_table.json                       (union flags)
    outputs/sim/rank_churn.json                       (full vs clean leaderboard)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

from medvlm_contam.analysis import (
    aggregate_flags,
    build_heatmap,
    flag_examples,
    load_audit_jsonl,
)
from medvlm_contam.audit import run_audit
from medvlm_contam.benchmarks.base import Benchmark, BenchmarkExample
from medvlm_contam.clean_benchmark import rank_churn
from medvlm_contam.models.mock import MockVLMScorer


# ---------------------------------------------------------------------------
# Synthetic benchmarks
# ---------------------------------------------------------------------------


class _SyntheticBench(Benchmark):
    def __init__(
        self,
        name: str,
        n_contam: int,
        n_clean: int,
        contam_prompt: str,
        clean_prompt: str,
        contam_answer: str,
        clean_answer: str,
    ) -> None:
        self.name = name
        self.canonical_order_description = "synthetic; contaminated items first"
        items: list[BenchmarkExample] = []
        for i in range(n_contam):
            items.append(
                BenchmarkExample(
                    example_id=f"{name}::contam::{i:04d}",
                    image_path=None,
                    prompt=f"{contam_prompt} (q{i})",
                    answer=contam_answer,
                    metadata={"contaminated": True},
                )
            )
        for j in range(n_clean):
            items.append(
                BenchmarkExample(
                    example_id=f"{name}::clean::{j:04d}",
                    image_path=None,
                    prompt=f"{clean_prompt} (q{j})",
                    answer=clean_answer,
                    metadata={"contaminated": False},
                )
            )
        self.items = items

    def __iter__(self) -> Iterator[BenchmarkExample]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)


def _build_benchmarks() -> list[Benchmark]:
    return [
        _SyntheticBench(
            "sim_chest",
            n_contam=40,
            n_clean=40,
            contam_prompt="Question about the lung",
            clean_prompt="Question about another organ",
            contam_answer="lung",
            clean_answer="heart",
        ),
        _SyntheticBench(
            "sim_path",
            n_contam=30,
            n_clean=50,
            contam_prompt="What does the pathology slide show",
            clean_prompt="Generic radiograph question",
            contam_answer="adenocarcinoma",
            clean_answer="normal",
        ),
    ]


def _build_models() -> list[MockVLMScorer]:
    return [
        # Heavily contaminated (memorizes both signature terms).
        MockVLMScorer(
            name="mock-vlm-contaminated",
            memorized_substrings=("lung", "adenocarcinoma"),
            seed=1,
        ),
        # Partially contaminated (lung only).
        MockVLMScorer(
            name="mock-vlm-partial",
            memorized_substrings=("lung",),
            seed=2,
        ),
        # Clean.
        MockVLMScorer(name="mock-vlm-clean", memorized_substrings=(), seed=3),
    ]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def _simulate_correctness(
    bench: Benchmark, model: MockVLMScorer
) -> dict[str, bool]:
    """Synthesize correctness: memorized items are always correct;
    non-memorized items are correct ~40% of the time, deterministically."""
    out = {}
    for ex in bench:
        memo = model._is_memorized(ex.prompt, ex.answer)
        rng = model._rng_for(ex.example_id)
        out[ex.example_id] = bool(memo or rng.random() < 0.4)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="outputs/sim")
    ap.add_argument("--mink-pp-quantile", type=float, default=0.10)
    args = ap.parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    benches = _build_benchmarks()
    models = _build_models()

    summary_paths: list[Path] = []
    # Per-example flag sources. NOTE: exchangeability is a benchmark-level
    # significance signal, not a per-example signal — it goes into the
    # heatmap, not into the union-of-flags used for the clean-benchmark
    # construction. Only per-example detectors (Min-K%++, image-NN) feed
    # the clean-subset machinery.
    flagged_by_detector: dict[str, set[str]] = {"mink_pp": set()}
    benchmark_level_significant: dict[str, dict[str, bool]] = {}
    correctness_by_model_bench: dict[str, dict[str, dict[str, bool]]] = {}

    for bench in benches:
        correctness_by_model_bench[bench.name] = {}
        benchmark_level_significant[bench.name] = {}
        for scorer in models:
            tag = f"{bench.name}__{scorer.name.replace('/', '_')}"
            jsonl = out_dir / f"{tag}.jsonl"
            summary = run_audit(
                bench, scorer, output_jsonl=jsonl, topk_for_mink_pp=32, progress=False
            )
            summary_paths.append(jsonl.with_suffix(".summary.json"))

            exch = summary.get("exchangeability", {})
            benchmark_level_significant[bench.name][scorer.name] = bool(
                exch.get("significant_at_0_01", False)
            )

            records = load_audit_jsonl(jsonl)
            mink_flags = flag_examples(records, mink_pp_quantile=args.mink_pp_quantile)
            flagged_by_detector["mink_pp"].update(mink_flags)

            correctness_by_model_bench[bench.name][scorer.name] = _simulate_correctness(
                bench, scorer
            )

    # Heatmap.
    heat = build_heatmap(summary_paths)
    (out_dir / "heatmap.json").write_text(
        json.dumps(
            {
                "models": heat["models"],
                "benchmarks": heat["benchmarks"],
                "p_value": heat["p_value"].tolist(),
                "significant_at_0_01": heat["significant_at_0_01"].tolist(),
                "n_examples": heat["n_examples"].tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Flag table.
    flag_table = aggregate_flags(
        {k: sorted(v) for k, v in flagged_by_detector.items()}
    )
    flag_table["benchmark_level_exchangeability"] = benchmark_level_significant
    (out_dir / "flag_table.json").write_text(json.dumps(flag_table, indent=2), encoding="utf-8")

    # Rank-churn — per benchmark.
    churn = {}
    for bench in benches:
        per_bench_flags = [
            eid
            for eid in flag_table["flag_table"]
            if eid.startswith(f"{bench.name}::")
        ]
        result = rank_churn(
            correctness_by_model_bench[bench.name], flagged_ids=per_bench_flags
        )
        churn[bench.name] = {
            "per_model": [asdict(m) for m in result.per_model],
            "kendall_tau": result.kendall_tau,
            "spearman_rho": result.spearman_rho,
            "max_abs_delta": result.max_abs_delta,
            "max_delta_model": result.max_delta_model,
            "n_full": result.n_full,
            "n_clean": result.n_clean,
        }
    (out_dir / "rank_churn.json").write_text(json.dumps(churn, indent=2), encoding="utf-8")

    print(f"Wrote {len(summary_paths)} audit summaries + heatmap/flag_table/rank_churn to {out_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
