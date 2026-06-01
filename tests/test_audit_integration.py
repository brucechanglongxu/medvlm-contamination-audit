"""End-to-end audit driver integration test with a mock scorer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from medvlm_contam.audit import run_audit
from medvlm_contam.benchmarks.base import Benchmark, BenchmarkExample
from medvlm_contam.models.mock import MockVLMScorer


class _TinyContaminatedBench(Benchmark):
    """Tiny in-memory benchmark whose first N examples mention 'lung'.

    With MockVLMScorer(memorized_substrings=('lung',)), the first chunk
    of examples will receive higher log-likelihood — i.e. the
    canonical-order Spearman statistic should be significantly negative.
    """

    name = "tiny_contam"
    canonical_order_description = "in-memory; contaminated items first"

    def __init__(self, n_contam: int = 30, n_clean: int = 30) -> None:
        self.items: list[BenchmarkExample] = []
        for i in range(n_contam):
            self.items.append(
                BenchmarkExample(
                    example_id=f"contam::{i:04d}",
                    image_path=None,
                    prompt=f"Question {i} about the lung",
                    answer="lung",
                    metadata={"contaminated": True},
                )
            )
        for j in range(n_clean):
            self.items.append(
                BenchmarkExample(
                    example_id=f"clean::{j:04d}",
                    image_path=None,
                    prompt=f"Question {j} about another organ",
                    answer="heart",
                    metadata={"contaminated": False},
                )
            )

    def __iter__(self) -> Iterator[BenchmarkExample]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)


def test_audit_driver_writes_jsonl_and_summary(tmp_path: Path):
    bench = _TinyContaminatedBench()
    scorer = MockVLMScorer(memorized_substrings=("lung",), seed=42)

    out = tmp_path / "audit.jsonl"
    summary = run_audit(
        bench,
        scorer,
        output_jsonl=out,
        topk_for_mink_pp=32,
        progress=False,
    )

    # JSONL written, one row per example.
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(rows) == len(bench)
    sample = rows[0]
    for key in (
        "benchmark",
        "example_id",
        "model",
        "n_answer_tokens",
        "sum_logprob",
        "mean_logprob",
        "mink_pp_score",
        "metadata",
    ):
        assert key in sample
    assert sample["benchmark"] == bench.name
    assert sample["model"] == scorer.name

    # Summary sidecar.
    sidecar = out.with_suffix(".summary.json")
    assert sidecar.exists()
    summary_disk = json.loads(sidecar.read_text())
    assert summary_disk == summary

    # The exchangeability test should pick up the planted contamination
    # (the first 30 examples have systematically higher log-likelihood).
    exch = summary["exchangeability"]
    assert exch["p_value"] < 0.01
    assert exch["significant_at_0_01"] is True
