"""End-to-end audit driver.

Wires a :class:`Benchmark` loader to a :class:`VLMScorer` and runs the
configured detectors. Writes per-(benchmark, example, model) results to a
JSONL artifact that downstream notebooks consume for heatmaps and
rank-churn analyses.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from tqdm import tqdm

from .benchmarks.base import Benchmark, BenchmarkExample
from .detectors.exchangeability import exchangeability_test
from .detectors.mink_pp import mink_pp_score
from .models.base import ScoredAnswer, VLMScorer


@dataclass
class PerExampleRecord:
    benchmark: str
    example_id: str
    model: str
    n_answer_tokens: int
    sum_logprob: float
    mean_logprob: float
    mink_pp_score: Optional[float]
    elapsed_s: float
    metadata: dict


def run_audit(
    benchmark: Benchmark,
    scorer: VLMScorer,
    *,
    output_jsonl: Path,
    topk_for_mink_pp: int = 64,
    max_examples: Optional[int] = None,
    progress: bool = True,
) -> dict:
    """Score every example with ``scorer`` and write per-example records.

    Returns a summary dict including the benchmark-level exchangeability
    test on the per-example log-likelihoods (canonical iteration order).
    """
    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    per_example: list[PerExampleRecord] = []
    sum_logliks: list[float] = []

    examples: Iterable[BenchmarkExample] = benchmark
    if max_examples is not None:
        examples = (ex for i, ex in enumerate(examples) if i < max_examples)

    iterator = tqdm(examples, desc=f"{benchmark.name}|{scorer.name}", disable=not progress)

    with output_jsonl.open("w", encoding="utf-8") as f:
        for ex in iterator:
            t0 = time.perf_counter()
            scored: ScoredAnswer = scorer.score(
                ex.image_path, ex.prompt, ex.answer, topk=topk_for_mink_pp
            )
            elapsed = time.perf_counter() - t0

            mink = None
            if scored.topk_logprobs is not None and scored.token_logprobs.size > 0:
                mink = mink_pp_score(
                    scored.token_logprobs, topk_logprobs=scored.topk_logprobs
                )

            rec = PerExampleRecord(
                benchmark=benchmark.name,
                example_id=ex.example_id,
                model=scorer.name,
                n_answer_tokens=int(scored.token_logprobs.size),
                sum_logprob=scored.sum_logprob,
                mean_logprob=scored.mean_logprob,
                mink_pp_score=mink,
                elapsed_s=elapsed,
                metadata=ex.metadata,
            )
            per_example.append(rec)
            sum_logliks.append(rec.sum_logprob)
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

    summary: dict = {
        "benchmark": benchmark.name,
        "model": scorer.name,
        "n_examples": len(per_example),
        "per_example_jsonl": str(output_jsonl),
    }

    if len(sum_logliks) >= 4:
        res = exchangeability_test(np.asarray(sum_logliks))
        summary["exchangeability"] = {
            "canonical_loglik": res.canonical_loglik,
            "p_value": res.p_value,
            "n_permutations": res.n_permutations,
            "n_examples": res.n_examples,
            "significant_at_0_01": res.significant,
        }

    summary_path = output_jsonl.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
