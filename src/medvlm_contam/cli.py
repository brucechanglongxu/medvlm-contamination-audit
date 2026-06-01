"""Command-line entry point: ``medvlm-audit``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="medvlm-audit",
        description="Run a contamination audit of a medical VLM on a benchmark.",
    )
    p.add_argument("--benchmark", required=True, choices=["slake_en", "pathvqa"])
    p.add_argument(
        "--benchmark-root",
        default=None,
        help="Local cache root for the benchmark. Defaults to data/raw/<benchmark>.",
    )
    p.add_argument("--split", default="test")
    p.add_argument(
        "--model",
        required=True,
        help="HuggingFace model id, e.g. Qwen/Qwen2.5-VL-7B-Instruct",
    )
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--output", required=True, help="Output JSONL path.")
    p.add_argument("--max-examples", type=int, default=None)
    p.add_argument("--topk", type=int, default=64, help="Top-K logprobs for Min-K%%++.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.benchmark == "slake_en":
        from .benchmarks.slake import SlakeEnglish

        bench = SlakeEnglish(
            root=args.benchmark_root or "data/raw/slake",
            split=args.split,
            max_examples=args.max_examples,
        )
    elif args.benchmark == "pathvqa":
        from .benchmarks.pathvqa import PathVQA

        bench = PathVQA(
            root=args.benchmark_root or "data/raw/pathvqa",
            split=args.split,
            max_examples=args.max_examples,
        )
    else:  # pragma: no cover — argparse choices guard this
        raise SystemExit(f"unknown benchmark: {args.benchmark}")

    from .audit import run_audit
    from .models.hf_vlm import HFVisionScorer

    scorer = HFVisionScorer(args.model, device=args.device, dtype=args.dtype)
    summary = run_audit(
        bench,
        scorer,
        output_jsonl=Path(args.output),
        topk_for_mink_pp=args.topk,
        max_examples=args.max_examples,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
