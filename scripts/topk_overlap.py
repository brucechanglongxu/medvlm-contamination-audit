"""Cross-model overlap of top-K Min-K%++ anomalous examples.

If two models were contaminated through the same source (e.g. the same
visual-instruction-tuning mix), their top-K anomalously-easy examples
should overlap far more than chance. We compute pairwise Jaccard
similarity of top-K sets across all 4 audited models on each benchmark.
"""

from __future__ import annotations

import json
import os
from itertools import combinations
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TOPK_DIR = REPO / "outputs" / "tail_enrichment_topk"
# Optional: directory holding the per-cell ``{bench}__{tag}.jsonl`` files used
# only to record per-benchmark example counts. Set MEDVLM_DATA_ROOT to point at
# it; if unset, example counts default to 0 and overlap statistics are
# unaffected.
DATA_ROOT = Path(os.environ.get("MEDVLM_DATA_ROOT", REPO / "data"))
OUT = REPO / "outputs" / "topk_overlap.json"


def discover() -> tuple[list[str], dict[str, list[str]], dict[str, int]]:
    """Auto-discover (bench, model_tag) cells from topk JSON files.

    Returns (all_benches, bench_to_models, bench_to_n_examples).
    n_examples is read from the matching ``{bench}__{tag}.jsonl`` row count.
    """
    bench_to_models: dict[str, list[str]] = {}
    for p in TOPK_DIR.glob("*__*.json"):
        model_tag, _, bench = p.stem.rpartition("__")
        if not model_tag or not bench:
            continue
        bench_to_models.setdefault(bench, []).append(model_tag)
    for b in bench_to_models:
        bench_to_models[b].sort()
    bench_to_n: dict[str, int] = {}
    for b, models in bench_to_models.items():
        if not models:
            continue
        jl = DATA_ROOT / f"{b}__{models[0]}.jsonl"
        if jl.exists():
            with jl.open() as fh:
                bench_to_n[b] = sum(1 for _ in fh)
        else:
            bench_to_n[b] = 0
    return sorted(bench_to_models), bench_to_models, bench_to_n


def load_topk(model: str, bench: str) -> set[str]:
    path = TOPK_DIR / f"{model}__{bench}.json"
    return {r["example_id"] for r in json.loads(path.read_text())}


def jaccard(a: set, b: set) -> float:
    if not (a or b):
        return 0.0
    return len(a & b) / len(a | b)


def expected_overlap(k: int, n: int) -> float:
    # E[|A cap B|] = k*k/n for two iid random size-k subsets of {1..n}
    return (k * k) / n


def main() -> None:
    BENCHES, BENCH_TO_MODELS, N_EXAMPLES = discover()
    if not BENCHES:
        raise SystemExit(f"no topk cells found under {TOPK_DIR}")
    print(
        f"discovered {len(BENCHES)} benchmarks; cohort sizes: "
        + ", ".join(f"{b}={len(BENCH_TO_MODELS[b])}" for b in BENCHES)
    )
    summary: dict = {}
    K = 25
    print(f"{'bench':9s}  {'pair':75s}  {'|cap|':>5s}  {'|cup|':>5s}  {'jaccard':>8s}  {'E[|cap|]':>9s}  {'lift':>6s}")
    print("-" * 132)
    for bench in BENCHES:
        n = N_EXAMPLES.get(bench, 0)
        e_cap = expected_overlap(K, n) if n > 0 else 0.0
        cell: dict = {
            "benchmark": bench,
            "n_examples": n,
            "K": K,
            "expected_intersection_size": e_cap,
            "pairs": [],
        }
        models = BENCH_TO_MODELS.get(bench, [])
        if len(models) < 2:
            print(f"{bench:9s}  (skip: only {len(models)} model(s))")
            summary[bench] = cell
            continue
        sets = {m: load_topk(m, bench) for m in models}
        for a, b in combinations(models, 2):
            inter = sets[a] & sets[b]
            union = sets[a] | sets[b]
            j = jaccard(sets[a], sets[b])
            lift = len(inter) / e_cap if e_cap > 0 else 0.0
            short_a = a.split("__")[-1][:35]
            short_b = b.split("__")[-1][:35]
            print(
                f"{bench:9s}  {short_a + ' vs ' + short_b:75s}  {len(inter):5d}  {len(union):5d}  {j:8.3f}  {e_cap:9.3f}  {lift:5.1f}x"
            )
            cell["pairs"].append(
                {
                    "a": a,
                    "b": b,
                    "intersection": len(inter),
                    "union": len(union),
                    "jaccard": j,
                    "lift_vs_chance": lift,
                }
            )
        summary[bench] = cell
        print()

    OUT.write_text(json.dumps(summary, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
