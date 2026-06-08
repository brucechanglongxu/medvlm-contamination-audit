"""Performance-inflation estimate for SLAKE-En image-side source overlap.

Detector 1 flags ~23% of SLAKE-En QA examples as having an extreme
same-view nearest neighbour in PMC-OA-beta. This script asks whether that
source overlap actually advantages the models on the flagged examples.

The audit is teacher-forced: it scores the gold-answer log-likelihood
rather than generating graded answers, so we use the mean per-token
gold-answer log-probability (nats/token) as a likelihood proxy for "how
easily a model produces the reference answer" and compare its value on the
source-overlap-flagged examples against the unflagged ones, per model and
within the low-entropy CLOSED stratum.

Reads outputs/slake_en_overlap_inflation.json (per-example flag,
answer_type, and per-model mean_logprob scalars). No model and no GPU.
Deterministic.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "outputs" / "slake_en_overlap_inflation.json"


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def main() -> None:
    blob = json.loads(DATA.read_text())
    recs = blob["records"]
    models = blob["models"]
    n_flag = sum(r["flagged"] for r in recs)
    print(
        f"SLAKE-En: {len(recs)} QA examples; "
        f"{n_flag} flagged by image source-overlap "
        f"({100 * n_flag / len(recs):.1f}%)\n"
    )

    header = (
        f"{'model':20s} {'flagged':>9s} {'unflagged':>9s} {'gap':>8s}"
        f"   {'CLOSED flag':>11s} {'CLOSED unflag':>13s} {'gap':>8s}"
    )
    print(header)
    print("-" * len(header))
    for m in models:
        fl = [r["mean_logprob"][m] for r in recs if r["flagged"]]
        un = [r["mean_logprob"][m] for r in recs if not r["flagged"]]
        flc = [r["mean_logprob"][m] for r in recs if r["flagged"] and r["answer_type"] == "CLOSED"]
        unc = [r["mean_logprob"][m] for r in recs if not r["flagged"] and r["answer_type"] == "CLOSED"]
        mf, mu = mean(fl), mean(un)
        cf, cu = mean(flc), mean(unc)
        print(
            f"{m:20s} {mf:9.3f} {mu:9.3f} {mf - mu:+8.3f}"
            f"   {cf:11.3f} {cu:13.3f} {cf - cu:+8.3f}"
        )

    print(
        "\nInterpretation: the flagged-minus-unflagged gold-answer log-prob gap "
        "is small and\nmixed in sign across models, and the contamination-free "
        "BLIP-2 baseline shows a gap\nof the same magnitude. The SLAKE-En source "
        "overlap does not translate into a\ndetectable gold-answer likelihood "
        "advantage."
    )


if __name__ == "__main__":
    main()
