"""Render the cross-model top-K Jaccard heatmap.

For each benchmark, plot a 4x4 matrix of pairwise Jaccard similarity of
top-K Min-K%++ anomalous example sets between models. Diagonal is 1.0.
Strong off-diagonal cells = candidates for shared contamination source.

Reads outputs/topk_overlap.json. Writes figures/topk_jaccard.png.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
OVERLAP = REPO / "outputs" / "topk_overlap.json"
OUT = REPO / "figures" / "topk_jaccard.png"

SHORT = {
    "OpenGVLab__InternVL3-8B": "InternVL3-8B",
    "Qwen__Qwen2.5-VL-7B-Instruct": "Qwen2.5-VL",
    "StanfordAIMI__CheXagent-8b": "CheXagent-8b",
    "llava-hf__llava-onevision-qwen2-7b-ov-hf": "LLaVA-OV",
}
BENCH_LABEL = {"pathvqa": "PathVQA", "slake_en": "SLAKE-En", "vqa_rad": "VQA-RAD"}


def main() -> None:
    data = json.loads(OVERLAP.read_text())
    benches = [b for b in data.keys() if b in BENCH_LABEL]
    n = 4

    fig, axes = plt.subplots(1, len(benches), figsize=(13, 4.2), constrained_layout=True)

    for ax, bench in zip(axes, benches):
        d = data[bench]
        models = list(SHORT.keys())
        M = np.eye(n)
        for p in d["pairs"]:
            if p["a"] not in models or p["b"] not in models:
                continue
            i, j = models.index(p["a"]), models.index(p["b"])
            M[i, j] = p["jaccard"]
            M[j, i] = p["jaccard"]

        im = ax.imshow(M, vmin=0.0, vmax=1.0, cmap="Reds")
        labels = [SHORT[m] for m in models]
        ax.set_xticks(range(n), labels, rotation=45, ha="right", fontsize=8.5)
        ax.set_yticks(range(n), labels, fontsize=8.5)
        ax.set_title(f"{BENCH_LABEL[bench]}\n(n={d['n_examples']}, K={d['K']})", fontsize=10)

        for i in range(n):
            for j in range(n):
                v = M[i, j]
                color = "white" if v > 0.5 else "black"
                weight = "bold" if v >= 0.5 and i != j else "normal"
                ax.text(
                    j, i, f"{v:.2f}",
                    ha="center", va="center",
                    color=color, fontsize=9, fontweight=weight,
                )

    cbar = fig.colorbar(im, ax=axes, shrink=0.7, label="Jaccard(top-25 Min-K%++ sets)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
