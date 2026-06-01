"""Threshold (tau / alpha) sensitivity of the image nearest-neighbour detector.

Detector 1 flags a benchmark image as a near-duplicate of a pretraining-corpus
image when its cosine nearest-neighbour distance in SigLIP space falls below a
threshold tau, where tau is the alpha-quantile of a null distribution of
natural-image NN distances.  This script reads the already-computed alpha sweeps
(outputs/image_nn*/alpha_sweep.json) and renders a single figure showing that
the SLAKE-En near-duplicate signal is robust across the entire operating range
of tau, while the PathVQA control stays at the floor -- i.e. the result is not
an artefact of the alpha = 0.01 choice used in the main text.

No model and no GPU: this only re-plots committed JSON.  Deterministic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BACKBONES = [
    ("outputs/image_nn/alpha_sweep.json", "SigLIP B/16", "#1f77b4"),
    ("outputs/image_nn_so400m/alpha_sweep.json", "SigLIP SoVIT-400m", "#ff7f0e"),
]
# alpha used in the main text.
ALPHA_MAIN = 0.01


def load_sweep(path: Path) -> list[dict]:
    return json.load(open(path))["sweep"]


def make_figure(sweeps: list[tuple[str, str, list[dict]]], out_fig: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # Panel (a): SLAKE-En image-flag fraction vs alpha, with PathVQA control.
    for label, color, sweep in sweeps:
        alphas = [r["alpha"] for r in sweep]
        slake = [r["slake_en__img_pct"] for r in sweep]
        path = [r["pathvqa__img_pct"] for r in sweep]
        ax1.plot(alphas, slake, "o-", color=color, label=f"SLAKE-En ({label})")
        ax1.plot(
            alphas,
            path,
            "s--",
            color=color,
            alpha=0.55,
            label=f"PathVQA control ({label})",
        )
    ax1.axvline(ALPHA_MAIN, ls=":", lw=1.2, color="k", label=r"main-text $\alpha=0.01$")
    ax1.set_xscale("log")
    ax1.set_xlabel(r"null-quantile level $\alpha$  (threshold $\tau=Q_\alpha$)")
    ax1.set_ylabel("images flagged (%)")
    ax1.set_title("(a) SLAKE-En near-duplicate rate is\nmonotone and far above the control")
    ax1.legend(fontsize=7, loc="upper left")
    ax1.grid(alpha=0.3)

    # Panel (b): triple-hit count (image + text + answer agreement) vs alpha.
    for label, color, sweep in sweeps:
        alphas = [r["alpha"] for r in sweep]
        triple = [r["slake_en__triple"] for r in sweep]
        ax2.plot(alphas, triple, "o-", color=color, label=f"SLAKE-En ({label})")
    ax2.axvline(ALPHA_MAIN, ls=":", lw=1.2, color="k", label=r"main-text $\alpha=0.01$")
    ax2.set_xscale("log")
    ax2.set_xlabel(r"null-quantile level $\alpha$  (threshold $\tau=Q_\alpha$)")
    ax2.set_ylabel("triple-hit images (count)")
    ax2.set_title("(b) Triple hits (image+question+answer)\npersist across the threshold sweep")
    ax2.legend(fontsize=7, loc="upper left")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-fig", default="figures/tau_sensitivity.png")
    args = ap.parse_args(argv)

    sweeps = []
    for path, label, color in BACKBONES:
        sweeps.append((label, color, load_sweep(Path(path))))

    make_figure([(lbl, clr, sw) for lbl, clr, sw in sweeps], Path(args.out_fig))

    print("tau-sensitivity sweep (image-flag % | triple hits):")
    for label, _color, sweep in sweeps:
        print(f"  {label}:")
        for r in sweep:
            print(
                f"    alpha={r['alpha']:.3f}  tau={r['threshold']:.5f}  "
                f"SLAKE img={r['slake_en__img_pct']:5.2f}%  triple={r['slake_en__triple']:2d}  "
                f"PathVQA img={r['pathvqa__img_pct']:.2f}%"
            )
    print(f"\n  wrote {args.out_fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
