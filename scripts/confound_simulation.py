"""Synthetic demonstration of the cohort-median Min-K%++ confound.

This script makes the negative result of the paper (Detectors 3 and 4 are
confounded by inter-model calibration heterogeneity) precise and
reproducible *without* any model or GPU. It instantiates the additive +
multiplicative calibration model of the Min-K%++ score under the null
hypothesis of NO contamination, and shows that:

  (1) a clean, never-contaminated model whose calibration gain exceeds the
      cohort median is nonetheless flagged by the cohort-relative
      tail-enrichment criterion Pr[Delta > 100] > 5%, and

  (2) the false-positive flag rate of a clean high-gain model grows with
      the number of calibration-outlier (low-gain) models in the cohort.

Formal model.  For benchmark example i and model m, write the per-example
Min-K%++ score as

    s_{i,m} = g_m * e_i + b_m + eps_{i,m},                          (1)

where e_i >= 0 is an intrinsic, model-independent "easiness" (large for
low-entropy closed-form answers), g_m > 0 is a model-specific calibration
gain, b_m is an additive offset, and eps is zero-mean noise.  A genuine
contamination signal would add a term delta_{i,m} > 0 on memorized cells;
HERE delta == 0 everywhere (the null).

The cohort-relative statistic of Detector 3 is

    Delta_{i,target} = s_{i,target} - median_{j != target} s_{i,j}
                     = (g_target - median_j g_j) * e_i
                       + (b_target - median_j b_j) + noise.            (2)

Equation (2) is the whole story: under the null, Delta_i has a right tail
driven by the easiness tail {e_i} scaled by the *gain gap*
Gamma_target = g_target - median_{j != target} g_j.  Whenever
Gamma_target > 0, Pr[Delta_i > tau] is strictly positive for any tau,
with no contamination present.  The flag therefore measures
"is this model's gain above the cohort median", not "did this model see
these examples".  An external clean baseline with above-median gain is
flagged identically -- which is exactly the BLIP-2 result in the paper.

Usage::

    python scripts/confound_simulation.py \
        --out-json outputs/confound_simulation.json \
        --out-fig  figures/confound_simulation.png

Deterministic: fixed seed, no network, no GPU.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Calibration model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelCalib:
    """Calibration parameters of one model under the score model (1)."""

    name: str
    gain: float       # g_m : multiplicative calibration gain (> 0)
    offset: float     # b_m : additive offset (nats)
    contaminated: bool = False  # ground truth; False everywhere in the null


def sample_easiness(n: int, rng: np.random.Generator) -> np.ndarray:
    """Per-example intrinsic easiness e_i >= 0.

    Mirrors the SLAKE-En structure: a majority OPEN stratum with modest
    easiness and a minority CLOSED low-entropy stratum with a heavy right
    tail (the closed-form yes/no / single-organ items that dominate the
    observed Delta tail).
    """
    n_closed = int(round(0.40 * n))  # ~40% CLOSED, as in SLAKE-En (416/1061)
    n_open = n - n_closed
    # OPEN: small non-negative easiness.
    e_open = np.abs(rng.normal(0.0, 1.0, size=n_open))
    # CLOSED: heavy-tailed easiness (low answer-token entropy -> extreme
    # calibrated z-scores under a high-gain model). Exponential tail in nats.
    e_closed = rng.exponential(scale=900.0, size=n_closed)
    e = np.concatenate([e_open, e_closed])
    rng.shuffle(e)
    return e


def score_matrix(
    easiness: np.ndarray,
    models: list[ModelCalib],
    rng: np.random.Generator,
    noise_sd: float = 2.0,
) -> np.ndarray:
    """Return the (n_examples, n_models) Min-K%++ score matrix under (1)."""
    n = easiness.size
    S = np.empty((n, len(models)), dtype=np.float64)
    for j, m in enumerate(models):
        eps = rng.normal(0.0, noise_sd, size=n)
        S[:, j] = m.gain * easiness + m.offset + eps
    return S


def cohort_delta(S: np.ndarray, target_idx: int) -> np.ndarray:
    """Delta_i for the target model vs the median of the *other* models."""
    other = np.delete(S, target_idx, axis=1)
    med = np.median(other, axis=1)
    return S[:, target_idx] - med


def tail_flag_rate(delta: np.ndarray, threshold: float = 100.0) -> float:
    """Pr[Delta > threshold] -- the Detector 3 tail-enrichment statistic."""
    return float(np.mean(delta > threshold))


# ---------------------------------------------------------------------------
# Experiment 1: clean baseline is flagged at parity with "suspect" models
# ---------------------------------------------------------------------------


def experiment_parity(rng: np.random.Generator) -> dict:
    """Reproduce the SLAKE-En pattern: two low-gain negative-outlier models
    drag the cohort median down; every above-median-gain model -- including
    a provably clean external baseline -- fires the tail flag."""
    # Cohort mirrors the paper: two negative-outlier (low-gain) models and
    # three high-gain models, one of which is a contamination-free baseline.
    models = [
        ModelCalib("InternVL3 (low-gain outlier)", gain=0.05, offset=0.0),
        ModelCalib("Qwen2.5-VL (low-gain outlier)", gain=0.05, offset=0.0),
        ModelCalib("CheXagent (high-gain)", gain=1.00, offset=0.0),
        ModelCalib("LLaVA-OneVision (high-gain)", gain=1.00, offset=0.0),
        ModelCalib("BLIP-2 (clean baseline, high-gain)", gain=1.00, offset=0.0),
    ]
    n = 1061  # SLAKE-En test size
    e = sample_easiness(n, rng)
    S = score_matrix(e, models, rng)

    rows = []
    for j, m in enumerate(models):
        d = cohort_delta(S, j)
        rows.append(
            {
                "model": m.name,
                "contaminated": m.contaminated,
                "gain": m.gain,
                "delta_max": float(d.max()),
                "pr_delta_gt_100": tail_flag_rate(d, 100.0),
                "flagged": tail_flag_rate(d, 100.0) > 0.05,
            }
        )
    return {"n_examples": n, "rows": rows}


# ---------------------------------------------------------------------------
# Experiment 2: false-positive rate vs cohort composition
# ---------------------------------------------------------------------------


def experiment_composition(rng: np.random.Generator) -> dict:
    """Vary the number of low-gain outliers in the cohort and measure the
    false-positive flag rate of a single clean high-gain probe model.

    For each cohort with k low-gain outliers (k = 0..4) plus a fixed set of
    high-gain models, we add one extra clean high-gain probe and report
    whether it is flagged, averaged over repeated draws."""
    n = 1061
    n_repeats = 200
    ks = list(range(0, 5))
    results = []
    for k in ks:
        flag_rates = []
        for _ in range(n_repeats):
            models = [
                ModelCalib(f"outlier{i}", gain=0.05, offset=0.0) for i in range(k)
            ]
            # Two high-gain "anchor" cohort members.
            models += [
                ModelCalib("anchorA", gain=1.00, offset=0.0),
                ModelCalib("anchorB", gain=1.00, offset=0.0),
            ]
            # The clean probe whose false-positive rate we measure.
            probe_idx = len(models)
            models.append(ModelCalib("clean_probe", gain=1.00, offset=0.0))

            e = sample_easiness(n, rng)
            S = score_matrix(e, models, rng)
            d = cohort_delta(S, probe_idx)
            flag_rates.append(tail_flag_rate(d, 100.0))
        flag_rates = np.asarray(flag_rates)
        results.append(
            {
                "n_low_gain_outliers": k,
                "mean_pr_delta_gt_100": float(flag_rates.mean()),
                "false_positive_flag_prob": float(np.mean(flag_rates > 0.05)),
            }
        )
    return {"n_examples": n, "n_repeats": n_repeats, "sweep": results}


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def make_figure(parity: dict, comp: dict, out_fig: Path, rng: np.random.Generator) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # Panel (a): tail-flag rate per model; clean baseline at parity.
    names = [r["model"].split(" (")[0] for r in parity["rows"]]
    rates = [100.0 * r["pr_delta_gt_100"] for r in parity["rows"]]
    is_clean_baseline = ["baseline" in r["model"] for r in parity["rows"]]
    is_low_gain = [r["gain"] < 0.5 for r in parity["rows"]]
    colors = []
    for clean, low in zip(is_clean_baseline, is_low_gain):
        if low:
            colors.append("#9e9e9e")       # low-gain outliers
        elif clean:
            colors.append("#d62728")       # clean baseline (the false positive)
        else:
            colors.append("#1f77b4")       # high-gain "suspect" models
    ax1.bar(range(len(names)), rates, color=colors)
    ax1.axhline(5.0, ls="--", lw=1, color="k", label="flag threshold (5%)")
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax1.set_ylabel(r"$\Pr[\Delta > 100]$  (%)")
    ax1.set_title("(a) No model is contaminated, yet every\nabove-median-gain model is flagged")
    ax1.legend(fontsize=8, loc="upper left")

    # Panel (b): false-positive flag prob of a clean probe vs cohort comp.
    ks = [r["n_low_gain_outliers"] for r in comp["sweep"]]
    fp = [100.0 * r["false_positive_flag_prob"] for r in comp["sweep"]]
    ax2.plot(ks, fp, "o-", color="#d62728")
    ax2.set_xlabel("# calibration-outlier (low-gain) models in cohort")
    ax2.set_ylabel("false-positive flag prob. of a\nclean high-gain model (%)")
    ax2.set_xticks(ks)
    ax2.set_ylim(-3, 103)
    ax2.set_title("(b) Adding low-gain outliers manufactures\nfalse positives for clean models")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=20260531)
    ap.add_argument("--out-json", default="outputs/confound_simulation.json")
    ap.add_argument("--out-fig", default="figures/confound_simulation.png")
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    parity = experiment_parity(rng)
    comp = experiment_composition(rng)

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {"seed": args.seed, "parity": parity, "composition": comp}
    out_json.write_text(json.dumps(payload, indent=2))

    make_figure(parity, comp, Path(args.out_fig), rng)

    # Console summary.
    print(f"[seed={args.seed}]  parity experiment (no contamination anywhere):")
    for r in parity["rows"]:
        flag = "FLAGGED" if r["flagged"] else "clean"
        print(
            f"  {r['model']:<42s} gain={r['gain']:.2f}  "
            f"Pr[D>100]={100*r['pr_delta_gt_100']:5.1f}%  "
            f"Dmax={r['delta_max']:7.0f}  -> {flag}"
        )
    print("\n  composition sweep (false-positive flag prob of a clean probe):")
    for r in comp["sweep"]:
        print(
            f"    {r['n_low_gain_outliers']} outliers -> "
            f"{100*r['false_positive_flag_prob']:5.1f}% of draws flag the clean probe"
        )
    print(f"\n  wrote {out_json}")
    print(f"  wrote {args.out_fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
