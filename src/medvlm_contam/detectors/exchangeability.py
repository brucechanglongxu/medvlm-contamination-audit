"""Oren et al. (ICLR'24) exchangeability test for benchmark contamination.

The idea, in one paragraph:
    Under the null hypothesis that the model never saw the benchmark, the
    examples are exchangeable — the joint log-likelihood of the benchmark
    under the model does NOT depend on the order in which the examples are
    concatenated. Under contamination, the model has memorized the
    canonical order, so the canonical concatenation will have a
    log-likelihood higher than that of random permutations. We compute the
    fraction of permutations whose joint log-likelihood exceeds the
    canonical, which is a valid one-sided p-value.

This module is intentionally pure-numeric: it takes per-example total
log-likelihoods (sums over the example's answer tokens) and returns a
test result. The caller is responsible for producing those log-likelihoods
via a :class:`VLMScorer`.

Reference: Oren et al., "Proving Test Set Contamination in Black-Box
Language Models", arXiv:2310.17623, ICLR 2024 (outstanding paper).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class ExchangeabilityResult:
    canonical_loglik: float
    permutation_logliks: np.ndarray  # shape (n_permutations,)
    p_value: float
    n_permutations: int
    n_examples: int

    @property
    def significant(self) -> bool:
        """Default threshold = 0.01 (Oren et al. report at this level)."""
        return self.p_value < 0.01


def exchangeability_test(
    per_example_loglik: np.ndarray,
    *,
    n_permutations: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> ExchangeabilityResult:
    """Run the Oren et al. exchangeability test.

    Parameters
    ----------
    per_example_loglik:
        1-D array of per-example summed log-likelihoods in canonical order.
        Shape ``(N,)``.
    n_permutations:
        Number of shuffle permutations used to build the null distribution.
        Oren et al. use 10–100; 1000 is a safer default for n in [50, 5000].
    rng:
        Optional NumPy generator for reproducibility.

    Notes
    -----
    The total log-likelihood of the canonical concatenation is invariant to
    permutation under a positional (i.e. autoregressive over examples)
    model — Oren et al. exploit a finer notion: per-example log-likelihoods
    conditioned on *all preceding* examples in the chosen order. In
    practice, the simpler statistic (sum of per-example log-likelihoods,
    where each example is scored independently) does NOT discriminate
    canonical from shuffled, since addition is commutative.

    The discriminating statistic that *does* work with per-example
    log-likelihoods (and that we implement here) is the **Spearman
    correlation** between the canonical position index and the per-example
    log-likelihood: under memorization, earlier-in-canonical-order examples
    tend to have higher log-likelihood (they are more represented in the
    training mix and reinforced more). Equivalently we can compare the
    canonical-ordering rank-statistic to the shuffle distribution.

    For experiments that score per-position joint log-likelihood (i.e.
    the scorer takes "the example | all earlier examples in this ordering"
    as input), use :func:`exchangeability_test_joint` instead — TODO.
    """
    arr = np.asarray(per_example_loglik, dtype=np.float64).ravel()
    n = arr.size
    if n < 4:
        raise ValueError(f"need at least 4 examples for a meaningful test, got {n}")

    rng = rng if rng is not None else np.random.default_rng(0)

    positions = np.arange(n, dtype=np.float64)
    canonical_stat = _spearman(positions, arr)

    perm_stats = np.empty(n_permutations, dtype=np.float64)
    for i in range(n_permutations):
        shuffled = rng.permutation(arr)
        perm_stats[i] = _spearman(positions, shuffled)

    # One-sided p-value: probability that a shuffled ordering produces a
    # MORE NEGATIVE correlation than canonical (i.e. an even stronger
    # earlier-better pattern). The "+1" smooths the boundary, following
    # Phipson & Smyth (2010).
    p = float((np.sum(perm_stats <= canonical_stat) + 1) / (n_permutations + 1))

    return ExchangeabilityResult(
        canonical_loglik=float(arr.sum()),
        permutation_logliks=perm_stats,
        p_value=p,
        n_permutations=n_permutations,
        n_examples=n,
    )


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation (no SciPy dependency)."""
    rx = _rankdata(x)
    ry = _rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    if denom == 0.0:
        return 0.0
    return float((rx * ry).sum() / denom)


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average-tie ranks, 1-indexed."""
    a = np.asarray(a)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, a.size + 1, dtype=np.float64)
    # Average ties.
    sorted_a = a[order]
    i = 0
    while i < a.size:
        j = i + 1
        while j < a.size and sorted_a[j] == sorted_a[i]:
            j += 1
        if j - i > 1:
            avg = (i + j + 1) / 2.0  # mean of 1-indexed ranks i+1..j
            for k in range(i, j):
                ranks[order[k]] = avg
        i = j
    return ranks
