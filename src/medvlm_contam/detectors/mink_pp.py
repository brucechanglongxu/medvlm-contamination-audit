"""Min-K%++ membership-inference score (Zhang et al., EMNLP'24 best paper).

Reference: arXiv:2409.14781.

Given per-token log-probabilities AND the per-position distribution over the
full vocabulary, Min-K%++ computes, for each token position, a calibrated
score:

    s_t = (log p(x_t) - mu_t) / sigma_t

where ``mu_t`` and ``sigma_t`` are the mean and standard deviation of
``log p(v)`` over the full vocabulary at position ``t`` (i.e. the
position's reference distribution under the model). The final score per
example is the MEAN of the K% LOWEST ``s_t`` values.

Memorized text has unusually high ``s_t`` everywhere, so its
bottom-K% mean is also elevated relative to non-memorized text — that's
the discriminative signal.

If the full-vocab moments are not available (very large V), the caller may
substitute the top-K-truncated approximation by passing
``topk_logprobs`` from a :class:`ScoredAnswer`; the resulting score is a
documented approximation, not the exact Min-K%++.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def _moments_from_topk(topk_logprobs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Approximate (mu_t, sigma_t) over the vocabulary from top-K logprobs.

    This is an approximation: it treats the top-K probability mass as the
    reference distribution. It overestimates ``mu_t`` and underestimates
    ``sigma_t`` for skewed distributions. Use only when full-vocab moments
    are unavailable.
    """
    p = np.exp(topk_logprobs)  # (T, K)
    # Renormalize within top-K so weights sum to 1 along K.
    p = p / np.clip(p.sum(axis=-1, keepdims=True), 1e-12, None)
    mu = (p * topk_logprobs).sum(axis=-1)  # (T,)
    var = (p * (topk_logprobs - mu[:, None]) ** 2).sum(axis=-1)
    sigma = np.sqrt(np.clip(var, 1e-12, None))
    return mu, sigma


def mink_pp_score(
    token_logprobs: np.ndarray,
    *,
    full_vocab_logprobs: Optional[np.ndarray] = None,
    topk_logprobs: Optional[np.ndarray] = None,
    k_percent: float = 20.0,
) -> float:
    """Compute the Min-K%++ score for one example.

    Parameters
    ----------
    token_logprobs:
        ``(T,)`` array of ``log p(x_t | x_<t)`` for the chosen tokens.
    full_vocab_logprobs:
        Optional ``(T, V)`` array of per-position log-probabilities over
        the full vocabulary. Preferred when memory allows.
    topk_logprobs:
        Optional ``(T, K)`` array of the top-K log-probabilities per
        position. Used to approximate the reference moments if
        ``full_vocab_logprobs`` is not provided.
    k_percent:
        K%% — the fraction of the lowest-scoring tokens averaged in the
        final statistic. Zhang et al. recommend 20%.

    Returns
    -------
    float
        The Min-K%++ score (higher → more memorization evidence).
    """
    token_logprobs = np.asarray(token_logprobs, dtype=np.float64)
    if token_logprobs.ndim != 1 or token_logprobs.size == 0:
        raise ValueError("token_logprobs must be a non-empty 1-D array")

    if full_vocab_logprobs is not None:
        flp = np.asarray(full_vocab_logprobs, dtype=np.float64)
        if flp.ndim != 2 or flp.shape[0] != token_logprobs.size:
            raise ValueError("full_vocab_logprobs must have shape (T, V)")
        p = np.exp(flp - flp.max(axis=-1, keepdims=True))
        p = p / p.sum(axis=-1, keepdims=True)
        mu = (p * flp).sum(axis=-1)
        sigma = np.sqrt(np.clip((p * (flp - mu[:, None]) ** 2).sum(axis=-1), 1e-12, None))
    elif topk_logprobs is not None:
        mu, sigma = _moments_from_topk(np.asarray(topk_logprobs, dtype=np.float64))
    else:
        raise ValueError("must pass either full_vocab_logprobs or topk_logprobs")

    z = (token_logprobs - mu) / sigma  # (T,)
    # MIN-K%%: the K%% LOWEST z values per the paper's definition.
    k = max(1, int(np.ceil((k_percent / 100.0) * z.size)))
    lowest_k = np.partition(z, k - 1)[:k]
    return float(lowest_k.mean())


def mink_pp_batch(
    token_logprobs_list: Sequence[np.ndarray],
    *,
    topk_logprobs_list: Optional[Sequence[np.ndarray]] = None,
    full_vocab_logprobs_list: Optional[Sequence[np.ndarray]] = None,
    k_percent: float = 20.0,
) -> np.ndarray:
    """Vectorized-loop wrapper for many examples. Returns ``(N,)`` array."""
    n = len(token_logprobs_list)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = mink_pp_score(
            token_logprobs_list[i],
            full_vocab_logprobs=(
                full_vocab_logprobs_list[i] if full_vocab_logprobs_list is not None else None
            ),
            topk_logprobs=(
                topk_logprobs_list[i] if topk_logprobs_list is not None else None
            ),
            k_percent=k_percent,
        )
    return out
