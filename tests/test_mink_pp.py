import numpy as np

from medvlm_contam.detectors.mink_pp import mink_pp_score, mink_pp_batch


def _synthetic_topk(T: int, K: int, low_logp: float, rng: np.random.Generator):
    """Build a (T, K) array of plausible top-K logprobs whose softmax-mean ~= low_logp."""
    # Place a single high-probability token near 0 and decay the rest.
    base = -np.linspace(0.5, 5.0, K)
    arr = np.broadcast_to(base, (T, K)).copy()
    arr += rng.normal(0.0, 0.05, size=arr.shape)
    return arr


def test_mink_pp_memorized_scores_higher_than_typical():
    rng = np.random.default_rng(42)
    T, K = 30, 32

    topk = _synthetic_topk(T, K, low_logp=-2.0, rng=rng)

    # Typical (non-memorized) tokens: log p ~ middle of the top-K distribution.
    typical_tok_logp = topk.mean(axis=-1)
    # Memorized tokens: log p close to the max (top of the distribution).
    memorized_tok_logp = topk.max(axis=-1) - 0.05

    s_typical = mink_pp_score(typical_tok_logp, topk_logprobs=topk, k_percent=20.0)
    s_memorized = mink_pp_score(memorized_tok_logp, topk_logprobs=topk, k_percent=20.0)

    assert s_memorized > s_typical


def test_mink_pp_batch_shape_and_values():
    rng = np.random.default_rng(0)
    K = 16
    items = []
    topks = []
    for T in (5, 10, 20):
        tk = _synthetic_topk(T, K, low_logp=-2.0, rng=rng)
        items.append(tk.mean(axis=-1))
        topks.append(tk)
    out = mink_pp_batch(items, topk_logprobs_list=topks)
    assert out.shape == (3,)
    assert np.all(np.isfinite(out))


def test_mink_pp_requires_reference_distribution():
    import pytest

    with pytest.raises(ValueError):
        mink_pp_score(np.array([-1.0, -2.0, -3.0]))
