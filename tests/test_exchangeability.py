import numpy as np

from medvlm_contam.detectors.exchangeability import exchangeability_test


def test_exchangeability_null_uniform_noise_is_nonsignificant():
    rng = np.random.default_rng(123)
    # Pure noise: positions carry no signal about log-likelihood.
    loglik = rng.normal(loc=-10.0, scale=1.0, size=200)
    res = exchangeability_test(loglik, n_permutations=500, rng=np.random.default_rng(7))
    assert 0.0 < res.p_value < 1.0
    # Under the null, the canonical statistic should land in the bulk of
    # the permutation distribution → p-value not extreme.
    assert res.p_value > 0.05
    assert not res.significant


def test_exchangeability_detects_memorization_signature():
    """Simulate memorization: log-likelihood decreases with canonical position."""
    rng = np.random.default_rng(0)
    n = 200
    # Earlier-in-canonical-order examples have higher log-likelihood.
    base = -np.linspace(0.0, 5.0, n)
    loglik = base + rng.normal(0.0, 0.3, size=n)
    res = exchangeability_test(loglik, n_permutations=1000, rng=np.random.default_rng(1))
    assert res.p_value < 0.01
    assert res.significant
    assert res.n_examples == n
    assert res.permutation_logliks.shape == (1000,)


def test_exchangeability_rejects_too_few_examples():
    import pytest

    with pytest.raises(ValueError):
        exchangeability_test(np.array([1.0, 2.0, 3.0]))
