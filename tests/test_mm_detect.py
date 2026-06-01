import numpy as np

from medvlm_contam.detectors.mm_detect import (
    option_perturbation_score,
    slot_guessing_score,
)
from medvlm_contam.models.mock import MockVLMScorer


def test_option_perturbation_detects_memorization():
    """Memorized scorers should NOT show a relabel gap if they memorize the
    *content* (which is what our mock does — the gold-answer text is the
    same string under any relabeling)."""
    scorer = MockVLMScorer(memorized_substrings=("lung",), seed=1)
    res = option_perturbation_score(
        scorer,
        image_path=None,
        prompt="What organ is shown?",
        choices=["lung", "heart", "kidney", "liver"],
        gold_index=0,
        n_perturbations=4,
        rng=np.random.default_rng(0),
    )
    # The mock memorizes by content, so canonical and perturbed scores
    # should be very close — the gap should be near zero.
    assert abs(res.gap) < 0.5
    assert res.n_perturbations >= 1


def test_option_perturbation_label_dependent_memorization():
    """A scorer that memorizes the FULL prompt-text (including the
    canonical letter ordering) should show a large positive gap when
    options are relabeled — that's the MM-Detect signature."""
    # The substring "A) lung" appears only in the canonical labeling.
    scorer = MockVLMScorer(memorized_substrings=("A) lung",), seed=2)
    res = option_perturbation_score(
        scorer,
        image_path=None,
        prompt="What organ is shown?",
        choices=["lung", "heart", "kidney", "liver"],
        gold_index=0,
        n_perturbations=8,
        rng=np.random.default_rng(0),
    )
    assert res.gap > 1.0  # canonical >> perturbed


def test_slot_guessing_returns_logprob_for_masked_word():
    scorer = MockVLMScorer(seed=3)
    res = slot_guessing_score(
        scorer,
        image_path=None,
        prompt="What lobe of the lung contains the lesion?",
        rng=np.random.default_rng(0),
    )
    assert res.masked_token  # picked something
    assert res.n_question_tokens == len(
        "What lobe of the lung contains the lesion?".split()
    )
    assert res.logprob < 0
