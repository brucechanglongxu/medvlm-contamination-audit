"""Loss-attack membership-inference baseline.

References:
    - Yeom et al., "Privacy Risk in Machine Learning: Analyzing the
      Connection to Overfitting", CSF 2018.
    - Carlini et al., "Quantifying Memorization Across Neural Language
      Models", ICLR 2023 (arXiv:2202.07646).

The classical loss-attack score for an example is simply the negative
per-token loss (i.e. the mean log-probability the model assigns to the
answer): memorized examples have abnormally low loss.

This module is intentionally trivial — it operates on values that the
audit driver already writes to JSONL (``mean_logprob``), so it requires
no extra model inference. It is included as the canonical baseline
against which Min-K%%++ (Zhang et al.) is benchmarked, and so that the
flag table can report agreement between two statistically distinct
detectors (Min-K%%++ uses *per-position* moments; loss-attack uses only
the chosen-token loss).
"""

from __future__ import annotations

import numpy as np


def loss_attack_score(mean_logprob: float) -> float:
    """Return the loss-attack score for one example.

    Higher score → more memorization evidence (lower loss).
    """
    return float(-mean_logprob)


def loss_attack_batch(mean_logprobs: np.ndarray) -> np.ndarray:
    """Vectorized loss-attack score over a batch of examples."""
    return -np.asarray(mean_logprobs, dtype=np.float64)
