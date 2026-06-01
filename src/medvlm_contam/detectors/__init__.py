"""Contamination detectors.

Pure-numeric detectors (operate on per-example or per-token log-probs):
    - :mod:`exchangeability` — Oren et al. (ICLR'24) shuffle test
    - :mod:`mink_pp`         — Zhang et al. (EMNLP'24) Min-K%++ membership score
    - :mod:`loss_attack`     — Yeom et al. (CSF'18) / Carlini et al. (ICLR'23)
      classical loss-based membership-inference baseline

Detectors that depend on the model interface (image / cross-modal):
    - :mod:`mm_detect`       — MM-Detect (Song et al., EMNLP'25 Findings) ports
    - :mod:`image_nn`        — image-side near-duplicate detection
"""

from .exchangeability import exchangeability_test, ExchangeabilityResult
from .mink_pp import mink_pp_score, mink_pp_batch
from .loss_attack import loss_attack_score, loss_attack_batch
from .mm_detect import (
    OptionPerturbationResult,
    SlotGuessingResult,
    option_perturbation_score,
    slot_guessing_score,
)
from .image_nn import (
    ImageNNFlags,
    calibrate_threshold,
    detect_image_duplicates,
)

__all__ = [
    "exchangeability_test",
    "ExchangeabilityResult",
    "mink_pp_score",
    "mink_pp_batch",
    "loss_attack_score",
    "loss_attack_batch",
    "option_perturbation_score",
    "OptionPerturbationResult",
    "slot_guessing_score",
    "SlotGuessingResult",
    "ImageNNFlags",
    "calibrate_threshold",
    "detect_image_duplicates",
]
