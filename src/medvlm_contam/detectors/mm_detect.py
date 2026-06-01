"""MM-Detect (Song et al., EMNLP'25 Findings) cross-modal contamination probes.

Reference: arXiv:2411.03823.

We port two of MM-Detect's probes to medical VLMs:

1. **Option perturbation** (``option_perturbation_score``). For an MCQ item,
   the model assigns probability to each labeled choice. Under
   contamination the model's preference for the *gold* choice survives
   relabeling the option letters (A↔C, B↔D, …) — i.e. the model has
   memorized "lung" rather than "the option labeled (A)". Under the null,
   relabeling drops the preference toward chance.

   Operationalized as the *relabel-invariance gap*: the model's
   probability mass on the **gold answer text** under canonical labeling
   minus the same quantity under a permuted labeling, averaged over a
   sample of permutations. Large positive values indicate memorization.

2. **Slot guessing** (``slot_guessing_score``). For an open-ended item, we
   mask out a content-bearing token in the **question** (not the image)
   and ask whether the model can still reconstruct it conditional on
   (image, masked-question). MM-Detect's finding: contaminated examples
   show abnormally high reconstruction accuracy because the model has
   memorized the canonical question wording paired with the image. We
   return the model's log-probability of the original masked token under
   the masked context. Calibrated against a non-contaminated control
   benchmark to extract a per-example significance.

Both probes operate through the :class:`VLMScorer` interface — they do not
care which concrete VLM is plugged in.
"""

from __future__ import annotations

import string
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ..models.base import VLMScorer


# ---------------------------------------------------------------------------
# Option perturbation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OptionPerturbationResult:
    canonical_logprob_gold: float
    perturbed_logprob_gold_mean: float
    gap: float  # canonical - perturbed_mean ; large positive ⇒ memorization
    n_perturbations: int


def _format_mcq(prompt: str, choices: Sequence[str], letters: Sequence[str]) -> str:
    """Render an MCQ in 'A) lung\\nB) heart\\n...' canonical form."""
    body = "\n".join(f"{l}) {c}" for l, c in zip(letters, choices))
    return f"{prompt}\n{body}"


def option_perturbation_score(
    scorer: VLMScorer,
    *,
    image_path: Optional[Path],
    prompt: str,
    choices: Sequence[str],
    gold_index: int,
    n_perturbations: int = 8,
    rng: Optional[np.random.Generator] = None,
) -> OptionPerturbationResult:
    """Run the option-perturbation probe on one MCQ example.

    The scored answer is the **gold option text** (not the option letter),
    so that the probe measures memorization of the *answer content*, not
    of the label.
    """
    if not (0 <= gold_index < len(choices)):
        raise ValueError(f"gold_index {gold_index} out of range for {len(choices)} choices")
    if len(choices) < 2:
        raise ValueError("option-perturbation requires at least 2 choices")

    rng = rng if rng is not None else np.random.default_rng(0)
    letters = list(string.ascii_uppercase[: len(choices)])
    gold_text = choices[gold_index]

    canonical_prompt = _format_mcq(prompt, choices, letters)
    canonical = scorer.score(image_path, canonical_prompt, gold_text)
    canonical_lp = canonical.sum_logprob

    perturbed_lps = []
    for _ in range(n_perturbations):
        perm = list(range(len(choices)))
        rng.shuffle(perm)
        # Skip identity permutations.
        if perm == list(range(len(choices))):
            continue
        perm_choices = [choices[i] for i in perm]
        perm_prompt = _format_mcq(prompt, perm_choices, letters)
        perturbed = scorer.score(image_path, perm_prompt, gold_text)
        perturbed_lps.append(perturbed.sum_logprob)

    perturbed_mean = float(np.mean(perturbed_lps)) if perturbed_lps else canonical_lp
    return OptionPerturbationResult(
        canonical_logprob_gold=float(canonical_lp),
        perturbed_logprob_gold_mean=perturbed_mean,
        gap=float(canonical_lp - perturbed_mean),
        n_perturbations=len(perturbed_lps),
    )


# ---------------------------------------------------------------------------
# Slot guessing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SlotGuessingResult:
    masked_token: str
    logprob: float  # log p(token | image, prompt with the token replaced by [MASK])
    n_question_tokens: int


_MASK_TOKEN = "[MASK]"


def slot_guessing_score(
    scorer: VLMScorer,
    *,
    image_path: Optional[Path],
    prompt: str,
    mask_word: Optional[str] = None,
    rng: Optional[np.random.Generator] = None,
) -> SlotGuessingResult:
    """Mask one content word in ``prompt`` and score the model's recovery.

    If ``mask_word`` is not given, we deterministically (per ``rng``)
    select the longest alphabetic word as a proxy for "content-bearing",
    falling back to the median-length word for ties. Numbers and
    punctuation are excluded.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    tokens = prompt.split()
    candidates = [
        (i, t) for i, t in enumerate(tokens) if t.strip(string.punctuation).isalpha()
    ]
    if not candidates:
        raise ValueError("no maskable content word found in prompt")

    if mask_word is not None:
        idx = next((i for i, t in candidates if t.strip(string.punctuation) == mask_word), None)
        if idx is None:
            raise ValueError(f"mask_word {mask_word!r} not found in prompt")
    else:
        # Pick the longest word; break ties by RNG for reproducibility.
        max_len = max(len(t) for _, t in candidates)
        longest = [(i, t) for i, t in candidates if len(t) == max_len]
        idx, _ = longest[int(rng.integers(len(longest)))]

    original = tokens[idx]
    core = original.strip(string.punctuation)
    masked_tokens = tokens.copy()
    masked_tokens[idx] = original.replace(core, _MASK_TOKEN)
    masked_prompt = (
        " ".join(masked_tokens)
        + f"\nWhat single word should replace {_MASK_TOKEN}? Answer with the word only."
    )
    scored = scorer.score(image_path, masked_prompt, core)
    return SlotGuessingResult(
        masked_token=core,
        logprob=scored.sum_logprob,
        n_question_tokens=len(tokens),
    )
