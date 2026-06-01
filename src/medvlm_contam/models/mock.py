"""Deterministic mock VLM scorer for unit + integration tests.

Produces reproducible token-level log-probs as a function of (prompt,
answer) via a seeded RNG, without loading any real model. Used to test
the audit driver, MM-Detect probes, and contamination simulations.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from .base import ScoredAnswer, VLMScorer


class MockVLMScorer(VLMScorer):
    """Pure-Python VLM scorer mock.

    Parameters
    ----------
    name:
        Identifier written into the audit JSONL.
    memorized_substrings:
        If any of these substrings appears in (prompt, answer), the mock
        returns higher-than-baseline log-probs to simulate memorization.
    base_logp:
        Mean log-prob for non-memorized tokens.
    memorized_logp:
        Mean log-prob for memorized tokens (should be closer to 0).
    vocab_size:
        Vocabulary size used to construct synthetic top-K distributions.
    """

    def __init__(
        self,
        name: str = "mock-vlm",
        *,
        memorized_substrings: Sequence[str] = (),
        base_logp: float = -3.0,
        memorized_logp: float = -0.5,
        vocab_size: int = 32_000,
        seed: int = 0,
    ) -> None:
        self.name = name
        self.memorized_substrings = tuple(memorized_substrings)
        self.base_logp = float(base_logp)
        self.memorized_logp = float(memorized_logp)
        self.vocab_size = int(vocab_size)
        self._seed = int(seed)

    def _rng_for(self, *parts: str) -> np.random.Generator:
        key = "||".join(parts).encode("utf-8")
        digest = hashlib.sha256(key).digest()[:8]
        seed = (int.from_bytes(digest, "big") ^ self._seed) & 0xFFFFFFFF
        return np.random.default_rng(seed)

    def _is_memorized(self, prompt: str, answer: str) -> bool:
        haystack = (prompt + " " + answer).lower()
        return any(s.lower() in haystack for s in self.memorized_substrings)

    def score(
        self,
        image_path: Optional[Path],
        prompt: str,
        answer: str,
        *,
        topk: int = 0,
    ) -> ScoredAnswer:
        rng = self._rng_for(str(image_path), prompt, answer)
        # One synthetic token per whitespace-separated word, minimum 1.
        n_tokens = max(1, len(answer.split()))
        mu = self.memorized_logp if self._is_memorized(prompt, answer) else self.base_logp
        token_logprobs = rng.normal(loc=mu, scale=0.3, size=n_tokens)
        token_logprobs = np.minimum(token_logprobs, -1e-3)  # log-prob must be < 0
        # Synthetic token ids in [0, vocab_size).
        token_ids = rng.integers(0, self.vocab_size, size=n_tokens)

        topk_logprobs = None
        if topk and topk > 0:
            # Build a plausible top-K: chosen token at position 0, then a
            # decaying tail. Place the chosen-token logprob first per row.
            tail = -np.linspace(0.5, 5.0, topk)
            topk_logprobs = np.broadcast_to(tail, (n_tokens, topk)).copy()
            topk_logprobs[:, 0] = token_logprobs  # ensure chosen ≤ max of top-K
            topk_logprobs += rng.normal(0.0, 0.02, size=topk_logprobs.shape)

        return ScoredAnswer(
            token_ids=token_ids,
            token_logprobs=token_logprobs,
            topk_logprobs=topk_logprobs,
        )

    def score_many(
        self,
        items: Sequence[tuple[Optional[Path], str, str]],
        *,
        topk: int = 0,
    ) -> Iterable[ScoredAnswer]:
        for image_path, prompt, answer in items:
            yield self.score(image_path, prompt, answer, topk=topk)
