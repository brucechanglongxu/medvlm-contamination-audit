"""VLM scorer base interface.

Detectors depend only on this interface — never on a concrete model class —
so swapping in a new VLM is a one-file change.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Protocol, Sequence

import numpy as np


@dataclass(frozen=True)
class ScoredAnswer:
    """Token-level scoring of an answer sequence conditioned on (image, prompt).

    Attributes
    ----------
    token_ids:
        Integer ids of the answer tokens that were scored.
    token_logprobs:
        log p(token_t | image, prompt, token_{<t}) for each scored token.
        Shape (T,).
    topk_logprobs:
        For each scored position, the log-probabilities of the top-K tokens
        in the *full vocabulary* — used by Min-K%++ to estimate the per-token
        reference distribution. Shape (T, K). May be ``None`` if the scorer
        cannot expose them efficiently.
    """

    token_ids: np.ndarray
    token_logprobs: np.ndarray
    topk_logprobs: Optional[np.ndarray] = None

    @property
    def sum_logprob(self) -> float:
        return float(self.token_logprobs.sum())

    @property
    def mean_logprob(self) -> float:
        return float(self.token_logprobs.mean()) if self.token_logprobs.size else 0.0


class VLMScorer(Protocol):
    """Score answer tokens conditioned on an (image, prompt) pair.

    Implementations are free to batch internally. They MUST be deterministic
    given the same inputs (no sampling, no dropout) — detectors compare
    log-probs across canonical and shuffled orderings and require this.
    """

    name: str

    def score(
        self,
        image_path: Optional[Path],
        prompt: str,
        answer: str,
        *,
        topk: int = 0,
    ) -> ScoredAnswer:  # pragma: no cover - interface
        ...

    def score_many(
        self,
        items: Sequence[tuple[Optional[Path], str, str]],
        *,
        topk: int = 0,
    ) -> Iterable[ScoredAnswer]:  # pragma: no cover - interface
        ...
