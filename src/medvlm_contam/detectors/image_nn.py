"""Image-side near-duplicate contamination detector (MM-Detect Method C).

Given (a) the set of benchmark images and (b) a corpus that proxies the
model's pretraining mix (e.g. a LAION-COCO sample for general VLMs, or
PMC-OA for medical VLMs), embed both with an :class:`ImageEmbedder`,
nearest-neighbor each benchmark image into the corpus, and flag every
benchmark image whose NN distance falls below a calibrated threshold.

The threshold is calibrated against a **null corpus**: a held-out set of
benchmark images known NOT to be in the pretraining mix (e.g. a
post-cutoff benchmark or a synthetic perturbation set). The flagging
FPR is set to a user-specified ``alpha`` (default 0.01).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ..embedders import ImageEmbedder, nearest_neighbor


@dataclass
class ImageNNFlags:
    benchmark_ids: list[str]
    nn_distance: np.ndarray  # (N,) distance to nearest corpus image
    nn_index: np.ndarray  # (N,) index into the corpus
    threshold: float
    flagged_mask: np.ndarray  # (N,) bool
    embedder_name: str
    metric: str
    alpha: float
    null_distance_quantiles: dict = field(default_factory=dict)

    @property
    def n_flagged(self) -> int:
        return int(self.flagged_mask.sum())

    @property
    def fraction_flagged(self) -> float:
        n = self.flagged_mask.size
        return float(self.n_flagged / n) if n else 0.0


def calibrate_threshold(
    null_distances: np.ndarray, *, alpha: float = 0.01
) -> tuple[float, dict]:
    """Pick the threshold so the null FPR equals ``alpha``.

    Distances are "smaller is more similar", so the threshold is the
    ``alpha``-quantile of the null distance distribution: any benchmark
    image whose NN distance is at most this value would have happened by
    chance with probability ``alpha`` under the null.
    """
    null = np.asarray(null_distances, dtype=np.float64).ravel()
    if null.size == 0:
        raise ValueError("null distance array is empty")
    threshold = float(np.quantile(null, alpha))
    quantiles = {
        "q01": float(np.quantile(null, 0.01)),
        "q05": float(np.quantile(null, 0.05)),
        "q25": float(np.quantile(null, 0.25)),
        "q50": float(np.median(null)),
        "min": float(null.min()),
        "max": float(null.max()),
    }
    return threshold, quantiles


def detect_image_duplicates(
    embedder: ImageEmbedder,
    *,
    benchmark_image_paths: Sequence[Path],
    benchmark_ids: Sequence[str],
    corpus_image_paths: Sequence[Path],
    null_image_paths: Optional[Sequence[Path]] = None,
    null_threshold: Optional[float] = None,
    alpha: float = 0.01,
) -> ImageNNFlags:
    """Run the full Detector-C pipeline end-to-end.

    Parameters
    ----------
    embedder:
        Any :class:`ImageEmbedder`. pHash for fast first-pass, SigLIP /
        OpenCLIP for semantic duplicates.
    benchmark_image_paths, benchmark_ids:
        The images being audited and their stable IDs.
    corpus_image_paths:
        The proxy pretraining corpus.
    null_image_paths:
        Held-out benchmark or post-cutoff images used to calibrate the
        per-detector FPR. Required if ``null_threshold`` is not given.
    null_threshold:
        Pre-computed threshold to bypass calibration (useful when the same
        embedder + corpus is reused across many audits).
    alpha:
        Target null false-positive rate.
    """
    if len(benchmark_image_paths) != len(benchmark_ids):
        raise ValueError("benchmark_image_paths and benchmark_ids must have same length")

    metric = embedder.metric

    bench_emb = embedder.embed_many(list(benchmark_image_paths))
    corpus_emb = embedder.embed_many(list(corpus_image_paths))
    nn_idx, nn_dist = nearest_neighbor(bench_emb, corpus_emb, metric=metric)

    null_quantiles: dict = {}
    if null_threshold is None:
        if null_image_paths is None:
            raise ValueError(
                "either null_image_paths or null_threshold must be provided"
            )
        null_emb = embedder.embed_many(list(null_image_paths))
        _, null_dist = nearest_neighbor(null_emb, corpus_emb, metric=metric)
        null_threshold, null_quantiles = calibrate_threshold(null_dist, alpha=alpha)

    flagged = nn_dist <= null_threshold

    return ImageNNFlags(
        benchmark_ids=list(benchmark_ids),
        nn_distance=nn_dist,
        nn_index=nn_idx,
        threshold=float(null_threshold),
        flagged_mask=flagged,
        embedder_name=embedder.name,
        metric=metric,
        alpha=float(alpha),
        null_distance_quantiles=null_quantiles,
    )
