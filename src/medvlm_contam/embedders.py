"""Image embedders for the image-side near-duplicate detector.

Two implementations are provided:

- :class:`PHashEmbedder` — pure-numpy + Pillow perceptual hash (64-bit
  DCT-based pHash). Zero heavyweight deps; fast; works on CPU. Use as the
  default for quick first-pass duplicate flagging.
- :class:`OpenCLIPEmbedder` — semantic embeddings via ``open_clip``
  (SigLIP, OpenCLIP, etc.). Catches paraphrased / re-rendered duplicates
  that pHash misses. Optional dependency (``[embed]`` extra).

All embedders implement the :class:`ImageEmbedder` Protocol and return
``np.ndarray`` of shape ``(D,)`` per image. For pHash the array is a
binary vector (0/1) with ``D=64``; for SigLIP it's a unit-norm float32
vector with ``D=512`` or ``1024`` depending on the backbone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Protocol, Sequence

import numpy as np


class ImageEmbedder(Protocol):
    name: str
    dim: int
    metric: str  # "cosine" or "hamming"

    def embed(self, image_path: Path) -> np.ndarray:  # pragma: no cover
        ...

    def embed_many(self, image_paths: Sequence[Path]) -> np.ndarray:  # pragma: no cover
        ...


# ---------------------------------------------------------------------------
# pHash — DCT-based perceptual hash
# ---------------------------------------------------------------------------


class PHashEmbedder:
    """64-bit DCT perceptual hash.

    Standard recipe:
        1. Convert to grayscale, resize to 32x32.
        2. Apply 2-D DCT-II.
        3. Take the top-left 8x8 low-frequency block (skip DC).
        4. Threshold by the block median to get a 64-bit binary vector.

    Hamming distance between two pHashes correlates with perceptual
    similarity. We return the bits as a length-64 ``uint8`` array of 0/1.
    """

    name = "phash64"
    dim = 64
    metric = "hamming"

    def __init__(self, size: int = 32, low_block: int = 8) -> None:
        self.size = size
        self.low_block = low_block

    def embed(self, image_path: Path) -> np.ndarray:
        from PIL import Image

        img = Image.open(image_path).convert("L").resize(
            (self.size, self.size), Image.BICUBIC
        )
        arr = np.asarray(img, dtype=np.float64)
        dct = _dct2(arr)
        block = dct[: self.low_block, : self.low_block].flatten()
        # Drop the DC component (index 0) before computing the median.
        ref = np.median(block[1:])
        bits = (block > ref).astype(np.uint8)
        return bits  # length 64

    def embed_many(self, image_paths: Sequence[Path]) -> np.ndarray:
        return np.stack([self.embed(p) for p in image_paths], axis=0)


def _dct2(a: np.ndarray) -> np.ndarray:
    """2-D DCT-II via 1-D DCTs along each axis."""
    return _dct1(_dct1(a, axis=0), axis=1)


def _dct1(a: np.ndarray, axis: int) -> np.ndarray:
    # Lazy import scipy.fft so the package works without scipy on path.
    try:
        from scipy.fft import dct  # type: ignore

        return dct(a, type=2, axis=axis, norm="ortho")
    except ImportError:  # pragma: no cover
        # Fall back to a slow O(N^2) implementation.
        n = a.shape[axis]
        k = np.arange(n)
        m = np.arange(n)
        basis = np.cos(np.pi * (2 * m + 1)[:, None] * k[None, :] / (2 * n))
        moved = np.moveaxis(a, axis, -1)
        out = moved @ basis
        scale = np.full(n, np.sqrt(2.0 / n))
        scale[0] = np.sqrt(1.0 / n)
        out = out * scale
        return np.moveaxis(out, -1, axis)


# ---------------------------------------------------------------------------
# open_clip / SigLIP embedder (optional)
# ---------------------------------------------------------------------------


class OpenCLIPEmbedder:
    """Semantic embedder via ``open_clip``.

    Defaults to SigLIP-So400m, which is the strongest open-weights image
    encoder in the 400M-param range as of 2026. For larger throughput on
    H100, use ``model_name="ViT-L-14"`` with ``pretrained="openai"``.

    Lazily imports torch + open_clip so the rest of the package stays
    importable without those deps.
    """

    metric = "cosine"

    def __init__(
        self,
        model_name: str = "ViT-SO400M-14-SigLIP",
        pretrained: str = "webli",
        device: str = "auto",
        batch_size: int = 32,
    ) -> None:
        import open_clip  # type: ignore
        import torch

        self.name = f"openclip::{model_name}::{pretrained}"
        device = self._resolve_device(device)
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        model = model.to(device).eval()
        self._model = model
        self._preprocess = preprocess
        self._device = device
        self._torch = torch
        self.batch_size = batch_size
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224, device=device)
            self.dim = int(model.encode_image(dummy).shape[-1])

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def _load_batch(self, paths: Sequence[Path]):
        from PIL import Image

        imgs = [Image.open(p).convert("RGB") for p in paths]
        batch = self._torch.stack([self._preprocess(im) for im in imgs])
        return batch.to(self._device)

    def embed(self, image_path: Path) -> np.ndarray:
        return self.embed_many([image_path])[0]

    def embed_many(self, image_paths: Sequence[Path]) -> np.ndarray:
        torch = self._torch
        outs = []
        with torch.no_grad():
            for i in range(0, len(image_paths), self.batch_size):
                chunk = list(image_paths[i : i + self.batch_size])
                batch = self._load_batch(chunk)
                feats = self._model.encode_image(batch).float()
                feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                outs.append(feats.cpu().numpy())
        return np.concatenate(outs, axis=0)


# ---------------------------------------------------------------------------
# Distance / similarity utilities
# ---------------------------------------------------------------------------


def pairwise_distance(
    query: np.ndarray, corpus: np.ndarray, *, metric: str
) -> np.ndarray:
    """Return a ``(n_query, n_corpus)`` distance matrix.

    ``metric == "cosine"``  — assumes unit-norm rows; returns ``1 - cos``.
    ``metric == "hamming"`` — counts differing bits, returns int array.
    """
    if metric == "cosine":
        sim = query @ corpus.T
        return 1.0 - sim
    if metric == "hamming":
        # Broadcast XOR over the binary vectors; sum along the bit axis.
        q = query.astype(np.uint8)[:, None, :]
        c = corpus.astype(np.uint8)[None, :, :]
        return (q != c).sum(axis=-1)
    raise ValueError(f"unknown metric: {metric}")


def nearest_neighbor(
    query: np.ndarray, corpus: np.ndarray, *, metric: str
) -> tuple[np.ndarray, np.ndarray]:
    """For each query row, return (nn_index, nn_distance) into ``corpus``."""
    d = pairwise_distance(query, corpus, metric=metric)
    idx = d.argmin(axis=1)
    dist = d[np.arange(d.shape[0]), idx]
    return idx, dist
