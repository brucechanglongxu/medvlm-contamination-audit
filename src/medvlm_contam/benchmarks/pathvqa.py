"""PathVQA loader.

PathVQA (He et al., 2020) is a fully-public pathology VQA benchmark. We
load via the HuggingFace mirror ``flaviagiammarino/path-vqa``.

Each item has a question, an answer (open-ended, often yes/no), and an
image stored as a PIL image inside the HF dataset. Because the upstream
images are embedded in the parquet, the loader materializes them to disk
under ``root / images / {example_id}.png`` on first access so the rest of
the pipeline can treat them as files.

Canonical iteration order is the upstream split order.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterator, Optional

from .base import Benchmark, BenchmarkExample


class PathVQA(Benchmark):
    name = "pathvqa"
    canonical_order_description = (
        "Upstream HuggingFace row order from flaviagiammarino/path-vqa, "
        "per split, no re-shuffling."
    )

    def __init__(
        self,
        root: str | Path = "data/raw/pathvqa",
        split: str = "test",
        hf_repo: str = "flaviagiammarino/path-vqa",
        max_examples: Optional[int] = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.hf_repo = hf_repo
        self.max_examples = max_examples
        self.images_dir = self.root / "images" / split
        self._dataset = None
        self._length: Optional[int] = None

    def _load_dataset(self):
        if self._dataset is not None:
            return self._dataset
        try:
            from datasets import load_dataset  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "datasets not installed. `pip install medvlm-contam[hf]`."
            ) from e
        self._dataset = load_dataset(self.hf_repo, split=self.split, cache_dir=str(self.root))
        if self.max_examples is not None:
            self._dataset = self._dataset.select(range(self.max_examples))
        self._length = len(self._dataset)
        return self._dataset

    def _materialize_image(self, image, example_id: str) -> Path:
        self.images_dir.mkdir(parents=True, exist_ok=True)
        # Hash example_id for filesystem safety.
        digest = hashlib.sha1(example_id.encode("utf-8")).hexdigest()[:16]
        out = self.images_dir / f"{digest}.png"
        if out.exists() and out.stat().st_size > 0:
            # Validate the on-disk file is a readable image; if a previous run
            # left a corrupt/half-written PNG, drop it and re-materialize.
            try:
                from PIL import Image as _PILImage  # local import to avoid cycles
                with _PILImage.open(out) as probe:
                    probe.verify()
                return out
            except Exception:
                try:
                    out.unlink()
                except FileNotFoundError:
                    pass
        # PNG only supports a subset of modes (RGB, RGBA, L, LA, P, I, 1).
        # Some PathVQA images are CMYK / palette / RGBA-with-alpha which can
        # break PIL's PNG encoder or downstream consumers — normalize to RGB.
        img = image
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        try:
            img.save(out, format="PNG")
        except Exception:
            # Best-effort cleanup so we don't leave a half-written file that
            # later raises UnidentifiedImageError.
            if out.exists():
                out.unlink()
            raise
        return out

    def __iter__(self) -> Iterator[BenchmarkExample]:
        ds = self._load_dataset()
        for i, row in enumerate(ds):
            qid = f"pathvqa::{self.split}::{i:06d}"
            image = row.get("image")
            image_path = self._materialize_image(image, qid) if image is not None else None
            yield BenchmarkExample(
                example_id=qid,
                image_path=image_path,
                prompt=row["question"],
                answer=str(row["answer"]),
                choices=None,
                metadata={"split": self.split, "row_index": i},
            )

    def __len__(self) -> int:
        if self._length is None:
            self._load_dataset()
        return int(self._length or 0)
