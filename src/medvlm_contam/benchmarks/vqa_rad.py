"""VQA-RAD loader.

VQA-RAD (Lau et al., 2018) is a fully-public radiology VQA benchmark
of ~3500 question/answer pairs over ~315 radiology images (head CT,
chest X-ray, abdomen). We load via the HuggingFace mirror
``flaviagiammarino/vqa-rad`` — same maintainer / schema convention as
the PathVQA mirror, with ``image`` as an embedded PIL image and
``question`` / ``answer`` text fields.

Like PathVQA, images are materialized to disk on first iteration so
the rest of the pipeline can treat them as files. Canonical iteration
order is the upstream split order.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterator, Optional

from .base import Benchmark, BenchmarkExample


class VQARad(Benchmark):
    name = "vqa_rad"
    canonical_order_description = (
        "Upstream HuggingFace row order from flaviagiammarino/vqa-rad, "
        "per split, no re-shuffling."
    )

    def __init__(
        self,
        root: str | Path = "data/raw/vqa_rad",
        split: str = "test",
        hf_repo: str = "flaviagiammarino/vqa-rad",
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
        self._dataset = load_dataset(
            self.hf_repo, split=self.split, cache_dir=str(self.root)
        )
        if self.max_examples is not None:
            self._dataset = self._dataset.select(range(self.max_examples))
        self._length = len(self._dataset)
        return self._dataset

    def _materialize_image(self, image, example_id: str) -> Path:
        self.images_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha1(example_id.encode("utf-8")).hexdigest()[:16]
        out = self.images_dir / f"{digest}.png"
        if out.exists() and out.stat().st_size > 0:
            try:
                from PIL import Image as _PILImage
                with _PILImage.open(out) as probe:
                    probe.verify()
                return out
            except Exception:
                try:
                    out.unlink()
                except FileNotFoundError:
                    pass
        img = image
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        try:
            img.save(out, format="PNG")
        except Exception:
            if out.exists():
                out.unlink()
            raise
        return out

    def __iter__(self) -> Iterator[BenchmarkExample]:
        ds = self._load_dataset()
        for i, row in enumerate(ds):
            qid = f"vqa_rad::{self.split}::{i:06d}"
            image = row.get("image")
            image_path = (
                self._materialize_image(image, qid) if image is not None else None
            )
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
