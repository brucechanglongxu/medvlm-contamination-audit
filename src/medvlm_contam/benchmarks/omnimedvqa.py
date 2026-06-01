"""OmniMedVQA loader.

OmniMedVQA (Hu et al., 2024) is a large-scale multimodal medical VQA
benchmark with ~127k MCQ questions spanning 12 modalities (CT, MRI,
X-ray, OCT, ultrasound, dermoscopy, microscopy, fundus, endoscopy,
pathology, retinography, mammography) drawn from 73 source datasets.

We load via the HuggingFace mirror ``foreverbeliever/OmniMedVQA``.
Schema (per row):
    image:            embedded PIL image
    question:         str
    option_A/B/C/D:   str  (some rows have fewer options; missing => "")
    gt_answer:        str  (the letter, e.g. "A")
    modality_type:    str
    dataset:          str  (source sub-dataset name)

Canonical iteration order is the upstream row order on the split, no
re-shuffling --- consistent with the Oren-style exchangeability test
convention used by the other benchmark loaders in this package.

The audit's answer-scoring detectors are answer-conditioned, so we
serialize each example as a closed-form MCQ prompt and use the gold
option *text* (not just the letter) as the target answer. This makes
the per-token log-likelihood comparable across models with different
tokenization of single letters.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterator, Optional

from .base import Benchmark, BenchmarkExample


_OPTION_KEYS = ("option_A", "option_B", "option_C", "option_D")


class OmniMedVQA(Benchmark):
    name = "omnimedvqa"
    canonical_order_description = (
        "Upstream HuggingFace row order from foreverbeliever/OmniMedVQA, "
        "per split, no re-shuffling."
    )

    def __init__(
        self,
        root: str | Path = "data/raw/omnimedvqa",
        split: str = "train",
        hf_repo: str = "foreverbeliever/OmniMedVQA",
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

    @staticmethod
    def _format_prompt(question: str, options: list[tuple[str, str]]) -> str:
        """Render the MCQ as a single prompt string.

        ``options`` is a list of (letter, text) pairs in canonical order.
        """
        lines = [question.strip()]
        for letter, text in options:
            lines.append(f"({letter}) {text.strip()}")
        lines.append("Answer:")
        return "\n".join(lines)

    @staticmethod
    def _resolve_answer(row: dict, options: list[tuple[str, str]]) -> str:
        """Return the gold option *text* (not the letter).

        Falls back to the letter itself if the option text is empty or
        the gold letter does not match any option (defensive).
        """
        gold = str(row.get("gt_answer", "")).strip()
        for letter, text in options:
            if letter == gold and text:
                return text.strip()
        return gold

    def __iter__(self) -> Iterator[BenchmarkExample]:
        ds = self._load_dataset()
        n = len(ds)
        skipped = 0
        for i in range(n):
            qid = f"omnimedvqa::{self.split}::{i:06d}"
            # Per-row decode with try/except: OmniMedVQA contains a
            # small number of rows whose ``image`` blob is not a valid
            # image file (PIL raises UnidentifiedImageError). Skipping
            # them keeps the audit interpretable -- the canonical-order
            # exchangeability test is still defined on the surviving
            # row subset, and the skip count is reported in the run
            # log for reproducibility.
            try:
                row = ds[i]
            except Exception as e:
                skipped += 1
                print(
                    f"[omnimedvqa] skipping row {i}: decode error: "
                    f"{type(e).__name__}: {e}",
                    flush=True,
                )
                continue
            image = row.get("image")
            try:
                image_path = (
                    self._materialize_image(image, qid)
                    if image is not None
                    else None
                )
            except Exception as e:
                skipped += 1
                print(
                    f"[omnimedvqa] skipping row {i}: image materialise "
                    f"error: {type(e).__name__}: {e}",
                    flush=True,
                )
                continue
            options: list[tuple[str, str]] = []
            for key in _OPTION_KEYS:
                text = str(row.get(key, "") or "")
                if text:
                    letter = key.split("_", 1)[1]
                    options.append((letter, text))
            prompt = self._format_prompt(str(row.get("question", "")), options)
            answer = self._resolve_answer(row, options)
            yield BenchmarkExample(
                example_id=qid,
                image_path=image_path,
                prompt=prompt,
                answer=answer,
                choices=tuple(t for _, t in options) if options else None,
                metadata={
                    "split": self.split,
                    "row_index": i,
                    "modality_type": str(row.get("modality_type", "")),
                    "dataset": str(row.get("dataset", "")),
                    "gt_letter": str(row.get("gt_answer", "")),
                },
            )

    def __len__(self) -> int:
        if self._length is None:
            self._load_dataset()
        return int(self._length or 0)
