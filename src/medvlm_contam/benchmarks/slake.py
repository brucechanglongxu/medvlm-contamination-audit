"""SLAKE loader.

SLAKE (Liu et al., ISBI 2021) is a fully-public bilingual medical VQA
benchmark. We load the English split from the HuggingFace mirror
``BoKelvin/SLAKE`` (default split). The canonical iteration order is the
upstream row order, which mirrors the JSON files in the original release —
this is the order that would be preserved by any naive scraper.

Usage::

    from medvlm_contam.benchmarks.slake import SlakeEnglish
    for ex in SlakeEnglish(root="data/raw/slake"):
        ...

The constructor downloads on first use via ``datasets.load_dataset`` and
caches into ``root``. If the HF dataset is unreachable, point ``root`` to
a local directory containing the canonical ``train.json`` / ``test.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Optional

from .base import Benchmark, BenchmarkExample


class SlakeEnglish(Benchmark):
    name = "slake_en"
    canonical_order_description = (
        "Upstream JSON row order from BoKelvin/SLAKE (train+validation+test), "
        "filtered to English (q_lang == 'en')."
    )

    def __init__(
        self,
        root: str | Path = "data/raw/slake",
        split: str = "test",
        hf_repo: str = "BoKelvin/SLAKE",
        max_examples: Optional[int] = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.hf_repo = hf_repo
        self.max_examples = max_examples
        self._records: Optional[list[dict]] = None

    # ------------------------------------------------------------------ loading

    def _load_records(self) -> list[dict]:
        if self._records is not None:
            return self._records

        local_json = self.root / f"{self.split}.json"
        if local_json.exists():
            with local_json.open("r", encoding="utf-8") as f:
                records = json.load(f)
        else:
            records = self._load_from_hf()

        # English only, preserve upstream order.
        en = [r for r in records if r.get("q_lang", "en") == "en"]
        if self.max_examples is not None:
            en = en[: self.max_examples]
        self._records = en
        return en

    def _load_from_hf(self) -> list[dict]:
        try:
            from datasets import load_dataset  # type: ignore
        except ImportError as e:  # pragma: no cover - environmental
            raise RuntimeError(
                "datasets not installed. `pip install medvlm-contam[hf]` "
                "or place the SLAKE JSON files under `root`."
            ) from e

        ds = load_dataset(self.hf_repo, split=self.split, cache_dir=str(self.root))
        return [dict(row) for row in ds]

    # --------------------------------------------------------------- iteration

    def __iter__(self) -> Iterator[BenchmarkExample]:
        for rec in self._load_records():
            qid = str(rec.get("qid") or rec.get("id") or rec.get("question_id"))
            img_name = rec.get("img_name") or rec.get("image")
            image_path = (self.root / "imgs" / img_name) if img_name else None
            yield BenchmarkExample(
                example_id=f"slake_en::{self.split}::{qid}",
                image_path=image_path,
                prompt=rec["question"],
                answer=str(rec["answer"]),
                choices=None,  # SLAKE is open-ended / closed (yes/no), not MCQ
                metadata={
                    "split": self.split,
                    "answer_type": rec.get("answer_type"),
                    "modality": rec.get("modality"),
                    "location": rec.get("location"),
                    "content_type": rec.get("content_type"),
                },
            )

    def __len__(self) -> int:
        return len(self._load_records())
