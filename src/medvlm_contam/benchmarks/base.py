"""Common types for benchmark loaders."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional


@dataclass(frozen=True)
class BenchmarkExample:
    """One audited example.

    Attributes
    ----------
    example_id:
        Stable identifier within the benchmark (used as the join key in
        ``medical_contam_flags_2026.jsonl``).
    image_path:
        Path to the image file on disk. ``None`` for text-only items.
    prompt:
        Question / instruction text in canonical form (no chat template).
    answer:
        Reference answer string. For multiple-choice, the canonical letter or
        the full chosen option text (the loader documents which).
    choices:
        Optional choice strings (for MCQ benchmarks).
    metadata:
        Arbitrary loader-defined fields (split, subspecialty, source URL, etc.).
        Kept opaque to detectors; written through to the output JSONL.
    """

    example_id: str
    image_path: Optional[Path]
    prompt: str
    answer: str
    choices: Optional[tuple[str, ...]] = None
    metadata: dict = field(default_factory=dict)


class Benchmark:
    """Abstract benchmark loader.

    Subclasses implement :meth:`__iter__` and :attr:`name`. The canonical
    iteration order matters for the Oren-style exchangeability test: it MUST
    match the order in which the benchmark would have appeared in a scraped
    dump (e.g. the upstream file order, alphabetical filename, etc.). The
    loader documents the chosen canonical order in its docstring.
    """

    name: str = "base"
    canonical_order_description: str = "subclass must override"

    def __iter__(self) -> Iterator[BenchmarkExample]:  # pragma: no cover - abstract
        raise NotImplementedError

    def __len__(self) -> int:  # pragma: no cover - abstract
        raise NotImplementedError

    def to_list(self) -> list[BenchmarkExample]:
        return list(self)


def write_jsonl(path: Path, examples: Iterable[BenchmarkExample]) -> int:
    """Materialize examples to a JSONL sidecar for reproducibility."""
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            row = {
                "example_id": ex.example_id,
                "image_path": str(ex.image_path) if ex.image_path else None,
                "prompt": ex.prompt,
                "answer": ex.answer,
                "choices": list(ex.choices) if ex.choices else None,
                "metadata": ex.metadata,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n
