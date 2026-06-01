from pathlib import Path

from medvlm_contam.benchmarks.base import BenchmarkExample, write_jsonl


def test_benchmark_example_basic_fields():
    ex = BenchmarkExample(
        example_id="x::1",
        image_path=Path("/tmp/x.jpg"),
        prompt="What organ?",
        answer="lung",
        metadata={"split": "test"},
    )
    assert ex.example_id == "x::1"
    assert ex.metadata["split"] == "test"
    assert ex.choices is None


def test_write_jsonl_roundtrip(tmp_path):
    import json

    out = tmp_path / "examples.jsonl"
    n = write_jsonl(
        out,
        [
            BenchmarkExample("a::1", None, "q1", "a1", metadata={"k": 1}),
            BenchmarkExample("a::2", Path("/tmp/i.png"), "q2", "a2", choices=("a", "b")),
        ],
    )
    assert n == 2
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert rows[0]["example_id"] == "a::1"
    assert rows[1]["choices"] == ["a", "b"]
    assert rows[1]["image_path"] == "/tmp/i.png"
