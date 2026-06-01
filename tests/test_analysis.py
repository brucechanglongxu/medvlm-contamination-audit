import json
from pathlib import Path

from medvlm_contam.analysis import (
    aggregate_flags,
    build_heatmap,
    flag_examples,
    load_audit_jsonl,
)


def _write_audit(path: Path, model: str, benchmark: str, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            base = {
                "benchmark": benchmark,
                "model": model,
                "example_id": r["example_id"],
                "n_answer_tokens": r.get("n_answer_tokens", 3),
                "sum_logprob": r.get("sum_logprob", -10.0),
                "mean_logprob": r.get("mean_logprob", -3.0),
                "mink_pp_score": r.get("mink_pp_score"),
                "elapsed_s": 0.01,
                "metadata": r.get("metadata", {}),
            }
            f.write(json.dumps(base) + "\n")
    return path


def _write_summary(path: Path, summary: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary))
    return path


def test_load_audit_jsonl_roundtrip(tmp_path):
    p = _write_audit(
        tmp_path / "a.jsonl",
        model="m1",
        benchmark="b1",
        rows=[
            {"example_id": "e1", "mink_pp_score": -0.5},
            {"example_id": "e2", "mink_pp_score": -2.1},
        ],
    )
    recs = load_audit_jsonl(p)
    assert len(recs) == 2
    assert recs[0].example_id == "e1"
    assert recs[0].mink_pp_score == -0.5
    assert recs[0].benchmark == "b1"


def test_build_heatmap_combines_summaries(tmp_path):
    s1 = _write_summary(
        tmp_path / "b1__m1.summary.json",
        {
            "benchmark": "b1",
            "model": "m1",
            "n_examples": 100,
            "exchangeability": {
                "canonical_loglik": -1000.0,
                "p_value": 0.003,
                "n_permutations": 1000,
                "n_examples": 100,
                "significant_at_0_01": True,
            },
        },
    )
    s2 = _write_summary(
        tmp_path / "b1__m2.summary.json",
        {
            "benchmark": "b1",
            "model": "m2",
            "n_examples": 100,
            "exchangeability": {
                "canonical_loglik": -1100.0,
                "p_value": 0.4,
                "n_permutations": 1000,
                "n_examples": 100,
                "significant_at_0_01": False,
            },
        },
    )
    s3 = _write_summary(
        tmp_path / "b2__m1.summary.json",
        {
            "benchmark": "b2",
            "model": "m1",
            "n_examples": 50,
            # No exchangeability block — should yield NaN.
        },
    )
    heat = build_heatmap([s1, s2, s3])
    assert heat["benchmarks"] == ["b1", "b2"]
    assert heat["models"] == ["m1", "m2"]
    # b1/m1 significant
    assert heat["significant_at_0_01"][0, 0]
    # b1/m2 not significant
    assert not heat["significant_at_0_01"][0, 1]
    # b2/m1 missing exchangeability ⇒ NaN p-value
    import numpy as np
    assert np.isnan(heat["p_value"][1, 0])
    assert heat["n_examples"][1, 0] == 50


def test_flag_examples_quantile_picks_top_fraction(tmp_path):
    p = _write_audit(
        tmp_path / "a.jsonl",
        model="m",
        benchmark="b",
        rows=[
            {"example_id": f"e{i}", "mink_pp_score": float(i)} for i in range(10)
        ],
    )
    recs = load_audit_jsonl(p)
    flagged = flag_examples(recs, mink_pp_quantile=0.20)  # top 20%
    assert set(flagged) == {"e8", "e9"}


def test_aggregate_flags_counts_multi_detector_overlap():
    table = aggregate_flags(
        {"detA": ["x1", "x2", "x3"], "detB": ["x2", "x4"]}
    )
    assert table["n_flagged_total"] == 4
    assert table["per_detector_counts"] == {"detA": 3, "detB": 2}
    assert table["n_flagged_by_multiple"] == 1  # x2
    assert table["flag_table"]["x2"] == ["detA", "detB"]
