import numpy as np

from medvlm_contam.clean_benchmark import (
    clean_subset_ids,
    rank_churn,
    union_flagged_ids,
)


def test_union_flagged_ids_dedupes_across_detectors():
    out = union_flagged_ids(
        {"exch": ["a", "b"], "mink": ["b", "c"], "image_nn": ["d"]}
    )
    assert out == {"a", "b", "c", "d"}


def test_clean_subset_preserves_order():
    ids = ["a", "b", "c", "d", "e"]
    assert clean_subset_ids(ids, ["c", "a"]) == ["b", "d", "e"]


def test_rank_churn_no_flags_is_identical_leaderboard():
    correctness = {
        "modelA": {"x1": True, "x2": True, "x3": False, "x4": True},
        "modelB": {"x1": True, "x2": False, "x3": True, "x4": False},
        "modelC": {"x1": False, "x2": False, "x3": False, "x4": True},
    }
    res = rank_churn(correctness, flagged_ids=[])
    assert res.kendall_tau == 1.0
    assert res.max_abs_delta == 0.0
    assert res.n_clean == res.n_full == 4


def test_rank_churn_flagging_inflated_items_reorders_leaderboard():
    # modelA gets the flagged items right (inflating its full accuracy);
    # modelB doesn't. Removing flagged items should drop modelA more.
    correctness = {
        "modelA": {f"x{i}": (i < 5) for i in range(10)},   # 50% overall, 100% on first 5
        "modelB": {f"x{i}": (i % 3 == 0) for i in range(10)},  # ~40% spread evenly
        "modelC": {f"x{i}": True for i in range(10)},  # always right (perfect)
    }
    flagged = [f"x{i}" for i in range(5)]  # the "contaminated" first 5
    res = rank_churn(correctness, flagged_ids=flagged)

    by_model = {m.model: m for m in res.per_model}
    # modelA's accuracy collapses to 0 on the clean subset.
    assert by_model["modelA"].full_accuracy == 0.5
    assert by_model["modelA"].clean_accuracy == 0.0
    assert by_model["modelA"].delta > 0.3
    # modelC is unaffected (perfect everywhere).
    assert by_model["modelC"].clean_accuracy == 1.0
    assert res.max_delta_model == "modelA"
    assert res.n_clean == 5
    assert res.n_full == 10


def test_rank_churn_mismatched_universes_errors():
    import pytest

    with pytest.raises(ValueError):
        rank_churn(
            {
                "modelA": {"x1": True, "x2": False},
                "modelB": {"x1": True},  # missing x2
            },
            flagged_ids=[],
        )
