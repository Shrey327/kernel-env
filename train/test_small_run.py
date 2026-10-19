import pytest

from small_run import compare


def episode(task, correct, speed):
    return {"task_id": task, "turns": [{"stage": "timed" if correct else "compile",
            "correct": correct, "speedup": speed}]}


def test_comparison_counts_failures_and_separates_groups():
    before = [episode("a", True, 0.8), episode("b", False, 0)]
    after = [episode("a", False, 0), episode("b", True, 0.5)]
    result = compare(before, after, {"all": ["a", "b"], "heldout": ["b"]})
    assert result["all"]["before"]["mean_best_speedup_including_failures"] == 0.4
    assert result["all"]["after"]["mean_best_speedup_including_failures"] == 0.25
    assert result["heldout"]["before"]["correct_pct"] == 0
    assert result["heldout"]["after"]["correct_pct"] == 100
    with pytest.raises(ValueError, match="task sets"):
        compare(before, after[:1], {"all": ["a", "b"]})
    with pytest.raises(ValueError, match="infrastructure"):
        compare(before, [{**e, "error": "disconnect"} for e in after], {"all": ["a", "b"]})
    with pytest.raises(ValueError, match="missing or repeated"):
        compare(before, after, {"all": ["a", "b", "c"]})
