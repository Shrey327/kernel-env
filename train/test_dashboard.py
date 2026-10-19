from dashboard import view


def test_live_view_distinguishes_rollouts_from_updates_and_escapes_code():
    ep = {"task_id": "task-a", "step": 0, "completions": ["<script>bad()</script>"],
          "turns": [{"stage": "timed", "correct": True, "speedup": 0.8, "reward": 0.3}]}
    snap = {"manifest": {"steps": 16, "groups": {"training_tasks": ["task-a"], "official_heldout": ["task-b"]}},
            "train": [ep], "before": [ep]}
    state = view(snap, cost=2.5)
    assert state["completed"] == 0 and not state["steps"][0]["updated"]
    assert state["kernel_evals"] == 2 and state["cost"] == 2.5
    assert state["groups"]["training_tasks"]["before"]["correct_pct"] == 100
    assert state["groups"]["official_heldout"]["before"]["episodes"] == 0
    assert "<script>bad()" not in state["sections"]["train"]
    assert "&lt;script&gt;" in state["sections"]["train"]
    state = view(snap, " 6%| | 1/16 [00:53<13:28]\nPHASE after:")
    assert state["steps"][0]["updated"] and state["phase"] == "Final evaluation"
    state = view(snap, '{"event":"step","task_id":"task-b","turn":1,"stage":"compile"}\n')
    assert state["kernel_evals"] == 1 and "task-b" in state["activity"]
    snap["trainer"] = {"global_step": 1, "log_history": [{"step": 1, "epoch": 0.25, "loss": -0.02,
                       "reward": 0.3, "learning_rate": 5e-5, "grad_norm": 0.01}]}
    state = view(snap)
    assert state["metrics"][1]["loss"] == -0.02 and state["steps"][0]["epoch"] == 0.25
    assert state["steps"][0]["learning_rate"] == 5e-5
