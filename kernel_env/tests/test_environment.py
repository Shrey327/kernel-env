import json
from pathlib import Path

import pytest
from openenv.core.env_server.serialization import serialize_observation

from kernel_env import KernelAction, KernelEnv, KernelObservation
from kernel_env.server.evaluator import MockEvaluator
from kernel_env.server.kernel_env_environment import KernelEnvironment
from kernel_env.server.rubrics import kernel_rubric
from kernel_env.server.tasks import Task, load_tasks
from kernels import GOOD, with_forward

FIXTURES = Path(__file__).parent / "fixtures"


def timed(speedup):
    return {**MockEvaluator.DEFAULT, "speedup": speedup}


@pytest.fixture
def tasks():
    return load_tasks(str(FIXTURES), levels=(1,))


def test_load_tasks(tasks):
    assert list(tasks) == ["level1/19_ReLU"]
    assert tasks["level1/19_ReLU"].split in ("train", "eval")


def test_reset_returns_prompt(tasks):
    env = KernelEnvironment(MockEvaluator(), tasks, max_turns=2)
    obs = env.reset(task_id="level1/19_ReLU")
    assert "class Model" in obs.problem and "@triton.jit" in obs.problem
    assert obs.turn == 0 and not obs.done
    assert env.state.task_id == "level1/19_ReLU"


def test_seeded_task_choice_is_deterministic():
    many = {f"level1/{i}_t": Task(f"level1/{i}_t", 1, "") for i in range(50)}
    env = KernelEnvironment(MockEvaluator(), many)
    picks = {env.reset(seed=7).task_id for _ in range(3)}
    assert len(picks) == 1 and many[picks.pop()].split == "train"
    assert many[env.reset(seed=1, split="eval").task_id].split == "eval"


def test_episode_rewards_and_done(tasks):
    ev = MockEvaluator([timed(2.0), {"stage": "correctness", "correct": False, "feedback": "mismatch"}])
    env = KernelEnvironment(ev, tasks, max_turns=3)
    env.reset(seed=0, task_id="level1/19_ReLU")

    obs = env.step(KernelAction(code=GOOD))
    assert obs.stage == "timed" and obs.reward == pytest.approx(1.3) and not obs.done

    obs = env.step(KernelAction(code=GOOD))
    assert obs.stage == "correctness" and obs.reward == 0.0

    obs = env.step(KernelAction(code=with_forward("return torch.relu(x)")))
    assert obs.stage == "format" and obs.reward == 0.0 and "torch.relu" in obs.feedback
    assert obs.done and len(ev.calls) == 2  # rejected code never reaches the GPU

    assert env.state.best_speedup == 2.0
    assert [h["stage"] for h in env.state.history] == ["timed", "correctness", "format"]
    with pytest.raises(RuntimeError):
        env.step(KernelAction(code=GOOD))

    seeds = [c["seeds"] for c in ev.calls]
    assert len(seeds[0]) == 5 and seeds[0] != seeds[1]


def test_suspicious_speedup_flagged(tasks):
    env = KernelEnvironment(MockEvaluator([timed(500.0)]), tasks)
    env.reset(task_id="level1/19_ReLU")
    env.step(KernelAction(code=GOOD))
    assert env.state.history[0].get("suspicious")


@pytest.mark.parametrize(
    "correct, speedup, reward",
    [(False, 10.0, 0.0), (True, 1.0, 0.3), (True, 0.5, 0.1), (True, 4.0, 2.3), (True, 1000.0, 3.3)],
)
def test_rubric(correct, speedup, reward):
    obs = KernelObservation(correct=correct, speedup=speedup)
    assert kernel_rubric()(None, obs) == pytest.approx(reward)


def test_client_parses_server_payload():
    obs = KernelObservation(task_id="t", stage="timed", correct=True, speedup=1.5, turn=1, done=True, reward=0.9)
    result = KernelEnv._parse_result(None, serialize_observation(obs))
    assert result.observation == obs and result.reward == 0.9 and result.done


def test_eval_log_lines(tasks, tmp_path, monkeypatch):
    log = tmp_path / "evals.jsonl"
    monkeypatch.setenv("KERNEL_ENV_EVAL_LOG", str(log))
    env = KernelEnvironment(MockEvaluator([timed(2.0)]), tasks)
    env.reset(seed=0, task_id="level1/19_ReLU")
    env.step(KernelAction(code=GOOD))
    env.step(KernelAction(code=with_forward("return torch.relu(x)")))
    recs = [json.loads(line) for line in log.read_text().splitlines()]
    assert [r["event"] for r in recs] == ["reset", "step", "step"]
    assert {r["episode_id"] for r in recs} == {env.state.episode_id}
    assert recs[1]["stage"] == "timed" and recs[1]["gpu_id"] == "mock" and recs[1]["eval_s"] >= 0
    assert recs[2]["stage"] == "format" and recs[2]["queue_wait_s"] == 0.0
    assert env.state.history[0]["code_sha"] == recs[1]["code_sha"]


def test_reference_timing_cached_per_task():
    ev = MockEvaluator([timed(2.0), timed(3.0), timed(1.0)])
    ev.evaluate("ref A", GOOD, "triton", [1])
    ev.evaluate("ref A", GOOD, "triton", [2])
    ev.evaluate("ref B", GOOD, "triton", [3])
    assert ev.calls[0]["ref_cache"] is None
    assert ev.calls[1]["ref_cache"]["t_ref_ms"] == MockEvaluator.DEFAULT["t_ref_ms"]
    assert ev.calls[2]["ref_cache"] is None  # different task


@pytest.mark.parametrize(
    "stage, correct, speedup, reward",
    [("format", False, None, 0.0), ("compile", False, None, 0.0), ("runtime", False, None, 0.02),
     ("timeout", False, None, 0.02), ("correctness", False, None, 0.05), ("timed", True, 0.5, 0.1)],
)
def test_partial_credit(stage, correct, speedup, reward):
    obs = KernelObservation(stage=stage, correct=correct, speedup=speedup)
    assert kernel_rubric("speedup", partial=True)(None, obs) == pytest.approx(reward)
    assert kernel_rubric("speedup", partial=False)(None, obs) == pytest.approx(reward if correct else 0.0)
