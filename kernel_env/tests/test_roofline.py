import pytest

from kernel_env import KernelAction
from kernel_env.server.evaluator import MockEvaluator, apply_roofline
from kernel_env.server.kernel_env_environment import KernelEnvironment
from kernel_env.server.roofline import peaks, roofline
from kernel_env.server.tasks import load_tasks
from kernels import GOOD
from test_environment import FIXTURES

H100 = "NVIDIA H100 80GB HBM3"
RELU = dict(flops=0, nbytes=2 * 4096 * 393216 * 4, dtype="torch.float32")  # KernelBench level1/19 at full size


def test_peak_lookup_prefers_specific_names():
    assert peaks("NVIDIA L40S")[3] == 0.864 and peaks("NVIDIA L4")[3] == 0.3
    assert peaks("NVIDIA H100 PCIe")[3] == 2.0 and peaks(H100)[3] == 3.35
    assert peaks("Tesla T4") is None


def test_relu_is_memory_bound():
    r = roofline(RELU["flops"], RELU["nbytes"], RELU["dtype"], H100, t_ref_ms=4.2, t_new_ms=4.0)
    assert r["bound"] == "memory" and r["t_min_ms"] == pytest.approx(3.846, rel=1e-3)
    assert r["roofline_frac"] == pytest.approx(3.846 / 4.0, rel=1e-3)


def test_square_matmul_is_compute_bound_on_tf32():
    n = 4096
    r = roofline(2 * n**3, 3 * n * n * 4, "torch.float32", H100, t_ref_ms=2.5)
    assert r["bound"] == "compute" and r["arithmetic_intensity"] == pytest.approx(682.7, rel=1e-3)
    assert r["t_min_ms"] == pytest.approx(2 * n**3 / 494.7e12 * 1e3, rel=1e-6)


def timed(t_new, **kw):
    return {"stage": "timed", "correct": True, "speedup": 4.2 / t_new, "t_ref_ms": 4.2, "t_new_ms": t_new,
            "gpu": H100, "flops": 0, "bytes": RELU["nbytes"], "dtype": "torch.float32", "feedback": "Correct.", **kw}


def test_apply_roofline_adds_feedback():
    res = apply_roofline(timed(4.0))
    assert res["stage"] == "timed" and "memory-bound" in res["feedback"] and "yours 96%" in res["feedback"]


def test_faster_than_hardware_is_rejected():
    res = apply_roofline(timed(1.0))
    assert res["stage"] == "correctness" and not res["correct"] and res["roofline_violation"]
    assert "below the hardware floor" in res["feedback"]


def test_unknown_gpu_leaves_result_alone():
    res = timed(1.0, gpu="Tesla T4")
    assert apply_roofline(res) == res


@pytest.mark.parametrize("mode, reward", [("speedup", 0.3 + 0.07), ("roofline", 0.3 + 2.7 * 3.846 / 4.0)])
def test_reward_modes(monkeypatch, mode, reward):
    monkeypatch.setenv("KERNEL_ENV_REWARD", mode)
    env = KernelEnvironment(MockEvaluator([timed(4.0)]), load_tasks(str(FIXTURES), levels=(1,)))
    env.reset(task_id="level1/19_ReLU")
    obs = env.step(KernelAction(code=GOOD))
    assert obs.bound == "memory" and obs.roofline_frac == pytest.approx(0.9615, rel=1e-3)
    assert obs.reward == pytest.approx(reward, abs=0.01)
