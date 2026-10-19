"""Real evaluation through worker.py. Runs only on a machine with CUDA + triton."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)

from kernel_env.server.evaluator import Evaluator  # noqa: E402
from kernels import ALIAS, CUDA_GOOD, GOOD, HANG, WRONG  # noqa: E402

REF = (Path(__file__).parent / "fixtures/level1/19_ReLU.py").read_text()
SEEDS = [1, 2, 3]


@pytest.fixture(scope="module")
def ev():
    return Evaluator(gpus=["0"], timeout_s=120, warmup_ms=5, rep_ms=20)


def test_good_kernel_is_correct_and_timed(ev):
    res = ev.evaluate(REF, GOOD, "triton", SEEDS)
    assert res["stage"] == "timed" and res["correct"], res
    assert res["speedup"] > 0  # Numerical parity/known-fusion controls are separate strict preflight gates.
    assert res["bound"] == "memory" and res["flops"] == 0 and res["bytes"] == 2 * 256 * 16384 * 4
    assert 0.05 < res["roofline_frac"] <= 1.0 and 0.05 < res["ref_roofline_frac"] <= 1.0, res
    assert "Roofline on" in res["feedback"]
    assert len(res['timing_samples_ms']) == 3
    assert res['timing_method'] == 'fresh_paired_cuda_graph_event_interval_v5'
    assert min(res['timing_samples_ms']) <= res['t_new_ms'] <= max(res['timing_samples_ms'])


def test_cuda_load_inline_kernel(ev):
    res = ev.evaluate(REF, CUDA_GOOD, "cuda", SEEDS)
    assert res["stage"] == "timed", res


def test_wrong_output_rejected(ev):
    res = ev.evaluate(REF, WRONG, "triton", SEEDS)
    assert res["stage"] == "correctness" and "mismatch" in res["feedback"]


def test_aliasing_input_rejected(ev):
    res = ev.evaluate(REF, ALIAS, "triton", SEEDS)
    assert res["stage"] == "correctness" and "aliases" in res["feedback"]


def test_hang_times_out():
    res = Evaluator(gpus=["0"], timeout_s=20, warmup_ms=5, rep_ms=20).evaluate(REF, HANG, "triton", SEEDS)
    assert res["stage"] == "timeout"


def test_reference_vs_itself_is_stable(ev):
    # identical kernel timed repeatedly: speedup spread is the reward noise floor
    results = [ev.evaluate(REF, GOOD, "triton", [seed,seed+1,seed+2]) for seed in range(5)]
    speedups = [r['speedup'] for r in results]
    assert max(speedups) / min(speedups) < 1.15, speedups
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'train'))
    from reward_noise import noise_aware_returns
    episodes = [{'task_id':'relu','codes':[GOOD],'turns':[{**r,'reward_mode':'roofline',
                 'reward':0.3+2.7*min(r['roofline_frac'],1)}]} for r in results]
    rewards, stats = noise_aware_returns(episodes,5)
    assert len(set(rewards)) == 1 and stats['learning_return_std'] == 0


def test_matmul_roofline_is_compute_bound(ev):
    import os

    path = Path(os.environ.get("KERNELBENCH_DIR", "/nonexistent"), "level1/1_Square_matrix_multiplication_.py")
    if not path.exists():
        pytest.skip("KernelBench not available")
    from kernels import MATMUL

    res = ev.evaluate(path.read_text(), MATMUL, "triton", SEEDS[:1])
    print({k: res.get(k) for k in ("stage", "speedup", "t_ref_ms", "t_new_ms", "t_min_ms", "bound",
                                   "arithmetic_intensity", "roofline_frac", "ref_roofline_frac", "gpu")})
    assert res["stage"] == "timed" and not res.get("roofline_violation"), res
    assert res["bound"] == "compute" and res["arithmetic_intensity"] > 100
    assert 0 < res["roofline_frac"] <= 1.0


def test_shared_gpu_keeps_timings_clean():
    # 2 workers share the GPU for compile/correctness; benchmarks take an exclusive lock
    from concurrent.futures import ThreadPoolExecutor

    shared = Evaluator(gpus=["0"], timeout_s=120, warmup_ms=5, rep_ms=20, workers_per_gpu=2)
    with ThreadPoolExecutor(4) as pool:
        res = list(pool.map(lambda s: shared.evaluate(REF, GOOD, "triton", [s, s + 1]), range(4)))
    print([r.get("phases") for r in res])
    assert all(r["stage"] == "timed" for r in res), res
    assert all("bench" in r["phases"] and "lock_wait" in r["phases"] for r in res)
    speedups = [r["speedup"] for r in res]
    assert max(speedups) / min(speedups) < 1.15, speedups


def test_very_slow_kernel_is_correct_and_timed(ev):
    from kernels import SLOW

    res = ev.evaluate(REF, SLOW, "triton", SEEDS)
    print(res.get("phases"), res.get("t_new_ms"))
    assert res["stage"] == "timed" and res["correct"] and res["speedup"] < 0.02, res
    # The wall-time cutoff can fall on either side of this fixture across GPUs.
    if res['timing_method'] == 'correctness_wall_slow':
        assert 'very slow' in res['feedback']
    else:
        assert res['timing_method'] == 'fresh_paired_cuda_graph_event_interval_v5'
        assert len(res['timing_samples_ms']) == 3


TOY_REF = """
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return torch.relu(x * self.scale)

def get_inputs():
    return [torch.randn(64, 512)]

def get_init_inputs():
    return [512]
"""


def test_prompt_v2_example_is_correct(ev):
    from kernel_env.server.tasks import EXAMPLE_V2

    res = ev.evaluate(TOY_REF, EXAMPLE_V2, "triton", SEEDS)
    assert res["stage"] == "timed" and res["correct"], res


@pytest.mark.parametrize('cycles',[100000,20000000])
def test_nondefault_stream_work_is_rejected(ev,cycles):
    code = GOOD.replace("super().__init__()", "super().__init__(); self.stream = torch.cuda.Stream()")
    # Append a delayed side-stream launch after a correct output has been produced.
    code = code.replace("return y", f"with torch.cuda.stream(self.stream):\n            torch.cuda._sleep({cycles})\n        return y")
    res = ev.evaluate(REF, code, "triton", SEEDS[:1])
    assert res['stage']=='format' and res['correct'] is False, res
    assert 'side-stream' in res['feedback'], res


def test_capture_time_side_stream_is_rejected(ev):
    # Calls 1-8 cover 3 correctness trials, eval+train profiles and the eager warmup; later calls are graph
    # capture only, so this side stream escapes both eager profiles and the captured replay.
    code = GOOD.replace("super().__init__()", "super().__init__(); self.stream = torch.cuda.Stream(); self.calls = 0")
    code = code.replace("return y", "self.calls += 1\n        if self.calls > 8:\n"
                        "            with torch.cuda.stream(self.stream):\n                torch.cuda._sleep(100000)\n        return y")
    res = ev.evaluate(REF, code, "triton", SEEDS)
    assert res['stage']=='format' and res['correct'] is False, res
    assert 'side-stream' in res['feedback'], res


def test_identity_copy_with_unused_jit_kernel_is_rejected(ev):
    ref = REF.replace("return torch.relu(x)", "return x")
    code = GOOD.replace("return y", "return x.clone()")
    # Remove actual kernel invocation, retaining an unused @jit definition.
    code = "\n".join(line for line in code.splitlines() if "relu_kernel[" not in line)
    res=ev.evaluate(ref,code,"triton",SEEDS[:1])
    assert res['stage']=='format' and 'No executed' in res['feedback'], res
