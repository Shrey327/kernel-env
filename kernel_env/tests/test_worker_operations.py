"""Exercise the same dispatch guard on CPU tensors without compiling candidate kernels."""

import pytest

torch = pytest.importorskip("torch")

from kernel_env.server.worker import HostOperationError, _evaluate_candidate, kernel_only_mode


@pytest.mark.parametrize("operation", [
    lambda x: x + 1,
    lambda x: x - 1,
    lambda x: x * 2,
    lambda x: x / 2,
    lambda x: 1.0 / x,
    lambda x: x @ x.t(),
    lambda x: x.sum(),
    lambda x: torch.relu(x),
    lambda x: x.add_(1),
    lambda x: torch.add(x, 1, out=torch.empty_like(x)),
    lambda x: x[0, 0].item(),
])
def test_host_tensor_compute_is_rejected(operation):
    x = torch.ones(2, 2)
    with pytest.raises(HostOperationError, match="use a custom kernel"):
        with kernel_only_mode("cpu"):
            operation(x)


def test_allocation_layout_copies_and_scalar_grid_math_are_allowed():
    x = torch.ones(2, 3)
    with kernel_only_mode("cpu"):
        y = torch.empty_like(x)
        y.copy_(x)
        view = y.t().contiguous().view(6)
        converted = view.to(torch.float64)
        zeros = torch.zeros_like(x)
        grid = (x.numel() + 1023) // 1024
    assert grid == 1 and torch.equal(converted, torch.ones(6, dtype=torch.float64))
    assert torch.equal(zeros, torch.zeros(2, 3))


def test_cpu_autotuner_statistics_are_not_blocked_by_cuda_guard():
    with kernel_only_mode("cuda"):
        assert torch.quantile(torch.tensor([1.0, 2.0, 3.0]), 0.5) == 2


@pytest.mark.parametrize("copy", [lambda x: x.to("cpu"), lambda x: torch.empty(2).copy_(x)])
def test_off_device_copy_cannot_bypass_the_guard(copy):
    # Meta exercises device routing without allocating a CUDA tensor or running any kernel.
    x = torch.empty(2, device="meta")
    with pytest.raises(HostOperationError):
        with kernel_only_mode("meta"):
            copy(x)


def test_audited_host_arithmetic_fails_even_though_ast_accepts_it():
    from kernel_env.server.static_checks import check_source
    from kernels import with_forward

    assert check_source(with_forward("return x * (1.0 / x)")) == []
    x = torch.ones(2, 2)
    with pytest.raises(HostOperationError):
        with kernel_only_mode("cpu"):
            x * (1.0 / x)


@pytest.mark.parametrize("compute_on_call", [1, 3])
def test_worker_rejects_host_compute_in_correctness_or_benchmark(tmp_path, monkeypatch, compute_on_call):
    # This is a trusted, hand-written test module, not model-generated source.
    path = tmp_path / "candidate.py"
    path.write_text(f"""import torch
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
    def forward(self, x):
        self.calls += 1
        if self.calls >= {compute_on_call}:
            return x * 0
        return torch.zeros_like(x)
""")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    inputs = lambda seed: [torch.ones(2, 2)]

    def bench(model, args, samples=None):
        model(*args)
        return 1.0

    result = _evaluate_candidate(
        {"code_path": str(path), "seeds": [1, 2], "atol": 0.01, "rtol": 0.01},
        torch, torch.device("cpu"), lambda x: torch.zeros_like(x), [], inputs, bench,
        {"t_ref_ms": 1.0, "gpu": "cpu-test"},
    )
    assert result["stage"] == "format" and result["correct"] is False
    assert "use a custom kernel" in result["feedback"]


@pytest.mark.parametrize('warmed_seconds', [0.01, 1.2])
def test_slow_timing_cutoff_excludes_first_call(tmp_path, monkeypatch, warmed_seconds):
    import contextlib
    from types import SimpleNamespace
    from kernel_env.server import worker

    path = tmp_path / 'candidate.py'
    path.write_text('''import torch
class ModelNew(torch.nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)
''')
    times = iter([0, 10, 20, 20+warmed_seconds, 30, 30+warmed_seconds])
    monkeypatch.setattr(worker, 'time', SimpleNamespace(monotonic=lambda: next(times)))
    monkeypatch.setattr(worker, 'Phase', lambda name: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    calls = []

    def bench(model, inputs, samples):
        calls.append(True)
        samples.extend([1.0, 1.0, 1.0])
        return 1.0

    result = _evaluate_candidate(
        {'code_path': str(path), 'seeds': [1, 2, 3], 'atol': 0.01, 'rtol': 0.01},
        torch, torch.device('cpu'), lambda x: torch.zeros_like(x), [],
        lambda seed: [torch.ones(2, 2)], bench, {'t_ref_ms': 1.0, 'gpu': 'cpu-test'})
    assert result['correct']
    assert bool(calls) == (warmed_seconds < 1)
    if calls:
        assert result['timing_method'] == 'three_warmed_benchmark_medians'
        assert result['t_new_ms'] == 1
    else:
        assert result['timing_method'] == 'correctness_wall_slow'
        assert result['t_new_ms'] == pytest.approx(warmed_seconds*1000)
