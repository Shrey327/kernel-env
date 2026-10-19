"""CPU validation with trusted controls; no model-generated source or GPU kernel is executed."""

import math

import pytest

torch = pytest.importorskip("torch")

from kernel_env.server.worker import FROBENIUS, SOFTMAX, adversarial_inputs, numerical_error


def error(ref, actual, task):
    return numerical_error(torch, ref, actual, task, atol=1e-2, rtol=1e-2)


def softmax_input():
    return torch.rand(4, 8192, generator=torch.Generator().manual_seed(7))


@pytest.mark.parametrize("failure", ["zeros", "chunks", "wrong_stride", "incomplete", "wrong_rows"])
def test_softmax_old_tolerance_accepts_wrong_results(failure):
    x = softmax_input()
    ref = torch.softmax(x, dim=1)
    if failure == "zeros":
        actual = torch.zeros_like(x)
    elif failure == "chunks":
        actual = torch.softmax(x.reshape(4, -1, 1024), dim=-1).reshape_as(x)
    elif failure == "wrong_stride":
        # The audited launch used stride(1) for rows, leaving most output unwritten.
        actual = torch.zeros_like(x)
        for row in range(x.shape[0]):
            actual.view(-1)[row:row + 1024] = torch.softmax(x.view(-1)[row:row + 1024], dim=0)
    elif failure == "incomplete":
        actual = ref.clone()
        actual[:, 1024:] = 0
    else:
        actual = ref.roll(1, dims=0)  # sums still equal 1; invariant alone is insufficient
    assert torch.allclose(actual, ref, atol=1e-2, rtol=1e-2)  # reproduce the old false positive
    assert error(ref, actual, SOFTMAX) is not None


def test_full_size_softmax_probabilities_are_below_old_absolute_tolerance():
    # rand() lies in [0,1], so every reference probability is bounded by e / width.
    assert math.e / 393216 < 1e-2


@pytest.mark.parametrize("trial", range(5))
def test_softmax_valid_control_on_adversarial_inputs(trial):
    x = adversarial_inputs(torch, [softmax_input()], SOFTMAX, trial)[0]
    ref = torch.softmax(x, dim=1)
    exp = torch.exp(x - x.amax(dim=1, keepdim=True))
    actual = exp / exp.sum(dim=1, keepdim=True)
    assert error(ref, actual, SOFTMAX) is None
    if trial == 2:
        assert ref.max() > 0.99
        assert len(set(ref.argmax(dim=1).tolist())) == x.shape[0]


@pytest.mark.parametrize("failure", ["zeros", "per_slice", "incomplete", "wrong_norm"])
def test_frobenius_old_tolerance_accepts_wrong_results(failure):
    x = torch.rand(4, 3, 256, 256, generator=torch.Generator().manual_seed(11))
    ref = x / torch.linalg.vector_norm(x)
    if failure == "zeros":
        actual = torch.zeros_like(x)
    elif failure == "per_slice":
        actual = x / torch.linalg.vector_norm(x, dim=(-2, -1), keepdim=True)
    elif failure == "incomplete":
        actual = ref.clone()
        actual[1:] = 0
    else:
        actual = ref * 0.5
    assert torch.allclose(actual, ref, atol=1e-2, rtol=1e-2)
    assert error(ref, actual, FROBENIUS) is not None


@pytest.mark.parametrize("trial", range(5))
def test_frobenius_valid_control_on_adversarial_inputs(trial):
    x = torch.rand(4, 3, 16, 32, generator=torch.Generator().manual_seed(11))
    x = adversarial_inputs(torch, [x], FROBENIUS, trial)[0]
    ref = x / torch.linalg.vector_norm(x)
    actual = x / x.square().sum().sqrt()
    assert error(ref, actual, FROBENIUS) is None
    if trial == 3:
        assert (ref != 0).sum() == 2
        assert ref.flatten()[0] == pytest.approx(0.6)
        assert ref.flatten()[-1] == pytest.approx(-0.8)


@pytest.mark.parametrize("task", [SOFTMAX, FROBENIUS])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_or_unwritten_normalization_output_fails(task, value):
    ref = torch.full((2, 8192), 1 / 8192) if task == SOFTMAX else torch.full((2, 8192), 1 / 128)
    actual = ref.clone()
    actual[0, 0] = value
    assert "finite" in error(ref, actual, task)


@pytest.mark.parametrize("task", [SOFTMAX, FROBENIUS])
def test_normalization_retains_relative_error_allowance(task):
    ref = torch.full((2, 8192), 1 / 8192) if task == SOFTMAX else torch.full((2, 8192), 1 / 128)
    assert error(ref, ref * 1.005, task) is None
    assert error(ref, -ref, task) is not None


def test_unchanged_task_keeps_configured_tolerance_and_inputs():
    x = torch.rand(3, 4)
    for trial in range(5):
        assert adversarial_inputs(torch, [x], "level1/1_Square_matrix_multiplication_", trial)[0] is x
    ref = torch.zeros(2, 3)
    actual = torch.full_like(ref, 0.009)
    assert error(ref, actual, "level1/1_Square_matrix_multiplication_") is None
    assert numerical_error(torch, ref, actual, "other", atol=0.001, rtol=0.01)


def test_worker_applies_task_specific_validation(tmp_path, monkeypatch):
    from kernel_env.server.worker import _evaluate_candidate

    # Trusted control module: zeros pass the old Softmax comparison, but must never be timed now.
    path = tmp_path / "candidate.py"
    path.write_text("""import torch
class ModelNew(torch.nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)
""")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    def bench(*args):
        pytest.fail("invalid output reached timing")

    result = _evaluate_candidate(
        {"code_path": str(path), "seeds": [1, 2, 3, 4, 5], "atol": 0.01, "rtol": 0.01, "task_id": SOFTMAX},
        torch, torch.device("cpu"), lambda x: torch.softmax(x, dim=1), [],
        lambda seed: [softmax_input()], bench, {"t_ref_ms": 1.0, "gpu": "cpu-test"},
    )
    assert result["stage"] == "correctness" and result["correct"] is False
    assert "row must sum to 1" in result["feedback"]


@pytest.mark.parametrize('task,activation',[
    ('level1/19_ReLU',torch.relu),
    ('level1/20_LeakyReLU',torch.nn.functional.leaky_relu),
    ('level1/31_ELU',torch.nn.functional.elu),
    ('level1/32_HardTanh',torch.nn.functional.hardtanh),
])
def test_activation_copy_passes_old_positive_inputs_but_fails_signed_inputs(task,activation):
    x=torch.rand(4,8192)
    assert torch.allclose(x,activation(x),atol=.01,rtol=.01)
    x=adversarial_inputs(torch,[x],task,1)[0]
    assert x.min() < -16 and x.max()>16
    assert numerical_error(torch,activation(x),x.clone(),task,.01,.01)
    assert numerical_error(torch,activation(x),activation(x),task,.01,.01) is None


@pytest.mark.parametrize('task,fn', [
    ('level1/94_MSELoss', lambda x, y: (x-y).square().mean()),
    ('level1/96_HuberLoss', torch.nn.functional.smooth_l1_loss),
])
def test_regression_losses_reject_constant_answers_across_trials(task, fn):
    original = [torch.rand(32, 128), torch.rand(32, 128)]
    constant = fn(*original)
    for trial in (1, 2):
        inputs = adversarial_inputs(torch, [x.clone() for x in original], task, trial)
        ref = fn(*inputs)
        assert error(ref, ref.clone(), task) is None
        assert error(ref, constant, task)
        assert [x.shape for x in inputs] == [x.shape for x in original]


@pytest.mark.parametrize('task,fn', [
    ('level1/47_Sum_reduction_over_a_dimension', lambda x: x.sum(dim=1, keepdim=True)),
    ('level1/48_Mean_reduction_over_a_dimension', lambda x: x.mean(dim=1)),
])
def test_reductions_reject_constant_answers_on_zero_inputs(task, fn):
    x = torch.rand(3, 1024, 5)
    constant = fn(x)
    inputs = adversarial_inputs(torch, [x.clone()], task, 1)
    ref = fn(*inputs)
    assert error(ref, ref.clone(), task) is None
    assert error(ref, constant, task)


def test_all_activation_domains_are_disclosed_and_signed():
    from kernel_env.server.worker import ACTIVATION_TASKS
    from kernel_env.server.tasks import Task, build_prompt
    for task in ACTIVATION_TASKS:
        x = adversarial_inputs(torch, [torch.linspace(0, 1, 129)], task, 1)[0]
        assert x.min() == -32 and x.max() == 32
        assert 'signed inputs' in build_prompt(Task(task, 1, ''))


@pytest.mark.parametrize('layout',['contiguous','transposed','expanded','scalar','empty'])
def test_chunked_numerics_preserve_tolerances_and_check_wrong_tail(monkeypatch,layout):
    import kernel_env.server.worker as worker
    original=worker.paired_tensor_chunks
    monkeypatch.setattr(worker,'paired_tensor_chunks',lambda a,b:original(a,b,max_bytes=16))
    ref=torch.arange(60,dtype=torch.float32).reshape(5,12)
    if layout=='transposed':ref=ref.t()
    elif layout=='expanded':ref=ref[:1].expand(5,12)
    elif layout=='scalar':ref=torch.tensor(2.)
    elif layout=='empty':ref=torch.empty(0)
    actual=ref.contiguous().clone()+.005
    assert numerical_error(torch,ref,actual,'other',.01,.01) is None
    if actual.numel():
        actual.reshape(-1)[-1]+=10
        assert '1/' in numerical_error(torch,ref,actual,'other',.01,.01)


def test_chunked_numerics_keep_nan_policy_and_normalization_invariants(monkeypatch):
    import kernel_env.server.worker as worker
    pairs,chunks=worker.paired_tensor_chunks,worker.tensor_chunks
    monkeypatch.setattr(worker,'paired_tensor_chunks',lambda a,b:pairs(a,b,max_bytes=16))
    monkeypatch.setattr(worker,'tensor_chunks',lambda a:chunks(a,max_bytes=16))
    monkeypatch.setattr(worker,'CHUNK_BYTES',16)
    nonfinite=torch.tensor([float('nan'),float('inf'),-float('inf')])
    assert numerical_error(torch,nonfinite,nonfinite.clone(),'other',.01,.01) is None
    for task in (SOFTMAX,FROBENIUS):
        ref=torch.full((3,17),1/17) if task==SOFTMAX else torch.full((3,17),1/math.sqrt(51))
        assert error(ref,ref.clone(),task) is None
        wrong=ref.clone();wrong[-1,-1]=0
        assert error(ref,wrong,task)
        wrong=ref.clone();wrong[-1,-1]=float('nan')
        assert 'finite' in error(ref,wrong,task)


def test_softmax_chunked_row_blocks_handle_transposed_output(monkeypatch):
    import kernel_env.server.worker as worker
    monkeypatch.setattr(worker,'CHUNK_BYTES',32)
    ref=torch.full((17,3),1/17).t()
    assert not ref.is_contiguous()
    assert error(ref,ref.clone(),SOFTMAX) is None
    wrong=ref.clone();wrong[-1,-1]=0
    assert error(ref,wrong,SOFTMAX)
