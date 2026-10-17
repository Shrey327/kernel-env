import pytest

from kernel_env.server.static_checks import check_source
from kernels import CUDA_GOOD, GOOD, HEADER, KERNEL, with_forward


def test_good_triton_passes():
    assert check_source(GOOD) == []


def test_copied_get_inputs_allowed():
    assert check_source(GOOD + "\ndef get_inputs():\n    return [torch.rand(4, 4)]\n") == []


def test_layer_named_like_tensor_op_allowed():
    # params held in an nn layer whose attribute name matches a tensor op (seen in Qwen3.5 baseline)
    src = GOOD.replace("super().__init__()\n", "super().__init__()\n        self.conv3d = nn.Conv3d(3, 8, 3)\n", 1)
    src = src.replace("x = x.contiguous()", "w = self.conv3d.weight\n        x = x.contiguous()")
    assert "self.conv3d" in src and check_source(src) == []
    called = src.replace("return y", "return self.conv3d(y)")
    assert any("nn layer" in e for e in check_source(called))


def test_good_cuda_passes():
    assert check_source(CUDA_GOOD, "cuda") == []


@pytest.mark.parametrize(
    "src, expect",
    [
        (with_forward("return torch.relu(x)"), "torch.relu"),
        (with_forward("return F.relu(x)"), "torch.nn.functional.relu"),
        (with_forward("return self.act(x)"), "nn layer"),
        (with_forward("return x.clamp(min=0)"), ".clamp"),
        (with_forward("return x @ x.t()"), "'@'"),
        (with_forward("mm = torch.matmul\nreturn mm(x, x)"), "reference to PyTorch op"),
        (with_forward("f = x.clamp\nreturn f(min=0)"), "reference to tensor op"),
        (with_forward("return relu(x)", header=HEADER + "from torch import relu\n"), "torch.relu"),
        (with_forward("return getattr(torch, 'relu')(x)"), "getattr"),
        (with_forward("return x", extra="import os\n"), "import of 'os'"),
        (with_forward("return x", extra="from subprocess import run\n"), "import from 'subprocess'"),
        (with_forward("return x", extra="torch.allclose = lambda *a, **k: True\n"), "assigning attributes"),
        (with_forward("return x.__class__"), "dunder"),
        (with_forward("return torch.compile(lambda t: t)(x)"), "torch.compile"),
        (HEADER + "class ModelNew(nn.Module):\n    def forward(self, x):\n        return x\n", "no @triton.jit"),
        (HEADER + KERNEL, "no top-level class ModelNew"),
        ("def broken(:\n", "SyntaxError"),
    ],
)
def test_rejects(src, expect):
    errors = check_source(src)
    assert any(expect in e for e in errors), errors


def test_cuda_requires_load_inline():
    assert any("load_inline" in e for e in check_source(GOOD, "cuda"))


def test_prompt_v2_example_is_valid(monkeypatch):
    from kernel_env.server.tasks import EXAMPLE_V2, Task, build_prompt

    assert check_source(EXAMPLE_V2) == []  # the worked example must itself pass the code checks
    task = Task("level1/19_ReLU", 1, "class Model: ...")
    assert EXAMPLE_V2 in build_prompt(task)
    monkeypatch.setenv("KERNEL_ENV_PROMPT", "1")
    assert EXAMPLE_V2 not in build_prompt(task) and "Reply with the complete module source only" in build_prompt(task)
