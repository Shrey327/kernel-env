"""Hybrid contract: vendor GEMM/conv allowed in host code, all other compute still custom, no precision drop."""
import pytest
import torch
import torch.nn.functional as F

from kernel_env.server.static_checks import check_source
from kernel_env.server.worker import HostOperationError, kernel_only_mode

FUSED = '''
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def leaky_kernel(X, Y, N, S, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    x = tl.load(X + i, i < N) * S
    tl.store(Y + i, tl.where(x >= 0, x, 0.01 * x), i < N)

class ModelNew(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.gemm = nn.Linear(i, o)
    def forward(self, x):
        y = F.linear(x, self.gemm.weight, self.gemm.bias).contiguous()
        out = torch.empty_like(y)
        leaky_kernel[(triton.cdiv(y.numel(), 1024),)](y, out, y.numel(), 2.0, B=1024)
        return out
'''


def test_strict_contract_still_rejects_library_calls(monkeypatch):
    monkeypatch.delenv('KERNEL_ENV_CONTRACT', raising=False)
    assert any('torch.nn.functional.linear' in e for e in check_source(FUSED))


def test_hybrid_allows_vendor_gemm_and_conv_only(monkeypatch):
    monkeypatch.setenv('KERNEL_ENV_CONTRACT', 'hybrid')
    assert check_source(FUSED, task_id='level2/76_Gemm_Add_ReLU') == []
    assert check_source(FUSED, task_id='level1/1_Square_matrix_multiplication_'), 'level-1 stays kernel-only'
    for banned in ('F.relu(y)', 'torch.sum(y)', 'y.clamp(0)', 'self.gemm(x)', 'x @ self.gemm.weight'):
        errors = check_source(FUSED.replace('.contiguous()', '.contiguous(); z = ' + banned), task_id='level2/76_Gemm_Add_ReLU')
        assert errors, banned


def _run(mode, fn):
    with mode:
        return fn()


def test_runtime_guard_allows_fp32_library_ops_and_blocks_the_rest():
    x, w, b = torch.randn(4, 8), torch.randn(16, 8), torch.randn(16)
    hybrid = kernel_only_mode('cpu', hybrid=True)
    assert _run(hybrid, lambda: F.linear(x, w, b)).shape == (4, 16)
    assert _run(hybrid, lambda: torch.matmul(x, w.t())).shape == (4, 16)
    assert _run(hybrid, lambda: F.conv2d(torch.randn(1, 3, 8, 8), torch.randn(4, 3, 3, 3))).shape == (1, 4, 6, 6)
    with pytest.raises(HostOperationError):_run(hybrid, lambda: torch.relu(x))
    with pytest.raises(HostOperationError):_run(hybrid, lambda: x.sum())
    with pytest.raises(HostOperationError, match='Reduced-precision'):
        _run(hybrid, lambda: F.linear(x.bfloat16(), w.bfloat16()))
    with pytest.raises(HostOperationError):_run(kernel_only_mode('cpu', hybrid=False), lambda: F.linear(x, w, b))


def test_hybrid_runtime_scope_defaults_to_level2_only(monkeypatch):
    monkeypatch.setenv('KERNEL_ENV_CONTRACT', 'hybrid')
    x, w = torch.randn(4, 8), torch.randn(16, 8)
    assert _run(kernel_only_mode('cpu', task_id='level2/76_Gemm_Add_ReLU'), lambda: F.linear(x, w)).shape == (4, 16)
    with pytest.raises(HostOperationError):
        _run(kernel_only_mode('cpu', task_id='level1/1_Square_matrix_multiplication_'), lambda: F.linear(x, w))


def test_fewshot_block_is_prompt_only_disjoint_and_matches_pinned_reference(monkeypatch):
    import os
    from pathlib import Path
    from kernel_env.server import fewshot
    from kernel_env.server.tasks import load_tasks, build_prompt
    root = os.environ.get('KERNELBENCH_DIR')
    if not root:
        pytest.skip('pinned KernelBench source required')
    tasks = load_tasks()
    for level, (task_id, reference, _) in fewshot.EXAMPLES.items():
        assert tasks[task_id].split == 'train' and tasks[task_id].level == level
        assert Path(root, task_id + '.py').read_text().strip() == reference
    import sys
    sys.path.insert(0, str(Path(__file__).parents[2] / 'train'))
    from better_protocol import CURRICULUM, FUSION_CURRICULUM
    from campaign import RESERVED, CANDIDATES
    used = set(CURRICULUM) | set(FUSION_CURRICULUM) | set(RESERVED) | set(CANDIDATES)
    assert not {fewshot.STRICT_TASK, fewshot.HYBRID_TASK} & used
    monkeypatch.setenv('KERNEL_ENV_CONTRACT', 'hybrid')
    plain = build_prompt(tasks['level2/76_Gemm_Add_ReLU'])
    monkeypatch.setenv('KERNEL_ENV_FEWSHOT', '1')
    shown = build_prompt(tasks['level2/76_Gemm_Add_ReLU'])
    assert fewshot.HYBRID_SOLUTION in shown and fewshot.HYBRID_SOLUTION not in plain
    assert fewshot.STRICT_SOLUTION in build_prompt(tasks['level1/19_ReLU'])
    assert shown.index(fewshot.HYBRID_SOLUTION) < shown.index('Answer with one')
