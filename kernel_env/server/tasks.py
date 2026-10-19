"""KernelBench task loading, train/eval split, and prompt building."""

import hashlib
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple

EVAL_FRACTION_PCT = 10  # held-out share of each level


@dataclass(frozen=True)
class Task:
    task_id: str  # "level1/19_ReLU"
    level: int
    src: str  # reference source: Model, get_inputs, get_init_inputs

    @property
    def split(self) -> str:
        h = int(hashlib.sha1(self.task_id.encode()).hexdigest(), 16)
        return "eval" if h % 100 < EVAL_FRACTION_PCT else "train"


@lru_cache(maxsize=None)
def load_tasks(root: str = "", levels: Tuple[int, ...] = (1, 2, 3)) -> Dict[str, Task]:
    """Load `<root>/level{N}/*.py` (KernelBench layout). Root defaults to $KERNELBENCH_DIR."""
    base = Path(root or os.environ.get("KERNELBENCH_DIR", ""))
    if not base.is_dir():
        raise FileNotFoundError(f"KernelBench dir not found: {base!s} (set KERNELBENCH_DIR)")
    tasks = {}
    for level in levels:
        for f in sorted((base / f"level{level}").glob("*.py")):
            tid = f"level{level}/{f.stem}"
            tasks[tid] = Task(tid, level, f.read_text())
    if not tasks:
        raise FileNotFoundError(f"No KernelBench tasks under {base}")
    return tasks


RULES = {
    "triton": (
        "Write custom Triton kernels (`@triton.jit`) and call them from `ModelNew.forward`."
    ),
    "cuda": (
        "Write custom CUDA kernels compiled with `torch.utils.cpp_extension.load_inline` "
        "and call them from `ModelNew.forward`."
    ),
}


def build_prompt(task: Task, language: str = "triton", max_turns: int = 4) -> str:
    """Prompt version from KERNEL_ENV_PROMPT ("2" default; "1" = the prompt used for the v1-v3 baselines)."""
    if os.environ.get("KERNEL_ENV_PROMPT", "2") == "1" or language != "triton":
        prompt = _prompt_v1(task, language, max_turns)
    else:
        prompt = _prompt_v2(task, max_turns)
    if task.task_id in {"level1/23_Softmax", "level1/37_FrobeniusNorm_"}:
        prompt += ("\nValidation for this normalization task also uses constant, peaked or signed inputs, "
                   "atol=min(configured_atol, max(abs(reference))*1e-5), the configured rtol, "
                   "and normalization invariants. "
                   "Softmax rows must sum to 1; Frobenius normalization uses one global norm, not per-slice norms.")
    from .worker import ACTIVATION_TASKS, REDUCTION_TASKS, REGRESSION_LOSSES
    if task.task_id in ACTIVATION_TASKS:
        prompt += ('\nValidation also uses signed inputs in [-32,32] and zeros at the original shape. '
                   'Implement the full activation, including negative inputs; timing uses the original generator.')
    if task.task_id in REDUCTION_TASKS:
        prompt += '\nCorrectness also checks zero and shifted/scaled inputs at the original shape; timing uses the original generator.'
    if task.task_id in REGRESSION_LOSSES:
        prompt += '\nCorrectness also checks identical prediction/target tensors and zero predictions with target 2 at the original shape; timing uses the original generator.'
    prompt += '\nCandidate CUDA work must use a single stream; side-stream work is rejected.'
    if os.environ.get("KERNEL_ENV_CONTRACT", "strict") == "hybrid" and task.level == 2:
        prompt = prompt.replace(STRICT_RULES, HYBRID_RULES)
    from . import fewshot
    if fewshot.enabled() and ANSWER_MARKER in prompt:
        prompt = prompt.replace(ANSWER_MARKER, fewshot.block(task.level) + ANSWER_MARKER, 1)
    return prompt


ANSWER_MARKER = "\n\nAnswer with one ```python block"


STRICT_RULES = """2. All math runs inside `@triton.jit` kernels. In `forward` and helpers you may only allocate outputs
   (torch.empty, torch.empty_like, torch.zeros), call .contiguous()/.view()/.reshape(), read .shape/.stride(),
   and launch kernels.
3. Never call PyTorch compute outside kernels: no torch.matmul, torch.relu, torch.softmax, torch.sum, etc.,
   no torch.nn.functional (F.*), no `@`, no tensor methods like x.sum() or x.clamp(), and never call a layer
   such as self.conv(x) or self.linear(x). To use a layer's weights, pass self.conv.weight and
   self.conv.bias to your kernel."""
HYBRID_RULES = """2. Matrix multiplies and convolutions may use the vendor library: torch.matmul, torch.mm, torch.bmm,
   F.linear, F.conv1d/2d/3d and F.conv_transpose1d/2d/3d, in the reference dtype (no casting to half/bf16).
   Pass layer weights explicitly, e.g. F.conv2d(x, self.conv.weight, self.conv.bias, ...).
   All other math (activations, normalization, reductions, elementwise chains) runs in `@triton.jit`
   kernels; fusing it into as few kernels as possible is where the speedup comes from.
3. Apart from those library calls: no other torch compute, no other F.* calls, no `@`, no tensor methods
   like x.sum() or x.clamp(), and never call a layer such as self.conv(x) or self.linear(x). A submission
   with no executed custom kernel is rejected; one only as fast as the reference earns no speed bonus."""


def _prompt_v1(task: Task, language: str, max_turns: int) -> str:
    return f"""You are optimizing a PyTorch model with custom GPU kernels.

Reference implementation (task {task.task_id}):
```python
{task.src.strip()}
```

Write a Python module defining `class ModelNew(nn.Module)` with the same `__init__` signature
and the same outputs as `Model`, but faster. {RULES[language]}

Rules:
- Allowed imports: torch, triton (and triton.language), math, typing, functools.
- Outside kernels, only allocate tensors (torch.empty/zeros/... ) and reshape/view them;
  no torch compute ops, torch.nn.functional, nn layer calls, or torch.compile.
- Parameters may be held in nn modules / nn.Parameter created in `__init__`.
- Outputs are checked on random inputs (atol=rtol=1e-2), then timed against the reference.
- You get {max_turns} submissions; each returns compile errors, mismatch info, or timings.

Keep reasoning short and code comments brief: replies are capped in length, and a reply cut off
before the module is complete is rejected. Reply with the complete module source only."""


# v2: a complete valid answer for a toy reference `torch.relu(x * self.scale)`. Baselines v1-v3 showed about a
# third of turns rejected for computing with PyTorch outside kernels, i.e. the rules were not understood.
EXAMPLE_V2 = """import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def scale_relu_kernel(x_ptr, w_ptr, y_ptr, n, d, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    w = tl.load(w_ptr + offs % d, mask=mask)
    tl.store(y_ptr + offs, tl.maximum(x * w, 0.0), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))  # same parameters as Model, created the same way

    def forward(self, x):
        x = x.contiguous()
        y = torch.empty_like(x)  # allocating is allowed; all math happens in the kernel
        n = x.numel()
        scale_relu_kernel[(triton.cdiv(n, 1024),)](x, self.scale, y, n, x.shape[-1], BLOCK=1024)
        return y"""


def _prompt_v2(task: Task, max_turns: int) -> str:
    return f"""Rewrite this PyTorch model with custom Triton GPU kernels so it gives the same outputs, faster.

Reference implementation (task {task.task_id}):
```python
{task.src.strip()}
```

Rules (a submission that breaks one is rejected before it runs):
1. Define a top-level `class ModelNew(nn.Module)` with the same `__init__` arguments as `Model`. Create the same
   nn modules and parameters in the same order, so weights initialize identically.
2. All math runs inside `@triton.jit` kernels. In `forward` and helpers you may only allocate outputs
   (torch.empty, torch.empty_like, torch.zeros), call .contiguous()/.view()/.reshape(), read .shape/.stride(),
   and launch kernels.
3. Never call PyTorch compute outside kernels: no torch.matmul, torch.relu, torch.softmax, torch.sum, etc.,
   no torch.nn.functional (F.*), no `@`, no tensor methods like x.sum() or x.clamp(), and never call a layer
   such as self.conv(x) or self.linear(x). To use a layer's weights, pass self.conv.weight and
   self.conv.bias to your kernel.
4. Imports allowed: torch, torch.nn, triton, triton.language, math, typing, functools. Do not include
   Model, get_inputs or get_init_inputs.
5. Outputs are checked on random inputs (atol=rtol=1e-2), then timed against the reference.
   You get {max_turns} submissions; each returns errors, mismatches or timings.

Example: for a reference computing `torch.relu(x * self.scale)`, a valid answer is:
```python
{EXAMPLE_V2}
```

Answer with one ```python block containing the complete module and nothing else. Keep it short (under about
150 lines, brief comments): replies are length-capped, and a reply cut off before the module ends is rejected."""
