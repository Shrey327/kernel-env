"""Candidate sources shared by tests (ReLU task)."""

HEADER = """
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
"""

KERNEL = """
@triton.jit
def relu_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(y_ptr + offs, tl.maximum(x, 0.0), mask=mask)
"""

GOOD = HEADER + KERNEL + """
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.contiguous()
        y = torch.empty_like(x)
        n = x.numel()
        relu_kernel[(triton.cdiv(n, 1024),)](x, y, n, BLOCK=1024)
        return y
"""


def with_forward(body: str, extra: str = "", header: str = HEADER) -> str:
    """ReLU kernel module whose ModelNew.forward body is replaced (4-space indented lines)."""
    lines = "\n".join("        " + ln for ln in body.strip().splitlines())
    return header + extra + KERNEL + f"""
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = nn.ReLU()

    def forward(self, x):
{lines}
"""


WRONG = GOOD.replace("tl.maximum(x, 0.0)", "x")  # identity, not relu
ALIAS = with_forward("relu_kernel[(triton.cdiv(x.numel(), 1024),)](x, x, x.numel(), BLOCK=1024)\nreturn x")
HANG = with_forward("while True:\n    pass")

CUDA_GOOD = """
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

src = '''
__global__ void relu_k(const float* x, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = x[i] > 0.f ? x[i] : 0.f;
}
torch::Tensor relu(torch::Tensor x) {
  auto y = torch::empty_like(x);
  int n = x.numel();
  relu_k<<<(n + 255) / 256, 256>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
  return y;
}
'''
mod = load_inline(name="relu_ext", cpp_sources="torch::Tensor relu(torch::Tensor x);",
                  cuda_sources=src, functions=["relu"])

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return mod.relu(x.contiguous())
"""

MATMUL = """
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def mm_kernel(a, b, c, M, N, K, sam, sak, sbk, sbn, scm, scn,
              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        x = tl.load(a + rm[:, None] * sam + (k + rk)[None, :] * sak,
                    mask=(rm[:, None] < M) & ((k + rk)[None, :] < K), other=0.0)
        y = tl.load(b + (k + rk)[:, None] * sbk + rn[None, :] * sbn,
                    mask=((k + rk)[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(x, y)
    tl.store(c + rm[:, None] * scm + rn[None, :] * scn, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        M, K = A.shape
        N = B.shape[1]
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)
        grid = (triton.cdiv(M, 128), triton.cdiv(N, 128))
        mm_kernel[grid](A, B, C, M, N, K, A.stride(0), A.stride(1), B.stride(0), B.stride(1),
                        C.stride(0), C.stride(1), BM=128, BN=128, BK=32)
        return C
"""

# correct but ~seconds per call: 100k sequential launches
SLOW = with_forward("""y = torch.empty_like(x)
for _ in range(100000):
    relu_kernel[(triton.cdiv(x.numel(), 1024),)](x, y, x.numel(), BLOCK=1024)
return y""")
