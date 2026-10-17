"""Roofline floor for a task on a GPU: t_min = max(FLOPs / peak compute, bytes / peak bandwidth).

FLOPs come from torch.utils.flop_counter on the reference (matmul/conv/attention only, so
elementwise work counts as 0) and bytes are the compulsory traffic: inputs + parameters read once,
outputs written once. Both are lower bounds, so t_min is a floor no correct kernel can beat,
except tasks that legitimately skip input data (slicing), which KernelBench rarely has.
"""

from typing import Dict, Optional

# Dense datasheet peaks: (FP32 SIMT, TF32 tensor, BF16/FP16 tensor) in TFLOP/s, HBM in TB/s.
PEAKS = {
    "H100 PCIe": (51.2, 378.0, 756.0, 2.0),
    "H100": (66.9, 494.7, 989.4, 3.35),  # SXM, "NVIDIA H100 80GB HBM3"
    "H200": (66.9, 494.7, 989.4, 4.8),
    "A100-SXM4-40GB": (19.5, 156.0, 312.0, 1.555),
    "A100": (19.5, 156.0, 312.0, 2.039),
    "L40S": (91.6, 183.0, 362.0, 0.864),
    "L4": (30.3, 60.0, 121.0, 0.3),
}


def peaks(gpu: Optional[str]):
    for key in sorted(PEAKS, key=len, reverse=True):  # "L40S" before "L4", "H100 PCIe" before "H100"
        if gpu and key in gpu:
            return PEAKS[key]
    return None


def roofline(flops, nbytes, dtype, gpu, t_ref_ms, t_new_ms=None) -> Optional[Dict]:
    p = peaks(gpu)
    if p is None or not nbytes or flops is None or not t_ref_ms:
        return None
    _, tf32, half, bw = p
    # fp32 tasks: TF32 tensor cores are the roof, since tl.dot uses TF32 by default and passes atol=rtol=1e-2
    compute = {"torch.float32": tf32, "torch.float16": half, "torch.bfloat16": half}.get(dtype)
    if compute is None:
        return None
    t_compute = flops / (compute * 1e12) * 1e3
    t_memory = nbytes / (bw * 1e12) * 1e3
    t_min = max(t_compute, t_memory)
    r = {
        "arithmetic_intensity": flops / nbytes,
        "bound": "compute" if t_compute > t_memory else "memory",
        "t_min_ms": t_min,
        "ref_roofline_frac": t_min / t_ref_ms,
    }
    if t_new_ms:
        r["roofline_frac"] = t_min / t_new_ms
    return r


def describe(r: Dict, gpu: str) -> str:
    s = (f"Roofline on {gpu}: {r['bound']}-bound (arithmetic intensity {r['arithmetic_intensity']:.3g} FLOP/byte), "
         f"floor {r['t_min_ms']:.4f} ms; PyTorch reaches {r['ref_roofline_frac']:.0%} of it")
    if "roofline_frac" in r:
        s += f", yours {r['roofline_frac']:.0%}"
    return s + "."
