---
title: Kernel Env
emoji: ⚡
sdk: docker
app_port: 8000
tags:
  - openenv
---

# kernel_env

OpenEnv environment that trains agents to write fast, correct GPU kernels (Triton by default,
CUDA via `load_inline`) for [KernelBench](https://github.com/ScalingIntelligence/KernelBench) tasks.

## Episode

1. `reset(seed=?, task_id=?, split="train"|"eval", language="triton")` → `observation.problem`
   holds the PyTorch reference `Model` and the rules.
2. `step(KernelAction(code=...))` (up to `max_turns`, default 4) → static checks → compile →
   correctness on 5 hidden random-input seeds (atol=rtol=1e-2) → timing vs eager PyTorch.
   `observation.stage` / `feedback` say how far it got and why.

Reward per step: `0` unless correct, else `0.3 + clamp(log2(speedup), -0.2, 3.0)`, or with
`KERNEL_ENV_REWARD=roofline`, `0.3 + 2.7 · min(t_min / t_new, 1)`.

Roofline (`server/roofline.py`): FLOPs counted on the reference (`torch.utils.flop_counter`, matmul/conv/attention),
compulsory bytes (inputs + parameters + outputs), dense datasheet peaks per GPU (H100, H200, A100, L40S, L4).
`t_min = max(FLOPs / peak, bytes / bandwidth)`; fp32 tasks use the TF32 tensor-core roof, since `tl.dot`
uses TF32 and passes the tolerance. Observations carry `bound`, `arithmetic_intensity`, `t_min_ms`,
`roofline_frac`, `ref_roofline_frac`; timings under 0.8 × `t_min` are rejected as impossible.

## Run

GPU host (NVIDIA + nvidia-container-toolkit):

```bash
docker build -t kernel_env:latest -f server/Dockerfile .
docker run --gpus all -p 8000:8000 -e KERNEL_ENV_GPUS=0 kernel_env:latest
```

Modal (see `../modal_app.py`): `modal run modal_app.py::gpu_tests` runs the GPU tests;
`modal deploy modal_app.py` serves the env with Modal proxy auth (one eval GPU per container,
autoscaled). Create a proxy auth token in the Modal dashboard (Settings → Proxy Auth Tokens) and
export `MODAL_PROXY_KEY` / `MODAL_PROXY_SECRET`; `KernelEnv` sends them as `Modal-Key` / `Modal-Secret`.

No GPU (Mac dev): mock evaluator — real static checks, fake timings.

```bash
git clone https://github.com/ScalingIntelligence/KernelBench /tmp/KernelBench
KERNEL_ENV_EVALUATOR=mock KERNELBENCH_DIR=/tmp/KernelBench/KernelBench uv run server
```

Client:

```python
from kernel_env import KernelAction, KernelEnv

with KernelEnv(base_url="http://localhost:8000").sync() as env:
    obs = env.reset(seed=0).observation
    result = env.step(KernelAction(code=my_source))
    print(result.observation.stage, result.observation.feedback, result.reward)
```

## Config (env vars)

| Var | Default | |
|---|---|---|
| `KERNELBENCH_DIR` | `/opt/KernelBench/KernelBench` in image | task root (`level1/`, `level2/`, `level3/`) |
| `KERNEL_ENV_GPUS` | `0` | comma list of GPU ids used for evaluation (one eval per GPU at a time) |
| `KERNEL_ENV_EVALUATOR` | `gpu` | `mock` for machines without CUDA |
| `KERNEL_ENV_MAX_TURNS` | `4` | submissions per episode |
| `KERNEL_ENV_MAX_SESSIONS` | `64` | concurrent WebSocket sessions |
| `KERNEL_ENV_EVAL_LOG` | unset | also append the per-reset/step JSON log lines (always on stdout) to this file |
| `KERNEL_ENV_REWARD` | `speedup` | `roofline`: reward share of hardware roof reached instead of speedup vs PyTorch |
| `KERNEL_ENV_ROOFLINE_REJECT` | `0.8` | reject timings below this fraction of the roofline floor (physically impossible) |

Keep evaluation GPUs separate from training GPUs, or timings (and the reward) get noisy.

## Tests

```bash
uv run --extra dev pytest tests            # Mac-safe; GPU worker tests skip without CUDA
```
