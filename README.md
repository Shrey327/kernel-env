# KernelEnv

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv) reinforcement learning environment where a language model gets a PyTorch module from [KernelBench](https://github.com/ScalingIntelligence/KernelBench) and has to write a faster Triton kernel for it. Every answer is compiled, checked for correctness on fresh inputs, and timed against PyTorch on a real H100.

The hard part was not RL. It was building a judge the model could not fool.

**Write-up:** [Teaching a 7B Model to Write GPU Kernels with RL (and Every Way It Tried to Cheat)](https://medium.com/@shreyansh.pathak27/teaching-a-7b-model-to-write-gpu-kernels-with-rl-and-every-way-it-tried-to-cheat-69a9cc9987ba)
**Environment:** [Hugging Face Space](https://huggingface.co/spaces/Shreyansh327/kernel-env) · **Trained model:** [Qwen2.5-Coder-7B-KernelEnv-Fusion-RL](https://huggingface.co/Shreyansh327/Qwen2.5-Coder-7B-KernelEnv-Fusion-RL)

## Results

Qwen2.5-Coder-7B-Instruct, LoRA SFT warm start then GRPO (64 steps, group size 8), about **$60** of Modal GPU time. Base, SFT and RL models were scored on the same 30 tasks with 4 samples each (120 attempts per model), same prompts, same evaluator.

| Correctness | Base | SFT | RL |
|---|---|---|---|
| Overall | 15.8% | 15.0% | **20.0%** |
| Training tasks | 81.2% | 68.8% | **93.8%** |
| Reserved transfer tasks | 15.6% | 21.9% | **21.9%** |
| Held-out level 2 | 2.8% | 0% | **5.6%** |
| Held-out level 1 | 0% | 0% | 0% |

On transfer tasks, RL's correct kernels were also faster: **1.64×** mean speedup versus 1.30× for the base model. Best single kernel: a held-out fused GELU at **8.8×** eager PyTorch, reaching 85% of the memory roofline where PyTorch reached 10%.

Read these honestly: RL is best or tied in every group, but every 95% bootstrap interval touches or crosses zero. Thirty tasks with four samples each is not enough to prove the gain.

## How an episode works

1. The environment picks a KernelBench task (pinned commit) and shows the PyTorch reference plus the rules.
2. The model returns a `ModelNew` class whose `forward` launches its own Triton kernels.
3. A fresh, isolated worker process on an H100 runs static checks, compiles the code, checks numerics against the reference on 5 unseen random seeds, then times both.
4. Compile errors, mismatch stats or timings go back to the model, which gets up to 3 tries.
5. The best try's reward feeds GRPO.

**Reward:** wrong or invalid scores 0. Correct scores `0.25 + clip(log2(speedup lower bound), 0, 2)`. Log speedup stops one 100× outlier from dominating the group advantages.

## What the judge blocks

Every submission has to pass all of these before it is timed.

| Exploit | Defense |
|---|---|
| Calling PyTorch instead of writing a kernel | AST checks; at least one `@triton.jit` kernel required; runtime dispatch guard on tensor ops in `forward` |
| Monkeypatching the judge, `getattr`/dunder/`exec` escapes | No attribute writes outside `self`; banned builtins; import allow list |
| Slowing the reference at import time | Reference timed before the candidate is imported |
| Memorized or cached outputs | 5 fresh random seeds per step, never shown to the model |
| Returning an input as the output, writing into inputs | Storage alias check; float64 checksums of inputs before and after |
| Faking a result on stdout | Worker writes its result to a file |
| Hiding work on an unjoined side CUDA stream | Profiler check that only one stream ran |
| Segfaults, illegal memory access, hangs | Fresh subprocess per eval in its own process group; timeouts kill child compilers |
| Skipped work that looks impossibly fast | Times below 0.8× the roofline floor are rejected as physically impossible |

The roofline (`kernel_env/server/roofline.py`) also exposes precision tricks: on fp32 tasks `tl.dot` silently uses TF32 tensor cores and still passes a 1e-2 tolerance, so each kernel's share of the correct roof is reported alongside its speedup.

## Trustworthy timing

A reward is only useful if it is larger than its own noise. The first setup had run-to-run noise of 0.031 against a 0.073 gap that needed detecting. The fix:

- Capture both the reference and the candidate as CUDA graphs to remove launch overhead from the measurement.
- Run 7 paired rounds, alternating reference and candidate, so clock drift hits both equally.
- Compute a noise band from the interquartile range and reward the conservative lower bound of the speedup.

Noise dropped to **0.0085**.

## Repository layout

| Path | What |
|---|---|
| `kernel_env/` | The OpenEnv environment: server, evaluator, isolated GPU worker, static checks, roofline, CUDA-graph timing, tests. See [`kernel_env/README.md`](kernel_env/README.md). |
| `train/` | Baseline evaluation, GRPO training with TRL and vLLM, run contracts and dashboards. See [`train/README.md`](train/README.md). |
| `hf_release/` | Hugging Face Space and model card packaging |
| `modal_*.py`, `launch_*.py` | Modal entry points for evaluation and training campaigns |
| `PLAN.md` | Original design notes |

## Quick start

GPU host (NVIDIA + nvidia-container-toolkit):

```bash
cd kernel_env
docker build -t kernel_env:latest -f server/Dockerfile .
docker run --gpus all -p 8000:8000 -e KERNEL_ENV_GPUS=0 kernel_env:latest
```

No GPU (mock evaluator: real static checks, fake timings):

```bash
git clone https://github.com/ScalingIntelligence/KernelBench /tmp/KernelBench
cd kernel_env
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

Keep evaluation GPUs separate from training GPUs, or timing noise leaks into the reward.

## Known gaps

- Plain tensor arithmetic (`x + y`) in host code is caught by the runtime dispatch guard but not yet by static checks. The guard is not a native-code sandbox.
- Held-out level 1 tasks stay at 0% for every model at this budget.
- Next: re-evaluate with 16 samples per task to tighten the intervals, and train longer on fusion tasks.
