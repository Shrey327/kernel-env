---
title: Kernel Env — OpenEnv GPU Kernel RL
emoji: ⚡
colorFrom: green
colorTo: gray
sdk: static
app_file: index.html
pinned: false
license: bsd-3-clause
tags:
  - openenv
  - reinforcement-learning
  - triton
  - cuda
  - kernelbench
models:
  - Shreyansh327/Qwen2.5-Coder-7B-KernelEnv-GRPO-16
---

# Kernel Env

An OpenEnv environment for iterative Triton/CUDA kernel writing, with execution feedback,
correctness-gated rewards and real GPU timing on KernelBench tasks.

**This free Space hosts documentation, evidence and downloadable environment source.
It is static: there is no live `/ws` endpoint, GPU evaluator, model inference or training here.**
The environment itself runs on an NVIDIA GPU with Docker or the included Modal runner.
Creating a new Docker Space currently requires a paid Hugging Face plan; no subscription,
GPU hardware or inference endpoint was enabled for this publication.

## Run the real environment on your own GPU

```bash
git clone https://huggingface.co/spaces/Shreyansh327/kernel-env
cd kernel-env
docker build -t kernel-env:latest .
docker run --rm --gpus all -p 127.0.0.1:8000:8000 \
  -e KERNEL_ENV_GPUS=0 kernel-env:latest
```

Requires an NVIDIA CUDA-capable host, Docker and NVIDIA Container Toolkit. The first build
downloads CUDA/PyTorch and pinned KernelBench sources. The server provides OpenEnv schemas,
health and WebSocket sessions on port 8000. The image pins OpenEnv 0.3.0 and uses the existing
PyTorch 2.7.1/CUDA 12.8 development base, including nvcc for custom CUDA extensions.
This release Dockerfile was statically checked; its complete image was not built locally.
The evaluator source passed a prior 116-test real-GPU suite; that is separate from an image build.

In another shell, install the client and send source:

```bash
python -m pip install "openenv-core[core]==0.3.0" ./kernel_env
python examples/client_smoke.py --url http://127.0.0.1:8000
```

`examples/relu.py` is a real custom Triton example. The smoke script checks the server's ReLU
task, submits the example and requires a correct GPU result. Never point it at this static
Space URL: it needs the server started above. Do not expose the execution server to untrusted
users; the source checks and subprocess boundary are not a security sandbox for native code.

`modal_app.py` and `train/` contain the cloud runner and bounded training recipe. Running them
allocates paid cloud resources. No Modal application was deployed by this publication.

## How it works

`reset(task_id, seed)` returns a reference PyTorch module. The policy submits a complete
`ModelNew` source through `step(KernelAction(...))`. Static checks, compilation, hidden-seed
correctness trials and synchronized GPU timing yield errors or a reward. The model can repair
its submission within the episode's turn limit (default four; the proof run used two).

In roofline mode, incorrect submissions receive zero; correct kernels receive
`0.3 + 2.7 * min(t_min / t_kernel, 1)`. They do not have to beat PyTorch to earn reward.
See [environment documentation](kernel_env/README.md) for checks, limitations and configuration.

## Small-run evidence

The separately published [LoRA adapter](https://huggingface.co/Shreyansh327/Qwen2.5-Coder-7B-KernelEnv-GRPO-16)
received 16 GRPO updates from a pristine Qwen2.5-Coder-7B-Instruct base on four activation tasks.
Matched greedy evaluation on ten fixed tasks, two submissions each:

| Correct tasks | Before | After |
|---|---:|---:|
| Training | 3/4 | 3/4 |
| Reserved transfer | 2/3 | 2/3 |
| Official held-out | 0/3 | 1/3 |
| Overall, best of two | 5/10 | 6/10 |
| Overall, first turn | 5/10 | 5/10 |

The new GELU success occurred on turn two and passed an additional five-seed signed-input replay.
It measured 8.83× eager PyTorch for that task only; existing solved kernels were essentially
unchanged in speed. One seed and ten related-family tasks do not establish robust or
unseen-family generalization. Public benchmark pretraining contamination is not excluded.

The current experiment and verification reported $3.48551555 Modal compute usage.
This excludes historical runs, future storage and billing lag. Publishing this static Space
and adapter starts no training or GPU inference service.

The [model repository](https://huggingface.co/Shreyansh327/Qwen2.5-Coder-7B-KernelEnv-GRPO-16/tree/main/evaluation)
contains comparison data, optimizer traces, a validation record and experiment provenance.

## Revisions and attribution

- KernelBench: `423217d9fda91e0c2d67e4a43bf62f96f6d104f1` (downloaded during image build).
- Base model: `c03e6d358207e414f1eca0bb1891e29f1db0e242`.
- OpenEnv-derived template files retain their Meta copyright headers; see `LICENSE`.
- Model weights are distributed separately under Apache 2.0; see that repository's license.
- This source snapshot includes local evaluator fixes used in the experiment. It is not
  represented as an upstream merged release.
