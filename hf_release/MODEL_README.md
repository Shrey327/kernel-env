---
license: apache-2.0
base_model: Qwen/Qwen2.5-Coder-7B-Instruct
base_model_relation: adapter
library_name: peft
pipeline_tag: text-generation
language:
  - en
tags:
  - lora
  - grpo
  - trl
  - openenv
  - triton
  - gpu-kernels
  - reinforcement-learning
---

# Qwen2.5-Coder-7B · KernelEnv GRPO, 16 updates

An experimental **LoRA adapter**, not a standalone set of 7B model weights, trained with
real GPU execution feedback through [Kernel Env](https://huggingface.co/spaces/Shreyansh327/kernel-env).
Load it on the pinned Qwen2.5-Coder-7B-Instruct base below.

The matched pilot observed best-of-two task correctness changing **5/10 → 6/10**.
First-turn correctness stayed **5/10 → 5/10**. This is a small observed refinement gain,
not a broad improvement or a claim to beat PyTorch across KernelBench.

## Load the trained model

```bash
pip install torch transformers peft accelerate
```

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_id = "Qwen/Qwen2.5-Coder-7B-Instruct"
base_revision = "c03e6d358207e414f1eca0bb1891e29f1db0e242"
adapter_id = "Shreyansh327/Qwen2.5-Coder-7B-KernelEnv-GRPO-16"

tokenizer = AutoTokenizer.from_pretrained(base_id, revision=base_revision)
base = AutoModelForCausalLM.from_pretrained(
    base_id, revision=base_revision, torch_dtype=torch.bfloat16, device_map="auto"
)
model = PeftModel.from_pretrained(base, adapter_id).eval()

# Use the actual reference and rules returned by a running KernelEnv.reset().
prompt = open("task_prompt.txt").read()
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=2048, do_sample=False)
code = tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(code)
```

GPU memory is required for practical inference; loading the bf16 base alone needs roughly
15 GB plus runtime overhead. Publishing the adapter does not provide a free hosted inference
endpoint. The two-turn evaluation also supplies GPU feedback after the first answer; the
single generation above does not reproduce the full two-turn evaluation.

## Training

| Setting | Value |
|---|---|
| Starting point | Pristine Qwen2.5-Coder-7B-Instruct, pinned revision above |
| LoRA | rank 16, alpha 32, dropout 0.05; q/k/v/o/up/down/gate projections |
| Optimization | 16 actual updates, four recorded epochs; initial LR 5e-5; clip 0.05 |
| Loss | Dr. GRPO; episode-level group-relative advantages |
| Rollout | Four trajectories/group, two turns, 2,048 tokens/turn, temperature 0.8 |
| Return | r1 + 0.4 × r2; feedback tokens masked from the loss |
| Reward | Zero for incorrect; roofline reward for correct GPU kernels; no partial credit |
| Tasks | ReLU, LeakyReLU, Sigmoid, Tanh, from pinned KernelBench |
| Hardware | Two H100 80GB GPUs; model on one, evaluator on the other |
| Seed | 42 |
| Data | 64 training episodes, 128 submitted kernels; no SFT warm-up |

The saved adapter was merged into 196 base tensors and the merged model freshly reloaded
for evaluation. All 16 recorded gradient norms were finite and nonzero. These integrity checks
are necessary controls, not independent evidence that every update helped.

## Matched evaluation

One greedy episode per task before and after, with the same two-attempt limit and evaluator.

| Task split | Before | After |
|---|---:|---:|
| Training: ReLU, LeakyReLU, Sigmoid, Tanh | 3/4 | 3/4 |
| Reserved transfer: Swish, HardSigmoid, HardTanh | 2/3 | 2/3 |
| Official held-out: Softmax, LogSoftmax, MinGPTNewGelu | 0/3 | 1/3 |
| Overall best of two | 5/10 | 6/10 |
| Overall first turn | 5/10 | 5/10 |

The additional MinGPT GELU solution appeared on turn two after a compilation error. It used
an erf formulation that agreed with the reference's tanh approximation within benchmark
tolerance, not bit for bit. Timing was about 0.1838 ms versus 1.6239 ms eager PyTorch
(8.8347× for this one task). Five further original-shape signed-input seeds passed, with maximum
absolute error 0.0004734993. Existing solved kernels' runtimes were essentially unchanged.

## Evidence and reproducibility

- `evaluation/comparison.json`: per-split before/after results, including failures.
- `evaluation/manifest.json`: exact tasks, sampling budget and phase durations.
- `evaluation/metrics.json`: all recorded optimizer traces.
- `evaluation/signed_validation.json`: additional signed-input replay.
- `evaluation/gelu_after.py`: the generated successful kernel, for inspection.
- `evaluation/provenance.json`: runtime versions, revisions and experiment source hashes.
- `release.json`: adapter hash, source experiment and publication metadata.

Experiment runtime: Torch 2.13.0, Triton 3.7.1, TRL 1.16.0, Transformers 5.17.0,
PEFT 0.21.2, vLLM 0.31.0 and OpenEnv 0.3.0. These record the measured run; the
environment's downloadable Docker recipe uses its existing PyTorch 2.7.1 base.
KernelBench revision: `423217d9fda91e0c2d67e4a43bf62f96f6d104f1`.

## Limits and intended use

Research prototype for GPU kernel generation and execution-feedback experiments. One training
seed, ten evaluation tasks and related operator families do not establish statistically robust
or unseen-family generalization. Public pretraining contamination is not excluded. No claim of
equivalence to CUDA Agent, Dr. Kernel or Kevin is made; this is episode-level GRPO.

Generated source can fail, be numerically wrong, or crash a GPU worker. Review and evaluate it
in an isolated environment; static guards are not a general sandbox for arbitrary native code.
No further training, inference deployment or paid GPU allocation accompanies this upload.

The base is Apache 2.0 licensed; the adapter is released under Apache 2.0. The original
adapter tensors are unchanged. Publication pins the adapter config's base `revision`, which
was null in the training export, to the exact experiment revision.
