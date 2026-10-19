# Small OpenEnv learning experiment

This pilot measures whether a short, real-GPU GRPO run improves generated kernels.
It uses the existing OpenEnv WebSocket client, evaluator, reward rubric and trainer.
It does not require beating PyTorch.

```bash
TRAIN_GPU=H100:2 modal run modal_app.py::job \
  --max-turns 2 --reward roofline --workers-per-gpu 1 \
  --cmd 'timeout 5400 python small_run.py --out /runs/proof-NEW --steps 16'
```

Use a fresh output directory. The command bounds execution to 90 minutes; monitor
the app's CPU/memory charges as well as GPU charges before reusing a dollar budget.
One GPU generates/trains; the other evaluates kernels. The evaluator is reached
over the container's loopback interface.

The runner evaluates the pristine Qwen2.5-Coder-7B-Instruct model, trains a rank-16
LoRA adapter for 16 GRPO steps, merges it using the existing merger, then evaluates
the freshly loaded merged model with the same greedy decoding and two-turn limit.
Each group has four independently sampled training episodes. Environment feedback
tokens are excluded from the loss. Rewards require numerically correct kernels;
the roofline reward distinguishes correct implementations by measured runtime.

The task list is fixed before training:

| Group | Tasks | Interpretation |
| --- | --- | --- |
| Training | ReLU, LeakyReLU, Sigmoid, Tanh | Learning on these tasks only |
| Reserved transfer | Swish, HardSigmoid, HardTanh | Related tasks excluded from optimizer updates |
| Official held-out | Softmax, LogSoftmax, MinGPTNewGelu | Tasks in the repository's deterministic eval split |

There is one greedy episode per task before and after. Correctness is measured
both on turn one and on the best of two turns. The mean best speedup including
failures assigns zero to failed episodes; speed among correct episodes alone can
be misleading when the solved task set changes. Three held-out tasks and one
training seed provide pilot evidence, not a statistical claim about KernelBench.
Related operator families appear in training and evaluation.

Artifacts on the `kernel-env-runs` Modal volume include `manifest.json`,
`before.jsonl`, `adapter/`, `merged/`, `after.jsonl`, `comparison.json` and phase
logs. The JSONL files retain generated source, errors, rewards and timings.
The 2026-10-19 run also saves exact model/package/GPU/source provenance and a
container-side budget watchdog. Inspect actual optimizer gradients, adapter
weights and generated evaluation results before claiming a learning gain.

The design borrows discounted interaction rewards and conservative gradient
clipping from [Kevin](https://cognition.com/blog/kevin-32b), and verification and
speed-objective cautions from [Dr. Kernel](https://arxiv.org/abs/2602.05885) and
[TritonRL](https://arxiv.org/abs/2510.17891). It uses the repository's GRPO trainer;
it does not implement Dr. Kernel's TRLOO or reproduce
[CUDA Agent](https://cuda-agent.github.io/)'s staged actor–critic training.

The 2026-10-19 run completed: 16 updates / 4 epochs; matched best-of-two
correctness improved from 5/10 to 6/10. The extra success was held-out MinGPT GELU
(0/3 to 1/3 official held-out tasks); training and reserved-transfer correctness
were unchanged. First-turn correctness stayed 5/10. The GELU kernel also passed
five new signed-input seeds at the original shape, within benchmark tolerance.

Reported Modal spend was $3.48551555 including GPU checks and the additional
replay; all GPU jobs stopped. Billing can lag and persistent storage may accrue
separately. Results, adapter weights, source/version provenance and billing are
saved under `runs/proof-20261019/results/`.

The private [dashboard](https://kernel-lab-20261019.shreyansh-pathak273.chatgpt.site)
includes all optimizer traces and episodes. Its
[methodology page](https://kernel-lab-20261019.shreyansh-pathak273.chatgpt.site/methodology.html)
explains the architecture, diagrams, reward policy, protocol and limitations.
This is a small observed gain from one seed, not robust evidence of broad
generalization.
