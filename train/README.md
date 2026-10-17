# train/ — baseline eval + GRPO for kernel_env

| File | What |
|---|---|
| `baseline_eval.py` | Frozen model on held-out tasks, multi-turn (env's `max_turns`). Prints compile %, correct % (best of episode and turn 1), faster-than-ref %, mean/median speedup of correct episodes, stage histograms, suspicious-flag count; writes per-episode JSONL. |
| `grpo_kernel.py` | TRL 1.16 `GRPOTrainer` + `rollout_func`. One dataset row = one KernelBench task id; TRL repeats it `--num_generations` times (= one GRPO group). Each rollout is a full episode: generate → `env.step` → feedback turn → … Feedback tokens get `env_mask=0` (not trained). Reward = `Σ γ^t r_t` (`--return_mode discounted --gamma 0.9`) or best turn (`--return_mode best`). |
| `test_train.py` | Mac checks: code extraction, return math, token stitching == chat template, rollout vs a mock server. |

Both scripts enumerate task ids from a local KernelBench copy (`KERNELBENCH_DIR`); the train/eval
split is the server's hash split, so it matches. Default `--levels 1`.

## Modal

`modal_app.py::job` (repo root) runs one command in `/root/train` on a `TRAIN_GPU` container,
with the env server on the last `--eval-gpus` GPUs (`$KERNEL_ENV_URL`) and the command on the rest.
Outputs go to `/runs` (Modal volume `kernel-env-runs`); HF models are cached in `kernel-env-hf-cache`.
Optional secrets: `TRAIN_SECRETS=wandb,huggingface` (names of your Modal secrets).

```bash
# baseline (fast path): `vllm serve` on GPU 0 with MTP speculative decoding, kernel evals on GPUs 1-3;
# episodes run concurrently, so one episode's generation overlaps another's kernel eval
TRAIN_GPU=H100:4 modal run modal_app.py::job --eval-gpus 3 --cmd 'python baseline_eval.py --env-url $KERNEL_ENV_URL \
  --model Qwen/Qwen3.5-9B --max-new-tokens 8192 --max-model-len 65536 --spec-mtp 2 \
  --vllm-args "--language-model-only --reasoning-parser qwen3" --out /runs/baseline_qwen35_9b.jsonl'

# GRPO, Qwen3.5-9B + LoRA: 4 training GPUs (vLLM colocated), 4 kernel-eval GPUs (sizing below)
TRAIN_GPU=H100:8 modal run modal_app.py::job --eval-gpus 4 --reward roofline --cmd 'accelerate launch --num_processes 4 grpo_kernel.py \
  --model_name_or_path Qwen/Qwen3.5-9B --dtype bfloat16 --bf16 true --env_url $KERNEL_ENV_URL \
  --use_peft --lora_r 64 --lora_alpha 128 --lora_target_modules all-linear \
  --use_vllm --vllm_mode colocate --vllm_gpu_memory_utilization 0.3 \
  --num_generations 8 --per_device_train_batch_size 8 --gradient_accumulation_steps 2 \
  --max_completion_length 4096 --learning_rate 1e-5 --gradient_checkpointing true \
  --eval_strategy steps --eval_steps 25 --num_generations_eval 1 --report_to none --output_dir /runs/grpo_q35_9b'
```

Eval GPU type = training GPU type here (one container). Speedups are only comparable on the same GPU type.

### Sizing (Qwen3.5-9B, P = 9.65 B params incl. vision tower)

| | Formula | Qwen3.5-9B |
|---|---|---|
| bf16 weights | 2·P | 19.3 GB |
| Full fine-tune state (bf16 weights + grads, fp32 master + Adam m, v) | 16·P | 154 GB: DDP does not fit on 80 GB; ZeRO-3 needs 154/n GB per GPU **plus** activations and vLLM |
| LoRA (frozen bf16 base + adapters ≈ 1–2 % of P with fp32 Adam) | 2·P + 16·P_lora | ≈ 22 GB per GPU before activations |
| Colocated vLLM | `vllm_gpu_memory_utilization` × 80 GB | 0.3 → 24 GB: 19.3 GB weights + ≈ 5 GB KV cache |
| Rollouts per optimizer step R | per_device_bs × processes × grad_accum | 8 × 4 × 2 = 64 (8 groups of 8) |
| Kernel evals per step | ≤ R × max_turns | ≤ 256 |
| Eval-queue time per step | evals × eval_s ÷ eval GPUs | 256 × ~10 s ÷ 4 ≈ 11 min |

`eval_s` (≈ 10 s here: process start + Triton compile + 5 correctness trials + 2 benchmarks) is an
estimate; read the measured p50/p95 under "Where the time went" in `report.py` output, and size
`--eval-gpus` so the eval queue does not dominate the step. Full fine-tuning instead of LoRA needs
ZeRO-3 (an accelerate DeepSpeed config) on ≥ 4 training GPUs with vLLM in server mode on its own GPU.

## GPU box

Keep eval GPUs and training GPUs disjoint, or timing noise leaks into the reward. Example for 8 GPUs:
GPUs 0–1 run the env (one kernel eval per GPU at a time), 2–7 train.

```bash
# 1. env server on eval GPUs only
docker build -t kernel_env:latest -f kernel_env/server/Dockerfile kernel_env
docker run -d --gpus '"device=0,1"' -p 8000:8000 \
  -e KERNEL_ENV_GPUS=0,1 -e KERNEL_ENV_MAX_SESSIONS=256 kernel_env:latest
```

Inside the container the two leased devices show up as 0,1, hence `KERNEL_ENV_GPUS=0,1`.

```bash
# 2. trainer env (separate venv; needs vllm for the GPU paths)
cd train && uv venv --python 3.12 && uv pip install trl vllm peft datasets -e ../kernel_env
git clone https://github.com/ScalingIntelligence/KernelBench /opt/KernelBench
export KERNELBENCH_DIR=/opt/KernelBench/KernelBench

# 3. baseline on held-out split (vLLM on a training GPU, not an eval GPU)
CUDA_VISIBLE_DEVICES=2 python baseline_eval.py --env-url http://localhost:8000 \
  --model Qwen/Qwen2.5-Coder-7B-Instruct --backend vllm --out baseline_7b.jsonl

# 4. GRPO: vLLM server on GPU 2, training on GPUs 3-7
CUDA_VISIBLE_DEVICES=2 trl vllm-serve --model Qwen/Qwen2.5-Coder-7B-Instruct &
CUDA_VISIBLE_DEVICES=3,4,5,6,7 accelerate launch --num_processes 5 grpo_kernel.py \
  --model_name_or_path Qwen/Qwen2.5-Coder-7B-Instruct --dtype bfloat16 --bf16 true \
  --env_url http://localhost:8000 \
  --use_vllm --vllm_mode server \
  --num_generations 8 --per_device_train_batch_size 8 --gradient_accumulation_steps 4 \
  --max_completion_length 4096 --temperature 1.0 --learning_rate 1e-6 \
  --gradient_checkpointing true --return_mode discounted --gamma 0.9 \
  --eval_strategy steps --eval_steps 25 --per_device_eval_batch_size 8 --num_generations_eval 1 \
  --logging_steps 1 --report_to wandb --output_dir runs/grpo_7b
```

Sizing notes:
- Concurrent env sessions = rollouts in flight across all processes (`per_device_train_batch_size × steps_per_generation × num_processes`); keep it ≤ `KERNEL_ENV_MAX_SESSIONS`.
- Episode wall time is bound by the eval-GPU pool: N sessions queue for 2 GPUs. `--env_timeout_s` (default 900) must cover that queueing; add eval GPUs before raising it.
- `--max_completion_length` is per turn; an episode completion is up to `max_turns ×` that plus feedback, so memory grows with turns.
- Single-GPU-trainer alternative: `--use_vllm --vllm_mode colocate`.

## Seeing what happened

| Source | What | Where |
|---|---|---|
| `baseline_eval.py` stdout | `[turn k] live, gen s, slowest step, stage mix, correct %` per turn; one line per finished episode | terminal / Modal logs |
| `--out run.jsonl` | one line per episode, written as it finishes: completions, per-turn stage/feedback/speedup/timings (`gen_s`, `step_s`), `episode_id`, env history (`gpu_id`, `queue_wait_s`, `eval_s`, `code_sha`) | `/runs/` on Modal |
| env server | one JSON line per `reset`/`step` (`"event": "step"`, same `episode_id`/`code_sha`); also appended to `$KERNEL_ENV_EVAL_LOG` | stdout; `/runs/evals-<time>.jsonl` in Modal jobs |
| `--wandb PROJECT` | per-turn metrics, summary, episode table (Modal: `TRAIN_SECRETS=wandb`) | W&B |
| `report.py run.jsonl -o run.html` | stage funnel, turn-by-turn progress, speedup distribution, timing p50/p95, every episode with code, feedback and diff between turns | static HTML |

```bash
modal volume get kernel-env-runs baseline_qwen35_9b.jsonl .
python report.py baseline_qwen35_9b.jsonl -o baseline.html --title "Qwen3.5-9B baseline"
grep '"stage": "timeout"' evals-*.jsonl     # server log: join to episodes on episode_id
```

## Logged metrics

`kernel/{compile_pct, correct_pct, correct_turn1_pct, faster_than_ref_pct, mean_speedup, suspicious, turns}`,
under train and `eval_` (held-out, every `--eval_steps`). Episodes the env flags as suspicious
(correct and >50× speedup, from `env.state().history`) are logged as warnings and appended with
their code to `<output_dir>/suspicious.jsonl`. Read those before trusting a speedup jump.

## Mac (no CUDA)

```bash
mkdir -p /tmp/kb/level1 && for n in 1 2 3 12 19; do cp ../kernel_env/tests/fixtures/level1/19_ReLU.py /tmp/kb/level1/${n}_ReLU.py; done
(cd ../kernel_env && KERNEL_ENV_EVALUATOR=mock KERNELBENCH_DIR=/tmp/kb .venv/bin/uvicorn server.app:app --port 8765) &
KERNEL_ENV_URL=http://127.0.0.1:8765 .venv/bin/python -m pytest -q test_train.py
KERNELBENCH_DIR=/tmp/kb .venv/bin/python grpo_kernel.py --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct --dtype float32 \
  --env_url http://127.0.0.1:8765 --output_dir /tmp/run --use_cpu --report_to none --save_strategy no \
  --num_generations 2 --per_device_train_batch_size 2 --max_completion_length 96 --max_steps 2 --logging_steps 1 \
  --eval_strategy steps --eval_steps 2 --per_device_eval_batch_size 2 --num_generations_eval 2
```

Copies of the ReLU fixture under several ids put tasks in both splits (`1_`, `12_` hash to eval).
Mock evaluator: real static checks, every passing submission is "correct, 1.00×" (reward 0.3).
