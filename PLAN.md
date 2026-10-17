# Architecture Design: `kernel_env` — RL env for writing fast GPU kernels

Grounded in installed `openenv-core==0.2.2` (`Environment[ActT, ObsT, StateT]`, `create_app`,
`EnvClient`, `openenv.core.rubrics.{Rubric, Sequential, Gate, WeightedSum}`, `openenv init` template).

## Overview

Agent gets a PyTorch reference module (`Model`) plus input shapes. It must return `ModelNew`
whose `forward` runs custom GPU kernels (Triton by default, CUDA C++ via `load_inline` optional).
Env compiles it, checks numerics against the reference on fresh random inputs, times both, and
returns reward = correctness gate + speedup bonus. Episodes are multi-turn: agent sees compile
errors / mismatch stats / timings and refines (Kevin-32B showed multi-turn refinement beats single-shot).

Task source: **KernelBench** (L1 single ops, L2 fusions, L3 full archs) — standard, ~250 problems,
known baselines. Train on a split, hold out a fixed eval set to measure real progress.

Hard part is not OpenEnv plumbing — it is **trustworthy evaluation**: crash/hang isolation, stable
timing, and closing reward hacks (kernel RL is notorious for these). Most of the design below is there.

## Design Decisions

| Decision | Rationale | Trade-offs |
|---|---|---|
| Triton default, CUDA opt-in (`language` field) | Python-native, LLMs write it better, faster compile, portable across NVIDIA archs | Peak perf ceiling lower than hand CUDA/CUTLASS |
| Multi-turn episode, `max_turns=4` | Feedback loop = how humans write kernels (RFC 000: minimize human-agent divergence) | Longer rollouts, more GPU eval per episode |
| Action = full source string, not diffs | Simple, stateless eval, no patch-apply failures | More tokens per turn |
| Every eval in a **fresh subprocess** (`python -I worker.py`, timeout) | Segfaults, illegal memory access, infinite loops, CUDA context poisoning can't kill the server | ~1–3 s process + compile overhead per step |
| One eval per GPU at a time (async GPU lease pool) | Concurrent kernels on same GPU wreck timing | Throughput bounded by #GPUs → scale with GPUs, not sessions |
| `SUPPORTS_CONCURRENT_SESSIONS=True`, high `max_concurrent_envs` | Sessions mostly idle waiting for LLM; GPU pool is the real limiter | Need shared singleton pool across sessions |
| Timing via `triton.testing.do_bench` (warmup, L2 flush, median) for ref **and** candidate in same worker | Same conditions both sides, robust to noise | Ref re-timed each step (could cache per task+GPU type later) |
| Reward via OpenEnv `Rubric` tree (`Sequential` fail-fast + `Gate`) | Uses core abstractions (RFC 004), each component inspectable/loggable | — |
| Correctness: 5 seeds, `atol=rtol=1e-2` (fp32), compare vs ref in same worker | KernelBench convention; multiple seeds kill input-specific hacks | Tolerance per-dtype needed for fp16/bf16 tasks |
| Observation feedback truncated (compile log ≤2 KB, top-k mismatches) | Design for LLM context efficiency (RFC 000) | Agent may miss deep error info |
| Step-based action now, MCP tools later | v1 needs only `submit(code)`; MCP (`profile`, `run_tests`) is natural v2 per two-interface model | — |
| Trainer GPUs ≠ eval GPUs | Training load on same GPU makes speedup reward noise | Need ≥2 GPUs total |

### Reward (per step)

```
invalid format / compile fail / runtime error / timeout  -> 0.0
ran but wrong output                                     -> 0.0   (log stage for analysis)
correct                                                  -> 0.3 + clamp(log2(speedup), -0.2, 3.0)
speedup = t_ref_eager / t_candidate
```
Log-speedup keeps 100× outliers from dominating GRPO advantages. Optional second baseline:
`torch.compile` time, logged (not rewarded) in v1. Episode return = discounted sum (trainer side) or
best-of-episode — configurable.

### Anti-reward-hacking checks (in worker, must all pass before timing)

1. **Static (AST)**: `ModelNew.forward` must launch ≥1 custom kernel (`@triton.jit` fn or `load_inline` module); deny-list torch compute in forward (`torch.matmul`, `F.*`, `torch.nn` layers, `torch.compile`, `.cuda()` sync tricks); deny `import` of `subprocess`, `os.system`, `ctypes`, `threading`, network.
2. **No reference tampering**: ref timed **before** importing candidate; reference module loaded from env's copy, not agent text; static checks block monkeypatching (no attribute assignment outside `self`, no setattr/getattr/dunders). Ref outputs are computed per trial (not all up front) to bound GPU memory on big KernelBench inputs.
3. **Fresh inputs per trial** with unseen seeds → kills cached/memoized outputs.
4. **Output checks**: shape, dtype, device, no NaN/Inf, output must not alias an input, inputs unmodified after call.
5. **Timing integrity**: `torch.cuda.synchronize()` around timed region; reject if candidate launches on non-default stream without sync; sanity cap speedup (e.g. >50× on L1 → flag for review, still log).
6. **Isolation**: per-step temp dir, private `TRITON_CACHE_DIR`/`TORCH_EXTENSIONS_DIR`, own process group (timeout kills nvcc children), `CUDA_VISIBLE_DEVICES` = leased GPU only, network/OS modules blocked by import allow-list.

## Implementation Plan

Scaffold with `openenv init kernel_env`, then:

### Files to Create
1. `kernel_env/models.py` — `KernelAction(code: str, language: Literal["triton","cuda"]="triton")`;
   `KernelObservation(task_id, problem: str, stage: Literal["ready","format","compile","runtime","correctness","timed"], feedback: str, correct: bool|None, max_abs_err, speedup, t_ref_us, t_new_us, turn, max_turns)`;
   `KernelState(State)` + `task_id, level, best_speedup, history: list[dict]`.
2. `kernel_env/client.py` — `KernelEnv(EnvClient[KernelAction, KernelObservation, KernelState])`, `_step_payload`/`_parse_result`/`_parse_state`.
3. `kernel_env/server/tasks.py` — load KernelBench (pinned commit, vendored or HF dataset), deterministic train/eval split, `reset(seed, task_id=...)` picks task; prompt builder (ref code + shapes + GPU name + rules).
4. `kernel_env/server/worker.py` — standalone script run in subprocess: static checks → ref outputs → import candidate → correctness trials → `do_bench` both → JSON result on stdout.
5. `kernel_env/server/evaluator.py` — async `GPUPool` (lease/release per GPU via `asyncio.Queue`), spawn worker with timeout + env vars, parse/truncate result, ref-timing cache keyed `(task_id, gpu_name)`.
6. `kernel_env/server/rubrics.py` — `ExecutesRubric`, `CorrectnessRubric`, `SpeedupRubric`; `rubric = Sequential(Gate(ExecutesRubric()), Gate(CorrectnessRubric()), SpeedupRubric())`.
7. `kernel_env/server/kernel_environment.py` — `KernelEnvironment(Environment[...])`: `reset` picks task, returns problem; `step_async` → evaluator → fill obs → `self._apply_rubric`; `done` when `turn==max_turns` or speedup target hit.
8. `kernel_env/server/app.py` — `create_app(KernelEnvironment, KernelAction, KernelObservation, env_name="kernel_env", max_concurrent_envs=64)`.
9. `kernel_env/server/Dockerfile` — `BASE_IMAGE` = CUDA 12.x runtime + torch + triton (e.g. `pytorch/pytorch:*-cuda12*-devel` for nvcc when CUDA mode on); run with `--gpus all --network none`.
10. `kernel_env/tests/` — fixture kernels (good, wrong, each hack type) + mock-evaluator tests runnable on Mac.
11. `train/grpo_kernel.py` — TRL `GRPOTrainer` + rollout loop over `KernelEnv` client (multi-turn), eval on held-out split every N steps.
12. `kernel_env/openenv.yaml`, `pyproject.toml`, `README.md` — from template.

### Files to Modify
None (greenfield).

### Implementation Order
1. `openenv init kernel_env`; write `models.py` + `client.py`.
2. `worker.py` standalone — run by hand on GPU box against 3 KernelBench L1 tasks with hand-written Triton kernels. **Gate: timing of ref-vs-ref within ±5%.**
3. Anti-hack fixtures; make each one fail in worker.
4. `evaluator.py` GPU pool + subprocess/timeout; `rubrics.py`.
5. `kernel_environment.py` + `app.py`; local `uvicorn` on GPU box; client smoke test.
6. Dockerfile; `openenv build` / `docker run --gpus all`; `openenv validate`.
7. Baseline eval: run a frozen model (e.g. Qwen2.5-Coder-7B / 32B) zero-shot on held-out set → record compile %, correct %, mean speedup.
8. GRPO training script; small run on L1 train split; watch reward + held-out metrics + hack-flag rate.
9. v2: MCP tools (`profile` via `ncu`/`torch.profiler` summary, `run_tests`), L2/L3 curriculum, bf16 tasks.

## Verification Plan
- **Unit (Mac, mock GPU)**: models round-trip, rubric math, prompt builder, AST deny-list on fixtures.
- **Worker (GPU)**: good kernel → correct, speedup>0; wrong kernel → `correctness` stage; `while True` → timeout; illegal address → `runtime`, server alive; calls `torch.matmul` / caches output / mutates input / aliases output → rejected.
- **Timing stability**: ref vs itself 20× → stddev of speedup < 3%.
- **E2E**: `KernelEnv.from_docker_image(...)` reset/step loop; 16 concurrent sessions on 2 GPUs, no timing drift.
- **Training signal**: GRPO smoke run, held-out correct% and median speedup rise; manually inspect top-10 speedup kernels each eval for hacks.

## RFC Required?
**NO** — standard env built on existing `Environment`/`EnvClient`/`Rubric` abstractions. MCP tool surface in v2 may warrant one only if it needs new core APIs.

## Open decisions (defaults chosen)
- **GPU host**: Mac M4 has no CUDA → server must run on NVIDIA box (cloud VM: Lambda/RunPod/Modal; ≥1 eval GPU + training GPUs). Dev/unit tests on Mac via mock evaluator.
- **Kernel language**: Triton (default).
- **Policy model + trainer**: Qwen2.5-Coder-7B-Instruct + TRL GRPO.

## Status (2026-10-17)
Scaffolded `kernel_env/` (steps 1–6 code-complete; GPU verification + steps 7–9 pending). Mac: 31 tests pass, live server smoke with mock evaluator passes.
Modal (2026-10-17): `modal_app.py` — 37/37 tests pass on L40S (real Triton + CUDA eval); env deployed behind Modal proxy auth (401 without token); `job` smoke (0.5B baseline, 2×L40S, env on its own GPU) ran end-to-end.
Roofline (2026-10-17): per-task FLOPs/bytes/arithmetic intensity + datasheet peaks → `t_min`, `roofline_frac`; timings < 0.8·t_min rejected; optional `KERNEL_ENV_REWARD=roofline`. L40S check: 4096³ fp32 matmul, PyTorch 28% of TF32 roof, naive Triton tl.dot 60% (2.1× speedup comes from TF32). GRPO for 9B: LoRA default (full FT = 154 GB state), 4 eval GPUs per 4 training GPUs.
