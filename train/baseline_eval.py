"""Baseline: frozen model on the held-out KernelBench split, multi-turn, via the kernel_env server.

    python baseline_eval.py --env-url http://localhost:8000 --model Qwen/Qwen2.5-Coder-7B-Instruct --backend vllm

Each eval task is one episode of up to `max_turns` submissions; after each submission the model
sees the env feedback (stage, errors, timings) and may refine. Reports compile %, correct %,
mean/median speedup (best correct submission per episode) and a stage histogram.

Also holds the helpers grpo_kernel.py shares (code extraction, feedback message, metrics).
"""

import argparse
import json
import os
import re
import shlex
import statistics
import subprocess
import threading
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from kernel_env import KernelAction, KernelEnv
from kernel_env.server.tasks import load_tasks

STAGE_ORDER = ["format", "compile", "runtime", "timeout", "correctness", "timed"]
_FENCE = re.compile(r"```(\w*)[^\n]*\n(.*?)```", re.DOTALL)


def episode_return(rewards: list, mode: str, gamma: float) -> float:
    if mode == 'best':
        return max(rewards, default=0.0)
    return sum(gamma**t * r for t,r in enumerate(rewards))


def extract_code(text: str) -> str:
    """Last ```python block, else last fenced block, else the raw text (prompt asks for bare source)."""
    blocks = _FENCE.findall(text)
    py = [body for lang, body in blocks if lang.lower() in ("python", "py")]
    if py or blocks:
        return (py or [body for _, body in blocks])[-1].strip()
    # unterminated fence (generation cut off): drop the opening line
    if "```" in text:
        return text.split("```", 1)[1].split("\n", 1)[-1].strip()
    return text.strip()


def feedback_message(obs, truncated: bool = False) -> str:
    """User turn shown to the model after a submission."""
    msg = f"Submission {obs.turn}/{obs.max_turns}: stage={obs.stage}."
    if truncated:
        msg = ("Your previous reply hit the token limit and was cut off before the module was complete. "
               "Reply with the module source only, no commentary.\n" + msg)
    if obs.correct and obs.speedup is not None:
        msg += f" Correct, speedup {obs.speedup:.2f}x (ref {obs.t_ref_ms:.3f} ms, yours {obs.t_new_ms:.3f} ms)."
    if obs.feedback:
        msg += f"\n{obs.feedback}"
    if getattr(obs, 'profiling_summary', None):
        msg += f"\nMeasured CUDA profile: {obs.profiling_summary}"
    if obs.turn < obs.max_turns:
        msg += "\nImprove it (fix errors first, then make it faster). Reply with the complete module source only."
    return msg


def summarize(episodes: list) -> dict:
    """episodes: dicts with `turns` = [{"stage", "correct", "speedup", "reward"}, ...]."""
    n = max(len(episodes), 1)
    furthest = [max((STAGE_ORDER.index(t["stage"]) for t in ep["turns"] if t["stage"] in STAGE_ORDER), default=0) for ep in episodes]
    best = [max((t["speedup"] for t in ep["turns"] if t["correct"]), default=None) for ep in episodes]
    sp = [s for s in best if s is not None]
    roof = [max((t["roofline_frac"] for t in ep["turns"] if t["correct"] and t.get("roofline_frac")), default=None)
            for ep in episodes]
    roof = [r for r in roof if r is not None]
    return {
        "episodes": len(episodes),
        "infrastructure_failures": sum(bool(ep.get("infrastructure_error")) or any(t["stage"] == "infrastructure" for t in ep["turns"]) for ep in episodes),
        "compile_pct": 100 * sum(f >= STAGE_ORDER.index("runtime") for f in furthest) / n,
        "correct_pct": 100 * len(sp) / n,
        "correct_turn1_pct": 100 * sum(bool(ep["turns"] and ep["turns"][0]["correct"]) for ep in episodes) / n,
        "faster_than_ref_pct": 100 * sum(s > 1.0 for s in sp) / n,
        "mean_speedup": statistics.fmean(sp) if sp else None,
        "mean_speedup_including_failures": sum(sp)/n,
        "clear_faster_pct": 100*sum(any(t.get("correct") and (t.get("speedup_lower_bound") or 0)>1 for t in ep["turns"]) for ep in episodes)/n,
        "median_speedup": statistics.median(sp) if sp else None,
        "median_roofline_frac": statistics.median(roof) if roof else None,
        "stage_hist_all_turns": dict(Counter(t["stage"] for ep in episodes for t in ep["turns"])),
        "stage_hist_furthest": dict(Counter(STAGE_ORDER[f] for f in furthest)),
        "suspicious": sum(bool(h.get("suspicious")) for ep in episodes for h in ep.get("history", [])),
        **_pass_at_k(episodes, best),
    }


def _pass_at_k(episodes, best) -> dict:
    """With several samples per task (--samples k): pass@k = share of tasks with any correct episode."""
    by_task = {}
    for ep, b in zip(episodes, best):
        by_task.setdefault(ep.get("task_id"), []).append(b is not None)
    k = max((len(v) for v in by_task.values()), default=1)
    if k < 2:
        return {}
    return {"tasks": len(by_task), "samples_per_task": k,
            "pass@1_pct": 100 * sum(sum(v) / len(v) for v in by_task.values()) / len(by_task),
            f"pass@{k}_pct": 100 * sum(any(v) for v in by_task.values()) / len(by_task)}


def split_task_ids(split: str, levels, kernelbench_dir: str = "") -> list:
    """Same deterministic split as the server (hash of task id), enumerated from a local KernelBench copy."""
    tasks = load_tasks(kernelbench_dir, tuple(levels))
    return sorted(t for t, task in tasks.items() if task.split == split)


def run_episodes(generate, env_url: str, task_ids: list, split: str = "eval", env_timeout_s: float = 900.0,
                 on_done=None, on_turn=None) -> list:
    """Turn-synchronous batched episodes: one generate() call per turn for all live episodes.

    on_done(ep) fires as each episode finishes (for incremental writes); on_turn(turn, stats) after each turn.
    """
    envs = [KernelEnv(base_url=env_url, message_timeout_s=env_timeout_s).sync() for _ in task_ids]
    pool = ThreadPoolExecutor(max_workers=max(len(envs), 1))
    try:
        obs = list(pool.map(lambda e, t: e.reset(task_id=t, split=split).observation, envs, task_ids))
        convs = [[{"role": "user", "content": o.problem}] for o in obs]
        eps = [{"task_id": t, "turns": [], "completions": []} for t in task_ids]
        live = list(range(len(envs)))
        turn = 0
        while live:
            turn += 1
            t0 = time.monotonic()
            texts = generate([convs[i] for i in live])
            gen_s = time.monotonic() - t0

            def step(i, code):
                t = time.monotonic()
                return envs[i].step(KernelAction(code=code)), time.monotonic() - t

            results = list(pool.map(step, live, [extract_code(t) for t in texts]))
            for i, text, (r, step_s) in zip(live, texts, results):
                o = r.observation
                eps[i]["completions"].append(text)
                eps[i]["turns"].append({"stage": o.stage, "correct": bool(o.correct), "speedup": o.speedup or 0.0,
                                        "reward": r.reward or 0.0, "feedback": o.feedback,
                                        "t_ref_ms": o.t_ref_ms, "t_new_ms": o.t_new_ms, "t_min_ms": o.t_min_ms,
                                        "bound": o.bound, "roofline_frac": o.roofline_frac,
                                        "ref_roofline_frac": o.ref_roofline_frac,
                                        "gen_s": gen_s, "step_s": step_s})
                convs[i] += [{"role": "assistant", "content": text}, {"role": "user", "content": feedback_message(o)}]
                if r.done:
                    st = envs[i].state()
                    eps[i].update(episode_id=st.episode_id, history=st.history)
                    if on_done:
                        on_done(eps[i])
            if on_turn:
                on_turn(turn, {"live": len(live), "gen_s": gen_s,
                               "stages": dict(Counter(eps[i]["turns"][-1]["stage"] for i in live)),
                               "max_step_s": max(s for _, s in results)})
            live = [i for i, (r, _) in zip(live, results) if not r.done]
        return eps
    finally:
        for e in envs:
            e.close()
        pool.shutdown()


def run_episodes_pipelined(chat, env_url: str, task_ids: list, split: str = "eval", env_timeout_s: float = 3600.0,
                           on_done=None, on_step=None, seeds=None) -> list:
    """One independent loop per episode (generate -> step -> ...), all concurrent.

    Against a vLLM server, continuous batching overlaps one episode's generation with another's
    kernel eval, instead of every episode waiting for the slowest one each turn. An episode that
    raises is kept with its error so one failure never loses the run.
    """
    if seeds is not None and len(seeds) != len(task_ids):
        raise ValueError("one generation seed is required per episode")

    def one(job):
        task_id, seed = job
        ep = {"task_id": task_id, "turns": [], "completions": [], "reasoning": []}
        if seed is not None:
            ep["sample_seed"] = seed
        env = KernelEnv(base_url=env_url, message_timeout_s=env_timeout_s).sync()
        try:
            o = env.reset(task_id=task_id, split=split, **({"seed": seed} if seed is not None else {})).observation
            conv = [{"role": "user", "content": o.problem}]
            while True:
                t0 = time.monotonic()
                text, finish, reasoning, ntok = chat(conv, seed=seed + len(ep["turns"])) if seed is not None else chat(conv)
                gen_s = time.monotonic() - t0
                t1 = time.monotonic()
                r = env.step(KernelAction(code=extract_code(text)))
                step_s = time.monotonic() - t1
                o, truncated = r.observation, finish == "length"
                ep["completions"].append(text)
                ep["reasoning"].append(reasoning)
                ep["turns"].append({"stage": o.stage, "correct": bool(o.correct), "speedup": o.speedup or 0.0,
                                    "reward": r.reward or 0.0, "feedback": o.feedback,
                                    "t_ref_ms": o.t_ref_ms, "t_new_ms": o.t_new_ms, "t_min_ms": o.t_min_ms,
                                    "timing_samples_ms": o.timing_samples_ms, "timing_method": o.timing_method,
                                    "ref_timing_samples_ms": o.ref_timing_samples_ms,
                                    "speedup_lower_bound": o.speedup_lower_bound, "speedup_upper_bound": o.speedup_upper_bound,
                                    "reward_mode": o.reward_mode,"reward_spec":o.reward_spec,
                                            "correctness_reward":o.correctness_reward,"performance_reward":o.performance_reward,
                                            "conservative_speedup":o.conservative_speedup,"rewarded_speedup":o.rewarded_speedup,"measurement_scope":o.measurement_scope,
                                            "graph_repeats":o.graph_repeats,"timing_calibration":o.timing_calibration,"warmup_evidence":o.warmup_evidence,
                                    "bound": o.bound, "roofline_frac": o.roofline_frac,
                                    "ref_roofline_frac": o.ref_roofline_frac, "truncated": truncated,
                                    "completion_tokens": ntok, "gen_s": gen_s, "step_s": step_s})
                if on_step:
                    on_step(ep)
                if r.done:
                    break
                conv += [{"role": "assistant", "content": text}, {"role": "user", "content": feedback_message(o, truncated)}]
            st = env.state()
            ep.update(episode_id=st.episode_id, history=st.history)
        except Exception as e:  # keep partial episode
            ep["error"] = repr(e)
            ep["infrastructure_error"] = True
        finally:
            env.close()
        if on_done:
            on_done(ep)
        return ep

    with ThreadPoolExecutor(max_workers=min(8, max(len(task_ids), 1))) as pool:
        return list(pool.map(one, zip(task_ids, seeds if seeds is not None else [None] * len(task_ids))))


def start_vllm_server(model: str, port: int, max_model_len: int, tp: int, spec_mtp: int, extra: str):
    """`vllm serve` subprocess; returns (proc, chat_factory)."""
    cmd = ["vllm", "serve", model, "--port", str(port), "--max-model-len", str(max_model_len),
           "--tensor-parallel-size", str(tp), *shlex.split(extra)]
    if spec_mtp:  # model's own multi-token-prediction layer drafts tokens; greedy output is identical to plain decoding
        cmd += ["--speculative-config", json.dumps({"method": "mtp", "num_speculative_tokens": spec_mtp})]
    print("starting:", shlex.join(cmd), flush=True)
    proc = subprocess.Popen(cmd)
    deadline = time.monotonic() + 1800
    while True:
        if proc.poll() is not None:
            raise RuntimeError(f"vllm serve exited with {proc.returncode}")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
            return proc
        except OSError:
            if time.monotonic() > deadline:
                proc.kill()
                raise RuntimeError("vllm serve not healthy after 30 min")
            time.sleep(3)


def make_chat(base_url: str, model: str, temperature: float, max_new_tokens: int, enable_thinking: bool):
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key="EMPTY", timeout=7200)

    def chat(conv, seed=None):
        r = client.chat.completions.create(
            model=model, messages=conv, temperature=temperature, max_tokens=max_new_tokens,
            **({"seed": seed} if seed is not None else {}),
            extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}})
        c = r.choices[0]
        return (c.message.content or "", c.finish_reason, getattr(c.message, "reasoning_content", None),
                r.usage.completion_tokens if r.usage else None)

    return chat


def make_generate(backend: str, model: str, temperature: float, max_new_tokens: int, tp: int, batch_size: int,
                  enable_thinking: bool = False, max_model_len: int = 32768):
    # templates without a thinking switch (Qwen2.5) ignore the kwarg
    tmpl = {"enable_thinking": enable_thinking}
    if backend == "vllm":
        from vllm import LLM, SamplingParams

        llm = LLM(model=model, tensor_parallel_size=tp, max_model_len=max_model_len)
        params = SamplingParams(temperature=temperature, max_tokens=max_new_tokens)
        return lambda convs: [o.outputs[0].text for o in llm.chat(convs, params, use_tqdm=False, chat_template_kwargs=tmpl)]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model, padding_side="left")
    lm = AutoModelForCausalLM.from_pretrained(model, dtype="auto", device_map="auto")

    def generate(convs):
        out = []
        for k in range(0, len(convs), batch_size):
            enc = tok.apply_chat_template(convs[k:k + batch_size], add_generation_prompt=True, padding=True,
                                          return_tensors="pt", return_dict=True, **tmpl).to(lm.device)
            with torch.no_grad():
                ids = lm.generate(**enc, max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                                  temperature=temperature or None, pad_token_id=tok.pad_token_id or tok.eos_token_id)
            out += tok.batch_decode(ids[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return out

    return generate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env-url", default="http://localhost:8000")
    p.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    p.add_argument("--backend", choices=["vllm-server", "openai", "vllm", "hf"], default="vllm-server",
                   help="vllm-server: start `vllm serve` and pipeline episodes (fastest); openai: use --base-url; "
                        "vllm/hf: in-process, turn-synchronous batches")
    p.add_argument("--base-url", default="http://127.0.0.1:8001/v1", help="OpenAI-compatible endpoint (openai backend)")
    p.add_argument("--port", type=int, default=8001, help="vllm-server port")
    p.add_argument("--max-model-len", type=int, default=65536)
    p.add_argument("--spec-mtp", type=int, default=0, help="speculative decoding with N MTP draft tokens (Qwen3.5: 2)")
    p.add_argument("--vllm-args", default="", help="extra `vllm serve` flags, e.g. '--language-model-only'")
    p.add_argument("--tp", type=int, default=1, help="vLLM tensor parallel size")
    p.add_argument("--split", default="eval")
    p.add_argument("--levels", type=int, nargs="+", default=[1])
    p.add_argument("--kernelbench-dir", default=os.environ.get("KERNELBENCH_DIR", ""))
    p.add_argument("--task-ids", nargs="*", help="override task list")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--samples", type=int, default=1, help="episodes per task (use temperature > 0) for pass@k")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=None, help="paired per-sample generation seeds for pipelined evaluation")
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--enable-thinking", action="store_true", help="Qwen3-style <think>; raise --max-new-tokens with it")
    p.add_argument("--batch-size", type=int, default=8, help="HF backend generate batch")
    p.add_argument("--out", default="baseline_results.jsonl")
    p.add_argument("--wandb", metavar="PROJECT", help="log progress, summary and an episode table to W&B")
    args = p.parse_args()

    task_ids = args.task_ids or split_task_ids(args.split, args.levels, args.kernelbench_dir)
    task_ids = task_ids[: args.limit] if args.limit else task_ids
    task_ids = [t for t in task_ids for _ in range(args.samples)]
    print(f"{len(task_ids)} {args.split} tasks, model {args.model} ({args.backend})")
    server = None
    if args.backend == "vllm-server":
        server = start_vllm_server(args.model, args.port, args.max_model_len, args.tp, args.spec_mtp, args.vllm_args)
        args.base_url = f"http://127.0.0.1:{args.port}/v1"
    if args.backend in ("vllm-server", "openai"):
        chat = make_chat(args.base_url, args.model, args.temperature, args.max_new_tokens, args.enable_thinking)
    else:
        generate = make_generate(args.backend, args.model, args.temperature, args.max_new_tokens, args.tp,
                                 args.batch_size, args.enable_thinking, args.max_model_len)
    run = None
    if args.wandb:
        import wandb

        run = wandb.init(project=args.wandb, config=vars(args), job_type="baseline")

    done, lock = [], threading.Lock()
    out = open(args.out, "w")
    t_start = time.monotonic()

    def on_done(ep):
        with lock:
            done.append(ep)
            out.write(json.dumps(ep) + "\n")
            out.flush()
            best = max((t["speedup"] for t in ep["turns"] if t["correct"]), default=None)
            print(f"[{len(done)}/{len(task_ids)} done, {time.monotonic() - t_start:.0f}s] {ep['task_id']}: "
                  f"{' > '.join(t['stage'] for t in ep['turns'])}" + (f", best {best:.2f}x" if best else "")
                  + (f", ERROR {ep['error']}" if ep.get("error") else ""), flush=True)

    def on_step(ep):
        t = ep["turns"][-1]
        tok = t.get("completion_tokens")
        rate = f", {tok / t['gen_s']:.0f} tok/s" if tok and t["gen_s"] else ""
        print(f"  {ep['task_id']} turn {len(ep['turns'])}: {t['stage']}"
              + (f" {t['speedup']:.2f}x" if t["correct"] else "") + (" [TRUNCATED]" if t.get("truncated") else "")
              + f" | gen {t['gen_s']:.0f}s ({tok} tok{rate}) | env step {t['step_s']:.0f}s", flush=True)
        if run:
            run.log({"gen_s": t["gen_s"], "step_s": t["step_s"], "completion_tokens": tok or 0,
                     "truncated": int(bool(t.get("truncated"))), "correct": int(t["correct"])})

    def on_turn(turn, stats):
        s = summarize(done) if done else {}
        print(f"[turn {turn}] {stats['live']} live, gen {stats['gen_s']:.0f}s, slowest step {stats['max_step_s']:.0f}s, "
              f"stages {stats['stages']}; done {len(done)}, correct {s.get('correct_pct', 0):.0f}%", flush=True)
        if run:
            run.log({"turn": turn, "live": stats["live"], "gen_s": stats["gen_s"], "max_step_s": stats["max_step_s"],
                     **{f"stage/{k}": v for k, v in stats["stages"].items()}})

    try:
        if args.backend in ("vllm-server", "openai"):
            seeds = [args.seed + i % args.samples for i in range(len(task_ids))] if args.seed is not None else None
            eps = run_episodes_pipelined(chat, args.env_url, task_ids, args.split, on_done=on_done, on_step=on_step,
                                        seeds=seeds)
        else:
            eps = run_episodes(generate, args.env_url, task_ids, args.split, on_done=on_done, on_turn=on_turn)
    finally:
        out.close()
        if server:
            server.terminate()
    print(f"wall time {time.monotonic() - t_start:.0f}s")
    summary = summarize(eps)
    print(json.dumps(summary, indent=2))
    if run:
        run.summary.update({k: v for k, v in summary.items() if not isinstance(v, dict)})
        cols = ["task_id", "episode_id", "stages", "best_speedup", "turns"]
        rows = [[ep["task_id"], ep.get("episode_id"), " > ".join(t["stage"] for t in ep["turns"]),
                 max((t["speedup"] for t in ep["turns"] if t["correct"]), default=None), len(ep["turns"])] for ep in eps]
        run.log({"episodes": wandb.Table(columns=cols, data=rows)})
        run.finish()


if __name__ == "__main__":
    main()
