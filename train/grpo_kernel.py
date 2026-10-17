"""GRPO on kernel_env: TRL GRPOTrainer + multi-turn rollout_func over the KernelEnv client.

    accelerate launch grpo_kernel.py --model_name_or_path Qwen/Qwen2.5-Coder-7B-Instruct \
        --env_url http://EVAL_BOX:8000 --num_generations 8 --use_vllm --output_dir runs/grpo

All GRPOConfig / ModelConfig fields are CLI flags (TrlParser). Dataset rows are KernelBench task
ids; rollout_func resets one env session per row (TRL already repeats each row `num_generations`
times = one GRPO group per task), then alternates model turn -> env.step -> feedback turn until the
env ends the episode. The whole episode is one completion: model tokens train (env_mask=1), feedback
tokens don't (env_mask=0). Reward = discounted sum of per-turn env rewards, or best turn.
Held-out eval = TRL's own eval loop over eval-split task ids (`--eval_strategy steps --eval_steps N`).
"""

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from datasets import Dataset
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOConfig, GRPOTrainer, ModelConfig, TrlParser, get_peft_config

from baseline_eval import extract_code, feedback_message, split_task_ids, summarize
from kernel_env import KernelAction, KernelEnv

log = logging.getLogger("grpo_kernel")


@dataclass
class KernelArgs:
    env_url: str = field(default="http://localhost:8000", metadata={"help": "kernel_env server (on eval GPUs)"})
    levels: list[int] = field(default_factory=lambda: [1], metadata={"help": "KernelBench levels to use"})
    kernelbench_dir: str = field(default=os.environ.get("KERNELBENCH_DIR", ""), metadata={"help": "local copy, for task ids"})
    return_mode: str = field(default="discounted", metadata={"help": "discounted | best"})
    enable_thinking: bool = field(default=False, metadata={"help": "Qwen3-style <think> in the chat template (baseline: off)"})
    gamma: float = field(default=0.9, metadata={"help": "discount over turns (return_mode=discounted)"})
    env_timeout_s: float = field(default=900.0, metadata={"help": "per env message; covers GPU-pool queueing"})
    max_eval_tasks: int = field(default=0, metadata={"help": "cap held-out tasks per eval (0 = all)"})
    task_file: str = field(default="", metadata={"help": "JSON list of training task ids (curriculum); default: whole train split"})


def episode_return(rewards: list, mode: str, gamma: float) -> float:
    if mode == "best":
        return max(rewards, default=0.0)
    return sum(gamma**t * r for t, r in enumerate(rewards))


def text_tokenizer(processing_class):
    """VLM checkpoints (e.g. Qwen3.5) load a Processor: its chat template returns nested lists and it has no
    eos_token_id. Rollouts are text-only, so use the wrapped tokenizer."""
    return getattr(processing_class, "tokenizer", processing_class)


def feedback_ids(tok, text: str, enable_thinking: bool = False) -> list:
    """Tokens after an assistant turn's EOS: the feedback user turn + next assistant header.

    Cut after the assistant turn's EOS (second-to-last EOS; a default system prompt may add one in front)
    rather than diffing two renders: templates such as
    Qwen3.5's re-render earlier assistant turns differently (they drop the empty think block), so the
    conversation is not prefix-stable. The rollout stitches the model's own tokens, so only this suffix matters.
    """
    msgs = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}, {"role": "user", "content": text}]
    full = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=False,
                                   enable_thinking=enable_thinking)  # templates without the switch ignore it
    eos = [i for i, t in enumerate(full) if t == tok.eos_token_id]
    return full[eos[-2] + 1:]


def make_rollout_func(args: KernelArgs, suspicious_path: str):
    def rollout_func(prompts: list, trainer) -> dict:
        tok = text_tokenizer(trainer.processing_class)
        mode = "train" if trainer.model.training else "eval"
        split = "train" if mode == "train" else "eval"
        envs = [KernelEnv(base_url=args.env_url, message_timeout_s=args.env_timeout_s).sync() for _ in prompts]
        pool = ThreadPoolExecutor(max_workers=len(envs))
        try:
            obs = list(pool.map(lambda e, t: e.reset(task_id=t, split=split).observation, envs, prompts))
            prompt_ids = [tok.apply_chat_template([{"role": "user", "content": o.problem}], add_generation_prompt=True,
                                                  tokenize=True, return_dict=False, enable_thinking=args.enable_thinking)
                          for o in obs]
            n = len(prompts)
            comp, mask, lps = [[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)]
            eps = [{"task_id": t, "turns": [], "codes": [], "completions": [], "step": trainer.state.global_step,
                    "mode": mode} for t in prompts]
            have_lp, live = True, list(range(n))
            while live:
                new_ids, new_lps = trainer._generate_single_turn([prompt_ids[i] + comp[i] for i in live], None, {}, 1)
                have_lp &= new_lps is not None  # HF generate gives none; vLLM does
                texts = tok.batch_decode(new_ids, skip_special_tokens=True)
                codes = [extract_code(t) for t in texts]
                results = list(pool.map(lambda i, c: envs[i].step(KernelAction(code=c)), live, codes))
                still = []
                for j, (i, r) in enumerate(zip(live, results)):
                    ids, o = list(new_ids[j]), r.observation
                    comp[i] += ids
                    mask[i] += [1] * len(ids)
                    lps[i] += list(new_lps[j]) if have_lp else []
                    truncated = not ids or ids[-1] != tok.eos_token_id
                    eps[i]["codes"].append(codes[j])
                    eps[i]["completions"].append(texts[j])
                    eps[i]["turns"].append({"stage": o.stage, "correct": bool(o.correct),
                                            "speedup": o.speedup or 0.0, "reward": r.reward or 0.0,
                                            "roofline_frac": o.roofline_frac, "ref_roofline_frac": o.ref_roofline_frac,
                                            "t_ref_ms": o.t_ref_ms, "t_new_ms": o.t_new_ms, "t_min_ms": o.t_min_ms,
                                            "bound": o.bound, "feedback": o.feedback, "truncated": truncated,
                                            "completion_tokens": len(ids)})
                    if r.done:
                        continue
                    # close a truncated turn ourselves so the template stays well-formed
                    extra = ([] if not truncated else [tok.eos_token_id]) + feedback_ids(tok, feedback_message(o, truncated), args.enable_thinking)
                    comp[i] += extra
                    mask[i] += [0] * len(extra)
                    lps[i] += [0.0] * len(extra) if have_lp else []
                    still.append(i)
                live = still
            for ep, env in zip(eps, envs):
                st = env.state()
                ep.update(episode_id=st.episode_id, history=st.history)
            # every episode, per rank, for report.py and for reading what the policy is learning
            rank = getattr(getattr(trainer, "accelerator", None), "process_index", 0)
            with open(os.path.join(os.path.dirname(suspicious_path), f"rollouts-rank{rank}.jsonl"), "a") as f:
                for ep in eps:
                    f.write(json.dumps(ep) + "\n")
        finally:
            for e in envs:
                e.close()
            pool.shutdown()

        returns = [episode_return([t["reward"] for t in ep["turns"]], args.return_mode, args.gamma) for ep in eps]
        log_episodes(trainer, mode, eps, suspicious_path)
        return {"prompt_ids": prompt_ids, "completion_ids": comp, "logprobs": lps if have_lp else None,
                "env_mask": mask, "episode_return": returns}

    return rollout_func


def log_episodes(trainer, mode: str, eps: list, suspicious_path: str):
    s = summarize(eps)
    for k in ("compile_pct", "correct_pct", "correct_turn1_pct", "faster_than_ref_pct", "mean_speedup",
              "median_roofline_frac", "suspicious"):
        if s[k] is not None:
            trainer._metrics[mode][f"kernel/{k}"].append(float(s[k]))
    trainer._metrics[mode]["kernel/turns"].append(sum(len(ep["turns"]) for ep in eps) / max(len(eps), 1))
    # env flags speedup > 50x as possible reward hack: keep the code for manual review
    for ep in eps:
        for h in ep["history"]:
            if h.get("suspicious") or h.get("roofline_violation"):
                log.warning("suspicious %.1fx speedup on %s turn %d", h["speedup"], ep["task_id"], h["turn"])
                with open(suspicious_path, "a") as f:
                    f.write(json.dumps({"step": trainer.state.global_step, "mode": mode, "task_id": ep["task_id"],
                                        **h, "code": ep["codes"][h["turn"] - 1]}) + "\n")


def env_return(prompts, completions, episode_return, **kwargs):
    return episode_return


def main():
    args, cfg, model_args = TrlParser((KernelArgs, GRPOConfig, ModelConfig)).parse_args_and_config()
    cfg.model_init_kwargs = {**(cfg.model_init_kwargs or {}), "dtype": model_args.dtype or "auto"}
    train_ids = split_task_ids("train", args.levels, args.kernelbench_dir)
    eval_ids = split_task_ids("eval", args.levels, args.kernelbench_dir)
    if args.task_file:
        with open(args.task_file) as f:
            chosen = json.load(f)
        leaked = set(chosen) - set(train_ids)  # never train on held-out (or unknown) tasks
        if leaked:
            raise ValueError(f"task_file has tasks outside the train split: {sorted(leaked)[:5]}")
        train_ids = chosen
    eval_ids = eval_ids[: args.max_eval_tasks] if args.max_eval_tasks else eval_ids
    log.info("%d train / %d eval tasks", len(train_ids), len(eval_ids))
    os.makedirs(cfg.output_dir, exist_ok=True)

    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=env_return,
        args=cfg,
        train_dataset=Dataset.from_dict({"prompt": train_ids}),
        eval_dataset=Dataset.from_dict({"prompt": eval_ids}) if cfg.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        rollout_func=make_rollout_func(args, os.path.join(cfg.output_dir, "suspicious.jsonl")),
    )
    # resume after a crash or a Modal retry: newest checkpoint-* in output_dir, if any
    last = get_last_checkpoint(cfg.output_dir)
    if last:
        log.info("resuming from %s", last)
    trainer.train(resume_from_checkpoint=last)
    trainer.save_model(cfg.output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
