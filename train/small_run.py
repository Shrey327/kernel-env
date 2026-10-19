"""Bounded OpenEnv RL pilot: frozen baseline -> GRPO LoRA -> identical evaluation.

TRAIN_GPU=H100:2 modal run modal_app.py::job --max-turns 2 --reward roofline \
    --cmd 'timeout 5400 python small_run.py --out /runs/proof-20261019'
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from baseline_eval import split_task_ids, summarize

TRAIN = ["level1/19_ReLU", "level1/20_LeakyReLU", "level1/21_Sigmoid", "level1/22_Tanh"]
TRANSFER = ["level1/25_Swish", "level1/28_HardSigmoid", "level1/32_HardTanh"]
HELDOUT = ["level1/23_Softmax", "level1/24_LogSoftmax", "level1/88_MinGPTNewGelu"]


def compare(before, after, groups):
    """Compare the same tasks; never infer a training gain from within-episode refinement."""
    if sorted(e["task_id"] for e in before) != sorted(e["task_id"] for e in after):
        raise ValueError("before/after task sets differ")
    if any(e.get("error") for e in before + after):
        raise ValueError("evaluation contains infrastructure errors")
    result = {}
    for name, ids in groups.items():
        result[name] = {}
        for label, episodes in [("before", before), ("after", after)]:
            chosen = [e for e in episodes if e["task_id"] in ids]
            if sorted(e["task_id"] for e in chosen) != sorted(ids):
                raise ValueError(f"missing or repeated task in {name}/{label}")
            summary = summarize(chosen)
            # Including failures prevents a policy solving fewer easy tasks from appearing faster.
            summary["mean_best_speedup_including_failures"] = sum(
                max((t["speedup"] for t in e["turns"] if t["correct"]), default=0)
                for e in chosen
            ) / len(chosen)
            result[name][label] = summary
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    p.add_argument("--steps", type=int, default=16)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    groups = {"training_tasks": TRAIN, "reserved_transfer": TRANSFER, "official_heldout": HELDOUT}
    assert set(TRAIN + TRANSFER) <= set(split_task_ids("train", [1]))
    assert set(HELDOUT) <= set(split_task_ids("eval", [1]))
    assert not set(TRAIN) & set(TRANSFER + HELDOUT)
    manifest = {"model": a.model, "steps": a.steps, "groups": groups,
                "eval_temperature": 0.0, "max_new_tokens": 2048, "seed": 42,
                "created_at_unix": time.time(), "torch_baseline": "eager"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    task_file = out / "train_tasks.json"
    task_file.write_text(json.dumps(TRAIN))

    def run(args, name):
        print(f"PHASE {name}: {args}", flush=True)
        start = time.monotonic()
        with (out / f"{name}.log").open("w") as log:
            proc = subprocess.Popen([sys.executable, "-u", *args], stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            if proc.wait():
                raise RuntimeError(f"{name} failed; see {log.name}")
        manifest.setdefault("phase_seconds", {})[name] = time.monotonic() - start
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    def evaluate(model, label):
        run(["baseline_eval.py", "--env-url", os.environ["KERNEL_ENV_URL"], "--model", model,
             "--backend", "vllm", "--temperature", "0", "--max-new-tokens", "2048",
             "--max-model-len", "8192", "--task-ids", *TRAIN, *TRANSFER, *HELDOUT,
             "--out", str(out / f"{label}.jsonl")], label)

    evaluate(a.model, "before")
    run(["grpo_kernel.py", "--model_name_or_path", a.model, "--dtype", "bfloat16", "--bf16", "true",
         "--env_url", os.environ["KERNEL_ENV_URL"], "--task_file", str(task_file),
         "--use_peft", "--lora_r", "16", "--lora_alpha", "32", "--lora_target_modules", "all-linear",
         "--use_vllm", "--vllm_mode", "colocate", "--vllm_gpu_memory_utilization", "0.25",
         "--vllm_max_model_length", "8192", "--num_generations", "4",
         "--per_device_train_batch_size", "4", "--gradient_accumulation_steps", "1",
         "--max_completion_length", "2048", "--max_steps", str(a.steps), "--learning_rate", "5e-5",
         "--gradient_checkpointing", "true", "--max_grad_norm", "0.05", "--loss_type", "dr_grpo",
         "--return_mode", "discounted", "--gamma", "0.4", "--temperature", "0.8",
         "--seed", "42", "--logging_steps", "1", "--report_to", "none", "--save_steps", "8",
         "--save_total_limit", "2", "--output_dir", str(out / "adapter")], "train")
    run(["merge_lora.py", "--base", a.model, "--adapter", str(out / "adapter"),
         "--out", str(out / "merged")], "merge")
    evaluate(str(out / "merged"), "after")
    episodes = [[json.loads(l) for l in (out / f"{label}.jsonl").read_text().splitlines()]
                for label in ("before", "after")]
    result = compare(*episodes, groups)
    (out / "comparison.json").write_text(json.dumps(result, indent=2))
    print("COMPARISON", json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
