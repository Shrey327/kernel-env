"""Merge a PEFT LoRA adapter into the base checkpoint's own safetensors files.

    python merge_lora.py --base Qwen/Qwen3.5-9B --adapter /runs/grpo_pilot_v2/checkpoint-10 --out /runs/merged_ckpt10

W' = W + (alpha / r) * B @ A for every adapted module, written back under the base checkpoint's tensor
names. Unlike PEFT's merge_and_unload + save_pretrained, this keeps the original config and every other
tensor (e.g. Qwen3.5's MTP draft layer), so vLLM serves the result exactly like the base model.
"""

import argparse
import json
import math
import os
import shutil

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file


def adapter_deltas(adapter_dir: str) -> dict:
    cfg = json.load(open(os.path.join(adapter_dir, "adapter_config.json")))
    if cfg.get("use_dora"):
        raise ValueError("DoRA adapters are not supported")
    scale = cfg["lora_alpha"] / (math.sqrt(cfg["r"]) if cfg.get("use_rslora") else cfg["r"])
    sd = load_file(os.path.join(adapter_dir, "adapter_model.safetensors"))
    mods = {}
    for k, v in sd.items():
        for part in (".lora_A.", ".lora_B."):
            if part in k:
                path = k.split(part)[0].removeprefix("base_model.model.")
                mods.setdefault(path, {})[part.strip(".")] = v
    return {p: scale * (ab["lora_B"].float() @ ab["lora_A"].float()) for p, ab in mods.items()}


def map_to_base(paths, base_keys) -> dict:
    """adapter module path -> base tensor name; exact match, else a unique suffix match."""
    out = {}
    for p in paths:
        key = p + ".weight"
        if key in base_keys:
            out[p] = key
            continue
        hits = [b for b in base_keys if b.endswith("." + key)]
        if len(hits) != 1:
            raise KeyError(f"cannot map adapter module {p!r} to one base tensor (candidates: {hits[:4]})")
        out[p] = hits[0]
    return out


def merge(base: str, adapter_dir: str, out_dir: str) -> int:
    snap = snapshot_download(base) if not os.path.isdir(base) else base
    index_path = os.path.join(snap, "model.safetensors.index.json")
    if os.path.exists(index_path):
        weight_map = json.load(open(index_path))["weight_map"]
    else:  # single-file checkpoint
        weight_map = {k: "model.safetensors" for k in load_file(os.path.join(snap, "model.safetensors"))}
    deltas = adapter_deltas(adapter_dir)
    target = {b: deltas[p] for p, b in map_to_base(deltas, set(weight_map)).items()}

    os.makedirs(out_dir, exist_ok=True)
    for name in os.listdir(snap):  # config, tokenizer, processor, index: copied as-is
        src = os.path.join(snap, name)
        if os.path.isfile(src) and not name.endswith(".safetensors"):
            shutil.copy(src, os.path.join(out_dir, name))
    merged = 0
    for shard in sorted(set(weight_map.values())):
        tensors = load_file(os.path.join(snap, shard))
        for k in tensors:
            if k in target:
                w = tensors[k]
                if w.shape != target[k].shape:
                    raise ValueError(f"{k}: base {tuple(w.shape)} vs delta {tuple(target[k].shape)}")
                tensors[k] = (w.float() + target[k]).to(w.dtype)
                merged += 1
        save_file(tensors, os.path.join(out_dir, shard), metadata={"format": "pt"})
    if merged != len(target):
        raise RuntimeError(f"merged {merged} of {len(target)} adapted tensors")
    return merged


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    n = merge(a.base, a.adapter, a.out)
    print(f"merged {n} tensors from {a.adapter} into {a.base} -> {a.out}")


if __name__ == "__main__":
    main()
