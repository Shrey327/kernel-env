"""Pick GRPO training tasks from a probe run (baseline_eval.py --split train --samples N).

    python make_curriculum.py probe_train_l1.jsonl -o curriculum_l1.json

GRPO learns only from groups whose rewards differ. A task the policy never gets past compiling gives an
all-zero group, so keep tasks where some sample was correct, or at least ran end to end (wrong output
still earns partial credit, so its group has reward differences).
"""

import argparse
import json
from collections import defaultdict

from baseline_eval import STAGE_ORDER

KEEP_FROM = STAGE_ORDER.index("correctness")  # ran to completion; "timed" = correct


def choose(episodes: list) -> tuple:
    best = defaultdict(int)  # task -> furthest stage index over all samples and turns
    for ep in episodes:
        for t in ep["turns"]:
            best[ep["task_id"]] = max(best[ep["task_id"]], STAGE_ORDER.index(t["stage"]))
    keep = sorted(t for t, s in best.items() if s >= KEEP_FROM)
    solved = sorted(t for t, s in best.items() if s == STAGE_ORDER.index("timed"))
    return keep, solved, len(best)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("probe")
    p.add_argument("-o", "--out", default="curriculum.json")
    args = p.parse_args()
    eps = [json.loads(line) for line in open(args.probe) if line.strip()]
    keep, solved, total = choose(eps)
    with open(args.out, "w") as f:
        json.dump(keep, f, indent=1)
    print(f"{len(keep)}/{total} tasks kept ({len(solved)} solved at least once) -> {args.out}")


if __name__ == "__main__":
    main()
