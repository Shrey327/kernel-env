"""Reward: correctness gate, then a speed score chosen by KERNEL_ENV_REWARD.

speedup (default): 0.3 + clamp(log2(t_ref / t_new), -0.2, 3.0). Correct-but-slower still earns >= 0.1;
  log scale keeps rare 100x outliers from dominating group-normalized advantages (GRPO).
roofline: 0.3 + 2.7 * min(t_min / t_new, 1), the share of the hardware roof reached. Comparable across
  memory-bound tasks (where ~1x vs PyTorch is already near optimal) and compute-bound ones. Falls back
  to the speedup score when the roofline is unknown (GPU not in the peak table).
partial credit (KERNEL_ENV_PARTIAL_CREDIT=1): incorrect submissions that compiled earn a little, so
  all-fail GRPO groups still have reward differences: 0.02 if it crashed or timed out at runtime, 0.05
  if it ran but gave wrong output. Correct kernels always earn >= 0.1, so this never outranks them.
"""

import math
import os

from openenv.core.rubrics import Gate, Rubric, Sequential


class Correct(Rubric):
    def forward(self, action, observation) -> float:
        return 1.0 if observation.correct else 0.0


class Speedup(Rubric):
    def forward(self, action, observation) -> float:
        s = observation.speedup or 0.0
        return 0.3 + min(max(math.log2(s), -0.2), 3.0) if s > 0 else 0.0


class Roofline(Rubric):
    def __init__(self):
        super().__init__()
        self.fallback = Speedup()

    def forward(self, action, observation) -> float:
        f = observation.roofline_frac
        return 0.3 + 2.7 * min(f, 1.0) if f else self.fallback(action, observation)


PARTIAL_CREDIT = {"runtime": 0.02, "timeout": 0.02, "correctness": 0.05}


class KernelReward(Rubric):
    def __init__(self, speed: Rubric, partial: bool):
        super().__init__()
        self.speed = speed
        self.partial = partial

    def forward(self, action, observation) -> float:
        if observation.correct:
            return self.speed(action, observation)
        return PARTIAL_CREDIT.get(observation.stage, 0.0) if self.partial else 0.0


def kernel_rubric(mode: str = "", partial: bool = None) -> Rubric:
    mode = mode or os.environ.get("KERNEL_ENV_REWARD", "speedup")
    if partial is None:
        partial = os.environ.get("KERNEL_ENV_PARTIAL_CREDIT") == "1"
    return KernelReward(Sequential(Gate(Correct()), {"speedup": Speedup, "roofline": Roofline}[mode]()), partial)
