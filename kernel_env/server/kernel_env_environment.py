# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Kernel Env Environment Implementation.

reset() picks a KernelBench task and returns the prompt; each step() evaluates one ModelNew
submission (static checks -> compile -> correctness on hidden seeds -> timing) and returns
feedback the agent can use to refine. Episode ends after `max_turns` submissions.
"""

import asyncio
import hashlib
import json
import logging
import os
import random
import threading
import time
from typing import Any, Dict, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import KernelAction, KernelObservation, KernelState
    from .evaluator import Evaluator, default_evaluator
    from .rubrics import kernel_rubric
    from .tasks import Task, build_prompt, load_tasks
except ImportError:
    from models import KernelAction, KernelObservation, KernelState
    from server.evaluator import Evaluator, default_evaluator
    from server.rubrics import kernel_rubric
    from server.tasks import Task, build_prompt, load_tasks

log = logging.getLogger(__name__)

N_TRIALS = 5
SUSPICIOUS_SPEEDUP = 50.0  # flagged in state.history for manual review of possible reward hacks
RESULT_FIELDS = ("stage", "feedback", "correct", "max_abs_err", "speedup", "t_ref_ms", "t_new_ms",
                 "t_min_ms", "bound", "arithmetic_intensity", "roofline_frac", "ref_roofline_frac",
                 "timing_samples_ms", "ref_timing_samples_ms", "speedup_lower_bound", "speedup_upper_bound", "generated_cuda_fraction", "profiling_summary", "timing_method", "measurement_scope", "graph_repeats", "timing_calibration", "replay_profiling_summary", "warmup_evidence")
_log_lock = threading.Lock()


def emit(event: str, **fields: Any) -> None:
    """One JSON line per event on stdout (Modal / docker logs); also appended to $KERNEL_ENV_EVAL_LOG if set."""
    line = json.dumps({"event": event, "ts": round(time.time(), 3), **fields}, default=str)
    print(line, flush=True)
    path = os.environ.get("KERNEL_ENV_EVAL_LOG")
    if path:
        with _log_lock, open(path, "a") as f:
            f.write(line + "\n")


class KernelEnvironment(Environment[KernelAction, KernelObservation, KernelState]):
    """Multi-turn GPU kernel optimization over KernelBench tasks."""

    # Episode state is per instance; GPU access is serialized by the shared Evaluator.
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(
        self,
        evaluator: Optional[Evaluator] = None,
        tasks: Optional[Dict[str, Task]] = None,
        max_turns: Optional[int] = None,
    ):
        reward_mode = os.environ.get("KERNEL_ENV_REWARD", "speedup")
        super().__init__(rubric=kernel_rubric(reward_mode))
        self._reward_mode = reward_mode
        self._evaluator = evaluator or default_evaluator()
        self._tasks = tasks if tasks is not None else load_tasks()
        self.max_turns = max_turns or int(os.environ.get("KERNEL_ENV_MAX_TURNS", "4"))
        self._state = KernelState(episode_id=str(uuid4()))
        self._task: Optional[Task] = None
        self._rng = random.Random()

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        split: str = "train",
        language: str = "triton",
        **kwargs: Any,
    ) -> KernelObservation:
        """Start an episode on `task_id`, or a seeded random task from `split` ("train"/"eval")."""
        self._reset_rubric()
        self._rng = random.Random(seed)
        if task_id is None:
            pool = sorted(t for t, task in self._tasks.items() if task.split == split)
            if not pool:
                raise ValueError(f"no tasks in split {split!r}")
            task_id = self._rng.choice(pool)
        if task_id not in self._tasks:
            raise ValueError(f"unknown task_id {task_id!r}")
        self._task = self._tasks[task_id]
        self._state = KernelState(
            episode_id=episode_id or str(uuid4()), task_id=task_id, level=self._task.level
        )
        emit("reset", episode_id=self._state.episode_id, task_id=task_id, split=split, seed=seed)
        return KernelObservation(
            task_id=task_id,
            problem=build_prompt(self._task, language, self.max_turns),
            max_turns=self.max_turns,
            reward_mode=self._reward_mode,
            done=False,
            reward=0.0,
        )

    def step(self, action: KernelAction, timeout_s: Optional[float] = None, **kwargs: Any) -> KernelObservation:  # type: ignore[override]
        if self._task is None:
            raise RuntimeError("call reset() before step()")
        if self._state.step_count >= self.max_turns:
            raise RuntimeError("episode is done; call reset()")
        self._state.step_count += 1

        # hidden per-step input seeds: outputs can't be memorized across turns
        seeds = [self._rng.randrange(2**31) for _ in range(N_TRIALS)]
        res = self._evaluator.evaluate(self._task.src, action.code, action.language, seeds, task_id=self._task.task_id)

        obs = KernelObservation(
            task_id=self._task.task_id,
            turn=self._state.step_count,
            max_turns=self.max_turns,
            done=self._state.step_count >= self.max_turns,
            reward_mode=self._reward_mode,
            **{k: res[k] for k in RESULT_FIELDS if res.get(k) is not None},
        )
        if self._reward_mode=='calibrated_speedup_v2' and obs.correct:
            from .cuda_graph_timing import METHOD
            if obs.timing_method!=METHOD:raise RuntimeError('corrected reward requires exact warmed graph timing protocol')
        obs.reward = self._apply_rubric(action, obs)
        if self._reward_mode=='calibrated_speedup_v2':
            from .calibrated_reward import SPEC_VERSION,CORRECTNESS
            obs.reward_spec=SPEC_VERSION
            obs.correctness_reward=CORRECTNESS if obs.correct else 0.
            obs.performance_reward=float(obs.reward)-obs.correctness_reward
            obs.conservative_speedup=obs.speedup_lower_bound if obs.correct else None
            obs.rewarded_speedup=min(4.,max(1.,obs.speedup_lower_bound)) if obs.correct else None

        entry = {
            "turn": obs.turn, "stage": obs.stage, "speedup": obs.speedup, "reward": obs.reward,
            "roofline_frac": obs.roofline_frac, "roofline_violation": bool(res.get("roofline_violation")),
            "gpu_id": res.get("gpu_id"), "queue_wait_s": res.get("queue_wait_s"), "eval_s": res.get("eval_s"),
            "code_sha": hashlib.sha1(action.code.encode()).hexdigest()[:12],
            "phases": res.get("phases"),  # worker seconds per phase: inputs, JIT, correctness, bench, lock wait
            "timing_samples_ms": obs.timing_samples_ms, "timing_method": obs.timing_method,
            "measurement_scope":obs.measurement_scope,"timing_calibration":obs.timing_calibration,
            "reward_spec":obs.reward_spec,"correctness_reward":obs.correctness_reward,
            "performance_reward":obs.performance_reward,"conservative_speedup":obs.conservative_speedup,"rewarded_speedup":obs.rewarded_speedup,
        }
        if obs.correct and (obs.speedup or 0) > SUSPICIOUS_SPEEDUP:
            entry["suspicious"] = True
            log.warning("suspicious speedup %.1fx on %s (episode %s)", obs.speedup, obs.task_id, self._state.episode_id)
        self._state.history.append(entry)
        emit(
            "step", episode_id=self._state.episode_id, task_id=obs.task_id, **entry,
            t_ref_ms=obs.t_ref_ms, t_new_ms=obs.t_new_ms, t_min_ms=obs.t_min_ms, bound=obs.bound,
            ref_roofline_frac=obs.ref_roofline_frac, max_abs_err=obs.max_abs_err,
            gpu=res.get("gpu"), feedback_head=obs.feedback[:200],
        )
        if obs.correct:
            self._state.best_speedup = max(self._state.best_speedup, obs.speedup or 0.0)
        return obs

    async def step_async(self, action: KernelAction, timeout_s: Optional[float] = None, **kwargs: Any) -> KernelObservation:  # type: ignore[override]
        # evaluation blocks for seconds; keep the server event loop free for other sessions
        return await asyncio.to_thread(self.step, action, timeout_s, **kwargs)

    @property
    def state(self) -> KernelState:
        return self._state
