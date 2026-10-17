# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Kernel Env Environment.

The agent receives a PyTorch reference `Model` and submits source code for a
`ModelNew` that computes the same thing with custom GPU kernels.
"""

from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field

Stage = Literal["ready", "format", "compile", "runtime", "timeout", "correctness", "timed"]


class KernelAction(Action):
    """Full Python source defining `ModelNew`."""

    code: str = Field(..., description="Python source defining class ModelNew")
    language: Literal["triton", "cuda"] = Field(
        default="triton", description="Kernel language used inside ModelNew"
    )


class KernelObservation(Observation):
    """Result of evaluating a submission (or the task prompt on reset)."""

    task_id: str = Field(default="", description="KernelBench task id, e.g. level1/19_ReLU")
    problem: str = Field(default="", description="Task prompt; only set on reset")
    stage: Stage = Field(
        default="ready",
        description="Furthest stage reached: format < compile < runtime/timeout < correctness < timed",
    )
    feedback: str = Field(default="", description="Truncated error / mismatch / timing summary")
    correct: Optional[bool] = Field(default=None, description="Outputs matched reference")
    max_abs_err: Optional[float] = Field(default=None, description="Worst abs error across trials")
    speedup: Optional[float] = Field(default=None, description="t_ref / t_new (eager PyTorch ref)")
    t_ref_ms: Optional[float] = Field(default=None, description="Reference runtime, median ms")
    t_new_ms: Optional[float] = Field(default=None, description="Candidate runtime, median ms")
    t_min_ms: Optional[float] = Field(default=None, description="Roofline floor: max(FLOPs/peak, bytes/bandwidth)")
    bound: Optional[Literal["memory", "compute"]] = Field(default=None, description="Which roof sets t_min_ms")
    arithmetic_intensity: Optional[float] = Field(default=None, description="Counted FLOPs / compulsory bytes")
    roofline_frac: Optional[float] = Field(default=None, description="t_min / t_new: share of hardware peak reached")
    ref_roofline_frac: Optional[float] = Field(default=None, description="t_min / t_ref for the PyTorch reference")
    turn: int = Field(default=0, description="Submissions made this episode")
    max_turns: int = Field(default=4, description="Submissions allowed this episode")


class KernelState(State):
    """Episode state."""

    task_id: str = Field(default="")
    level: int = Field(default=0)
    best_speedup: float = Field(default=0.0, description="Best speedup among correct submissions")
    history: List[Dict[str, Any]] = Field(default_factory=list, description="Per-turn stage/speedup")
