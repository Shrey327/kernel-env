# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kernel Env Environment Client."""

import asyncio
import os
from typing import Dict

from openenv.core import EnvClient
from websockets.asyncio.client import connect as ws_connect
from openenv.core.client_types import StepResult

from .models import KernelAction, KernelObservation, KernelState


class EnvironmentInfrastructureError(RuntimeError):
    """An evaluator failure, not a policy outcome to train against."""


class KernelEnv(EnvClient[KernelAction, KernelObservation, KernelState]):
    """
    Client for the Kernel Env Environment.

    Example:
        >>> with KernelEnv(base_url="http://localhost:8000").sync() as env:
        ...     result = env.reset(seed=0)
        ...     print(result.observation.problem)
        ...     result = env.step(KernelAction(code=my_triton_source))
        ...     print(result.observation.stage, result.reward)

    Reset kwargs: `task_id="level1/19_ReLU"` to pick a task, `split="eval"` for held-out tasks.
    """

    async def connect(self) -> "KernelEnv":
        # Modal endpoints are deployed with requires_proxy_auth; send the token if one is configured.
        key, secret = os.environ.get("MODAL_PROXY_KEY"), os.environ.get("MODAL_PROXY_SECRET")
        if self._ws is not None:
            return self
        headers = {"Modal-Key": key, "Modal-Secret": secret} if key and secret else {}
        for attempt in range(4):
            try:
                self._ws = await ws_connect(
                    self._ws_url, open_timeout=self._connect_timeout,
                    max_size=self._max_message_size, additional_headers=headers,
                )
                break
            except Exception as error:
                # Retry connection establishment only; a step may already have executed.
                if attempt == 3 or not isinstance(error, (TimeoutError, OSError)):
                    raise EnvironmentInfrastructureError(
                        f"Failed to connect after {attempt + 1} attempt(s): {error}"
                    ) from error
                await asyncio.sleep(0.25 * 2**attempt)
        return self

    def _step_payload(self, action: KernelAction) -> Dict:
        return action.model_dump(exclude={"metadata"})

    def _parse_result(self, payload: Dict) -> StepResult[KernelObservation]:
        if payload.get("observation", {}).get("stage") == "infrastructure":
            raise EnvironmentInfrastructureError(payload["observation"].get("feedback", "Evaluator failed"))
        obs = KernelObservation(
            **payload.get("observation", {}),
            done=payload.get("done", False),
            reward=payload.get("reward"),
        )
        return StepResult(observation=obs, reward=payload.get("reward"), done=obs.done)

    def _parse_state(self, payload: Dict) -> KernelState:
        return KernelState(**payload)
