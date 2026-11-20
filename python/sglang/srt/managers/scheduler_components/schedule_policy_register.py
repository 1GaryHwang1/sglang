"""Runtime schedule policy register."""


from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path
import json
import os

import logging
from typing import TYPE_CHECKING

from sglang.srt.managers.io_struct import (
    SchedulePolicyRegisterReqInput, 
    SchedulePolicyRegisterReqOutput,
)

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)

def handle_schedule_policy_register(scheduler: Scheduler, recv_req: SchedulePolicyRegisterReqInput
) -> SchedulePolicyRegisterReqOutput:
    """Register the scheduler's policy at runtime. The instance do not need to be
    idle."""
    register_policy_name = recv_req.policy_name.strip().lower()
    module_name = None

    def _fail(msg: str) -> SchedulePolicyRegisterReqOutput:
        logger.warning(
            "Schedule policy %s register rejected: %s", register_policy_name, msg
        )
        return SchedulePolicyRegisterReqOutput(
            success=False,
            message=msg,
            new_register_schedule_policy=register_policy_name
        )

    if scheduler._schedule_policy_register_in_progress:
        return _fail("another schedule policy register is already in progress")

    if not register_policy_name:
        return _fail("policy_name must not be empty")

    scheduler._schedule_policy_register_in_progress = True

    try:
        path = Path(recv_req.register_path).resolve(strict=True)
        if not path.is_file() or path.suffix != ".py":
            raise ValueError("register_path must point to a Python file")

        if register_policy_name in scheduler.policy.policy_registry:
            raise ValueError(f"Policy name already registered: {register_policy_name}")

        if not recv_req.register_path.strip():
            raise ValueError("register_path must not be empty")

        module_name = f"_sglang_schedule_policy_{register_policy_name}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Cannot load Python module: {path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        rank_fn = getattr(module, recv_req.func_name, None)
        if not callable(rank_fn):
            raise ValueError(
                f"{recv_req.func_name!r} is missing or is not callable"
            )

        scheduler.policy.register(register_policy_name, rank_fn)

        logger.info("Registered custom schedule policy: %s", register_policy_name)
        return SchedulePolicyRegisterReqOutput(
            success=True,
            message="ok",
            new_register_schedule_policy=register_policy_name,
        )
    except Exception as e:
        if module_name is not None:
            sys.modules.pop(module_name, None)

        logger.exception("Failed to register schedule policy: %s", register_policy_name)
        return SchedulePolicyRegisterReqOutput(
            success=False,
            message=str(e),
            new_register_schedule_policy=register_policy_name,
        )
    finally:
        scheduler._schedule_policy_register_in_progress = False
