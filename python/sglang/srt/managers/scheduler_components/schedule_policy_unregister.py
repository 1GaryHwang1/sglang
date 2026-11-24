"""Runtime schedule policy unregister."""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

from sglang.srt.managers.io_struct import (
    SchedulePolicyUnregisterReqInput,
    SchedulePolicyUnregisterReqOutput,
)

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)

def handle_schedule_policy_unregister(
    scheduler: Scheduler, recv_req: SchedulePolicyUnregisterReqInput
) -> SchedulePolicyUnregisterReqOutput:
    """Unregister the scheduler's policy at runtime. The instance do not need to be
    idle."""
    unregister_policy_name = recv_req.policy_name.strip().lower()

    try:
        scheduler.policy.unregister(unregister_policy_name)
    except ValueError as e:
        logger.warning(
            "Schedule policy unregistration rejected (%s): %s", unregister_policy_name, e
        )
        return SchedulePolicyUnregisterReqOutput(
            success=False,
            message=str(e),
            unregister_schedule_policy=unregister_policy_name,
        )

    sys.modules.pop(f"_sglang_schedule_policy_{unregister_policy_name}", None)

    logger.info("Unregistered schedule policy: %s", unregister_policy_name)
    return SchedulePolicyUnregisterReqOutput(
        success=True,
        message="ok",
        unregister_schedule_policy=unregister_policy_name,
    )