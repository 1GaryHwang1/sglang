"""Runtime schedule policy switch."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Optional, Tuple

from sglang.srt.managers.io_struct import (
    SchedulePolicySwitchReqInput, 
    SchedulePolicySwitchReqOutput,
)
from sglang.srt.managers.schedule_policy import (
    CacheAgnosticPolicy, 
    CacheAwarePolicy, 
    SchedulePolicy
)
from sglang.srt.runtime_context import get_context

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)

def handle_schedule_policy_switch(scheduler: Scheduler, recv_req: SchedulePolicySwitchReqInput
) -> SchedulePolicySwitchReqOutput:
    """Switch the scheduler's policy at runtime. The instance do not need to be
    idle."""
    old_schedule_policy = scheduler.policy.policy.value
    new_schedule_policy = (recv_req.new_schedule_policy or "").lower()

    def _fail(msg: str) -> SchedulePolicySwitchReqOutput:
        logger.warning(
            "Schedule policy switch rejected (%s -> %s): %s", old_schedule_policy, new_schedule_policy, msg
        )
        return SchedulePolicySwitchReqOutput(
            success=False,
            message=msg,
            old_schedule_policy=old_schedule_policy,
            new_schedule_policy=new_schedule_policy,
        )
    
    if new_schedule_policy == old_schedule_policy:
        return SchedulePolicySwitchReqOutput(
            success=True,
            message="already in target policy",
            old_schedule_policy=old_schedule_policy,
            new_schedule_policy=new_schedule_policy,
        )

    scheduler._schedule_policy_switch_in_progress = True

    try:
        new_policy = SchedulePolicy(
            policy=new_schedule_policy,
            tree_cache=scheduler.tree_cache,
            enable_hierarchical_cache=scheduler.enable_hierarchical_cache,
            enable_priority_scheduling=scheduler.enable_priority_scheduling,
            schedule_low_priority_values_first=scheduler.schedule_low_priority_values_first,
        )
        scheduler.policy = new_policy 
        scheduler.schedule_policy = new_schedule_policy
        get_context().override("scheduler.schedule_policy_switch", schedule_policy=new_schedule_policy)
        return SchedulePolicySwitchReqOutput(
            success=True, 
            message="ok", 
            old_schedule_policy=old_schedule_policy, 
            new_schedule_policy=new_schedule_policy
        )
    except Exception as e:
        logger.critical(
                "Schedule policy switch (%s -> %s) failed; "
                "%s",
                old_schedule_policy,
                new_schedule_policy,
                e,
            )
        return _fail(f"schedule policy switch failed; restart required: {e}")
    finally:
        scheduler._schedule_policy_switch_in_progress = False
