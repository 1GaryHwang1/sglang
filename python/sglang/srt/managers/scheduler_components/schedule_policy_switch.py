"""Runtime schedule policy switch."""


from __future__ import annotations

import json
import os

import logging
from typing import TYPE_CHECKING

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

def record_waiting_snapshot(
    scheduler: Scheduler,
    *,
    switch_id: int,
    phase: str,
    new_schedule_policy: str,
) -> None:
    path = os.environ.get("SGLANG_SCHEDULE_SWITCH_TRACE")
    if not path:
        return

    record = {
        "switch_id": switch_id,
        "phase": phase,
        "policy": scheduler.policy.policy.value,
        "new_schedule_policy": new_schedule_policy,
        "waiting": [
            {
                "rid": req.rid,
                "max_new_tokens": req.sampling_params.max_new_tokens,
                "enqueue_time": req.time_stats.wait_queue_entry_time,
            }
            for req in scheduler.waiting_queue
        ],
    }

    try:
        with open(path, "a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        logger.exception("Failed to write scheduling snapshot")

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

    scheduler._schedule_policy_switch_sequence += 1
    switch_id = scheduler._schedule_policy_switch_sequence

    record_waiting_snapshot(
        scheduler=scheduler,
        switch_id=switch_id,
        phase="before_switch",
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

        scheduler._pending_policy_switch_snapshot = (
            switch_id,
            new_schedule_policy,
        )

        logger.info("Schedule policy switch succeeded: %s -> %s", old_schedule_policy, new_schedule_policy)
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
