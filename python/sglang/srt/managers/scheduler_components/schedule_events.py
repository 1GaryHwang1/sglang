"""Scheduling events and the internal handler signature."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache


class SchedulingEvent(Enum):
    """Request and batch callback contract for normal generation scheduling.

    record: the request is being received, before its first waiting-queue entry.
    in queue: the request belongs to waiting_queue, including requests that were
    not admitted in an earlier round and requests requeued after preemption.
    candidate: the request is eligible for selection in the current round.
    runnable: the request is selected for the current execution batch.
    running: the request participates in the current forward pass. This execution
    state is distinct from the Scheduler.running_batch container.
    settled: the round's results have been applied to the request.

    These are logical stages, not separate queues or exclusive request states.
    Candidate and later stages may contain both prefill and decode requests.
    Unfinished decode and ongoing chunk-prefill requests can become candidates
    for another round without joining waiting_queue or emitting ON_SORT. A
    decode candidate that is not selected remains a candidate for a later round;
    this is not preemption.

    In non-overlap mode, a round is settled before the next round is scheduled.
    In overlap mode, the Scheduler may select and submit the next round before
    applying the previous round's results on the CPU. For the same request,
    ON_BATCH and ON_COMPUTE for round N+1 may precede ON_STOP for round N.
    Callbacks still run sequentially on the Scheduler thread; they do not run
    concurrently with one another in the ordinary overlap loop.

    Req is a shared, mutable object, not a snapshot of a particular round.
    Output and completion fields reflect results already applied on the CPU;
    scheduling fields may already describe a later round. GPU-side token
    handoff can let the next forward proceed before Req.output_ids is updated.
    ON_SORT likewise uses the currently available state without waiting for
    outstanding results. A handler must not assume the previous ON_STOP has
    run when handling ON_BATCH or ON_COMPUTE. Policies that need to match data
    across rounds are responsible for retaining that data themselves.

    Emit callbacks for the transition that actually occurs, even when a shared
    helper handles multiple transitions. Every registered policy with a handler
    receives the event, including policies that are not currently active.
    ENABLE, ON_PREEMPT, and DISABLE receive one request as ``req``. ON_SORT
    receives ``reqs`` and an engine view once per ranking pass. ON_BATCH,
    ON_COMPUTE, and ON_STOP receive ``reqs`` and ``batch`` once per batch event.
    ``reqs`` contains only requests participating in this event; ON_STOP excludes
    requests whose results were discarded. Batch metadata may still describe
    other rows, so per-request updates must iterate ``reqs``, not ``batch.reqs``.
    """

    # record -> in queue: emit once after successful first entry into waiting_queue.
    # Requeueing after preemption emits ON_PREEMPT without emitting ENABLE again.
    ENABLE = "enable"

    # in queue -> candidate: emit immediately before waiting_queue is ranked so
    # handlers can update the data used by that rank. Emit once per actual
    # ranking pass with all requests being ranked. Unselected requests stay queued.
    ON_SORT = "on_sort"

    # candidate -> runnable: emit once for the selected execution batch, after
    # resource checks, preemption, and batch mixing finalize the participants.
    # Applies to prefill, every chunk-prefill round, and every decode round.
    # Under overlap, the previous round's ON_STOP may not have run yet.
    ON_BATCH = "on_batch"

    # candidate -> in queue: emit after an active request is actually preempted
    # and successfully requeued. Applies to priority and memory-pressure
    # preemption, once per occurrence. This transition does not emit ENABLE.
    ON_PREEMPT = "on_preempt"

    # runnable -> running: emit once for the execution batch after inputs are
    # prepared, immediately before submitting the forward pass for GPU execution.
    # This marks CPU submission, not the actual GPU kernel start time, and does
    # not imply that the previous round's results have been applied to Req.
    ON_COMPUTE = "on_compute"

    # running -> settled: emit once after the round's results, including output and
    # completion state, have been applied to the request and before its custom
    # scheduling data is removed. Include intermediate prefill chunks that produce
    # no output token. Do not emit for results discarded because the request was
    # already terminated or preempted. Under overlap, later rounds may already
    # have been submitted, so ON_STOP need not match the number of ON_COMPUTE
    # events and Req scheduling fields need not describe the settled round.
    ON_STOP = "on_stop"

    # Any live state -> terminal: emit once after completion, cancellation, or
    # abort is confirmed, before removing the request's custom scheduling data.
    # Normal completion emits ON_STOP for the final round before DISABLE; a
    # request terminated without a forward pass does not require ON_STOP.
    # Emit DISABLE only once. Discarded results from rounds submitted earlier
    # must not include the request in further events or access its cleaned-up data.
    DISABLE = "disable"


EVENT_INPUTS = {
    SchedulingEvent.ENABLE: ("req","engine_view"),
    SchedulingEvent.ON_SORT: ("reqs", "engine_view"),
    SchedulingEvent.ON_BATCH: ("reqs", "batch"),
    SchedulingEvent.ON_PREEMPT: ("req",),
    SchedulingEvent.ON_COMPUTE: ("reqs", "batch"),
    SchedulingEvent.ON_STOP: ("reqs", "batch"),
    SchedulingEvent.DISABLE: ("req",),
}


@dataclass(frozen=True, slots=True)
class SchedulingEngineView:
    """Read-only membership from the current planning pass; Req stays mutable."""
    tree_cache: BasePrefixCache
    enable_hierarchical_cache: bool
    enable_priority_scheduling: bool
    schedule_low_priority_values_first: bool

    running_reqs: tuple[Req, ...]


# Internal signature after registration-time argument adaptation and data binding.
EventHandler = Callable[..., None]
