"""Check terminal hook sites without loading a model or allocating GPU KV.

Run inside the development container:
    python test/manual/schedule_policy_switch/verify_custom_policy_hook_abort_paths.py

Uses real Scheduler methods, Req objects, registration and hook binding. Only
cache/IPC services and decode capacity/retraction results are substituted. This
checks terminal-branch control flow; the HTTP suite checks real GPU execution.
"""

from __future__ import annotations

import json
from array import array
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import verify_custom_policy_hooks as hooks
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_batch import FINISH_ABORT, Req
from sglang.srt.managers.schedule_policy import SchedulePolicy
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.schedule_events import SchedulingEvent
from sglang.srt.sampling.sampling_params import SamplingParams


def make_scheduler():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.policy = SchedulePolicy.__new__(SchedulePolicy)
    scheduler.policy.policy_registry = {}
    scheduler.policy.event_handlers = {event: {} for event in SchedulingEvent}
    for name in hooks.POLICIES:
        scheduler.policy.register(
            name,
            hooks.rank,
            initialize_policy_fn=hooks.initialize_policy,
            hooks={
                SchedulingEvent.ENABLE: hooks.enable,
                SchedulingEvent.DISABLE: hooks.disable,
            },
        )
        scheduler.policy.policy_registry[name].policy_data.label = name
    scheduler.policy.policy = scheduler.policy.policy_registry[hooks.POLICIES[0]]
    scheduler.waiting_queue = []
    scheduler.disaggregation_mode = DisaggregationMode.NULL
    scheduler.tree_cache = SimpleNamespace(
        finish=lambda *args: None,
        req_to_token_pool=SimpleNamespace(mamba_allocator=None),
    )
    scheduler.enable_priority_scheduling = False
    scheduler.abort_on_priority_when_disabled = True
    scheduler.schedule_low_priority_values_first = True
    scheduler.max_queued_requests = 1
    scheduler.processed_tokens_counter = 0
    scheduler._prefetch_kvcache = lambda req: None
    scheduler.sent = []
    scheduler.ipc_channels = SimpleNamespace(
        send_to_tokenizer=SimpleNamespace(
            send_output=lambda output, req: scheduler.sent.append(output)
        )
    )
    scheduler.beam_coordinator = SimpleNamespace(retire_group=lambda req: None)
    return scheduler


def make_req(rid, priority=None):
    return Req(
        rid,
        "hello",
        array("q", [42, 43]),
        SamplingParams(max_new_tokens=8),
        priority=priority,
        disagg_mode=DisaggregationMode.NULL,
    )


def admit(scheduler, req):
    assert scheduler._add_request_to_queue(req)
    scheduler.policy.trigger_event(SchedulingEvent.ENABLE, req)


def check_terminal(scheduler, req):
    assert req.finished() and isinstance(req.finished_reason, FINISH_ABORT)
    for entry in scheduler.policy.policy_registry.values():
        metrics = entry.policy_data
        assert metrics.counters["enable"] == metrics.counters["disable"] == 1
        assert metrics.terminal_requests == metrics.initializations == 1
        assert not metrics.live_rids and not entry.request_data
    records = [
        json.loads(line) for line in hooks.SNAPSHOT_FILE.read_text().splitlines()
    ]
    for policy in hooks.POLICIES:
        trace = [
            item
            for item in records
            if item["policy"] == policy and item["rid"] == req.rid
        ]
        assert [item["event"] for item in trace] == ["enable", "disable"]
        assert trace[-1]["req"]["finished"]
        assert req.rid in trace[-1]["request_data_rids"]
        assert trace[-1]["metrics"]["remaining_tokens"] == 8


def run_checks():
    scheduler = make_scheduler()
    assert not scheduler._add_request_to_queue(
        make_req("priority-rejected", priority=1)
    )
    scheduler.max_queued_requests = 0
    assert not scheduler._add_request_to_queue(make_req("queue-rejected"))
    assert all(
        not any(entry.policy_data.counters.values())
        for entry in scheduler.policy.policy_registry.values()
    )
    print("PASS：首次入队拒绝，不触发 ENABLE/DISABLE，也不建立请求数据")

    scheduler = make_scheduler()
    req = make_req("requeue-rejected")
    admit(scheduler, req)
    scheduler.waiting_queue.clear()
    scheduler.max_queued_requests = 0
    assert not scheduler._add_request_to_queue(req, is_retracted=True)
    check_terminal(scheduler, req)
    print("PASS：回队失败，终止状态→DISABLE→清理；两个策略指标均正确")

    scheduler = make_scheduler()
    req = make_req("already-aborted")
    admit(scheduler, req)
    reason = FINISH_ABORT("original reason", 503)
    req.finished_reason = reason
    scheduler._release_aborted_request(req)
    check_terminal(scheduler, req)
    assert req.finished_reason is reason
    print("PASS：已有终止原因保留，DISABLE 仍能访问请求数据")

    scheduler = make_scheduler()
    req = make_req("decode-oom")
    admit(scheduler, req)
    scheduler.waiting_queue.clear()
    reason = FINISH_ABORT("Out of memory", 500)
    req.to_finish = reason

    class OOMBatch:
        reqs = [req]
        batch_is_full = True

        def batch_size(self):
            return len(self.reqs)

        def filter_batch(self):
            self.reqs = [item for item in self.reqs if not item.finished()]

        def is_empty(self):
            return not self.reqs

        def check_decode_mem(self):
            return False

        def retract_decode(self):
            req.reset_for_retract()
            self.reqs = []
            return [], 0.5, [req]

    scheduler.decode_offload_manager = None
    scheduler.token_to_kv_pool_allocator = SimpleNamespace(available_size=lambda: 256)
    scheduler.new_token_ratio_tracker = SimpleNamespace(current=0.1)
    scheduler.metrics_reporter = SimpleNamespace(enable_metrics=False)
    batch = OOMBatch()
    scheduler.update_running_batch(batch)
    check_terminal(scheduler, req)
    assert req.finished_reason is reason and req.to_finish is None
    assert req.retraction_count == 1
    assert scheduler.sent[0].finished_reason == reason.to_json()
    scheduler.update_running_batch(batch)
    check_terminal(scheduler, req)
    print("PASS：decode OOM 直接终止，DISABLE 恰好一次，保留原因，不触发 ON_PREEMPT")


def main():
    with TemporaryDirectory(prefix="sglang-hook-aborts-") as directory:
        hooks.SNAPSHOT_FILE = Path(directory) / "events.jsonl"
        # Abort IPC attaches a serving weight version, unrelated to these hooks.
        with patch(
            "sglang.srt.managers.scheduler.get_serving",
            return_value=SimpleNamespace(weight_version="test"),
        ):
            run_checks()
    print("PASS：终止出口定向检查全部通过（没有加载模型）")


if __name__ == "__main__":
    main()
