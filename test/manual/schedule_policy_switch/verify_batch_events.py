"""Check batch event dispatch through real Scheduler methods without a model.

Run inside the development container:
    python test/manual/schedule_policy_switch/verify_batch_events.py

Model/cache services are substituted. These checks cover CPU control flow and
argument binding, including the ordinary overlap submission branch.
"""

from __future__ import annotations

import unittest
from array import array
from contextlib import nullcontext
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_batch import FINISH_LENGTH, Req
from sglang.srt.managers.schedule_policy import SchedulePolicy
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.schedule_events import SchedulingEvent
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


def noop(*args, **kwargs):
    pass


def rank(waiting_queue):
    pass


def enable(req, request_data):
    request_data[req.rid] = {}


def on_sort(reqs, engine_view, policy_data):
    policy_data.events.append(("sort", reqs, engine_view))


def on_batch(reqs, batch, policy_data):
    policy_data.events.append(("batch", reqs, batch))


def on_compute(*, batch, policy_data):
    # A batch-only handler must work without declaring reqs.
    policy_data.work += batch.extend_num_tokens or 0
    policy_data.events.append(("compute", batch))


def on_stop(reqs, batch, request_data, policy_data):
    assert all(req.rid in request_data for req in reqs)
    policy_data.events.append(("stop", reqs, batch))


def disable(req, request_data, policy_data):
    assert req.finished() and req.rid in request_data
    policy_data.events.append(("disable", req))


def make_req(rid):
    return Req(rid, "test", array("q", [42, 43]), SamplingParams(max_new_tokens=8))


def make_scheduler():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.scheduler_stage_metrics = None
    scheduler.tree_cache = SimpleNamespace()
    scheduler.enable_hierarchical_cache = False
    scheduler.enable_priority_scheduling = False
    scheduler.schedule_low_priority_values_first = False
    scheduler.running_batch = SimpleNamespace(reqs=[])
    scheduler.policy = SchedulePolicy.__new__(SchedulePolicy)
    scheduler.policy.engine_view = scheduler.get_scheduling_engine_view(
        scheduler.running_batch
    )
    scheduler.policy.policy_registry = {}
    for name in ("active", "observer"):
        scheduler.policy.register(
            name,
            rank,
            initialize_policy_fn=lambda: SimpleNamespace(work=0, events=[]),
            hooks={
                SchedulingEvent.ENABLE: enable,
                SchedulingEvent.ON_SORT: on_sort,
                SchedulingEvent.ON_BATCH: on_batch,
                SchedulingEvent.ON_COMPUTE: on_compute,
                SchedulingEvent.ON_STOP: on_stop,
                SchedulingEvent.DISABLE: disable,
            },
        )
    scheduler.policy.policy = scheduler.policy.policy_registry["active"]
    scheduler.enable_fpm = False
    scheduler.processed_tokens_counter = 0
    return scheduler


class RankReached(Exception):
    pass


class ForwardReached(Exception):
    pass


class BatchEventTests(unittest.TestCase):
    def test_sort_uses_current_planning_batch(self):
        scheduler = make_scheduler()
        waiting = [make_req("a"), make_req("b")]
        running = make_req("current-running")
        scheduler.waiting_queue = waiting
        scheduler.running_batch = SimpleNamespace(reqs=[make_req("stale-running")])
        planning_batch = SimpleNamespace(reqs=[running], batch_is_full=False)
        scheduler.grammar_manager = SimpleNamespace(has_waiting_grammars=lambda: False)
        scheduler.enable_priority_preemption = False
        scheduler.is_hybrid_swa = False
        scheduler.chunked_req = None
        scheduler.min_free_slots_delayer = None
        scheduler.get_num_allocatable_reqs = lambda *args, **kwargs: 4

        ranked_views = []

        def stop_at_rank(waiting_queue, engine_view):
            ranked_views.append(engine_view)
            raise RankReached()

        scheduler.policy.register("stop_at_rank", stop_at_rank)
        scheduler.policy.policy = scheduler.policy.policy_registry["stop_at_rank"]
        with self.assertRaises(RankReached):
            scheduler._get_new_batch_prefill_raw(None, planning_batch)

        for name in ("active", "observer"):
            events = scheduler.policy.policy_registry[name].policy_data.events
            self.assertEqual(len(events), 1)
            event, reqs, engine_view = events[0]
            self.assertEqual(
                (event, reqs, engine_view.running_reqs),
                ("sort", tuple(waiting), (running,)),
            )
            self.assertIs(engine_view, ranked_views[0])
            self.assertIs(engine_view.tree_cache, scheduler.tree_cache)
            with self.assertRaises(FrozenInstanceError):
                engine_view.running_reqs = ()

    def test_final_batch_is_reported_once(self):
        scheduler = make_scheduler()
        reqs = [make_req("a"), make_req("b")]
        batch = SimpleNamespace(reqs=reqs)
        running = SimpleNamespace(reqs=[], is_prefill_only=False)
        scheduler.process_pending_chunked_abort = noop
        scheduler._process_hicache_events = noop
        scheduler.dllm_config = None
        scheduler.chunked_req = None
        scheduler.enable_hisparse = False
        scheduler._should_defer_prefill = lambda: False
        scheduler.get_new_batch_prefill = lambda running_batch: SimpleNamespace(
            batch_to_run=batch, running_batch=running_batch
        )
        scheduler.require_mlp_sync = False
        scheduler.dp_attn_adapter = SimpleNamespace(
            maybe_prepare_mlp_sync_batch=lambda batch, **kwargs: batch,
            maybe_convert_decode_to_extend=lambda batch: batch,
        )
        scheduler._arm_prefill_decode_interval = noop
        scheduler.ngram_embedding_manager = SimpleNamespace(
            prepare_for_forward=lambda batch, **kwargs: batch
        )
        with patch("sglang.srt.managers.scheduler.set_schedule_time_batch"):
            plan = scheduler.get_next_batch_to_run(running, None)
        self.assertIs(plan.batch_to_run, batch)
        for entry in scheduler.policy.policy_registry.values():
            self.assertEqual(entry.policy_data.events, [("batch", tuple(reqs), batch)])

    def test_compute_counts_batch_work_once_in_both_loops(self):
        for overlap in (False, True):
            for mode, tokens in (
                (ForwardMode.EXTEND, 128),
                (ForwardMode.MIXED, 129),
                (ForwardMode.DECODE, 0),
            ):
                with self.subTest(overlap=overlap, mode=mode):
                    scheduler = make_scheduler()
                    batch = SimpleNamespace(
                        reqs=[make_req("a"), make_req("b")],
                        forward_mode=mode,
                        extend_num_tokens=tokens,
                        spec_algorithm=SpeculativeAlgorithm.NONE,
                        req_pool_indices=None,
                    )
                    scheduler.metrics_reporter = SimpleNamespace(
                        record_scheduler_active=noop
                    )
                    scheduler.forward_ct = 0
                    scheduler._sched_idled = False
                    scheduler.scripted_scheduler_hook = None
                    scheduler.profiler_manager = SimpleNamespace(
                        _profile_batch_predicate=noop
                    )
                    scheduler.forward_sleep_time = None
                    scheduler.disaggregation_mode = DisaggregationMode.NULL
                    scheduler.is_generation = True
                    scheduler.enable_overlap = overlap
                    scheduler.enable_pdmux = False
                    scheduler.spec_algorithm = SpeculativeAlgorithm.NONE
                    scheduler.future_map = SimpleNamespace(resolve_seq_lens_cpu=noop)
                    scheduler._confidence_budget_prepare = None
                    scheduler.forward_stream_ctx = nullcontext()
                    scheduler.forward_stream = SimpleNamespace(wait_stream=noop)
                    scheduler.schedule_stream = object()
                    scheduler._forward_isolation = lambda *args, **kwargs: nullcontext()
                    resolved = []

                    def forward(batch, **kwargs):
                        self.assertEqual(resolved, [True])
                        for entry in scheduler.policy.policy_registry.values():
                            self.assertEqual(entry.policy_data.work, tokens)
                            self.assertEqual(
                                entry.policy_data.events, [("compute", batch)]
                            )
                        raise ForwardReached()

                    scheduler.model_worker = SimpleNamespace(
                        forward_batch_generation=forward
                    )
                    with (
                        patch(
                            "sglang.srt.managers.scheduler.resolve_forward_inputs",
                            side_effect=lambda *args: resolved.append(True),
                        ),
                        self.assertRaises(ForwardReached),
                    ):
                        scheduler.run_batch(batch)
                    self.assertEqual(scheduler.processed_tokens_counter, tokens)

    def test_stop_filters_stale_requests_then_disables_and_cleans_up(self):
        scheduler = make_scheduler()
        live, ending, stale = [make_req(name) for name in ("live", "ending", "stale")]
        for req in (live, ending):
            scheduler.policy.trigger_event(
                SchedulingEvent.ENABLE,
                req=req,
                engine_view=scheduler.get_scheduling_engine_view(
                    scheduler.running_batch
                ),
            )
        ending.finished_reason = FINISH_LENGTH(1)
        stale.finished_reason = FINISH_LENGTH(1)
        stale.scheduling_disabled = True
        batch = SimpleNamespace(
            reqs=[live, ending, stale], forward_mode=ForwardMode.DECODE
        )
        applied = [live, ending, stale]
        scheduler.batch_result_processor = SimpleNamespace(
            process_batch_result_decode=lambda *args: applied
        )
        scheduler.publish_load_snapshot = lambda **kwargs: None
        scheduler.load_publisher = SimpleNamespace(publish_load_stat=noop)
        scheduler.load_inquirer = SimpleNamespace(get_loads=noop)
        scheduler.tree_cache = SimpleNamespace(flush_pending_backups=noop)
        scheduler._record_step_counters = noop
        scheduler.metrics_reporter = SimpleNamespace(
            log_batch_result_stats=noop, update_device_timer=noop
        )
        scheduler._maybe_clear_mm_inputs = noop
        scheduler.maybe_send_health_check_signal = noop

        scheduler.process_batch_result(batch, object())
        for entry in scheduler.policy.policy_registry.values():
            self.assertEqual(
                entry.policy_data.events,
                [("stop", (live, ending), batch), ("disable", ending)],
            )
            self.assertEqual(set(entry.request_data), {"live"})

        # A later overlap result for terminated requests emits no more hooks.
        applied.clear()
        scheduler.process_batch_result(batch, object())
        for entry in scheduler.policy.policy_registry.values():
            self.assertEqual(len(entry.policy_data.events), 2)

    def test_registration_rejects_per_request_batch_handler(self):
        scheduler = make_scheduler()
        with self.assertRaises(TypeError):
            scheduler.policy.register(
                "invalid", rank, hooks={SchedulingEvent.ON_COMPUTE: lambda req: None}
            )
        self.assertNotIn("invalid", scheduler.policy.policy_registry)


if __name__ == "__main__":
    unittest.main(verbosity=2)
