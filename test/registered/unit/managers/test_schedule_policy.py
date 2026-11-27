import unittest
from array import array
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.schedule_policy import SchedulePolicy
from sglang.srt.managers.scheduler_components.schedule_events import (
    SchedulingEngineView,
    SchedulingEvent,
)
from sglang.srt.managers.scheduler_components.scheduling_policy import (
    hrrn,
    shortest_prefill_first,
)
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_req(rid, origin_input_text, origin_input_ids, sampling_params=None, **kwargs):
    if sampling_params is None:
        sampling_params = SamplingParams()
    return Req(
        rid,
        origin_input_text,
        array("q", origin_input_ids),
        sampling_params,
        **kwargs,
    )


class TestSchedulePolicyHRRN(CustomTestCase):
    def make_policy(self):
        return SchedulePolicy(
            policy="hrrn",
            engine_view=SchedulingEngineView(
                tree_cache=RadixCache.create_simulated(),
                enable_hierarchical_cache=True,
                enable_priority_scheduling=False,
                schedule_low_priority_values_first=False,
                running_reqs=(),
            ),
        )

    def enqueue(self, policy, *reqs):
        for req in reqs:
            policy.trigger_event(
                SchedulingEvent.ENABLE, req=req, engine_view=policy.engine_view
            )

    def submit_work(self, policy, tokens):
        policy.trigger_event(
            SchedulingEvent.ON_COMPUTE,
            reqs=(),
            batch=SimpleNamespace(extend_num_tokens=tokens),
        )

    def test_calc_priority_hrrn(self):
        """Equal waiting work gives shorter uncached prompts higher ratios."""
        policy = self.make_policy()
        short = _make_req("short", "a", [1])
        long = _make_req("long", "a" * 10, list(range(10)))
        aged = _make_req("aged", "a" * 3, [1, 2, 3])
        queue = [long, aged, short]
        self.enqueue(policy, *queue)
        self.submit_work(policy, 1000)
        policy.calc_priority(queue)
        self.assertEqual([req.rid for req in queue], ["short", "aged", "long"])

    def test_calc_priority_hrrn_aging_overtakes_short(self):
        """An older long request outranks a newly enqueued short request."""
        policy = self.make_policy()
        old = _make_req("long_old", "a" * 100, list(range(100)))
        new = _make_req("short_new", "a", [1])
        self.enqueue(policy, old)
        self.submit_work(policy, 100000)
        self.enqueue(policy, new)
        queue = [new, old]
        policy.calc_priority(queue)
        self.assertEqual([req.rid for req in queue], ["long_old", "short_new"])

    def test_calc_priority_hrrn_cached_length_affects_order(self):
        """Cache hits reduce service work; equal ratios use the request ID."""
        policy = self.make_policy()
        more_cached = _make_req("a", "x" * 100, list(range(100)))
        less_cached = _make_req("b", "x" * 100, list(range(100)))
        tie = _make_req("c", "x" * 100, list(range(100)))
        more_cached.num_matched_prefix_tokens = 90
        queue = [tie, less_cached, more_cached]
        self.enqueue(policy, *queue)
        self.submit_work(policy, 1000)
        with patch.object(hrrn, "_compute_prefix_matches", return_value=set()):
            policy.calc_priority(queue)
        self.assertEqual([req.rid for req in queue], ["a", "b", "c"])


class TestShortestPrefillFirst(CustomTestCase):
    def setUp(self):
        self.policy = SchedulePolicy(
            policy="shortest-prefill-first",
            engine_view=SchedulingEngineView(
                tree_cache=RadixCache.create_simulated(),
                enable_hierarchical_cache=True,
                enable_priority_scheduling=False,
                schedule_low_priority_values_first=False,
                running_reqs=(),
            ),
        )

    def make_req(self, rid, uncached, *, cached=0, arrived=0):
        req = _make_req(rid, "", list(range(uncached + cached)))
        req.full_untruncated_fill_ids = req.origin_input_ids[:]
        req.num_matched_prefix_tokens = cached
        req.prefix_indices = list(range(cached))
        req.time_stats.wait_queue_entry_time = arrived
        return req

    def test_calc_priority_uses_uncached_work(self):
        cached = self.make_req("cached", 16, cached=4096)
        short = self.make_req("short", 32)
        long = self.make_req("long", 1024)
        queue = [long, short, cached]
        with patch.object(
            shortest_prefill_first, "_compute_prefix_matches", return_value=set()
        ):
            self.policy.calc_priority(queue)
        self.assertEqual([req.rid for req in queue], ["cached", "short", "long"])

    def test_equal_work_uses_arrival_time(self):
        older = self.make_req("z", 32, arrived=1)
        newer = self.make_req("a", 32, arrived=2)
        queue = [newer, older]
        with patch.object(
            shortest_prefill_first, "_compute_prefix_matches", return_value=set()
        ):
            self.policy.calc_priority(queue)
        self.assertEqual(queue, [older, newer])

    def test_duplicate_prefix_is_deprioritized(self):
        duplicate = self.make_req("duplicate", 1)
        other = self.make_req("other", 1024)
        queue = [duplicate, other]
        with patch.object(
            shortest_prefill_first,
            "_compute_prefix_matches",
            return_value={duplicate.rid},
        ):
            self.policy.calc_priority(queue)
        self.assertEqual(queue, [other, duplicate])

    def test_retracted_output_is_part_of_uncached_work(self):
        replay = self.make_req("replay", 16, cached=1024)
        replay.output_ids.extend([0] * 64)
        short = self.make_req("short", 32)
        queue = [replay, short]
        with patch.object(
            shortest_prefill_first, "_compute_prefix_matches", return_value=set()
        ):
            self.policy.calc_priority(queue)
        self.assertEqual(queue, [short, replay])

    def test_chunk_limit_reserves_complete_short_prefills(self):
        continuation = self.make_req("continuation", 16384)
        waiting = [self.make_req("a", 512), self.make_req("b", 1024)]
        self.assertEqual(
            self.policy.shortest_prefill_chunk_limit(continuation, waiting, 4096, 256),
            2560,
        )

    def test_reservation_rounds_to_pages_and_keeps_continuation_progress(self):
        continuation = self.make_req("continuation", 16384)
        self.assertEqual(
            self.policy.shortest_prefill_chunk_limit(
                continuation, [self.make_req("short", 257)], 4096, 256
            ),
            3584,
        )
        self.assertEqual(
            self.policy.shortest_prefill_chunk_limit(
                continuation, [self.make_req("short", 3840)], 4096, 256
            ),
            256,
        )

    def test_no_reservation_when_request_cannot_fit_or_is_not_shorter(self):
        continuation = self.make_req("continuation", 8192)
        for waiting, budget in [
            ([], 4096),
            ([self.make_req("same", 8192)], 4096),
            ([self.make_req("too-large", 4096)], 4096),
            ([self.make_req("short", 1)], 256),
        ]:
            with self.subTest(budget=budget, waiting=[req.rid for req in waiting]):
                self.assertIsNone(
                    self.policy.shortest_prefill_chunk_limit(
                        continuation, waiting, budget, 256
                    )
                )

    def test_other_policy_keeps_normal_chunk_limit(self):
        self.policy.activate("hrrn")
        self.assertIsNone(
            self.policy.shortest_prefill_chunk_limit(
                self.make_req("continuation", 8192),
                [self.make_req("short", 512)],
                4096,
                256,
            )
        )


class TestSchedulingEngineView(unittest.TestCase):
    def setUp(self):
        self.engine_view = SchedulingEngineView(
            tree_cache=RadixCache.create_simulated(),
            enable_hierarchical_cache=False,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
            running_reqs=(),
        )
        self.policy = SchedulePolicy("fcfs", engine_view=self.engine_view)

    def test_existing_and_new_requests_use_their_initialization_view(self):
        existing = _make_req("existing", "a", [1])
        new = _make_req("new", "b", [2])
        running = _make_req("running", "c", [3])
        registration_view = replace(self.engine_view, running_reqs=(running,))
        self.policy.engine_view = registration_view

        def initialize_policy(engine_view):
            return SimpleNamespace(initial_view=engine_view, initializations=0)

        def initialize_request(req, *, engine_view, policy_data):
            policy_data.initializations += 1
            return tuple(item.rid for item in engine_view.running_reqs)

        def enable(req, engine_view, request_data, policy_data):
            request_data[req.rid] = initialize_request(
                req, engine_view=engine_view, policy_data=policy_data
            )

        self.policy.register(
            "view_probe",
            lambda waiting_queue: None,
            initialize_policy_fn=initialize_policy,
            initialize_request_fn=initialize_request,
            hooks={SchedulingEvent.ENABLE: enable},
        )
        entry = self.policy.policy_registry["view_probe"]
        entry.initialize_request(existing, engine_view=registration_view)
        entry.initialize_request(existing, engine_view=registration_view)
        self.policy.trigger_event(
            SchedulingEvent.ENABLE, req=new, engine_view=self.engine_view
        )

        self.assertEqual(entry.request_data, {"existing": ("running",), "new": ()})
        self.assertIs(entry.policy_data.initial_view, registration_view)
        self.assertEqual(entry.policy_data.initializations, 2)

    def test_unsupported_rank_parameters_do_not_register_or_initialize(self):
        for rank_fn in (
            lambda waiting_queue, running_batch: None,
            lambda waiting_queue, processed_tokens: None,
        ):
            with self.subTest(rank_fn=rank_fn):
                initialized = []
                with self.assertRaisesRegex(ValueError, "Unsupported rank parameter"):
                    self.policy.register(
                        "invalid",
                        rank_fn,
                        initialize_policy_fn=lambda: initialized.append(True),
                    )
                self.assertNotIn("invalid", self.policy.policy_registry)
                self.assertEqual(initialized, [])


if __name__ == "__main__":
    unittest.main()
