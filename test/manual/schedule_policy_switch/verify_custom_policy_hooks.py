"""Verify hook order and custom metrics against real Scheduler request state.

Use a fresh, dedicated server for each mode, in the same container as this client:

    bash test/manual/schedule_policy_switch/start_custom_policy_hooks_server.sh
    python test/manual/schedule_policy_switch/verify_custom_policy_hooks.py

Restart with the retract profile to exercise the decode-memory retraction site:

    bash test/manual/schedule_policy_switch/start_custom_policy_hooks_server.sh /models/Qwen3-0.6B retract
    python test/manual/schedule_policy_switch/verify_custom_policy_hooks.py --mode retract

The server imports this file twice to create independent active/observer policies.
Snapshots are appended to /tmp/sglang-policy-hooks.jsonl. --snapshot-file and
--server-policy-file support shared filesystems with different client paths.
Only normal, non-overlap, non-speculative generation is covered. Retraction uses
SGLang's existing test switch, not artificial callbacks or a simulated GPU OOM.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import requests

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


POLICIES = ("manual_hooks_active", "manual_hooks_observer")
SNAPSHOT_FILE = Path("/tmp/sglang-policy-hooks.jsonl")
EVENTS = (
    "enable",
    "on_sort",
    "on_batch",
    "on_preempt",
    "on_compute",
    "on_stop",
    "disable",
)


@dataclass
class PolicyMetrics:
    label: str
    counters: dict[str, int] = field(default_factory=lambda: dict.fromkeys(EVENTS, 0))
    live_rids: set[str] = field(default_factory=set)
    initializations: int = 0
    terminal_requests: int = 0


@dataclass
class RequestMetrics:
    output_budget: int
    prompt_tokens: int
    phase: str = "waiting"
    counters: dict[str, int] = field(default_factory=lambda: dict.fromkeys(EVENTS, 0))
    generated_tokens: int = 0
    remaining_tokens: int = 0
    prefill_rounds: int = 0
    decode_rounds: int = 0
    no_output_rounds: int = 0
    queue_visits: int = 1
    last_extend_idx: int = 0
    last_decode_idx: int = 0
    round_kind: str = ""
    output_before_compute: int = 0
    compute_started_at: float = 0.0
    compute_seconds: float = 0.0
    waiting_seconds: float = 0.0
    score: int | None = None


def initialize_policy() -> PolicyMetrics:
    return PolicyMetrics(__name__.removeprefix("_sglang_schedule_policy_"))


def initialize_request(req: Req, policy_data: PolicyMetrics) -> RequestMetrics:
    policy_data.initializations += 1
    policy_data.live_rids.add(req.rid)
    return RequestMetrics(
        output_budget=req.sampling_params.max_new_tokens,
        prompt_tokens=len(req.origin_input_ids),
        remaining_tokens=req.sampling_params.max_new_tokens - len(req.output_ids),
    )


def _record(event, req, request_data, policy_data):
    data = request_data[req.rid]
    data.counters[event] += 1
    policy_data.counters[event] += 1
    extend = req.extend_range
    snapshot = {
        "policy": policy_data.label,
        "event": event,
        "rid": req.rid,
        "at": time.monotonic(),
        "sampled_at": time.perf_counter(),
        "req": {
            "output_tokens": len(req.output_ids),
            "budget": req.sampling_params.max_new_tokens,
            "prompt_tokens": len(req.origin_input_ids),
            "priority": req.priority,
            "finished": req.finished(),
            "finish_reason": req.finished_reason.to_json() if req.finished() else None,
            "retractions": req.retraction_count,
            "is_retracted": req.is_retracted,
            "extend_idx": req.extend_batch_idx,
            "decode_idx": req.decode_batch_idx,
            "extend_start": extend.start if extend is not None else None,
            "extend_end": extend.end if extend is not None else None,
            "pool_index": req.kv.req_pool_idx,
            "queue_entered_at": req.time_stats.wait_queue_entry_time,
        },
        "metrics": asdict(data),
        "policy_metrics": {
            "counters": dict(policy_data.counters),
            "live_rids": sorted(policy_data.live_rids),
            "initializations": policy_data.initializations,
            "terminal_requests": policy_data.terminal_requests,
        },
        # At DISABLE the request must still be present, then disappear before
        # the next request's ENABLE/ON_SORT or the final cleanup probe.
        "request_data_rids": sorted(request_data),
        "policy_object_id": hex(id(policy_data)),
        "request_object_id": hex(id(data)),
    }
    with SNAPSHOT_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(snapshot, ensure_ascii=False) + "\n")


def enable(req, request_data, policy_data):
    request_data[req.rid] = initialize_request(req, policy_data)
    _record("enable", req, request_data, policy_data)


def on_sort(reqs, request_data, policy_data):
    for req in reqs:
        data = request_data[req.rid]
        data.waiting_seconds = time.perf_counter() - req.time_stats.wait_queue_entry_time
        data.score = req.priority * 10000 + data.remaining_tokens
        _record("on_sort", req, request_data, policy_data)


def rank(waiting_queue, request_data, policy_data):
    # The ranking really consumes metrics updated by ON_SORT/ON_STOP.
    for req in waiting_queue:
        data = request_data[req.rid]
        assert data.counters["on_sort"] > 0, "ON_SORT did not prepare the metric"
        assert data.score == req.priority * 10000 + (
            req.sampling_params.max_new_tokens - len(req.output_ids)
        ), f"stale rank inputs: {req.rid}"
    waiting_queue.sort(
        key=lambda req: (
            request_data[req.rid].score,
            req.time_stats.wait_queue_entry_time,
        )
    )


def on_batch(reqs, request_data, policy_data):
    for req in reqs:
        data = request_data[req.rid]
        data.phase = "runnable"
        if req.decode_batch_idx > data.last_decode_idx:
            data.round_kind = "decode"
            data.decode_rounds += 1
        else:
            data.round_kind = "prefill"
            data.prefill_rounds += 1
        data.last_extend_idx = req.extend_batch_idx
        data.last_decode_idx = req.decode_batch_idx
        _record("on_batch", req, request_data, policy_data)


def on_preempt(req, request_data, policy_data):
    data = request_data[req.rid]
    data.phase = "waiting"
    data.queue_visits += 1
    data.last_extend_idx = data.last_decode_idx = 0
    data.score = None
    _record("on_preempt", req, request_data, policy_data)


def on_compute(reqs, request_data, policy_data):
    for req in reqs:
        data = request_data[req.rid]
        data.phase = "running"
        data.output_before_compute = len(req.output_ids)
        data.compute_started_at = time.monotonic()
        _record("on_compute", req, request_data, policy_data)


def on_stop(reqs, request_data, policy_data):
    for req in reqs:
        data = request_data[req.rid]
        delta = len(req.output_ids) - data.output_before_compute
        data.generated_tokens += delta
        data.remaining_tokens = data.output_budget - len(req.output_ids)
        data.no_output_rounds += int(delta == 0)
        data.compute_seconds += time.monotonic() - data.compute_started_at
        data.phase = "settled"
        _record("on_stop", req, request_data, policy_data)


def disable(req, request_data, policy_data):
    data = request_data[req.rid]
    data.phase = "terminal"
    policy_data.terminal_requests += 1
    policy_data.live_rids.remove(req.rid)
    _record("disable", req, request_data, policy_data)


class HookVerifier:
    def __init__(self, args):
        from verify_custom_policy_switch import CustomPolicySwitchVerifier

        self.client = CustomPolicySwitchVerifier(
            args.base_url, args.server_policy_file, args.blocker_tokens
        )
        self.client.prefix = f"hooks-{time.time_ns()}"
        self.snapshot_file = args.snapshot_file
        self.mode = args.mode
        self.expected = {}
        self.futures = {}
        self.request_args = {}

    def rid(self, name):
        return f"{self.client.prefix}-{name}"

    def read_records(self):
        if not self.snapshot_file.exists():
            return []
        return [
            item
            for line in self.snapshot_file.read_text(encoding="utf-8").splitlines()
            if line.endswith("}")
            and (item := json.loads(line))["rid"].startswith(self.client.prefix)
        ]

    def wait_event(self, name, event, future=None):
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            matches = [
                item
                for item in self.read_records()
                if item["rid"] == self.rid(name)
                and item["event"] == event
                and item["policy"] == POLICIES[0]
            ]
            if matches:
                return matches[-1]
            if future is not None and future.done():
                future.result()
                raise AssertionError(f"{name} completed without {event}")
            time.sleep(0.02)
        raise TimeoutError(f"没有观察到 {name}/{event}")

    def generate(
        self, name, tokens=8, prompt_tokens=16, priority=100, http_abort=False
    ):
        response = requests.post(
            f"{self.client.base_url}/generate",
            json={
                "rid": self.rid(name),
                "input_ids": [42 + index % 64 for index in range(prompt_tokens)],
                "priority": priority,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": tokens,
                    "ignore_eos": True,
                },
            },
            timeout=120,
        )
        if name.endswith("rejected"):
            assert response.status_code == 503, response.text
            return None
        if http_abort:
            assert response.status_code == 503, response.text
            return {"http_status": 503, "finish_reason": {"type": "abort"}}
        response.raise_for_status()
        return response.json()["meta_info"]

    def submit(self, executor, name, **kwargs):
        self.request_args[name] = {
            "tokens": 8,
            "prompt_tokens": 16,
            "priority": 100,
            **kwargs,
        }
        future = executor.submit(self.generate, name, **kwargs)
        self.futures[name] = future
        return future

    def finish(self, name, future, *, aborted=False):
        meta = future.result(timeout=120)
        self.wait_event(name, "disable")
        assert (meta["finish_reason"]["type"] == "abort") == aborted, meta
        self.expected[name] = (meta, aborted)
        return meta

    def cancel(self, name, future):
        self.client.abort(self.rid(name))
        return self.finish(name, future, aborted=True)

    def run_normal(self, executor):
        for name, prompt_tokens in (("short", 16), ("chunk", 512)):
            self.finish(name, self.submit(executor, name, prompt_tokens=prompt_tokens))
            print(f"场景完成：{name}，prompt={prompt_tokens}，输出=8", flush=True)

        tail = self.submit(executor, "mixed-decode", tokens=self.client.blocker_tokens)
        self.client.wait_for_load(1, 0, tail)
        prefill = self.submit(executor, "mixed-prefill", prompt_tokens=512)
        self.finish("mixed-prefill", prefill)
        self.client.check_blocker(tail)
        self.cancel("mixed-decode", tail)
        print("场景完成：chunk prefill 与已有 decode 混合执行", flush=True)

        chunk = self.submit(executor, "chunk-cancel", prompt_tokens=8192)
        first_chunk = self.wait_event("chunk-cancel", "on_stop", chunk)
        assert first_chunk["req"]["output_tokens"] == 0
        self.cancel("chunk-cancel", chunk)
        print("场景完成：尚未完成的 chunk prefill 被取消", flush=True)

        # Higher-priority running requests prevent waiting ones from preempting.
        blockers = []
        for number in (1, 2):
            name = f"queue-blocker-{number}"
            future = self.submit(
                executor, name, tokens=self.client.blocker_tokens, priority=0
            )
            self.client.wait_for_load(number, 0, future)
            blockers.append((name, future))
        queued = []
        for number in (1, 2):
            name = f"queued-{number}"
            future = self.submit(executor, name, http_abort=number == 2)
            self.client.wait_for_load(2, number, blockers[0][1])
            queued.append((name, future))
        self.generate("intake-rejected")
        replacement = self.submit(executor, "queue-replacement", priority=50)
        # Queue priority ties evict the most recently enqueued request.
        self.finish("queued-2", queued[1][1], aborted=True)
        self.client.wait_for_load(2, 2, blockers[0][1])
        self.cancel("queued-1", queued[0][1])
        self.cancel("queue-replacement", replacement)
        for name, future in blockers:
            self.cancel(name, future)
        print("场景完成：首次入队拒绝、等待队列淘汰、等待取消及运行中取消", flush=True)

        # Fill both running slots, then introduce a better-priority request.
        blockers = []
        for number in (1, 2):
            name = f"priority-victim-{number}"
            future = self.submit(executor, name, tokens=self.client.blocker_tokens)
            self.client.wait_for_load(number, 0, future)
            blockers.append((name, future))
        urgent = self.submit(executor, "urgent", priority=0)
        self.finish("urgent", urgent)
        records = self.read_records()
        victims = {item["rid"] for item in records if item["event"] == "on_preempt"}
        assert victims, "优先级抢占场景没有真正发生抢占"
        # Wait for a requeued victim to execute again, not merely to be aborted.
        for name, future in blockers:
            if self.rid(name) in victims:
                self.wait_event(name, "on_sort", future)
                deadline = time.monotonic() + 20
                while time.monotonic() < deadline:
                    if any(
                        item["rid"] == self.rid(name)
                        and item["event"] == "on_compute"
                        and item["req"]["retractions"] > 0
                        for item in self.read_records()
                    ):
                        break
                    self.client.check_blocker(future)
                    time.sleep(0.02)
                else:
                    raise AssertionError("抢占请求未恢复执行")
            self.cancel(name, future)
        print("场景完成：优先级抢占、重新排序及恢复执行", flush=True)

        # Both policies keep receiving hooks while FCFS is active.
        self.client.switch_policy("fcfs")
        self.finish("inactive", self.submit(executor, "inactive"))
        self.client.switch_policy(POLICIES[1])
        self.finish("observer-active", self.submit(executor, "observer-active"))
        print("场景完成：两个策略均未激活时继续更新，以及切换激活观察策略", flush=True)

    def run_retract(self, executor):
        futures = [
            (f"memory-{number}", self.submit(executor, f"memory-{number}", tokens=96))
            for number in (1, 2)
        ]
        for name, future in futures:
            self.finish(name, future)
        assert any(item["event"] == "on_preempt" for item in self.read_records()), (
            "没有发生 decode retract；请用 retract 启动配置"
        )
        print(
            "场景完成：decode retract 路径抢占并恢复，两个请求均生成 96 token",
            flush=True,
        )

    def verify_metrics(self):
        records = self.read_records()
        by_policy = {
            name: [item for item in records if item["policy"] == name]
            for name in POLICIES
        }
        for policy, items in by_policy.items():
            assert items, f"没有收到策略 {policy} 的事件"
            shared_ids = {item["policy_object_id"] for item in items}
            assert len(shared_ids) == 1, "策略共享对象被替换"
            global_counts = Counter()
            initialized, terminal = set(), set()
            groups = defaultdict(list)
            for item in items:
                rid, event = item["rid"], item["event"]
                groups[rid].append(item)
                global_counts[event] += 1
                if event == "enable":
                    assert rid not in initialized
                    initialized.add(rid)
                if event == "disable":
                    assert rid not in terminal, f"重复 DISABLE: {rid}"
                    terminal.add(rid)
                shared = item["policy_metrics"]
                assert shared["counters"] == dict.fromkeys(EVENTS, 0) | dict(
                    global_counts
                )
                assert shared["initializations"] == len(initialized)
                assert shared["terminal_requests"] == len(terminal)
                assert set(shared["live_rids"]) == initialized - terminal
                assert rid in item["request_data_rids"], "回调前请求数据已被清理"
                assert not (terminal - {rid}) & set(item["request_data_rids"]), (
                    "此前结束的请求数据没有被框架清理"
                )

            for name, (meta, aborted) in self.expected.items():
                trace = groups[self.rid(name)]
                self.verify_request(name, trace, meta, aborted)
            assert self.rid("intake-rejected") not in groups, "首次入队拒绝触发了 hook"
            assert initialized == terminal, "测试结束仍有未终止请求"
            assert all(global_counts[event] > 0 for event in EVENTS), global_counts
            print(f"\n{policy} 共享指标：{dict(global_counts)}", flush=True)
            print(
                f"  初始化={len(initialized)}，终止={len(terminal)}，存活=0", flush=True
            )

        first, second = (by_policy[name] for name in POLICIES)
        assert first[0]["policy_object_id"] != second[0]["policy_object_id"]
        for name in self.expected:
            traces = [
                [item for item in items if item["rid"] == self.rid(name)]
                for items in (first, second)
            ]
            assert [item["event"] for item in traces[0]] == [
                item["event"] for item in traces[1]
            ]
            assert (
                traces[0][-1]["request_object_id"] != traces[1][-1]["request_object_id"]
            )
            assert (
                traces[0][-1]["metrics"]["counters"]
                == traces[1][-1]["metrics"]["counters"]
            )

        if self.mode == "normal":
            compute = [
                item for item in first if item["event"] in ("on_compute", "on_stop")
            ]
            pair = {self.rid("mixed-decode"), self.rid("mixed-prefill")}
            assert any(
                a["event"] == b["event"] == "on_compute"
                and {a["rid"], b["rid"]} == pair
                and {a["metrics"]["round_kind"], b["metrics"]["round_kind"]}
                == {"prefill", "decode"}
                for a, b in zip(compute, compute[1:])
            ), "未观察到真正的混合 batch"
        print(
            "\nPASS：7 个 hook 的顺序、更新结果、策略隔离和数据清理检查全部通过",
            flush=True,
        )

    def verify_request(self, name, trace, meta, aborted):
        assert trace[0]["event"] == "enable" and trace[-1]["event"] == "disable", name
        assert len({item["request_object_id"] for item in trace}) == 1
        counts = Counter()
        previous = ""
        output_before = 0
        no_output = 0
        computing = None
        previous_compute_seconds = 0.0
        prefill_rounds = decode_rounds = 0
        last_extend = last_decode = 0
        for item in trace:
            event, req, metric = item["event"], item["req"], item["metrics"]
            counts[event] += 1
            assert metric["counters"] == dict.fromkeys(EVENTS, 0) | dict(counts), (
                name,
                item,
            )
            assert metric["remaining_tokens"] == req["budget"] - req["output_tokens"], (
                name,
                item,
            )
            assert metric["generated_tokens"] == req["output_tokens"], (name, item)
            assert metric["queue_visits"] == 1 + counts["on_preempt"]
            if event in ("enable", "on_sort", "on_preempt"):
                assert metric["phase"] == "waiting" and req["queue_entered_at"] > 0
            if event == "on_sort":
                assert previous in ("enable", "on_sort", "on_preempt"), (
                    name,
                    previous,
                    event,
                )
                assert metric["waiting_seconds"] >= 0
                observed_wait = item["sampled_at"] - req["queue_entered_at"]
                assert 0 <= observed_wait - metric["waiting_seconds"] < 0.1
                assert (
                    metric["score"]
                    == req["priority"] * 10000 + req["budget"] - req["output_tokens"]
                )
            elif event == "on_batch":
                assert previous in ("on_sort", "on_stop"), (name, previous, event)
                assert metric["phase"] == "runnable" and req["pool_index"] is not None
                assert counts["on_batch"] == counts["on_stop"] + 1
                assert (
                    metric["prefill_rounds"] + metric["decode_rounds"]
                    == counts["on_batch"]
                )
                if req["decode_idx"] > last_decode:
                    decode_rounds += 1
                    assert metric["round_kind"] == "decode"
                else:
                    assert req["extend_idx"] > last_extend
                    prefill_rounds += 1
                    assert metric["round_kind"] == "prefill"
                last_extend, last_decode = req["extend_idx"], req["decode_idx"]
                assert metric["prefill_rounds"] == prefill_rounds
                assert metric["decode_rounds"] == decode_rounds
            elif event == "on_compute":
                assert previous == "on_batch" and metric["phase"] == "running"
                assert counts["on_compute"] == counts["on_batch"]
                output_before = req["output_tokens"]
                computing = item
            elif event == "on_stop":
                assert previous == "on_compute" and metric["phase"] == "settled"
                assert counts["on_stop"] == counts["on_compute"]
                if not aborted:
                    assert req["finished"] == (req["output_tokens"] == req["budget"])
                delta = req["output_tokens"] - output_before
                compute_req = computing["req"]
                middle_chunk = (
                    computing["metrics"]["round_kind"] == "prefill"
                    and compute_req["extend_end"]
                    < compute_req["prompt_tokens"] + output_before
                )
                assert delta == (0 if middle_chunk else 1), (name, delta, computing)
                no_output += int(delta == 0)
                assert metric["no_output_rounds"] == no_output
                elapsed = metric["compute_seconds"] - previous_compute_seconds
                observed = item["at"] - computing["metrics"]["compute_started_at"]
                assert 0 < elapsed <= observed and observed - elapsed < 0.1
                previous_compute_seconds = metric["compute_seconds"]
            elif event == "on_preempt":
                assert previous == "on_stop", (name, previous)
                assert req["is_retracted"] and req["pool_index"] is None
                assert req["extend_idx"] == req["decode_idx"] == 0
                assert metric["score"] is None
                last_extend = last_decode = 0
            elif event == "disable":
                assert previous in ("enable", "on_sort", "on_stop", "on_preempt")
                assert req["finished"] and metric["phase"] == "terminal"
            assert req["retractions"] == counts["on_preempt"], (name, item)
            previous = event
        final = trace[-1]
        metric = final["metrics"]
        requested = self.request_args[name]
        assert metric["output_budget"] == requested["tokens"]
        assert metric["prompt_tokens"] == requested["prompt_tokens"]
        assert counts["enable"] == counts["disable"] == 1
        if "http_status" in meta:
            assert aborted and metric["generated_tokens"] == 0
            assert counts["on_compute"] == 0
        else:
            assert metric["generated_tokens"] == meta["completion_tokens"], (
                name,
                meta,
                metric,
            )
        if not aborted:
            assert meta["completion_tokens"] == final["req"]["budget"]
            assert trace[-2]["event"] == "on_stop" and trace[-2]["req"]["finished"]
        if name in ("short", "inactive", "observer-active", "cleanup-probe"):
            assert counts["on_batch"] == counts["on_compute"] == counts["on_stop"] == 8
            assert metric["prefill_rounds"] == 1 and metric["decode_rounds"] == 7
            assert metric["no_output_rounds"] == counts["on_preempt"] == 0
        elif name == "chunk":
            assert counts["on_stop"] == 11, metric
            assert metric["prefill_rounds"] == 4 and metric["decode_rounds"] == 7
            assert metric["no_output_rounds"] == 3
            prefills = [
                item
                for item in trace
                if item["event"] == "on_batch"
                and item["metrics"]["round_kind"] == "prefill"
            ]
            assert [
                (item["req"]["extend_start"], item["req"]["extend_end"])
                for item in prefills
            ] == [(0, 128), (128, 256), (256, 384), (384, 512)]
        elif name in ("queued-1", "queued-2", "queue-replacement"):
            assert counts["on_batch"] == counts["on_compute"] == counts["on_stop"] == 0
        elif name == "chunk-cancel":
            assert counts["on_stop"] > 0 and metric["generated_tokens"] == 0
            assert metric["no_output_rounds"] == counts["on_stop"]
        print(
            f"  PASS {name}: 输出={metric['generated_tokens']}，prefill={metric['prefill_rounds']}，decode={metric['decode_rounds']}，无输出轮次={metric['no_output_rounds']}，抢占={counts['on_preempt']}",
            flush=True,
        )

    def run(self):
        info, state = self.client.get_state()
        assert info["disable_overlap_schedule"], "此测试要求关闭 overlap"
        assert info["disaggregation_mode"] == "null"
        assert info["chunked_prefill_size"] == 128 and info["disable_radix_cache"]
        assert info["enable_mixed_chunk"] and info["schedule_low_priority_values_first"]
        assert state["effective_max_running_requests_per_dp"] == 2
        self.client.wait_for_load(0, 0)
        original = state["schedule_policy"]
        print(
            "每请求指标：事件计数、输出/剩余额度、prefill/decode轮次、无输出轮次、排队次数、等待时间、排序分数及执行耗时。\n每策略共享指标：各事件总数、初始化/终止数、存活请求集合；两策略对象彼此独立。",
            flush=True,
        )
        try:
            for name in POLICIES:
                result = self.client.request_json(
                    "POST",
                    "/schedule_policy_register",
                    json={
                        "policy_name": name,
                        "register_path": self.client.server_policy_file,
                        "rank_fn_name": "rank",
                    },
                )
                assert result["success"], result
            self.client.switch_policy(POLICIES[0])
            with ThreadPoolExecutor(max_workers=6) as executor:
                try:
                    if self.mode == "normal":
                        self.run_normal(executor)
                    else:
                        self.run_retract(executor)
                    self.client.wait_for_load(0, 0)
                    self.finish("cleanup-probe", self.submit(executor, "cleanup-probe"))
                    self.client.wait_for_load(0, 0)
                finally:
                    for name, future in self.futures.items():
                        if not future.done():
                            self.client.abort(self.rid(name))
            self.verify_metrics()
        finally:
            self.client.switch_policy(original)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--server-policy-file", default=str(Path(__file__).resolve()))
    parser.add_argument("--snapshot-file", type=Path, default=SNAPSHOT_FILE)
    parser.add_argument("--blocker-tokens", type=int, default=4096)
    parser.add_argument("--mode", choices=("normal", "retract"), default="normal")
    HookVerifier(parser.parse_args()).run()


if __name__ == "__main__":
    main()
