"""Verify unregister/re-register through HTTP while requests are still live.

Start a dedicated normal, non-overlap server with
start_custom_policy_switch_server.sh, then run this file in the same container:

    python test/manual/schedule_policy_switch/verify_custom_policy_unregister.py

The server imports this file as two independent policies. The observer keeps
references to the target's test data to check that unregister clears the request
table and stops updating the old shared state. It never holds the target module.
This is test-only observation; no production inspection endpoint is required.

Snapshots go to /tmp/sglang-policy-unregister.jsonl. For different client/server
paths, use --server-policy-file and --snapshot-file on a shared filesystem.
Only normal, non-overlap, non-speculative generation is covered.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import requests


TARGET = "manual_unregister"
OBSERVER = "manual_unregister_observer"
SNAPSHOT_FILE = Path("/tmp/sglang-policy-unregister.jsonl")


@dataclass
class PolicyData:
    generation: int = field(default_factory=time.time_ns)
    request_initializations: int = 0
    rank_calls: int = 0
    hook_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class RequestData:
    output_budget: int
    initial_output: int
    observed_output: int
    settled_rounds: int = 0


_policy_data = None
_request_data_ref = None
_observed_target_data = None
_observed_target_requests = None


def _record(event, req=None):
    global _observed_target_data, _observed_target_requests
    policy_name = __name__.removeprefix("_sglang_schedule_policy_")
    target_module = sys.modules.get(f"_sglang_schedule_policy_{TARGET}")
    observation = None
    if policy_name == OBSERVER:
        if target_module is not None:
            _observed_target_data = target_module._policy_data
            _observed_target_requests = target_module._request_data_ref
        if _observed_target_data is not None:
            observation = {
                "generation": _observed_target_data.generation,
                "hook_counts": dict(_observed_target_data.hook_counts),
                "request_data_size": (
                    len(_observed_target_requests)
                    if _observed_target_requests is not None
                    else 0
                ),
            }
    data = _request_data_ref or {}
    record = {
        "policy": policy_name,
        "event": event,
        "rid": req.rid if req is not None else None,
        "output_tokens": len(req.output_ids) if req is not None else None,
        "finished": req.finished() if req is not None else None,
        "policy_data": asdict(_policy_data),
        "request_data": {rid: asdict(value) for rid, value in data.items()},
        "target_module_present": target_module is not None,
        "observed_target": observation,
    }
    with SNAPSHOT_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")


def initialize_policy():
    global _policy_data
    _policy_data = PolicyData()
    _record("initialize_policy")
    return _policy_data


def initialize_request(req, policy_data):
    policy_data.request_initializations += 1
    data = RequestData(
        output_budget=req.sampling_params.max_new_tokens,
        initial_output=len(req.output_ids),
        observed_output=len(req.output_ids),
    )
    _record("initialize_request", req)
    return data


def _event(event, req, request_data, policy_data):
    global _request_data_ref
    _request_data_ref = request_data
    assert req.rid in request_data, f"{req.rid}: request data is missing at {event}"
    policy_data.hook_counts[event] = policy_data.hook_counts.get(event, 0) + 1
    if event == "on_stop":
        data = request_data[req.rid]
        data.observed_output = len(req.output_ids)
        data.settled_rounds += 1
    _record(event, req)


def enable(req, request_data, policy_data):
    assert req.rid not in request_data, f"Duplicate ENABLE: {req.rid}"
    request_data[req.rid] = initialize_request(req, policy_data)
    _event("enable", req, request_data, policy_data)


def on_sort(reqs, request_data, policy_data):
    for req in reqs:
        _event("on_sort", req, request_data, policy_data)


def on_batch(reqs, request_data, policy_data):
    for req in reqs:
        _event("on_batch", req, request_data, policy_data)


def on_preempt(req, request_data, policy_data):
    _event("on_preempt", req, request_data, policy_data)


def on_compute(reqs, request_data, policy_data):
    for req in reqs:
        _event("on_compute", req, request_data, policy_data)


def on_stop(reqs, request_data, policy_data):
    for req in reqs:
        _event("on_stop", req, request_data, policy_data)


def disable(req, request_data, policy_data):
    assert req.finished(), f"{req.rid}: DISABLE before terminal state"
    _event("disable", req, request_data, policy_data)


def rank(waiting_queue, request_data, policy_data):
    policy_data.rank_calls += 1
    for req in waiting_queue:
        assert request_data[req.rid].observed_output == len(req.output_ids)
    waiting_queue.sort(
        key=lambda req: (
            request_data[req.rid].output_budget,
            req.time_stats.wait_queue_entry_time,
        )
    )
    _record("rank")


class UnregisterVerifier:
    def __init__(self, args):
        # Server imports need no client helper on sys.path.
        from verify_custom_policy_switch import CustomPolicySwitchVerifier

        self.client = CustomPolicySwitchVerifier(
            args.base_url, args.server_policy_file, args.blocker_tokens
        )
        self.snapshot_file = args.snapshot_file
        self.log_offset = len(self._read_log())
        self.registered = set()
        self.futures = {}

    def _read_log(self):
        if not self.snapshot_file.exists():
            return []
        # A snapshot may still be appended while the client reads the file.
        return [
            json.loads(line)
            for line in self.snapshot_file.read_text().splitlines()
            if line.endswith("}")
        ]

    def records(self):
        return self._read_log()[self.log_offset :]

    def control(self, operation, payload, *, success=True, message=""):
        response = requests.post(
            f"{self.client.base_url}/schedule_policy_{operation}",
            json=payload,
            timeout=10,
        )
        result = response.json()
        assert response.status_code == (200 if success else 400), result
        assert result["success"] is success, result
        assert message in result["message"], result
        return result

    def register(self, name):
        self.control(
            "register",
            {
                "policy_name": name,
                "register_path": self.client.server_policy_file,
                "rank_fn_name": "rank",
            },
        )
        self.registered.add(name)
        initial = [
            item
            for item in self.records()
            if item["policy"] == name and item["event"] == "initialize_policy"
        ][-1]
        assert initial["policy_data"]["request_initializations"] == 0
        assert initial["policy_data"]["rank_calls"] == 0
        assert initial["policy_data"]["hook_counts"] == {}
        generation = initial["policy_data"]["generation"]
        print(f"注册成功：{name}，数据代次={generation}，初始计数全部为 0", flush=True)
        return generation

    def unregister(self, name):
        result = self.control("unregister", {"policy_name": f" {name.upper()} "})
        assert result["unregister_schedule_policy"] == name, result
        self.registered.remove(name)

    def submit(self, executor, name, tokens):
        rid = f"{self.client.prefix}-{name}"
        future = executor.submit(self.client.generate, rid, tokens)
        self.futures[rid] = future
        return rid, future

    def wait_record(self, policy, event, rid, *, generation=None):
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            for item in reversed(self.records()):
                if (
                    item["policy"] == policy
                    and item["event"] == event
                    and item["rid"] == rid
                    and (
                        generation is None
                        or item["policy_data"]["generation"] == generation
                    )
                ):
                    return item
            time.sleep(0.02)
        raise AssertionError(
            f"没有观察到 {policy}/{event}/{rid}，检查 --snapshot-file 路径"
        )

    def generate_probe(self, name):
        rid = f"{self.client.prefix}-{name}"
        meta = self.client.generate(rid, 8)["meta_info"]
        assert meta["completion_tokens"] == 8, meta
        assert meta["finish_reason"]["type"] != "abort", meta
        self.client.wait_for_load(0, 0)
        return rid

    def check_detached(self, generation, *, since):
        records = self.records()[since:]
        observed = [
            item
            for item in records
            if item["policy"] == OBSERVER
            and not item["target_module_present"]
            and item["observed_target"] is not None
        ]
        assert observed, "没有观察到注销后的执行事件"
        counts = observed[0]["observed_target"]["hook_counts"]
        for item in observed:
            old = item["observed_target"]
            assert old["generation"] == generation
            assert old["request_data_size"] == 0, "注销没有清空旧请求数据"
            assert old["hook_counts"] == counts, "注销后旧策略的指标仍在变化"
        assert not any(item["policy"] == TARGET for item in records), (
            "注销后目标策略仍收到回调"
        )
        return counts

    def run_lifecycle(self, executor):
        observer_generation = self.register(OBSERVER)
        first_generation = self.register(TARGET)
        self.client.switch_policy(TARGET)
        for name, message in (
            (TARGET, "Switch to another policy"),
            ("fcfs", "FCFS is required"),
            ("manual_unregister_missing", "Unknown policy"),
            ("", "Unknown policy"),
        ):
            self.control(
                "unregister", {"policy_name": name}, success=False, message=message
            )
        assert self.client.get_state()[1]["schedule_policy"] == TARGET
        probe = self.generate_probe("before-unregister")
        self.wait_record(TARGET, "disable", probe)
        print("PASS：当前策略/FCFS/未知名称/空名称保护，拒绝后策略仍能工作", flush=True)

        blocker_rid, blocker = self.submit(
            executor, "blocker", self.client.blocker_tokens
        )
        self.client.wait_for_load(1, 0, blocker)
        self.wait_record(TARGET, "on_stop", blocker_rid)
        cases = [("A", 16), ("B", 128), ("C", 64)]
        queued = []
        for name, tokens in cases:
            rid, future = self.submit(executor, name, tokens)
            queued.append((name, tokens, rid, future))
            self.client.wait_for_load(1, len(queued), blocker)

        self.client.switch_policy("fcfs")
        self.unregister(TARGET)
        removed_at = len(self.records())
        self.client.check_blocker(blocker)
        self.client.wait_for_load(1, 3, blocker)
        self.control(
            "switch",
            {"new_schedule_policy": TARGET},
            success=False,
            message="Unknown schedule policy",
        )
        self.control(
            "unregister",
            {"policy_name": TARGET},
            success=False,
            message="Unknown policy",
        )
        assert self.client.get_state()[1]["schedule_policy"] == "fcfs"

        absent_rid, absent = self.submit(executor, "while-unregistered", 32)
        self.client.wait_for_load(1, 4, blocker)
        self.client.abort(absent_rid)
        meta = absent.result(timeout=120)["meta_info"]
        assert meta["finish_reason"]["type"] == "abort", meta
        self.wait_record(OBSERVER, "disable", absent_rid)
        frozen_counts = self.check_detached(first_generation, since=removed_at)
        assert frozen_counts.get("disable", 0) == 1, (
            "注销策略错误触发了活跃请求的 DISABLE"
        )
        print(f"旧代冻结指标：{frozen_counts}，旧 request_data 大小=0", flush=True)
        print(
            "PASS：带 1 个运行请求和 3 个等待请求注销；旧指标停止、请求数据清空、请求继续处理",
            flush=True,
        )

        self.client.check_blocker(blocker)
        second_generation = self.register(TARGET)
        assert second_generation != first_generation, "重新注册复用了旧策略数据"
        expected_live = {blocker_rid, *(rid for _, _, rid, _ in queued)}
        new_records = [
            item
            for item in self.records()
            if item["policy"] == TARGET
            and item["policy_data"]["generation"] == second_generation
        ]
        initializations = [
            item for item in new_records if item["event"] == "initialize_request"
        ]
        assert {item["rid"] for item in initializations} == expected_live
        assert len(initializations) == 4, "已有请求漏初始化或重复初始化"
        assert all(item["policy_data"]["hook_counts"] == {} for item in initializations)
        assert any(
            item["rid"] == blocker_rid and item["output_tokens"] > 0
            for item in initializations
        ), "没有覆盖正在生成输出的请求补初始化"
        print(
            "补初始化快照："
            + ", ".join(
                f"{item['rid'].removeprefix(self.client.prefix + '-')}"
                f"(已有输出={item['output_tokens']})"
                for item in initializations
            ),
            flush=True,
        )

        self.client.switch_policy(TARGET)
        self.client.check_blocker(blocker)
        self.client.abort(blocker_rid)
        meta = blocker.result(timeout=120)["meta_info"]
        assert meta["finish_reason"]["type"] == "abort", meta
        starts = []
        for name, tokens, rid, future in queued:
            meta = future.result(timeout=120)["meta_info"]
            assert meta["completion_tokens"] == tokens, meta
            assert meta["finish_reason"]["type"] != "abort", meta
            starts.append((meta["forward_entry_time"], name))
            final = self.wait_record(
                TARGET, "disable", rid, generation=second_generation
            )
            assert final["request_data"][rid]["observed_output"] == tokens
        assert len({started for started, _ in starts}) == 3
        order = [name for _, name in sorted(starts)]
        assert order == ["A", "C", "B"], order
        self.client.wait_for_load(0, 0)
        print(
            "PASS：同名重新注册补初始化全部已有请求，切换后实际排序 A → C → B",
            flush=True,
        )

        fresh_rid = self.generate_probe("after-reregister")
        final = self.wait_record(
            TARGET, "disable", fresh_rid, generation=second_generation
        )
        assert final["policy_data"]["request_initializations"] == 5
        assert final["policy_data"]["hook_counts"]["enable"] == 1
        assert final["policy_data"]["hook_counts"]["disable"] == 5
        assert final["request_data"][fresh_rid]["settled_rounds"] == 8
        assert set(final["request_data"]) == {fresh_rid}, "完成请求的数据仍残留"
        new_records = [
            item
            for item in self.records()
            if item["policy"] == TARGET
            and item["policy_data"]["generation"] == second_generation
        ]
        for initialized in initializations:
            first_hook = next(
                item
                for item in new_records
                if item["rid"] == initialized["rid"]
                and item["event"] not in ("initialize_policy", "initialize_request")
            )
            data = first_hook["request_data"][initialized["rid"]]
            assert data["initial_output"] == initialized["output_tokens"], (
                "已有请求重新初始化时没有记录当时的输出进度"
            )
        print(f"新代指标：{final['policy_data']}", flush=True)

        self.client.switch_policy("fcfs")
        self.unregister(TARGET)
        removed_again_at = len(self.records())
        final_rid = self.generate_probe("observer-only")
        observer = self.wait_record(OBSERVER, "disable", final_rid)
        self.check_detached(second_generation, since=removed_again_at)
        assert observer["policy_data"]["generation"] == observer_generation
        assert observer["policy_data"]["request_initializations"] == 8
        assert observer["policy_data"]["hook_counts"]["enable"] == 8
        assert observer["policy_data"]["hook_counts"]["disable"] == 8
        print(f"观察策略指标：{observer['policy_data']}", flush=True)
        print(
            "PASS：新请求初始化、完成清理和第二次注销正确；观察策略状态持续保留",
            flush=True,
        )

        self.unregister(OBSERVER)
        stopped_at = len(self.records())
        self.generate_probe("no-custom-policy")
        assert len(self.records()) == stopped_at, "全部注销后仍有自定义回调"
        print("PASS：全部注销后不再产生自定义事件，FCFS 生成仍正常", flush=True)

    def run(self):
        info, state = self.client.get_state()
        assert info["enable_metrics"] and info["disable_overlap_schedule"]
        assert (
            info["disaggregation_mode"] == "null"
            and info["speculative_algorithm"] is None
        )
        assert state["effective_max_running_requests_per_dp"] == 1, (
            "请使用 start_custom_policy_switch_server.sh"
        )
        assert info["max_queued_requests"] is None or info["max_queued_requests"] >= 4
        self.client.wait_for_load(0, 0)
        original = state["schedule_policy"]
        print(
            "共享指标：注册代次、请求初始化次数、排序次数、各 hook 次数。\n每请求指标：输出预算、初始化时输出数、已结算输出数、结算轮次。",
            flush=True,
        )
        try:
            self.client.switch_policy("fcfs")
            with ThreadPoolExecutor(max_workers=5) as executor:
                try:
                    self.run_lifecycle(executor)
                finally:
                    for rid, future in self.futures.items():
                        if not future.done():
                            self.client.abort(rid)
            print(
                "\nPASS：注销、保护校验、停止更新、请求保留、同名重注册及排序全部通过",
                flush=True,
            )
        finally:
            self.client.switch_policy(original)
            for name in tuple(self.registered):
                self.unregister(name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--server-policy-file", default=str(Path(__file__).resolve()))
    parser.add_argument("--snapshot-file", type=Path, default=SNAPSHOT_FILE)
    parser.add_argument("--blocker-tokens", type=int, default=4096)
    UnregisterVerifier(parser.parse_args()).run()


if __name__ == "__main__":
    main()
