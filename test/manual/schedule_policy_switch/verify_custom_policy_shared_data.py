"""Verify shared data owned by each registered policy in one Scheduler process.

Start a fresh server with start_custom_policy_switch_server.sh, then run:
    python test/manual/schedule_policy_switch/verify_custom_policy_shared_data.py

Run in the server's container so the client can read the snapshots in /tmp.
For separate filesystems, share the server's snapshot file and pass its client
path with --snapshot-file, plus the server's script path with --server-policy-file.
The server imports this file and shared_data_longest_policy.py beside it.
Both export fixed initializer and hook names. Fixed policy names
require a fresh server for each run. Short prompts avoid chunking/retraction;
PD-specific queues and sharing across Scheduler processes are not covered.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import requests

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.scheduler_components.schedule_events import (
        SchedulingEngineView,
    )


SHORTEST_POLICY = "manual_shared_shortest"
LONGEST_POLICY = "manual_shared_longest"
SNAPSHOT_FILE = Path("/tmp/sglang-shared-policy-data.jsonl")


@dataclass
class PolicyData:
    label: str
    output_sign: int
    request_initializations: int = 0
    rank_calls: int = 0
    # Test-only history survives request cleanup to detect lost/shared state.
    initialized_rids: set[str] = field(default_factory=set)
    ranked_rounds: set[int] = field(default_factory=set)


@dataclass
class RequestData:
    output_budget: int
    shared: PolicyData


# Each registration imports a separate module. Keep the factory's actual return
# value here to check that request initialization and ranking receive that object.
_created_data: PolicyData | None = None
_policy_initializations = 0


def _initialize_policy(label: str, output_sign: int) -> PolicyData:
    global _created_data, _policy_initializations
    _policy_initializations += 1
    assert _policy_initializations == 1, "策略共享数据被重复初始化"
    _created_data = PolicyData(label=label, output_sign=output_sign)
    return _created_data


def initialize_policy() -> PolicyData:
    return _initialize_policy("shortest", 1)


def initialize_request(req: Req, policy_data: PolicyData) -> RequestData:
    assert policy_data is _created_data, "请求初始化收到的共享对象不正确"
    assert req.rid not in policy_data.initialized_rids, f"重复初始化：{req.rid}"
    policy_data.initialized_rids.add(req.rid)
    policy_data.request_initializations += 1
    return RequestData(req.sampling_params.max_new_tokens, policy_data)


def initialize_request_keyword(req: Req, *, policy_data: PolicyData) -> RequestData:
    """Exercise keyword-only shared-data injection as well."""
    return initialize_request(req, policy_data)


def enable(
    req: Req,
    request_data: dict[str, RequestData],
    policy_data: PolicyData,
) -> None:
    if req.rid not in request_data:
        request_data[req.rid] = initialize_request(req, policy_data)


def _record_snapshot(
    number: int,
    waiting_queue: list[Req],
    request_data: dict[str, RequestData],
    policy_data: PolicyData,
    engine_view: SchedulingEngineView,
) -> None:
    """Send actual Scheduler-process values to the manual test's client output."""
    snapshot = {
        "round": number,
        "policy": (
            SHORTEST_POLICY if policy_data.label == "shortest" else LONGEST_POLICY
        ),
        "sample_rid": waiting_queue[0].rid,
        "policy_initializations": _policy_initializations,
        "policy_data": {
            "object_id": hex(id(policy_data)),
            "label": policy_data.label,
            "output_sign": policy_data.output_sign,
            "request_initializations": policy_data.request_initializations,
            "rank_calls": policy_data.rank_calls,
            "initialized_rids": sorted(policy_data.initialized_rids),
            "ranked_rounds": sorted(policy_data.ranked_rounds),
        },
        "request_data": [
            {
                "rid": req.rid,
                "object_id": hex(id(request_data[req.rid])),
                "output_budget": request_data[req.rid].output_budget,
                "shared_object_id": hex(id(request_data[req.rid].shared)),
                "enqueue_time": req.time_stats.wait_queue_entry_time,
            }
            for req in waiting_queue
        ],
        "running_rids": [req.rid for req in engine_view.running_reqs],
    }
    with SNAPSHOT_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(snapshot, ensure_ascii=False) + "\n")


def _rank(
    waiting_queue: list[Req],
    request_data: dict[str, RequestData],
    policy_data: PolicyData,
    engine_view: SchedulingEngineView,
    expected_label: str,
) -> None:
    assert _policy_initializations == 1, "共享数据初始化次数不是一次"
    assert policy_data is _created_data, "排序收到的共享对象被替换或与其他策略串用"
    assert policy_data.label == expected_label, "不同策略的共享数据串用"
    assert policy_data.request_initializations == len(policy_data.initialized_rids)

    live_reqs = [
        req for req in [*waiting_queue, *engine_view.running_reqs] if not req.finished()
    ]
    assert set(request_data) == {req.rid for req in live_reqs}, (
        "请求数据缺失或在完成/取消后未清理"
    )
    assert len({id(data) for data in request_data.values()}) == len(request_data), (
        "不同请求复用了同一个 RequestData 对象"
    )
    for req in live_reqs:
        data = request_data[req.rid]
        assert data.shared is policy_data, "同策略的请求没有引用同一个共享对象"
        assert data.output_budget == req.sampling_params.max_new_tokens
        assert req.rid in policy_data.initialized_rids

    for suffix in ("-backfill-running", "-backfill-waiting"):
        assert any(rid.endswith(suffix) for rid in policy_data.initialized_rids), (
            f"注册时遗漏已有请求：{suffix}"
        )

    new_rounds = set()
    for req in waiting_queue:
        if not req.rid.endswith(("-A", "-B", "-C")):
            continue
        number = int(req.rid.rsplit("-", 2)[1])
        # Every registered policy must initialize requests while inactive, too.
        for previous in range(1, number):
            for name in ("A", "B", "C"):
                suffix = f"-{previous}-{name}"
                assert any(
                    rid.endswith(suffix) for rid in policy_data.initialized_rids
                ), f"策略未激活时遗漏请求初始化：{suffix}"
        previous_rank = {("shortest", 3): 1, ("longest", 4): 2}.get(
            (expected_label, number)
        )
        if previous_rank is not None:
            assert policy_data.rank_calls > 0
            assert previous_rank in policy_data.ranked_rounds, (
                "切回策略后丢失之前的排序状态"
            )
        if number not in policy_data.ranked_rounds:
            new_rounds.add(number)
        policy_data.ranked_rounds.add(number)

    policy_data.rank_calls += 1
    # The shared object's configuration must actually affect scheduling order.
    waiting_queue.sort(
        key=lambda req: (
            policy_data.output_sign * request_data[req.rid].output_budget,
            req.time_stats.wait_queue_entry_time,
        )
    )
    for number in sorted(new_rounds):
        _record_snapshot(number, waiting_queue, request_data, policy_data, engine_view)


def rank_shortest(
    waiting_queue: list[Req],
    policy_data: PolicyData,
    request_data: dict[str, RequestData],
    engine_view: SchedulingEngineView,
) -> None:
    _rank(waiting_queue, request_data, policy_data, engine_view, "shortest")


def rank_longest(
    waiting_queue: list[Req],
    *,
    request_data: dict[str, RequestData],
    engine_view: SchedulingEngineView,
    policy_data: PolicyData,
) -> None:
    _rank(waiting_queue, request_data, policy_data, engine_view, "longest")


class SharedPolicyDataVerifier:
    def __init__(
        self,
        base_url: str,
        server_policy_file: str,
        blocker_tokens: int,
        snapshot_file: Path,
    ):
        # The server imports this file without needing client helpers on sys.path.
        from verify_custom_policy_switch import CustomPolicySwitchVerifier

        self.client = CustomPolicySwitchVerifier(
            base_url, server_policy_file, blocker_tokens
        )
        self.client.prefix = f"shared-data-{self.client.prefix}"
        self.snapshot_file = snapshot_file
        self.shared_object_ids: dict[str, str] = {}

    @staticmethod
    def print_data_model() -> None:
        print(
            "\n本测试使用的数据（共享范围是单个 Scheduler 进程内的单个策略）：\n"
            "  框架提供：waiting_queue、engine_view、req。\n"
            "    engine_view.running_reqs：本轮规划中的运行请求。\n"
            "    req.sampling_params.max_new_tokens：初始化请求的输出预算。\n"
            "    req.time_stats.wait_queue_entry_time：排序分数相同时按入队时间排序。\n"
            "  policy_data：每个策略独立一份 PolicyData，供本策略全部请求共享。\n"
            "    label：策略标识；output_sign：1=短输出优先，-1=长输出优先。\n"
            "    request_initializations：请求初始化累计次数；rank_calls：排序累计次数。\n"
            "    initialized_rids：已初始化请求的历史；ranked_rounds：参与排序的测试轮次。\n"
            "  request_data：每个策略独立一张表，按 rid 保存每条请求的 RequestData。\n"
            "    output_budget：该请求独有的输出预算，从 max_new_tokens 复制。\n"
            "    shared：引用本策略的 policy_data，所有请求都指向同一个对象。\n"
            "  初始化函数接收 req、policy_data；排序函数接收 waiting_queue、\n"
            "    engine_view、policy_data、request_data。\n"
            "  对象地址来自 Scheduler 进程，用于比较共享对象及同时存活的请求对象。",
            flush=True,
        )

    def print_round_snapshot(self, number: int, policy_name: str) -> None:
        assert self.snapshot_file.is_file(), (
            f"无法读取服务端快照 {self.snapshot_file}；请在服务端容器运行测试，"
            "或用 --snapshot-file 指定共享文件在客户端的路径"
        )
        snapshots = [
            json.loads(line)
            for line in self.snapshot_file.read_text(encoding="utf-8").splitlines()
            if line
        ]
        snapshot = next(
            (
                item
                for item in snapshots
                if item["round"] == number
                and item["policy"] == policy_name
                and item["sample_rid"].startswith(f"{self.client.prefix}-")
            ),
            None,
        )
        assert snapshot is not None, f"缺少第 {number} 轮排序数据快照"
        data = snapshot["policy_data"]
        object_id = data["object_id"]
        previous_id = self.shared_object_ids.get(policy_name)
        if previous_id is not None:
            assert object_id == previous_id, "切回策略后共享对象发生变化"
        else:
            assert object_id not in self.shared_object_ids.values(), (
                "不同策略共用了同一个共享对象"
            )
            self.shared_object_ids[policy_name] = object_id

        def short_rid(rid: str) -> str:
            return rid.removeprefix(f"{self.client.prefix}-")

        history = [short_rid(rid) for rid in data["initialized_rids"]]
        running = [short_rid(rid) for rid in snapshot["running_rids"]]
        print(
            f"\n第 {number} 轮首次排序时的数据快照：{policy_name}\n"
            f"  policy_data（本策略一份）：PolicyData@{object_id}\n"
            f"    label={data['label']}，output_sign={data['output_sign']}，"
            f"共享数据初始化次数={snapshot['policy_initializations']}\n"
            f"    request_initializations={data['request_initializations']}，"
            f"rank_calls={data['rank_calls']}，ranked_rounds={data['ranked_rounds']}\n"
            f"    initialized_rids（含已完成/取消请求的历史）={history}\n"
            "  request_data（本次排序的 waiting 请求，每请求各一份）：",
            flush=True,
        )
        for request in snapshot["request_data"]:
            print(
                f"    {short_rid(request['rid'])}: "
                f"RequestData@{request['object_id']}，"
                f"output_budget={request['output_budget']}，"
                f"shared→PolicyData@{request['shared_object_id']}，"
                f"入队时间={request['enqueue_time']:.6f}",
                flush=True,
            )
        print(
            f"  engine_view.running_reqs（框架数据）：{running}\n"
            "  排序键：output_sign × output_budget，然后按入队时间排序。",
            flush=True,
        )
        if previous_id is not None:
            print(f"  PASS：共享对象地址与上次一致：{previous_id}", flush=True)

    def register_policies_with_existing_requests(self) -> None:
        client = self.client
        futures = {}
        with ThreadPoolExecutor(max_workers=2) as executor:
            try:
                running_id = f"{client.prefix}-backfill-running"
                blocker = executor.submit(
                    client.generate, running_id, client.blocker_tokens
                )
                futures[running_id] = blocker
                client.wait_for_load(1, 0, blocker)

                waiting_id = f"{client.prefix}-backfill-waiting"
                futures[waiting_id] = executor.submit(client.generate, waiting_id, 16)
                client.wait_for_load(1, 1, blocker)

                for name, rank, policy_file in (
                    (
                        SHORTEST_POLICY,
                        "rank_shortest",
                        client.server_policy_file,
                    ),
                    (
                        LONGEST_POLICY,
                        "rank_longest",
                        str(
                            Path(client.server_policy_file).with_name(
                                "shared_data_longest_policy.py"
                            )
                        ),
                    ),
                ):
                    client.check_blocker(blocker)
                    result = client.request_json(
                        "POST",
                        "/schedule_policy_register",
                        json={
                            "policy_name": name,
                            "register_path": policy_file,
                            "rank_fn_name": rank,
                        },
                    )
                    assert result["success"], result
                    _, state = client.get_state()
                    assert state["schedule_policy"] == "fcfs", (
                        "注册策略不应改变当前策略"
                    )
                    print(
                        f"注册成功：{name}（独立共享对象，补初始化已有请求）",
                        flush=True,
                    )

                client.check_blocker(blocker)
                # Cancel the waiting request before freeing the running slot.
                client.abort(waiting_id)
                meta = futures[waiting_id].result(timeout=120)["meta_info"]
                assert meta["finish_reason"]["type"] == "abort", meta
                client.wait_for_load(1, 0, blocker)
                client.abort(running_id)
                meta = blocker.result(timeout=120)["meta_info"]
                assert meta["finish_reason"]["type"] == "abort", meta
            finally:
                for rid, future in futures.items():
                    if not future.done() and not future.cancel():
                        try:
                            client.abort(rid)
                        except requests.RequestException as error:
                            print(f"清理请求 {rid} 失败：{error}", flush=True)
        client.wait_for_load(0, 0)

    def run(self) -> None:
        client = self.client
        self.print_data_model()
        info, state = client.get_state()
        assert info["enable_metrics"], "请使用 --enable-metrics"
        assert info["disaggregation_mode"] == "null", "本测试仅覆盖非 PD 模式"
        assert state["effective_max_running_requests_per_dp"] == 1, (
            "请使用 start_custom_policy_switch_server.sh 启动服务"
        )
        load = client.get_load()
        assert load["num_running_reqs"] == load["num_waiting_reqs"] == 0, (
            "请使用独立测试服务，并先等待其他请求结束"
        )
        original_policy = state["schedule_policy"]
        try:
            client.switch_policy("fcfs")
            self.register_policies_with_existing_requests()
            client.verify_round(1, SHORTEST_POLICY, ["A", "C", "B"])
            self.print_round_snapshot(1, SHORTEST_POLICY)
            client.verify_round(2, LONGEST_POLICY, ["B", "C", "A"])
            self.print_round_snapshot(2, LONGEST_POLICY)
            client.switch_policy("fcfs")
            print("两种自定义策略均未激活，继续接收新请求", flush=True)
            client.verify_round(3, SHORTEST_POLICY, ["A", "C", "B"])
            self.print_round_snapshot(3, SHORTEST_POLICY)
            client.verify_round(4, LONGEST_POLICY, ["B", "C", "A"])
            self.print_round_snapshot(4, LONGEST_POLICY)

            # A final rank for each policy also checks cleanup of the last request.
            for name in (SHORTEST_POLICY, LONGEST_POLICY):
                client.switch_policy(name)
                probe = client.generate(f"{client.prefix}-{name}-probe", 8)
                meta = probe["meta_info"]
                assert meta["completion_tokens"] == 8, meta
                assert meta["finish_reason"]["type"] != "abort", meta
                client.wait_for_load(0, 0)
            print(
                "\nPASS：共享数据初始化一次、策略间隔离、请求间共享、"
                "未激活时初始化新请求、切换后保留及请求数据清理全部通过",
                flush=True,
            )
        finally:
            client.switch_policy(original_policy)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument(
        "--server-policy-file",
        default=str(Path(__file__).resolve()),
        help="服务端可读取的本脚本路径；在同一容器运行时可省略",
    )
    parser.add_argument("--blocker-tokens", type=int, default=4096)
    parser.add_argument(
        "--snapshot-file",
        type=Path,
        default=SNAPSHOT_FILE,
        help="服务端 /tmp/sglang-shared-policy-data.jsonl 在客户端可读取的路径",
    )
    args = parser.parse_args()
    if args.blocker_tokens <= 0:
        parser.error("--blocker-tokens 必须大于 0")
    SharedPolicyDataVerifier(
        args.base_url,
        args.server_policy_file,
        args.blocker_tokens,
        args.snapshot_file,
    ).run()


if __name__ == "__main__":
    main()
