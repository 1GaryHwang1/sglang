"""Verify per-policy request data on a dedicated, single-running-request server.

Start the server with start_custom_policy_switch_server.sh, then run this script
in the same container. Use --server-policy-file when the server sees a different
path. The server imports the initializer and rank function from this file.
The fixed policy name requires a fresh server for each run.

This test uses short prompts without chunking. At each rank, request_data must
contain exactly the unfinished waiting and running requests. Later ranks and
probe requests therefore also detect data left behind after completion/abort.
Retraction and PD-specific queues are outside this test's scope.
"""

from __future__ import annotations

import argparse
import math
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import requests

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch


POLICY_NAME = "manual_request_data"
CASES = [("A", 16), ("B", 128), ("C", 64)]


@dataclass
class RequestData:
    rid: str
    output_budget: int
    enqueue_time: float


def initialize_request(req: Req) -> RequestData:
    enqueue_time = req.time_stats.wait_queue_entry_time
    assert math.isfinite(enqueue_time) and enqueue_time > 0, (
        f"{req.rid}: 初始化前尚未记录入队时间"
    )
    return RequestData(
        rid=req.rid,
        output_budget=req.sampling_params.max_new_tokens,
        enqueue_time=enqueue_time,
    )


def enable(req: Req, request_data: dict[str, RequestData]) -> None:
    if req.rid not in request_data:
        request_data[req.rid] = initialize_request(req)


def rank_by_request_data(
    waiting_queue: list[Req],
    request_data: dict[str, RequestData],
    running_batch: ScheduleBatch,
) -> None:
    """Check initialization/cleanup, then sort using only the stored data."""
    live_reqs = [
        req for req in [*waiting_queue, *running_batch.reqs] if not req.finished()
    ]
    live_rids = {req.rid for req in live_reqs}
    missing = live_rids - request_data.keys()
    stale = request_data.keys() - live_rids
    assert not missing, f"请求缺少初始化数据：{sorted(missing)}"
    assert not stale, f"结束或取消后仍残留请求数据：{sorted(stale)}"

    for req in live_reqs:
        data = request_data[req.rid]
        assert isinstance(data, RequestData), (req.rid, type(data))
        assert data.rid == req.rid, (req.rid, data)
        assert data.output_budget == req.sampling_params.max_new_tokens, (
            req.rid,
            data,
        )

    waiting_queue.sort(
        key=lambda req: (
            request_data[req.rid].output_budget,
            request_data[req.rid].enqueue_time,
        )
    )


class CustomPolicyDataVerifier:
    def __init__(
        self, base_url: str, server_policy_file: str, blocker_tokens: int
    ) -> None:
        # Only the client needs these helpers. The server can import the policy
        # functions without adding this directory to its module search path.
        from verify_custom_policy_switch import CustomPolicySwitchVerifier

        self.client = CustomPolicySwitchVerifier(
            base_url, server_policy_file, blocker_tokens
        )
        self.client.prefix = f"request-data-{self.client.prefix}"

    def register_policy(self) -> None:
        result = self.client.request_json(
            "POST",
            "/schedule_policy_register",
            json={
                "policy_name": POLICY_NAME,
                "register_path": self.client.server_policy_file,
                "rank_fn_name": "rank_by_request_data",
            },
        )
        assert result["success"], result
        _, state = self.client.get_state()
        assert state["schedule_policy"] == "fcfs", "注册策略不应自动切换当前策略"
        print(f"注册成功：{POLICY_NAME}（包含初始化函数）", flush=True)

    def verify_round(
        self,
        number: int,
        description: str,
        *,
        register_when_queued: bool = False,
        cancel_waiting: bool = False,
    ) -> None:
        client = self.client
        client.wait_for_load(0, 0)
        print(f"\n第 {number} 轮：{description}", flush=True)
        futures: dict[str, Future[dict[str, Any]]] = {}
        blocker_id = f"{client.prefix}-{number}-blocker"

        with ThreadPoolExecutor(max_workers=5) as executor:
            try:
                blocker = executor.submit(
                    client.generate, blocker_id, client.blocker_tokens
                )
                futures[blocker_id] = blocker
                client.wait_for_load(1, 0, blocker)
                print("占位成功：running=1", flush=True)

                test_futures = []
                for name, tokens in CASES:
                    rid = f"{client.prefix}-{number}-{name}"
                    future = executor.submit(client.generate, rid, tokens)
                    futures[rid] = future
                    test_futures.append(future)
                    client.wait_for_load(1, len(test_futures), blocker)
                    print(
                        f"{name} 已入队：max_new_tokens={tokens}，"
                        f"waiting={len(test_futures)}",
                        flush=True,
                    )

                if register_when_queued:
                    self.register_policy()

                if cancel_waiting:
                    cancel_id = f"{client.prefix}-{number}-cancel"
                    cancelled = executor.submit(client.generate, cancel_id, 32)
                    futures[cancel_id] = cancelled
                    client.wait_for_load(1, 4, blocker)
                    client.abort(cancel_id)
                    meta = cancelled.result(timeout=120)["meta_info"]
                    assert meta["finish_reason"]["type"] == "abort", meta
                    client.wait_for_load(1, 3, blocker)
                    print("额外等待请求已取消，随后检查其数据是否清理", flush=True)

                client.switch_policy(POLICY_NAME)
                client.check_blocker(blocker)
                client.abort(blocker_id)
                meta = blocker.result(timeout=120)["meta_info"]
                assert meta["finish_reason"]["type"] == "abort", meta

                records = []
                for (name, tokens), future in zip(CASES, test_futures):
                    meta = future.result(timeout=120)["meta_info"]
                    assert meta["completion_tokens"] == tokens, meta
                    assert meta["finish_reason"]["type"] != "abort", meta
                    started_at = meta.get("forward_entry_time")
                    assert (
                        isinstance(started_at, (int, float))
                        and math.isfinite(started_at)
                        and started_at > 0
                    ), f"{name} 缺少有效的 forward_entry_time：{meta}"
                    records.append((started_at, name))
                    print(f"{name}: 开始执行时间={started_at:.6f}", flush=True)

                assert len({started_at for started_at, _ in records}) == len(CASES), (
                    "请求开始时间相同，无法判断顺序；请使用配套启动脚本"
                )
                actual_order = [name for _, name in sorted(records)]
                expected_order = ["A", "C", "B"]
                print("预期顺序：", " → ".join(expected_order), flush=True)
                print("实际顺序：", " → ".join(actual_order), flush=True)
                assert actual_order == expected_order, actual_order
            finally:
                for rid, future in futures.items():
                    if not future.done() and not future.cancel():
                        try:
                            client.abort(rid)
                        except requests.RequestException as error:
                            print(f"清理请求 {rid} 失败：{error}", flush=True)

        client.wait_for_load(0, 0)
        # Trigger another rank after the last completion, so even the last
        # request's data must have been removed from the policy's dictionary.
        probe = client.generate(f"{client.prefix}-{number}-probe", 8)
        meta = probe["meta_info"]
        assert meta["completion_tokens"] == 8, meta
        assert meta["finish_reason"]["type"] != "abort", meta
        client.wait_for_load(0, 0)
        print(f"PASS：第 {number} 轮初始化、排序及数据清理检查通过", flush=True)

    def run(self) -> None:
        client = self.client
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
            self.verify_round(
                1,
                "请求入队后注册，验证已有 waiting/running 请求补初始化",
                register_when_queued=True,
            )
            self.verify_round(2, "自定义策略激活时，验证后续新请求初始化")
            client.switch_policy("fcfs")
            self.verify_round(
                3,
                "自定义策略未激活时初始化新请求，并取消一条等待请求",
                cancel_waiting=True,
            )
            print(
                "\nPASS：已有请求、新请求、未激活策略初始化及完成/取消清理全部通过",
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
    args = parser.parse_args()
    if args.blocker_tokens <= 0:
        parser.error("--blocker-tokens 必须大于 0")
    CustomPolicyDataVerifier(
        args.base_url, args.server_policy_file, args.blocker_tokens
    ).run()


if __name__ == "__main__":
    main()
