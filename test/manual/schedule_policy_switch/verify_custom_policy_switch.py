"""Verify runtime registration and scheduling order on a dedicated test server.

Run in the server's container, or set --server-policy-file to this file's path
inside the container. The server imports rank_shortest_output_first from here.
"""

import argparse
import math
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests


def rank_shortest_output_first(waiting_queue):
    """Example custom policy: shorter requested outputs run first."""
    waiting_queue.sort(
        key=lambda req: (
            req.sampling_params.max_new_tokens,
            req.time_stats.wait_queue_entry_time,
        )
    )


class CustomPolicySwitchVerifier:
    def __init__(self, base_url, server_policy_file, blocker_tokens):
        self.base_url = base_url.rstrip("/")
        self.server_policy_file = server_policy_file
        self.blocker_tokens = blocker_tokens
        self.prefix = f"custom-switch-{uuid.uuid4().hex[:12]}"
        self.policy_name = "manual_shortest_output"

    def request_json(self, method, path, *, timeout=10, **kwargs):
        response = requests.request(
            method, f"{self.base_url}{path}", timeout=timeout, **kwargs
        )
        if not response.ok:
            raise RuntimeError(
                f"{method} {path} 失败：HTTP {response.status_code} {response.text}"
            )
        return response.json()

    def get_state(self):
        info = self.request_json("GET", "/server_info")
        states = info["internal_states"]
        assert len(states) == 1, "本测试需要单个 DP 副本"
        return info, states[0]

    def get_load(self):
        data = self.request_json("GET", "/v1/loads", params={"include": "core"})
        assert len(data["loads"]) == 1, "本测试需要单个 DP 副本"
        return data["loads"][0]

    @staticmethod
    def check_blocker(blocker):
        if blocker.done():
            blocker.result()
            raise RuntimeError(
                "占位请求提前结束，本轮测试无效；请增大 --blocker-tokens 后重试"
            )

    def wait_for_load(self, running, waiting, blocker=None):
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if blocker is not None:
                self.check_blocker(blocker)
            load = self.get_load()
            if (
                load["num_running_reqs"] == running
                and load["num_waiting_reqs"] == waiting
            ):
                return
            time.sleep(0.05)
        raise TimeoutError(
            f"等待 running={running}、waiting={waiting} 超时，最后一次状态：{load}"
        )

    def switch_policy(self, name):
        result = self.request_json(
            "POST", "/schedule_policy_switch", json={"new_schedule_policy": name}
        )
        assert result["success"], result
        assert result["new_schedule_policy"] == name, result
        _, state = self.get_state()
        assert state["schedule_policy"] == name, state["schedule_policy"]
        print(
            f"策略切换：{result['old_schedule_policy']} → {name}",
            flush=True,
        )

    def generate(self, rid, tokens):
        return self.request_json(
            "POST",
            "/generate",
            timeout=120,
            json={
                "rid": rid,
                "text": "Please count upwards from one.",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": tokens,
                    "ignore_eos": True,
                },
            },
        )

    def abort(self, rid):
        response = requests.post(
            f"{self.base_url}/abort_request", json={"rid": rid}, timeout=10
        )
        response.raise_for_status()

    def verify_round(self, number, target_policy, expected_order):
        self.wait_for_load(0, 0)
        print(f"\n第 {number} 轮：切换到 {target_policy}", flush=True)
        futures = {}
        cases = [("A", 16), ("B", 128), ("C", 64)]
        blocker_id = f"{self.prefix}-{number}-blocker"
        with ThreadPoolExecutor(max_workers=4) as executor:
            try:
                blocker = executor.submit(
                    self.generate, blocker_id, self.blocker_tokens
                )
                futures[blocker_id] = blocker
                self.wait_for_load(1, 0, blocker)
                print("占位成功：running=1", flush=True)

                test_futures = []
                for name, tokens in cases:
                    rid = f"{self.prefix}-{number}-{name}"
                    future = executor.submit(self.generate, rid, tokens)
                    futures[rid] = future
                    test_futures.append(future)
                    self.wait_for_load(1, len(test_futures), blocker)
                    print(
                        f"{name} 已入队：max_new_tokens={tokens}，"
                        f"waiting={len(test_futures)}",
                        flush=True,
                    )

                self.switch_policy(target_policy)
                self.check_blocker(blocker)
                self.abort(blocker_id)
                blocker.result(timeout=120)

                records = []
                for (name, tokens), future in zip(cases, test_futures):
                    meta = future.result(timeout=120)["meta_info"]
                    assert meta["completion_tokens"] == tokens, meta
                    started_at = meta.get("forward_entry_time")
                    assert (
                        isinstance(started_at, (int, float))
                        and math.isfinite(started_at)
                        and started_at > 0
                    ), f"{name} 缺少有效的 forward_entry_time：{meta}"
                    records.append((started_at, name))
                    print(f"{name}: 开始执行时间={started_at:.6f}", flush=True)

                assert len({record[0] for record in records}) == len(cases), (
                    "请求开始时间相同，无法判断先后顺序；请使用配套启动脚本"
                )
                actual_order = [name for _, name in sorted(records)]
                print("预期顺序：", " → ".join(expected_order), flush=True)
                print("实际顺序：", " → ".join(actual_order), flush=True)
                assert actual_order == expected_order, (
                    f"调度顺序错误：预期 {expected_order}，实际 {actual_order}"
                )
                print(f"PASS：第 {number} 轮实际执行顺序正确", flush=True)
            finally:
                # Abort only this round's unfinished requests before joining threads.
                for rid, future in futures.items():
                    if not future.done() and not future.cancel():
                        try:
                            self.abort(rid)
                        except requests.RequestException as error:
                            print(f"清理请求 {rid} 失败：{error}", flush=True)

    def run(self):
        info, state = self.get_state()
        assert info["enable_metrics"], "请使用 --enable-metrics"
        assert info["disaggregation_mode"] == "null", "本测试仅覆盖 normal 模式"
        assert state["effective_max_running_requests_per_dp"] == 1, (
            "请使用 --max-running-requests 1，或使用配套启动脚本"
        )
        load = self.get_load()
        assert load["num_running_reqs"] == load["num_waiting_reqs"] == 0, (
            "请先等待其他请求结束，并使用独立的测试服务"
        )
        original_policy = state["schedule_policy"]
        try:
            self.switch_policy("fcfs")
            result = self.request_json(
                "POST",
                "/schedule_policy_register",
                json={
                    "policy_name": self.policy_name,
                    "func_name": "rank_shortest_output_first",
                    "register_path": self.server_policy_file,
                },
            )
            assert result["success"], result
            _, registered_state = self.get_state()
            assert registered_state["schedule_policy"] == "fcfs", (
                "注册策略不应自动切换当前策略"
            )
            print(f"注册成功：{self.policy_name}", flush=True)

            self.verify_round(1, self.policy_name, ["A", "C", "B"])
            self.verify_round(2, "fcfs", ["A", "B", "C"])
            # Reuse the original registration to detect registry loss on switches.
            self.verify_round(3, self.policy_name, ["A", "C", "B"])
            self.wait_for_load(0, 0)
            print(
                "\nPASS：注册、首次切换、切回 FCFS、再次切回自定义策略全部通过",
                flush=True,
            )
        finally:
            self.switch_policy(original_policy)


def main():
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
    CustomPolicySwitchVerifier(
        args.base_url, args.server_policy_file, args.blocker_tokens
    ).run()


if __name__ == "__main__":
    main()
