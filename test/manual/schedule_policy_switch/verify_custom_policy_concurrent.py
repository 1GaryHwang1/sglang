"""Verify repeated policy switches under concurrent load using scheduler snapshots.

Run against start_custom_policy_concurrent_server.sh in the same container.
The server imports four rank functions with different parameter lists from here.
Three policies register at startup; the fourth registers during the request stream.
"""

import argparse
import asyncio
import inspect
import json
import math
import os
import time
import uuid
from fractions import Fraction
from pathlib import Path

import aiohttp


def record_rank_inputs(function, waiting_queue, **inputs):
    """Record actual rank arguments next to the scheduler's sorting snapshot."""
    record = {
        "phase": "custom_rank_inputs",
        "function": function,
        "queue_rids": [req.rid for req in waiting_queue],
        "inputs": inputs,
    }
    with open(
        os.environ["SGLANG_SCHEDULE_SWITCH_TRACE"], "a", encoding="utf-8"
    ) as file:
        file.write(json.dumps(record) + "\n")


def running_inputs(running_batch):
    from sglang.srt.managers.schedule_batch import ScheduleBatch

    assert isinstance(running_batch, ScheduleBatch), type(running_batch)
    assert running_batch.batch_size() == len(running_batch.reqs)
    return {
        "running_count": running_batch.batch_size(),
        "running_rids": [req.rid for req in running_batch.reqs],
        "running_output_lengths": [
            req.sampling_params.max_new_tokens for req in running_batch.reqs
        ],
    }


def token_inputs(waiting_queue, processed_tokens):
    assert type(processed_tokens) is int and processed_tokens >= 0
    arrivals = {req.rid: req.arrival_processed_tokens for req in waiting_queue}
    assert all(0 <= arrival <= processed_tokens for arrival in arrivals.values())
    return {"processed_tokens": processed_tokens, "arrival_processed_tokens": arrivals}


def rank_shortest_output_first(waiting_queue):
    waiting_queue.sort(
        key=lambda req: (
            req.sampling_params.max_new_tokens,
            req.time_stats.wait_queue_entry_time,
        )
    )
    record_rank_inputs("rank_shortest_output_first", waiting_queue)


def rank_running_output_first(waiting_queue, running_batch):
    inputs = running_inputs(running_batch)
    running_lengths = set(inputs["running_output_lengths"])
    waiting_queue.sort(
        key=lambda req: (
            req.sampling_params.max_new_tokens not in running_lengths,
            -req.sampling_params.max_new_tokens,
            req.time_stats.wait_queue_entry_time,
        )
    )
    record_rank_inputs("rank_running_output_first", waiting_queue, **inputs)


def rank_token_aged_first(waiting_queue, *, processed_tokens):
    inputs = token_inputs(waiting_queue, processed_tokens)
    waiting_queue.sort(
        key=lambda req: (
            -Fraction(
                processed_tokens - req.arrival_processed_tokens,
                req.sampling_params.max_new_tokens,
            ),
            req.time_stats.wait_queue_entry_time,
        )
    )
    record_rank_inputs("rank_token_aged_first", waiting_queue, **inputs)


def rank_loaded_token_aged_first(waiting_queue, processed_tokens, running_batch):
    inputs = {
        **token_inputs(waiting_queue, processed_tokens),
        **running_inputs(running_batch),
    }
    waiting_queue.sort(
        key=lambda req: (
            -Fraction(
                processed_tokens - req.arrival_processed_tokens,
                req.sampling_params.max_new_tokens + inputs["running_count"],
            ),
            req.time_stats.wait_queue_entry_time,
        )
    )
    record_rank_inputs("rank_loaded_token_aged_first", waiting_queue, **inputs)


def build_actions(run_id, action_count, switch_positions, policies, register_position):
    switches = {
        position: policies[index % len(policies)]
        for index, position in enumerate(switch_positions)
    }
    output_lengths = (128, 256, 512, 1024)
    return [
        {"index": index, "type": "register", "policy": policies[-1]}
        if index == register_position
        else {"index": index, "type": "switch", "policy": switches[index]}
        if index in switches
        else {
            "index": index,
            "type": "generate",
            "rid": f"custom-concurrent-{run_id}-{index:04d}",
            "max_new_tokens": output_lengths[(index - 1) % len(output_lengths)],
        }
        for index in range(1, action_count + 1)
    ]


async def on_request_chunk_sent(session, trace_config_ctx, params):
    sent = trace_config_ctx.trace_request_ctx
    if sent is not None:
        sent.set()


class ConcurrentVerifier:
    def __init__(self, args):
        self.args = args
        self.run_id = uuid.uuid4().hex[:12]
        self.custom_policies = {
            "manual_shortest": "rank_shortest_output_first",
            "manual_running": "rank_running_output_first",
            "manual_token_aged": "rank_token_aged_first",
            "manual_combined": "rank_loaded_token_aged_first",
        }
        self.trace_offset = 0
        self.last_switch_id = -1
        self.tasks = []
        self.request_actions = []
        self.report = {
            "run_id": self.run_id,
            "actions": build_actions(
                self.run_id,
                args.actions,
                args.switch_positions,
                list(self.custom_policies),
                args.register_position,
            ),
            "registrations": [],
            "switches": [],
            "responses": [],
            "peak_observed_running": 0,
            "trace_file": str(args.trace_file),
        }

    async def request_json(self, session, method, path, **kwargs):
        async with session.request(
            method,
            f"{self.args.base_url.rstrip('/')}{path}",
            timeout=aiohttp.ClientTimeout(total=30),
            **kwargs,
        ) as response:
            if response.status >= 400:
                raise RuntimeError(
                    f"{method} {path} 失败：HTTP {response.status} "
                    f"{await response.text()}"
                )
            return await response.json()

    async def get_load(self, session):
        data = await self.request_json(
            session, "GET", "/v1/loads", params={"include": "core"}
        )
        assert len(data["loads"]) == 1, "本测试需要单个 DP 副本"
        load = data["loads"][0]
        self.report["peak_observed_running"] = max(
            self.report["peak_observed_running"], load["num_running_reqs"]
        )
        return load

    async def switch_policy(self, session, policy):
        result = await self.request_json(
            session,
            "POST",
            "/schedule_policy_switch",
            json={"new_schedule_policy": policy},
        )
        assert result["success"] and result["new_schedule_policy"] == policy, result
        print(f"策略切换：{result['old_schedule_policy']} → {policy}", flush=True)
        return result

    async def register_policy(self, session, policy_name, action_index=None):
        function = self.custom_policies[policy_name]
        before = await self.request_json(session, "GET", "/server_info")
        current_policy = before["internal_states"][0]["schedule_policy"]
        result = await self.request_json(
            session,
            "POST",
            "/schedule_policy_register",
            json={
                "policy_name": policy_name,
                "rank_fn_name": function,
                "register_path": self.args.server_policy_file,
            },
        )
        assert result["success"], result
        after = await self.request_json(session, "GET", "/server_info")
        assert after["internal_states"][0]["schedule_policy"] == current_policy, (
            "注册策略不应改变当前策略"
        )
        record = {
            **result,
            "policy_name": policy_name,
            "rank_fn_name": function,
            "parameters": list(inspect.signature(globals()[function]).parameters),
            "action_index": action_index,
            "current_policy": current_policy,
        }
        self.report["registrations"].append(record)
        print(
            f"注册成功：{policy_name} → {function}"
            f"{inspect.signature(globals()[function])}",
            flush=True,
        )
        return record

    async def wait_for_backlog(self, session):
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            load = await self.get_load(session)
            if load["num_running_reqs"] >= 2 and load["num_waiting_reqs"] >= 2:
                return load
            if all(task.done() for task in self.tasks):
                break
            await asyncio.sleep(0.05)
        raise RuntimeError(
            f"切换前未形成并发和等待队列：{load}。请使用配套启动脚本，"
            "或增加首次切换前的请求数；本次不能验证排序正确性。"
        )

    def read_snapshots(self):
        if not self.args.trace_file.exists():
            return []
        with self.args.trace_file.open(encoding="utf-8") as file:
            file.seek(self.trace_offset)
            # Ignore an unfinished final line while the scheduler is appending it.
            lines = file.read().split("\n")[:-1]
        return [json.loads(line) for line in lines if line]

    async def wait_for_first_rank(self, policy):
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            snapshots = self.read_snapshots()
            for index, after in enumerate(snapshots):
                if (
                    after["phase"] == "after_first_rank"
                    and after["switch_id"] > self.last_switch_id
                    and after["new_schedule_policy"] == policy
                ):
                    before = next(
                        record
                        for record in snapshots
                        if record["phase"] == "before_switch"
                        and record["switch_id"] == after["switch_id"]
                    )
                    assert index > 0, "缺少自定义排序的参数记录"
                    rank_inputs = snapshots[index - 1]
                    assert rank_inputs["phase"] == "custom_rank_inputs", rank_inputs
                    self.last_switch_id = after["switch_id"]
                    return before, after, rank_inputs
            await asyncio.sleep(0.05)
        raise TimeoutError(
            f"未找到 {policy} 的首次排序快照：{self.args.trace_file}。"
            "确认客户端能读取服务端的 SGLANG_SCHEDULE_SWITCH_TRACE 文件。"
        )

    def verify_snapshots(self, result, before, after, rank_inputs):
        assert before["policy"] == result["old_schedule_policy"], before
        assert after["policy"] == result["new_schedule_policy"], after
        for snapshot in (before, after):
            queue = snapshot["waiting"]
            assert len(queue) >= 2, "快照不足两条请求，本次排序验证无效"
            ids = [req["rid"] for req in queue]
            assert len(set(ids)) == len(ids), "等待队列出现重复请求"
            assert all(
                rid.startswith(f"custom-concurrent-{self.run_id}-") for rid in ids
            ), "等待队列混入其他请求，请使用独立的测试服务"

        # New arrivals can change membership between snapshots; validate the
        # after snapshot using its own requests instead of assuming identical sets.
        policy = after["policy"]
        function = self.custom_policies[policy]
        queue = after["waiting"]
        assert rank_inputs["function"] == function, rank_inputs
        assert rank_inputs["queue_rids"] == [req["rid"] for req in queue], (
            "参数记录与首次排序快照不属于同一次排序"
        )
        inputs = rank_inputs["inputs"]
        parameters = inspect.signature(globals()[function]).parameters
        if "running_batch" in parameters:
            count = inputs["running_count"]
            assert type(count) is int and count > 0, inputs
            assert (
                count
                == len(inputs["running_rids"])
                == len(inputs["running_output_lengths"])
            )
            assert set(inputs["running_rids"]).isdisjoint(rank_inputs["queue_rids"])
        if "processed_tokens" in parameters:
            processed = inputs["processed_tokens"]
            assert type(processed) is int and processed > 0, inputs
            arrivals = inputs["arrival_processed_tokens"]
            assert set(arrivals) == set(rank_inputs["queue_rids"])
            assert all(0 <= value <= processed for value in arrivals.values())
        if function == "rank_shortest_output_first":
            keys = [(req["max_new_tokens"], req["enqueue_time"]) for req in queue]
        elif function == "rank_running_output_first":
            running_lengths = set(inputs["running_output_lengths"])
            keys = [
                (
                    req["max_new_tokens"] not in running_lengths,
                    -req["max_new_tokens"],
                    req["enqueue_time"],
                )
                for req in queue
            ]
        elif function in ("rank_token_aged_first", "rank_loaded_token_aged_first"):
            extra = inputs.get("running_count", 0)
            keys = [
                (
                    -Fraction(
                        processed - arrivals[req["rid"]], req["max_new_tokens"] + extra
                    ),
                    req["enqueue_time"],
                )
                for req in queue
            ]
        else:
            raise AssertionError(f"未定义预期排序规则：{policy}")
        assert keys == sorted(keys), f"{policy} 首次排序错误：{keys}"

    async def generate(self, session, action, sent):
        record = dict(action)
        try:
            async with session.post(
                f"{self.args.base_url.rstrip('/')}/generate",
                json={
                    "rid": action["rid"],
                    "text": "Please count upwards from one.",
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": action["max_new_tokens"],
                        "ignore_eos": True,
                    },
                },
                trace_request_ctx=sent,
            ) as response:
                response.raise_for_status()
                meta = (await response.json())["meta_info"]
                record["meta_info"] = meta
                assert meta["id"] == action["rid"], meta
                assert meta["completion_tokens"] == action["max_new_tokens"], meta
                assert (meta.get("finish_reason") or {}).get("type") != "abort", meta
                started = meta.get("forward_entry_time", 0)
                assert math.isfinite(started) and started > 0, meta
        except Exception as error:
            record["error"] = repr(error)
        finally:
            sent.set()
        return record

    async def cleanup_requests(self, session):
        async def abort(rid):
            async with session.post(
                f"{self.args.base_url.rstrip('/')}/abort_request",
                json={"rid": rid},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                response.raise_for_status()

        pending = [
            action["rid"]
            for action, task in zip(self.request_actions, self.tasks)
            if not task.done()
        ]
        await asyncio.gather(*(abort(rid) for rid in pending), return_exceptions=True)
        if self.tasks:
            _, unfinished = await asyncio.wait(self.tasks, timeout=30)
            for task in unfinished:
                task.cancel()

    async def run(self):
        trace = aiohttp.TraceConfig()
        trace.on_request_chunk_sent.append(on_request_chunk_sent)
        error = None
        original_policy = None
        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=0),
            timeout=aiohttp.ClientTimeout(total=self.args.request_timeout, connect=30),
            trace_configs=[trace],
        ) as session:
            try:
                info = await self.request_json(session, "GET", "/server_info")
                states = info["internal_states"]
                self.report["overlap_enabled"] = not info["disable_overlap_schedule"]
                assert len(states) == 1, "本测试需要单个 DP 副本"
                assert info["disaggregation_mode"] == "null", "本测试仅覆盖 normal 模式"
                assert info["enable_metrics"], "请使用 --enable-metrics"
                assert not info["enable_priority_scheduling"], "请关闭优先级调度"
                assert states[0]["effective_max_running_requests_per_dp"] > 1, (
                    "这是高并发测试，请勿使用 --max-running-requests 1"
                )
                load = await self.get_load(session)
                assert load["num_running_reqs"] == load["num_waiting_reqs"] == 0, (
                    "请先等待其他请求结束，并使用独立的测试服务"
                )
                original_policy = states[0]["schedule_policy"]
                self.report["initialization"] = await self.switch_policy(
                    session, "fcfs"
                )
                for policy_name in list(self.custom_policies)[:-1]:
                    await self.register_policy(session, policy_name)
                self.trace_offset = (
                    self.args.trace_file.stat().st_size
                    if self.args.trace_file.exists()
                    else 0
                )

                for action in self.report["actions"]:
                    if action["type"] == "generate":
                        sent = asyncio.Event()
                        self.request_actions.append(action)
                        self.tasks.append(
                            asyncio.create_task(self.generate(session, action, sent))
                        )
                        # Body sent does not imply the scheduler has enqueued it.
                        await sent.wait()
                        continue
                    load = await self.wait_for_backlog(session)
                    print(
                        f"第 {action['index']} 个操作：running={load['num_running_reqs']}，"
                        f"waiting={load['num_waiting_reqs']}",
                        flush=True,
                    )
                    if action["type"] == "register":
                        record = await self.register_policy(
                            session, action["policy"], action["index"]
                        )
                        record["load_before_registration"] = load
                        record["load_after_registration"] = await self.get_load(session)
                        assert (
                            record["load_after_registration"]["num_running_reqs"] >= 2
                            and record["load_after_registration"]["num_waiting_reqs"]
                            >= 2
                        ), "注册结束时负载已消失，本次无法验证带负载注册"
                        record["registered_under_load"] = True
                        print("PASS：运行中注册成功，当前策略保持不变", flush=True)
                        continue
                    result = await self.switch_policy(session, action["policy"])
                    before, after, rank_inputs = await self.wait_for_first_rank(
                        action["policy"]
                    )
                    check = {
                        **result,
                        "action_index": action["index"],
                        "load_before_switch": load,
                        "before": before,
                        "after": after,
                        "rank_inputs": rank_inputs,
                    }
                    self.report["switches"].append(check)
                    self.verify_snapshots(result, before, after, rank_inputs)
                    check["order_correct"] = True
                    check["parameters_correct"] = True
                    print(
                        f"PASS：switch_id={after['switch_id']}，"
                        f"waiting {len(before['waiting'])} → {len(after['waiting'])}，"
                        f"首次排序及传参正确，inputs={list(rank_inputs['inputs'])}",
                        flush=True,
                    )

                print(f"发送完毕，等待 {len(self.tasks)} 条推理请求完成", flush=True)
                await asyncio.gather(*self.tasks)
                failures = [
                    task.result() for task in self.tasks if "error" in task.result()
                ]
                assert not failures, f"{len(failures)} 条推理请求失败，详见结果文件"
                assert self.report["peak_observed_running"] >= 2, "未观察到实际并发"
                assert len(self.report["registrations"]) == len(self.custom_policies)
                assert (
                    sum(
                        record.get("registered_under_load", False)
                        for record in self.report["registrations"]
                    )
                    == 1
                ), "未验证运行中的策略注册"
            except Exception as caught:
                error = caught
                self.report["error"] = repr(caught)
                await self.cleanup_requests(session)
            finally:
                values = await asyncio.gather(*self.tasks, return_exceptions=True)
                self.report["responses"] = [
                    value
                    if isinstance(value, dict)
                    else {**action, "error": repr(value)}
                    for action, value in zip(self.request_actions, values)
                ]
                if original_policy is not None:
                    try:
                        await self.switch_policy(session, original_policy)
                    except Exception as caught:
                        self.report["restore_error"] = repr(caught)
                        error = error or caught

        output = self.args.output or Path(f"/tmp/custom-concurrent-{self.run_id}.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"注册策略数={len(self.report['registrations'])}，"
            f"操作数={len(self.report['actions'])}，推理请求={len(self.tasks)}，"
            f"排序验证通过次数={sum(check.get('order_correct', False) for check in self.report['switches'])}，"
            f"传参验证通过次数={sum(check.get('parameters_correct', False) for check in self.report['switches'])}，"
            f"观测到的最大 running={self.report['peak_observed_running']}",
            flush=True,
        )
        print(f"结果文件：{output}", flush=True)
        if error is not None:
            raise RuntimeError(f"测试失败，详情见 {output}：{error}") from error
        print(
            "PASS：运行中注册成功，高并发请求完成，全部切换的首次排序及传参正确",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--actions", type=int, default=200)
    parser.add_argument(
        "--switch-positions",
        type=int,
        nargs="+",
        default=[32, 56, 80, 104, 128, 152, 176, 200],
    )
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument(
        "--register-position",
        type=int,
        default=100,
        help="第四个策略的注册位置，必须早于第四次切换，默认第 100 个操作",
    )
    parser.add_argument(
        "--server-policy-file",
        default=str(Path(__file__).resolve()),
        help="服务端可读取的本脚本路径，内含四种参数签名的排序函数",
    )
    parser.add_argument(
        "--trace-file",
        type=Path,
        default=Path(
            os.environ.get(
                "SGLANG_SCHEDULE_SWITCH_TRACE",
                "/tmp/sglang-custom-policy-concurrent.jsonl",
            )
        ),
        help="客户端可读取的服务端快照文件路径",
    )
    parser.add_argument("--output", type=Path, help="结果 JSON 路径，默认写入 /tmp")
    args = parser.parse_args()
    if (
        args.actions <= len(args.switch_positions)
        or len(args.switch_positions) < 4
        or args.switch_positions != sorted(set(args.switch_positions))
        or not all(1 < position <= args.actions for position in args.switch_positions)
        or args.request_timeout <= 0
    ):
        parser.error("操作数和超时必须有效，至少提供四个递增且不重复的切换位置")
    if (
        not 1 < args.register_position < args.switch_positions[3]
        or args.register_position in args.switch_positions
    ):
        parser.error("注册位置必须早于第四次切换、晚于首个操作，且不能与切换位置重复")
    asyncio.run(ConcurrentVerifier(args).run())


if __name__ == "__main__":
    main()
