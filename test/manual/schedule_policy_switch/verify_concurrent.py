"""Send 200 operations, with a policy switch as operation 100."""

import asyncio
import json
import time
import uuid
from pathlib import Path

import aiohttp

BASE_URL = "http://127.0.0.1:30000"
ACTION_COUNT = 200
SWITCH_POSITION = 100


def build_actions(run_id):
    output_lengths = [128, 256, 512, 1024]
    actions = []

    for index in range(1, ACTION_COUNT + 1):
        if index == SWITCH_POSITION:
            actions.append({"index": index, "type": "switch", "policy": "lof"})
        else:
            actions.append({
                "index": index,
                "type": "generate",
                "rid": f"switch-{run_id}-{index:04d}",
                "max_new_tokens": output_lengths[(index - 1) % len(output_lengths)],
            })

    return actions


async def get_load(session):
    async with session.get(
        f"{BASE_URL}/v1/loads",
        params={"include": "core"},
        timeout=aiohttp.ClientTimeout(total=5),
    ) as response:
        response.raise_for_status()
        return (await response.json())["loads"][0]


async def on_request_chunk_sent(session, trace_config_ctx, params):
    sent = trace_config_ctx.trace_request_ctx
    if sent is not None:
        sent.set()


async def generate(session, action, sent):
    record = dict(action)
    try:
        async with session.post(
            f"{BASE_URL}/generate",
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
            if meta.get("id") != action["rid"]:
                raise RuntimeError(f"响应 ID 与请求 ID 不一致：{meta.get('id')}")
            finish_reason = meta.get("finish_reason") or {}
            if finish_reason.get("type") == "abort":
                raise RuntimeError(f"请求被中止：{finish_reason}")
    except Exception as exc:
        record["error"] = str(exc)
    finally:
        # 连接失败也要唤醒发送循环，错误会保存在该请求的结果中。
        sent.set()
    return record


async def switch_policy(session, policy):
    async with session.post(
        f"{BASE_URL}/schedule_policy_switch",
        json={"new_schedule_policy": policy},
        timeout=aiohttp.ClientTimeout(total=30),
    ) as response:
        response.raise_for_status()
        result = await response.json()

    if not result["success"] or result["new_schedule_policy"] != policy:
        raise RuntimeError(f"策略切换失败：{result}")

    print(
        f"策略切换：{result['old_schedule_policy']}"
        f" → {result['new_schedule_policy']}",
        flush=True,
    )
    return result


async def main():
    run_id = uuid.uuid4().hex[:8]
    actions = build_actions(run_id)
    report = {
        "run_id": run_id,
        "action_count": len(actions),
        "request_count": sum(action["type"] == "generate" for action in actions),
        "actions": actions,
        "switches": [],
        "responses": [],
    }

    trace = aiohttp.TraceConfig()
    trace.on_request_chunk_sent.append(on_request_chunk_sent)

    # 不让客户端连接池把前半段请求排队到控制请求之后。
    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=0),
        timeout=aiohttp.ClientTimeout(total=600, connect=30),
        trace_configs=[trace],
    ) as session:
        load = await get_load(session)
        if load["num_running_reqs"] or load["num_waiting_reqs"]:
            raise RuntimeError("请先等待其他请求结束、队列清空")

        # 测试准备操作，不计入下面的 200 个操作。
        report["initialization"] = await switch_policy(session, "fcfs")

        tasks = []
        try:
            for action in actions:
                if action["type"] == "generate":
                    sent = asyncio.Event()
                    tasks.append(asyncio.create_task(generate(session, action, sent)))
                    # 等到请求体进入 HTTP 发送阶段，继续发下一条；不等待生成结束。
                    # 此处不保证 Scheduler 已经接收或处理该请求。
                    await sent.wait()
                else:
                    print(
                        f"第 {action['index']} 个操作：切换到 {action['policy']}，"
                        f"此前已发起 {len(tasks)} 条推理请求",
                        flush=True,
                    )
                    result = await switch_policy(session, action["policy"])
                    report["switches"].append({
                        **result,
                        "action_index": action["index"],
                        "client_ack_time": time.time(),
                    })

            print(f"操作发送完毕，等待 {len(tasks)} 条推理请求完成", flush=True)
        except Exception as exc:
            report["send_error"] = str(exc)
        finally:
            report["responses"] = await asyncio.gather(*tasks)

    output = Path(__file__).with_name(f"concurrent-{run_id}.json")
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    errors = [record for record in report["responses"] if "error" in record]
    print("操作总数：", report["action_count"])
    print("计划推理请求数：", report["request_count"])
    print("已完成推理请求数（含失败）：", len(report["responses"]))
    print("序列内成功切换次数：", len(report["switches"]))
    print("请求异常数量：", len(errors))
    print("结果文件：", output)

    if "send_error" in report:
        raise RuntimeError(f"操作发送失败：{report['send_error']}")
    if errors:
        raise RuntimeError(f"有 {len(errors)} 条推理请求失败，详情见结果文件")

    print("固定操作序列已执行完毕，请通过服务端快照检查排序正确性。")


if __name__ == "__main__":
    asyncio.run(main())
