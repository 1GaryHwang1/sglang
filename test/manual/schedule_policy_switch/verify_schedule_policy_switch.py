import time
from concurrent.futures import ThreadPoolExecutor

import requests

BASE_URL = "http://127.0.0.1:30000"


def get_load():
    response = requests.get(
        f"{BASE_URL}/v1/loads",
        params={"include": "core"},
        timeout=5,
    )
    response.raise_for_status()
    return response.json()["loads"][0]


def generate(rid, max_new_tokens):
    return requests.post(
        f"{BASE_URL}/generate",
        json={
            "rid": rid,
            "text": "Please count upwards from one.",
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": max_new_tokens,
                "ignore_eos": True,
            },
        },
        timeout=120,
    )

def wait_for_waiting(expected, blocker):
    deadline = time.monotonic() + 10

    while time.monotonic() < deadline:
        if blocker.done():
            blocker.result().raise_for_status()
            raise RuntimeError("占位请求提前结束，本轮测试无效")

        load = get_load()
        if (
            load["num_running_reqs"] == 1
            and load["num_waiting_reqs"] == expected
        ):
            return

        time.sleep(0.1)

    raise TimeoutError(f"等待 waiting={expected} 超时")

def switch_policy(policy):
    response = requests.post(
        f"{BASE_URL}/schedule_policy_switch",
        json={"new_schedule_policy": policy},
        timeout=10,
    )
    response.raise_for_status()

    result = response.json()
    assert result["success"], result
    assert result["new_schedule_policy"] == policy, result

    print(
        f"策略切换：{result['old_schedule_policy']}"
        f" → {result['new_schedule_policy']}"
    )

def main():
    load = get_load()
    assert load["num_running_reqs"] == 0, "请先等待其他请求结束"
    assert load["num_waiting_reqs"] == 0, "请先等待队列清空"

    with ThreadPoolExecutor(max_workers=4) as executor:
        # 后台发送，主线程继续检查服务端状态。
        switch_policy("fcfs")
        blocker = executor.submit(generate, "switch-test-blocker", 4096)

        try:
            deadline = time.monotonic() + 15

            while time.monotonic() < deadline:
                if blocker.done():
                    blocker.result().raise_for_status()
                    raise RuntimeError("占位请求提前结束")

                load = get_load()
                if load["num_running_reqs"] == 1:
                    print("占位成功：running=1")
                    break

                time.sleep(0.1)
            else:
                raise TimeoutError("未观察到占位请求运行")

            test_futures = []

            for name, tokens in [("A", 16), ("B", 128), ("C", 64)]:
                future = executor.submit(
                    generate,
                    f"switch-test-{name}",
                    tokens,
                )
                test_futures.append(future)

                wait_for_waiting(len(test_futures), blocker)
                print(
                    f"{name} 已入队，"
                    f"running=1，waiting={len(test_futures)}"
                )

            switch_policy("lof")

            if blocker.done():
                raise RuntimeError("切换期间占位请求已结束，本轮测试无效")

        finally:
            # 当前先验证占位，验证后清理请求。
            response = requests.post(
                f"{BASE_URL}/abort_request",
                json={"rid": "switch-test-blocker"},
                timeout=5,
            )
            response.raise_for_status()

        blocker.result()

        records = []

        for name, future in zip(["A", "B", "C"], test_futures):
            response = future.result()
            response.raise_for_status()

            meta = response.json()["meta_info"]
            started_at = meta["forward_entry_time"]

            records.append((name, started_at))
            print(f"{name}: 开始执行时间={started_at:.6f}")

        # 按服务端实际开始执行的时间排序。
        records.sort(key=lambda item: item[1])
        actual_order = [name for name, _ in records]
        expected_order = ["B", "C", "A"]

        print("预期顺序：", " → ".join(expected_order))
        print("实际顺序：", " → ".join(actual_order))

        assert actual_order == expected_order, (
            f"调度顺序错误：预期 {expected_order}，实际 {actual_order}"
        )

        print("PASS：切换 LOF 后，实际执行顺序正确")

        print("三个测试请求均已完成")


if __name__ == "__main__":
    main()