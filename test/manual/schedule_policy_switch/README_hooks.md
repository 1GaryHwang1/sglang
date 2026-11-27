# Custom scheduling hook checks

These manual tests cover normal, non-overlap, non-speculative generation. Run
them in the development container, from the repository root, on a dedicated
server. Policy names are fixed, so restart the server before repeating a run.
No changes to production code are required by these tests.

`enable`, `on_preempt`, and `disable` receive one `req`. `on_sort` receives
`reqs` and an optional `engine_view` once per ranking pass; `enable` can also
declare `engine_view`. `on_batch`,
`on_compute`, and `on_stop` receive `reqs` and an optional `batch` once per batch
event. These tests iterate `reqs` inside the callbacks to retain per-request
metrics. `on_stop` sees only requests whose results were applied. The engine
view's `running_reqs` comes from the current planning pass, and its membership
is read-only; the request objects themselves remain mutable.

## Batch event dispatch without a model

```bash
python test/manual/schedule_policy_switch/verify_batch_events.py
```

This check calls real Scheduler planning, submission, and result-processing
methods with substituted model/cache services. It verifies one callback per
batch, keyword-only and batch-only handlers, the current planning view, and
stop-before-disable cleanup. Both ordinary overlap and non-overlap submission
branches are exercised; no GPU forward is run.

## Normal execution, cancellation, and priority preemption

Start the server in one terminal:

```bash
bash test/manual/schedule_policy_switch/start_custom_policy_hooks_server.sh
```

Run the client in a second terminal in the same container:

```bash
python test/manual/schedule_policy_switch/verify_custom_policy_hooks.py
```

The launcher defaults to `/models/Qwen3-0.6B`. A different model and additional
server options can be passed after the profile:

```bash
bash test/manual/schedule_policy_switch/start_custom_policy_hooks_server.sh \
  /models/Qwen3-0.6B normal --mem-fraction-static 0.3
```

The server has two running slots, two waiting slots, a 128-token chunk budget,
mixed prefill/decode, and lower-integer-first priority scheduling. Prefix caching
is disabled so chunk boundaries have deterministic expectations. Two independent
policies register all seven fixed-name hooks; both continue updating while
inactive. Their ranking function actually consumes the custom score prepared by
`on_sort`, including the remaining-output metric maintained by `on_stop`.

| Scenario | Independent expectation |
| --- | --- |
| 16 input tokens, 8 output tokens | 1 prefill + 7 decode rounds; each execution hook fires 8 times |
| 512 input tokens, 8 output tokens, no other requests | 4 prefill + 7 decode rounds; exactly 3 rounds produce no output |
| Chunk prefill mixed with an existing decode request | Both requests emit compute events before either emits stop; their round kinds differ |
| Cancel a middle prefill chunk | Zero output; each completed chunk emits stop; one disable |
| Reject a new request at a full waiting queue | HTTP 503; no initialization or hooks |
| Higher-priority request evicts a queued request | HTTP 503 for the victim; disable before cleanup; no compute for the victim |
| Cancel queued/running requests | Queued requests have no execution events; running requests settle before disable |
| Higher-priority request preempts a running request | Retraction count equals preempt count; requeue, sort, and resumed compute are observed |
| Switch to FCFS, then to the observer policy | Both registered policies still receive identical event sequences and retain their own objects |
| Final probe request | No previously terminated request remains in the framework's request-data tables |

## Decode retraction site

Stop the previous server, then start a fresh instance:

```bash
bash test/manual/schedule_policy_switch/start_custom_policy_hooks_server.sh \
  /models/Qwen3-0.6B retract
```

Run:

```bash
python test/manual/schedule_policy_switch/verify_custom_policy_hooks.py --mode retract
```

This profile enables `SGLANG_TEST_RETRACT` with an interval of 16 forwards. It
exercises the real `update_running_batch -> retract_decode -> requeue` path
without depending on GPU memory exhaustion. Two requests must complete their
96-token outputs, at least one real retraction must occur, and no second enable
is allowed for a requeued request. This tests the memory-retraction hook site;
it does not claim to test an actual CUDA OOM.

## Terminal branches without a model

```bash
python test/manual/schedule_policy_switch/verify_custom_policy_hook_abort_paths.py
```

This separate check does not require an HTTP server. It uses real Scheduler
methods, Req objects, and policy registration/binding, with substituted cache,
IPC, and decode capacity/retraction services. It covers priority/queue intake
rejections, failed requeue, preservation of an existing abort reason, and the
`reqs_to_abort` branch after decode retraction. It verifies that disable runs
once, sees terminal state and existing request data, updates both policies, then
the framework clears the data. This is a control-flow check, not a GPU test.

## Metrics and evidence

Per-request metrics include event counts, generated/remaining output tokens,
prefill/decode/no-output rounds, queue visits, waiting time, ranking score, and
elapsed execution time. Per-policy metrics include event totals, initialization
and terminal counts, and the live request set.

Every callback records both the updated custom metrics and a separate snapshot
of Req state. The client checks event transitions, exact counts, chunk ranges,
output deltas, timestamps, retraction counters, and the actual HTTP results. It
also checks that the two policy objects differ, their event totals agree, and
their request data survives switching until termination. Elapsed time is checked
against local timestamps, not treated as a measurement of GPU kernel duration.

Event snapshots are written to `/tmp/sglang-policy-hooks.jsonl`; generated logs
do not belong in Git. When client/server paths differ, use `--server-policy-file`
and `--snapshot-file` with paths on a shared filesystem. The client output lists
the metrics and prints a PASS result only after all assertions pass.
