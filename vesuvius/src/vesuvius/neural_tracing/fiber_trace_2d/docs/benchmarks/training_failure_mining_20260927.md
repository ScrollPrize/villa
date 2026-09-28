# Training failure mining

## Runtime update: bounded direct-trace scheduling

The approved no-argument runner is now
`bash /home/hendrik/villa3/scripts/run_training_failure_mining`.
Its control file `scripts/training_failure_mining_control.json` selects
`start`, `stop`, or `build`, and currently overrides runtime GPUs to **0–3**.
Logs and stage status are `workflow.log` and `workflow_status.json` in the root below.

Initial execution found three PHerc0139 single-control-point annotations with
no eligible span: preparation now records these exclusions (943 runnable fibers).
Bounds validation ignores extrapolated tails, matching the CP-bounded protocol.
The failed thread-per-case startup diagnostics are preserved in
`evaluations/classic/failed_thread_per_case_startup/`.

The direct live beam evaluator now uses a bounded OS-worker pool (the existing
`workers` setting, 32 here) with stackful, same-worker resumable case contexts.
An inference wait yields its context and CPU permit; bridge publications wake
the pool. Guarded 8-MiB virtual stacks are demand-paged, not fully committed.
Blocked readiness predicates are also checked on a 10-ms timer for timeouts.
The prior 256-case/thread limit no longer applies to this direct-trace path.
Fiberlet/anchor construction still uses its existing separate scheduling path.

Reference-prefetch geometry uses the unchanged exact calculation in a 32-process
spawn pool; jobs cover eligible runs and results are unioned by shared cache root.
Inference parameters, remaining-distance priority, source cache identity and
low-priority background-prefetch semantics are unchanged.

Validation: 600 suspended cases complete with four OS workers; 28 Python tests;
23 native replay and 86 native tracer test cases, including queued real-socket
heartbeat and timeout subcases. Native code is built inside villa3 and the
training configs point to that binary.

Rerun observation (2026-09-27 UTC): Classic full stage started 13:41:15;
native launch config was ready at 13:42:07 (52 seconds, including preflight
reference/model work, versus roughly eight minutes before). All 1,524 run
prefetch jobs completed. Live waiters reached approximately 3,043 without the
old 256-case failure. At elapsed 87.1–117.1 seconds, inferred tiles increased
256 → 1,576: **44 tiles/s**. GPUs 0–3 each showed 100% utilization and
42,267 MiB memory usage. Roughly 249,000 queued
tiles imply about 95 minutes of inference at that instantaneous rate, not an
end-to-end ETA for all models/reproduction. This is a startup observation, not
a completed benchmark result. Cache namespaces were unchanged.

## Dataset and models

Use the locked `train` split of
`/ephemeral/hendrik/fiber3d_train/2026-09-09_x4_collection/split_95_5_length_seed20260910`.
There are 946 fibers: 197 PHerc0139, 74 PHerc0332, 77 PHerc1299
(collection label PHerc1229), 5 PHerc1451, and 593 PHercParis4
(463 eval_fibers + 130 fibers). Do not mix the held-out test split into this export.

Configs in `fiber_trace_3d/configs/`:

- `eval_train_mining_classic_20260927.json`: Classic best91.5k.
- `eval_train_mining_lowres_20260927.json`: refinement 212757 best EMA999 u22000.
- `eval_train_mining_fullres_20260927.json`: fullres refinement 213753 best EMA999.
- `failure_union_training_20260927.json`: reproduce, union, fill inspection corridors, export.

GPU selection is 0–5. The evaluation configs retain their existing model hashes,
cache namespaces, inference parameters and worker settings. Fibers are shuffled
reproducibly within each collection with seed 20260927. All three evaluations
must complete before union extraction; their summaries are the extraction inputs.

## Extraction semantics

There is **no failure-count quota**, per scroll or globally. `extract_all` derives
an unreachable selection bound from the complete candidate inventory instead of
using the previous arbitrary 100,000-case bound. All configured models contribute
failures, even if a model's aggregate performance is worse than Classic.
The original self-proximity exclusion, reproduction search in both directions,
minimum/padded start checks, deduplication, extra 500-base-voxel context and
200-base-voxel interval merge remain in use. Candidates that cannot reproduce
under those checks still do not become extracted pieces.

The extraction global worklist, batching and inference dispatch are unchanged;
the native direct-trace scheduler/startup were subsequently updated as described above.
Extraction does not terminate work early. Filtering for
500 locations per scroll and fiber diversity is deliberately a separate future
step over this full extraction.

Exports preserve source control points and segment metadata; crop boundaries
inherit the containing segment. No re-optimization of the reference is done.

## Output and cache reuse

Root: `/ephemeral/hendrik/fiber3d_train2/evaluations/train_failure_union_20260927_v1`

- `evaluations/{classic,lowres,fullres}/`: full training evaluation artifacts.
- `reproduction_search/`: existing parallel reproduction workflow and checkpoints.
- `fibers/<scroll>/`: deduplicated reference pieces.
- `chunk_subsets/<scroll>/`: Zarr metadata and links to required published chunks.
- `vc3d_channels/<scroll>/`: links to those subsets, with portable Lasagna manifests.
- `manifest.json`, `cache_coverage.json`: provenance and required-chunk coverage.

Normalized scroll names: PHerc0139, PHerc0332, PHerc1299, PHerc1451, PHercParis4.
Both Paris4 collections share one export. Existing inference cache roots remain
authoritative: only missing chunks are inferred; exported chunk links do not copy
the cache or include scratch files. `export_chunk_subsets.chunk_mode` can instead
be `copy` for a standalone subset. `rsync -aL` dereferences the default links.
The test extraction is not changed.

## Execution sequence

For each of the three evaluation configs use the existing evaluation runner's
`prepare`, `validate-full`, then `full` modes. Then use its `failure-union` mode
with `failure_union_training_20260927.json`. Use code from this checkout: the
old approved launcher under `/home/hendrik/villa` resolves its own checkout and
does **not** automatically execute these changes in villa3. Do not launch that
stale code against these configs. No new command approval or run was requested
as part of preparing this implementation.

Preparation validation: unit tests for both train/test membership auditing,
source CP/constraint preservation, uncapped model inventory and per-scroll chunk
subset links. See the runtime update and workflow status for execution progress.

## 2026-09-27 interval-check failure and repair

### Explicit resource policy (user instruction, 2026-09-27)

All three mining model configs now explicitly retain the tuned settings:
128 accumulator processes, 114688 hot records, 2048 spill records per segment,
128 input workers and 128 publication workers. GPU allocation remains 0–3.
Do not change these settings, or couple CPU/RAM settings to GPU count, without
explicit user instruction. Counts are preserved exactly (not rescaled by chunk
size). The undersized Classic attempt was stopped at 20:37 UTC and archived as
`evaluations/classic/stopped_undersized_accumulator`; published inference caches
remain unchanged. Workflow restarted with PID 710058. No accumulation algorithm
or tracing/model parameters were changed in this resource-setting correction.

Post-start verification: inference initially waited for the previous manager
grant because four orphaned workers from the stopped run survived SIGINT.
The approved launcher terminated those exact workers after verifying their
orphan status, original process group and command/pipe identities; the manager
then automatically released the old grant. No manager state was edited.
The new run has 128 accumulation processes. Two GPU samples separated by
approximately one minute showed GPUs 0–3 all at 100%, each using 42267 MiB.
At elapsed 419.8–465.0 seconds, completed tiles increased from 2432 to 4276:
40.80 tiles/s, versus the previous approximately 8.5 tiles/s. At the end of
that interval, 128 tiles were GPU-submitted, accumulation backlog was zero,
21372 records were hot and zero spilled. These are early post-start results,
not evidence about later pressure once the hot pool fills. 368 trace artifacts
were already present (some using retained inference). Resource settings remain
as explicitly requested.

Classic stopped at 16:27 UTC after 128,361 tiles and 57,246 published
chunks. The failure was post-trace measurement, not missing inference:
case 2293 (Paris4 eval collection, reverse, source
`anon_20260902T164827563_000384.json`, eligible run 1) completed its full
19,394-vertex reference. Its selection length was 154719.34786287573 base
voxels; replay ended at 154719.3478628727. The approximately 3e-9 difference
exceeded the old absolute 1e-9 interval assertion. Selection prefix sums and
directional replay summation need not round identically.

The outcome checker now allows a floating-point accumulation bound
`max(1e-9, 8 * double_epsilon * vertex_count * max(1, length))` for the
endpoint comparison. Both direct and fiberlet replay use the same check.
Completion is tested against replay's own endpoint, then credited with the
canonical selection length. Nonfinite intervals and genuine mismatches remain
errors. Tracing, acceptance ellipses, failure counting, scheduling and inference
cache identities are unchanged. A regression uses the actual failing lengths,
checks both replay types, summary acceptance, incomplete coverage, a 0.01-voxel
interval mismatch, and NaN rejection.

Build/test command: `bash /home/hendrik/villa3/scripts/run_training_failure_mining`
with control action `build`; this also runs the reference replay benchmark
tests now. Published inference remains reusable; the failed output is retained.
Verification passed: 23 replay, 86 tracer, and 18 reference replay benchmark
test cases, plus the standalone cooperative case-queue test. The extraction
was not restarted as part of this repair.

## 2026-09-28 post-kernel-upgrade resume

At 12:29:56 UTC resumed with the same approved no-argument command:
`bash /home/hendrik/villa3/scripts/run_training_failure_mining`.
Control selects GPUs `0,1,2,3,4,5,6,7`; no other resource tuning changed.
Kernel is now 6.8.0-1062-oracle and the ephemeral write/fsync probe passes.
Classic and lowres completed 3048-case summaries are reused. Partial fullres
output is preserved in `evaluations/fullres/full.interrupted.20260928T122956296927Z`.
The evaluator cannot append partial trace output, so fullres tracing reruns using
published inference caches. Prepared bindings/configs and cache identities are
unchanged; subsequent failure-union stage remains uncapped and unchanged.
Archival is explicitly enabled for fullres only and runs under the workflow lock.
Tests: `python -m unittest discover -s scripts -p test_training_failure_mining.py`
(five passing cases). Startup PID42153; log is the existing workflow.log.

## Parallel cache startup (next launches)

### Revised mining scope, 2026-09-28

The fullres training evaluation was stopped. Only best-lowres training failures
(1784 recorded episodes before filtering/reproduction/union) feed extraction.
Classic/fullres failures are excluded. The existing reproduction search runs
unchanged, then adds 500-base-voxel context and merges overlapping pieces.
Highres inference fills only those inspection corridors. Its separate completed
producer summary is selected by `fill_missing_inference.summary`; this does not
add any failures from that summary. Cache roots/checkpoint match the interrupted
training run. Current runtime tuning is retained in inference_overrides.

Run command remains `bash /home/hendrik/villa3/scripts/run_training_failure_mining`.
Control selects evaluation_names=[lowres], devices0-7. Launched13:02:32 UTC,
PID76919. Old session42153 orphan CPU workers were verified and terminated to
release its stale manager grant; published inference retained. Extractor tests:
21 passed using `python -m pytest -o pythonpath=/home/hendrik/villa3/vesuvius/src
-q vesuvius/tests/neural_tracing/test_short_failure_benchmark.py`.

### Startup implementation

### Bounded failure-search scheduling (2026-09-28)

The previous search handed301134 trials to an eager stack allocator and sorted
them longest-first. It did not cancel native longer-distance work. The new
queue reuses at most8 coroutine stacks per native worker; descriptors retain
shared source polylines and materialize only admitted trial intervals. At192
workers that is at most1536 stacks, versus301134. Global trials run shortest
first; ordinary benchmark ordering is unchanged. Valid adjacent n/n+100 audits
prune larger queued trials for the same candidate/direction. Already-running
trials finish; smaller trials are never pruned. Native audit decisions are
checked against the existing Python saved-geometry audit. Cancelled trials are
reported separately and never interpreted as no-failure traces.

Inference inherits producer settings unchanged (removed extractor's clamps on
input/accumulator/publication workers and slot counts). Explicit CPU grant192,
all8 GPUs. Approved runner remains unchanged; control action build runs the
workspace RelWithDebInfo build and native tests, start resumes extraction.

Validation: standalone g++13 -O2 queue test with301134 trivial callbacks took
0.038–0.040s across3 runs (median0.039s); this is not tracing throughput. Native
23 replay,86 tracer,18 reference tests pass. Python47 extractor/namespace tests
pass. Previous live native utilization was~564% lifetime CPU; resumed PID101496
at13:26:43 UTC. Production comparison pending in this entry.

Live follow-up13:30:20 UTC:17200 trials processed (360 skipped);5535 new
inference tiles and1169 published chunks at109.4s live-stage elapsed. Native
process lifetimeCPU4044% at55s vs prior564% at313s; these are observational
utilization samples, not matched-work speedup measurements. All8 GPUs sampled
65–100% busy. Checked500 real trace artifacts against Python audit:0 mismatches,
12 valid reproductions. Full-stage runtime remains pending.

`coordinate_native` now prepares uncapped reference-prefetch requests with
`lasagna.demand_predict3d.prepare_background_requests` before starting inference.
It uses the existing `input_workers` setting (128 here), bounded batches of
four requests per worker, worker-local Zarr handles, and the same completeness,
metadata and decode checks as serial readiness. Missing-chunk contributor
calculation also runs in these workers. Registration records merge in original
order; priority heaps and publication queues remain coordinator-owned. Python
contributor math can still be GIL-limited. Explicit capped-background operation
retains its existing incremental path. No cache identity or tuning changes.

The active extraction was not stopped/restarted. At elapsed 1107.9s it had
processed 20,217 tiles and published 21,161 chunks using the old loaded code.
The new path logs prepared counts, worker count and elapsed startup time.

CPU test command (no build or GPU required):
`/home/hendrik/.venv_las/bin/python -m pytest -q -s lasagna/tests/test_demand_predict3d.py -k parallel_startup`

Controlled latency benchmark: 256 requests, 2ms simulated blocking validation
per request, actual dependency/registration logic, three repetitions, 16 workers.
Serial mean 0.5370s (min/median/max 0.5358/0.5371/0.5380s); parallel mean
0.0460s (0.0438/0.0448/0.0493s). This verifies latency hiding, not production
filesystem throughput. Real production startup speed remains unmeasured until
the next requested launch; the ongoing run is intentionally left untouched.

Validation: 25 namespace tests passed; full demand suite 77 passed / 1 failed
in 65.68s. The failure is the unchanged legacy scratch-cleanup preservation test
(`test_legacy_stale_scratch_is_cleaned_on_resume`); startup-specific tests pass.
