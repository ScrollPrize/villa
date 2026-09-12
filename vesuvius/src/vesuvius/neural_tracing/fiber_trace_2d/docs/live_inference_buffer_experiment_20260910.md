# Result-buffer experiment and accumulator OS profile — 2026-09-10

Outcome:128 result buffers do NOT improve completed inference throughput over8.
They allow GPUs to run ahead but accumulate a larger unfinished backlog. OS
measurements identify kernel CPU/page-fault overhead in accumulator workers,
not GPU arithmetic, scheduler starvation, or predominantly disk-read waits.
Exact kernel function responsible is not yet isolated.

## Setup and implementation

Same classic best91_5k/PHerc0139 CT L1 workload and10 test fibers as
live_inference_accumulation_fix_20260910.md. Eight H10080GB GPUs, input512³,
overlap96/border32,32³ output chunks, bf16 model/f16 accumulator,8 accumulator
processes/4 publication threads. Shared native AVX-512 addition unchanged.
Native sampler/build unchanged (GCC13 RelWithDebInfo, Linux amd64).

Added optional `inference.result_slots`, defaulting to existing input-slot count.
Input slots remain8×256MiB=2GiB. Result slots are64MiB each:8→128 changes their
RAM from512MiB→8GiB and total shared slot allocation2.5→10GiB. No input batch,
tile geometry, precision or loader-worker changes. Memory preflight:
/dev/shm1008GiB free, host1.9TiB available, /ephemeral8.1TiB free.

Production default is NOT changed. The workload's **profile-only** override is
128, with optional `accumulator_os_metrics: true` for the diagnostic rerun.
Actual capacities and bytes are recorded in report.json. GPU results cannot be
mistaken for completed tiles: a result slot is released only after every touched
chunk acknowledges accumulation. Initial/final backlogs are both recorded.

## Results

Artifacts below `/ephemeral/hendrik/fiber3d_train2/evaluations/inference_profiles/`:

| Run | Result slots | Window | GPU results received | Fully accumulated tiles | 128³-equivalent completed tiles/s | Requested chunks published |
|---|---:|---:|---:|---:|---:|---:|
| Previous corrected baseline `20260910T115814021990` | 8 |40.0029s|71|71|113.592|43|
| Buffer experiment `20260910T121643381845` |128|40.0032s|135|70|111.991|43|
| OS diagnostic repeat `20260910T122023690046` |128|40.0040s|170|71|113.589|43|

Counts are512³ input tiles; equivalent128³ throughput multiplies by64. These
rates describe input work, not unique output coverage after overlap. Each run
has full source/checkpoint/config/reference/implementation identities and raw
samples. All use cProfile plus wall/event timings; the last adds sampled OS
metrics. Cache warmth and processed tile counts differ, so do not interpret
small differences as significant improvements/regressions.

128-buffer run: accumulation backlog8→73, GPU queued/active8→1, free result
slots112→54. The OS repeat: accumulation backlog11→110, GPU queued/active9→2,
free result slots108→16. GPU-active includes submitted work not yet returned,
so it may exceed GPU count. There was still result capacity at both cutoffs.
More received results therefore reflect work queued behind accumulation, not
higher completed throughput. Both produced the same43 requested chunks.

## OS diagnosis (40.004s window, eight accumulator processes)

| Counter | Sum over eight workers |
|---|---:|
| User CPU |9.24s|
| System/kernel CPU |289.17s|
| Kernel share of user+system CPU |96.90%|
| Main-thread runnable scheduler wait |0.297s|
| Minor page faults |19,290,464|
| Major page faults |22,644|
| Kernel-accounted storage read bytes |0|
| Kernel-accounted storage write bytes |9,613,385,728 (8.95GiB)|

Main-thread state samples:2996 running,127 sleeping,13 uninterruptible/D out
of3136 (~95.5% R,0.4% D). Wait samples mostly0 (running or hidden),124 pipe_read,
15 folio_wait_bit_common and4 futex_wait_queue. The configured interval is50ms;
actual392 samples/worker over40s indicates roughly102ms effective sampling under
load. Counts are samples, NOT exact wait durations. /proc reads are optional and
Linux-specific; unavailable/denied counters remain absent on other platforms.

All8125 acknowledged tasks reported **avx512**, not NumPy fallback. Task wall
time mean36.392ms, p502.513ms, p9520.672ms, max5.065s. Submitted queue+ack delay
mean0.905s, p500.265s, p952.380s, max19.065s. This queue metric starts only at
the worker queue submission; it EXCLUDES earlier time in the coordinator's
pending descriptor deque and includes acknowledgement delivery. Worker resource
deltas cover all work during the window, including partial tasks at boundaries;
acknowledged-task timing samples need not sum to those resource deltas.

Kernel delay accounting is disabled (`/proc/sys/kernel/task_delayacct = 0`), so
zero block-I/O delay counters are NOT evidence of no I/O waiting. Independent
state samples and CPU totals show workers predominantly executing in kernel
mode rather than sleeping. `write_bytes` counts storage bytes dirtied/caused by
these processes, not completed disk-device traffic or its latency; `rchar/wchar`
are logical syscall bytes, reported separately. Zero storage read bytes does not
exclude memory mapping/page-cache faults or filesystem metadata activity.

Conclusion: sparse mmap-backed accumulation is causing a page-fault/kernel-work
storm. Full-volume C-order arrays make small3D chunk rows widely strided in
memory, and interleaved chunk ownership can make workers map/write nearby file
pages. Those are plausible mechanisms, not a proven attribution to a specific
kernel path. The next useful investigation is page-friendly accumulator layout
or ownership and kernel fault profiling, not simply enlarging the result queue.
No such layout change was made in this experiment.

## Synchronization audit

- NO scratch pwrite, fsync, fdatasync, mmap.flush or native msync. Native
  `accumulator_add.cpp` only adds array rows; no file synchronization.
- During normal processing, disk durability occurs when final Zarr chunks are
  published: fsync completed channel files/directories, then commit ready ledger.
- Strictly, it is NOT "only Zarr files": SQLite setup/cleanup metadata commits
  and atomic provenance/profile-report writes also sync outside the accumulation
  path. This distinction was reported to the user. This buffer experiment did
  not change lifecycle metadata durability.
- Queue notifications, exclusive ownership and GPU transfer/event synchronization
  are not disk flushes. OS-managed dirty-page writeback remains possible without
  explicit sync calls.

## Run and validation

Existing approved command, no new launcher or dependency:

```sh
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py profile
```

Default workload profile now requests128 result slots and OS metrics. Edit
`profile.result_slots` to8 for the control; production evaluation still defaults
to8 result slots. Reports are timestamped and isolated; old results are preserved.

```sh
bash /home/hendrik/villa2/vesuvius/scripts/run_fiber3d_native_replay_tests /tmp/vc-live-native-cmake /home/hendrik/.venv_las/bin/python
/home/hendrik/.venv_las/bin/python -m pytest --noconftest -q vesuvius/tests/neural_tracing/test_live_inference_profile.py vesuvius/tests/scripts/test_classic_fiber_evaluation.py
```

Tests cover unequal capacities (1/7 result slots with4 inputs), zero capacity
rejection, byte accounting, slot ownership, late demand, readiness and failures;
Linux stat-field indices/units and missing proc data. Existing native/tracing
tests retained. Final native wrapper:62 tracer cases+1 cache case+31 Python
tests passed17.45s; profile/config suite5passed1.58s. `git diff --check` clean.
Independent review verified counters and ownership and flagged
queue-timing interpretation plus a report snapshot race. The phase snapshot is
now copied under its lock before summarizing, preventing background writers from
mutating it; that reporting-only fix followed the OS run. Raw CPU/fault/backlog
metrics and conclusions do not depend on the phase snapshot.

No full trace evaluation restarted. Production defaults unchanged; no commit
requested. Existing600s native timeout, input conversion and deferred native-read
cancellation are untouched. No macOS/arm64 performance claim.
