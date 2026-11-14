# Opt-in read coordination profiling

For bounded native CLI diagnostics, set `VC_READER_COORDINATION_TRACE=/path/to/new/trace.jsonl` before starting the process. Ordinary reads do not create a monitor or register thread/job records. Enabling tracing starts one monitor sampling every 10 ms; updates on participating threads use thread-local atomic flags. Registration takes a registry lock only when a thread or parallel job is first recorded. This is diagnostic overhead: compare profiling-disabled controls and do not treat enabled timings as production performance.

Each JSON line contains a steady-clock timestamp and concurrent counts:

| Field | Meaning |
|---|---|
| `planning` | Calling threads inside a custom-interpolation read, including its execution/waits; not exclusively planning CPU time |
| `tasks` / `chunk_tasks` | Threads inside any reader task / specifically chunk-resolution tasks |
| `producers` | Threads inside an exact-key load producer; they may be running or blocked internally |
| `chunk_waiters` | Threads obtaining another producer's shared result; an already-ready result may pass briefly through this scope |
| `batch_waiters` | Callers at a parallel-for completion barrier |
| `registering` | Threads in in-flight-key registration, including both lock acquisition and its short protected work |
| `helper_tasks` / `helper_chunk_waiters` | Task/wait counts specifically on reader helper threads |
| `ready` / `chunk_ready` | Unclaimed task items across live jobs / specifically chunk-resolution jobs; unclaimed does not guarantee a chunk is independent of every active producer |
| `io` | Selected-channel file open and range-read calls, including their synchronous waits |
| `decode` | Selected-channel Blosc block decompressor calls |
| `allocate` | Major selected-channel output/scratch allocation or resize scopes, including buffer initialization |
| `copy` | Selected Blosc frame/channel copy scopes |

The counters are overlapping scopes, not disjoint time percentages. In particular, I/O, decompression, allocation and copying are subsets of producer work; a producer count is not a count of CPUs executing. Counts are sampled, not transactionally consistent, and very short scopes can be missed. A scope includes internal waits as well as execution. Use OS CPU/scheduler observations to distinguish the two.

On Linux, additionally setting `VC_READER_COORDINATION_WCHAN=1` samples `/proc/self/task/TID/wchan` for producer threads and changes the monitor interval to 30 ms. `producer_wchan` contains aggregate kernel wait-channel observations. A `0` value is not proof of CPU execution, and reading the wait channel happens slightly after the corresponding flag snapshot. This adds profiling I/O; compare the basic trace and disabled control. This extra observation is omitted on other platforms.

The normal-sampling integration can record exact same-host coordinate replay input by setting `VC_READER_COORDINATE_REPLAY=/path/to/new/coordinates.bin`. Records are native-endian uint64 point count, float32 base-coordinate spacing, then that many XYZ float32 triples. This is a disposable benchmark fixture, not a portable persistent data format. It records each normal-coordinate batch independently, not chunk data, and has its own optional write cost. Capture separately from the timing control. Do not enable capture and point replay in the same process.

These diagnostics are intended for standalone, non-forking native processes. They are not a fork-safe live-monitor API. They do not alter source data, interpolation arithmetic, cache policy, thread counts or scheduling decisions. Trace destinations must be writable; keep generated files out of source control. Use `test_native_coordination_trace` to check that known blocked callers/helpers and queued tasks are observed together; its fixtures are automatically cleaned.

## Source syscall timing

`VC_READER_SCHEDULER_PROFILE=1` adds per-stage cumulative `file_open`, `file_stat`, `file_read` and `file_close` counters to scheduler snapshots. Calls are counted per executing thread and one in 64 is timed; ordinary reads with profiling disabled do not read clocks. `file_read.units` counts requested bytes. For these four metrics specifically, `wait_ns` is sampled wall time and `hold_ns` is sampled executing-thread CPU time, not mutex hold time. Thread CPU timing uses `CLOCK_THREAD_CPUTIME_ID` where available; otherwise it is zero and must not be interpreted as blocked time. The stat counter currently covers selected-channel local file loads; the other counters cover shared chunk-file helpers.

Subtract snapshots from the same steady-state phase. Mean wall or CPU time per sampled call is the corresponding nanosecond sum divided by `samples`. Scaling sampled means by `count` estimates aggregate worker time, not elapsed pipeline duration. Wall minus CPU includes blocking, scheduling delays and CPU throttling; it does not identify storage-device latency on its own. Combine with kernel wait channels, process I/O counters and profiling-disabled controls before choosing I/O concurrency. Warm page-cache reads can still perform significant filesystem and memory-copy CPU work.
