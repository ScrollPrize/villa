# Reusable fiber test and experiment runner

`reader-multichannel` runs paired1/7-channel public ROI workloads,16/64callers,
10GiB/512MiB cache budgets, independent exact oracle, and matched memcpy/decode
controls. `reader-multichannel-report` generates output-byte throughput and
physical-decode amplification tables. No production tracing/layout changes.

`reader-layout` builds/tests the generic blocked ROI copier and runs the locked
1/16/64-caller standalone reader benchmark plus bounded-cache streaming. It uses
the same no-argument command below; local inputs/results/builds remain on the
evaluation filesystem. No tracer or CUDA kernel runs in these measurements.
`reader-layout-pool` validates partial idle-buffer trimming and reruns bounded
streaming against the frozen pre-layout binary. `reader-layout-report` and
`reader-layout-pool-report` generate the respective Markdown/JSON reports.

Approve this command once and reuse the identical command:

```sh
/home/hendrik/villa3/scripts/run_fiber_tests
```

It takes no CLI arguments. Select named suites in the untracked
`local/fiber_tests.json`, for example:

```json
{"suites": ["runner-tests", "manager-tests"]}
```

Without a config it runs only the launcher unit tests. The runner prints the
configuration and selected tests, runs sequentially, and stops on the first
failure. Unknown keys/suites, duplicate suites and missing harnesses fail before
execution. There are no arbitrary shell commands or pytest argument passthrough.
No installs, commits, production restarts or direct GPU allocation are performed.

| Suite | Work |
|---|---|
| `runner-tests` | Launcher validation/unit tests; CPU only |
| `manager-tests` | Manager command/partial-grant/cleanup regressions; synthetic GPU inventory, temporary local sockets |
| `loader-build` | Existing isolated queued-wake CMake configure/build harness |
| `loader-tests` | Native cache/queue and related regression harnesses; may build their test targets |
| `loader-scaling` | Existing matched 1/16/64-worker experiment; requests its GPU/CPU grant through the manager |
| `loader-report` | Recreate the scaling report from saved measurements |
| `reader-isolated` | Build and measure the frozen independent native-reader baseline |
| `reader-isolated-after` | Intermediate reader candidate and native regressions |
| `reader-final` | Final native reader build, regressions, 1/4/16/64-worker scaling |
| `reader-fixed-pool` | Compare frozen/final bulk readers with a fixed 64-worker pool |
| `reader-capacity` | 64/128/224-worker native read/decode/copy/cache controls |
| `reader-roi` | Actual `Volume::readZYX` ROI reads, cold decoded cache and resident cache |
| `reader-regressions` | Rebuild/run native queue, cache, hints, Zarr and ROI regression targets |
| `reader-report` | Recreate independent reader tables from completed measurement JSONs |
| `reader-roi-multi-before` | Frozen public multi-chunk ROI benchmark (do not rebuild after source changes) |
| `reader-roi-after` | Build/regressions and matched public multi-chunk ROI benchmark with keyed wakeups |
| `reader-roi-single-after` | Matched single-chunk public ROI benchmark |
| `reader-wake-capacity` | Internal diagnostic confirming the wakeup fix at64/128/224workers |
| `reader-roi-report` | Public ROI before/after tables and full timing distributions |
| `reader-scaling-diagnosis` | Matched ROI/control inputs at1/16/64callers, physical IO/page faults and reused-buffer controls |
| `reader-scaling-report` | Summarize completed diagnostic counters and timing distributions |
| `reader-decode-reuse` | Isolated decode allocation candidate, native regressions and matched ROI/reused-decode controls |
| `reader-decode-reuse-report` | Before/after distributions including512MiB bounded-cache streaming and allocation counters |
| `reader-decode-reuse-tests` | Rebuild/rerun native regression targets without repeating performance measurements |

Loader suites require the existing `local/test_queued_*` harnesses and build/data
fixtures in this workspace; they fail explicitly on fresh clones without those
fixtures. Scaling retains the harness's resume behavior: existing completed
measurements are reused, not silently retimed. Change the experiment's output
directory when fresh timings are intended. New experiments should add a named
suite, not an arbitrary command option. Announce changed suite selection before
running it; approval of this launcher does not authorize unrelated work.

The `reader-*` suites likewise require the local `test_reader*` /
`test_isolated_chunk_reader*` harnesses and locked Paris4 inputs. They run a
standalone C++ benchmark, **without tracing, inference or CUDA kernels**. The
manager currently requires a GPU promise even for CPU work; these harnesses
request a nonexclusive 1 MiB token on one device plus the measured CPU resources.
Builds and scratch for new reader candidates live under the evaluations root.
The generic benchmark target and portable configuration are documented in
[`chunk_channel_hints.md`](../volume-cartographer/docs/chunk_channel_hints.md).

The launcher uses the existing `/home/hendrik/.venv_las/bin/python` environment.
On other hosts set `VILLA_TEST_PYTHON` to an existing interpreter with pytest.
It sets repository imports and single-thread BLAS defaults, and never installs
an environment. Long GPU work stays under the existing manager-aware harness.
