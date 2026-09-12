# PR draft: Overlap streamed model reads during line reoptimization

This is a checked-in draft and agent handoff, **not an opened GitHub PR**.
Repository: `ScrollPrize/villa`. Head branch: `line-annotation-model-prefetch`.
Intended base: `main`. Do not open a PR without the owner's authorization.

## Proposed PR body

**In one sentence:** Reduce waiting for streamed model data when reoptimizing a
fiber in VC3D's line annotation GUI.

**One real example:** The user traced PHerc1299 with fiber
`dj_20260908T152535598_000001.json` (3735 saved points, eight controls). Re-solving
its last control span with isolated cold and disk-warm model caches reproduced
the streaming delay without reading or caching the raw scan volume.

**Before:** Model reads dominated reoptimization despite millisecond-scale
Ceres compute; fully cached models were much faster. The user reported:
"When it's all downloaded, it's much faster."

**After this PR:** Model chunks can be queued along the existing line and ahead
of native tracing/normal growth, and independent temporary Lasagna tails can run
concurrently. Required reads, scoring and solver math retain their semantics.
The speculative queue has global count/byte bounds, low priority, independent
cancellation, and ordinary-demand retry after speculative failure.

**Proof:** The retained implementation's historical PHerc1299 measurements used
Ubuntu amd64, GCC 13.3, RelWithDebInfo, one native scoring thread, 64 model readers,
512 MiB configured cache, real S3 data, and the same frozen input and settings.
Each cold/warm pair uses a new cache, then a fresh process with that disk cache.

| Historical version | Pairs | Cold solve mean (min / median / max), s | Disk-warm mean, s | Cold downloaded MB mean |
|---|---:|---:|---:|---:|
| Prior branch, prefetch disabled |3|5.991 (5.856 /6.029 /6.089)|1.312|11.867|
| Initial straight-prefetch implementation (`fa167075d`) |6|4.452 (4.107 /4.298 /5.382)|1.324|17.310|
| Plus normal-growth window and parallel temporary tails (`cdbffc05b`) |6|3.796 (3.597 /3.760 /4.123)|1.324|17.373|

The last step was 14.7% faster than the initial enabled implementation, with
unchanged warm timing. These are **historical pre-merge comparisons, not a new
benchmark against current `main`**, and not a latency guarantee. Prefetch used
more bandwidth than disabled (about 46% more cold bytes). Cold remained slower
than cached. Exact geometry, decisions and shared downloaded bytes matched.
See [full PHerc1299 measurements and commands](../benchmarks/line_model_prefetch_pherc1299.md)
and [initial design, tests and profiling record](../benchmarks/line_model_prefetch.md).
Post-merge validation is recorded below separately.

**Why / where this is useful:** This came from a human using File > Data Catalog
and adding control points in the line annotation GUI. It targets Lasagna/fiber
model streaming, not raw-scan rendering. It can overlap reads with tracing or
the user's next edit without changing the optimized fiber.

- [ ] I personally verified that the example and proof above were produced by
      this PR on the stated data. (Owner to check; the agent cannot attest for them.)

## Details

- Queue-only speculative admission is capped at 16 requests and 32 MiB decoded
  estimates, further limited by shared read concurrency and decoded-cache size.
  Already running requests may finish after cancellation; this is not bandwidth
  reservation or preemption of an HTTP transfer.
- The GUI warmup worker uses immutable inputs and coalesces/cancels edit jobs.
  Speculative prediction metadata is opened off the GUI thread. The first
  synchronous normal pass on opening an existing line is not eliminated.
- Native lookahead is **straight only**. Guided/curved experiments were removed
  after they did not produce a reliable latency benefit. Do not resurrect
  `VC3D_LINE_MODEL_PROJECTION`; that switch is absent.
- `VC3D_LINE_MODEL_PREFETCH=0` is the startup opt-out for new warmup/moving
  prefetch and concurrent temporary reinit tails. Legacy scalar normal prefetch
  is unchanged. `VC3D_LINE_PERF_LOG=1` enables the diagnostic timings.
- Shared cache pinning/worker-drain fixes and speculative-origin retry tests
  accompany the scheduling changes. Required model-read failures are not hidden.
- Sharded/incompatible model grids skip unsupported speculative paths. macOS
  and arm64 runtime performance have not been tested on this Ubuntu host.
- Real-network timings vary; warm processes can still download optional extras.
  Process-wide remote counters include overlapping consumers and are not exact
  per-solve request attribution. A saved-state re-solve is not the unavailable
  historical click sequence.

## Agent handoff and publication status

- Source before merge: `7ad90b8c9ee628b7d794c4ca1f7e4ba42643a23b`, exactly the
  pre-guided-experiment tree from `cdbffc05b`.
- Merged upstream: `be09a85035059fd83471b1632b5898c62f2c65b1` (`origin/main`
  fetched on 2026-09-12).
- Merge/code commit: `ed59e1bd90572f47d88be740310f855889937f5b`.
- Conflict resolution retained main's process-wide `remoteCachePathFs()` for
  normal datasets, prediction datasets, attachment preparation and warmup;
  removed obsolete per-package cache-root calls; retained both sets of CMake
  tests and the shared unchanged optimization defaults.
- Earlier Codex/Fable plan and implementation reviews converged before the
  main merge. Those are not being represented as new post-merge reviews.
- Post-merge validation passed on 2026-09-12: VC3D and the benchmark built;
  all 23 selected CTest targets, 29 Python tests, and 61 offscreen GUI smoke
  checks passed (including 465 invalid-input probes). Details and commands below.
- The exact PHerc1299 fiber passed three cold/warm pairs with one scoring
  thread and one pair with four scoring threads. Geometry/decision signatures
  matched the corresponding pre-merge straight-prefetch references; shared
  downloaded payloads matched, with zero failed or pending requests at exit.
- No PR has been created. This file is intentionally committed so an agent
  checking out this branch can find the draft and the remaining human tasks.

### Local evidence locations (not uploaded datasets)

Original fiber SHA256:
`2e9a9fc9f59a6c9a7cc0187b7ddf8365e4bc67b4fa113d7b67824dd34ffae4a1`.

- Frozen input: `/tmp/villa-pherc1299-prefetch.IJmtzD/baseline-on/input/`.
- Historical kept implementation: `/tmp/villa-pherc1299-prefetch.IJmtzD/candidate-final/`.
- Post-merge evidence root: `/tmp/villa-line-prefetch-merge.VZlf7o/`.
- Post-merge runs: `pherc1299-threads1/` and `pherc1299-threads4/` under that
  root, including per-run commands, traces, inventories, hashes and summaries.
  `ctest-LastTest.log` and `gui-smoke-result.json` preserve test evidence;
  `merged-build/` freezes the measured executable and six repository libraries.
- These `/tmp` artifacts exist on the original workstation only. Another
  machine needs the owner's saved fiber and matching public model manifests;
  the runner refuses mismatched input/settings and existing output directories.
  Never delete the user's application cache or modify the original fiber.

### Before an actual PR submission

1. Obtain owner review and human-written commentary required by
   `CONTRIBUTING.md`; the quoted report above is motivation, not approval.
2. Add the requested before/after GUI or terminal screenshot evidence, and
   check the personal-verification box only after the owner verifies it.
3. If claiming speedup over then-current main, benchmark that exact baseline
   with matching inputs/settings; do not relabel the historical table above.
4. Check whether main has advanced, refresh the merge/tests if needed, and
   copy the proposed body into a PR only when the owner authorizes submission.

### Build and run

With the existing configured dependencies (no installation):

```bash
AGENTS_AGENT_MODE=1 cmake --build volume-cartographer/build \
  --target VC3D vc_fiber_trace_metric -j6
VC3D_LINE_MODEL_PREFETCH=1 VC3D_LINE_PERF_LOG=1 volume-cartographer/build/bin/VC3D
```

The benchmark runner is `scripts/benchmark_line_model_remote.py`; the exact
input paths and invocation are documented in the linked PHerc1299 record.

### Post-merge validation details

Tested code commit: `ed59e1bd90572f47d88be740310f855889937f5b`; subsequent
publication changes are documentation only. Ubuntu amd64, GCC 13.3,
RelWithDebInfo (`-O2 -g`, existing LTO/architecture flags unchanged), no installs
or solver precision changes. The build and GUI smoke finished before timing.

Same 3735-point/eight-control saved fiber, dirty span 6, prefetch enabled, 64
model readers, default legacy readers, 512 MiB configured cache, no pre-solve
corridor warmup. Each cold run starts with only manifests/remote markers; each
warm partner is a fresh process using the paired disk cache. The original fiber
hash above was rechecked and unchanged; the user's cache was not cleared.

| Post-merge check | Pairs | Cold solve mean (min / median / max), s | Disk-warm solve mean (min / median / max), s | Cold / warm downloaded MB mean |
|---|---:|---:|---:|---:|
| One native scoring thread |3|3.970 (3.585 / 4.061 / 4.264)|1.325 (1.304 / 1.325 / 1.345)|17.301 / 0.607|
| Four native scoring threads |1|3.182 (single sample)|0.388 (single sample)|16.742 / 1.166|

These are post-merge regression checks, **not** new before/after comparisons
against current main. The four-thread sample is only a batch-path smoke check,
not enough to estimate performance variability. All eight outputs have 2835
points, one native span, two native tails and zero fallback spans. Signatures
are compared within matching thread configurations, not across thread counts.
Executable/library hashes were stable throughout; every shared remote payload
matched. Optional warm-cache downloads are included rather than called zero-I/O.

One-thread stage means: span tracing 1.449 s, reinitialization 1.426 s, native
tails 1.094 s; reported normal preparation inside reinit 0.706 s and Ceres compute
about 0.003 s. Remaining latency is still largely model preparation. Full-process
means, including startup and the bounded speculative drain, were 4.722 s cold
and 1.520 s warm (four-thread sample: 3.953 / 0.806 s).

The 23-target CTest selection includes new cache/prefetch/queue coverage and main's
settings/project/metadata integration tests. Two legacy dataset-dependent
subcases in the optimizer target logged skips because `las008`/`5.json` fixtures
were absent; the PHerc1299 benchmark above ran with real data. This is not a
claim that the entire repository test suite, an interactive human GUI session,
or macOS/arm64 was tested after the merge.

Exact selected build and test commands:

```bash
AGENTS_AGENT_MODE=1 cmake --build volume-cartographer/build --target \
  VC3D vc_fiber_trace_metric test_model_prefetch test_lasagna_remote_prefetch \
  test_chunk_cache test_remote_file_cache test_lasagna_line_optimizer \
  test_lasagna_batch_window test_lasagna_manifest test_chunk_cache_persist \
  test_chunked_plane_sampler test_fiber_trace_review test_fiber_trace3d \
  test_lasagna_normal_sampler test_line_model_warmup_queue test_vc_settings \
  test_volume_pkg test_volume_pkg_more test_volume_pkg_full test_open_data_manifest \
  test_lasagna_project_volumes test_line_annotation_generated_views \
  test_line_annotation_optimization_queue test_surface_area_units \
  test_voxel_size_metadata -j6
AGENTS_AGENT_MODE=1 VC3D_CONFIG_DIR=/tmp/villa-line-prefetch-merge.VZlf7o/test-config \
  QT_QPA_PLATFORM=offscreen ctest --test-dir volume-cartographer/build --output-on-failure \
  -R '^(test_model_prefetch|test_lasagna_remote_prefetch|test_chunk_cache|test_remote_file_cache|test_lasagna_line_optimizer|test_lasagna_batch_window|test_lasagna_manifest|test_chunk_cache_persist|test_chunked_plane_sampler|test_fiber_trace_review|test_fiber_trace3d|test_lasagna_normal_sampler|line_model_warmup_queue|vc_settings|test_volume_pkg|test_volume_pkg_more|test_volume_pkg_full|test_open_data_manifest|test_lasagna_project_volumes|test_line_annotation_generated_views|line_annotation_optimization_queue|test_surface_area_units|test_voxel_size_metadata)$'
AGENTS_AGENT_MODE=1 python3 -m unittest discover \
  -s volume-cartographer/scripts/tests -p 'test_line_model*.py'
AGENTS_AGENT_MODE=1 VC3D_CONFIG_DIR=/tmp/villa-line-prefetch-merge.VZlf7o/gui-config \
  python3 volume-cartographer/apps/VC3D/agent_bridge/test/smoke_offscreen.py \
  --vc3d /home/djosey/Documents/villa/volume-cartographer/build/bin/VC3D --rpc-timeout 20
```

The two config directories each contain an isolated `VC3D.ini` setting
`viewer/remote_cache_dir` to a matching `test-cache` or `gui-cache` directory
under the evidence root, plus `project/auto_open=false` and
`project/show_open_data_catalog_on_startup=false`. Preserve that isolation when
rerunning. CMake dependency fetch/update was disabled for this configured build.
HTTP fixture tests need loopback access; the smoke test needs local GUI/socket
permissions but runs offscreen.

Exact three-pair remote check:

```bash
AGENTS_AGENT_MODE=1 python3 volume-cartographer/scripts/benchmark_line_model_remote.py \
  --output-dir /tmp/villa-line-prefetch-merge.VZlf7o/pherc1299-threads1 \
  --binary /tmp/villa-line-prefetch-merge.VZlf7o/merged-build/bin/vc_fiber_trace_metric \
  --fiber-json /tmp/villa-pherc1299-prefetch.IJmtzD/baseline-on/input/fiber.json \
  --fiber-manifest /tmp/villa-pherc1299-prefetch.IJmtzD/baseline-on/input/fiber/PHerc1299-20260309130042-las-sd1-aac02eb8.lasagna.json \
  --normal-manifest /tmp/villa-pherc1299-prefetch.IJmtzD/baseline-on/input/normal/PHerc1299-20260309130042-lasagna-20260724.lasagna.json \
  --dirty-span 6 --prefetch 1 --trials 3 \
  --compare-run /tmp/villa-pherc1299-prefetch.IJmtzD/candidate-final
```

For the recorded four-thread check, the command used output directory
`/tmp/villa-line-prefetch-merge.VZlf7o/pherc1299-threads4`, added `--threads 4`,
used `--trials 1`, and compared against
`/tmp/villa-guided-prefetch.ff9dfn/straight-threads4`. Despite that historical
scratch directory's name, the reference ran in **straight** mode. Never reuse
an existing output directory for a new trial; choose a new name.
