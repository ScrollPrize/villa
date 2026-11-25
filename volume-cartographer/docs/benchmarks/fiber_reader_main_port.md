# Streaming and GPU chunk tracing across the vc_reader migration

## Purpose and revisions

This is a real-data integration/performance check of the continuous full-volume
tracer and GPU chunk constructor after moving the streaming FW/BW changes onto
main. It is not a new model or tracing-parameter sweep.

- Before: `c2305418e`, based on `8ff06f762`, using the old decoded-chunk cache.
- Main base: `9c343a164`.
- After: code at `0750edcad` (GPU runs at documentation-only descendant
  `3d8f60cf7`), including the FW/BW cherry-pick `a178d9c3a` and a CUDA
  build fix. The CUDA targets now consume `vc_reader::headers` and C++20; the
  device source includes the narrow shared sampling-math header. Arithmetic
  flags, including `--fmad=false`, were not changed.

The sole cherry-pick conflict was the streaming progress report: main removed
the obsolete asynchronous CT-suspension counter. The port retains main's reader
implementation and adds the new reverse-check counters, not the old reader.

## Workload and method

Measured on 2026-11-05, Ubuntu x86-64, 96 physical / 192 logical CPU cores,
approximately 1.5 TiB RAM and RTX PRO 6000 Blackwell GPUs. Both native builds use
GCC 13.3, RelWithDebInfo and LTO. CUDA builds use CUDA 13.4 and architecture 120.
These are profiling-build measurements, not Release performance claims.

Input is the completed PHerc0332 prediction:

```
/data/hendrik/las_manager_inference/PHerc0332-20251211183505-las-sd1-4c4f7467/artifacts/PHerc0332-20251211183505-las-sd1-4c4f7467.zarr
```

It stores seven byte channels, including presence, direction, width and normal,
in 64-cubed spatial chunks. The selected level has four base voxels per stored
voxel. CT checks use PHerc0332/20251211183505, level 2, through the normal local
source cache. No neural inference runs in this benchmark.

The fixed workload is 32 contiguous tracing cores (512 base voxels per core):
X=8..11, Y=15..18, Z=5..6. This 4x4x2 group was selected once for the largest
total saved-page size in the completed PHerc0332 production run. The same keys,
seeds, predictions and parameters are used before/after.

Continuous tracing uses 96 workers, 2,240 active-fiber slots, eight seed workers,
32 open tiles and a maximum of 4,096 base voxels per direction. It includes the
production delayed FW/BW checks: interval 512, four-edge reverse history,
convergence after 1,024 base voxels within 8 base voxels / 10 degrees, one
converged point. Width modulation is cosine 30%, with a hard width cutoff of
20% and joint modulated-presence cutoff 0.25.

GPU construction requests 16 CPU workers and eight contexts per worker; the
manager grants seven GPUs (physical indices 1-7), 4 GiB VRAM each and 14 CPU
workers for every measured run. Physical GPU 0 is excluded by the local-node
allowlist. No other task is stopped and no manager journal is reset. It keeps
the same chunk geometry and cosine modulation but disables reverse replay and
the hard width cutoff, which that backend does not implement. This is a matched
GPU-before/GPU-after comparison, not a CPU/GPU algorithm equivalence claim.
Chunk consistency filtering remains enabled with the existing 10/20 ellipse.

Each process starts with an empty decoded cache and fresh output, while the
filesystem/source caches are warm. Three repeats per mode are requested; normal
and pressure order reverses on alternating repeats. Before runs precede after
runs. Native wall time excludes Python startup/teardown; complete process time
is recorded separately. Builds do not overlap the after measurements. The first
old CPU repeat briefly overlapped the end of compilation; its other repeats
did not. No system cache flush or global service interruption is performed.

Normal cache cap is 64 GiB total; pressure cap is 1 GiB total. Main creates
independent prediction, normal and CT caches, so the harness divides this total
by three for streaming and by two for GPU construction. The old implementation
has a shared decoded cache. These caps exclude working tensors and live chunk
references. The removed legacy `pin_sampling_region` setting is omitted only
for the main-based configuration.

## Results

Three unprofiled repeats per cell, native wall time in seconds:

| Pipeline / total cache cap | Before mean (min / median / max) | After mean (min / median / max) | Median change |
| --- | --- | --- | --- |
| Continuous tracer / 64 GiB | 56.123 (54.822 / 55.637 / 57.910) | 61.950 (57.396 / 62.082 / 66.372) | +11.6% |
| Continuous tracer / 1 GiB | 62.674 (61.370 / 62.067 / 64.585) | 72.798 (70.130 / 70.366 / 77.898) | +13.4% |
| GPU chunk construction / 64 GiB | 9.299 (9.146 / 9.232 / 9.517) | 5.448 (5.181 / 5.512 / 5.650) | -40.3% |
| GPU chunk construction / 1 GiB | 9.521 (9.449 / 9.466 / 9.647) | 6.764 (6.412 / 6.854 / 7.026) | -27.6% |

This port works for streaming but does not preserve the old throughput. Mean
CPU time rises from 4,051 to 4,841 CPU-seconds with ample cache, and from 4,935
to 6,072 under pressure. It is not merely a slower startup.

Output means across the three repeats:

| Mode | Before / after pieces | Before / after retained base voxels | Before / after length-weighted modulated presence |
| --- | --- | --- | --- |
| 64 GiB | 822.7 / 817.7 | 1,023,013 / 1,023,509 | 0.76650 / 0.76647 |
| 1 GiB | 825.0 / 824.0 | 1,029,885 / 1,025,997 | 0.76664 / 0.76670 |

All runs see the same 14,780 seed candidates. Retained-length means differ by
less than 0.4%; collision scheduling makes inventories non-identical. These
figures do not establish equal GT accuracy.

All GPU runs construct 21,261 paths and flag the same 942 as conflicting. Saved
coordinates, endpoint reasons, seeds, maxima and conflict flags match exactly
between before and after for each of the six paired runs (192 chunk pairs).
This checks parsed values rather than compressed-file bytes: timing metadata
and backend configuration necessarily differ. The preserved startup-failure
directories contain no benchmark results and are excluded.

The GPU times include CPU preparation, GPU tracing, conflict filtering and
writing all 32 fresh output chunks; they are not pure CUDA kernel times. Mean
sums of per-chunk elapsed timers in the 64-GiB runs:

| Timer | Before summed seconds | After summed seconds |
| --- | --- | --- |
| Seed construction | 5.452 | 4.407 |
| GPU packing | 14.357 | 6.384 |
| GPU kernel events | 0.762 | 0.816 |
| Conflict filtering, including normal reads | 82.102 | 41.327 |

These timers overlap across workers and are not an additive wall-time
decomposition. The kernel timer is essentially unchanged; the complete chunk
workflow is faster in preparation/filtering, not because its tracing math was
reduced. Conversely, the streaming CPU workflow remains slower. This comparison
spans the reader migration; it is not a measurement of stock current main
versus only the FW/BW patch.

## Cache behavior under test

The old implementation uses background eviction, reserve watermarks and
compact missing-chunk tracking. Its three 1-GiB streaming runs evicted about
136-141 GiB each, with 326-339 hard-limit passes: this genuinely exercises
eviction rather than merely allocating a small configured cache.

Main's `vc_reader` uses a mutex-protected, byte-bounded LRU with synchronous
tail eviction. Missing entries are charged 64 bytes and share the LRU. It does
not have the old background-watermark/bitmap design or its progress counters.
Its tail eviction is O(1); it is not the former all-entry eviction scan. The
pressure benchmark therefore tests the replacement implementation, not a
claim that all old cache optimizations survived the migration unchanged.

### Separate CPU profiles

One additional pressure run per revision was sampled with `gprofng collect app
-p 10 -a off`; these runs are excluded from the timing table. The profiles and
complete function reports are in `before-profile` and `after-profile` below the
artifact directory. Exclusive sampled CPU attribution:

| Work | Before CPU seconds (% of total) | After CPU seconds (% of total) |
| --- | --- | --- |
| Principal compact-tensor axis calculation | 607.77 (13.70%) | 707.33 (12.31%) |
| Compact-tensor corner accumulation | 404.90 (9.13%) | 658.44 (11.46%) |
| All libzstd frames | 102.07 (2.30%) | 599.88 (10.44%) |
| Entire sampled process | 4,434.82 (100%) | 5,745.87 (100%) |

These selected rows do not partition the whole process. Inlining and LTO affect
function attribution. The substantial extra decompression is direct evidence
of increased reader/cache work under pressure; it is not evidence that mutex
contention alone explains the regression. Separate prediction/normal caches
can decode the same packed-channel chunks independently, and partitioned caps
cannot share spare capacity as the old global cache did. Their individual
contributions have not been isolated. CPU sampling does not measure time spent
asleep on a lock.

## Reproduction and evidence

Host-local harnesses (not production code):

```
/home/hendrik/villa3/local/benchmark_main_port.py
/home/hendrik/villa3/local/report_main_port.py
```

Run from the original checkout, keeping the two builds separate:

```sh
/home/hendrik/.venv_las/bin/python local/benchmark_main_port.py before \
  volume-cartographer/build/fiber-stream-data-host --kind stream
BENCH_REPO=/home/hendrik/villa3-main-port \
  /home/hendrik/.venv_las/bin/python local/benchmark_main_port.py after \
  /home/hendrik/villa3-main-port/volume-cartographer/build/reader-port --kind stream
```

The corresponding GPU commands use `--kind gpu` and manager admission.
Completed runs are skipped; incomplete outputs are rejected, not silently
treated as completed measurements. Full commands, selected keys, input configs,
revisions, native progress logs, per-process CPU/I/O samples, output geometry
and summaries are retained under:

```
/data/hendrik/fiber3d_train2/evaluations/reader_port_20261105_v1/
```

## Correctness and limitations

Both builds pass 108 native test cases across trace, chunk, reverse-consistency
and reference-replay suites. The old build passes 25 streaming/evaluation Python
tests; the main-based build passes 44 streaming/evaluation/chunk-workflow Python
tests. Main's native reader cache and cached-sampling tests also pass.

Example Python validation from the main-based checkout:

```sh
PYTHONPATH=vesuvius/src:vc_reader/src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/hendrik/.venv_las/bin/python -m pytest -q \
  volume-cartographer/python/tests/test_fiber_stream_volume.py \
  volume-cartographer/python/tests/test_fiber_evaluation.py \
  volume-cartographer/python/tests/test_fiber_chunk_volume.py \
  --vc-build-python volume-cartographer/build/reader-port/python
```

Continuous tracing has scheduling-dependent live collision ownership, so
parallel repeats do not produce identical piece inventories even on the same
revision. Compare retained length, piece counts and fit scores alongside speed;
do not label these piece-count differences as GT failures. This 32-core stress
test is not a full-scroll duration measurement, a cold-network benchmark, or a
substitute for the reference-fiber accuracy benchmark. Only Linux/amd64/CUDA was
exercised here; macOS and arm64 were not tested.
