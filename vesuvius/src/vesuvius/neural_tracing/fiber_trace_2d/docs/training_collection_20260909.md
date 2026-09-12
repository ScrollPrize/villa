# Five-scroll JSON fiber training

## S1a-only comparison using current training

`fiber_trace_3d/configs/train_s1a_nml_16x8.json` changes data, run name and
test/checkpoint intervals from the current collection setup. Training uses the eight original
`fibers_s1a_*.nml` files and the established inverse affine into the Paris4 SD1
scan. Testing uses the original held-out `/ephemeral/hendrik/fiber3d_train/data/test/*.json`
without that affine, not the five-scroll test collection.

Retains GroupNorm/BF16, stride-4 output, current augmentations, LR0.01, local
batch16/global128, 2M updates and test/checkpoint interval100. No embeddings.
Both splits use dense supervision and continuous arc-length sampling within
PHercParis4 (the only scroll). This reproduces the old data selection, **not**
the old control-point sampling, normalization, or CP-only JSON test metric.
Manager startup/calibration/adaptation is unchanged; no manual resource setup.

```bash
source /home/hendrik/.venv_las/bin/activate
CONFIG=/home/hendrik/villa/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/train_s1a_nml_16x8.json
python -m vesuvius.neural_tracing.fiber_trace_3d.train "$CONFIG"
```

Creates a fresh `s1a_nml_128_sd1_16x8_<timestamp>` run under
`/ephemeral/hendrik/fiber3d_train2/runs`. No resume weights by default.

## Five-scroll collection

Config: `fiber_trace_3d/configs/train_collection_20260909_5scroll_16x8.json`.
Based on `train_s1a_nml_8gpu_fast.json`; sampling strategy, dataset lists, run name, explicit
`embedding.enabled=false` and test/checkpoint intervals (1000 instead of 100)
differ. Initial evaluation remains enabled. LR 0.01, 100-step warmup, SD1 input (half
resolution), 128³ patches, GroupNorm/BF16, stride-4 output, target 16 samples/GPU
and effective batch 128. The manager may change physical parallelism while
preserving that effective batch. Logs go to `/ephemeral/hendrik/fiber3d_train2/runs`.

Snapshot root:
`/ephemeral/hendrik/fiber3d_train/2026-09-09_x4_collection/split_95_5_length_seed20260910`.

| Directory | Actual scroll / scan | Train | Test |
|---|---|---:|---:|
| PHerc0139 | PHerc0139 / 20260102150214 | 197 | 10 |
| PHerc0332 | PHerc0332 / 20251211183505 | 74 | 3 |
| PHerc1229 | **PHerc1299** / 20260309130042 | 77 | 5 |
| PHerc1451 | PHerc1451 / 20260319101107 | 5 | 2 |
| PHercParis4_eval_fibers | PHercParis4 / 20260411134726 | 463 | 24 |
| PHercParis4_fibers | PHercParis4 / 20260411134726 | 130 | 6 |

All 82 PHerc1229 files explicitly identify PHerc1299 in their embedded source
URLs. PHerc0332 is identified by traced-segment manifest paths; not every file
contains provenance. JSON coordinates are base-volume coordinates, so no NML
transform is applied. No NML files or normal manifests are used.

Preserves the supplied split exactly (946 train, 50 test), including both Paris4
collections. It is a per-source fiber-length split, not spatially disjoint;
74 branch references cross splits. The loader does not expand branch references.
PHerc0332's test length is ~9.5%, because whole fibers are retained.
The source symlink targets must remain available. The copied SOURCES.md's old
0/996 parser-compatibility report is obsolete: current parser accepts all 996.

## Launch

```bash
source /home/hendrik/.venv_las/bin/activate
CONFIG=/home/hendrik/villa/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/train_collection_20260909_5scroll_16x8.json
python -m vesuvius.neural_tracing.fiber_trace_3d.train "$CONFIG"
```

This is the managed entry point, not torchrun. Startup measures memory with real
training/evaluation under exclusive ownership when no matching calibration is
available, then registers adaptive training. See
[cooperative training](cooperative_training.md). Do not relabel an old config's
calibration after changing supervision.

## Dense supervision and balanced sampling (2026-09-11)

The collection now uses `sampling_strategy: "scroll_balanced_arclength"`.
Every train/test dataset has `dense_supervision: true` and an explicit
`scroll_id`. Each scroll has probability 1/5, including Paris4's two folders
combined. Within that scroll a continuous distance is uniform over the combined
polyline lengths: a fiber twice as long is twice as likely. Control-point and
dense-vertex counts do not determine probability. Lengths are in base-volume
coordinates, independent of selected Zarr level and augmentation.

The patch surrounds the interpolated location and uses the same geometric and
value augmentation pipeline as before. Only its selected fiber supplies labels:
all its line segments intersecting the patch (including later re-entry) produce
dense centerline direction/presence supervision. Unselected fibers are not added
to positives. Other output cells remain negative, as with NML training.
This is not a radius-expanded positive tube.

The same loader, queue and target materializer serve managed/unmanaged training,
testing and prefetch. `arc_sampling.py` stores cumulative segment/fiber lengths;
draws depend on seed and data index, not rank, batch size or worker scheduling.
Augmentation retains its separate stream-index seeds. No global sample list is
constructed. Segment AABBs are rejected in a vectorized pass before exact
clipping, avoiding a Python clipping loop over the entire fiber.

Testing disables augmentation and repeats the same fixed location range every
time. `test_control_points: 0` retains the held-out CP count (1942) as its sample
budget, but these are now 1942 continuous sampled locations, not exhaustive CP
coverage. A positive count sets another fixed budget.

Compatibility: other configs retain `sampling_strategy: "control_points"`.
JSON `dense_supervision` defaults false; NML is always dense. Balanced mode
requires dense labels, nonempty `scroll_id` and positive finite length for every
fiber. Repeated vertices are harmless. Explicit `sample_mode="flat"` still
selects CPs for CP-based tools. Continuous samples report `control_point_index=-1`.

The already-running process is not changed. Restart to adopt these settings.
Earlier CP-only test losses/best-checkpoint scores are **not comparable** with
the new dense/balanced evaluation; use a separate run for a clean comparison.

## Sampling regression tests

Set `suites` to `["arc-sampling", "sampling-regression"]` in
`vesuvius/scripts/fiber3d_cooperative_tests.json`, then run the fixed caller:

```bash
/home/hendrik/villa/vesuvius/scripts/run_fiber3d_cooperative_tests
```

These suites are CPU-only (CUDA hidden); no manager allocation or GPU job is
started. The local collection parser audit skips if the data is unavailable.

For focused GPU integration, select `suites: ["cuda-arc"]` and one GPU device
in that same control file, then use the same launcher. It acquires exclusive
manager ownership before launching tests and releases after exit. The test uses
the collection U-Net with synthetic images/continuous fiber locations at 128^3,
batch 1, stride 4, FP32/BF16: CPU/CUDA targets, backward/update and repeatable
evaluation. Allocator memory is capped at 4 GiB. This is not a throughput test.

## Validation

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=vesuvius/src /home/hendrik/.venv_las/bin/python -m pytest -q vesuvius/tests/neural_tracing/test_collection_training_config.py
```

Checks baseline settings, parser acceptance, counts, declared volume provenance
and resolved-file split disjointness. Dataset audit skips on machines without
this local snapshot. No GPU training, volume downloads or image/alignment visual
inspection were performed for this config-only task.
