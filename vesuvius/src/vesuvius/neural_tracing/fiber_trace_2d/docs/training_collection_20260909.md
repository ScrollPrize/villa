# Five-scroll JSON fiber training

Config: `fiber_trace_3d/configs/train_collection_20260909_5scroll_16x8.json`.
Based on `train_s1a_nml_8gpu_fast.json`; dataset lists, run name, explicit
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
PHerc0332's test length is ~9.5%, because whole fibers are retained. Sampling
remains the existing control-point-based scheme, not new equal-scroll weighting.
The source symlink targets must remain available. The copied SOURCES.md's old
0/996 parser-compatibility report is obsolete: current parser accepts all 996.

## Launch

```bash
source /home/hendrik/.venv_las/bin/activate
CONFIG=/home/hendrik/villa/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/train_collection_20260909_5scroll_16x8.json
python -m vesuvius.neural_tracing.fiber_trace_3d.train "$CONFIG"
```

This is the managed entry point, not torchrun. A measured calibration matching
this config is required first; none existed when the config was created. See
[cooperative training](cooperative_training.md) for the exclusive calibration
command and GPU UUID/budget arguments. Do not reuse or relabel an old config's
calibration. Training auto-discovers the new report under the run root's
`.memory-calibrations/`. The manager auto-start/report discovery improvements
were present but still uncommitted when this config was created.

## Validation

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=vesuvius/src /home/hendrik/.venv_las/bin/python -m pytest -q vesuvius/tests/neural_tracing/test_collection_training_config.py
```

Checks baseline settings, parser acceptance, counts, declared volume provenance
and resolved-file split disjointness. Dataset audit skips on machines without
this local snapshot. No GPU training, volume downloads or image/alignment visual
inspection were performed for this config-only task.
