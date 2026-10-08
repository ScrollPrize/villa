# Inference-only tile repeatability — 2026-10-01

## Outcome

Same-batch repeats were bit-identical for all seven float output planes and all three encoded channels. Changing batch size changed predictions without any accumulation. Actual production inference uses **bf16**, not IEEE fp16. fp16 was substantially less sensitive in absolute error and final byte changes, but this does not establish improved tracing accuracy.

## Method and reproduction

Run `bash /home/hendrik/villa3/scripts/run_training_failure_mining` with action `reviewed_tile_repeatability` in scripts/training_failure_mining_control.json. Implementation: scripts/reviewed_tile_repeatability.py. Exclusive manager GPU0, 32 CPU cores; other GPUs unused by this test. Completed successfully.

Frozen raw u31500 checkpoint: `/ephemeral/hendrik/fiber3d_train2/evaluations/reviewed_fibers_20260930_v3/halfsigma_current_raw_v3/frozen/live_inference/_models/ffd7aff24441e82e53b5f3fef69ab19d24a7b632e1dee4dac6487401694d244d/snapshot.pt`.
SHA256: `ffd7aff24441e82e53b5f3fef69ab19d24a7b632e1dee4dac6487401694d244d`.
Four canonical 256³ CT tiles around earlier divergence probes: two Paris4 and two PHerc0139. Input level1, source-to-base scale2, overlap96/border32. Exact probes, tile origins, raw input and normalized input hashes are recorded in the companion JSON. Shared native reader, production preprocessing, model adapter and channel finalizer. Inputs normalized once, reused, and hashes rechecked at completion. No blending, accumulation, output cache publication or tracing. All voxels including borders and low-presence regions are compared (67,108,864 per plane).

GPU: NVIDIA H100 80GB HBM3; Torch 2.11.0+cu128; CUDA 12.8.
Backend flags: `{"cudnn_allow_tf32":true,"cudnn_benchmark":false,"cudnn_deterministic":false,"deterministic_algorithms":false,"matmul_allow_tf32":false}`.
Autocast-off returns float32 but retains TF32 convolution allowance: not a strict IEEE-fp32 control.
“Raw” means adapter-returned clamped float32 products; actual native model output dtypes confirmed bf16/fp16/float32 respectively.

## Full matrix

Values below are percentages of voxel values that differ against that precision's baseline (B4 for bf16/fp16, B1 for off), except the explicitly marked cross-precision rows. B3 means batches [3,1]. Reversed batches are realigned before comparison.

| Precision | Comparison | Raw presence changed | Presence bytes changed | nx bytes changed | ny bytes changed |
|---|---|---:|---:|---:|---:|
| bf16 | repeat1 | 0.000000% | 0.000000% | 0.000000% | 0.000000% |
| bf16 | repeat2 | 0.000000% | 0.000000% | 0.000000% | 0.000000% |
| bf16 | batch1 | 18.654546% | 3.274107% | 25.565512% | 19.749424% |
| bf16 | batch2 | 18.654546% | 3.274107% | 25.565512% | 19.749424% |
| bf16 | batch3 | 5.752689% | 1.084238% | 7.785012% | 6.463683% |
| bf16 | reverse_batch4 | 0.000000% | 0.000000% | 0.000000% | 0.000000% |
| bf16 | repeat_after_shapes | 0.000000% | 0.000000% | 0.000000% | 0.000000% |
| fp16 | repeat1 | 0.000000% | 0.000000% | 0.000000% | 0.000000% |
| fp16 | repeat2 | 0.000000% | 0.000000% | 0.000000% | 0.000000% |
| fp16 | batch1 | 20.546758% | 0.416853% | 4.136322% | 3.131342% |
| fp16 | batch2 | 20.546758% | 0.416853% | 4.136322% | 3.131342% |
| fp16 | batch3 | 10.275836% | 0.232835% | 1.940010% | 1.506589% |
| fp16 | reverse_batch4 | 0.000000% | 0.000000% | 0.000000% | 0.000000% |
| fp16 | repeat_after_shapes | 0.000000% | 0.000000% | 0.000000% | 0.000000% |
| off | repeat1 | 0.000000% | 0.000000% | 0.000000% | 0.000000% |
| off | repeat2 | 0.000000% | 0.000000% | 0.000000% | 0.000000% |
| fp16 | versus_bf16_baseline | 97.977915% | 8.930379% | 37.650570% | 31.869228% |
| off | versus_bf16_baseline | 99.999768% | 8.767639% | 37.358147% | 31.631126% |

## Error magnitude and rounding

B1 versus B4:
- bf16 presence: raw MAE 0.00012872125, max 0.0078125 (0–1 scale); encoded MAE 0.0328092 byte units, max 2 bytes.
- fp16 presence: raw MAE 0.00001630626, max 0.001220703125; encoded MAE 0.00416853 byte units, max 1 byte.
- In every comparison, zero presence voxels had equal raw outputs but different encoded bytes. Rounding itself was deterministic. It hides many float differences, while boundary crossings turn small differences into discrete byte changes.
- nx/ny include nonlinear direction reconstruction and hemisphere encoding, not simply independent rounding of model channels. Large byte differences are not angular errors; this test does not calculate angular error or restrict metrics to high-presence voxels.

Fixed-batch bf16/fp16 repeats, reversed order and repeats after shape changes were exactly equal across all seven raw planes. Both autocast-off B1 repeats also exactly matched.
The fp16 versus bf16 cross-precision row compares B4 with B4. The off versus bf16 row compares B1 with B4 and therefore mixes precision and batch-shape effects; do not use it as a precision-only attribution.

This demonstrates an upstream batch-shape dependence, independent of accumulation, that can contribute to dynamic-batch end-to-end variation. It does not isolate the responsible GPU operation/kernel, measure cross-GPU or fresh-process repeatability, or establish a better accuracy setting. No production defaults changed.

## Validation and artifacts

`/home/hendrik/.venv_las/bin/python -m pytest -q scripts/test_reviewed_tile_repeatability.py scripts/test_reviewed_accumulation_sweep.py`: **6 passed**. Tests cover signed byte differences, exact equality, nonfinite rejection, rounding boundaries, canonical tile selection and existing accumulation configuration. `git diff --check` passed.

Raw results: `/ephemeral/hendrik/fiber3d_train2/evaluations/reviewed_fibers_20260930_v3/tile_repeatability_u31500_v1/results.json`.
Companion repository JSON preserves all plane metrics, backend flags and input hashes. Per-row wall time includes finalization/comparison and is not GPU inference latency.
Independent plan/code review completed by profile_plan_review; its precision, raw-output and batch-confounding caveats are reflected above.

