# Float32/TF32-allowed inference batch sweep — 2026-10-01

## Method

Follow-up to reviewed_tile_repeatability_20261001.md. Same frozen raw u31500 model, same four 256³ tiles (two Paris4, two PHerc0139), shared native reader, production normalization and finalizer. Identical normalized GPU tensors reused; hashes verified after the test. No accumulation/blending/publication/tracing. Exclusive GPU0, 32 CPU cores. Autocast disabled; native model outputs confirmed torch.float32. TF32 convolutions permitted, not proof that particular kernels used TF32.

Backend flags: {"cudnn_allow_tf32":true,"cudnn_benchmark":false,"cudnn_deterministic":false,"deterministic_algorithms":false,"matmul_allow_tf32":false}.
Checkpoint SHA256: ffd7aff24441e82e53b5f3fef69ab19d24a7b632e1dee4dac6487401694d244d.
GPU: NVIDIA H100 80GB HBM3, torch 2.11.0+cu128, CUDA 12.8.

Baseline batch1 [1,1,1,1]. Compare two baseline repeats, batch2 [2,2], batch3 [3,1], batch4 [4], reversed invocation order at batch1, then final baseline repeat. All output voxels including borders and low presence are counted: 67,108,864 per plane. Raw means clamped adapter output, not preclamp tensors.

## Results against batch1

| Comparison | Raw presence changed | Raw presence MAE (0–1) | Raw presence max (0–1) | Presence bytes changed | nx bytes changed | ny bytes changed |
|---|---:|---:|---:|---:|---:|---:|
| repeat1 | 0.000000% | 0.00000000e+0 | 0.00000000e+0 | 0.000000% | 0.000000% | 0.000000% |
| repeat2 | 0.000000% | 0.00000000e+0 | 0.00000000e+0 | 0.000000% | 0.000000% | 0.000000% |
| batch2 | 99.986750% | 1.77110099e-5 | 6.71297312e-4 | 0.455002% | 4.233971% | 3.197567% |
| batch3 | 74.989974% | 1.26384772e-5 | 6.40213490e-4 | 0.325254% | 3.350468% | 2.349560% |
| batch4 | 99.986896% | 1.80278798e-5 | 6.35534525e-4 | 0.463176% | 4.106352% | 3.129788% |
| reverse_batch1 | 0.000000% | 0.00000000e+0 | 0.00000000e+0 | 0.000000% | 0.000000% | 0.000000% |
| repeat_after_shapes | 0.000000% | 0.00000000e+0 | 0.00000000e+0 | 0.000000% | 0.000000% | 0.000000% |

Every fixed-batch repeat, reversed batch1 order and final baseline repeat was exactly equal across all seven raw planes and all three encoded channels. Every batch-shape comparison had maximum presence-byte difference 1. No identical raw presence values produced unequal encoded bytes.

Nearly all float32 values change at different batch shapes, but their absolute differences are small; unequal-value fraction alone should not be compared between precisions as an error magnitude. Batch4-versus1 presence MAE is 0.00001802788 and encoded change rate 0.463176%, versus previous fp16 0.00001630626 / 0.416853% and bf16 0.00012872125 / 3.274107% for the same B1/B4 pairing. Thus the TF32-allowed float32 path is similar in magnitude to fp16 and substantially less batch-sensitive than bf16 on these inputs, not bitwise batch invariant.

The lower B3 aggregate partly reflects its unchanged B1 remainder tile. nx/ny include nonlinear direction reconstruction/hemisphere mapping; byte differences are not angular errors. No strict TF32-disabled experiment, cross-GPU/fresh-process experiment or downstream tracing accuracy comparison was performed. Production defaults unchanged.

## Reproduction and validation

Control action `reviewed_tile_float32_sweep` in scripts/training_failure_mining_control.json, then approved command:

```bash
bash /home/hendrik/villa3/scripts/run_training_failure_mining
```

This dispatches scripts/reviewed_tile_repeatability.py --float32-sweep through the existing GPU manager; preserves prior results in a separate directory. Completed exit0.
Raw output: /ephemeral/hendrik/fiber3d_train2/evaluations/reviewed_fibers_20260930_v3/tile_repeatability_float32_u31500_v1/results.json.
Companion repository JSON contains input hashes/origins and per-plane MAE/RMSE/max. Row wall time includes CPU finalization/comparison; not inference latency.

`/home/hendrik/.venv_las/bin/python -m pytest -q scripts/test_reviewed_tile_repeatability.py scripts/test_reviewed_accumulation_sweep.py`: 6 passed.
Independent follow-up review passed. `git diff --check` clean.

