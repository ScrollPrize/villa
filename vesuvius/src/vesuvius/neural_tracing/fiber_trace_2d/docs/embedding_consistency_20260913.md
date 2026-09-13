# Embedding tile consistency — 2026-09-13

Raw checkpoint: update **5,300**, **678,400 samples**.
SHA256: `f4e261f81de165a76bde523b7d864b6001ddefa543037ce958590cbf6e8ee258`.

Ten held-out JSON fibers (two per scroll), the lexicographically first eligible files; each contributes a centered 512-base-voxel GT arc sampled every4 base voxels (129 points; about1.23mm per window at2.4µm). This is a deterministic pilot, not all50 fibers.

Input level1;256³ input tiles, overlap64, border16; native output stride4 (output/base factor8), stored-chunk size32 output voxels. BF16 model, existing native VC3D CT reader and inference preprocessing, shared canonical tile geometry and Gaussian-downsampled blend ramp. All contributors to all eight interpolation corners are blended before trilinear sampling and L2 normalization.

This diagnostic blends signed raw vectors in FP32. Embedding publication does not exist yet, so this is NOT an f16 accumulator/Zarr roundtrip. Storage-boundary statistics classify the continuous field at hypothetical chunk boundaries; they do not test storage serialization. Tile samples with blend weight≤0.05 are excluded from same-tile/cross-tile comparison, not from blending.

## Cosine consistency

Means below use matched4-base-voxel arclength separation. “Switch” means a change of dominant contributor at the largest-weight interpolation corner (a proxy for the blend handoff, not a discontinuous field selection).

| Scroll | Same point, overlapping tiles | Within one tile | Blended interior | Blended switch | Storage crossing | Switch pairs |
|---|---:|---:|---:|---:|---:|---:|
| PHerc0139 | 0.999991 | 0.999963 | 0.999980 | 0.999998 | 0.999964 | 4 |
| PHerc0332 | 0.999995 | 0.999986 | 0.999983 | 0.999993 | 0.999995 | 2 |
| PHerc1299 | 0.999996 | 0.999981 | 0.999979 | 0.999999 | 0.999971 | 2 |
| PHerc1451 | 0.999995 | 0.999996 | 0.999996 | 0.999999 | 0.999995 | 3 |
| PHercParis4 | 0.999991 | 0.999988 | 0.999990 | 0.999992 | 0.999994 | 4 |
| Equal-scroll mean | 0.999994 | 0.999983 | 0.999986 | 0.999996 | 0.999984 | — |

## Longer along-fiber comparisons

Equal-scroll means, pooling pairs within each scroll. All distances are GT arclength separations, not straight-line displacement.

| Separation (base voxels) | Within tile | Blended interior | Blended switch |
|---:|---:|---:|---:|
| 4 | 0.999983 | 0.999986 | 0.999996 |
| 16 | 0.999979 | 0.999981 | 0.999990 |
| 64 | 0.999976 | 0.999981 | 0.999949 |
| 128 | 0.999982 | 0.999985 | 0.999962 |

## Off-fiber controls: consistency is not separation

Controls lie >80 base voxels from the entire selected GT polyline, conservatively outside its20/80 exclusion ellipse. They are unlabeled: they can be background or other fibers. This is not exactly the training candidate distribution. Each control is compared with a GT point from the same tile. High-presence means predicted presence≥0.5.

| Scroll | Controls | Cosine>0.9 | High-presence controls | High-presence cosine>0.9 | High-presence cosine p05 / median / p95 |
|---|---:|---:|---:|---:|---|
| PHerc0139 | 2161 | 27.3% | 154 | 31.8% | -0.9941 / -0.8511 / 1.0000 |
| PHerc0332 | 1832 | 25.6% | 215 | 28.4% | -0.9859 / -0.9140 / 1.0000 |
| PHerc1299 | 1141 | 26.2% | 142 | 24.6% | -0.9849 / -0.8818 / 1.0000 |
| PHerc1451 | 1400 | 23.9% | 96 | 26.0% | -0.9899 / -0.8542 / 1.0000 |
| PHercParis4 | 1365 | 25.5% | 209 | 23.4% | -0.9954 / -0.8925 / 1.0000 |

Across the10 window-mean embedding directions, pairwise cosine min/median/max = 0.999982 / 0.999993 / 0.999999. Cross-window similarity is not an objective violation by itself (training is patch-local), but near-identical GT directions plus high-cosine off-GT controls warn against treating continuity as fiber identity.

## Norms and blending

| Scroll | GT norm mean | Corner cancellation mean | Worst corner cancellation | Zero GT vectors |
|---|---:|---:|---:|---:|
| PHerc0139 | 32.320 | 0.996116 | 0.804819 | 0 |
| PHerc0332 | 30.933 | 0.990396 | 0.663128 | 0 |
| PHerc1299 | 39.029 | 0.999887 | 0.994647 | 0 |
| PHerc1451 | 40.836 | 0.999175 | 0.974966 | 0 |
| PHercParis4 | 33.817 | 0.998524 | 0.924959 | 0 |

Cancellation is ||Σw e||/Σw||e|| at all eight support corners, including corners with zero trilinear coefficient. Thus worst-corner values need not imply a bad actual GT query; GT norm/continuity are reported separately.

## Interpretation

- This pilot shows no appreciable GT discontinuity at tile handoffs. Vector blending is not currently the main obstacle indicated by these samples.
- Many off-GT points remain almost indistinguishable from GT in cosine. The representation may primarily separate broad foreground/background groups rather than individual neighboring fibers. This is a warning, not a demonstrated tracing failure or proof of global collapse.
- Review these results before enabling embedding tracing. A local similarity score can be continuous yet offer little discrimination.

## Reproduction and validation

```bash
/home/hendrik/villa/vesuvius/scripts/run_fiber3d_experiment embedding-consistency
```

Settings are in `vesuvius/scripts/fiber3d_experiment.json` under `embedding_consistency`; use a new output directory for a fresh run. Use the frozen snapshot for exact checkpoint reuse. The launcher uses the existing exclusive, non-pausable manager command; running training releases its GPUs and resumes after analysis.

Measured analysis wall time: **48.3s**, one GPU, 48 tile forwards. Includes source opening/CT reads/forward/CPU metrics, excludes manager waiting and initial checkpoint/model load. This is not a throughput benchmark.

CPU synthetic tests: affine interpolation/arclength, identical-tile seam invariance, opposing-tile disagreement/cancellation (3 passed).

Raw per-fiber points, vectors, cosine samples, distributions, tiles and settings: `/ephemeral/hendrik/fiber3d_train2/evaluations/embedding_consistency_20260913_v1/results.json`.
No tracer or training behavior was changed. macOS/arm64 and embedding storage precision were not tested; only this Ubuntu/CUDA environment was measured.
