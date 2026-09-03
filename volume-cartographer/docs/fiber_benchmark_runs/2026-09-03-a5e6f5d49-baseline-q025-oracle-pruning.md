# Baseline quality-threshold pruning: 2026-09-03

## Provenance

| Item | Value |
| --- | --- |
| Villa base revision | `a5e6f5d498a67ff7a2a400314f2bfd96260df5e8` plus the uncommitted quality-threshold change recorded with this document |
| Build | CMake `Release` |
| Binary SHA-256 | `6d321398b2b4a6899a14fb96446d416be4c700a6a942ba86c7a0117c904a7b9a` |
| Repetitions | 1 per threshold |

## Inputs

| Item | Value |
| --- | --- |
| Complete ordinary traces | `$VES/data/workdir3/crop_traces.zarr` |
| Trace `.zattrs` SHA-256 | `78fb6aff07c5490051382e250b4cc8458cb90e0b1ed86e6e4be6e1f3c478c5b3` |
| Crop, base XYZ | `10240,22016,6144` to `11264,23040,7168` |
| Normals | `$VES/data/lasagna3d_inf/las008_s1_full/las_008.lasagna.json` |
| Normal manifest SHA-256 | `77834e54d9e2dfde4c10b6dba8610ba881a5ad509d1c82d6409346a931e3aa29` |
| References | `$VES/data/test_datasets/2026-09-01_fiber_stack2`, tag `hendrik_crop1` |
| Selected-run stdout SHA-256 | `e046eb134299ad411b21e5f49155df77d2f1d559880a593bdf33046f9361630c` |

## Command

```bash
/usr/bin/time -f 'wall_s=%e user_s=%U sys_s=%S max_rss_kib=%M' \
  volume-cartographer/build/bin/vc_fiber_trace_chunk direction-ablation \
  "$VES/data/workdir3/crop_traces.zarr" \
  --normal-manifest "$VES/data/lasagna3d_inf/las008_s1_full/las_008.lasagna.json" \
  --output /tmp/baseline-load-q025-fibers \
  --direction-dominance 0.9 \
  --piece-length 512 \
  --bp-only \
  --bp-inference sum-product-mixed \
  --quality-threshold 0.25 \
  --winding-fixed-orientation \
  --reference-prune-offenders \
  --reference-prune-policy oracle-inliers \
  --reference-oracle-magnitude-weight 1 \
  --reference-oracle-accept-message-limit \
  --reference-fiber-dir "$VES/data/test_datasets/2026-09-01_fiber_stack2" \
  --reference-fiber-tag hendrik_crop1
```

## Threshold Sweep

All candidates use the same stored traces, inputs, Release binary, and pruning
settings. Accuracy is the oracle round-zero result before any piece removal.

| Maximum density | Traces | Exact | Wrong | Missing | Exact / estimated | Wall time |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.245 | 416 | 22 | 3 | 0 | 88.0% | 38.62 s |
| **0.250** | **451** | **22** | **3** | **0** | **88.0%** | **40.99 s** |
| 0.2525 | 468 | 20 | 5 | 0 | 80.0% | 60.43 s |
| 0.256482 | 500 | 20 | 5 | 0 | 80.0% | 58.60 s |
| 0.260 | 525 | 21 | 4 | 0 | 84.0% | 295.89 s |

`0.250` is selected because it ties the best reference accuracy and retains
more traces than `0.245`. This is reference-tuned selection on the benchmark
crop, not an unbiased validation result.

## Selected Result

| Piece metric | Result |
| --- | ---: |
| Initial pieces | 1,221 |
| Retained, all | 935 |
| Removed | 286 |
| Retained in reference range | 429 |
| Retained other | 506 |
| Problematic fraction | 40.00% |
| Removed / retained-reference | 66.67% |

| Constraint metric | Result |
| --- | ---: |
| Initial unique constraints | 57,893 |
| Removed-incident | 24,937 |
| Retained infringed | 10,758 |
| Problematic | 35,695 (61.66%) |
| Retained fulfilled | 22,198 |
| Problematic / retained-fulfilled | 160.80% |

Round zero produced 22 exact, three wrong, and zero missing estimates among the
25 evaluable references. Oracle pruning reached 24 exact, zero wrong, and one
missing after three accepted removal checkpoints. The run took 40.99 seconds
wall, 915.51 seconds user, 5.52 seconds system, and 541,476 KiB peak RSS.

## Online Diagnostic

For completeness, applying `0.25` during ordinary trace generation accepted
674 traces only after attempting 13,915 candidates and rejecting 13,233. It
took 988.04 seconds wall and produced 20 exact, five wrong, and zero missing at
oracle round zero. It is not the selected baseline point because online
rejection changes seed coverage; baseline tuning intentionally filters the
complete stored trace cohort only when BP loads it.
