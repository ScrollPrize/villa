# No-overtrace quality-threshold pruning: 2026-09-03

## Provenance

| Item | Value |
| --- | --- |
| Villa base revision | `a5e6f5d498a67ff7a2a400314f2bfd96260df5e8` plus the uncommitted no-overtrace/quality-threshold change recorded with this document |
| Build | CMake `Release` |
| Binary SHA-256 | `6d321398b2b4a6899a14fb96446d416be4c700a6a942ba86c7a0117c904a7b9a` |
| Repetitions | 1 |

## Inputs

| Item | Value |
| --- | --- |
| Fiberlets | `$VES/data/s1/PHercParis4.volpkg/volumes/fiberlets.zarr` |
| Crop, base XYZ | `10240,22016,6144` to `11264,23040,7168` |
| Generated traces | `/tmp/fiber-crop-1024-no-overtrace-q035.zarr` |
| Trace `.zattrs` SHA-256 | `de3fa1a0e6cb1acfcf751ec9c244ccffb197b802c68f0f5b6145739b7f1078c8` |
| Normals | `$VES/data/lasagna3d_inf/las008_s1_full/las_008.lasagna.json` |
| Normal manifest SHA-256 | `77834e54d9e2dfde4c10b6dba8610ba881a5ad509d1c82d6409346a931e3aa29` |
| References | `$VES/data/test_datasets/2026-09-01_fiber_stack2`, tag `hendrik_crop1` |

## Commands

```bash
/usr/bin/time -f 'wall_s=%e user_s=%U sys_s=%S max_rss_kib=%M' \
  volume-cartographer/build/bin/vc_fiber_trace_chunk trace \
  "$VES/data/s1/PHercParis4.volpkg/volumes/fiberlets.zarr" \
  --normal-manifest "$VES/data/lasagna3d_inf/las008_s1_full/las_008.lasagna.json" \
  --bbox 10240 22016 6144 11264 23040 7168 \
  --lookahead 384 \
  --stop-at-covered \
  --quality-threshold 0.35 \
  --output /tmp/fiber-crop-1024-no-overtrace-q035.zarr \
  --obj /tmp/fiber-crop-1024-no-overtrace-q035.obj
```

```bash
/usr/bin/time -f 'wall_s=%e user_s=%U sys_s=%S max_rss_kib=%M' \
  volume-cartographer/build/bin/vc_fiber_trace_chunk direction-ablation \
  /tmp/fiber-crop-1024-no-overtrace-q035.zarr \
  --normal-manifest "$VES/data/lasagna3d_inf/las008_s1_full/las_008.lasagna.json" \
  --output /tmp/no-overtrace-online-q035-fibers \
  --direction-dominance 0.9 \
  --piece-length 512 \
  --bp-only \
  --bp-inference sum-product-mixed \
  --winding-fixed-orientation \
  --reference-prune-offenders \
  --reference-prune-policy oracle-inliers \
  --reference-oracle-magnitude-weight 1 \
  --reference-oracle-accept-message-limit \
  --reference-fiber-dir "$VES/data/test_datasets/2026-09-01_fiber_stack2" \
  --reference-fiber-tag hendrik_crop1
```

## Results

The online threshold accepted 544 traces and 241,680 base voxels of traced arc.
It attempted every remaining eligible seed because a rejected trace creates no
coverage: the trace run took 683.01 seconds wall, 14,931.97 seconds user, and
12.59 GiB peak RSS.

Before any oracle-pruning removal, reference round zero produced 21 exact,
four wrong, and zero missing estimates among the 25 evaluable references. One
of the complete 26-reference stack lacked sufficient evidence. A controlled
rerun of the old 25%-fraction cohort with the same binary produced 20 exact,
five wrong, and zero missing among the same 25 evaluable references.

| Piece metric | Result |
| --- | ---: |
| Initial pieces | 807 |
| Retained, all | 666 |
| Removed | 141 |
| Retained in reference range | 329 |
| Retained other | 337 |
| Retained piece arc | 230,116.8 base voxels |
| Problematic fraction | 30.00% |
| Removed / retained-reference | 42.86% |

| Constraint metric | Result |
| --- | ---: |
| Initial unique constraints | 13,747 |
| Removed-incident | 4,019 |
| Retained infringed | 2,803 |
| Problematic | 6,822 (49.63%) |
| Retained fulfilled | 6,925 |
| Problematic / retained-fulfilled | 98.51% |

The oracle reached 24 exact, zero wrong, and one missing reference after seven
accepted removal checkpoints. The pruning run took 27.11 seconds wall, 747.43
seconds user, and 5.78 seconds system.

## Interpretation

This threshold was selected using the reference crop, so the result is a tuned
operating point rather than an unbiased benchmark. It improves the controlled
pre-pruning reference result from 80% to 84% exact and the final result from
24/0/2 to 24/0/1 relative to the historical fixed-quarter benchmark. It also
reduces the problematic-to-retained-fulfilled constraint ratio from 177.93% to
98.51%, while retaining a smaller, higher-quality candidate graph.
