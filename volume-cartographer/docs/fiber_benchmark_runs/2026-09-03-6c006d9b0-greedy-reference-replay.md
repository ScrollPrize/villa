# Greedy direct reference replay: 2026-09-03

## Provenance

| Item | Value |
| --- | --- |
| Villa revision | `6c006d9b0a1477cfcb618ab1b02e3f2855349f9c` |
| Tracked source state | Clean |
| Build | CMake `Release`, GCC 16.1.1 |
| Binary SHA-256 | `666c5eed8b617c59fab76f92fba36f6114b04c769f2cfbf94f6833e6c9d25d83` |
| Host | `staticsheep`, Linux x86_64, AMD Ryzen 9 5950X, 16 cores / 32 threads, 62 GiB RAM |
| Repetitions | 1 |

## Inputs And Settings

| Item | Value |
| --- | --- |
| Tracer | Historical dense-prediction greedy tracer; beam width 1, lookahead 1 |
| Fiber prediction | `$VES/data/s1/PHercParis4.volpkg/volumes/fiber_s1_002.lasagna.json` |
| Normals | `$VES/data/lasagna3d_inf/las008_s1_full/las_008.lasagna.json` |
| References | `$VES/data/test_datasets/2026-09-01_fiber_stack2`, tag `hendrik_crop1` |
| Crop, half-open base XYZ | `[10240,22016,6144) .. [11264,23040,7168)` |
| Cases | 26 selected sources, 24 in-crop runs, 48 directed cases |
| Trace geometry | prediction-to-base 8, trace-to-base 2, step 4 trace voxels / 8 base voxels |
| Direction search | 25 degree cone, 5 degree angular step, 25-point cone grid |
| Smoothness | total 2.0, normal 0.1, tangent 10.0 |
| Threshold | 20 base voxels normal, 80 base voxels tangential |
| Reset | exact reference endpoint/prediction initialization; 8 base voxels after failure |
| Base voxel size | 2.4 um |

| Artifact | SHA-256 |
| --- | --- |
| Fiber prediction manifest | `3fd8a291e201309fdd67cc919c3ab383c752357120658925222e46f0cba770a9` |
| Normal manifest | `77834e54d9e2dfde4c10b6dba8610ba881a5ad509d1c82d6409346a931e3aa29` |
| Reference inventory | `1a2a5c0d608f8b5b6cf9ceb361a78ff163eea640422662d669d89a33eeca3b90` |

The dense Fiber prediction is an independent historical input. Its manifest
hash differs from the source manifest recorded by the current Fiberlet dataset,
so this run does not claim identical learned predictions between policies.

## Command

```bash
/usr/bin/time -f 'wall_s=%e user_s=%U sys_s=%S max_rss_kib=%M' \
  volume-cartographer/build/bin/vc_fiber_trace_chunk \
  reference-replay-benchmark - \
  --replay-tracer greedy \
  --fiber-manifest "$VES/data/s1/PHercParis4.volpkg/volumes/fiber_s1_002.lasagna.json" \
  --normal-manifest "$VES/data/lasagna3d_inf/las008_s1_full/las_008.lasagna.json" \
  --bbox 10240 22016 6144 11264 23040 7168 \
  --reference-fiber-dir "$VES/data/test_datasets/2026-09-01_fiber_stack2" \
  --reference-fiber-tag hendrik_crop1 \
  --base-voxel-size-um 2.4 \
  --output /tmp/reference-replay-greedy-6c006d9b0.json
```

## Results

| Metric | Result |
| --- | ---: |
| Total tested directed length | 101.036 mm |
| Failure events | 13 |
| Cases with failures | 11 / 48 |
| Mean segment length | 7.217 mm |
| Mean segment length / tested length | 7.143% |
| Whole command | 0.49 s wall, 3.51 s user, 6.42 s system |
| Maximum RSS | 274,592 KiB |

All 13 events were anisotropic distance failures. The corrected summary rows
above use `tested directed length / (failures + 1)`; the archived version-3
JSON uses the deprecated distance-per-failure fields and had SHA-256
`9d31363a262eb01a7e4fc093e2a1bcf100a15ac3dedcea7f25da760e48a25a92`.
