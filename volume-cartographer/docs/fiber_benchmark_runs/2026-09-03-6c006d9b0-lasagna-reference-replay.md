# Lasagna normal-transport reference replay: 2026-09-03

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
| Tracer | Reference-tangent-initialized Lasagna normal transport control |
| Normals | `$VES/data/lasagna3d_inf/las008_s1_full/las_008.lasagna.json` |
| References | `$VES/data/test_datasets/2026-09-01_fiber_stack2`, tag `hendrik_crop1` |
| Crop, half-open base XYZ | `[10240,22016,6144) .. [11264,23040,7168)` |
| Cases | 26 selected sources, 24 in-crop runs, 48 directed cases |
| Transport | 16 base voxels per step; invalid normals retain the previous direction |
| Threshold | 20 base voxels normal, 80 base voxels tangential |
| Reset | exact reference endpoint/tangent initialization; 16 base voxels after failure |
| Global optimization | Disabled |
| Base voxel size | 2.4 um |

| Artifact | SHA-256 |
| --- | --- |
| Normal manifest | `77834e54d9e2dfde4c10b6dba8610ba881a5ad509d1c82d6409346a931e3aa29` |
| Reference inventory | `1a2a5c0d608f8b5b6cf9ceb361a78ff163eea640422662d669d89a33eeca3b90` |

Normals do not determine an independent fiber direction. This backend is a
control initialized with the reference endpoint tangent, not a learned direct
fiber tracer, and it does not run Lasagna's global line optimization.

## Command

```bash
/usr/bin/time -f 'wall_s=%e user_s=%U sys_s=%S max_rss_kib=%M' \
  volume-cartographer/build/bin/vc_fiber_trace_chunk \
  reference-replay-benchmark - \
  --replay-tracer lasagna \
  --normal-manifest "$VES/data/lasagna3d_inf/las008_s1_full/las_008.lasagna.json" \
  --bbox 10240 22016 6144 11264 23040 7168 \
  --reference-fiber-dir "$VES/data/test_datasets/2026-09-01_fiber_stack2" \
  --reference-fiber-tag hendrik_crop1 \
  --base-voxel-size-um 2.4 \
  --output /tmp/reference-replay-lasagna-6c006d9b0.json
```

## Results

| Metric | Result |
| --- | ---: |
| Total tested directed length | 101.036 mm |
| Failure events | 57 |
| Cases with failures | 34 / 48 |
| Mean segment length | 1.742 mm |
| Mean segment length / tested length | 1.724% |
| Whole command | 0.09 s wall, 0.09 s user, 1.27 s system |
| Maximum RSS | 70,776 KiB |

All 57 events were anisotropic distance failures. The corrected summary rows
above use `tested directed length / (failures + 1)`; the archived version-3
JSON uses the deprecated distance-per-failure fields and had SHA-256
`1183903ffe2eba7946ebc101e16847dc49165dc26f8d3afcbc7dde77071dd951`.
