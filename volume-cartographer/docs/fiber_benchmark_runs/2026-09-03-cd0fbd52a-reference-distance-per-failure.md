# Reference replay mean segment length: 2026-09-03

## Provenance

| Item | Value |
| --- | --- |
| Villa revision | `cd0fbd52a1fdc48952a8d44227f98f7e240d5bdb` |
| Tracked source state | Clean |
| Build | CMake `Release`, GCC 16.1.1 |
| Binary SHA-256 | `b63f0959bd85f62e0f46795df07fe64fcbfb57e99dbeb5ac402c730bbd6e6960` |
| Host | `staticsheep`, Linux x86_64, AMD Ryzen 9 5950X, 16 cores / 32 threads, 62 GiB RAM |
| Repetitions | 1 |

## Inputs And Settings

| Item | Value |
| --- | --- |
| Fiberlets | `$VES/data/s1/PHercParis4.volpkg/volumes/fiberlets.zarr` |
| Normals | `$VES/data/lasagna3d_inf/las008_s1_full/las_008.lasagna.json` |
| References | `$VES/data/test_datasets/2026-09-01_fiber_stack2`, tag `hendrik_crop1` |
| Crop, half-open base XYZ | `[10240,22016,6144) .. [11264,23040,7168)` |
| Search | beam 16, step 48, lookahead 384, state limit 1,000,000 |
| Threshold | 20 base voxels normal, 80 base voxels tangential |
| Reset | minimum advance 1 base voxel, seed window 32 base voxels |
| Base voxel size | 2.4 um |

| Artifact | SHA-256 |
| --- | --- |
| Fiberlet `.zattrs` | `d38b08d3fb06dd237adcd50da7228d963ac50a44dd6039628d5b37eb87a51e73` |
| Normal manifest | `77834e54d9e2dfde4c10b6dba8610ba881a5ad509d1c82d6409346a931e3aa29` |
| Reference inventory | `1a2a5c0d608f8b5b6cf9ceb361a78ff163eea640422662d669d89a33eeca3b90` |

## Command

```bash
/usr/bin/time -f '\nreal %e\nuser %U\nsys %S\nmax_rss_kib %M' \
  volume-cartographer/build/bin/vc_fiber_trace_chunk \
  reference-replay-benchmark \
  "$VES/data/s1/PHercParis4.volpkg/volumes/fiberlets.zarr" \
  --normal-manifest "$VES/data/lasagna3d_inf/las008_s1_full/las_008.lasagna.json" \
  --bbox 10240 22016 6144 11264 23040 7168 \
  --reference-fiber-dir "$VES/data/test_datasets/2026-09-01_fiber_stack2" \
  --reference-fiber-tag hendrik_crop1 \
  --base-voxel-size-um 2.4 \
  --output /tmp/reference-replay-cd0fbd52a.json
```

## Results

| Metric | Result |
| --- | ---: |
| Total tested directed length | 101.036 mm |
| Failure events | 6 |
| Mean segment length | 14.434 mm |
| Mean segment length / tested length | 14.286% |
| Whole command | 21.41 s wall, 117.98 s user, 14.28 s system |
| Maximum RSS | 13,276,964 KiB |

The corrected formula is `tested directed length / (failures + 1)`. These two
rows are recomputed from the archived total length and failure count. The
generated version-2 JSON used the deprecated distance-per-failure fields and
had SHA-256
`a08cc553dd80ef0000bba2d1b490576252566aa64f5505f386763694c05c48f5`.
