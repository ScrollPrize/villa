# Reference endpoint replay: 2026-09-03

## Provenance

| Item | Value |
| --- | --- |
| Villa revision | `1a70f9e57e47754df8379bf453ab73fddf757088` |
| Tracked source state | Clean |
| Build | CMake `Release`, GCC 16.1.1 |
| Binary SHA-256 | `dbe657478e1ceea1b9a14800bf6d840ff3027fd50daa88dbd218916c6c551ce2` |
| Host | `staticsheep`, Linux x86_64, AMD Ryzen 9 5950X, 16 cores / 32 threads, 62 GiB RAM |
| Cache state | Filesystem/OS cache not flushed; decoded process caches start empty |
| Repetitions | 1 |

## Inputs

| Item | Value |
| --- | --- |
| Fiberlets | `$VES/data/s1/PHercParis4.volpkg/volumes/fiberlets.zarr` |
| Fiberlet `.zattrs` SHA-256 | `d38b08d3fb06dd237adcd50da7228d963ac50a44dd6039628d5b37eb87a51e73` |
| Normals | `$VES/data/lasagna3d_inf/las008_s1_full/las_008.lasagna.json` |
| Normal manifest SHA-256 | `77834e54d9e2dfde4c10b6dba8610ba881a5ad509d1c82d6409346a931e3aa29` |
| References | `$VES/data/test_datasets/2026-09-01_fiber_stack2`, tag `hendrik_crop1` |
| Reference inventory SHA-256 | `1a2a5c0d608f8b5b6cf9ceb361a78ff163eea640422662d669d89a33eeca3b90` |
| Crop, half-open base XYZ | `[10240,22016,6144) .. [11264,23040,7168)` |

The 384-base-voxel lookahead materializes support from
`[9856,21632,5760)` through `[11648,23424,7552)`. The 26 selected source
fibers yield 24 nonempty in-crop runs and 48 directed cases; two sources have
no in-crop run.

## Settings

| Setting | Value |
| --- | ---: |
| Beam width | 16 |
| Beam step | 48 base voxels |
| Lookahead | 384 base voxels |
| Generated-state limit | 1,000,000 per iteration |
| Normal failure radius | 20 base voxels |
| Tangential failure radius | 80 base voxels |
| Match refinement | 1 step |
| Base voxel size | 2.4 um |
| Workers | 32 directed cases; one expansion thread per case |
| Endpoint seed | Required in first ordinary seed window |
| Failure policy | Stop on first failure of any kind |

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
  --output /tmp/reference-replay-1a70f9e57.json
```

## Results

| Metric | Result |
| --- | ---: |
| Directed cases | 48 |
| Completed | 43 (89.583%) |
| Failed | 5, all `distance_above_threshold` |
| Full directed reference length | 101.036 mm |
| Credited traced length | 91.246 mm |
| Mean credited length, all cases | 1.901 mm |
| Mean length to failure, failed cases | 0.465 mm |
| Length-weighted success | 90.311% |
| Graph preparation | 20.145 s wall, 128.518 s CPU |
| Replay cases | 0.155 s wall |
| Whole command | 21.37 s wall, 119.35 s user, 13.98 s system |
| Maximum RSS | 12,988,800 KiB |

The generated version-1 JSON had SHA-256
`fc2f9c91c436364d457db9152c95cabc9bc3d04653eafb35ce975edacb3a7d2f`.

