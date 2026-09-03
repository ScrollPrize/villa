# Staged Fiberlet reference replay: 2026-09-03

## Provenance

| Item | Value |
| --- | --- |
| Villa revision | `3046918b556c6a1af42b50b59e7c95853954c9f3` |
| Code state | Revision above; only benchmark-planning Markdown was modified during measurement |
| Build | CMake `Release`, GCC 16.1.1 |
| Binary SHA-256 | `ca92f44cf857272de450368105fe66f0d12853985787506f4b197d8f91d7f8c0` |
| Host | `staticsheep`, Linux x86_64, AMD Ryzen 9 5950X, 16 cores / 32 threads, 62 GiB RAM |
| Cache state | Filesystem/OS cache warm from the preceding staged crop run |
| Repetitions | 1 |

The source Zarr reports `build_state=partial`; see the staged pruning record
for the local-mirror limitation and full source inventory hash.

## Inputs And Settings

| Item | Value |
| --- | --- |
| Fiberlets | `$VES/data/s1/PHercParis4.volpkg/volumes/fiberlets.zarr` |
| Fiberlet `.zattrs` SHA-256 | `d38b08d3fb06dd237adcd50da7228d963ac50a44dd6039628d5b37eb87a51e73` |
| Normals | `$VES/data/lasagna3d_inf/las008_s1_full/las_008.lasagna.json` |
| Normal manifest SHA-256 | `77834e54d9e2dfde4c10b6dba8610ba881a5ad509d1c82d6409346a931e3aa29` |
| References | `$VES/data/test_datasets/2026-09-01_fiber_stack2`, tag `hendrik_crop1` |
| Established reference inventory ID | `1a2a5c0d608f8b5b6cf9ceb361a78ff163eea640422662d669d89a33eeca3b90` |
| Crop, half-open base XYZ | `[10240,22016,6144) .. [11264,23040,7168)` |
| Search | beam 16, step 48, lookahead 384, state limit 1,000,000 |
| Threshold | 20 base voxels normal, 80 base voxels tangential |
| Reset | minimum advance 1 base voxel, seed window 32 base voxels |
| Base voxel size | 2.4 um |

## Command

```bash
/usr/bin/time -f 'wall_s=%e user_s=%U sys_s=%S max_rss_kib=%M' \
  volume-cartographer/build/bin/vc_fiber_trace_chunk \
  reference-replay-benchmark \
  "$VES/data/s1/PHercParis4.volpkg/volumes/fiberlets.zarr" \
  --normal-manifest "$VES/data/lasagna3d_inf/las008_s1_full/las_008.lasagna.json" \
  --bbox 10240 22016 6144 11264 23040 7168 \
  --reference-fiber-dir "$VES/data/test_datasets/2026-09-01_fiber_stack2" \
  --reference-fiber-tag hendrik_crop1 \
  --base-voxel-size-um 2.4 \
  --output /tmp/reference-replay-staged-3046918b5.json \
  --stage 256,0,0,0 \
  --stage 256,128,128,128 \
  --stage 512,256,256,256
```

The effective filter policy was join angle `45` degrees, stored cost profile,
and per-entry state guard `5,000,000`. The stages processed the same 512, 343,
and 27 boxes and produced the same aggregate populations as the crop run. The
stage list is preserved by this command and record; version 3 of the replay
JSON does not embed filter provenance.

## Results

| Metric | Result |
| --- | ---: |
| Total tested directed length | 101.036 mm |
| Directed cases | 48 |
| Failure events | 7 |
| Cases with failures | 6 |
| Mean distance per failure | 14.434 mm |
| Distance per failure / tested length | 14.286% |
| Whole command | 243.75 s wall, 601.77 s user, 39.97 s system |
| Maximum RSS | 8,366,164 KiB |

The output JSON SHA-256 was
`5ccbfe34655cb13f81274289031b9b56b5e6f7111395127c0c0059e63c1a2998`.
The run evaluated all 48 directed cases completely. The metric is
`tested directed length / max(failures, 1)`; zero failures therefore means
100 percent.
