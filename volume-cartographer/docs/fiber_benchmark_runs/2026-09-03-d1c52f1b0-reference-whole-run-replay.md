# Whole-run reference endpoint replay: 2026-09-03

## Provenance

| Item | Value |
| --- | --- |
| Villa revision | `d1c52f1b0eba18c705165cbb1dc0e9906a126efa` |
| Tracked source state | Clean |
| Build | CMake `Release`, GCC 16.1.1 |
| Binary SHA-256 | `f570e17b7a3e791ab31ae88db739d26cfcca16e774bbcfa6408936abc4109f5b` |
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
| Endpoint seed | Required in first 32-base-voxel seed window |
| Failure policy | Record failure, advance at least 1 base voxel, scan successive 32-base-voxel windows, and continue to reference end |

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
  --output /tmp/reference-replay-d1c52f1b0.json
```

## Results

| Metric | Result |
| --- | ---: |
| Directed cases evaluated | 48 / 48 |
| Failure-free directions | 43 / 48 (89.583%) |
| Directions with failures | 5 |
| Failure events | 6, all `distance_above_threshold` |
| Full directed reference length | 101.036 mm |
| Credited seeded length | 101.036 mm |
| Mean credited length, all directions | 2.105 mm |
| Seeded replay spans | 54 |
| Mean seeded span | 1.871 mm |
| Mean span ending in failure | 0.441 mm |
| Failures per directed millimeter | 0.0594 |
| Length-weighted seeded coverage | 100.000% |
| Graph preparation | 20.289 s wall, 129.268 s CPU |
| Replay cases | 0.215 s wall |
| Whole command | 21.58 s wall, 119.44 s user, 15.22 s system |
| Maximum RSS | 13,229,500 KiB |

The seeded-coverage percentage is 100% because every failure was immediately
recoverable without a missing-seed gap. Failure count, density, failure-free
direction rate, and failed-span length carry the error information in this run.

## Failure Locations

Directional arcs start at the endpoint used by that case. Source arcs use the
forward orientation of the clipped reference run.

| Source | Direction | Event | Directional arc | Source arc | Span | Normal error | Tangential error |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | forward | 0 | 752.387 vx | 752.387 vx | 752.387 vx | 4.257 vx | 81.504 vx |
| 0 | reverse | 0 | 73.334 vx | 1019.090 vx | 73.334 vx | 23.321 vx | 19.982 vx |
| 0 | reverse | 1 | 205.559 vx | 886.865 vx | 132.225 vx | 2.847 vx | 80.919 vx |
| 10 | reverse | 0 | 33.395 vx | 973.839 vx | 33.395 vx | 25.485 vx | 22.165 vx |
| 14 | reverse | 0 | 56.039 vx | 1035.026 vx | 56.039 vx | 21.167 vx | 29.275 vx |
| 15 | reverse | 0 | 54.177 vx | 711.077 vx | 54.177 vx | 6.542 vx | 86.025 vx |

Source 0 reverse contains two distinct failures. The superseded first-failure
benchmark could report only the first one. Source indices 0, 10, 14, and 15
refer respectively to `waldkauz_20260828T064452831_000002.json`,
`waldkauz_20260901T105602115_000012.json`,
`waldkauz_20260901T160812289_000016.json`, and
`waldkauz_20260901T161032692_000017.json`.

The generated version-2 JSON had SHA-256
`a00376caf599940f7aa43985b166f1f4254e6b8259f3389c628738939c65778b`.
