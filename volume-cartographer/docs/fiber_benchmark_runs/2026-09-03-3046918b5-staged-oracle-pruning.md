# Staged uncapped oracle fiber pruning: 2026-09-03

## Provenance

| Item | Value |
| --- | --- |
| Villa revision | `3046918b556c6a1af42b50b59e7c95853954c9f3` |
| Code state | Revision above; only benchmark-planning Markdown was modified during measurement |
| Build | CMake `Release`, GCC 16.1.1 |
| Binary SHA-256 | `ca92f44cf857272de450368105fe66f0d12853985787506f4b197d8f91d7f8c0` |
| Host | `staticsheep`, Linux x86_64, AMD Ryzen 9 5950X, 16 cores / 32 threads, 62 GiB RAM |
| Cache state | Filesystem/OS cache not flushed |
| Pruning repetitions | 2; identical scientific counts |

The source Fiberlet Zarr reports `build_state=partial`. Its supplied encoded
extent covers the complete planned stage geometry, and both commands resolved
all requested chunks, but no authoritative remote chunk inventory was
available. This run therefore uses an unverified partial local mirror; sparse
absence is interpreted as empty according to the dataset contract.

## Inputs

| Item | Value |
| --- | --- |
| Fiberlets | `$VES/data/s1/PHercParis4.volpkg/volumes/fiberlets.zarr` |
| Fiberlet `.zattrs` SHA-256 | `d38b08d3fb06dd237adcd50da7228d963ac50a44dd6039628d5b37eb87a51e73` |
| Fiberlet sorted file/content inventory SHA-256 | `82734233d4cdffab00bc0993a2eae9958bb01a28f1f0a1bd6c609a2768b343bf` |
| Generated traces | `$VES/data/workdir3/fiber-crop-1024-staged-full/crop_traces.zarr` |
| Trace `.zattrs` SHA-256 | `2e0b5fbc2bfcf806a32cdec161a703d90401ff1514c7bfcb1760cf78acf75734` |
| Trace sorted file/content inventory SHA-256 | `076c2471135ea5b122fc6f15809f13be03d2860cf06cd639316c15c348622ad1` |
| Normals | `$VES/data/lasagna3d_inf/las008_s1_full/las_008.lasagna.json` |
| Normal manifest SHA-256 | `77834e54d9e2dfde4c10b6dba8610ba881a5ad509d1c82d6409346a931e3aa29` |
| References | `$VES/data/test_datasets/2026-09-01_fiber_stack2`, tag `hendrik_crop1` |
| Established reference inventory ID | `1a2a5c0d608f8b5b6cf9ceb361a78ff163eea640422662d669d89a33eeca3b90` |
| Pruning stdout log SHA-256 | `9ec052c3c0d0e1c23983897ec9dcbdbe06faa65952ecaf5cd5d0b747b88b23f1` |

## Trace Generation

```bash
volume-cartographer/build/bin/vc_fiber_trace_chunk trace \
  "$VES/data/s1/PHercParis4.volpkg/volumes/fiberlets.zarr" \
  --normal-manifest "$VES/data/lasagna3d_inf/las008_s1_full/las_008.lasagna.json" \
  --bbox 10240 22016 6144 11264 23040 7168 \
  --output "$VES/data/workdir3/fiber-crop-1024-staged-full/crop_traces.zarr" \
  --obj "$VES/data/workdir3/fiber-crop-1024-staged-full/traces.obj" \
  --volume "$VES/data/s1/PHercParis4.volpkg/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/2" \
  --stage 256,0,0,0 \
  --stage 256,128,128,128 \
  --stage 512,256,256,256
```

No fiber or attempt limit was supplied. The trace metadata records
`maximum_fibers=0`, `maximum_attempts=0`, lookahead `384`, beam width `16`,
state guard `1,000,000`, coverage angle `25` degrees, and coverage normal
radius `20` base voxels. Filter defaults were join angle `45` degrees, stored
cost profile, and per-entry state guard `5,000,000`.

The stages processed 512, 343, and 27 overlapping boxes. Their aggregate
Fiberlet counts were `5,903,905 -> 4,011,983`,
`2,082,011 -> 1,292,725`, and `632,703 -> 448,512`; these are per-box totals,
not unique dataset populations. The uncapped trace resolved all 26,640 seed
candidates: 2,062 attempted and accepted, 24,578 covered, zero remaining.
Filtering took about 2m45s, 1m17s, and 11s; graph materialization took 4.59s
and tracing took 1.04s. This one artifact-generation run is not a performance
estimate.

## Pruning Command

```bash
/usr/bin/time -f 'wall_s=%e user_s=%U sys_s=%S max_rss_kib=%M' \
  volume-cartographer/build/bin/vc_fiber_trace_chunk \
  direction-ablation \
  "$VES/data/workdir3/fiber-crop-1024-staged-full/crop_traces.zarr" \
  --normal-manifest "$VES/data/lasagna3d_inf/las008_s1_full/las_008.lasagna.json" \
  --output "$VES/data/workdir3/fiber-crop-1024-staged-full/fibers" \
  --direction-dominance 0.9 \
  --piece-length 512 \
  --bp-only \
  --bp-inference sum-product-mixed \
  --quality-fraction 0.25 \
  --winding-fixed-orientation \
  --reference-prune-offenders \
  --reference-prune-policy oracle-inliers \
  --reference-oracle-magnitude-weight 1 \
  --reference-fiber-dir "$VES/data/test_datasets/2026-09-01_fiber_stack2" \
  --reference-fiber-tag hendrik_crop1
```

The effective winding settings were phase `0.5`, measurement scale `0.822`,
Defect cost `100`, hard split continuity, magnitude weights
`0,0,0.5,4,1`, sign weights `1,0.5`, finite sign cost `44`, and hard signs
within 30 degrees. Oracle pruning used sign weight `1`, magnitude weight `1`,
retention `1`, at most 20 rounds, 32 pair candidates, and 3 minimum
observations.

## Results

| Piece metric | Result |
| --- | ---: |
| Initial pieces | 1,450 |
| Retained, all | 1,142 |
| Removed | 308 |
| Retained in reference range | 481 |
| Retained other | 661 |
| Problematic fraction | 39.04% |
| Removed / retained-reference | 64.03% |

| Constraint metric | Result |
| --- | ---: |
| Initial unique constraints | 78,925 |
| Removed-incident | 28,598 |
| Retained infringed | 17,398 |
| Retained Defect | 0 |
| Problematic | 45,996 (58.28%) |
| Retained fulfilled | 32,929 |
| Problematic / retained-fulfilled | 139.68% |

The final pruned solve converged with 1,134 active and 8 disabled pieces,
one active component, and zero retained sign conflicts. Its reference result
was 24 exact, zero wrong, and two missing windings. The two measured wall
times were 79.33 and 83.34 seconds: min 79.33, median/mean 81.34, max 83.34.
The logged repeat used 1,681.14 seconds user, 6.77 seconds system, and 542,608
KiB maximum RSS.
