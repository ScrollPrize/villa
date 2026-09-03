# Oracle fiber pruning: 2026-09-03

## Provenance

| Item | Value |
| --- | --- |
| Villa revision | `1a70f9e57e47754df8379bf453ab73fddf757088` |
| Tracked source state | Clean |
| Build | CMake `Release`, GCC 16.1.1 |
| Binary SHA-256 | `dbe657478e1ceea1b9a14800bf6d840ff3027fd50daa88dbd218916c6c551ce2` |
| Host | `staticsheep`, Linux x86_64, AMD Ryzen 9 5950X, 16 cores / 32 threads, 62 GiB RAM |
| Cache state | Filesystem/OS cache not flushed; normal cache warmed by the preceding endpoint run |
| Repetitions | 1 |

## Inputs

| Item | Value |
| --- | --- |
| Frozen traces | `$VES/data/workdir3/crop_traces.zarr` |
| Trace `.zattrs` SHA-256 | `78fb6aff07c5490051382e250b4cc8458cb90e0b1ed86e6e4be6e1f3c478c5b3` |
| Normals | `$VES/data/lasagna3d_inf/las008_s1_full/las_008.lasagna.json` |
| Normal manifest SHA-256 | `77834e54d9e2dfde4c10b6dba8610ba881a5ad509d1c82d6409346a931e3aa29` |
| References | `$VES/data/test_datasets/2026-09-01_fiber_stack2`, tag `hendrik_crop1` |
| Reference inventory SHA-256 | `1a2a5c0d608f8b5b6cf9ceb361a78ff163eea640422662d669d89a33eeca3b90` |

## Settings

The run retains the best 25% of 1,998 traces, producing 500 fibers and 1,361
pieces before the main-component filter (1,360 afterward), with 69,172 unique
constraints. Effective settings were: direction dominance `0.9`, nominal piece
length `512`, fixed-orientation mixed-state sum-product BP, temperature `1.25`,
mixed cost `1`, winding phase `0.5`, measurement scale `0.822`, winding Defect
cost `100`, hard split continuity, magnitude weights `0,0,0.5,4,1`, sign
weights `1,0.5`, finite sign cost `44`, and hard signs within 30 degrees of the
normal. Oracle pruning used sign weight `1`, magnitude weight `1`, retention
`1`, at most 20 rounds, 32 pair candidates, and 3 minimum observations.

## Command

The measured approved runner invocation was:

```bash
/tmp/vc_direction_ablation_runner.sh reference-prune oracle 1
```

Its expanded benchmark command is:

```bash
/usr/bin/time -f 'real=%e user=%U sys=%S' \
  volume-cartographer/build/bin/vc_fiber_trace_chunk \
  direction-ablation \
  "$VES/data/workdir3/crop_traces.zarr" \
  --normal-manifest "$VES/data/lasagna3d_inf/las008_s1_full/las_008.lasagna.json" \
  --output "$VES/data/workdir3/fiber-crop-1024/fibers" \
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

## Results

| Piece metric | Result |
| --- | ---: |
| Initial pieces | 1,360 |
| Retained, all | 997 |
| Removed | 363 |
| Retained in reference range | 454 |
| Retained other | 543 |
| Problematic fraction | 44.43% |
| Removed / retained-reference | 79.96% |

| Constraint metric | Result |
| --- | ---: |
| Initial unique constraints | 69,172 |
| Removed-incident | 33,299 |
| Retained infringed | 10,985 |
| Retained Defect | 0 |
| Problematic | 44,284 (64.02%) |
| Retained fulfilled | 24,888 |
| Problematic / retained-fulfilled | 177.93% |

The oracle converged after seven accepted checkpoints and produced 24 exact,
zero wrong, and two missing reference windings. Timing was 57.62 seconds wall,
1,251.99 seconds user, and 3.96 seconds system.

