# Fiber pruning benchmark

Recorded invocations are indexed in
[fiber_benchmark_results.md](fiber_benchmark_results.md).

This guide reproduces the supervised Fiber pruning benchmark on the fixed
PHercParis4 1024-base-voxel crop. It covers the complete data path from Fiber
and Lasagna inference through Fiberlet preprocessing, crop tracing, and the
final piece/constraint retention measurements.

The oracle pruning policy consumes manually ordered reference fibers. It is an
evaluation tool, not an unsupervised production algorithm.

## Frozen workload

| Item | Value |
| --- | --- |
| Sample | `PHercParis4` |
| Source volume | `20260411134726-2.400um-0.2m-78keV-masked.zarr` |
| Source volume shape ZYX | `75784,32696,32696` |
| Fiber source OME group | `1` |
| Lasagna source OME group | `2` |
| Fiber model run | `s1a_128_1_single_8x8_20260801_084232` |
| Fiber checkpoint | `snapshots/best91_5k.pt` |
| Fiber checkpoint SHA-256 | `f389da7914a6da34506f92204bf5441964e96599339dfe79dfc9c48b67165e17` |
| Fiber manager run UUID | `d74cca79-7f8e-48b4-9563-4286b1a8fc04` |
| Fiber prediction manifest SHA-256 | `9656fbb70146ca33e8866f81acb2f9d431f13d5a9b233dd44b821d6867440924` |
| Lasagna model run | `20260419_180421_conthr_1e-5_warp_2um_noss_dist` |
| Lasagna checkpoint | `snapshots/model_current.pt` |
| Lasagna checkpoint SHA-256 | `4cc0c40a846b296769b9f7982470559078662a1a0b0e1f74ff41ed9341c0d600` |
| Lasagna manager run UUID | `5b17ff6c-3c47-4f7f-8042-9326a683933b` |
| Lasagna prediction manifest SHA-256 | `e18d1074759ef6db32538963ab13b9071dc8af334af7d991bbf77442e58453ff` |
| Evaluation-normal manifest SHA-256 | `77834e54d9e2dfde4c10b6dba8610ba881a5ad509d1c82d6409346a931e3aa29` |
| Fiber prediction spacing | 8 base voxels |
| Fiberlet storage chunk side | 512 base voxels |
| Fiberlet algorithm fingerprint | `fnv1a64:edf68922eb871373` |
| Fiberlet dataset fingerprint | `d40e09c09ceb56a4b36b82b22a109ad7a612f2dbbc32e9cb15d4e6ee4fa30a2d` |
| Reference directory | `$VES/data/test_datasets/2026-09-01_fiber_stack2` |
| Reference tag | `hendrik_crop1` |
| Reference count and winding range | 26 fibers, `0.0..12.5` in filename order |
| Reference inventory SHA-256 | `1a2a5c0d608f8b5b6cf9ceb361a78ff163eea640422662d669d89a33eeca3b90` |

The recorded Fiberlet artifact uses compact directions, fixed nonlinear
`uint16` route costs, and a prediction-to-base scale of 8. Verify these values
in `fiberlets.zarr/.zattrs` before comparing a regenerated dataset with the
frozen result. The reference checksum is the SHA-256 of the concatenated,
lexicographically filename-sorted `sha256sum` output, computed from inside the
reference directory. Lexicographic filename order assigns virtual windings
`0.0, 0.5, ..., 12.5`.

## Coordinates and required support

All coordinates below are half-open base-volume **XYZ**, matching
`vc_fiber_trace_chunk --bbox`. Array shapes and chunk keys in Zarr metadata are
ZYX.

| Region | Minimum XYZ | Maximum XYZ | Side |
| --- | --- | --- | --- |
| Requested output crop | `10240,22016,6144` | `11264,23040,7168` | 1024 |
| Lookahead-expanded search | `9856,21632,5760` | `11648,23424,7552` | 1792 |
| 512-voxel storage-chunk envelope | `9728,21504,5632` | `11776,23552,7680` | 2048 |

The search range is the output crop plus the default 384-base-voxel lookahead
on every face. In Fiber prediction coordinates it is
`[1232,2704,720) -> [1456,2928,944)`, a 224-voxel cube. The final row is the
smallest 512-base-voxel-aligned Fiberlet storage region containing that search
range; it is useful when prioritizing sparse chunk transfer.

Do not add another maximum-Fiberlet-length margin. Graph materialization keeps
each complete Fiberlet incident to a search-box anchor, including the edge that
first exits the box. The source Fiberlet dataset must itself be complete for
those records and their endpoint dependencies.

## Environment and Release build

The commands below assume the standard checkout and data roots:

```bash
export SRC=/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3
export VES=/home/hendrik/business/aiconsulting/vesuviuschallenge
export VC_BUILD="$SRC/volume-cartographer/build"
export VOLUME="$VES/data/s1/PHercParis4.volpkg/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr"
export NORMALS="$VES/data/lasagna3d_inf/las008_s1_full/las_008.lasagna.json"
export FIBERLETS="$VES/data/s1/PHercParis4.volpkg/volumes/fiberlets.zarr"
export FROZEN_TRACES="$VES/data/workdir3/crop_traces.zarr"
export TRACES="$VES/data/workdir3/crop_traces_1024_benchmark.zarr"
export BENCH="$VES/data/workdir3/fiber-crop-1024/fibers"
export REFERENCES="$VES/data/test_datasets/2026-09-01_fiber_stack2"
```

Configure a Release build if the existing build directory is not already
Release, then build both commands:

```bash
cmake -S "$SRC/volume-cartographer" -B "$VC_BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

cmake --build "$VC_BUILD" \
  --target vc_fiberlets vc_fiber_trace_chunk test_fiber_trace_winding_bp \
  -j "$(nproc)"
```

Confirm the revision and build type in every benchmark log:

```bash
git -C "$SRC" rev-parse HEAD
rg '^CMAKE_BUILD_TYPE:STRING=Release$' "$VC_BUILD/CMakeCache.txt"
```

## Generate the predictions

The canonical data path uses `las_manager`. Its Fiberlet command accepts only
completed, uncropped whole-volume prediction bundles, so `--crop-xyzwhd` must
not be added to these two inference runs.

Run Fiber inference from OME group 1:

```bash
las_manager inference run \
  fiber3d/s1a_128_1_single_8x8_20260801_084232/best91_5k.pt \
  PHercParis4/20260411134726-2.400um-0.2m-78keV-masked.zarr \
  1 \
  -- \
  --devices all
```

Run Lasagna normal inference from OME group 2:

```bash
las_manager inference run \
  lasagna/20260419_180421_conthr_1e-5_warp_2um_noss_dist/model_current.pt \
  PHercParis4/20260411134726-2.400um-0.2m-78keV-masked.zarr \
  2 \
  -- \
  --devices all
```

Both commands return after reserving a managed run and starting its tmux job.
Record the printed run names, follow them until completion, and check their
portable provenance:

```bash
las_manager inference ls
las_manager tmux attach <inference-run>
```

Two normal artifacts have distinct roles. The manager Lasagna prediction above
was used when generating the frozen Fiberlets. Crop tracing, coverage, normal
alignment, and evaluation use the separately stored `las008_s1_full` manifest.
They are compatible in base frame and scale but must not be treated as the same
artifact. The exact local Fiber and evaluation-normal manifests are:

```text
$VES/data/s1/PHercParis4.volpkg/volumes/fiber_s1_002.lasagna.json
$VES/data/lasagna3d_inf/las008_s1_full/las_008.lasagna.json
```

Their recorded base shapes are respectively `75784,32694,32694` and
`75784,32693,32693` ZYX. This one-voxel extent difference is an accepted
manifest convention; neither should be substituted for the source volume's
`75784,32696,32696` ZYX shape when deriving crop coordinates.

Their historical locations are not dataset identity. A regenerated managed run
uses a generated name below its `artifacts/` directory; compare model hashes,
source volume, source level, manifest content, and output scales instead.

## Generate the Fiberlet dataset

Use the completed managed Fiber and Lasagna run names from the preceding step:

```bash
las_manager fiberlet run <fiber-inference-run> <lasagna-inference-run>

las_manager fiberlet ls
las_manager tmux attach <fiberlet-run>
```

The portable completed output is `<fiberlet-run>/artifacts/fiberlets.zarr`. Set
`FIBERLETS` to that path for a regenerated benchmark. The frozen benchmark path
defined above is a region-prioritized local mirror and reports
`build_state=partial`; this is not evidence that the manager's whole-volume
artifact was incomplete. It is sufficient only when every sparse record and
endpoint dependency required by the expanded crop is present.

Whole-volume preprocessing is sparse: absent input presence chunks remain
absent output chunks. The run still uses full-volume manifests so boundary
anchors and Fiberlets inside the required search range see the same context as
the whole-volume artifact.

## Trace the 1024 crop

Generate at most 2000 bidirectional crop traces. `--lookahead 384` is explicit
to freeze the otherwise identical default. Worker count is intentionally
omitted because the command defaults to all host CPUs.

```bash
mkdir -p "$VES/data/workdir3/fiber-crop-1024"

/usr/bin/time -f 'real=%e user=%U sys=%S' \
  "$VC_BUILD/bin/vc_fiber_trace_chunk" \
  trace \
  "$FIBERLETS" \
  --normal-manifest "$NORMALS" \
  --bbox 10240 22016 6144 11264 23040 7168 \
  --lookahead 384 \
  --max-attempts 2000 \
  --output "$TRACES" \
  --obj "$VES/data/workdir3/fiber-crop-1024/traces.obj" \
  --volume "$VOLUME/2" \
  2>&1 | tee "$VES/data/workdir3/fiber-crop-1024/trace.log"
```

`vc_fiber_trace_chunk trace` requires a new output path and does not overwrite
an existing trace dataset. To benchmark the already frozen trace cohort instead
of regenerating it, skip the trace command and use:

```bash
export TRACES="$FROZEN_TRACES"
```

The frozen trace artifact intentionally reports `build_state=partial`: tracing
was capped at 2000 attempts rather than run until every candidate anchor was
covered. It nevertheless contains the authoritative 1998 accepted traces for
this bounded benchmark. Its `.zattrs` records
the requested crop, lookahead, beam width 16, coverage radii 20 normal and 80
tangential base voxels, prediction-to-base scale 8, and source Fiberlet
fingerprint. Treat a different trace count or provenance as a different
workload rather than silently comparing it with the numbers below.

## Run the pruning benchmark

This command uses the best 25% of stored traces by cost density, splits them
into nominal 512-base-voxel pieces, solves the H/V and winding problem, fixes
the 26 reference fibers, and runs supervised oracle inlier pruning.

```bash
/usr/bin/time -f 'real=%e user=%U sys=%S' \
  "$VC_BUILD/bin/vc_fiber_trace_chunk" \
  direction-ablation \
  "$TRACES" \
  --normal-manifest "$NORMALS" \
  --output "$BENCH" \
  --direction-dominance 0.9 \
  --piece-length 512 \
  --bp-only \
  --bp-inference sum-product-mixed \
  --quality-fraction 0.25 \
  --winding-fixed-orientation \
  --reference-prune-offenders \
  --reference-prune-policy oracle-inliers \
  --reference-oracle-magnitude-weight 1 \
  --reference-fiber-dir "$REFERENCES" \
  --reference-fiber-tag hendrik_crop1 \
  2>&1 | tee "$VES/data/workdir3/fiber-crop-1024/benchmark.log"
```

All other scientific weights omitted here are deliberate current defaults. The
command prints their effective values in the run header. Preserve that header,
the Git revision, and the complete command whenever defaults are changed.

Prediction and Fiberlet generation are setup steps, not benchmark iterations.
For repeated measurements, reuse one validated Fiberlet dataset and one frozen
trace cohort, remove or overwrite only derived OBJ/log outputs as needed, and
rerun `direction-ablation`. Record CPU model, logical CPU count, RAM, OS,
compiler, Git revision, Release build type, and cache state. Report multiple
iterations with at least median and range; the frozen timing below is a single
reference run, not a multi-run mean.

## Benchmark interpretation

The pruning benchmark starts after quality filtering, piece splitting, and
main-component selection. The frozen 1024 workload has 1360 input pieces and
69,172 unique graph constraints at that point.

`fiber piece pruning benchmark` uses:

```text
problematic_%     = 100 * removed / (removed + retained_ref)
problematic/ref_% = 100 * removed / retained_ref
```

`retained_ref` contains final active H/V pieces whose calibrated winding lies
in the inclusive annotated interval `0.0..12.5`. `retained_other` includes
retained pieces outside that interval and non-active pieces. Removed pieces are
counted individually, not as whole source fibers.

`constraint pruning benchmark` counts each original graph constraint once,
even when it emits multiple orientation, magnitude, or sign factors:

```text
problematic = removed_incident + retained_infringed + retained_defect

problematic_% =
    100 * problematic / (problematic + retained_fulfilled)

problematic/retained_% =
    100 * problematic / retained_fulfilled
```

`removed_incident` means at least one endpoint piece was removed.
`retained_defect` means a retained endpoint finished as Defect.
`retained_infringed` means at least one evaluated factor term was infringed by
the authoritative conditioned solution. These classes form a union, so a
constraint is never counted twice.

The reference result is also reported as exact, wrong, or missing. This is a
separate accuracy diagnostic and does not replace the population benchmark.

## Frozen result

The recorded Release run produced:

| Piece metric | Count/value |
| --- | ---: |
| Initial pieces | 1360 |
| Retained pieces, all | 997 |
| Removed pieces | 363 |
| Retained in reference range | 454 |
| Retained other | 543 |
| Problematic fraction | 44.43% |
| Removed / retained-reference | 80.00% |

| Constraint metric | Count/value |
| --- | ---: |
| Initial unique constraints | 69,172 |
| Removed-incident | 33,299 |
| Retained infringed | 10,985 |
| Retained Defect | 0 |
| Problematic | 44,284 |
| Retained fulfilled | 24,888 |
| Problematic fraction | 64.02% |
| Problematic / retained-fulfilled | 177.90% |

The supervised direct oracle result found 24 exact, zero wrong, and two
missing references over the complete 26-reference stack. A single reference
run took 56.83 seconds wall, 1224.87 seconds user, and 2.15 seconds system on the
existing 32-worker host. Timing comparisons require the same host, Release
build, input artifacts, trace cohort, and warm/cold cache state.

## Validation

Run the focused regression suite after changing benchmark accounting:

```bash
"$VC_BUILD/bin/test_fiber_trace_winding_bp"
```

The current suite has 97 test cases. Before accepting a benchmark run, also
check:

1. The trace artifact reports 1998 input traces before quality filtering.
2. The evaluation retains 500 traces at `--quality-fraction 0.25`.
3. Main-component processing reports 1360 pieces and 69,172 constraints.
4. `final_active_uncalibrated=0`.
5. The piece and constraint benchmark denominators reconstruct their initial
   populations exactly.
6. No run reached an unexpected BP resource guard or message-limit state.

Verify the reference inventory without embedding its absolute directory in the
digest:

```bash
cd "$REFERENCES"
find . -maxdepth 1 -type f -name '*.json' -printf '%f\0' \
  | sort -z \
  | xargs -0 sha256sum \
  | sha256sum
```
