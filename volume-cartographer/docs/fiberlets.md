# Fiberlet anchor extraction

`vc_fiberlets anchors` is the first stage of anchor-seeded fiberlet
over-segmentation. It turns the canonical `presence/nx/ny` channels of a Fiber
Lasagna manifest into zero, one, or two local unoriented line anchors per
coarse prediction-grid cell. Connection search, path optimization, path
filtering, deduplication, and extension are not implemented by this stage.

## Input and coordinates

The command accepts a local Lasagna manifest, a local manifest with the
existing `lasagna-remote.json` sidecar, or a direct HTTP/S3 manifest. Direct
remote manifests require `--remote-cache-dir`; manifest and Zarr reads use the
shared cache-first Lasagna implementation. `--cache-gib` controls the decoded
chunk cache separately.

Only the exact canonical `presence`, `nx`, and `ny` channels are observations.
Extra manifest groups, including prefixed prediction triplets used by older
tracer configurations, do not enter the fit. The channels must be 3D `uint8`
arrays with equal ZYX shapes and effective spacing, but their chunk shapes may
differ. The manifest must provide a positive numeric `source_to_base`.

Integer stored-prediction indices are voxel centres. CLI spatial coordinates
and all JSON/OBJ spatial positions are always expressed in base voxels. A cell
has a globally fixed origin and owns the half-open ZYX range
`[cell * size, min((cell + 1) * size, shape))`. A crop selects every global
cell it intersects and samples that cell in full. It does not move or truncate
interior cells. Prediction-grid coordinates remain private solver values.

## Fit

For a voxel direction `d_i`, presence `p_i`, fixed cell-centred Gaussian
`g_i`, and two unoriented axes `u_0,u_1`, the solver maximizes

```text
sum_i g_i p_i max_k((d_i dot u_k)^2) / sum_cell_voxels g_i.
```

Assignment uses the larger squared dot with a deterministic component-zero
tie. Each component update is the principal eigenvector of its assigned
weighted direction dyads. Deterministic multistart assignment/PCA permits the
two lines to be non-orthogonal. It is not the first two, necessarily
orthogonal eigenvectors of one covariance matrix.

Component support and position use the same `g_i p_i dot^2` mass. The support
denominator includes all owned cell voxels, including invalid and zero-presence
samples. Empty, degenerate, and below-threshold components are discarded
independently; the remaining component is not refitted after rejection. The
output is therefore zero, one, or two anchors.

## Command

```bash
volume-cartographer/build/bin/vc_fiberlets anchors \
  /path/to/fiber.lasagna.json /tmp/fiber-anchors \
  --crop 13568,20224,18112,256,256,256
```

Cell size is restricted to 2 through 8 stored prediction voxels. If omitted,
Gaussian sigma is half the selected cell size. The presence floor and aligned
support threshold are inclusive. `--base-voxel-size-um` adds optional physical
reporting metadata but never changes the solve.

`--crop` is the half-open base-volume box `X,Y,Z,W,H,D`. Because stored
prediction indices are point centres, both boundaries map with
`ceil(base/prediction_to_base_scale)`, with numerical snapping at exact lattice
boundaries. The resulting interval selects complete global anchor cells. A crop
outside the prediction grid or containing no stored prediction sample fails.

The command prints prediction/base scale, cell side and diagonal in base
voxels, counts, and elapsed time. It writes:

- `anchors.json`: versioned machine input for later fiberlet stages. It stores
  a credential-free manifest locator and content hash, coordinate contract,
  parameters, aggregate rejection counts, and only non-empty cells.
- `anchors.obj`: diagnostic base-coordinate line glyphs. It is not an input to
  later stages.

Both files use same-directory temporary files followed by atomic replacement.
Timing and worker count are not stored because they are operational values;
identical inputs and numerical parameters produce byte-identical artifacts
across worker counts.

## Calibration

Choose the largest cell for which distinct sustained parallel fibers cannot
share a cell. Compare both the cell side and cube diagonal in base voxels with
the minimum sustained sheet/fiber separation, and inspect the OBJ over the
source volume. A representative crop should be run at cell sizes 2, 4, and 8
before selecting production thresholds.

## Integer path stage

`vc_fiberlets paths` consumes the authoritative `anchors.json`, the same fiber
manifest used to create it, and a separate regular Lasagna manifest providing
surface normals:

```bash
volume-cartographer/build/bin/vc_fiberlets paths \
  /path/to/fiber.lasagna.json \
  /tmp/fiber-anchors/anchors.json \
  /tmp/fiberlet-paths \
  --normal-manifest /path/to/lasagna.lasagna.json \
  --stats
```

The command verifies the anchor artifact's manifest hash, prediction shape and
prediction-to-base scale. A content-identical manifest may be relocated. Direct
remote fiber or normal manifests use the same required `--remote-cache-dir`
and cache-first behavior as the anchor command. Fiber directions and regular
Lasagna normals are distinct: the fiber manifest's `nx/ny` are never treated as
surface normals.

Target cells are selected from the integer shell
`radius-0.5 <= length(cell_offset) < radius+0.5`; the initial radius is four.
Endpoint axes must agree with their chord within 45 degrees. Every surviving
pair is solved independently, so an anchor can currently participate in many
paths.

`--corridor-radius` is measured in base voxels. If omitted, it defaults to one
anchor-cell width. Cell radius and shell width remain dimensionless cell-lattice
parameters.

The path graph contains only integer stored-prediction voxels. Exact sub-voxel
anchors are virtual endpoints connected through nearby integer voxels. A
cubic-Hermite reference bounds the corridor, and 26-neighbour moves must have
strictly positive chord progress. DP state retains the incoming move, allowing
one-step curvature without a cumulative history state.

At each valid prediction voxel, the best direction available in the discrete
forward stencil has zero direction penalty. Other directions pay squared angle
excess above that local quantization floor. Low presence and direction costs
are integrated by edge length. Curvature uses the native tracer's shared
Lasagna-normal tangent-plane/normal-tilt split, with isotropic fallback for an
invalid normal. Invalid fiber predictions have a finite default cost of 4 per
prediction voxel, allowing short gaps to be bridged.

The command writes:

- `fiberlets.json`: every shell pair, rejection/failure reason, objective
  breakdown, and successful base-coordinate polyline.
- `fiberlets.obj`: one named successful polyline per group in base coordinates.

`--stats` prints retained anchors, candidate pairs, pre-DP rejections,
searched-but-unscored failures, scored paths, accepted fiberlets, and
min/mean/max total objective scores for all scored and accepted paths. Rejected
endpoint pairs and failed searches have no path score and are counted as
unscored, never as zero. Empty score ranges print `n/a`. There is currently no
quality cutoff, so every scored path is accepted and the two score ranges are
expected to match.

For MeshLab compatibility, each fiberlet OBJ group writes every adjacent path
edge as an explicit two-vertex `l a b` element. It does not rely on support for
multi-index OBJ polyline records.

This is an overcomplete diagnostic collection. There is no path-quality cutoff,
degree selection, overlap deduplication, extension, H/V or winding assignment,
or final graph construction yet. Inspect the OBJ on a small crop before using
the output for later graph work.
