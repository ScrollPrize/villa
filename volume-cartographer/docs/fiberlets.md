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

Integer stored-prediction indices are voxel centres. A cell has a globally
fixed origin and owns the half-open ZYX range
`[cell * size, min((cell + 1) * size, shape))`. A crop selects every global
cell it intersects and samples that cell in full. It does not move or truncate
interior cells. Anchor JSON positions are stored once in prediction-grid XYZ;
base XYZ is derived by multiplying by `prediction_to_base_scale`.

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
  --cell-size 4 \
  --crop-prediction-xyzwhd 1696,2528,2264,32,32,32 \
  --threads 32 \
  --cache-gib 8
```

Cell size is restricted to 2 through 8 stored prediction voxels. If omitted,
Gaussian sigma is half the selected cell size. The presence floor and aligned
support threshold are inclusive. `--base-voxel-size-um` adds optional physical
reporting metadata but never changes the solve.

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
