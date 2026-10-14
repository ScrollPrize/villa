# Fixed-UV fiber surface fitting

## Scope and implementation

This opt-in experiment reconstructs a single unwrapped quad surface from the
frozen U/V assignments of retained H fiber pieces. It does not change the
winding solve, use V observations, read inference volumes, or require normals.

`fiber_surface.py` builds geometry and calls the **existing**
`optimizer.optimize` with `Model3D.from_tifxyz_crop`. Its five-level XYZ residual
pyramid and existing `smooth`, `step`, and `bend` losses are reused directly.
`opt_loss_fixed_uv.FixedUVTargets` supplies the new `fixed_uv` term. The adapter
in `vesuvius/.../fiber_lasagna_surface.py` uses canonical `hard_geometry` and the
shared normal-free `PolylineBank`; the old H/V surface interpolation is unchanged.

All coordinates are base voxels. H arc length increases or decreases U according
to the already resolved direction; V stays fixed along each H piece. Selection
requires both retained status and chosen-winding joint probability >0.5.

## Initialization and extrapolation

At each U column, evaluate all H curves spanning that U. Interpolate in V between
the nearest distinct H heights. Exact ties choose the lowest piece ID. Outside
that height range, extend the nearest secant, limiting extrapolation speed to
`max_extrapolation_slope` (default 1 XYZ voxel per UV voxel). Supported
interpolation and observations are never clamped or rejected by this limit.

A column with one H borrows the nearest multi-H column's **integrated bounded
transverse increments**, translated to its own observation. Empty U columns are
interpolated between known columns; outer ones use bounded secant extrapolation.
The finite output rectangle is the requested UV bounds, rounded out to the grid.
There must be at least one multi-H column and two supported U columns; otherwise
initialization fails explicitly rather than inventing a transverse direction.

`targets.npz` includes the support map:

| Class | Meaning |
|---|---|
| 0 | Interpolation between H supports |
| 1 | V-boundary extrapolation |
| 2 | Single-H column with borrowed transverse shape |
| 3 | Empty U column interpolated from supported columns |
| 4 | U-boundary extrapolation |

## Positional loss and stages

For each observation, freeze its quad and bilinear weights and penalize
`sum(arc_weight * ||S(U,V)-fiber_xyz||²) / mesh_step²`. Samples include piece
endpoints, all crossed grid edges, original polyline knots, and subdivisions at
the configured spacing. Trapezoidal arc weights avoid overweighting dense knots.
H curves have constant V, so their ordinary grid crossings are U/column edges.

This is a **full XYZ positional residual at fixed UV**, not a normal-only
residual and not a ray intersection search. XYZ vertices can move in all axes.
All pyramid levels evaluate on the same finest grid; changing trainable levels
does not change correspondence. A geometry-changing grow/crop is unsupported and
rejected. Fine-grid target evaluations differentiate into the shared pyramid.

`args.fixed_uv_ramp: [start,end]` specifies absolute, positive, geometrically
interpolated weights for the stage. Start cannot be zero; end cannot be smaller.
The surface wrapper also rejects weight decreases between stages. The tested
default is 600 iterations, LR 5, data weight 1→20, smooth 0.5, step 0.5, bend 5.
These are experimental settings, not a demonstrated global-topology guarantee.

## Run and output

Use the existing evaluator in this worktree:

```bash
/home/hendrik/.venv_las/bin/python \
  /home/hendrik/villa4/vesuvius/scripts/run_classic_fiber_evaluation.py \
  crop-benchmark --config \
  /home/hendrik/villa4/lasagna/configs/fiber_surface_pilot_geometry.json
```

The checked-in config pins a completed winding solution in villa2 as read-only
input. All output is in villa4. Existing output directories are rejected: use a
new `output_root` for another run. Outputs are `initial.tifxyz`, `surface.tifxyz`
(LZW-compressed float TIFFs), fixed targets/support map, source/code hashes,
effective config, progress history and initial/final metrics.

The CPU pilot uses 8 threads and requests **no GPU**. CUDA is supported through
the existing manager launcher: set `device: "cuda"`, `devices: "0"` (one physical
GPU) and `gpu_manager` settings. A shared reservation can use
`exclusive: false, vram_bytes_per_gpu: 4294967296, cpu_cores: 8`; size that budget
for the intended mesh. The managed child uses local `cuda:0`; resources are
released by the existing process-lifetime manager wrapper. CUDA execution/timing
has not yet been measured for this experiment; dispatch is regression-tested.

## Validation and limitations

See `../planning/fiber_surface/results.md` for measured pilot results. The
implementation is Python/PyTorch/NumPy, using existing Lasagna dependencies and
no new build or install. CPU tests use the established las venv; CUDA uses the
manager. No platform-specific numerical kernels were added.

The local opposing-triangle statistic detects some folded/twisted quads, **not
all self-intersections**. Regularization does not guarantee untangling. Targets
incompatible with each other at the chosen resolution remain in the fit and can
leave large errors or folds. Extrapolated regions are guessed geometry. The
current pilot is an inspection artifact, not a validated production surface.
# Joint sliding attachments (2026-10-14)

Set `sliding_uv: true` and supply `deformation_root` for the implicit original /
deformed comparison. The deformed retained H mask is authoritative for both;
`apply_offsets` chooses whether saved source/arc U/V offsets are added. Actual XYZ
curves are unchanged. Base states/widths and piece identities must match.

Each attachment has a trainable two-coordinate displacement. Its physical source
arc stays fixed, including endpoints, original/offset knots and initial U/V grid
crossings. Sampling is not repeated as UV moves. Nonconstant V and local U reversal
are supported. This mode currently processes the full selected regional input;
explicit `uv_bounds` is rejected rather than silently cropping or ignoring it.

The existing Lasagna optimizer updates mesh and attachment variables in the same
Adam step from iteration one. Normal-distance loss updates geometry only (UV and
normal detached); tangential-distance loss updates UV only (mesh and normal
detached). Normals are analytic bilinear-quad normals recomputed every evaluation,
not Lasagna volume reads. This gradient routing is deliberate, not an alternating
solve or the full derivative of an ordinary point-to-plane energy.

Regularization penalizes normalized displacement, changes in displacement along
the fiber, and excessive stretch/collapse relative to the input mapping. Shared
source endpoints are linked; gaps between pieces are not bridged. Weak mean-shift
penalty anchors the global translational gauge. Nominal soft allowances are 0.1
local winding width horizontally and 1000 base voxels vertically; beyond these
the penalty increases. They are not hard caps. Mesh padding and out-of-domain
penalties allow sliding without a zero-gradient clamp at the border.

Use the same evaluation runner with `lasagna/configs/fiber_surface_joint_original.json`
or `fiber_surface_joint_deformed.json`. These reserve non-exclusive 8 GiB grants
on GPUs 0 and 1 respectively. Each run uses one GPU. Results and limitations:
[joint_uv_results.md](../planning/fiber_surface/joint_uv_results.md).
`attachments.npz` saves initial/final grid UV, source IDs, piece IDs, arc positions,
XYZ targets and nominal movement scales. Grid UV becomes base UV using the
`targets.npz` grid step and origin. Existing fixed-UV mode is unchanged.

Regularization sweep: [smoothing_results.md](../planning/fiber_surface/smoothing_results.md).
3×smoothness reduced local folds; increased bending did not. Remaining folds
require further investigation; no settings were silently promoted to defaults.
Report-only `summarize_runs` settings reuse the shared geometry-mask calculation
and split counts by final attachment-cell occupancy without allocating GPUs.

## Local patch repair (opt-in)

Set `repair_from` to a completed joint-sliding surface directory. The existing
runner dispatches to `fiber_surface_repair.py`; it reads saved mesh and attachments
without reloading AFV or rerunning winding optimization. `repair_settings` selects
halos, `largest_first`, guide weights, optional `seam_weight`, and blend fractions.
Reconstruction freezes boundary vertices and attachment UV, solves a sparse
least-squares interior, and rejects new local defects or excessive fiber-fit
damage. `seam_weight` preserves the incoming boundary slope through a penalty
on changes in the frozen boundary Laplacian. Rejected patches remain unchanged.

The matched continuation control and repaired solve reuse the existing optimizer
with fresh Adam state; original UV remains the displacement reference. An opt-in
`SurfaceGuard` adds a soft signed-corner-area penalty. It does not guarantee
nonfolding, and stronger weights did not reliably cure recurrence in these tests.
Normals/areas of already defective reference cells are not protected.

Use configs `repair_tangent_original.json` / `repair_tangent_deformed.json` with
the same runner above. Outputs preserve before/repaired/control/continued tifxyz,
attachment coordinates, attempts, metrics, configuration and input/code hashes.
All sizes/errors are base voxels. Results, all variants and exact commands:
[repair_results.md](../planning/fiber_surface/repair_results.md).

## Matched CT and fiber overlays

Use the existing evaluation entrypoint (no GPU required):

```sh
/home/hendrik/.venv_las/bin/python /home/hendrik/villa4/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa4/lasagna/configs/render_deformed_surface.json
```

`fiber_surface_render.py` reuses the TIFXYZ loader/bilinear interpolation and
native VC blocking trilinear CT sampling, with remote-cache fetching rather than
interpreting an incomplete local mirror as an empty scan. It does not optimize
or reproject the surface. The config selects the surface, matching saved final
attachment NPZ, scan URL/level and UV-layout base voxels per pixel.

Future CT-only renders are `ct.jpg` (JPEG quality95, requested by user).
Other outputs remain LZW TIFFs: transparent RGBA `fibers.tif`, RGB
`ct_fibers.tif`, and `valid.tif`. All have identical pixel grids, no decorations
or independent resizing. Pixel0 maps to surface vertex0; attachment grid UV
multiplied by grid spacing/pixel size maps to the exact same pixels. U is right,
V is down. The physical surface scale can vary due to mesh distortion.

Colors identify source fibers. Lines connect saved final attachment UV samples
in source-arc order, separately for each cropped piece. They show the fitted H
fibers, not new nearest-XYZ correspondences or V fibers. `fibers.json` maps
colors/pieces back to source IDs and arc intervals. Lines now use8px (originally
2px). To refresh only overlays, use the same runner with config
`lasagna/configs/render_deformed_thick_fibers.json`; it validates input hashes and
reuses CT/mask without fetching or sampling. The scan is sampled at zero normal offset (no slab projection), level2
(4x downsample), at4 UV base voxels/pixel. Remaining surface folds and fitting
errors are visible rather than repaired or concealed. `summary.json` records
input hashes, mapping, coverage, native sampling statistics and timings.
# Connected H/V fitting (opt-in)

Use `connected_hv: true`, `sliding_uv: true`, `apply_offsets: true` with the
deformed `deformation_root`, and `normal_root` pointing to the sampled normal
curves. `v_data_weight` defaults to **0.1**; H and V data losses are normalized
separately by their supported arc lengths, so adding V does not dilute H.

At each saved H/V crossing, measure `offset = dot(H_xyz - V_xyz, inward_V_normal)`.
Linearly interpolate these signed offsets along V source arc; extend endpoints
constantly. Unconnected V fibers use the global mean of distinct crossings.
The frozen V target is `V_xyz + offset * inward_V_normal`. H targets stay on H.
Mesh gradients still use normal error, attachment gradients tangential error.
The normal sign convention cannot invert the target (flipping both normal and
signed offset leaves it unchanged). Invalid V normals cause an explicit error.

Each connected source-fiber component has one shared UV translation. Local
attachment motion is exactly zero at saved crossing nodes, but remains learnable
between and beyond them. This preserves relative crossing UV layout throughout
optimization, without another alignment solve. Arc/offset knots and mesh-line
crossings are included in sampling. Exact crossing UV uses the original signed
event distances, avoiding roundoff from serialized source arcs.

`targets.npz` and `attachments.npz` include family, component, crossing mask,
crossing attachment pairs, raw fiber XYZ, shifted target XYZ, normals and offsets;
attachments additionally save shared/local motion for faithful restoration.
The existing ruled initializer still uses H only. Existing H-only modes remain
available. The legacy post-fit sample crop is rejected for connected mode rather
than silently dropping connected neighbors; whole-component cropping is not yet
implemented. This change does not resolve the previously reported mesh folds.

CPU real-data smoke (three optimizer steps, **not a converged surface**):

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa4/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa4/lasagna/configs/fiber_surface_connected_smoke.json
```

Actual 800-step deformed H/V fit: `configs/fiber_surface_connected_solve.json`.
Render: `configs/render_connected_hv_deformed.json`; use the same runner above
with the selected config. Final outputs are in `data/connected_hv_deformed_solve_v1`
and `data/render_connected_hv_deformed_v1`. The latter contains `ct.jpg`, transparent
`fibers.tif`, combined `ct_fibers.tif`, and `inspection.tifxyz` for vc3d.
`column_range` is a half-open inspection crop; it does not alter solve topology.
The render is the central14..21 winding band, while the fit retained the full graph.
Normal RMS within this band is1.0124H /6.0967V-offset basevoxels. There are735folded
quads in that band; this output is not repaired or established as a new best result.

# Connected-component selection

Connected H/V fitting defaults to `component_min_fibers: 3`: retain entire saved
crossing-graph components containing at least three distinct source fibers and
both H and V. Multiple pieces of one source count once. This runs before grid
construction, ruled initialization and offset measurements. It removes isolated
fibers and H/V pairs, not low-degree members of larger connected groups.
Use `0` to replay the old unfiltered selection. Upstream solutions are unchanged.
Counts and retained source IDs are saved under `connected_hv.selection` in the
summary. Filtered-run config: `lasagna/configs/fiber_surface_connected_min3.json`.

## Stronger offset smoothing trial

`fiber_surface_uvsmooth10.json` and `fiber_surface_uvsmooth100.json` keep the same
800-step solve but set `sliding_settings.displacement_weight=0.1` (was0.01) and
`smooth_weight=10/100` (was1). This penalizes changes in the added UV offset along
source arc, not curvature of the input fiber mapping. Component motion remains
possible; crossing locks are unchanged. Mesh smoothing weights are unchanged.

Smooth100 reduces local-offset p95 from186.0 to35.3basevx and consecutive offset
slope-change RMS by88.2%, but H/V normal error rises1.07/5.05→1.89/10.90basevx and
folds215→279. This is the smoother-offset inspection candidate, not a repaired or
universally better surface. Baseline configs remain reproducible.
`audit_uv_offsets.py RUN_DIR ...` verifies matched attachments and reports the
displacement/slope statistics without modifying results or requiring a GPU.

## Geometry follows the UV attachments

`sliding_settings.geometry_tangent_weight` defaults to0 (normal-only geometry).
Positive values add tangential squared error to geometry evaluated at detached
current UV, with detached surface normals. At1 this is full positional fitting.
The new term never moves UV; the existing UV tangent loss/regularizers remain.
H uses actual XYZ, V its separately weighted normal-offset target. The weight is
saved with attachments and restored, including old-checkpoint default0.

Matched smooth100 trials reduce H full-XYZ RMS50.75→8.34/8.21/7.47basevx for
weights0.1/0.3/1, but folded quads279→358/418/517. Configs
`fiber_surface_geom01.json`, `fiber_surface_geom03.json`, `fiber_surface_geom1.json`
reproduce these trials. Inspect0.1 for the conservative alignment/fold tradeoff;
1 gives the closest positional fit, not a repaired mesh. Existing defaults stay.

Future renders default to8baseUV/pixel with4px fiber lines (previously4/8).
Full-render presets use these values; existing images and their frozen configs
are unchanged. The same central view is7351x3188 instead of14701x6376. CT scale,
UV extent and JPG/LZW TIFF output formats remain unchanged. Old overlay-only
refresh presets retain widths appropriate to their existing canvas.
