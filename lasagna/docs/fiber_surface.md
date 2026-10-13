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
