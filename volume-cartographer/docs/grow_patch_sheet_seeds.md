# GrowPatch seeds with projection and streamlines

In a VC3D volume viewer, Ctrl+right-click a point and choose **Create Segment
(GrowPatch)**. The dialog opens in **GrowPatch (default)**, using the existing
optimizer and normal-grid workflow. Its existing defaults are unchanged.

Select **Projection** or **Streamline** to create a patch using the same CT
sheet-normal estimator and integration routines as Manual Grow. Select a raw
CT volume for these modes; they do not require a normal-grid file.

- **Iterations:** cells on each side of the fixed seed. N iterations produce
  a `(2N+1) × (2N+1)` grid; these modes default to 20 and allow 1–1024
  iterations. Switching modes preserves separate optimizer/sheet iteration counts.
- **Mesh spacing:** output vertex spacing in level-0 voxels (default 20).
- **Max integration step:** maximum internal step in voxels (default 2).
  Zero uses one integration step per output vertex. Smaller steps do not
  change the output grid dimensions.
- **Integrator:** Midpoint (default), Heun, or RK4, enabled for Streamline.
- **Workers:** independent line-integration threads (default 1).
- **Min size:** minimum accepted surface area in cm², as for default GrowPatch.
  Set it to zero for small exploratory patches.

The seed's CT normal defines a tangent plane. The grid's central column follows
projected volume Z (Y when nearly parallel to the normal), in both directions.
Rows grow perpendicular to this column in the local tangent plane. The seed
remains the center vertex. This constructs a new coordinate patch rather than
running GrowPatch's optimizer. Missing normals or leaving the CT volume fail
the run without publishing a surface. These are seed-only modes: resume,
inpainting, correction points, neural tracing, and direction-field constraints
continue to use the default method. Created surfaces include a generations
channel and can subsequently be edited or extended in VC3D.

The command output reports the active stage, completed/total vertices or points,
percentage **of that stage**, and elapsed seconds. It flushes a heartbeat every
second, including while CT reads are blocked. Stages cover seed-normal
estimation, central-column growth, row-direction preparation, row growth, and
surface preparation for saving. A stage reaching 100% does not mean the whole
job has finished; the command runner reports completion after saving.

Normal following does not snap back onto a particular papyrus surface or
prevent distant parts of the grid from overlapping. Inspect the result,
especially across sharp bends and poorly resolved material. Thread count does
not change line integration order or coordinates.

## Command line

The same modes are available through `vc_grow_seg_from_seed`'s JSON params:

```json
{
  "mode": "seed",
  "growth_method": "streamline",
  "generations": 3,
  "step_size": 20,
  "walk_step_size": 2,
  "walk_integrator": "rk4",
  "thread_limit": 4,
  "min_area_cm": 0
}
```

`growth_method` accepts `default`, `projection`, and `streamline`. Omitting it
keeps the existing GrowPatch behavior. Explicit `--seed` coordinates containing
zero are honored instead of being treated as a request for random seed search.
Sheet-mode defaults, if omitted, are
20 generations, 20-voxel mesh spacing, a 2-voxel integration step, Midpoint,
and one worker. Supply the selected CT's level-0 seed coordinates:

```sh
build/bin/vc_grow_seg_from_seed --volume /path/to/ct.zarr \
  --params /path/to/params.json --seed X Y Z \
  --target-dir /path/to/new-output --segment-name sheet-seed
```

## Build and checks

From `volume-cartographer`:

```sh
ninja -C build VC3D vc_grow_seg_from_seed test_sheet_seed test_grow_patch_seed_dialog test_manual_add_extend
ctest --test-dir build -R '^(test_sheet_seed|grow_patch_seed_dialog|manual_add_extend)$' --output-on-failure
build/bin/VC3D
```

The geometry regression tests cover fixed seeds, planar/cylindrical geometry,
all integrators, unsigned normals, invalid parameters, query failures, and
serial/parallel equivalence. The dialog test covers default values, mode
controls, accepted parameters, and cancellation. Existing manual-extension
tests exercise the extracted shared integration routines.

## Real CT validation

On Ubuntu arm64, RelWithDebInfo, the seed `(2574.79541015625,
2697.93798828125, 4293.798828125)` from the existing real
`build/streamline-bench/input` PHerc0139 crop was grown against:

```
https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0139/volumes/20250728140407-9.362um-1.2m-113keV-masked.zarr
```

Using two generations, spacing 20, integration step 2, and `min_area_cm: 0`,
projection and all three streamline integrators produced finite 5×5 surfaces,
kept the seed exactly fixed, and saved the generations channel and positive
area metadata. RK4 outputs with one and four workers had byte-identical XYZ
TIFFs. Invalid mode, unsupported sheet-mode resume, and out-of-volume seeds
failed without publishing geometry.

After extracting the shared walker, the existing right-edge manual-growth
case (two cells, one round, RK4, integration step 2, one worker) remained
byte-identical to `build/streamline-bench/pass2-rk4-c2-t1/before-1` for all
three coordinate TIFFs. Full commands and results are in the local
`build/sheet-seed-validation/summary.json`. This validates a small real CT
patch; large patches, macOS, and amd64 have not been tested in this change.

Progress reporting was also checked through a live stdout pipe on this seed,
for projection with one worker and RK4 with four workers. Stage counts and
one-second heartbeats arrived before process exit; XYZ and generations TIFFs
remained byte-identical to the runs above. The existing `vc_extend_edge` binary
also retained identical output. The captured progress events and commands are
in `build/sheet-progress-validation/summary.json`.

To rebuild the command-line tool and run the progress regression checks:

```sh
ninja -C build vc_grow_seg_from_seed test_sheet_seed test_manual_add_extend
ctest --test-dir build -R '^(test_sheet_seed|manual_add_extend|test_grow_seg_store_mode)$' --output-on-failure
```

An already running seed process retains its old code. New launches from VC3D
pick up the rebuilt tool; progress reporting does not require restarting VC3D.
