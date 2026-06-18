# Flattening Implementation Status

Status date: 2026-05-21

This document summarizes the current `model-init=flatten` implementation in
`lasagna`, with enough context for a developer to continue the work.

## Current Variants

Flattening is selected with:

```json
{
	"args": {
		"model-init": "flatten"
	},
	"external_surfaces": [
		{"path": "/path/to/input.tifxyz"}
	]
}
```

There are currently two Adam-based variants:

1. `flatten_solver` omitted, `"torch"`, or `"inverse"`
   - Existing inverse-map path.
   - Optimizes `map_flatten_ms` as a regular output-grid map:
     `output_uv_pixel -> source_tifxyz_grid_coord`.
   - This is the original well-working bilinear Adam flattening path.

2. `flatten_solver: "forward"`
   - New source-UV path.
   - Optimizes `map_flatten_ms` as a source-grid UV field:
     `source_tifxyz_vertex -> output_uv_coord`.
   - Uses the same pyramid residual representation and Adam optimizer stages.
   - At save/export time, the optimized UV field is inverted onto a regular
     tifxyz canvas.

Use:

```json
{
	"args": {
		"model-init": "flatten",
		"flatten_solver": "forward"
	}
}
```

Reference configs:

- `lasagna/configs/flatten.json`: default inverse Adam path.
- `lasagna/configs/flatten_fast.json`: forward Adam, fixed 3 x 1000 steps.
- `lasagna/configs/flatten_long.json`: forward Adam, 3 auto stages.
- `lasagna/configs/flatten_forward.json`: forward Adam template matching the
  long auto-stage shape.

## Important Files

- `lasagna/fit.py`
  - Parses `--flatten-solver`.
  - Reads JSON `args.flatten_solver`.
  - Builds the flatten-only model.
  - Saves `model_final.pt` and `tifxyz/flatten.tifxyz`.

- `lasagna/model.py`
  - Owns flatten initialization and map integration.
  - `flatten_direction == "inverse"` keeps the old output-to-source map.
  - `flatten_direction == "forward"` uses a source-sized UV map.
  - `_flatten_invert_forward_uv_map(...)` performs CPU/Numpy export inversion.

- `lasagna/opt_loss_flatten.py`
  - Contains direction-aware flatten losses.
  - Inverse mode keeps the old sampled-surface loss.
  - Forward mode evaluates source quads directly against their optimized UV
    quads.

- `lasagna/tests/test_opt_loss_flatten.py`
  - Contains the current regression coverage for both inverse and forward
    flattening.

## Forward-Map Loss

The forward variant optimizes UVs on the source tifxyz vertex grid. For each
valid source quad:

- 3D source derivatives are computed from the original tifxyz quad.
- 2D UV derivatives are computed from the optimized UV quad.
- The symmetric-Dirichlet style term compares the UV metric against the 3D
  source metric, normalized by the measured `flatten_target_step`.

The forward path also reuses:

- `flatten_map_step`
  - Keeps local UV steps near one grid cell.
  - In forward mode this is masked by valid source vertices.

- `flatten_avg_offset`
  - Keeps the mean UV offset anchored to initialization.

- `flatten_orient`
  - Penalizes negative or low UV signed area.
  - In forward mode this is masked by valid source cells.

The optimizer is still plain Adam over `map_flatten_ms`; no sparse matrix solve
or SLIM local/global loop is active in this working path.

## Export Behavior

Both variants write the same normal VC3D-style tifxyz output:

```text
out_dir/
	model_final.pt
	tifxyz/
		flatten.tifxyz/
			x.tif
			y.tif
			z.tif
			meta.json
```

Invalid output pixels are written as `-1`.

For `flatten_solver: "forward"`:

1. The optimized source UV map is detached.
2. A regular output canvas is chosen from UV bounds plus
   `flatten_output_margin` padding, but never smaller than the old 20 percent
   larger source-based canvas.
3. Candidate source quads are found by a KD-tree over UV quad centers when
   `scipy.spatial.cKDTree` is available.
4. Each output UV pixel tries bilinear inverse coordinates inside candidate UV
   quads.
5. Successful inversions bilinearly sample the original source tifxyz.
6. Failed inversions remain invalid and export as `-1`.

The inversion is intentionally not differentiable. Optimization happens only on
the forward source UV map; inversion is an export step.

## Current Validation

The following checks have been run for the current forward-map implementation:

```bash
python -m py_compile lasagna/model.py lasagna/fit.py lasagna/opt_loss_flatten.py
python -m json.tool lasagna/configs/flatten_fast.json
python -m json.tool lasagna/configs/flatten_long.json
python -m json.tool lasagna/configs/flatten_forward.json
python -m unittest lasagna/tests/test_opt_loss_flatten.py
```

Additional synthetic smoke:

- Source-grid forward optimization reduced `flatten_sdir` from about
  `4.89938e-4` to `1.23279e-4` in 40 Adam steps on a small warped grid.
- `flatten_fast.json` and `flatten_long.json` load through
  `optimizer.load_stages_cfg(...)`.

## Known Limitations

- Forward export inversion is CPU/Numpy code. It should be acceptable for
  validation and current runs, but large outputs may need profiling.
- `scipy.spatial.cKDTree` is optional for small exports but required by the
  current implementation for large forward exports.
- Orientation is a penalty, not a hard constraint. Bad steps can still create
  folded UVs if weights or step caps are too loose.
- The symmetric-Dirichlet forward loss masks invalid, degenerate, or flipped
  UV quads out of the sdir average; `flatten_orient` is the term that should
  push those back.
- Checkpoints save `flatten_map_flat` as the export/output inverse map, not the
  raw optimized forward UV map. This keeps `fit2tifxyz` compatibility.

## Historical Note

A SLIM/local-global solver was tried before this forward Adam path, but it did
not perform well enough compared with the existing Adam/bilinear optimizer and
was reset out of the working implementation. If revisiting SLIM, do not treat
the old experiment as a known-good baseline. The current working direction is:

- keep the proven pyramid + Adam optimizer,
- optimize source-grid UVs for the forward variant,
- invert only at export.

## Suggested Next Work

Good follow-up tasks:

- Benchmark `flatten_fast.json` vs `flatten_long.json` on a representative
  real tifxyz and record wall time, final loss, valid output fraction, and
  visual quality.
- Profile `_flatten_invert_forward_uv_map(...)` on large outputs.
- Save the optimized forward UV map in checkpoints under a separate field,
  for example `flatten_forward_uv_flat`, while keeping `flatten_map_flat`
  as the compatible exported inverse map.
- Add a stricter line-search or projected update for forward UV orientation if
  real data shows fold spikes.
- Consider a faster spatial index for UV inversion if KD-tree candidate search
  becomes the export bottleneck.
