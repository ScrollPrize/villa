\# Model (2D winding-space → image-space fitting)

## Goal

- Implement the full model in [`model.py`](model.py), including:
	- parameterization (mesh positions, global transforms)
	- target creation (cosine signal, optional modulation)
	- any resampling helpers needed for losses/visualization

The fitting/optimization code should consume the model as a black box and not re-implement model math.

## Coordinate systems

### 1) Mesh / winding coordinates (model domain)

- The model domain is a 2D grid (a “mesh”) with indices:
	- **horizontal index**: winding number
		- 1 winding == 1 cosine wave
		- cosine peaks correspond to “on-winding” locations
		- indices are conceptually 0, 1, 2, … in winding units
	- **vertical index**: mesh dimension along the surface

- Resolution:
	- There is a **base (low-res) mesh resolution** which is the actual parameter mesh.
	- We may generate **resampled (higher-res) versions** of fields derived from the mesh for:
		- derived losses
		- visualization
		- stable gradient computation

### 2) Image coordinates (data domain)

- The input is an image in 2D pixel space.
- All `xy` tensors use pixel coordinates (see [`docs/modeling.md`](docs/modeling.md)).

## What the model computes

## FitResult (runtime contract)

- The model is evaluated exactly once per optimization step via [`model.Model2D.forward()`](model.py:153).
- It returns a [`model.FitResult`](model.py:21) that caches all derived tensors needed by losses/visualization.

FitResult fields:

	- `xy_lr`: base mesh grid in pixel coordinates as `(N,Hm,Wm,2)`.
	- `xy_hr`: evaluation grid (upsampled) in pixel coordinates as `(N,He,We,2)`.
	- `xy_conn`: per-mesh connection positions in pixel coordinates as `(N,Hm,Wm,3,2)` with `[...,0,:]` left-connection, `[...,1,:]` point, `[...,2,:]` right-connection.
	- `data_s`: [`fit_data.FitData`](fit_data.py:12) sampled at `xy_hr`.
	- `mask_hr`: validity mask for `xy_hr` `(N,1,He,We)`.
	- `mask_lr`: validity mask for `xy_lr` `(N,1,Hm,Wm)`.
	- `mask_conn`: validity mask for `xy_conn` `(N,1,Hm,Wm,3)`.

	Direction encoding is documented in [`docs/modeling.md`](docs/modeling.md).

Implementation note:

- FitResult stores internal tensors as `_...` and exposes read-only properties.

Rules:

- Losses/visualization must consume FitResult and must not recompute model grids or resample FitData.
- FitData does not depend on the model; it only provides a pixel-space sampler.

## Grow stages & const mask (local optimization)

- A stage can optionally include a `grow` block (handled by the optimizer) that expands the mesh size by reallocating the parameter pyramids.

- After `grow`, the optimizer can set a constant-mask on the model (see [`model.Model2D.const_mask_lr`](../model.py:113)) to keep the pre-existing mesh region fixed during local optimization.

- Implementation: the const region is enforced by masking gradients in the backward pass (parameter hooks) and **not** by doing any forward-time mixing:
	- `const_mask_lr` has shape `(1,1,Hm,Wm)` and uses `1` for “keep constant” and `0` for “optimize”.
	- During local-opt, the optimizer registers a backward hook on the base (highest-res) residual tensors `mesh_ms[0]` & `conn_offset_ms[0]` and multiplies their gradients by `(1 - const_mask_lr)`.
	- The model forward path is unchanged; it always uses the current parameters in [`model.Model2D.forward()`](../model.py:134).

- `opt_window` (local-opt setting): controls how much of the “old” region adjacent to the grown border is allowed to update.
	- Default is `1` (only the grown cells update).
	- `opt_window = k` allows updating the grown region plus a band of `k-1` cells into the previously-existing mesh adjacent to the inserted rectangle.

## Scale-space pyramids & grow (reconstruction)

- `mesh_ms` and `conn_offset_ms` are residual scale-space pyramids.
- During grow, we edit the **integrated/base** tensor and then rebuild a consistent residual pyramid:
	- integrate → resize/edit (expand) → reconstruct residual pyramid.
	- this avoids having to “grow” every residual level independently.

Implementation:

- integrate: [`Model2D._integrate_param_pyramid()`](../model.py:275)
- grow pipeline: [`Model2D._grow_param_pyramid_flat_edit()`](../model.py:263)
- reconstruct: [`Model2D._construct_param_pyramid_from_flat()`](../model.py:281)

## Line-offset modeling (conn_offset)


- The model has a learnable multi-scale `conn_offset_ms` tensor (same scale-space handling as `mesh_ms`).
	- `conn_offset_coarse()` reconstructs the base-mesh offsets used for outputs.
	- The coarse/base result has shape `(1,2,Hm,Wm)`.
	- channel 0: vertical offset for the *left* connection
	- channel 1: vertical offset for the *right* connection

	- Interpretation:
		- For a base mesh point `(i,j)`, the left/right connection endpoint is in the neighboring column `j-1` / `j+1`, but vertically shifted by `conn_offset[...,i,j]`.
		- `0` means connect to the same-row neighbor.
	- `+1` means connect to one row lower.
	- `-1` means connect to one row higher.
	- Fractional values are linearly interpolated along the vertical axis.

- The pixel-space connection points are exposed via `FitResult.xy_conn`.

- Optimization:
	- `conn_offset_ms` is optimized via the same stage plumbing as `mesh_ms` (including `min_scaledown` handling).

### Mapping: image → winding coordinates (via sampling)

- The model attaches a **2D image position** to every mesh point.
- The mesh point index **is** the winding-space coordinate.
- From these per-mesh-point image positions, the model builds a sampling grid that maps winding-space samples into the input image.

### Global transform (optional)

In addition to per-mesh positions, the model can apply a global transform before sampling:

- **rotation** around the image center
- **global scale** (phase scale / x-scale in winding direction)
- **global phase offset** (shift along winding direction)

This global transform is applied consistently to the sampling grid (and thus affects all losses that depend on sampled intensities).

## Target data definition

### Base target: cosine in winding direction

- Target intensity is defined in winding-space.
- Along the **horizontal (winding)** axis: cosine signal.
	- one full period per winding
	- peaks correspond to on-winding locations
- Along the **vertical** axis: constant (unless modulation depends on vertical position).

### Optional modulation

- The model can create a modulated target from the base cosine via per-sample (or per-mesh) modulation fields (e.g. amplitude & bias).
- Modulation may be parameterized on the low-res mesh and then resampled to the evaluation resolution.

## Model outputs (conceptual)

The model should provide:

- `sampling_grid`: winding-space → image-space mapping (for sampling the input image)
- `pred`: sampled image intensities in winding-space at the chosen evaluation resolution
- `target_plain`: base cosine target at evaluation resolution
- `target_mod`: modulated target at evaluation resolution

Losses and visualization code should be able to request these at a consistent resolution (or request resampled versions).

## Initialization

Model init is driven by:

- image shape (H,W)
- `mesh_step_px`: default vertical step size in pixels
- `winding_step_px`: default horizontal step size in pixels

Initialization rules:

	- The model initializes its winding-space domain to cover **~2× the image extent** at the assumed steps.
	- This gives headroom above/below and left/right.
	- Cropping/expansion utilities will be added later.
	- Canonical winding-space coordinates span about `[-0.5*W, 1.5*W]` and `[-0.5*H, 1.5*H]` in pixels.

## Current data contract (FitData)

- Model init uses the supervision container [`fit_data.FitData`](fit_data.py:10) (not raw tensors).
- Current loader behavior:
	- expects a directory with 4 float tif channels: `*_cos.tif`, `*_mag.tif`, `*_dir0.tif`, `*_dir1.tif`
	- optional crop is applied to all channels consistently
	- optional downscale is applied to all channels consistently
