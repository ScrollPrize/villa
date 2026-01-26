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
- For sampling we use normalized image coordinates (e.g. as required by `grid_sample`).

## What the model computes

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

## Initialization (planned API)

Model init is driven by:

- image shape (H,W)
- `mesh_step_px`: default vertical step size in pixels
- `winding_step_px`: default horizontal step size in pixels

Initialization rules:

- The model initializes its winding-space domain to cover **~2× the image extent** at the assumed steps.
	- This gives headroom above/below and left/right.
	- Cropping/expansion utilities will be added later.
	- Canonical winding-space coordinates span [-2,2] (image spans [-1,1]).

## Current data contract (FitData)

- Model init uses the supervision container [`fit_data.FitData`](fit_data.py:10) (not raw tensors).
- Current loader behavior:
	- expects a directory with 4 float tif channels: `*_cos.tif`, `*_mag.tif`, `*_dir0.tif`, `*_dir1.tif`
	- optional crop is applied to all channels consistently
	- optional downscale is applied to all channels consistently
