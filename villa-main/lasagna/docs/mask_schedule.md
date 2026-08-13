# Mask scheduling (stages)

This document specifies the mask scheduling feature used by optimization stages.

## Goals

- Support generic mask scheduling via stage configuration.
- Keep mask generation extensible via a `type` registry + free-form parameters.
- Apply scheduled masks through the existing image-space validity masking path so supported losses automatically respect it.

## JSON configuration

### 1) Per-stage mask definitions

Stage configuration JSON defines masks **inside each stage** (not in `base`).

- Required fields:
	- `label`: unique mask identifier
	- `type`: mask generator type
- All other fields are passed as a dict to the generator of that `type`.

Example:

```json
{
	"base": {
		"gradmag": 1.0
	},
	"stages": [
		{
			"name": "stage0",
			"steps": 2000,
			"masks": [
				{
					"label": "cw_pie",
					"type": "central_winding_pie_ema",
					"ramp_start_it": 0,
					"start_size_segs": 32,
					"ramp_stop_it": 2000,
					"blur_sigma": 7.0,
					"losses": ["gradmag"]
				}
			]
		}
	]
}
```

### 2) Per-mask loss selection

To make a scheduled mask actually affect optimization, each mask definition includes:

- `losses`: list of loss names the mask applies to (e.g. `["gradmag"]`).

Only losses whose name matches an entry in `losses` receive the scheduled mask multiplier.

## Scope & semantics

- Masks are **image-space only**.
- Scheduled masks are computed from the model’s EMA state (see below).
- Final effective mask for a loss is:

`effective_mask(loss_name) = base_img_validity_mask * product(stage_masks_applicable_to(loss_name))`

`base_img_validity_mask` remains the existing validity/crop mask derived from sampling coordinates.

## EMA tracking (model runtime state)

The model tracks exponential moving averages needed by mask generators:

- `xy_ema` (LR): EMA of the current winding grid `xy`.
- `xy_conn_ema` (LR): EMA of the current connection field `xy_conn` (used for normals / direction).

### Update mechanism

- EMA integration is done via a dedicated `update_ema(...)` function.
- The optimizer calls `update_ema(...)` on every iteration (or at least whenever model outputs are refreshed).
- EMA weight is fixed at `0.99`.

The EMA tensors are the authoritative source for scheduled mask computations.

## Mask generator interface

Mask generators are selected by `type` and receive:

- The full parameter dict (excluding `label` and `type`).
- Access to the model EMA tensors (`xy_ema`, `xy_conn_ema`).
- The current stage iteration `it` (0-based within the stage).
- The image size for the current optimization run.

They return an image-space mask tensor (float in `[0,1]`, shape compatible with the loss’ mask usage).

## Implemented mask types

### `central_winding_pie_ema`

This is the first (and currently only) scheduled mask type.

#### Parameters

- `ramp_start_it` (int): iteration at which the mask begins to ramp.
- `start_size_segs` (float/int): initial size in segments along the central winding.
- `ramp_stop_it` (int): iteration at which the ramp completes.
- `blur_sigma` (float): Gaussian blur sigma applied to the binary shape mask.
- `losses` (list[str]): loss names this scheduled mask applies to.

#### Definition

The mask is a “pie” wedge (or triangle) derived from the **central winding** and the **central XY position**, computed from EMA state.

1) **Select the central winding & center point**

- Use the central winding of the LR mesh.
- Compute the central XY position from EMA `xy_ema` and/or `xy_conn_ema` (for direction).

2) **Choose two boundary points along the winding**


- Compute a progression factor based on stage iteration:
	- If `it <= ramp_start_it`: progression = 0
	- If `it >= ramp_stop_it`: progression = 1
	- Else: linearly interpolate in `[0,1]`.
- Convert progression to an effective segment-span by ramping from `start_size_segs` (at/ before `ramp_start_it`) up to the full central-winding size in segments (≈ `Hm-1` for `xy_lr`) at/ after `ramp_stop_it`.
- Pick two mesh points by stepping (fractionally / interpolated) along the central winding:
	- At ramp start, boundary points are `center ± (span/2)` segments.
	- As ramp progresses, the span is adjusted according to the schedule.

3) **Compute boundary rays (using normals from `xy_conn_ema`)**

- For each boundary point, compute a normal direction using EMA `xy_conn_ema`.
- The direction is defined from connection-left to connection-right (conn-left → conn-right).
- Each boundary ray is a line passing through its boundary point along the computed normal.

4) **Construct the polygon**

- Extend both rays until they intersect the image borders.
- Additionally compute the intersection of the two rays:
	- If the rays intersect *within* the image bounds, use that intersection as the apex and draw a **triangle** (intersection + the two border intersection points).
	- Otherwise draw a **pie wedge** bounded by the two rays and the relevant image border intersections.

5) **Rasterize & blur**

- Rasterize the polygon in image space using OpenCV.
- Apply Gaussian blur with `blur_sigma`.
- Normalize/clamp to `[0,1]`.

The resulting float mask is used as a multiplier in `effective_mask(...)` for the configured losses.

## Mask application point (centralized)

Losses obtain their image-space mask through a shared “mask selection” helper.

- The helper takes at least:
	- the base validity mask (derived from XY/crop validity)
	- the current stage mask schedule (if any)
	- the `loss_name`
- If the active stage references a mask and `loss_name` is listed in that mask’s `losses`, the helper multiplies the scheduled mask into the base mask.

Initial support requirement:

- The scheduled masking must apply to the `gradmag` loss.
- The integration is generic so additional losses can opt-in later by name.
