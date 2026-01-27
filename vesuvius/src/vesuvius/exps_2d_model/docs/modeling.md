\# Modeling notes

## Coordinate conventions

- All `xy` tensors use **pixel coordinates**:
	- x in `[0, W-1]`, y in `[0, H-1]`.
- The `(x,y)` pair is always stored in the **last** dimension.
	- Examples:
		- sampling grid: `(N,He,We,2)`
		- base mesh: `(N,Hm,Wm,2)`
		- conn endpoints: `(N,Hm,Wm,3,2)`

Notes:

- When sampling with `grid_sample`, pixel coords must be converted to normalized coords `[-1,1]`.
- Validity masks are derived directly from the `xy` tensor being used and are never resized/interpolated.

## Direction encoding

Given a direction vector `(gx,gy)` (e.g. a gradient direction), define `theta = atan2(gy,gx)`.

Encodings (normal ambiguity handled by doubling the angle):

- `dir0 = 0.5 + 0.5 * cos(2*theta)`
- `dir1 = 0.5 + 0.5 * cos(2*theta + pi/4)`

Equivalent formulas (avoids computing `theta` explicitly):

- `cos(2*theta) = (gx^2 - gy^2) / (gx^2 + gy^2)`
- `sin(2*theta) = 2*gx*gy / (gx^2 + gy^2)`
- `cos(2*theta + pi/4) = (cos(2*theta) - sin(2*theta)) / sqrt(2)`

This matches `compute_frac_mag_dir()` (training) and the encoding used in [`opt_loss_dir.direction_loss_map()`](../opt_loss_dir.py:22).
