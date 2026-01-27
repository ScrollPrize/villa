\# Modeling notes

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
