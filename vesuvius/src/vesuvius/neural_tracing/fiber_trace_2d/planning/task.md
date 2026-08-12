# Task: integrate broader peak evidence for fiber anchors

Improve the direction-conditioned local-maximum signal used to place cell
anchors:

- Increase the default transverse peak sigma from `0.75` to `1.5` prediction
  voxels.
- Integrate evidence along the fitted fiber direction with a substantially
  larger Gaussian sigma. Choose it so a straight fiber contributes with roughly
  comparable weight while passing through multiple neighboring cells.

Keep candidate positions constrained to the rotating normal plane and owning
cell. Preserve the existing broad direction fit, deterministic local ascent,
subvoxel fit, support filtering, and NMS behavior.

The fiberlet artifacts remain experimental and unshipped. Update their strict
schema directly without compatibility or repair behavior.
