# Task: fix fallback-level units for generated surfaces

Fix interactive fallback-range selection so viewport scale is compared with
volume chunk extents only when it is expressed in pixels per level-0 volume
voxel.

Plane views have that affine volume-space relationship. Generated,
parameterized, and flattened surfaces do not: their camera scale is pixels per
surface parameter unit. Those views must not use that value in volume-space
chunk coverage calculations and should instead queue the bounded five-level
fallback range.

Preserve queue ordering, rendering, sampling, and the opt-in visual download
overlay.
