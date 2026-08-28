# Task: preserve the complete winding visibility mask

Fix the Napari winding viewer so Previous/Next circularly shifts the complete
visibility mask through every integer winding in the observed range. Missing
or empty OBJ state artifacts must not delete a mask bit. Materialize a logical
field for every winding/state slot, especially H and V, and preserve arbitrary
manual visibility patterns exactly while shifting.

Also add a per-reference-fiber error table before the aggregate reference-to-BP
benchmark table. Each row must report right, wrong, and right-fraction values
for perpendicular, parallel-same, parallel-other, and sum, using the same
calibrated gauge offsets and active-only observations as the aggregate.
