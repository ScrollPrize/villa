# Task: hard split continuity and aligned winding signs

The 2048-voxel crop admits too many visibly invalid active fiber pieces.

Make consecutive pieces created by splitting one source fiber a configurable
edge-local hard continuation: when both endpoints are active they must have the
same H/V and winding state, while a Defect endpoint neutralizes that edge and
may separate two independently active runs. Make dominant perpendicular and parallel signed winding
constraints hard when their connector is sufficiently aligned with the aligned
Lasagna normal, with a configurable threshold defaulting to 30 degrees. Less
reliable signs retain the configured finite sign penalty.

Also report, for every solve, the number and percentage of final constraints
infringed overall and by canonical constraint class, while distinguishing
constraints neutralized by a Defect endpoint. Re-run the 1024 and 2048 crops and
compare the result with the prior behavior.
