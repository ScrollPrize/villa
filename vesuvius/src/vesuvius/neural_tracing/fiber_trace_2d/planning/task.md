# Task: bounded fiberlet replay comparison

Add `--length N` to `vc_fiberlets fiberlet-replay`. The positive base-voxel
length starts at the first control point and limits graph extraction, greedy
replay, fiberlet replay, failure fractions, persisted reference geometry, and
optional failure visualizations to the same interval. Omission retains the full
remaining reference.
