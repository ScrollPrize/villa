# Task: Lasagna-oriented replay failure threshold

Change the replay failure evaluation used by both the classic greedy tracer and
the fiberlet tracer. Keep the configured base-voxel threshold unchanged along
the local Lasagna normal, but allow four times that threshold within the local
Lasagna tangent plane.

The forward reference matching itself remains unchanged. The new threshold is
used for dense trace-point failure decisions and for fiberlet seed acceptance.
The result must report enough component information to explain every decision.
An invalid local normal conservatively retains the existing isotropic threshold.
