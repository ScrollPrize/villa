# Task: detailed replay overview and restart markers

Increase the full replay overview JPEG to eight times its current top/side
strip resolution so CT and line detail are directly inspectable. Mark every
greedy and fiberlet failure/restart position with a vertical line so the
relevant errors are easy to locate. A bounded `--length` run must still show
only its selected interval.

Keep the existing per-failure textured strip artifacts unchanged and continue
to use the existing VC3D surface and shared CT rendering infrastructure.
