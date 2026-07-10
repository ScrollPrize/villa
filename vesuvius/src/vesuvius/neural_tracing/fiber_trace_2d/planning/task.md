# Task: Trace2CP Metric Command-Line Tool

Plan the `trace2cp` metric from `planning/plan.md` as a command-line inspection
tool, similar to the existing `--line-trace-vis` runner tool.

For now this should be a runner/export tool, not a training metric. It should
load the side-strip segment between two control points, run the existing
direction tracer from the first CP toward the second CP, compute the normalized
trace-to-target-CP score, print the score, and export a visualization that
overlays the traced line and score on the strip.
