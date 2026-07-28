Make native 3D Trace2CP metric-only by default.

The command line tool should still run the same native 3D Trace2CP tracing and
metric calculation, write the JSON summary, and print the metric lines. JPG
visualization and partial image updates should only run when explicitly enabled
with a visualization flag.
