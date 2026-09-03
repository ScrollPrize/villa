# Task: Fiber Crop Evaluation Benchmarks

Add a reference-endpoint replay benchmark for a combined Fiberlet crop. Load
tagged reference fibers, clip them to the requested base-XYZ crop, and trace
from both ends of every contiguous in-crop run with the current Fiberlet replay
search. Use the shared anisotropic replay threshold. Report mean traced length
to first failure in millimeters and aggregate success as traced reference
length divided by total directed reference length.

Provide reproducible Markdown run records for this benchmark and the existing
1024-crop oracle pruning benchmark. Add a separate Markdown results table that
indexes recorded runs. Every run record must include the villa Git revision,
commands, artifacts, effective settings and weights, timing, and results.
