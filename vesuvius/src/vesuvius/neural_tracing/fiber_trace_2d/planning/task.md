# Task: Whole-Run Reference Replay Benchmark

Update the reference endpoint replay benchmark so a failure does not terminate
the directional case. Resume replay after each failure and evaluate the entire
in-crop reference run, because one reference fiber may contain multiple failure
locations. Preserve both endpoint directions and the existing anisotropic
normal/tangential failure criterion.
