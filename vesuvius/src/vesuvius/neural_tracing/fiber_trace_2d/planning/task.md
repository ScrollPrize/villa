# Task: evaluate fixed nonlinear uint16 fiberlet costs

Evaluate a fixed, dataset-independent nonlinear `uint16` representation of
fiberlet edge cost density. Encode
`sqrt(clamp(total_cost / path_length_prediction_voxels / 256, 0, 1))` over the
complete uint16 range and reconstruct total edge cost from the decoded density
and existing stored path length.

The encoding must never derive ranges from chunks or observed samples. Retain
the existing raw-total scenarios unchanged, reuse the existing
compact-direction geometry cache, and run the full Paris4 radius-768 comparison
at `H=384`, `D=48`, beam 16, exact search, and 32 threads.
