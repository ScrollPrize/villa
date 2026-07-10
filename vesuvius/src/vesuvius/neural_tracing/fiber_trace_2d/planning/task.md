# Task: Coordinate Tensor Boundary Cleanup

Clean up fiber-strip coordinate handling so source-strip coordinate generation,
geometric coordinate augmentation, and line/control-point coordinate generation
do not bounce repeatedly between NumPy and PyTorch.

The desired behavior is one explicit conversion boundary in each direction:
metadata and Lasagna/VC3D inputs may enter as NumPy, torch coordinate work stays
as torch tensors on the configured device, and conversion back to NumPy happens
only at the final boundaries that require NumPy, such as VC3D coordinate
sampling, runner image export, and sample metadata consumed by existing callers.
