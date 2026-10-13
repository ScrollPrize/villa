# Fiber surface experiment specification

- Upstream winding/sign/HV solutions are immutable inputs.
- Use actual AFV polylines and their signed arc mapping, not visualized chords.
- Select H pieces with chosen-winding joint probability >0.5.
- UV constraints stay fixed across optimization and all mesh pyramid levels.
- The data residual is full positional XYZ at fixed UV (not normal-only).
- Extrapolation creates geometry, not extra supervision; record its extent.
- Reuse Lasagna model, optimizer and regularizers; do not clone their math.
- Preserve all existing workflows and output tifxyz base-voxel coordinates.
- No claim that regularization guarantees a globally nonintersecting surface.
