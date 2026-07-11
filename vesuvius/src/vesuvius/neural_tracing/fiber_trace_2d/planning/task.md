# Trace2CP Vertical Space Doubling

Increase Trace2CP segment strip vertical room by another factor of two.

The segment-strip height is currently `2 * patch_shape_hw[0]`; change it to
`4 * patch_shape_hw[0]` for Trace2CP loading, visualization, and test
evaluation paths. This gives the tracer more vertical space before hitting RF
margin or strip-edge stop conditions.
