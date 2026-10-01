# Local fiber sideband supervision and test visualization

Add separately normalized, Lasagna-normal-oriented left and right negative
sidebands around each densely supervised fiber. Preserve the existing
anisotropic Gaussian tube, but explicitly suppress presence outside its width
support in a swept rectangle covering the observed approximately +/-40-base-
voxel secondary peaks.

During configured training tests, accumulate the aligned predicted presence
cross-section over the complete deterministic test set and write/log the mean
and 10% exceedance (`q0.1`) images. Stop the previous training only after the
implementation is validated, then resume its latest checkpoint with the new
loss and visualization enabled.
