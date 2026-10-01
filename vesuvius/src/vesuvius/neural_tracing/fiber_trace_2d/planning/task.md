# Fiber cross-section evaluation

Add an opt-in training-test diagnostic which samples the predicted fiber
presence in a cross-section centered on the supervised fiber and oriented by
the Lasagna surface normal. The physical field is 160 base voxels in the
in-sheet transverse direction and 80 base voxels in the surface-normal
direction. Accumulate a per-pixel mean and bounded histograms, then write a
float TIFF mean image and per-pixel 0.1%, 1%, and 10% quantile float TIFFs.

Run the diagnostic on the best/current raw and EMA snapshots of the latest
Gaussian-refinement training run so their learned cross-sectional profiles can
be compared. This is evaluation-only and must not alter training targets,
metrics, refinement state, or checkpoint selection.

