# Plan: integrate broader peak evidence for fiber anchors

## Kernel and sampling

1. Keep broad Gaussian direction/position refinement unchanged. Change only the
   subsequent direction-conditioned peak response.
2. Raise the transverse peak sigma default to `1.5` prediction voxels.
3. Add `peakAxialSigmaPredictionVoxels`, defaulting to `1.5 * cell-size` in the
   CLI (`6` prediction voxels for the default four-voxel cells). A straight
   fiber one cell from the pivot then has Gaussian weight about `0.80`, and the
   ends of the centered three-cell span have weight about `0.61`.
4. Score each candidate with the normalized anisotropic Gaussian
   `exp(-0.5*(r_transverse/sigma_transverse)^2
        -0.5*(r_axial/sigma_axial)^2)` multiplied by the existing
   `presence * abs(prediction_direction dot component_axis)^2` signal. Apply
   the existing cutoff independently in normalized transverse and axial
   distance, where transverse distance is distance to the candidate line and
   axial distance is measured from the candidate in its fixed pivot-normal
   plane. The support is the intersection of those two cutoff bounds. The
   denominator sums the anisotropic Gaussian for every sampled in-volume site
   inside both bounds, including invalid, below-floor, zero-presence, and
   unassigned sites. Replace the peak stage's old fixed axial half-width with
   `gaussianCutoffSigmas * peakAxialSigmaPredictionVoxels`; retain the fixed
   axial half-width only for the preceding broad direction fit.
5. Expand the extraction sampling halo to the outward-rounded maximum of the
   broad bound and the orientation-independent peak bound
   `hypot(localWindow + cutoff*peakTransverseSigma,
          cutoff*peakAxialSigma)`.
   Keep invalid/zero samples in the response denominator as before.

## CLI and strict artifacts

6. Add `--axial-sigma` in base voxels to `anchors` and `fiber-replay`, print its
   effective base-voxel value, and store
   `peak_axial_sigma_prediction_voxels` in the anchor and diagnostic artifacts.
7. Require the new field in strict C++ and Python readers. Do not accept or
   repair old experimental artifacts.

## Tests and measurements

8. Add focused tests for the core defaults (`1.5` transverse and `6` axial for
   four-voxel cells), CLI axial-default recomputation for non-default cell
   sizes and prediction-to-base scales, config validation, exact anisotropic
   weighting, evidence beyond the old axial slab affecting peak selection,
   complete oblique/block-boundary halo sampling, strict missing/exact artifact
   fields across C++ and Python readers, and unchanged deterministic output
   across blocks/threads. Retain explicit owner-cell/pivot-plane and broad
   support/NMS preservation coverage.
9. Build the anchor/path/replay targets with `-j32`; run focused C++ and Python
   viewer tests, Ruff, Python compilation, and diff hygiene.
10. Run the existing small real replay before and after the change and report
    anchor-stage timing and output counts. This is a signal-quality experiment;
    final scientific quality still requires visual inspection on the user's
    representative replay.

## Spec update

- Document the anisotropic peak kernel, default transverse/axial sigmas,
  normalized cutoff, expanded halo, and strict axial-sigma artifact field.

## Docs updates

- Document the peak-kernel controls, units, and intended multi-cell axial
  integration.

## Changelog update

- Record broader anisotropic peak evidence for anchor placement.
