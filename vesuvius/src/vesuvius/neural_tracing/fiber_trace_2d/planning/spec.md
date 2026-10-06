# Training trace execution contract

## Loss reporting

- Individual loss metrics exclude their configured outer weights. Intrinsic
  projection and spatial falloff weights remain part of the loss definition.
- Combined losses use weighted constituents internally and are logged before
  any weight on the combination itself. Total reports the optimization objective.
- Reporting changes must not change gradients, target counts, or sampling.
  Disabled terms remain disabled; historical logs are not rewritten.

## Optional reviewed-fiber annotation heads

- Unconditioned fiber U-Net optionally adds six sign-ambiguous normal channels,
  positive full-width scalar in network input voxels, and binary parent H/V.
  Disabled heads preserve the legacy layout. Auxiliary outputs are explicit,
  not interpreted as direction branches or exported automatically as volumes.
- Width-annotated reviewed data bakes dense normal axes with the shared VC3D
  Lasagna sampler and CP display-field corrections. Store coordinate identity,
  validity and source provenance. Missing/invalid samples provide no normal or
  width label; unresolved sources and malformed corrections fail preparation.
- Carry aligned labels through eligible-span slicing and geometry; transform
  normals with inverse-transpose Jacobians and widths with transverse stretch.
  H/V is inherited from the parent and is not re-estimated after augmentation.
- Normal supervision uses direction's Gaussian tube rasterization and winning
  segment, six-channel sigmoid MSE, normal-axis projection weights and Gaussian
  falloff, normalized by six times total falloff. No labels for unannotated
  samples. Normal metrics are unweighted; configured weight applies to total.
  Width SmoothL1 uses the same Gaussian tube/falloff, normalized by labelled
  falloff mass; H/V BCE remains on the selected annotated line. GT-only
  global counts preserve one backward per microbatch and logical-batch normalization.
- With a width head, extra sideband BCE is multiplied by detached linear width
  similarity to transformed GT width: one at agreement, zero at20% relative
  deviation. Keep original GT-only sideband denominator; ordinary presence
  positive/distribution/background losses do not depend on width. Missing width
  labels yield no additional sideband penalty. Legacy no-width models unchanged.
- Sample overview width is one base voxel per grayscale unit, clipped0–255.
  Only direction-modulated presence views additionally use the same width gate,
  relative to predicted width sampled at the center CP; raw presence is unchanged.
- Exact prepass/forward count validation is per rank before collectives, in the
  same microbatch order. Global error consensus shares the statistics reduction;
  fractional masses are not compared across different collective vector layouts.
- Raw and EMA evaluation and checkpoints retain all enabled heads and metrics.
  Prepared baked training data requires no runtime Lasagna reads.
- Baked test copies preserve the exact held-out piece selection and geometry.
  Dense normal labels use the same native baker as training; H/V comes from full
  held-out parents, never cropped-piece estimates. Preserve original files and
  source/parent hashes, and record updated output hashes in the test manifest.

## Resident reviewed trace evaluation

- Trace tests become eligible only after the globally aggregated raw dense-test
  mean angle first falls strictly below10 degrees with valid angle support.
  Eligibility is latched across subsequent worsening, pause/reallocation and
  checkpoint resume. Dense/raw/EMA tests and cross-section outputs stay ungated.
- Reviewed trace evaluation uses the same prepared workload, precision,
  preprocessing, products, native tracing, and metrics as standalone evaluation.
- Managed training lends its resident raw-model GPU replicas at a completed
  test boundary. Optimizer and EMA tensors remain resident and unchanged.
- Evaluation does not spawn a replacement Python benchmark or model process.
  Native tracing and existing parallel CPU loading/accumulation remain shared.
- Manager polling continues during evaluation. Revocation stops new inference,
  joins borrowed GPU work and cleans shared buffers before ordinary GPU drain.
  The exact pending test survives reallocation and retries before more training.
- Dense/raw/EMA metrics keep their existing sample-step conventions. The
  independent raw best-by-restarts checkpoint retains exact frozen provenance.
- Debug unmanaged training must explicitly disable reviewed tracing; no hidden
  extra-model fallback is permitted. A resident test cannot substitute a second
  checkpoint for its embedding channels.
