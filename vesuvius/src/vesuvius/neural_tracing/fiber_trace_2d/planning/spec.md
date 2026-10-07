# Training trace execution contract

## Optional replay presence filtering

- Cached output-only length analysis may discard completed saved pieces below a minimum actual traced-polyline arc length in base coordinates (after rollback; equality retained). It must preserve native traces and online totals, expose discarded coverage/pieces, and distinguish surviving endpoint counts from original restarts. GT-error boundaries already split benchmark pieces, so this is GT-assisted retained-output analysis, not a demonstrated GT-free detector. Report actual arithmetic piece mean alongside historical virtual retained-distance/(surviving stops+1).

- Optional `--replay-presence-mode modulated` uses the actual winning candidate's `1 - fiberLocalAlignmentLoss`, captured during scoring in both sample and fused-corner paths and carried to committed replay points. This includes width scaling and all six positive-clamped direction dot products, but excludes separate smoothness/embedding penalties. Raw remains default. Width-off still includes direction modulation. New-segment seeds/reseeding retain raw presence requirements because their width/history is reinitialized.
- Cutoff0 disables the filter. In raw mode, positive cutoff in[0,1] accepts equality and tests raw unweighted presence at every newly committed forward point, maximum finite value among valid prediction options. Missing/invalid options count0; published-chunk readiness remains unchanged.
- A rejected point is filtered, not an ordinary error. Retain previous accepted prefix without FW/BW, or last reverse-verified prefix with FW/BW. No terminal reverse check is added by presence rejection. Record presence_filtered separately from reverse_consistency_filtered.
- Reuse existing GT-guided aligned local-maximum restart search within10/20 ellipse, additionally requiring presence>=cutoff. Always advance; exhaustion skips remaining reference. Initial supplied GT seed is not itself threshold-tested; failures before any committed point retain the original handling.
- No change to beam scoring, widths, sampling precision, scheduler or disabled protocol. Retained distance and filtered counts must accompany error counts.

## Optional cumulative-width tracing

- Explicit width inference exports the annotation head's positive softplus
  full width in network-input voxels as a float32 scalar product. It is part
  of cache identity and coherent chunk readiness; legacy caches remain valid
  only for their original products. Normal/HV heads are not implicit outputs.
- The opt-in tracer multiplies presence by
  max(0,1-abs(sample_width/mean_width-1)/relative_tolerance), initially0.20.
  Optional cosine weighting instead uses cos(pi*d/2) for d<1, else0,
  where d=abs(sample_width/mean_width-1)/relative_tolerance. Linear is default;
  zero tolerance disables gating. Width shape does not change inference identity.
  Seed and newly committed forward points contribute equally to a cumulative
  arithmetic mean. Beam candidates never contribute. Update before observers;
  shared-prefix commits can lag speculative frontier points.
- Reverse consistency copies the current forward mean and never initializes
  or updates it during the reverse trace. Forward restarts reset the mean.
  Invalid candidate widths give zero presence evidence; invalid seed/committed
  widths fail explicitly. Disabled width leaves original scoring unchanged.
- Existing inference parallelism and precision are preserved. Width export
  currently requires an unconditioned width head and does not support TTA.

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
