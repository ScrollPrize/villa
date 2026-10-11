# H/V-specific rejection maps

- User requests V hard disabled length fraction, plain disabled count, rejection
  probability, separately/againstH/asV/H. Added fraction-of-pieces too so raw
  abundance cannot be confused with a rejection rate. Both probability weightings.
- Follow-up bin sizes:10000vx U profiles and3000vx U/V heatmaps. Same saved20
  CCW/CW states, no solve/inference. Plan hv_rejection_plan.md reviewed/approved.
- Implement conditional-posterior placement plus MAP integer counts. Count each
  piece once per intersected cell; compute profiles independently to avoid V
  height multiplying the same piece's U count. Hard threshold strictly>.5.
- Shared exact segment/grid overlap replaces pooled length quadrature for future
  pooled runs too; historical pooled outputs remain untouched. Explicit normalized
  source arc lengths remove float32 signed-offset roundoff. Not a geometry refit.
- Initial test used exact equality for floating mass2999.9999999999995; changed
  test to1e-14relative tolerance. MAP integer counts remain exact-equality checked.
- Independent implementation review caught nonzero-origin boundary rounding
  creating a zero-length extra count. Fixed ceil-based upper-bin selection and
  explicit positive-overlap filtering; negative-origin/hi=0 regression now passes.
  Interrupted first plot run and restarted corrected calculation; review approved.
- User's final heatmap size3000vx is reflected in config and dynamic plot labels.
- Completed correctedCCW/CW runs:33.36s/31.07scalculation,98.5s/94.9sincluding
  figures. No GPU. Both MAP and posterior placements, all requested metrics,
  H/V overlays, differences, ratios, support and numeric exports. Full mass and
  MAP-integer assertions passed.28tests passed including plotting smoke; final
  independent review approved, both conventions visually inspected.
- Globally H/V disabled lengths37.2833%/32.2703%CCW and37.4941%/32.3329%CW;
  V is less rejected overall. No convincing common full-height V-only bands;
  strongest ratios often low support/small H denominator and change with MAP vs
  posterior placement. No overlap labels inferred. hv_rejection_results.md links
  all figures and records global counts/probabilities, limitations and timings.
