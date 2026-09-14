# Training span eligibility

- User requested launch: approved experiment production-start launched root
  1355260 at20260914_105433, fresh (no resume), regular JSON-only LR0.01.
  Log: /ephemeral/hendrik/fiber3d_train2/runs/.launches/5scroll_json_eligible_128_sd1_16x8_ema-20260914_105433/train.log.
  Manager task9ca62151-ab05-492b-a865-d8cd3f7b677c entering automatic exclusive
  startup calibration before adaptive training; root confirmed alive.
- Prior sweep/results preserved in05159442e and refinement docs.
- User specifies cspline/trace only, no length threshold; trim tails.
- Separate tracer evaluation unchanged; no training started.
- Implemented maximal accepted components before bounds checking; component
  range participates in stable identity. Synthetic `_array_records` remain
  already-selected test geometry; all regular file-backed loaders filter.
- Approved training-spans suite38passed/2CUDA skipped32.75s; real corpus946train
  sources→1624components,50test→90components. All5scrolls retain data. No GPU
  kernels changed; CPU validation exercises geometry/targets/sampling.
- Broader sampling regressions initially315passed,2failed,11skipped,2xfail:
  eligible fixture needed spline provenance; old packed-loss mock lacked the
  contrastive sum/count (unrelated existing stale test). Updated fixtures only,
  no production loss change. Rerun317passed/11skipped/2xfail26.06s at
  /ephemeral/hendrik/fiber3d_train2/cooperative-tests/20260914_105208_737537.
  Focused results:20260914_104925_629191 under the same output root.
- Independent implementation review passed. Duplicate ambiguous CP coordinates
  fail explicitly through existing exact locator/order check; corpus passes.
