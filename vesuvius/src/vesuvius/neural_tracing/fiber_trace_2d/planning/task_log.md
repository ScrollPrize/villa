# Native 3D Trace2CP Beam Search Log

## Notes

- Planning only so far.
- Current native 3D tracing commits greedily to one candidate per step.
- Current candidate density is `25x25 = 625` cone-disk samples, not an explicit
  one-degree angular sweep.
- Planned replacement uses explicit `5.0` degree tangent-plane angular steps
  inside the `25.0` degree cone, plus beam search over multiple step histories.

## Deviations / Deferred Items

- None so far.

## Validation

- Pending implementation.
