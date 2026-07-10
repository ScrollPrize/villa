# Task Log: Cached Fused Augmentation Maps For Line And CP Warp

## Planning Notes

- The current transform code has direct formulas, but the implementation is
  not yet strict enough about constructing and reusing one fused map object.
- Smooth control setup must move out of per-call mapping paths.
- Line points and CP must be transformed in one batched source-to-output call.
- Shared CP line/CP mapping should be reused across strip-z offsets that share
  the same source and augmentation params.
- `planning/specs.md` was updated with these requirements.

## Validation

- Not run yet for this implementation task.
