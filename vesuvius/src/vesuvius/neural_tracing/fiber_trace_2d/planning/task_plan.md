# BatchNorm Direction Model Plan

## Implementation

- Replace GroupNorm layers in `FiberStripDirectionNet` and `_ResidualBlock`
  with `nn.BatchNorm2d`.
- Remove the GroupNorm group-count helper because the model should not carry
  an unused GroupNorm fallback.
- Keep the public model config unchanged: `in_channels`, `hidden_channels`,
  and `depth`.

## Spec Update

- Update `planning/specs.md` so the default V0 direction model is documented
  as BatchNorm2d-based.

## Docs Updates

- Update `docs/code_structure.md` model documentation.
- Mark the normalization todo item complete.
- Replace `planning/task_log.md` with current-task notes only.
- Add one changelog line for the architecture change.

## Testing

- Run Python compile validation for the touched model and tests.
- Run the focused 2D fiber-trace loader tests.
