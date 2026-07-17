# Native 3D Trace2CP Default Tuning Plan

## Implementation

- Update `NativeTrace2CpConfig` defaults for beam lookahead, split smoothness
  weights, and core margin.
- Update native 3D Trace2CP CLI defaults to match.
- Keep parser `sample_index=None` so bare `--fiber-json` still enters
  whole-fiber mode, but resolve ordinary single-sample mode to sample index 13.
- Allow no-normal-sampler paths to fall back to isotropic smoothness when split
  smoothness defaults are present.

## Spec Update

- Document the current native 3D Trace2CP defaults:
  `--beam-lookahead-steps 1`, `--beam-width 8`,
  `--smoothness-normal-weight 0.1`, `--smoothness-tangent-weight 10.0`,
  ordinary sample-index fallback 13, and `--core-margin-voxels 20`.
- State that bare `--fiber-json` remains whole-fiber mode.

## Docs Updates

- Update `docs/code_structure.md` native 3D Trace2CP command examples and
  smoothness/beam default descriptions.
- Add a changelog entry for the default tuning.

## Tests

- Update native 3D Trace2CP default assertions.
- Add a CLI parser default test.
- Run the focused 3D fiber trace test file.
