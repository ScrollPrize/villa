# Native 3D Trace2CP Default Tuning

Make the following native 3D Trace2CP values the defaults for current tracing
experiments:

- `--beam-lookahead-steps 1`
- `--beam-width 8`
- `--smoothness-normal-weight 0.1`
- `--smoothness-tangent-weight 10.0`
- ordinary single-sample fallback `--sample-index 13`
- `--core-margin-voxels 20`

Keep bare `--fiber-json` whole-fiber mode intact; the sample-index fallback
must only apply when ordinary single-sample mode is resolved.
