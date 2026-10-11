# Fresh global U/V solve with 151 winding choices

Status: both fresh joint stages running. This supersedes the unsuccessful
warm expansion. No prior global U/V solution is used to initialize these runs.

## Initialization and stages

Upstream normal signs, H/V and tangent directions remain fixed. The same
6,606,100 constraints connect 489,663 pieces. Physical tensors are reused;
cached 100-winding initial guesses are not used for the fresh 151 initialization.
Radius quantiles set an initial winding center spanning roughly 7.6–136.4, with
Gaussian sigma 3 winding units and finite support over every index 0–150.
Widths are initialized from radii across that range. Shared and per-candidate
offsets start at zero; broken probability starts at 1%. No reference evidence.

Both directions independently run the original sequence at the new count:

1. Joint Adam: 10,000 steps, batch 4096, learning rate 0.03. Geometry, widths,
   assignment and rejection variables are trainable; no U/V checkpoint loaded.
2. Sign-charge16 refinement: 4,000 steps, same batch/rate, starting only from
   that direction's newly generated joint result. Geometry remains trainable.
3. Categorical assignment refinement: ten full-graph sweeps, charge4, step1,
   temperature0.25, batch2048. Geometry is fixed only in this final stage.

Thus ten requested final sweeps require the preparatory joint/sign stages;
this is not ten Adam minibatches. Existing loss settings are retained.
GPU manager: separate GPUs0/1, shared48GiB reservations for joint/sign stages,
then12GiB for categorical. Actual allocation is monitored. Old outputs preserved.

## Commands

Use the existing approved runner for each stage (in order within each direction):

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_uv_fresh151_ccw_joint.json
```

Other configs replace `ccw_joint` with `cw_joint`, `ccw_sign`, `cw_sign`,
`ccw_categorical`, or `cw_categorical`.

Outputs: `data/uv_fresh151_v1/{ccw,cw}_{joint,sign,categorical}/`.
Configuration and checkpoint SHA pins establish within-run stage provenance.
The previous 151 warm expansion remains archived separately, not promoted.

## Validation

40 regression tests passed: historical 100-state byte identity;151 fresh dimensions,
finite support/1% broken/zero offsets; geometry and width optimizer updates;
existing pair costs, expansion, projection and visualization tests. Independent
plan and code reviews approved. Initial production state confirms zero offsets,
1% broken probability, no rejected pieces and occupied indices4–146.
