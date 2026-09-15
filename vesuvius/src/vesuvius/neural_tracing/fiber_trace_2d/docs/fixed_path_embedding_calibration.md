# Fixed-path EMA scoring calibration

This is an opt-in diagnostic experiment, not a production-default change. It
freezes the inference models and path coordinates, then fits only an EMA-only
embedding scorer. Presence/direction come from EMA999 u11800; embeddings come
from paired current EMA999 u42000. The preceding baseline scored111 restarts.

## Data and objective

`embedding_fixed_path.py` reads completed replay artifacts and locks a split by
annotation-content hash before mining windows. Forward/reverse and eligible
spans remain together. The current50fibers split30fit/9selection/11heldout.
This is calibration-heldout data, not a pristine test: these fibers informed
earlier manual tuning. The existing external training/test split is not changed.

Failure windows use the last close-to-reference traced position preceding an
error. Both continuations share up to1024basevoxels of traced prefix. A Hermite
displacement correction connects the reference continuation continuously to the
same starting position and tangent. No direct jump onto GT is scored for free.
Reject short, tight-curvature or excessive prediction-angle routes. This does
not prove discrete cone-grid feasibility. Both alternatives have32samples at
approximately8basevoxel spacing; actual distances are recorded in the report.
Successful controls preserve the baseline relative score instead of forcing
GT to beat an already acceptable trace.

`vc_fiber_trace_metric score-fixed-paths REQUEST_JSON` reuses the native
`scoreCandidateTasks` and `beamStateFromFrontierCandidate` routines. The ordinary
prediction field/native chunk cache supplies all samples, with an explicit
readiness check. Export contains normalized embeddings, distances, base costs,
total costs and prediction angles. Missing channels reject a route. No Python
Zarr reader, new inference, or approximate Python base-cost implementation.

GPU PyTorch replays normalized EMA, scoring each sample against its parent state
before the update. Warm-up is recomputed at current half-life, then detached;
gradients run through the short branch only. Exact vector cancellation retains
the previous reference. Native parity is for this finite-prefix scorer, not the
original trace's full state. Smoothness state also starts at the finite prefix.

Fit positive bounded weight and half-life, optionally a smooth threshold and
temperature. The latter uses:

```text
weight * temperature * (
    softplus((cosine_distance - threshold) / temperature)
    - softplus(-threshold / temperature)
)
```

The subtraction makes zero distance cost zero. Temperature0 selects the exact
existing linear cost; threshold must then be0. New native flags are
`--embedding-distance-threshold` and `--embedding-distance-temperature`.
No anchor or delayed/gated state is involved.

Ranking uses the mean difference of native-base-plus-embedding costs, with
softplus margin. Controls preserve baseline score differences. Regularization
limits weight/half-life departure. Inverse-total-window-per-fiber weighting is
used, not exact equal per-class influence. Selection fibers choose a checkpoint;
heldout metrics never choose parameters. A512basevoxel-window diagnostic applies
the same fitted parameters without retraining. It independently mines windows,
so it is not a paired truncation-only experiment.

## Reusable commands

Run from `/home/hendrik/villa2`, no environment activation or installation needed.
The runner uses the installed las Python and injects this checkout's import paths.

```bash
bash /home/hendrik/villa2/vesuvius/scripts/run_fiber3d_native_build
bash /home/hendrik/villa2/vesuvius/scripts/run_fiber3d_native_replay_tests
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py fixed-path-extract --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_fixed_path_ema_fit_20260915_v1.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py fixed-path-fit --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_fixed_path_ema_fit_20260915_v1.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py fixed-path-report --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_fixed_path_ema_fit_20260915_v1.json
```

Replace `fit_20260915_v1` with `fit_long_20260915_v1` for longer-window extraction
and parameter evaluation. The extraction resumes matching per-case hashes;
`fixed-path-fit` explicitly refits/overwrites its fit report, except when
`evaluate_parameters_from` points to existing parameters. GPU work goes through
the same manager launcher. CPU-only training is not a fallback.

Real full tracing uses ordinary `prepare`, `validate-full`, `full` modes with
`eval_fixed_path_ema_linear_20260915_v1.json` and
`eval_fixed_path_ema_smooth_20260915_v1.json`. These retain the frozen role
checkpoints, source namespaces, eligibility,40/80ellipse,160excursion grace and
80reference-voxel restart grouping. Full runs acquire all8GPUs exclusively for
any new missing inference. Existing published chunks are reused; no rendering.

Artifacts are under
`/ephemeral/hendrik/fiber3d_train2/evaluations/fixed_path_ema_fit_20260915_v1/`:
`split.json`, `dataset.json`, per-case queries/exports, `fit_report.json`,
`comparison.json` and `comparison.md`. Long diagnostic and two real trace runs
have separate sibling directories. Reports clearly distinguish fixed-route
loss from actual restart metrics; lower surrogate loss alone is not success.

## Validation and portability

Native tests cover exported base/EMA costs, smooth mapping and invalid input.
Python tests cover parent-before-update, finite-difference half-life gradient,
zero-vector cancellation gradient, shared prefixes and content-grouped split.
Real fit checks native/PyTorch total-cost parity before optimization. Native
code uses portable C++/existing sampling; real fitting requires CUDA, while
synthetic unit tests may use CPU. Machine-specific configs are explicit.
