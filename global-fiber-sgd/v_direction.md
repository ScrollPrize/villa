# V-fiber consistent up direction

H state was committed as`3f5b443e8`, including
[h_direction_recommended.json](h_direction_recommended.json). Its selected solution
remains frozen; large data artifacts live under ignored`data/`.

## Model and conventions

The shared`fiber_direction.py`implementation now serves both H and V (renamed from
the former H-only module; approved H runner/config entrypoints still work).
Only the family selector and geometric prior differ:

- H: parallel H–H tangents, weak circulation prior around the umbilicus.
- V: parallel V–V tangents, weak local **positive-Z umbilicus tangent** prior.

The prior is the arc-weighted mean signed dot product between each stored-forward
fiber tangent and the local umbilicus tangent. It uses the same16basevoxel sampling
and z±8axis secants as H. On-axis samples remain valid for V. No normals are
flipped here, no H/V classification is reoptimized, no H–V tangent equality is
introduced, and whole original AFV fibers remain intact.

One sign applies to the entire V fiber. Positive means follow stored point order;
negative means traverse it backward. Canonical UP follows the local increasing-Z
umbilicus direction. DOWN is the exact global negation. This is NOT a hard
pointwise monotonicity constraint: folds may locally turn down; unsupported internal
hairpins cannot be guaranteed detected. Disconnected/isolated components depend
on the weak geometric prior.

## Solver

First run the sign-only full-batch CUDA Adam solve, then warm-start its exact logits
for sign/broken optimization. Same H settings:10000steps, lr.05, prior.001,
late joint entropy.001, fixed original event-weight denominator, keep-gated prior,
uniform mean broken penalty. Sweep .1/.07/.05/.03. Prefer zero retained conflicts
then maximum retained arc, no post-hoc pruning. See[v_direction_results.md](v_direction_results.md).

The accepted H solution is hash-bound during preparation. Its labels,184newH
rejections and CCW/CW sign arrays remain unchanged. New rejection acts only on V.
Rejecting a fiber means its direction evidence is inconsistent under this model,
not proof of a physical papyrus break.

## Output schema

Each full-source solution saves:

- `ids`, `labels`(0H,1V,2broken), `broken`: includes all earlier/H/V exclusions.
- `active`: original pre-V-rejection V indices; do not assume all remain enabled.
- `newly_broken`: ONLY new V rejections, separate from inherited rejections.
- `direction_sign_up/down`: +/-1on surviving V,0elsewhere. Exact opposites.
- `direction_sign_positive/negative`: aliases of up/down.
- `inherited_h_direction_sign_ccw/cw`: unchanged frozen H output, independent
  of reversing V. Never reverse these when choosing V down.
- `sign_probability_up/down`, `sign_logit_up/down`: conditional on V being kept.
- `joint_probability_up/down`: reverse/forward/broken probabilities, active-V rows.
- `component`: ORIGINAL V graph before rejection, not retained-graph components.

Summary/audits report remaining conflicts, original and retained denominators,
retained arc/events, isolated/unrelated rejections and individual restorability.
`all_original_v_sign_diagnostics`includes newly rejected V and is not a retained
subset metric. Baseline outputs remain immutable under`solve/sign_only/`.

## Run and visualize

```
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_v_direction.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_v_direction_review.json
```

One shared4GiB GPU-manager allocation covers baseline and all penalties. Preparation
and review/visualization are CPU-only. Completed outputs are protected; change
output paths for fresh runs. Use the same established curve/normal renderer:

- `disabled_v.obj` and CSV: all NEW rejected V fibers, red, full curves.
- `v_fibers.obj`:1000long spatially distributed surviving V fibers, green.
- `v_normals.ply`:same subset, outward display normals (stored normals inward).
- `surviving_v_directions.obj`:same curves plus UP tangent arrows, length256,
  radius8, every1024basevoxels. These are not sheet-normal arrows.
- `umbilicus.ply`, `fibers.mtl`, `v_direction_index.json`: axis and provenance.

The standard renderer also creates empty H/broken layers for this V-only selection;
the actual newly disabled layer is`disabled_v.obj`, not`broken_fibers.obj`.

## Validation

```
/home/hendrik/.venv_las/bin/python -m pytest -q --noconftest -o pythonpath=/home/hendrik/villa2/vesuvius/src vesuvius/tests/test_fiber_v_direction.py vesuvius/tests/test_fiber_h_direction.py vesuvius/tests/test_fiber_normal_sign.py vesuvius/tests/test_fiber_hv_stage.py vesuvius/tests/test_fiber_normal_sign_visualization.py
```

30tests cover original H losses/geometry, V axial/reversal/on-axis/tilt behavior,
immutable inheritance, breaks/gauges/denominators and visualization primitives.
Global U/V assignment remains paused; neither direction result is consumed there yet.
