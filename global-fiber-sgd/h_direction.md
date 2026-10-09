# H-fiber circulation direction: third optimization pass

## Current extension: optional whole-fiber breakage

The original sign-only result below remains available as a baseline. The new
`run_h_direction_breakage_v2.json` runs joint sign/broken optimization, preserving
the fitted sign logits exactly at initialization. Earlier normal/HV labels are
immutable; only currently retained H fibers may be newly marked broken.

With broken probability q and keep k=1-q:

```
pair = sum(k_a*k_b*(abs(dot)-dot*s_a*s_b)/2) / ORIGINAL sum(abs(dot))
prior = mean(k*(1-prior_agreement*s)/2)
broken = breakage_weight * mean(q)
joint = [k*(1-p), k*p, q]    # reverse, forward, broken
loss = pair + 0.001*prior + broken + annealed_entropy(joint)
```

Hard labels use the maximum joint probability. Rejected fibers get zero saved
direction sign; zero must never be treated as a valid direction. The selected
solution includes full-source `labels`, `broken`, `newly_broken` arrays, conditional
sign probabilities, and active-H joint probabilities in both CW/CCW conventions.
All earlier rejected/V entries retain their original labels. No post-hoc pruning.
Saved `component` IDs refer to the original H graph before new rejections; they
must not be interpreted as connectivity of the retained graph.

Report conflicts only between surviving endpoints, with both original and retained
denominators, retained arc/event percentages, and new rejections. Additional audits
count isolated rejections, rejection outside baseline conflicts, and whether any
rejected fiber could individually be restored without a direction conflict.
`all_original_h_sign_diagnostics` explicitly includes rejected H fibers and is
not a retained-only consistency metric.

Initial v1 softened warm signs to +/-4; this introduced artificial soft pair error
on highly connected consistent fibers and excessive rejection at low penalties.
It is diagnostic only, not recommended. v2 preserves saved confidence, leaves
signs trainable, and sweeps .1/.07/.05/.03. See
[breakage results](h_direction_breakage_results.md).

Run (approved runner, one managed shared 4GiB GPU grant for the sweep):

```
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_h_direction_breakage_v2.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_h_direction_breakage_review.json
```

Use new output directories for repeat runs; completed outputs are protected.
The review command is CPU-only and reuses established normal/curve visualization:

- `disabled_h.obj`: ALL newly disabled H curves, red, full original geometry.
- `disabled_h.csv`: IDs, lengths, baseline incident conflicts and broken probabilities.
- `h_fibers.obj`:1000surviving H curves, spatially distributed from the longest
  10000eligible fibers; blue, not a random statistical sample.
- `h_normals.ply`:outward display normal arrows for that same surviving subset.
- `surviving_h_directions.obj`:same1000curves plus CCW TANGENT arrows. These arrows
  show traversal direction, not surface normals. Native L0 XYZ, length256/radius8,
  sampled every1024basevoxels; sign applies to the entire original fiber.
- `umbilicus.ply`:thick orange umbilicus. Keep `fibers.mtl` next to OBJ files.
- `h_direction_index.json`:source IDs, signs, arrow positions and hashes.

The reused renderer also writes empty legacy `broken_fibers.obj` and
`v_fibers.obj` layers because its input selection contains only surviving H.
Use **disabled_h.obj**, not that empty legacy broken layer, for the184new rejections.

Rejection means inconsistent evidence under this model, not proof of anatomical
breakage. Global U/V remains paused and has not consumed these results.

## Original sign-only baseline

Pipeline order: normal-sign alignment → baked H/V classification → **H tangent
sign alignment** → future global winding/U/V assignment. The global assignment
task is paused pending additional user constraints; old U/V trials did not use
this stage and have not been rerun.

## Model

One scalar sign logit per whole retained H fiber; no fiber cutting or new rejection
state. Earlier normal/HV rejections remain excluded, and V fibers are not assigned
a tangent sign here. Original AFV point order defines the local tangent gauge.
A positive saved sign follows that order; negative reverses the complete curve.
No per-point flips: the signed tangent can rotate through bends/folds while the
whole-fiber traversal stays consistent.

At each retained H–H parallel constraint, sample a signed tangent using the shared
`tangents_at` clipped centered secant with16basevoxel half-span. Raw secants come
from original polylines, with no repair or resampling of stored geometry. Zero
or invalid secants are counted separately and do not become invented directions.
The current dataset has none. Preserve event IDs. Let c_e=t_a·t_b.

For sign probability p_i=sigmoid(logit_i), s_i=2p_i-1:

```
pair = sum_e (abs(c_e) - c_e*s_a*s_b)/2 / sum_e abs(c_e)
prior = mean_i (1 - a_i*s_i)/2
loss = pair + 0.001*prior + late_binary_entropy
```

This pair/prior implementation is shared with the earlier normal-sign optimizer;
its rejection gates are absent for this stage. Zero pair weight produces zero
pair cost, leaving the weak prior and explicitly reported gauge uncertainty.
Each parallel event contributes once with absolute-dot confidence; evidence is
not deduplicated into unique neighbor pairs.

## Circulation prior and convention

Sample each original segment at midpoint subintervals <=16basevoxels and weight
by represented arc length. At each sample use:

```
positive_circulation = normalize(local_umbilicus_axis × outward_radial_vector)
a_i = arc-weighted mean(dot(stored_forward_tangent, positive_circulation))
```

The local axis points toward increasing native Z, using catalogue-center secants
at z±8. Positive is right-handed circulation: counter-clockwise viewed from the
positive-axis end looking toward the negative end. Catalogue frame/voxel size
is checked against AFV metadata; no additional level scaling is applied.
Axis-center XY coordinates clamp outside the annotated Z range, consistently
with earlier priors. Zero-radius/invalid circulation samples contribute neither
numerator nor denominator; completely unsupported fiber gets a neutral0prior.

The CCW solution is canonical. CW is its **exact global negation**: signs/logits
negated, probabilities complemented. Both have identical pair agreement, so no
second optimization is needed. Later U/V experiments need only choose one global
circulation sign. They must also respect disconnected components: independent
component gauges are selected by weak geometric evidence, not guaranteed by
relational constraints.

## Run and verification

10000full-batch Adam steps, lr.05, prior.001, binary entropy0until50%, linearly
ramped to.001by80%, then fixed. Seed0, deterministic Torch; CUDA required for
production, shared GPU-manager launcher and4GiB grant. No installations.

```
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_h_direction.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_h_direction_report.json
```

Completed solve outputs refuse overwrite; use a new output_root for repeats.
Cached preparation includes source hashes, sampling settings and convention.
Results: [h_direction_results.md](h_direction_results.md).

Current output:
`data/h_direction_v1/solve/solution.npz`

- `ids`: all original source IDs, unchanged order.
- `active`: indices of eligible retained H fibers.
- `direction_sign_ccw`, `direction_sign_cw`: full source-sized int8 arrays;
  +/-1for H,0for non-H/excluded fibers.0is NOT a usable direction assignment.
- `sign_probability_*`, `sign_logit_*`, `component`: active-H-sized arrays;
  use `active` for mapping to original IDs.
- `convention`: explicit stored-point-order/circulation meaning.

`summary.json` records inputs/settings/baselines/final metrics; `history.json`
records progress; `audit.json` checks current input hashes and saved output.

Tests:

```
/home/hendrik/.venv_las/bin/python -m pytest -q --noconftest -o pythonpath=/home/hendrik/villa2/vesuvius/src vesuvius/tests/test_fiber_h_direction.py vesuvius/tests/test_fiber_normal_sign.py vesuvius/tests/test_fiber_hv_stage.py vesuvius/tests/test_fiber_normal_sign_visualization.py
```

22tests pass. Includes bit-identical old scalar tangent and normal-loss formulas,
reversal covariance, exact circulation-gauge symmetry, neutral/degenerate cases,
and a synthetic connected graph correcting a misleading local prior. Independent
audit recomputed all3603373pair outcomes, IDs/exclusions/component memberships,
CW symmetry and128raw AFV events(256tangents, max difference2.98e-8).

## Limitations

This is model consistency, not ground-truth orientation accuracy. Raw local
tangent agreement does not explicitly parallel-transport a sheet frame between
strong folds. It can therefore contain contradictory observations;1721remain.
No additional fibers or constraints are silently rejected for that disagreement.
Disconnected components and isolated fibers rely on the weak umbilicus prior.
Late entropy produces decisive probabilities, not calibrated confidence.
Normal signs, H/V labels and geometry remain unchanged. V-direction alignment and
U/V consumption of this stage are intentionally not implemented while global
assignment is paused.
