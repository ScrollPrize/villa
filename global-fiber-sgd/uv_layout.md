# Probabilistic global U/V layout

## Cost-target assignment experiment (2026-10-09)

`assignment_update=cost_target` selects `fiber_uv_categorical.py`. It reuses the
shared coordinates, pair costs, provenance, manager and strict readout. Requires
a shared-anchor warm start; **continuous offsets/phases/widths are frozen**.
This is assignment/rejection repair, not joint geometry optimization.

For each candidate winding, aggregate costs of ALL incident constraints against
each neighbor's unnormalized winding distribution. Broken neighbors contribute
zero. Multiply incident sums by N/E; broken's cost is R*(.5+.5*degree/mean_degree).
Fixed sign charges distinguish each individual winding pair, alongside the
original magnitude and soft geometric costs. No selected-pair blanket charge.
Continuations, per-event angle and node geometric priors are included.

Targets are detached `softmax(-conditional_cost/temperature)`. Direct per-row
cross-entropy SGD updates logits by `-step*(p-target)`, avoiding both the tiny
current-probability factor and accidental1/N shrink. All targets use the same
previous state. Small updates reduce synchronous reject/restore oscillation.
The temperature replaces old positive entropy sharpening; expected soft energy
replaces mixture likelihood. Thus this is not an optimizer-only ablation.

Run with the existing command, substituting the desired config:

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_uv_categorical_repair.json
```

Other configs: `run_uv_categorical_damped.json` (earlier diffuse initialization),
`run_uv_categorical_pilot_t0.25.json` / `_t1.json` (aggressive oscillating controls),
`run_uv_categorical_ccw.json` / `_cw.json` (full repair),
`run_uv_categorical_pilot_report.json` (read-only evidence audit plus report files).
Completed output roots refuse overwrite; change output_root for repetition.
Future full configs request shared12GiB per job on separate GPUs (CCW0, CW1),
released on completion. The initial full trials retain their frozen24GiB grants
on GPU0; changing launch configs does not alter running jobs. Batch4096 unchanged.
Pilot configs retain their historical reservations. No dependencies
or builds added. Reproducibility uses fixed full-graph order and deterministic
Torch operations; CPU audits test strict negative signs and q_broken>0.5.

The first repair pilot reaches0violations and71.9722%arc versus227and69.4125%.
This does not solve the existing continuation failures (0/5links survive) or
establish anatomical correctness. Probabilities remain uncalibrated; zero selected
sign violations does not imply a converged posterior.

## Current shared-anchor revision (2026-10-09)

Opt in with `layout_mode=shared_anchor`; the legacy formulation below remains
available for old results/configurations. Use run_uv_anchor_ccw.json and
run_uv_anchor_cw.json through the same approved crop-benchmark runner.

The accepted normal, H/V, H-direction and V-direction solutions are hash-bound.
All previous rejections are excluded (23,501 original fibers). Final V labels
must preserve H outputs and its embedded H sign arrays must match exactly.
Whole-fiber direction sign times original arc length replaces local geometric
sign guesses, even through folds. H CCW/CW alternatives flip H travel and angular
phase together; V remains UP and outward normal ordering does not flip.

Each piece now has303float parameters:101winding/reject logits, shared phase and
V adjustments,100U offsets and100V offsets. The anchor is the first point of the
piece. Effective offsets are arithmetic-mean centered across buckets (not
probability-weighted), removing shared-anchor/offset redundancy. With continuous
cumulative-width interpolation U(q):

```
phase_i = observed_first_phase_i + shared_phase_adjustment_i
q_ik = k + phase_i + centered_u_offset_ik
u_ik = U(q_ik)
v_ik = observed_first_z_i + 1000*(shared_v_adjustment_i + centered_v_offset_ik)
```

Shared phase is unbounded and never wrapped/clipped. U extrapolates first/last
width outside the finite100bucket range. U offsets are turn units, not raw voxel
distances; within a width interval their physical effect is width*offset. Weak
quadratic centering (.001) applies to U/V offsets; a strong squared hinge
(10000) applies to |Uoffset|>.25. The numeric threshold .25turns means90degrees,
not45degrees. This is a soft penalty, not a hard no-overlap guarantee. Report
offset quantiles/exceedance and minimum adjacent-bucket separation explicitly.
Shared phase can change integer turns; bucket IDs are relative hypotheses, not
absolute winding numbers. Report actual derived anchor q as well.

Constraint coordinates derive solely from these anchors and signed arc offsets.
For conditional retained probabilities pi, compatibility cost C:

```
L_match = keep_i*keep_j * -T*logsumexp(log(pi_ik)+log(pi_jl)-C_kl/T)
```

All100x100pairs participate; T=1. C includes the existing broad winding-distance
mixture and weak metric terms. Signed-order hinge remains a separate posterior
expectation so a small compatible mode cannot hide wrong-sign mass. Rejection
gates events but has an explicit degree-mixed cost. Late entropy sharpening and
width/angle/axial priors remain. Expected-cost mode is available as a control via
`pair_mode=expectation`; only the ordinary event aggregation changes in that
control, not the new coordinate model or continuation likelihood.

Continuation uses the same all-pair likelihood for derived endpoint U/V squared
distance (128base-vx scale), with link-sum/event-count normalization. This avoids
making each of a few continuation links thousands of times more influential
than an ordinary observation simply because there are few links. Its configurable
weight is10. The initial pilot retained historical per-link normalization and
weaker offset penalty; it is archived separately, not the final protocol.

Validation:49tests across layout and previous stages; synthetic seam/gradient,
likelihood reinforcement, rejection gating, frozen provenance, direction reversal,
whole-fiber folds and CPU audit equivalence. Production uses CUDA only, existing
GPU manager and runner, no new dependencies. Runs preserve original geometry,
event IDs and non-overlapping piece partitions. Results are model consistency,
not ground-truth winding accuracy.

### Exact shared-anchor commands

```
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_uv_anchor_ccw.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_uv_anchor_cw.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_uv_anchor_full_report.json
```

Completed output roots are protected. To repeat optimization, copy a config with
a fresh output_root; its verified prepared tensor cache can be reused. Reports
can be rerun without optimization. Pilot configs use the same runner, named
run_uv_anchor_pilot_ccw.json and run_uv_anchor_pilot_expected.json.

## Legacy formulation and historical runs

Experimental opt-in stage following accepted normal-sign and H/V stages. Original
AFV geometry, constraint DB and earlier solutions are immutable. This is a
constraint-consistency experiment, not a validated flattened scroll.

## Inputs and splitting

`run_uv_global.json` binds the original AFV and catalogue umbilicus, schema2
oriented constraint database, inward normal solution and accepted H/V solution
by SHA256/ID/geometry checks. Exclude all23240earlier rejected fibers. The full
input has489458fibers and6618825events after those exclusions. Contradictory
remaining H/V relations are not silently filtered; the new rejection state can
remove additional pieces.

Cuts occur only midway between consecutive perpendicular event base arcs. Greedy
earliest eligible cuts leave every resulting piece >=10mm (4166.6667basevoxels)
and >=3perpendicular events. Short original fibers remain intact. No endpoint
support test or constraint extraction is rerun after cutting.454original fibers
split into920pieces; overall489924pieces,466continuations. Ownership uses the
original event base arc and a deterministic right-side tie convention.

## Coordinate model

Each piece has301trainable floats:101logits (100windings+rejection),100fraction
adjustments,100Vadjustments. Constraint points have NO trainable variables.

W_k=100+softplus(raw_k); B_0=0; B_k=sum_{j<k}W_j.
For piece i and winding k:

```
f_ik = observed_midpoint_angle_fraction + 0.25*tanh(fraction_parameter_ik)
u_ik = B_k + W_k*f_ik
v_ik = midpoint_z + 1000*v_parameter_ik
H endpoint: (u_ik + signed_arc_offset, v_ik)
V endpoint: (u_ik, v_ik + signed_arc_offset)
```

Positive H arc follows the circumferential tangent around the axis; positive V
arc follows +z. Integrate signed physical segment lengths; a reversed stored
polyline gives the same relative map. These fixed maps approximate a flattening;
they do NOT learn local shear/stretch or a transported sheet-tangent frame through
folds. The ±quarter-turn fraction bound is an explicit first-model restriction.

Invert piecewise-linear U(q) using detached bucket selection and differentiable
within-bucket interpolation; continuous across seams. Extrapolate first/last
width outside[0,100], with a boundary penalty. No expensive iterative integral or
binary search over all hypotheses is required for their anchors; endpoint seam
crossings use `searchsorted`. Winding widths weakly regularized to circumferences
from geometric initialization; no physical monotonic-width constraint.

## Constraint model

For original A->B connector, project onto BOTH inward endpoint normals and negate
to obtain outward sign. Both must agree and have |cosine|>=.1. Otherwise retain
the event as unsigned.90.878%of full events have a known sign. No global normal
volume alignment or re-inference is performed.

For every pair of winding hypotheses:

```
delta = q_b - q_a - wrapped_angle_difference_ab + 0.5*(family_b-family_a)
family: H=0, V=1
```

Subtract geometric angular displacement so it is not mistaken for a layer step.
The half-layer offset appears once: V backing shares the chart of its inward H
layer but follows it in outward ordering, per requested anatomical convention.
It is not an inferred physical thickness. Strong folds/kolleses can violate this
simple local interpretation.

Measured mu is `parallel_winding_distance` for parallel observations, otherwise
central `winding_distance`. Do not round it. For half-grid hypotheses d>=0:

```
w_d = normalized exp(-0.5*((d-mu)/sigma)^2)
C(x) = -log(sum_d w_d * exp(-0.5*((x-d)/0.25)^2))
```

All half-grid possibilities through max(104,mu+8sigma) participate. C is tabulated
at1/16winding spacing in mu and x and bilinearly interpolated. Unit tests compare
to direct logsumexp (tested absolute error<.006). This is a numerical table
approximation, not exact arithmetic. Beyond its x extent it extrapolates linearly;
boundary diagnostics disclose out-of-range coordinates.

Known sign: x=max(sign*delta,0), plus `4*sign_weight*relu(-sign*delta)`.
Unknown sign: x=abs(delta), no ordering term. This is a **finite boundary-clamped
surrogate** for a sign-truncated likelihood. Wrong-sign states are not reflected
into high-likelihood valid distances, but truly hard support would require infinite
loss. Report remaining sign violations, never imply a hard guarantee.

Pair cost is E_{p_i,p_j}[C+ordering+weak_metric], not -log(E[likelihood]) and not
loss on mean winding. Every100x100assignment pair participates in each sampled
event. Rejection states omit attached pair costs but pay degree-weighted cost.
Weak circular phase priors apply to starts/midpoints/ends AND event points. Weak
V=z and local cylindrical connector priors stabilize the physical chart.

Continuations compare derived U/V endpoints with full posterior quadratic
expectation using centered moments; different winding indices can connect at a
seam. Centering avoids catastrophic cancellation at large U. Current experiment
normalizes this term per continuation link, separately from ordinary events;
this is an explicit weighting choice and may encourage rejection of split pieces.

## Optimization and limitations

CUDA only for production. Sparse embedding lookups/SparseAdam update touched
piece rows; width parameters use Adam. Sample4096events and4096uniform pieces per
step; all100hypotheses retained. Fixed seed0 and deterministic Torch algorithms.
Initial winding probabilities are radial-position Gaussians(sigma3) with logit
floor-14 to keep all100states numerically reachable. Rejection starts at1%.
First half freezes rejection-logit gradients (not its normalized probability).
Late entropy ramps from50%to80%of steps.10ksteps =6.1884sampled event passes,
not a guarantee every event was optimized or that the solve converged.

No prior/probability is anatomical ground truth. Radial initialization, weak
angle/axis geometry, factorized posteriors and local optima can affect results.
H/V is fixed and never relabeled. Disconnected components retain relative-gauge
ambiguities. First run is not promoted over earlier successful stages.

## Commands and outputs

Existing approved runner, no new command approval or dependencies:

```
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_uv_pilot.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_uv_global.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_uv_global_soft_distance.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_uv_report.json
```

Completed output directories are protected against overwrite. Shared GPU manager
requests24GiB on one GPU for full solve, releases on completion. Preparation is
CPU-only and cached. `solution.npz` stores piece identities/arcs/families,301state
parameters and100widths; `config.json` code/config provenance; `history.json`
fixed diagnostic sample and full final metrics. Full baseline is in summary.
`audit.json` recomputes full metrics and checks original event preservation,
gap-free partitions, minimum lengths/counts, frozen exclusions and source hashes.

Regression tests:

```
/home/hendrik/.venv_las/bin/python -m pytest -q --noconftest -o pythonpath=/home/hendrik/villa2/vesuvius/src vesuvius/tests/test_fiber_uv_layout.py vesuvius/tests/test_fiber_hv_stage.py vesuvius/tests/test_fiber_normal_sign_visualization.py
```

Full variants: sigma.5/sign10 and sigma1/sign100. This compares two intended loss
balances, NOT an isolated ablation. Both use identical data, seed,10ksteps,4096batch,
rejection10 and continuation10. Partial overlap on one shared managed GPU means
wall times are workload measurements, not isolated performance comparisons.

Latest results are in uv_layout_results.md once the full run/audit completes.
