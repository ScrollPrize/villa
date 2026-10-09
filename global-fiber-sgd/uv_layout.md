# Probabilistic global U/V layout

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
