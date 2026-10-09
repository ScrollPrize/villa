# Global U/V layout first experiment

## Model and implementation

- Frozen normal/HV outputs with exact source fingerprint checks; no re-extraction.
- Reuse AFV raw geometry reader, source validation and managed CUDA launcher.
- Cut at midpoints between perpendicular event arcs, requiring resulting pieces
  >=4166.6667basevoxels and >=3events. If impossible keep original. No observation
  revalidation/removal due to splitting.
- 100 winding hypotheses plus rejection per piece; fractional U-anchor and V-anchor
  for each. Positive widths W; B=cumsum(W); U=B[k]+fraction*W[k]. Constraint bases
  add signed arc along H:U or V:V. Piecewise-linear inverse U->q for seam crossings;
  discrete bucket indices detached, interpolation differentiated.
- H/V offsets0/+0.5 for outward layer order. Inward endpoint normals negate
  connector dot to give outward signs. Disagreement or near-tangency: unknown sign,
  retain unsigned soft evidence and report separately.
- Soft distance: normalized Gaussian hypotheses over half-winding grid >=0,
  configurable sigma(first .5/1), truncated at zero. Kernel mixture compatibility
  per assignment pair; high finite hinge loss for sign violations. Not a strict
  hard guarantee. Weight pair losses by joint posterior, not posterior means.
- Weak angle fraction prior at starts/ends/events with unwrapped angles; weak
  cylindrical tangential connector metric (after removing whole winding shift),
  positive width size/smoothness priors, weak V=z initialization/prior. Fixed signed
  arc map first; local shear/stretch is not introduced in this experiment.
- High-weight endpoint continuation across cuts, including winding seam crossings.
  Degree-weighted rejection and late entropy. CUDA Adam, all100states retained,
  batches/tiled exact pair losses avoiding global E*K*K allocation.
- Small deterministic induced-graph pilot then full graph run with diagnostics,
  provenance, saved state and independent full hard audit. No GT accuracy claim.

## Validation

### Independent review resolved

Use E_p_i,p_j[-log(sum_d w_d exp(-(D-d)^2/(2*tau^2)))], not
-log(E_p_i,p_j[...]). w_d is a normalized discrete Gaussian on half-grid d>=0,
support through max(100, ceil(mu+8sigma)); tau=.25. Compute its log kernel with
stable logsumexp and interpolate a precomputed cost table at .125winding steps.
Known sign uses D=sign*delta; soft magnitude uses abs(delta), with separate sign
penalty. delta=q_b-q_a-wrapped(theta_b-theta_a)/(2pi)+.5*(HV_b-HV_a).
The angle subtraction removes tangential displacement; family offset appears
exactly once. All relation types use half-grid support as an uncertainty model,
not category-rounding. Parallel uses median parallel_winding_distance.
Normals at BOTH endpoints project the SAME A->B connector. Unknown signs retained.
H positive arc follows circumferential direction, V positive arc axial direction;
per-segment signed arc integrates dot sign, so reversing stored polylines preserves
relative maps. Zero tangent projection contributes zero; report as geometric
approximation in folds. U origin B0=0, V weakly anchored to native z. Endpoint q
outside [0,100] linearly extrapolates end widths, with explicit boundary penalty.
Both continuation endpoints derived from their anchors; full posterior pairwise
distance moments are exact for quadratic continuation and cost normalized per link.
Sparse embedding gradients plus SparseAdam for touched piece rows; regularize a
uniform sampled node batch. Widths use Adam. Pilot then explicit full-run budget.

Synthetic/unit tests: eligible cuts, seam mapping, frozen exclusions, H/V sign
offsets, gradients, pair objective vs brute force, truncated Gaussian, continuation
and inconsistent graph. Pilot measures batch speed/memory before global run.
Report baseline/final ordering, retained events/arc, distance fit, continuation,
ambiguity, occupancy, sampled epochs and wall time.

## Spec update

New opt-in stage; earlier algorithms/artifacts unchanged. Explicit fixed-arc map,
finite high sign cost, weak angle/metric priors and upper bound100windings.

## Docs updates / changelog

Add uv_layout.md/results/configs with math/commands/results/limitations;
update task_log/status/current task and durable changelog.
