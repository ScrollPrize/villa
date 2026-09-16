# Single-anchor weighted fit experiment

This is an opt-in native C++ fitter inside the existing batched anchor extraction,
not another sampler or Python implementation. Legacy extraction is unchanged.

For positions x, presence p and unoriented directions d in one owned cell:

1. Initialize a at the cell center c; initialize u from the principal eigenvector
   of sum Gcell(x-c)*p^gamma*d*d^T.
2. Form weights Gcell(x-c)*Glocal(x-a)*p^gamma*abs(d dot u)^q.
3. Update u from the principal weighted tensor; move a halfway toward the weighted
   centroid. Repeat the configured number of iterations.

Fixed Gaussian sigma2 and moving sigma1.5 are in prediction voxels (16 and12base
voxels for current level3). Damping0.5; widths/covariance are not adapted.
The fixed Gaussian never follows a. All positions remain inside the owned cell
by convex updates; no fitting halo or cross-cell NMS is used.

Final support is sum Gcell*p*cos² / sum Gcell over all owned voxels (invalid and
below-floor samples have zero numerator). Existing observation floor0.05 and
minimum support0.05 remain. This support does not evaluate local presence at a.
Empty/unsupported cells emit nothing. Degenerate initialization uses the strongest
observed direction with stable ties; later zero mass/tied tensors retain state.
Directions are sign-invariant, canonicalized through existing helpers.

Cache identity includes experimental mode/schema and every weighted-fit setting.
Inference caches and legacy construction fingerprints remain unchanged. The mode
also works through the ordinary fiberlet backend, although this first sweep is
anchor-only and does not establish downstream graph quality.

## Matrix

Iterations0,2,4,8; presence exponent1,2; directional exponent0,2,4.
For zero iterations q is irrelevant: two initialization rows, not six duplicates.
Twenty experimental rows plus rerun legacy baseline, same short200/model/ellipse.
No selected/default replacement before inspecting results.

The comparison changes fitter, one-anchor multiplicity, owned-only support and
NMS together. Compare placement gaps/empty bins and fitted-axis GT angular error,
not anchor count alone. Angular statistics are conditional on accepted positions.
Also report the fraction of arc bins without any accepted anchor whose fitted axis
is within10/20/30degrees of the GT tangent. This provides a direction-qualified
coverage measure rather than relying only on anchor-weighted angle averages.

```bash
bash /home/hendrik/villa2/vesuvius/scripts/run_fiber3d_native_build
bash /home/hendrik/villa2/vesuvius/scripts/run_fiber3d_native_replay_tests
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py benchmark-full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_short200_anchor_weighted_fit_20260916_v1.json
```

Output: /ephemeral/hendrik/fiber3d_train2/evaluations/short200_anchor_weighted_fit_20260916_v1.
matrix.md/json are updated after each completed variant. Each variant contains
normal anchor benchmark outputs/provenance/ranking. Completed variants can be
reused only with identical sweep/workload/binary provenance. An interrupted
incomplete variant is retained, not silently overwritten.

GPU management remains demand-only; release after30idle seconds. CPU extraction
uses224logical-worker budget, one physical scroll at a time. Wall times include
cache/setup effects and are not controlled speed comparisons.
