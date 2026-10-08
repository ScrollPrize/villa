# Task: whole-fiber normal sign resolution with PyTorch

Two logits per original, unsplit fiber: sign probability and broken probability.
Only perpendicular event normal agreement is used; ignore winding magnitude/sign.
For s=2sigmoid(u)-1, b=sigmoid(v), c=dot(baked normals), objective is
sum((1-bi)(1-bj)(abs(c)-c*si*sj)/2)/sum(abs(c))
+weak*mean((1-a*s)/2)+broken_penalty*mean(b).
Fixed denominator (not retained-edge denominator); no degrees duplicated in
broken penalty. Broken effectively suppresses all incident events. All-broken
and current+1/radial initialization baselines must be reported. Report relaxed
loss separately from hard sign/broken readout, retained edge percentage and sign
violations before/after removal. No correctness labels available; agreement is
not anatomical ground-truth accuracy or a guarantee of a global optimum.

Umbilicus found via local catalogue: exact Paris4 scan, matching L0 grid/2.4um.
Reuse C++ Umbilicus frame loader and outward radial convention, not bbox center.
Reuse normal sweep with same exact event points to preserve baked gauge; derive
arc-weighted regular-grid outward-dot average, persist only scalar priors per
fiber with source IDs/provenance. Verify regenerated signed endpoint normals.
Source DB remains read-only. Export solution arrays separately, not baked normals.

Tensor layout: contiguous fiber IDs/prior floats, endpoint index arrays and dot
array. SQLite/JSON read once and cached as numpy arrays; GPU full-batch gathers.
Use GPU manager existing gated child lifecycle through approved evaluation runner.
One GPU is sufficient; no CPU optimization fallback (CPU synthetic tests allowed).
Adam baseline plus configurable SGD, deterministic seed/init, log percent metrics
and timing. Start Adam2000steps lr0.05, prior1e-3, penalties0.01/0.1/1.0, then
assess convergence and extend if still materially improving. No invented labels.

Tests: synthetic aligned/inverted normals, broken suppression/penalty, gauge
invariance, scalar-vs-vectorized loss, gradients, finite outputs, isolated fibers,
GPU full graph validation and objective/history. Existing native suite.
Spec update: new standalone sign-solve mode; no changes to prior extraction or BP.
Docs: equations, tensor/schema fields, provenance, commands, timings/results,
task log/status/changelog. Independent review required by AGENTS workflow.
