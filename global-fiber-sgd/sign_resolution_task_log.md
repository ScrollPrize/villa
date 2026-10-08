# Archived task log: whole-fiber normal sign resolution

- Prior orientation committed1a41f4321. Accepted model:sign/broken logits per
  unsplit fiber, signed endpoint dot on perpendicular events, very weak outward
  umbilicus prior. Broken suppresses all incident events; penalty prevents trivial
  all-broken solution. Plan sign_resolution_plan.md independently reviewed.
- Local cached catalogue identifies exact Paris4 scan umbilicus; downloaded via
  approved curl. Metadata states native2.4um/L0 matching32693x32693x75784 despite
  annotation filenameL2. Native shared frame loader/radial vector reused.
- Reused existing normal sweep for scalar prior export, trapezoidal arc weights
  on regular samples only. Checks all regenerated signed endpoint normals against
  schema2before export. No source database changes or repeated extraction.
- Full-batch CUDA PyTorch Adam/SGD with cached tensor arrays and manager-gated child.
  Reserve4GiB/4CPU ononeGPU; no CPU production fallback. Tests may use CPU.
- Native142tests pass. Initial pytest invocation used wrong relative import root;
  rerun with explicit workspace source path (no dependency/install changes).
- Five Python tests now pass, including cache invalidation on source replacement.
  Reviewer identified missing prior→DB content identity; fixed with before/after
  source/umbilicus SHA256 and embedded source hash. Freshv2prior export123.01s,
  all14,736,670signed endpoints exact;oldv1artifacts retained as pilot history.
- Pilot fullGPU2000steps/penalty completed~4.5s each,148.6MB peak allocated.
  Radial-only weighted mismatch1.4637%;penalty0.1retains97.1643%events with
  0.0008873%mismatch and0.54984%fibers broken. Penalty0.01over-disables; extend
  full trials to10000steps. No anatomical accuracy claim or global-optimality claim.
- Completed10000steps for0.01/0.1/1.0penalties:22.21/22.27/23.40s on shared H100,
  peak141.69MiB allocated. Penalty0.1marks2402fibers(0.4685%)broken and retains
  1,299,906events(98.4322%);137retained opposed events, weighted0.001069%.
  Penalty1breaks459fibers(0.0895%),retains99.5532%,weighted0.161454%opposition.
  Penalty0.01still breaks9.2253%fibers/discards50.2313%events;not recommended.
- NumPy independently verified all saved readouts/metrics. Tensor inputs43.13MB,
  preparation23.49s;43,776fibers isolated in perpendicular-only graph. Native v1/v2
  priors/weights/IDs exact; source DB SHA unchanged. GPU child/manager task finished.
- Deviation/limitation: no CPU production fallback; normalization is event-weighted
  and global minimum is not guaranteed. Low-penalty run still drifts/re-enables
  fibers at10000steps; do not claim convergence. No missing cases or endpoint
  exclusions in current data. SGD supported but actual full trials used Adam.
- Documentation sign_resolution.md, generated full table sign_resolution_results.md
  and JSON; run configs preserved. Source DB remains read-only; NPZ solutions
  contain source IDs, probabilities, logits, hard signs and broken mask.
