# Patch repair task log — 2026-10-14 (archived)

- User authorized reconstruction and continuation on both inputs.
- Independent review approved with corner-Jacobian/seam audits and matched guarded
  control. Frozen boundaries/UV, arc-normalized guides, original displacement
  reference, fresh Adam state. No global collision guarantee.
- Halos2/4/8, guide weights0/.1/1. Acceptance: fewer aggregate defects, no new
  defective cells/neighbor edges, local mean <=1.25*before+5base vx, global mean
  <=original+1base vx. Failed patches stay unchanged.
- Both continuations:600steps, LR1, dataweight20, smooth1.5/bend5/step.5, guard100
  relative to their own healthy starting geometry. UV starts at saved final
  positions, still regularized against original UV. Guard is soft.
- Synthetic/regression tests:7passed. Full paired experiments use approved runner
  and shared8GiB grants on GPUs0/1.
- Initial v1 interrupted after review caught inconsistent guard eligibility.
  v2 excludes near-collapse and neighboring orientation reversals; audits changed
  to mathematically equivalent affected-region checks (full-array audits took
  minutes). Both v2 runs completed in33.5/35.5s;58tests passed,1known unrelated
  test deselected. Immediate folds225→209 and233→206; continued195/200 vs
  no-repair controls198/216. No new defects from accepted repair itself; soft
  continuation guards do NOT prevent new defects, even though totals decrease.
- User requested larger patches first. Tested16/8/4/2 and area-descending order:
  immediate207/205, barely better. Review suggests fixed boundary positions alone
  do not preserve incoming tangent, causing most candidates to fail seam checks.
- Testing seam-Laplacian-change penalty10 and full/.5/.25 reconstruction steps.
  Guard now also explicitly excludes opposing diagonal triangle normals, beyond
  corner-vs-average tests. Matched controls re-run with same guard. Current
  defaults unchanged. Synthetic restoration and failed-fit acceptance tested.
- Large-first alone:207/205folds before continuation;189/200after (matching
  controls203/214). Seam-preserving+damped repair gives183/184 (225/233start),
  then200/195 after continuation. Mean error8.254/8.519 immediately after repair,
  6.919/6.775 after continuation. Strongerguard10000 ends208/195, not a cure.
- All8completed experiments and their matched controls recorded in
  repair_results.md and repair_metrics.json; full attempts in run summary.json.
  59tests passed,1known unrelated deselected. Actual repairs preserve outside
  boundaries and introduce no new diagnostic defects; continued solves do.
- No defaults promoted. Explicit limitation: this does not eliminate folds;
  finite soft orientation guard is not a hard injectivity constraint. No global
  self-intersection audit. Runtime33–55seconds per input including both controls
  and saves, non-exclusive grants on0/1. All trials finished.
