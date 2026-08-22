# Plan: restore aggregate fiberlet replay as the default

## Implementation

1. Add an explicit replay cost-evaluation mode with `fiberlet` and `stepped`
   values. Default the public replay config to `fiberlet`.
2. In `fiberlet` mode, rank with existing stored whole-edge and join costs.
   Score from the segment seed through the common horizon, include complete
   fiberlets and joins once, and prorate only the horizon-crossing fiberlet by
   edge length. The checkpoint is only a commitment boundary. Do not request
   cost-profile payloads or evaluate the integration grid.
3. Use the same aggregate costs in exact-search scoring and its relaxed
   cost-to-go bound, and in intermediate-pruned scoring. Both paths must remain
   profile-free.
4. Preserve the current subsegment/grid implementation behind explicit
   `stepped` mode. Keep its weight, delay, integration-step, and profile-blend
   controls unchanged.
5. Add `--cost-mode fiberlet|stepped` to the CLI. Reject stepped-only CLI
   controls unless `--cost-mode stepped` is explicit, preventing silently
   ignored settings.
6. Reject non-default stepped settings in aggregate mode at the core API, and
   record `cost_mode` in replay JSON while omitting inactive stepped fields.

## Spec Update

- Specify `fiberlet` as the default evaluator and its exact aggregate boundary
  semantics.
- Specify `stepped` as the explicit profile evaluator. Its existing numeric
  controls and defaults remain unchanged.
- State that switching modes does not change cache identity or stored payloads.

## Docs Updates

- Update the replay command and option reference.
- Correct the earlier implication that profile blend zero itself is the desired
  default. It merely reproduced failure arcs while still using the slower
  stepped machinery.
- Retain the measured two-failure result as evidence for aggregate-fiberlet
  selection, clearly distinguishing evaluator identity from equivalent results.
- Maintain `status.md`, `task_log.md`, and the changelog.

## Tests

- Add behavioral coverage where default aggregate mode and explicit stepped
  mode select different candidates.
- Prove default aggregate mode never calls `costProfile` in both exact and
  intermediate-pruned search, while stepped mode does.
- Cover the partial-horizon edge, entering-join, and checkpoint-inside-edge
  aggregate semantics.
- Cover CLI mode parsing, help defaults, rejection of stepped-only options in
  aggregate mode, and replay JSON mode serialization.
- Build `vc_fiberlets` and affected tests with `-j32`.
- Run the replay, storage, trace, and fiberlet path suites; report the known
  unrelated path-fixture failures separately.
- Measure a short hot-cache aggregate-versus-stepped replay with command,
  input, build type, wall/CPU time, and peak RSS.
- Run `git diff --check`.

## Changelog Update

- Record restoration of the fast aggregate evaluator as the public default and
  the explicit stepped-mode selection.
