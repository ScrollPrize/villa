# Plan

## User interruption: visualization first
Pause oracle implementation. Enable optional render on shortened benchmark binding,
using existing native inline inspection renderer and CT input-volume binding just
as regular live eval. No new visualization implementation. Rerun the same five
weighted cases with retained inference/construction cache in a dedicated vis root.
Test render binding/default preservation, run approved tests/runner, verify five
image sets plus metadata and report paths. Then wait for user inspection before
continuing the oracle. No oracle code changes have been made.

1. Reuse five directed pieces/model/weighted-i4 graph from mini5 pilot; archive prior planning. Inspect shared graph costs, successor legality, replay matcher/ellipse/grace and serialization.
2. Add opt-in native diagnostic uniform-cost search to shared replay implementation, reusing successor expansion, path evaluation and cost accounting (no duplicate graph or metric). Fixed original seed, legal simple paths/joins/initial direction; same40/80ellipse and160grace. No beam pruning or unsafe node-only dominance: reference matching and excursion are path dependent. Stop at actual reference progress target, not merely consumed path length. Nonnegative cumulative edge+join costs; terminal partial edges charged consistently. Deterministic ties, resource cap reports unresolved, not no-route. Return certified minimum only when min-cost complete label is popped; exhaustion is no-valid-route for this fixed seed/simple-path domain. Only ordinary fiberlet cost mode supported initially.
3. Run regular replay plus oracle sidecar on same graph/session with fixed first seed, stop before any reseeding. Save regular graph and oracle geometry/matches/component costs/stats. Oracle never changes reported ordinary benchmark stats or cache identity. Compare to first failing segment through matching GT progress; label uniform-in-edge clipping and current production terminal semantics explicitly. Full-piece minimum is not the production finite-lookahead density objective; separate objective mismatch from component differences.
4. Tests: cheap violating vs expensive valid paths, grace across edges, no valid path, deterministic results, state limit unproven, terminal progress and cost, defaults unchanged. Approved build/test runner; exact benchmark-full runner on mini5. Resource-bounded diagnostic if complexity high; report bounds/status rather than invent optimality.

## Spec update
Opt-in GT oracle diagnostic only, fixed seed/simple-path domain, nonnegative total cost, no beam prune, per-path matching/excursion, explicit certified/exhausted/resource-limit status. No default or construction changes.

## Docs/changelog
Persist method/limitations, exact commands, five-case cost comparisons and artifact paths. Archive mini5 planning, update current log/status and changelog.
