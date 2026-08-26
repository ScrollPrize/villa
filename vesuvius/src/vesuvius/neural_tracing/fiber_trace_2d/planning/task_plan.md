# Plan: mixed-integer crop-fiber labeling

## Constraint preparation

1. Add an exclusive maximum accepted winding distance of `1.5` to constraint
   extraction. Reject non-finite and out-of-range measured links before they
   enter reports, diagnostic OBJs, or optimization; retain same-trace
   continuity links at winding zero.
2. Keep deterministic piece and constraint ordering. Report valid links removed
   by the threshold through a distinct cutoff counter; retain the existing
   winding rejection counter for invalid/non-finite sampling.

## HiGHS MILP

3. Add a reusable core labeling API backed by the installed HiGHS C++ target.
   Represent each piece with active, H/V, and odd/even binaries, with H/V and
   parity forced to zero when inactive so broken has one canonical encoding.
4. For every link add bounded pair-active and gated H/V-difference and
   parity-difference auxiliaries. Linear envelopes force them to their binary
   AND/XOR values once the endpoint piece variables are binary, so they do not
   expand the branch-and-bound integer dimension.
5. Use the exact active-link objective
   `(1-pair_parallel) * pair_active + (2*pair_parallel-1) * hv_difference`
   plus
   `winding * pair_active + (abs(1-winding)-winding) * parity_difference`.
   Charge each broken piece `broken_cost_per_link * incident_degree`, default
   `0.5`, including continuity and measured links. This disables all incident
   link costs when broken without deleting graph structure.
6. Require an optimal HiGHS result at a reported configurable relative MIP gap
   (default `1e-4`, with `0` available for an exact proof), validate the returned binary solution, and
   canonicalize the unavoidable global H/V and parity symmetries independently
   in every active connected component: its lowest piece ID becomes H/even.
   Isolated zero-degree pieces are canonically broken. Report objective
   decomposition, model dimensions, solve time, and counts of all five labels.
   Run HiGHS without its verbose log and with deterministic solver settings.
7. Add explicit HiGHS package discovery and link ownership on
   `vc_fiber_tracer`; add `libhighs-dev`, Homebrew `highs`, and vcpkg `highs`
   to the existing Ubuntu, macOS, and Windows dependency definitions.

## CLI and visualization

8. Extend `vc_fiber_trace_chunk constraints` to solve immediately after
   extraction. Add `--broken-cost-per-link` for the default `0.5` coefficient
   and require it to be finite and nonnegative in both CLI and core APIs.
9. Write sampled piece polylines to deterministic suffixes
   `_h_even.obj`, `_h_odd.obj`, `_v_even.obj`, `_v_odd.obj`, and `_broken.obj`
   using the same output basename as the existing constraint diagnostics.
10. Print the cutoff, optimization summary, each class count, and all five
   paths. Preserve the three existing constraint diagnostic OBJs.

## Tests

11. Test the exclusive `1.5` cutoff, including invalid/non-finite winding, a
    retained value immediately below it, and retained hard continuity.
12. Test objective behavior on small exact graphs: parallel/same-winding,
    perpendicular/opposite-winding, broken-piece selection, disabled incident
    costs, deterministic canonical labels, and objective decomposition.
13. Test five-way OBJ naming, classification, stable piece identities, empty
    classes, invalid broken costs, and repeated symmetric/isolated graphs.
    Exercise CLI output naming and ensure failed solves cannot write labels.
    Build and run focused GCC and Clang tests, run
    `git diff --check`, and execute the representative 500-trace Release
    command to report model size, solution distribution, and solve time.

## Spec update

Extend `planning/specs.md` with the exclusive winding cutoff, the five-state
piece model, exact MILP variables/constraints/objective, broken cost, required
optimal status, output suffixes, and report fields.

## Documentation updates

Extend `volume-cartographer/docs/fiber_chunk_tracing.md` with the labeling
semantics, equations, HiGHS dependency, CLI option, five OBJ outputs, and
interpretation of broken pieces.

## Changelog

Add crop-trace MILP labeling and five-way OBJ visualization to the 2026-08-26
entry in `volume-cartographer/planning/changelog.md`.

## LP-relaxation tightening follow-up

14. Replace each edge-dependent expensive-relation auxiliary with a stable
    gated H/V or parity XOR variable: zero means equal active endpoint labels,
    one means different active endpoint labels, and a broken endpoint gates the
    value to zero. Use the direct signed objective coefficient; binary feasible
    assignments and costs must remain identical.
15. Build deterministic piece adjacency and bound the H/V and parity columns of
    the lowest piece in each input connected component to zero without fixing
    its active column. This is objective preserving; if that root is broken it
    deliberately does not remove the remaining split-component symmetry. For
    LP relaxation only, enumerate graph triangles once and add the four
    cut-polytope triangle inequalities independently for H/V and parity. Gate
    each edge inequality with the sum of the three active values, so one broken
    vertex still permits the opposite active edge to differ.
16. Report the number of gauge components, triangles, and triangle rows. Keep
    raw continuous piece values in CSV without thresholding or repair.
17. Add exact small-graph tests proving unchanged binary costs, gated broken
    behavior, rejection of the impossible three-differences triangle, and
    deterministic gauge fixing. Run GCC focused tests and compare 256 and 1024
    crop LP distributions, objectives, row counts, wall time, and memory.
18. Reuse the five-way piece OBJ writer for an explicitly diagnostic relaxed
    classification: active values at or above their mean survive, V and odd
    own exact `0.5`, and all lower-activity pieces are shown as broken. Prefix
    the five suffixes with `_relaxation_`, report the mean, and test exact
    threshold ownership and class counts.
19. Add relaxation-only CLI/config controls for HiGHS parallel execution and
    solver choice (`choose`, `simplex`, `hipo`, or `ipm`). Reject their use on
    the MILP path, report the requested backend, and retain HiGHS automatic LP
    selection as the default.
20. Build and run focused tests, then benchmark the same centered 384-base-
    voxel artifact sequentially with parallel automatic selection and parallel
    HiPO. Record wall/CPU time, peak RSS, objective, status, and visualization
    artifacts without changing the LP formulation.

The gated difference hull is explicit: `difference <= pair_active`, the two
signed endpoint lower bounds receive `pair_active - 1`, and the two XOR upper
bounds are `difference <= x_a + x_b` and
`difference <= 2 - x_a - x_b`. The direct coefficient is
`different_cost - same_cost`, so both positive and negative cost signs retain
the exact binary objective.

### Follow-up spec update

Document the stable gated-difference semantics, component gauge, LP-only
triangle cycle inequalities, their broken-piece gating, and new diagnostics.

### Follow-up documentation update

Explain why the local XOR envelope admitted the all-half solution, what the
triangle inequalities guarantee, and that longer-cycle cuts remain a possible
next tightening step if triangles are insufficient.

Document the explicit LP backend controls, their diagnostic-only scope, and
the fact that HiGHS may still choose a serial algorithm unless parallel mode or
HiPO is requested.
