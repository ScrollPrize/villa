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
