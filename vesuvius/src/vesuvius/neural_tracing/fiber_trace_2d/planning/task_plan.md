# Plan: compact Fiberlet crop lookahead state

## Contract

- This is a search-state representation optimization only. It must not change
  floating-point operations, graph traversal order, pruning boundaries,
  lexicographic tie-breaking, generated-state limits, or selected Fiberlets.
- The committed trace prefix remains one immutable visited-anchor set owned by
  `traceSide`; no branch may copy it.
- A rollout cycle is rejected when its target is in either the committed prefix
  or the current state's parent chain, matching the existing copied set.

## Implementation

1. Replace `LookaheadState::visited` and `LookaheadState::arcs` with a compact
   route-node arena containing only anchor, parent index, and arc from parent.
   Active frontier state keeps its route-node index, incoming arc, first arc,
   depth, accumulated loss, and accumulated prediction-space length. This
   avoids retaining a full arc payload for every historical generated state.
2. Keep frontier vectors as compact states whose ancestry uses arena indices.
   Append each accepted route node once and retain parent links for the arena
   lifetime. Avoid references into the arena across appends so vector
   reallocation is harmless; do not reserve the one-million-state hard cap.
3. Check rollout-local cycles by walking the short parent chain after checking
   the immutable committed-prefix set.
4. Reconstruct the full arc sequence only when creating a completion or when
   the existing intermediate density/lexicographic pruning boundary is reached.
   Build one `{state, reconstructed_arcs}` record per state at that boundary,
   sort by the exact existing `loss / length` followed by the full arc vector,
   and rebuild the frontier in that order. Do not introduce hashes or new keys.
5. Retain completion route-node/depth indices instead of materialized route
   vectors. Select the single required minimum by the exact existing density
   ordering. For exact-density ties, compare parent-linked arcs in forward
   lexicographic order without allocation. This is equivalent to sorting,
   truncating, and returning the first completion. Keep all arithmetic
   expressions and their order unchanged.

## Tests

- Add focused regressions for committed-prefix cycle rejection, current-node
  and ancestor rollout-cycle rejection, dead-end completion, horizon/crop-exit
  completion, and deterministic completion tie-breaking.
- Exercise intermediate pruning with more than `beamWidth * 64` equal-density
  states, and exercise the exact generated-state cap semantics with a low-cap
  branching graph.
- Run GCC Release `test_fiberlet_crop_trace` repeatedly and the relevant
  storage/path suites; run the Clang crop test when available.
- Profile a smaller representative crop with an available local profiler or
  focused internal timings. Benchmark the same Paris4 1024-base-voxel crop with
  500 attempts repeatedly in the existing Release build. Report iteration
  count, min/median/max wall and tracing time, user/system time,
  graph-preparation time, and computed/discarded candidates.
- Record peak route-arena population and estimated bytes on the canonical
  workload to confirm bounded practical memory despite the generated-state cap.
- Compare every generated OBJ against the committed pre-change baseline with
  `cmp`, and run `git diff --check`.

## Spec update

Specify that crop lookahead keeps the committed prefix immutable and uses
parent-linked rollout ancestry, materializing routes only for canonical
ranking/completion while preserving exact search semantics.

## Documentation updates

Document the allocation behavior and explain why the arena has no numerical or
search-result effect.

## Changelog

Add the measured lookahead-state optimization and exact-output validation to
the current crop-tracing performance entry.
