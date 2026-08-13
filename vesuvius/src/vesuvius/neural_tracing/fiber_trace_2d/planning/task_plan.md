# Plan: fiberlet graph replay

## Implementation

1. Replace radius-shell candidate enumeration with a deterministic filled
   cell neighborhood: all nonzero integer offsets whose length is below the
   existing outer half-open radius bound. Keep offsets symmetric and
   lexicographically ordered, and continue considering every retained component
   pair once under canonical `source_id < target_id` deduplication. Rename
   shell-only diagnostics directly because the format is experimental.
2. Add a hard DP sampled-fiber deviation limit, default 25 degrees. At every
   lattice transition require a valid dense fiber prediction and
   `abs(step dot fiber_axis) > cos(limit)` with no boundary epsilon. Integer
   lattice moves use the destination voxel prediction. Virtual endpoint
   attachments remain governed by the existing endpoint-axis rule because they
   only connect a fitted subvoxel anchor to the lattice. Keep Lasagna normals
   in the existing curvature split and serialize the effective limit.
3. Add a reusable fiberlet graph module. Nodes are retained anchor identities
   and base-coordinate positions; edges are every successful fiberlet with its
   exact polyline, endpoint tangents from the first/last distinct path points,
   length, and loss. Each edge supplies exact forward and reverse directed arcs.
   Precompute deterministic directed transitions and permit a join only when
   the tangent entering the anchor and tangent leaving it have an angle strictly
   less than 45 degrees.
4. Implement deterministic receding-horizon beam routing over directed graph
   edges. Seed from graph edges near and aligned with the beginning of the
   replay reference interval, rank lookahead routes by accumulated fiberlet
   loss per prediction-voxel path length with canonical identity tie breaks,
   commit one edge at a time, and prevent node cycles in committed and tentative
   states. Beam pruning is global per edge depth; only routes reaching the
   deepest available horizon compete. Ranking is additive loss divided by
   additive prediction-grid length, then canonical full directed-arc sequence.
5. Extract a shared exact forward-reference matcher. Greedy replay supplies its
   nominal step; graph replay supplies each actual nonzero dense segment length.
   Seed at the anchor nearest the reference-interval beginning among anchors
   within the inclusive failure threshold that have a strictly aligned outgoing
   arc; ties use node/arc identity. Match the seed and every committed point.
   Record the first point whose Euclidean distance to the monotone
   forward-window projection is strictly above `--fail`, without truncating its
   fiberlet. Keep later match records for diagnostics while preserving that
   first-failure identity. Before failure distinguish graph exhaustion,
   reference end, and invalid start; after failure use the postroll statuses in
   item 9.
6. Add `vc_fiberlets fiberlet-replay` with the same inputs and failure-tube
   extraction as `fiber-replay`, plus short `--beam` and `--lookahead` route
   controls. Share setup/extraction code between the two commands rather than
   copying the existing pipeline.
7. Publish strict graph JSON, route JSON, and a base-coordinate fiberlet-route OBJ in the
   content-addressed replay generation. Extend the replay JSON with graph-route
   status, matching records, route metrics, and artifact bindings. Extend the
   napari replay viewer/reload path with a stable distinct fiberlet-route line
   layer. Keep route visibility independent of the fiberlet-radius filter.
8. Make `--along` the sole longitudinal comparison extent. Derive the greedy
   postroll step budget from the effective base-coordinate trace step, retain
   the full greedy result for matching diagnostics, then reduce the requested
   value to one symmetric half-extent available on every reference/greedy side.
   Publish/display exact equal-sided arc-length crops and use the same bounds
   for the fiberlet graph interval. Persist and strictly validate requested and
   effective extents plus arc bounds. Remove `--after` directly.
9. Change graph replay into a two-phase trace. Before failure, retain the first
   strict over-threshold point and its candidate/path index. After failure,
   continue the same deterministic graph routing without applying another
   distance stop. Accumulate routed geometric arclength from the failure point
   while always completing selected fiberlets. Stop at the first target anchor
   whose accumulated postroll reaches or exceeds the effective `--along`
   distance; the bounded final-fiberlet overshoot is explicit metadata rather
   than a partial edge. Distinguish complete postroll from graph-exhausted
   truncation and persist requested/completed base-voxel distances.

## Tests

1. Update candidate-neighborhood tests to prove distances one through four are
   included once and offsets outside the half-open outer bound are excluded.
2. Add DP tests for accepted aligned steps, exact-25-degree rejection, invalid
   fiber-prediction rejection, invalid-Lasagna-normal fallback, and subvoxel
   endpoint attachment behavior.
3. Add graph tests for node/edge construction, endpoint tangent orientation,
   strict 45-degree joins, canonical transition ordering, and rejection of a
   sharp turn.
4. Add route tests for deterministic beam/lookahead choice, cycle prevention,
   exact forward matching, failure-threshold termination, and graph exhaustion.
5. Extend replay publication and Python viewer tests for graph/route artifacts,
   layer creation, filtering/reload, and strict malformed-data rejection.
6. Verify that a comparison trace distinct from the full diagnostic trace is
   hash-checked and displayed, and smoke-test the derived postroll budget.
7. Build affected targets with `-j32`, run focused C++ tests, and run the
   viewer Python test module. Run one local replay smoke command if the existing
   Paris4 inputs are accessible.
8. Add graph tests for anchor-bounded postroll completion across an additional edge and
   truncated postroll at graph exhaustion. Extend strict JSON/viewer validation
   for status-specific failure and postroll metadata, then rerun the user's
   `--along 512` workload when its shell-expanded input is identifiable.

## Spec Update

Replace shell-only pair generation with the filled neighborhood, specify the
hard 25-degree sampled-fiber step constraint, define the fiberlet graph
and strict directed join rule, and specify beam replay/matching/artifacts.
Specify the shared longitudinal extent and the distinction between complete
diagnostic trace data and comparison display geometry.

## Docs Update

Document the graph module, candidate and DP behavior, new CLI invocation,
route selection, replay bundle additions, and napari route layer.
Document that `--along` also owns greedy postroll and display cropping.
Document graph postroll measurement, completion/truncation statuses, and why a
route can still be shorter when the extracted graph is exhausted.

## Changelog

Record shorter-pair fiberlet generation, hard sampled-fiber feasibility,
fiberlet graph construction, and graph replay visualization.
