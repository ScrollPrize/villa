# Plan: lossless post-stage-two Fiberlet graph simplification

## Semantics and representation

1. Materialize the post-stage-two retained physical graph in the same centered
   half-open box and rebuild its directed transition adjacency with the same
   join-angle, prediction-validity, normal/tangent scoring, and selected edge
   cost view used by exact stage-two analysis.
2. Treat a directed Fiberlet arrival as the graph state. Classify its
   admissible successors after excluding reversal over the same physical
   Fiberlet. Report zero, one, and multiple-successor state counts.
3. Compute conservative forward reachability from directed entries and reverse
   reachability from directed exits. Remove states outside their intersection.
   This cannot remove a valid simple route because every such route is present
   in both reachability sets. It may conservatively retain a state when the two
   reachability witnesses cannot be concatenated without revisiting an anchor.
   Retain a physical Fiberlet when either direction remains live and carry an
   explicit live-direction mask through every later adjacency and macro step;
   never recreate its removed reverse orientation.
4. Remove all anchors not referenced by a remaining physical Fiberlet. Outside
   endpoints of crossing Fiberlets become boundary portals keyed by their
   original anchor identity, so entry-root revisit semantics remain unchanged.
5. A physical interior anchor is contractible only when exactly two remaining
   physical Fiberlets touch it and their mutual transition is admissible in
   both directions. Boundary portals, entry/exit-only anchors, branch anchors,
   and one-way continuations stop physical contraction.
6. Build maximal physical macro-Fiberlets by walking through contractible
   anchors. Each directed macro stores its ordered original directed Fiberlet
   IDs, complete ordered anchor sequence, per-edge and per-join losses, and
   per-edge lengths. Aggregate loss and length are diagnostics only;
   authoritative evaluation replays the original scalar sequence in the same
   order and association as physical expansion. Preserve cycles as
   uncontracted edges; never invent a self-loop macro.
7. Build macro transition adjacency from the final physical edge at the end of
   an incoming macro and the first physical edge of an outgoing macro. This is
   the precomputed regular join relation, not a relaxed connectivity rule.
8. Build deterministic directed rollout descriptors from macro states with one
   admissible successor. Stop at exits, branching states, repeated anchors, or
   cycles. Macro/rollout application exposes an atomic visited-anchor validator
   which rejects the complete candidate if any hidden target anchor already
   exists in the route history, then records the full hidden sequence on
   success.
9. Validate canonical physical IDs. One stored physical Fiberlet per exact
   anchor-key pair is already enforced, so exact same-endpoint duplicates are
   impossible. Do not remove distinct higher-cost physical or macro routes:
   preserving only the optimum would not preserve the valid route set, and
   visited-anchor histories can make an apparently dominated geometry useful.
   Report the structural duplicate count, which must remain zero.
10. Keep this as an exact in-memory macro graph/report. Do not serialize macros
    as ordinary Fiberlets: the current route lattice is defined by one endpoint
    pair and cannot encode concatenated geometry without resampling. Persistent
    macro serialization is a separate format task.

## CLI integration and reporting

1. Run simplification on each centered stage-two crop after exact stage-two
   retention. Input is the stage-two retained physical-ID set, not the larger
   stage-one graph.
2. Report each centered box independently. Macro partitions depend on the box
   boundary and have no global identity across overlapping centered boxes. For
   the current 512/256 experiment this is one report.
3. Print a compact simplification table for the common stage-two selection:
   physical Fiberlets before/after reachability, unused anchors removed,
   contractible anchors, physical macro count, physical edges represented by
   macros, and directed macro states.
4. Print continuation counts and chain distributions: zero/one/branching
   directed states, deterministic rollout count, mean/median/max physical
   Fiberlets per macro, and mean/median/max macros per rollout.
5. Keep detailed per-box lists behind `--stats`; headline counts remain visible
   by default.

## Tests

1. Add a deterministic fixture containing a bidirectional degree-two chain, a
   branch, a one-way forced continuation, a dead state, an unused anchor, and
   boundary entry/exit Fiberlets.
2. Verify forward/reverse reachability, complete unused-anchor removal,
   boundary portal identity, continuation-degree counts, and exact retained
   physical IDs.
3. Verify physical macro ordering, forward/reverse ordered edge/join scalars,
   lengths, live-direction masks, internal anchor sequence, and that expanding
   all macros reproduces the original physical sequence without reassociation.
4. Verify branching and one-way anchors are not physically contracted while a
   deterministic directed rollout may cross a one-way forced continuation.
5. Enumerate every simple entry-to-first-exit route in the bounded fixture
   before and after simplification and compare expanded directed physical
   sequences and ordered scalar evaluation. Include a hidden-middle-anchor
   revisit, exit to the entry-root portal, a one-live-direction edge, a cycle,
   and an exact-cost tie. Verify exact endpoint duplicates are rejected by the
   existing canonical-ID graph materialization.
6. Build `vc_fiberlets`, `test_fiberlet_storage`, and
   `test_fiberlet_paths` with 32 threads; run both tests and `git diff --check`.
7. Run the hot Paris4 512/256 two-stage command and record simplification
   counts and elapsed time.

## Spec update

Extend the chunk-route diagnostic contract with post-stage-two directed
reachability pruning, complete unused-anchor removal, boundary portals,
bidirectionally safe physical macro contraction, deterministic directed
rollouts, exact cost/length preservation, and the restriction that macros are
references to original Fiberlets rather than ordinary serialized Fiberlets.

## Docs update

Document simplification semantics and the additional post-stage-two tables in
`volume-cartographer/docs/fiberlets.md`, including why exact same-endpoint
physical duplicates cannot exist and why macro persistence needs a distinct
format.

## Changelog

Record lossless post-stage-two graph simplification and the measured Paris4
reductions without claiming that macro graphs are yet consumed by replay or
persisted as ordinary Fiberlet datasets.
