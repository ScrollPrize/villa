# Preserve certified constraints across splitting

Current correction supersedes cost exploration. Construct/certify local constraints on source fibers first; cuts may only remap their owning piece IDs and add continuation links. Never clip support or reject an event because a split crosses its measurement span. Normalize repeated parallel evidence by original source-trace pair before remapping, so additional splits cannot multiply weights. Explicit parallel ablation remains allowed and happens after common cut construction.

Audit additionally found reference cross extraction re-certifying already-cut pieces. Fix before accepting benchmark correctness: certify on reference.lines + bpSourceLines, then remap with the same core helper as ordinary extraction. Preserve canonical one-trace-per-piece presentation required by existing reference consumers. Missing owners from explicitly removed component intervals are separately counted; never map a gap to the next piece. Add short-piece and removed-gap regression.

Regression tests: tightly spaced orthogonal crossings with support crossing both ends of middle pieces must all survive; adding crossing-induced splits to a parallel pair must preserve its event measurements, support and weights. Preserve original two-sided disagreement tests, geometry/no-overlap and parallel ablation tests. Instrument certified/retained counts so real extraction proves no split-stage loss.

Build and full native suite via approved runners. Rerun grouped legacy and local parallel-on/off fixed-H/V hard controls, plus joint D10 orthogonal-only hard/flat10/distance10 comparisons. Same traces, normals, model, 500-message policy, caches; separate corrected artifact root, no overwrite of flawed historical runs. Record full percentage metrics, shift scores and event-retention audit with plot; mark prior graph experiments invalid for intended formulation. Independent review before implementation and after tests. No defaults promoted without results.

## Spec update
No spec.md exists. Clarify immutable certified constraint support/measurement through segmentation; no weakening of certification.

## Docs updates
Correct earlier support-clipping documentation; report corrected vs old graphs and results, exact commands, tests, limitations. Update task/status/log/changelog.

## Previous experiment

Next experiment: retain piece disabling, but replace constant continuation cut energy with lambda*abs(q_b-q_a), where q is integer winding plus the calibrated H/V phase offset. Thus half-winding changes cost lambda/2 and large jumps cost proportionately more. Ordinary constraints remain unchanged; no saturation. Reuse the existing soft-continuation path and diagnostics (discontinuity, not a removed zero-cost edge). Defect endpoints still pay the piece unary rather than continuation energy.

Test joint H/V, parallel off, D=10, lambda={0.1,1,10,100}, with matched hard and constant-C10 controls. Existing approved runners only; frozen traces and 500-iteration limits unchanged. Validate exact energy scaling, H/V half steps, sign symmetry, zero/defect behavior and unchanged constant/hard modes. Save initial and conditioned metrics and ladder-offset changes, full table and plot. No default promotion. Independent plan and implementation review required.

## Previous completed matrix

1. Add opt-in independent continuation cut cost to joint-grid BP. For two active endpoints, the enabled connection requires equal orientation and integer winding; otherwise it is cut at the configured constant cost. This is a minimized binary switch (finite Potts potential), not an ordinary measurement-disable prior. Defective endpoints already disconnect the edge and incur no additional cut cost. Preserve defaults and ordinary orthogonal factors even when aggregated with continuation on the same pair.
2. Make messages, decoded energy, projection, diagnostics and reference removal/readout honor this switch. Record cut connections separately from disabled pieces. Zero cost is exactly neutral, not a hidden residual continuity cost. Reject unsupported solver modes.
3. Run joint orientation/winding solves, not frozen prepass orientations, so active pieces may differ in H/V across a cut. Include matched joint hard-continuity controls plus previous fixed-orientation local and legacy results as context. Keep normal alignment at 500 iterations, source traces, extraction, reference data and winding budget fixed.
4. Add resumable cost sweep using existing benchmark runner and shared metric/provenance helpers. First coarse grid: piece defect cost {1,10,100} crossed with continuation {0.1,1,10,hard}, both parallel on/off (24 runs). Up to four concurrent runs, 48 threads each. Follow up around useful interior/boundary candidates only when warranted. No predictor/inference work.
5. Audit pre-BP geometry and unique orthogonal-event arc groups per piece. Coincident events are a group; report violations rather than assume one raw factor per piece. Existing largest-component filtering may change retained geometry downstream.
6. Validate native synthetic cases for active-active cuts, piece disabling, zero-cost neutrality, preserved ordinary factors, energies, invalid inputs and hard/default regression. Test sweep construction/resume/ranking. Execute through approved native build/test and evaluation runners only. Save full percentage tables including initial independent and reference-conditioned metrics, defects/cuts and convergence; do not promote a setting on removal score alone.
7. Because early joint controls are substantially worse, add a matched fixed-H/V slice at the original D=100: C={0.1,1,10,hard}, parallel on/off (8 supplemental trials). This isolates cut benefits from orientation-policy changes; fixed mode only permits winding changes across active cuts. Preserve both matrices in the same report.

## Spec update
No spec.md exists. New behavior is opt-in and documented; production defaults unchanged. Cutting continuity does not disable orthogonal evidence. Original plan.md is an unrelated aspirational 2D plan.

## Docs updates
Update native CLI documentation, add dated benchmark Markdown/JSON/SVG with input and command provenance, task log/status and durable changelog. Explicitly document joint versus frozen orientation controls, switch semantics, event grouping and convergence limitations.
