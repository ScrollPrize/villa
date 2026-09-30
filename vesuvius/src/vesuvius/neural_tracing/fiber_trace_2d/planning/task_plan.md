# N2 ordinary-only plan

1. Reuse the complete-graph N2 selection already tested; references are used only to select the cohort and later to read out the frozen solution, never as solver nodes/factors/initialization.
2. Add an explicit ordinary-only neighborhood option. Remap selected ordinary pieces/source lines/original IDs and cached reference cross links via existing subset helpers. Enter the existing ordinary BP pipeline, with component filtering off. Reuse independent reference calibration/ranking and existing correctness/shift parsers; no duplicate solver or metrics.
3. Keep original special post-split reference attachment semantics. Preserve all selected sources and pieces. Validate cohort identity versus previous fixed-reference runs and absence of reference nodes/factors/fixed states from the solver graph.
4. Build and test using approved runners; run three cost settings on cached2048 traces,64threads/run,500iterations, no parallel factors. Export full ladders, correctness, shift count/percentage, provenance and comparisons with conditioned diagnostics. Missing/ambiguous reference readouts must remain explicit.
5. Independently review this plan and wiring. No parameter sweep beyond the requested three. Requested latest scope is unconditioned correctness+shift; reference-oracle filtering score is not substituted by conditioned defect percentages and remains a separate evaluation if not run here.

## Spec update
No spec.md. No changes to extraction, reference attachment, inference or numerical policy. Document ordinary-only N2 behavior and reference-assisted readout calibration.

## Docs updates
Add dated report and machine-readable results, update fiber_chunk_tracing.md, task_log/status and changelog. Include exact commands and snapshot provenance.
