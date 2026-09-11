# Plan: saved-result analysis

1. Commit completed work separately by subproject (done: 75013f6cc, a238c4598, 175e3645b).
2. Inspect saved complete summary/case artifacts and shared metric definitions. Use existing failed-span reference-distance fields for mean before restart; distinguish terminal spans and corpus reliability metric. Aggregate millimeters before dividing, merge both Paris4 collections, identify PHerc1299 from provenance rather than PHerc1229 label.
3. Independently review plan/metric interpretation. Validate 100 complete directions, counts, units, summary consistency, and weighted aggregation. No benchmark rerun or behavior change.
4. Save reproducible read-only analysis and results; explain reference distance versus traced polyline length and terminal handling. Answer GPU-bound question from completed profile evidence.

## Spec update

No runtime/spec changes. Existing whole-corpus reliability convention remains unchanged; label distinct empirical span metrics explicitly.

## Docs updates

Archive preceding task log. Add saved-result analysis report with input path, formulas, counts and reproduction command; update status/task log/changelog.
