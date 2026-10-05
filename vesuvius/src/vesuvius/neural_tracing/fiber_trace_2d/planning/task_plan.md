# Seven-snapshot plan

1. Freeze all seven current files with hashes, sample counts and selection metrics.
2. Verify current raw/EMA sample counts agree; report any differences explicitly.
3. Execute existing trace-series-next sequentially for all seven with unchanged
   full91 workload (182 directions), native build, precision and tracer settings.
   Verify template SHA256 before every entry. Manager GPU0–5 only; do not stop training.
4. Validate completion and zero infrastructure failures, report each result and
   compare final per-scroll counts with previous set. Use samples, not updates.
5. Verify training alive after manager releases the final evaluation allocation.

## Spec update

None: reuse shared standalone benchmark and existing manager semantics.

## Docs updates

Record snapshots/results in task_log.md; prior results preserved in
evaluation_results_20261004.md and external comparison series directories.

## Tests / validation

Existing prepare, validate-full, complete_report; 182 directions and no
infrastructure failures per snapshot. No model/inference implementation edits.

## Changelog

Experiment only; no implementation change.
