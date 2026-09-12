# Three EMA evaluation plan

## Follow-up pair

Reuse the same runner for best99 then best999 from the active run. Add optional
validated unique variants list (supported exact names only), preserving default
three-entry order. Freeze both beforefirst; fresh output root, include three
prior EMA evaluations as additional baselines. Test requested two-entry order and
invalid/duplicate names. All admission, audit, report and no-training-mutation
contracts unchanged. Report after each, then verify original task resumes.
Observed prior-series cache-name collision: namespace must identify the full
evaluation output path, not its leaf. Provide explicit failed-only retry into
a fresh attempt directory, preserving the original frozen selection and logs;
test no automatic retry, exact selection retention and cross-series isolation.

1. Bind active combined run directory and training launch identity. Through approved experiment controller freeze current_ema_0p999.pt, best_ema_0p999.pt, best_ema_0p99.pt before any evaluation. Reuse freeze_checkpoint; read metadata from frozen files, validate EMA kind/decay/test metric, record step/samples/hash. Concurrent serialization or corrupt copies fail clearly, never silently substitute.
2. Extract existing frozen-evaluation invocation into one shared helper for trace_evaluate and the series. Preserve prepare/validate-full/full and exclusive manager admission. Add trace-series-prepare and trace-series-next to approved launcher; next executes only next entry in declared order and audits all100directions. No training stop/start/watch calls or signals.
3. Generate cumulative Markdown after each result using previous audit reports and classic complete_report. Validate identical scroll IDs/counts/reference lengths; average five means at20% each. Show per-scroll distributions with best/ties bold, no pooled overall score.
4. Execute three serial evaluations, report each immediately, verify training root alive and manager pause/resume.

## Spec update

No training, inference, tracing geometry, manager protocol or numerical changes. Snapshot variants use exact filenames, not cross-decay selection. Classic is longer-trained; new snapshots are later than600k.

## Docs updates

Save cumulative reports and frozen provenance under output root; summarize usage/results in docs and task log.

## Validation

CPU tests for freeze metadata/decay checks, exact variant order, failures not marked complete, no training stop/restart, equal-scroll means and highlighted extrema. Existing managed preflight/full100direction artifact audits for all three GPU evaluations. Independent plan/report review.

## Changelog

Record evaluation results/report paths and unchanged active training.
