# Plan

1. Use current.pt,best.pt,best_reviewed_trace.pt; freeze all3 with existing freeze_selection before any long evaluations. Validate raw kind/selection metric,save hashes,steps and metadata.
2. Reuse exact winning workload; replace only checkpoint,output,cache namespace and labels. Same inference precision/resources, GPU0–5,cohort and trace args. Distinct model identities; no native/default changes.
3. Reuse evaluate_config, measured_result, filter_counts and short-output analyze. Extract shared result collection if necessary, never duplicate tracing/inference/metrics. Run3 through approved wrapper; compare original/min512 metrics globally/per-scroll with prior winner. Audit182cases91runs,effective args/cohort and retained accounting. Report each completion.
4. Test config invariants and shared metrics; independent review; log/docs/changelog.

## Spec update
Correction: raw-only selection was incomplete. Include current and best-loss EMA99 and EMA999, freezing all four before continuation; validate EMA kind, decay and metric. Retain completed raw results without rerunning inference/tracing. Only one best-reviewed-trace file exists, no separate EMA restart selections. Existing GT-assisted and output-only filter caveats preserved. No changes to scheduler, GPU0–5, inference precision or tracing policy.

## Docs updates
New194246 MD/JSON with snapshots,metrics,per-scroll rows,provenance,timing,reproduction command. Prior reports retained. Task/status/log/changelog updated.

## Validation
Config test ensures only allowed identity/output changes. Existing metrics tests; min0/source parity and retained+discarded checks. Effective trace equality with winner. No native build needed. No resource tuning or new approval commands.
