# Plan

1. Inspect clean retrain b52438d64 and embedding1f24f17d2; source already contains earlier retrainb32f31a55. Read workflow and GPU-manager instructions; use this scoped plan before merge to avoid modifying overlapping active planning files. Independent review required.
2. Merge pinned embedding1f24f17d2 with --no-ff --no-commit into retrain. Do not touch other checkout or its uncommitted changes. Inspect automatic merge and conflicts; no substantive code conflict expected (retrain's unique commit is evaluation config/results/docs). Escalate unexpected semantic choices; do not discard features.
3. Archive both branches' active task/status/plan/log texts with commit provenance. Resolve planning conflicts by retaining durable history from both branches and making current active task/plan/log describe this merge. Preserve both changelog entries and all specs. Existing source production changes must remain byte-identical to source tip unless an explicitly reviewed fix is required.
4. Validate no conflict markers/unmerged paths, diff whitespace and exact inclusion of new evaluation JSON/config files. Run CPU-only GPU-manager and focused fiber training/queue/arc/config/metric tests using existing environment; no GPU grants or installations. Explain failures/limitations without unrelated fixes. Review code diff against both parents and rerun appropriate validation if fixes required.
5. Record conflict resolutions, features integrated and test commands/results in logs/docs. Stage only merge/plan changes, complete merge commit, verify clean tree/current branch/both ancestry paths. No push.

## Spec update

No new behavior specified. Preserve incoming dense-supervision/scroll-balanced arc sampling, unified cooperative queues/calibration and manager responsiveness/recovery specs alongside retrain corrected metric specs.

## Docs updates

Archive both planning histories with provenance, current merge task/log/status, changelog and concise merge record. Keep latest28k/best11k benchmark results intact.
