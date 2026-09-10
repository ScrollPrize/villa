# Request concurrency plan

1. Inspect loader/capacities and specs; previous layout/prefault findings remain in docs/live_inference_chunk_major_20260910.md.
2. Independent plan review.
3. Shared coordinator snapshot: pending requested/incidental chunks, unique queued foreground/background tiles, submitted reads split executor-queued/running/done-awaiting-dispatch, GPU submitted/uncollected (NOT hardware utilization), input/result capacities/free/owned, accumulation tasks and publication states. No qsize(), no scheduling changes.
4. Profile snapshots nominal250ms with actual elapsed timestamps/gaps, initial/final and distributions in report.json; extended console progress. Live same counters plus native RPC waiters every1s to buffered JSONL including final drain, no fsync. Coordinator sampling is not guaranteed during blocked operations. Reuse shared snapshot/formatter, no duplicate accounting.
5. Unit tests for read states/capacity identities and100+ pending chunks versus bounded reads; existing regressions and one approved short chunk-major/prefault-off/128result/8input profile. No capacity increase or full trace restart.

## Spec update

Define counters/cadence/limits, preserve model and scheduling semantics.

## Docs updates

Concurrency report: loader path, exact command, counts/distributions, limits and proposed next step; active log/status.

## Changelog update

Record observability and measured limiting stages, not an unmeasured speedup.
