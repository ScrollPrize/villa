# Plan: next collection snapshot evaluation

1. Read existing full-test workflow/specs, inspect current best.pt metadata and hash, confirm model native stride4. Freeze selected step3000 snapshot (384000samples, test/loss_total0.43289151958754807, SHA256ee4595cd702029ad1f2edbbd56ebaf8006991b57601f3bbc7499b0e1a4970b18) with existing freeze_checkpoint helper in /tmp staging so ongoing training cannot change preparation/audit. Live runner subsequently freezes it into global inference model cache as usual. Keep original training path/provenance in docs; no new snapshot-copy implementation.
2. Archive previous active log and independently review plan. Clone prior full cap8 config changing only checkpoint path/hash and fresh output/cache namespace. Exact6collections/5scrolls,50fibers100directions, inputL1,total factor4,native4,256tile96overlap32border32outputchunk,bf16,cap8,existing trace/ellipse/reset settings. No production-code changes planned, no installs, no retraining or GT-heading fix.
3. Prepare and validate-full with approved run_classic_fiber_evaluation.py entrypoint, then full with exclusive8GPU manager priority100/224logicalCPUs. Manager handles pausable training; never bypass exclusivity or mutate training manually. Fresh inference; existing source fetches reusable. Monitor to100/100 and successful cleanup; no stop at launch. Keep stable hash throughout.
4. Use existing saved-artifact analysis script (optional root, actual terminal-segment counts) and existing failure/percentile definitions. Validate all100artifacts, units/source identities/counts, zero warm inference, drained queues and released manager grant. Compare mean/median/P05/P95/failures/zero spans globally/per-scroll with classic and previous collection; retain all channels/provenance. One full trial, report timing limitations and no significance claim.
5. Record exact commands/build/config/data/snapshot hashes and results under docs/benchmarks, link existing native_trace2cp_benchmarks.md, update task_log/status/changelog. Native-stride batching already validated in previous task; configuration-only preparation validation and actual full run are primary tests; rerun relevant lightweight checks if code changes become necessary.

## Spec update

None: existing authoritative-checkpoint/native-stride/exclusive full-test specs apply unchanged. Keep GT-heading investigation deferred.

## Docs updates

New benchmark record for run20260911_182306, full paired metrics vs prior baselines, artifact audit and timing provenance. Archive previous log; update current workflow/history. Preserve measured old records unchanged.
