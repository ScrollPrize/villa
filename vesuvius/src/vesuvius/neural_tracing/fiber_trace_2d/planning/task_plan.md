# Demand-scoped GPU lease

## Startup follow-up
Defer manager bootstrap and the first Supervisor tick/registration until an input read is ready for
GPU dispatch. Before that, hold no grant and launch no GPU worker. Closing an
unused lease must not register a task. Retain existing idle release and pause
semantics after activation; no protocol changes. Test delayed reads, first
ready tile and close-before-demand with the approved CPU/live test caller.
Update specs, demand GPU docs and task log with this startup contract.

## Goal follow-up: cached CPU validation

After exact-repeat verification, finish the seven family refinements not
completed in the interrupted report: plain mean LRU own/transfer seeds,
windowed anchors own/transfer seeds, and normalized mean novelty/plain/hit
variants. Reuse the original saved axes/seeds unchanged. Run cached-only with
inference disabled, original controls, per-family repeats and best repeat.
Missing coverage is a real incomplete-test result, not permission to fill tiles
or silently skip a candidate. Keep the GPU sweep stopped.

Keep the GPU sweep stopped. Inspect saved unmatched failure regions, then
repeat LRU64 and support-softmin67 with `fill_missing=false` through the approved
experiment caller. Include original controls and duplicate softmin candidate;
compare exact replay hashes, not just counts. Missing coverage must fail closed,
never request GPUs. Preserve earlier interrupted search config in its committed
planning/embedding_anchor_boundaries_next.json artifact. Document results.

1. Stop sweep via approved caller; verify terminal zero grant.
2. Separate GPU lifetime from DemandWorkerPool CPU readers/accumulators.
   Gate GPU startup until authorization; drain submitted tiles before join;
   retain pending reads, accumulation and native tracing state.
3. Reuse manager Supervisor with pool cohort adapter. Lazy registration;
   self-pause after 30 idle seconds, unpause on demand; administrative pause
   remains authoritative. CPU tracing intentionally continues outside GPU grant.
4. Replace sweep-wide fixed reservation with shared demand-loop ownership.
   Cached-only variants reserve nothing. Preserve cache identity validation.
5. Native request waits must tolerate scheduling pause with heartbeat support,
   preserving disconnect/error and bounded coordinator failure detection.

## Spec update
Document 30s idle release, safe drain/join before acknowledgement, CPU continuation,
no admission while paused, locked manager wire protocol unchanged.

## Tests
Independent plan/implementation review. Approved cooperative test caller: fake
clock lifecycle tests, shared pool output equivalence across restarts, cleanup,
manager pause/reacquire, native heartbeat timeout. Small manager-granted GPU
smoke test where available; do not launch full sweep.

## Docs updates / changelog
Update tracing docs, status, task log and changelog. Previous boundary sweep is
interrupted; preserve partial report, do not label unrepeated winners final.
