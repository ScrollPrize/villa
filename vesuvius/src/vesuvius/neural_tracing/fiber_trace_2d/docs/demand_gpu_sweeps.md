# Cooperative GPU use in embedding sweeps

The sweep is CPU-driven. Cached tracing reserves no GPU. When missing inference
tiles appear, the shared demand pool starts CPU reads. Only when the first read
is ready does it bootstrap/register with the manager and request a GPU grant
through the existing Supervisor. Slow initial reads hold no GPU reservation;
closing before first demand never registers a task. GPU children are gated until the
manager records their identities and authorizes the grant.

- **Pause:** stop dispatching tiles, finish submitted tiles, join GPU children,
  then acknowledge release. CPU tracing, readers, accumulation and publication
  continue; pending tile requests retain their state and wait for resources.
- **Idle:** after 30 seconds without submitted/completed GPU tile work, release
  GPU memory/reservation using self-pause. A ready input tile clears self-pause
  and requests resources again. Unfinished slow reads do not cause churn.
- **Administrative pause:** independent of idle self-pause; new tiles cannot
  clear it. Resuming/reallocation recreates GPU children, not CPU trace state.
- **Accounting:** continuing CPU work is intentionally outside the GPU-scoped
  reservation while paused. The locked protocol still couples CPU/GPU grants;
  this does not claim separate manager-accounted CPU ownership during pause.
- **Failures:** manager disconnection drains GPU work through Supervisor;
  no new authorization while disconnected. Errors close workers/buffers before
  release; failed reconciliation leaves the server's ownership conservative.
- **Native wait:** heartbeat proves coordinator liveness, never chunk readiness.
  A scheduling pause can exceed the old request timeout; absent heartbeats or
  disconnected/erroring coordinator still fail. Updated native binary required.

Existing `gpu_manager` sweep settings supply priority, devices (default `0`),
CPU count and measured VRAM budget. Capabilities are pausable/adaptable; no
manager protocol changes or outer whole-sweep reservation are needed.

Approved experiment caller:

```bash
/home/hendrik/villa/vesuvius/scripts/run_fiber3d_experiment embedding-trace-sweep
/home/hendrik/villa/vesuvius/scripts/run_fiber3d_experiment embedding-stop-sweep
```

The interrupted boundary sweep was not restarted. Its partial report is retained
at `/ephemeral/hendrik/fiber3d_train2/evaluations/embedding_ema99_anchor_boundaries_20260914_v1/report.json`.
Use a new output directory for another sweep; this change does not add sweep
search-state resume. Completed cached tile output is retained and reusable.

`inference_compatibility.py` permits only the exact audited old/new source hashes
for this scheduling-only change. All other model, runtime, source and numerical
implementation hashes must match. No cache metadata is rewritten or bypassed.

Tests use the existing `run_fiber3d_cooperative_tests` caller. Suites
`demand-gpu-cpu` and `demand-gpu-live` cover queued-state preservation, output
equivalence, idle/reacquire and administrative pause. The GPU smoke test owns
its own small nonexclusive lease; it never bypasses the manager or wraps that
lease in an exclusive reservation.
