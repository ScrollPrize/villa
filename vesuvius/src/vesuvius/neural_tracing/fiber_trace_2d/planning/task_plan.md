# Manager merge and exclusive evaluation plan

1. Checkpoint current native/shared/Fiber work in scoped commits (done). Concurrency time series remains partially implemented and explicitly paused in its archived doc.
2. Merge fiber-training-manager. Keep both changelog histories/specs. Resolve train.py by combining sample-based progress/checkpoint metadata with shared optimizer/scaler restoration, RAM handoff, bounded accumulation, resumable tests and consolidated snapshot helper. Carry samples_seen through handoff. Keep README diagnostics. Replace conflicting active planning docs for this task; historical branch commits and cooperative docs retain prior findings.
3. Independent review of merge and integration plan. GPU manager playbook: Python3.11+Unix sockets/psutil, pytest synthetic inventory; locked protocol unchanged, no GPU allocations or daemon startup in tests. Native merge uses existing build/tests; absent root-playbook dependency script means no installs.
4. Add reusable CPU-only exclusive command launcher using existing Client/process identity helpers. Fixed non-pausable/non-adaptable request, priority100 by default, explicit allowed GPU UUIDs. Wait for initialize grant, require requested devices, preserve physical-to-visible device mapping. Launch a gated child in an owned process group, report its PID before allowing exec. No Torch/CUDA before grant. Poll/reconnect with retained token; non-pausable grant must stay identical. On completion/cancel/error terminate/join any remaining owned group processes before pause/released-zero/finish. Never release on uncertain cleanup; no protocol fields or hard enforcement of other clients.
5. Integrate live evaluator CLI and classic pilot/full/profile launcher using the reusable command wrapper; unmanaged explicit opt-out for compatibility/tests. Enable manager by default in classic evaluation config at priority100; missing service fails closed, no auto-start/install. Direct module use exposes manager socket/priority options/config. UUID inventory discovery uses nvidia-smi, respects CUDA_VISIBLE_DEVICES and passes only granted UUIDs into child with local0..N-1 ordinals. CPU and VRAM budgets derived from requested managed inventory; partial grants rejected before CUDA.
6. Synthetic manager/client tests for low-priority donor release barrier, no child work before applied, normal/failure/cancellation release after process cleanup, manager loss/reconnect behavior and nonpausable blockers. Merge regressions for sample-progress and cooperative state/training. No real training preemption or full evaluation restart without a separate run request.

## Spec update

High-priority exclusive fixed evaluation lifecycle, fail-closed service requirement, ownership/cleanup guarantees and cooperative-only limits. Sample-based progress survives RAM handoff.

## Docs updates

Merge conflict/resolution report; evaluation command/config docs and synthetic validation; current task log/status. Preserve paused queue task for later.

## Changelog update

Checkpoint hashes, merge and exclusive tracer integration; no unmeasured throughput claims.
