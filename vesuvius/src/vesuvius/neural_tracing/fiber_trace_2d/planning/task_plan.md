# Single-backward normalization and reliable termination

## Approved scope (2026-09-12)
Reject embedding and ordinary multi-branch loss modes at training startup and
loss entrypoints with explicit TODOs. Retain their inference compatibility.
The objective redesigns in items 2 and 5 below are DEFERRED, not implemented.
Supported conditioned counts are (12*N_GT, 2*N_positive_samples, 2*batch_size).
Use the normalized accumulator for every supported bounded training mode; no
component-backward fallback. Independent review confirmed this scope.

Shutdown: remove dead-owner counter reads and queue feeder joins during loader
cleanup, use shared TERM/KILL deadlines, and install a separate client-owned
stop guard tracking explicitly registered compute roots only. Never traverse
the supervisor's entire tree (it includes the shared manager). Test blocked
cleanup, TERM-ignoring workers, and an unregistered unrelated process.

## Loss contract
No global denominator depends on any prediction. Prediction-dependent branch
selection or negative importance sampling is local to its own sample only.
All global counts are available from the full canonical batch's targets before
its first forward; GPU activations remain microbatch-bounded.

1. Direction (ordinary and conditioned): retain summed weighted errors and
   target-derived element count (6 per selected GT voxel, twice for conditioned
   positive queries). Selection may depend on that sample's predictions but
   does not change the global count.
2. Ordinary positive presence: compute each sample's existing mean over its
   nonempty selected branches, then average over samples containing GT targets.
   The global denominator is the target-derived number of GT-positive samples.
   For one branch this preserves the current objective. For multiple branches,
   it replaces global nonempty-branch weighting with equal positive-sample
   weighting. Explicit approval required: this is a loss-definition change,
   not merely an optimization. Keep per-sample branch selection/quota.
3. Ordinary negative presence: retain per-sample/per-branch negative means;
   branch count is fixed and negative eligibility is target-derived. No change.
4. Conditioned positive/negative presence: retain fixed two-query grouping and
   target-derived eligibility. No prediction-dependent global count required.
5. Embedding positives: retain configured half-pair count and GT trace sampling.
   Embedding negatives: count the configured pair budget for each geometrically
   eligible sample, independently of predicted presence mass. If eligible
   presence mass is zero, contribute zero numerator but retain that sample's
   target-derived denominator. No uniform fallback, no sampling zero-probability
   points. Geometrically empty candidate sets have zero target-derived budget.
   Explicit approval required: currently zero predicted-mass samples disappear
   from the denominator, reweighting every other sample.
6. Metrics may depend on predictions but remain detached diagnostics. They must
   not feed the training normalization, learning-rate decisions except explicitly
   configured schedules, or gradient weights.

## Implementation after approval
- Extract shared per-sample grouping/count helpers so training, evaluation,
  target prepass and accumulation use the same definitions; no copied voxelizer.
- Stage owned host inputs for one local macro batch, promptly releasing shared
  queue leases even when queue capacity is smaller than the macro batch.
- Reuse target materialization on compact geometry on the training device,
  inside CUDA preparation stream/pause synchronization. No image H2D, model
  forward or repeated augmentation in the count prepass.
- SUM-reduce a small fixed vector of all active counts before model forward.
- One combined weighted loss/backward per microbatch, one gradient SUM per
  optimizer update; remove component banks in all production modes.
- Preserve BF16/FP16 autocast, FP32 gradient accumulation, scaler/clipping,
  canonical sample membership, zero-work ranks and resize/pause state.
- Assert actual forward counts match precomputed counts before optimizer step.
- No automatic partial-mode fallback to repeated component backwards.

## Tests and measurements
- Prove another sample's changed predictions cannot change this sample's loss
  coefficient/gradient; exercise empty selected branches, no eligible negatives,
  zero predicted presence, unequal partitions, microbatch1 and zero-work ranks.
- Compare full-batch vs accumulated gradients/Adam updates for every supported
  mode with exactly one backward call per microbatch; include FP32/BF16/FP16.
- Shared training/evaluation normalization contracts, lease-reuse/cancellation
  tests and managed two-GPU resize/resume test.
- Matched manager-exclusive short performance A/B, reporting global and per-GPU
  throughput and mean/median/tail update times. No uncontrolled GPU use.

## Reliable termination (authorized separately)
- Live evidence: host root1200528 remains alive in futex_wait_queue; only manager,
  resource tracker and forkserver children were visible; GPU memory is empty.
  Need user's GDB stack to identify precise blocking cleanup site.
- Remove dead-owner lock acquisition, corrupted queue draining and unbounded
  feeder joins from forced cleanup; shared deadlines, TERM then KILL and reap.
- Explicit stop must have independent escalation authority even if the training
  root is blocked in native code. Never kill detached manager service descendants.
- Never report release while owned live processes remain; D-state survivors get
  explicit failure reporting. Zombies are not live compute, but must be reaped.
- Fault-injection tests for held locks, blocked feeders, TERM-ignoring children
  and repeated stop; keep tests CPU-only and short.

## Spec update
Replace production component-backward banks with target-only prepass/single
backward for supported ordinary/conditioned modes. Embedding and ordinary
multi-branch losses raise explicit TODO errors; their objective redesign is
deferred and their inference remains supported. Manager protocol unchanged.

## Docs and changelog updates
Document the shared loss path, stop guard/session contract and limitations in
cooperative_training.md and gpu_manager/README.md. Record exact validation argv,
model-only benchmark methodology/results, failed live attachment and recovery in
task_log.md; add a concise changelog entry. No end-to-end throughput claim.
