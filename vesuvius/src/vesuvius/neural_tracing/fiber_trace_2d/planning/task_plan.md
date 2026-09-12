# Reproduction plan

1. Commit tested prerequisites; recover historical checkpoint configs, TensorBoard scalar sample axes and200k–300k old-test angle. Inspect historical source for sampling/voxelization/metric definitions. Never confuse training and test metrics.
2. Implement approved configurable experiment launcher using existing manager exclusive command/watchdog, not copied process ownership. inspect/start/status/stop/summarize. Immutable configs/source records/logs per trial; no CUDA during CPU inspection. Test shutdown and limit validation.
3. Modern fresh-weight control on exact historical deterministic CP test set/metric. Explicit test-only sampling/target overrides must not alter training. Reuse loader/model/loss, add experimental switches only as required; preserve production defaults.
4. Sequential300k-sample trials: SyncBN; stride4 raw outputs upsampled trilinear/nearest for full-resolution loss; full decoder dense then points-only NML; global64/LR.01 and128/LR.02; fixed on/off blur; historical mask. SyncBN uses physical optimizer batch, no gradient accumulation. Record all200k–300k evals, not cherry-picked minima.
5. Stop stacking on recovery; removal ablations against successful configuration identify critical differences. Repeat seed/matched pairs when ambiguous. Preserve named stage branches/commits. If no recovery exhaust requested options and evidence-driven followups.

## Testing
Independent review accepted plan with prerequisites: freeze logical test batch8,
CP seed/order/full-resolution target radius and equal batch-row aggregation,
including final partial batch. SyncBN must use real physical global batch under
exclusive manager lease. Launcher reuses tree-aware ExclusiveCommand and tracks
torchrun child sessions in independent StopGuard. No two-GPU accumulation proxy.
Compare6400-sample evaluation intervals for BS64/128; run to300800 samples (whole
evaluation boundary). Preserve parent-stage ancestry and exact delta. If needed
also inspect old local-rank versus new global normalization weighting.

CPU evaluation membership/voxelization and switch regressions; manager-exclusive CUDA smoke and real trials through approved fixed launcher. Record code/config/input identities, seed, GPUs, physical/global batch, sample/update counts, angle/loss and throughput; failures count as failures, not measurements. Validate old metric at equal sample counts.

## Spec update
Experimental reproduction configs may change norm, output/loss grid, dense/point targets, batch/LR, blur and masks. Production defaults and locked manager protocol unchanged. SyncBN fixed exclusive runs do not claim resize/accumulation equivalence.

## Docs updates
Brief task_log/status updates per stage; final reproduction documentation with commands/results and changelog. Previous task preserved in3a4a7b2c1. Independent review required before implementation.
