# Single fresh joint winding solve (archived)

- User explicitly requires ONE joint solve, strong winding sign penalties from
  start. No weak pre-stage, sign refinement, or categorical repair.
- Old staged processes already finished; pgrep found no active solver. No final
  categorical stage from that abandoned experiment was launched.
- New configs run_uv_joint151_strong_{ccw,cw}.json: fresh151, no warm_start,
  sign_charge_multiplier16, rejection_warmup0, geometry and widths trainable.
- Budget10event-pass equivalents =16,129updates of4096sampled events. This is
  minibatch sampling with replacement/importance weighting, not exhaustive sweeps.
- Existing40tests cover these implementations; config-only run change. Separate
  GPUs, shared24GiB reservations; approved command only.
- Independent plan/config review approved. Both runs launched. Strong sign charge
  applies from first update; existing entropy regularizer retains its within-run
  ramp, not a separate optimization stage.
- Saved-state audit now reconstructs fresh_initial_state when no checkpoint
  exists, rather than assuming a refinement.24solver tests passed again.
- Both GPU processes active ~97–98% utilization, ~11.5GiB allocator use each.
  Initial full-graph state retains100%length and has ~2.48Msign violations.
- User requests comparisons ONLY to earlier joint solves, excluding categorical
  runs. Saved joint_only_comparison.md with100/151 original and strong-joint rows,
  explicit budget/penalty/readout differences. Current3pass result retains38.41%
  length, violations0.0817%/0.0684%; not an overall improvement claim.
