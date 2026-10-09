# Current task: global probabilistic U/V layout

- User approved100windings, bakedH/V and previous exclusions, >=3perpendicular
  events per resulting >=1cm piece; constraint points have no variables.
- Plan uv_layout_plan.md; independent review requested before implementation.
- Prior results preserved in hv_stage_results.md/changelog.md.
- Independent review completed: exact expected negative log mixture, family/angle
  separation, source-direction invariance, unsigned evidence, continuation and
  GPU budget specified in plan. Full pair costs computed only for each batch.
- First timing pilot10kfibers/105184events:200steps3.09s, superseded because review
  caught a capped likelihood, cancellation in continuation moments, and underflow
  of remote initial hypotheses. Removed likelihood cap, centered continuation
  moments, initial logit floor-14 and added missing event phase priors. The prior
  pilot was already launched when fixes landed; rerun in solve_corrected, preserve
  previous artifact only as superseded timing, NOT a valid method result.
- Gaussian log-mixture table bilinear step1/16 winding, tested absolute error<.006
  against direct logsumexp. Fraction adjustment limited±.25turn, fixed arc maps;
  these first-model restrictions documented. Rejection warmup freezes only its
  logit gradients, not normalized probability. Sign cost finite10, not hard proof.
- Corrected pilot1000steps~14s. Rejection logit-20 yielded negligible gradients
  relative to Adam epsilon: replace with exactly1%initial rejection probability
  for full experiment (and validated pilot). No claim of useful rejection from
  earlier timing pilot. Final benchmark audit will use full baseline/full result.
- Before full CUDA launch, tightened signed-distance handling: known-sign wrong
  side uses zero-boundary soft cost plus uncapped corrective sign hinge, not a
  reflected distance distribution. This is a finite surrogate for zero-density
  invalid support, explicitly not a mathematically hard truncated likelihood.
  Pilot_rejection01 predates this final change; retained for timing/rejection
  sanity only. Full config uses sign10; stronger sign100 comparison planned if
  baseline leaves violations. Loaded NPZ once to avoid repeated whole-array reads.
- Full preparation148.41s:489458fibers,489924pieces,454split originals,466links,
  6618825events,90.878%known signs. Full10000steps/4096batch~6.188sampled epochs.
- Second configuration sigma1/sign100 queued alongside sigma.5/sign10 to compare
  stronger ordering versus soft noisy distance, not an isolated single-variable
  ablation. Continuation normalization/physical scale unchanged in both.
- Review flags rejection-gated continuation: separately averaged rare links can
  encourage rejecting their endpoints. Retained-link counts added to final audit;
  future normalization-only comparison should hold all other settings fixed.

- Paused at user request before H-direction pass. Both full trials already
  finished; no active U/V processes. sigma.5/sign10:7.7979%pieces rejected,
  82.1471%arc retained,7.9776%retained known-sign errors,49.0268%ambiguous,
  403.38s. Independent all-event audit passed;59/466continuations retained.
  sigma1/sign100:12.7079%rejected,73.1419%arc retained,6.9248%sign errors,
  65.5273%ambiguous,405.19s; second independent audit/report deferred while paused.
  These are preliminary, not recommended global layouts. No further U/V work.
