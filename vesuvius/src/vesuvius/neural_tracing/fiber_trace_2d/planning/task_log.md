# Three EMA evaluation log

- Final independent review verified360per-scroll cells/highlights and12macro
  scores: best9996.7422270083, +5.5312%vspriorbest99. Both100direction raw summaries
  reproduce audits exactly. Same original training task901af56b observedrunning
  generation13 in its active log after final grant; no stop/relaunch.
  Final CPU regression suite20passed; git diff --check clean.
- Follow-up best999 completed100directions/audit/exit0. Score6.742mm, new largest
  equal-scroll mean, vs newbest99 6.426 and priorbest99 6.389. Both entriesdone;
  original root alive. Full cumulative report saved in docs/benchmarks and
  intermediate/final reports retained under comparisonroot. No training/model
  changes. First report independently verified all330cells/highlights and score.
- Follow-up best99 retry completed100directions, audit passed, exit0; equal-scroll
  mean6.426mm vs priorbest99 6.389. Original training root alive. Cumulative
  comparison reported before launching best999; full distributions preserved in
  comparison_after_best99.md. Retry/cache changes independently reviewed: no blocker.
- First pair best99 attempt failed before inference: prediction-cache namespace
  used leaf directory best99 and collided with the prior series. Fixed shared
  invocation to derive a safe namespace from the full resolved output path.
  Added explicit failed-entry retry into a fresh retryN output, retaining frozen
  selection and failed logs; normal next still refuses failure. No state reset,
  deleted cache, refreeze or training restart. CPU suite20passed1.65s; retry launched.
- Follow-up pair frozen before evaluation: best99 update14700/1881600samples,
  loss0.373057779963, SHA256 dffd46a25b0ecca7f54373736b412cfbb92119f5c26c936ef7e3d968b4d99abf;
  best999 update15000/1920000samples, loss0.380881538767,
  SHA256 a55fd90653780ba7e37b3e838c1186bb62ca1e7884d4ada94de18a941eabf409.
  Fresh output combined_best_pair_20260912_19m; 19 CPU tests passed in2.01s.
  Independent review passed; report wording distinguishes later prior EMA rows
  from the600k comparisons. Best99 managed evaluation launched first.

- Follow-up requested: new current-state best99 then best999, original training
  still active near1.9Msamples. Explained per-track selection by aggregate dense
  test loss_total, not per metric or tracing score. Configurable validated variant
  subset reuses existing freeze/eval/audit/report path; no lifecycle changes.

- Active combined launch164113/root1975660, latest observed11000updates/1408000samples. Run compare_5scroll_s1a_128_sd1_16x8_ema_500k_20260912_164117.
- Requested currentEMA999, bestEMA999, bestEMA99 in order, reports after each, training stays alive; manager pauses only.
- Previous equal-scroll report committed82926d946 before new task.
- Plan reviewed independently: no blocker. Shared frozen-evaluation helper used
  by old selection and new exact-EMA series; no training lifecycle calls.
- CPU tests13passed2.07s. Freeze succeeded for all three beforefirst evaluation:
  current99911300/1446400samples/testloss.3861950273/hash3e87efbc;
  best9999800/1254400/.3855587457/hash7a4ac502;
  best99 at9800/1254400/.3819212223/hashab66f474.
  Full hashes/provenance in combined_three_ema_20260912/series.json.
- First managed evaluation launched with approved trace-series-next command.
- Current999 completed100directions/audit/exit0; equal-scroll mean6.240mm (rounded).
  Original training root verifiedalive. Full cumulative per-scroll table reported
  immediately; saved comparison_after_current999.md. Best999 launched next.
- Best999 completed100directions/audit/exit0; equal-scroll mean6.180mm (rounded).
  Original training root verifiedalive. Full cumulative comparison reported before
  launching best99. Best999 improves0139/Paris4 vs current999, loses1299/1451.
- Best99 completed100directions/audit/exit0, score6.389mm (rounded), narrowly above
  JSONEMA6.308. All three entries statusdone and originalrootalive. Training was
  observed running after eval1/2 under the same task901af56b and root1975660;
  final manager grant generation7 returned to training. No training signal sent.
- Final report preserved in docs/benchmarks/combined_three_ema_20260912.md.
  CPU tests rerun13passed1.61s; git diff --check passed. Each evaluation used
  approved run_fiber3d_experiment trace-series-next, exact frozen provenance,
  existing prepare/validate-full/full and independent raw100direction audit.
- Final resume verified: same task/root runninggeneration7, update11500,
  1472000samples, LR.01. No training lifecycle changes or deviations.
- Independent final review verified all300 per-scroll cells, every highlighted
  best/tie, all macro scores and requested checkpoint order/steps: no discrepancy.
- Independent implementation review passed; wording clarified that best checkpoint
  age is given by provenance, not necessarily the latest training step. Training
  manager task901af56b released its grant and is waiting; original root kept alive.
