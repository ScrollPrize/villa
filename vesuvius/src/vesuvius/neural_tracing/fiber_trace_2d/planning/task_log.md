# Three EMA evaluation log

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
