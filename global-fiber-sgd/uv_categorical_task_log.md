# Categorical task log

- Previous log archived in uv_sign_charge_task_log.md.
- Completed previous full multiplier16:CCW28760violations,64.4886%arc;
  CW27968,64.2597%arc. Pilot16settle:9violations,63.5233%arc.
- New mode:full incident-event aggregation, detached targets, SGD p-target logits.
- Explicit limitation:first trial freezes geometry/widths to isolate categorical
  update; expected soft energy replaces mixture likelihood. Not joint refinement.
- Independent plan/code review passed. Hoisted mean degree; added expected-energy
  telemetry and full conditional-energy derivative regression including priors
  and continuation.65tests passed8.39s, legacy paths included.
- Aggressive8updates*step4 oscillates between widespread rejection and recovery.
  T=.25:final3288violations/93.6321%arc;T=1:2867/87.5463%arc. Not promoted.
- Revised1update*step1, T=.25. From original anchor state reaches0violations but
  ~51%arc at sweep60. From pilot4 state227violations/69.4125%arc reaches0 and
  71.8092%arc at sweep60. Shows initialization dependence, not a global guarantee.
- Full CCW/CW repair configs queued through manager,10full sweeps,starting from
  previous full fixed-charge16 results. No early stopping at first zero.
- Target temperature replaces previous positive entropy sharpening; no hidden
  duplicate entropy term. The full conditional energy is not previous mixture
  likelihood, so this is not a pure optimizer-only controlled comparison.
- Final pilot repair:0violations,71.9722%arc,16.6517%pieces rejected,120.13s.
  Original227:137corrected retained,90disabled; total rejected1890→1666.
  All last10sweeps0violations. Original-anchor damped52.7570%arc at0violations.
- Independent saved-array audit agrees. Frozen geometry exact.0/5continuations
  retained and max target discrepancy.936 remain limitations, clearly reported.
- First pilot new-module hash omission documented. Full execution now captures
  both module hashes. No retroactive guessed provenance.
- Future full categorical launches: CCW GPU0, CW GPU1, shared12GiB reservation
  each (down from24GiB). Observed combined GPU memory was14.4GiB for both jobs;
  12GiB/job keeps headroom above roughly7.2GiB average, not a measured per-job peak.
  Batch4096 unchanged. Current frozen configs and manager grants untouched;
  no restarts or new launches. JSON/device/budget validation and diff check pass.
- Full10sweeps completed:CCW28760→110sign errors(0.0048921%),arc64.4886→62.4748%,
  rejected21.1274→22.7383%,755.14s;CW27968→104(0.0046532%),arc64.2597→62.3828%,
  rejected21.2675→22.7887%,754.85s. Peak Torch6.1316GiB/job. Retainedlinks1/466
  and0/466 respectively. Final solver audits saved; independent full artifact
  report not yet rerun. No posthoc filtering or further solve launched.
