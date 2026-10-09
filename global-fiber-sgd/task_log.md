# Current task: H direction alignment

## Active follow-up: breakage penalty and visualization

- Baseline audit: 1,721 conflicting events over 269 pairs/444 fibers; 59 pairs
  internally contradictory, guaranteeing at least 424 unsatisfied events without
  rejections. No single fiber flip improves weighted cost.
- Implement optional whole-H broken state; tune penalties; export newly rejected
  fibers and surviving H subset with a separate tangent-arrow OBJ. U/V paused.
- Plan: h_direction_breakage_plan.md; independent review requested.
- Independent plan review approved: fixed original pair denominator, explicitly
  gated prior, joint ternary entropy, and isolated/non-conflict rejection audit.
- Optional sigmoid broken state implemented via shared sign pair loss and shared
  H/V ternary probability helper; earlier normal/HV formulas unchanged. Warm-start
  sign-only logits clipped to +/-4; directions remain optimizable, not frozen.
- Managed full-data sweep launched via approved run_h_direction_breakage.json,
  single shared 4GiB GPU grant across all four penalties. No global U/V jobs.
-25 regression tests passed (4.72s). .1 finishes176new rejections/8conflicts;
  .03 finishes185new rejections/0conflicts. Lower penalties still running.
- Reuse existing normal/curve exporter for surviving H; add all-new-disabled OBJ
  and separate same-subset CCW tangent-arrow OBJ. Original full curves, not clips.
- Independent audit found an avoidable rejection at .03 (ID139999, degree1538,
  no hard conflicts even if restored). Initial clipping of saved logits invented
  soft pair uncertainty, creating rejection pressure on high-degree fibers.
  Corrected to preserve saved logits exactly (still trainable). v1 is diagnostic,
  not promoted. v2 refines sweep to .1/.07/.05/.03 with unchanged confidence;
  export chosen v2 only. No post-hoc deletion/pruning or hard-mask repair.
- v2 finished: penalties .1/.07/.05/.03 give176/177/184/184new rejections and
  8/7/0/0retained conflicts. Chosen .05 retains99.912734319417%Harc and
  3594993/3603373events(99.767440117912%). .03 has identical hard mask/result.
 10ksteps each, sharedH10047.97/49.19/51.56/73.83s (not isolated perf measurement).
- Independent full-event audit passes .05:184new rejects, all earlier labels
  unchanged, exact CW opposite, ID139999retained. None of the184can individually
  be restored with EITHER tangent sign without introducing conflicts. This is a
  local minimality check, not proof of globally minimum rejection/physical error.
- Final regression25tests passed4.54s, git diff --check clean. CPU-only approved
  run_h_direction_breakage_review.json now auditing sources and exporting views.
- Export completed: data/h_direction_breakage_v2/visualization_1000 contains
  all184disabled full curves (disabled_h.obj + CSV),1000long spatially distributed
  surviving H curves,4973outward-normal arrows and5075CCW tangent arrows in a
  separate same-subset OBJ. Whole directory23MiB (includes duplicate standard
  renderer layers); old empty broken/V layers are NOT the new rejected output.
- Independent artifact audit passed: exact184disabled IDs,1000retained IDs,
  5075unit signed tangent arrows; all OBJindices/finite vertices and hashes valid.
  Disabled2548vertices/2364segments; survivor29599vertices/28599segments;
  direction-overlay126024vertices/91350trianglefaces. Every tip equals
  point+length*tangent and signs match the chosen solution. No remaining blockers.

## Earlier phase of this direction task

- U/V explicitly paused. Both previously launched trials are finished; no active
  global processes (pgrep verified). Prior notes preserved in uv_layout_task_log.md.
- Plan h_direction_plan.md; independent review before implementation.
- Direction-only stage: one sign per retained whole H fiber, no new rejections.
- Independent plan review passed: define zero-weight/degenerate handling, report
  disconnected gauges, preserve event IDs, CW hard signs by exact negation.
- Shared tangent_at delegates to batched tangents_at; shared normal-sign pair
  objective extracted without changing its formula. H pass uses same term with
  no rejection gates. No U/V code integration or reruns.
- Minor robustness guards: empty parallel graphs, positive finite sampling spans,
  isolation on nonzero-weight edges. Production had zero degenerate/zero-dot events.
- Full prep84.12s:251658Hfibers,3603373parallel events,82250components,
  58793isolatedH. Managed H10010000steps25.796s,274.65MiBpeak CUDA tensors.
- Source-forward weighted disagreement12.32525%; umbilicus-only0.698138%;
  optimized0.047735%(1721events).1612Hdirections changed vs prior(0.640552%).
  Zero new rejections. Numerically ambiguous(<.9chosenprob)0%;1217components have
  weak signed circulation evidence, so certainty is not anatomical validation.
-22tests pass. Independent reviewer recomputed all3603373outcomes, unchanged
  IDs/masks, component labels, exactCWnegation and256raw endpoint tangents(max
  difference2.98e-8). GPU job finished; no U/V work resumed.
- Full commands/config and limitations in h_direction.md; generated report and
  source-hash audit use approved run_h_direction_report.json runner.
