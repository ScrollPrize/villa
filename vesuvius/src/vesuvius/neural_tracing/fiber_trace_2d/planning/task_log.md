# Fiber training reproduction log

- 2026-09-12: Prerequisites committed: cc80cf86b manager guard;3a4a7b2c1 single backward/training cleanup. Previous task log preserved there. Branch fiber-train-reproduce. No reproduction trials yet.
- Approved /home/hendrik/villa/vesuvius/scripts/run_fiber3d_experiment (mutable control) and git add/commit/switch/stash/merge. GPU trials manager-exclusive. No protocol changes.
- Independent review: freeze test logical batch8/CP ordering/row aggregation;
  fixed physical batch for SyncBN; verify separate torchrun sessions on cleanup.
- Historical Jul29 TB uses update axis,globalbatch64. Error17.51584deg at204800
  samples,18.29189at217600,19.93559at288000; intervening values17.5–39.6degrees.
  Do not interpret a single crossing as a stable<20degree window. Jul29/30 config
  and all scalars extracted CPU-only to reproduction/*.checkpoint/scalars.json.
- Modern control retains GN,stride4,modern aug,arclength training; test overrides
  CP-only JSON,control_points,batch8,full-resolution targets with trilinearly
  upsampled predictions. This changes evaluation only,not training supervision.
- CPU run20260912_101634_356799:27passed20.89s (historical-grid/loader override,
  existing exclusive ownership and stopguard faults). Jul30 test config matches
  Jul29; only later-leg LR differs,not used for fresh initialization.
