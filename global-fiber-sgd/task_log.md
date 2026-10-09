# V direction task log

- H state committed3f5b443e8, including hash-pinned selected H solution manifest.
  Large outputs remain ignored data/.36tests passed before commit.
- User requests same direction/breakage pass for V, consistently up.
- Plan v_direction_plan.md: share H pipeline, axis prior, freeze H solution,
  V sign baseline and breakage sweep, audit and analogous visualization.
- Global U/V remains paused. No source/training/inference changes or installations.
- Independent plan/code review approved. Mechanical rename to shared
  fiber_direction.py/review.py first,25regressiontests passed before family changes.
- Added V local axial prior, V–V filter, explicit up/down and family-neutral
  direction aliases, hash-bound frozen H outputs and combined labels.
-30tests pass4.62s; tests corrected to tolerance1e-14 for quadrature support
  1+/-2e-16 (no production precision/math change). H prior/loss unchanged.
- Full V input preparation:237800Vfibers,1842020parallel events. Approved
  run_v_direction.json; sign-only baseline then4breakageweights in one GPU grant.
- Preparation72.24s,117091components including86312isolatedV; no degenerate or
  zero-dot events. Signed baseline528conflicts(0.028664%);16sGPU10ksteps.
- First two penalties .1/.07 converge77rejections/0conflicts,99.953354%Varc
  retained,1838768retainedevents. Independent full-data audit passes, including
  frozen H arrays/labels and up/down gauges. None of77is individually restorable
  in either sign without conflict; none isolated or unrelated to baseline conflicts.
- Full sweep done; all4penalties converge same hard labels/signs(77newly broken,
  zero conflicts). Choose strongest tested penalty.1. Baseline15.88s; four10kstep
  solves29.99/39.21/40.66/40.07s on sharedH100. These are not isolated speed tests.
-41tests pass7.01s including paused U/V regression; git diff --check clean.
  CPU-only run_v_direction_review.json generating full audit, recommendation and
  all-disabled +1000survivingV normal/curve/up-tangent views.
- Export completed:77full disabled curves,1000survivingV curves,3960outward-normal
  arrows and4000UPtangent arrows. Hash-pinned v_direction_recommended.json,
  visualization data/v_direction_v1/visualization_1000. Report in v_direction_results.md.
- Independent rawAFV resampling128events/256tangents: maximum difference2.97e-8
  versus cachedfloat32tangents. H signs/labels unchanged; up/down aliases verified.
- Independent final OBJ audit passed:77disabled curves1120vertices/1043edges;
  1000survivors25828vertices/24828edges; direction overlay101828vertices/72000faces.
  All4000arrows independently recomputed from rawAFV (maxdirectionerror2.22e-16),
  endpoints+256basevx in savedUPdirection, all hashes/indices/survivor masks valid.
  No remaining blockers. Global U/V still paused.
