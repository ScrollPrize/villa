# Fixed-crop test refinement and EMA logging

Commit previous state first. Refine each deterministic test crop independently,
without whole-test-fiber updates. Freeze crop selection/transforms on original GT.
Persist crop-local curves across evaluations and snapshots. Log identical raw/EMA
component scalars. Restart from original non-refinement run
5scroll_json_eligible_128_sd1_16x8_ema_20260914_105436 at update11800,
retaining optimizer/EMA but starting fresh refinement state.

Correction: independent test pieces must NOT use border taper or pinned endpoints.
Retain the uniform 1/8-ellipse per-step cap. Stop current run and restart from the
original non-refinement update11800 after validation. Training taper stays unchanged.
