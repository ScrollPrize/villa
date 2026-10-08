# Independent fiber normal orientation

- User requests regular sampled per-fiber sign sweep and signed constraint normals,
  with separate fiber sign initialized+1; cross-fiber alignment deferred by design.
- Plan: normal_orientation_plan.md. Existing raw database preserved; native second
  pass avoids re-extraction. Invalid samples remain explicit, not fabricated.
- Independent plan and implementation review approved; sampler/default spacing is
  native spacing (16 here), batch size is256 fibers rather than a hard sample cap.
- Native142tests passed. Pilot6,612fibers/19,889constraints:831,845samples,
  13,611/39,778endpoint signs flipped, zero invalids;1.05s sweep/1.52s second pass.
  All old non-normal columns and endpoint axes validated exactly; source preserved.
- Full second pass launched using approved run_classic_fiber_evaluation.py runner.
- Full pass completed222.89s wall (45.46sweep):512,698fibers,7,368,335unchanged
  constraints;60,386,279samples,3,797,156endpoint flips(25.7667%),zero invalid
  endpoints,5invalid regular samples skipped with prior orientation retained.
  Output oriented_normals_v1/full.afc.sqlite is11.42GiB. No cross-fiber alignment
  and no signed winding inference (both intentionally deferred by user scope).
- Independent137.08s full validation passed:all fiber/non-normal event columns
  exact, all14,736,670endpoint axes unchanged except sign;integrity/FKs pass,
  every fiber normal_sign=+1. Persistent oriented_normals_full_validation.json.
- Deviations/limitations: batching is256fibers rather than a hard sample cap;
  first valid sample anchors gauge if first point invalid;5invalid intermediate
  samples skipped (all endpoints valid), no guaranteed continuity across gaps or
  true>90degree between-sample turns. Regular samples never persisted.
- Build/run commands and schema in constraint_database.md; timings and counts in
  normal_orientation_results.md. Unrelated .pytest_tmp_private_hot/ untouched.
