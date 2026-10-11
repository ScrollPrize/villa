# Separate V equality and weak individual Z prior

- Independently approved uv_vertical_split_plan.md. User requests equal V at
  constraint points and separate weak per-fiber V≈CT-Z calibration.
- Removed manufactured target_v from preparation, pair APIs, joint/categorical
  tensor readers, density diagnostics, visualization energy reader and benchmark.
  Old cache files can retain extra target_v bytes; no consumer uses them and
  they need not be rebuilt. New caches do not generate the field.
- Pair metric now min(((vB-vA)/128)^2,100)*.01. Weight/scale/cap unchanged.
- Shared vertical_layout_prior: v_prior_weight=.01, v_prior_scale=1000.
  Uniform per-piece start/middle/end averaging, unconditional probability mass
  and full-pass normalization unchanged. Independent of angular prior_weight.
- No derivative or monotonicity constraint on Z. Existing signed-arclength
  V parameterization unchanged; physical Z may reverse along surface V.
- No solve rerun, moments implementation or historical output overwrite.
- Independent implementation review approved all producer/consumer changes.
- Initial suite found one oracle fixture that disabled prior_weight but not the
  newly independent v_prior_weight. Fixed the fixture to disable both as intended
  for its continuation-only comparison, without loosening tolerances.
- Test command: /home/hendrik/.venv_las/bin/python -m pytest -q --noconftest
  -o pythonpath=/home/hendrik/villa2/vesuvius/src followed by
  vesuvius/tests/test_fiber_uv_{vertical,distance_objectives,pair_backend,layout,
  anchor,categorical,density,sign_charge,full_pass,global_accumulation}.py.
- Final result:81passed in20.32s, existing sparse-invariant warning only.
  git diff --check clean. No target_v references remain in production fiber_uv*.
