# Task Log: Native 3D Trace2CP GPU-Centric Beam Acceleration

## Plan

- Reuse the approved whole-fiber Trace2CP benchmark command after each change.
- Preserve current metric quality: the reference run should remain at 3
  restarts on the benchmark fiber unless a change is explicitly rejected.
- First target the measured hot path in point-field lookup: fewer GPU calls,
  one valid/output sampling pass, and one decode pass per query batch.
- Then reduce Python beam rebuild/synchronization overhead.
- Then move cached resident block routing off NumPy where practical.

## Deviations / Deferred Items

- Rejected experiments are listed below with measurements and were removed from
  the code path. No intentionally slower or quality-changing acceleration path
  remains enabled.

## Validation

- Baseline command:
  `PYTHONPATH=/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/volume-cartographer/build/python-bindings/python:/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/vesuvius/src:/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3 python -m vesuvius.neural_tracing.fiber_trace_3d.trace2cp_tool /home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/metric_sd2_s1_single.json --checkpoint /home/hendrik/business/aiconsulting/vesuviuschallenge/data/fiber/snapshots/s1a_128_2_single_8x8_20260727_161616/best_25_9k.pt --export-dir /tmp/trace2cp_sparse_default --fiber-json /home/hendrik/business/aiconsulting/vesuviuschallenge/data/train_fibers/fibers_test_paul_4/kb_20260605T150824406_000001.json --beam-lookahead-steps 2 --beam-width 8 --smoothness-normal-weight 0.1 --smoothness-tangent-weight 10.0 --core-margin-voxels 48 --inference-patch-shape-zyx 128 128 128 --inference-scaledown-power 2`
- Baseline result on current checkout:
  `restarts=3`, `err/kvx=0.2`, `err/m=19.6 (38.2mm)`,
  `trace_wall_s=202.528`, `trace_cpu_s=741.915`.
- Baseline top profile rows:
  `trace_candidate_score=148.877s`, `inference_forward=38.632s`,
  `field_sample_lookup=33.868s`, `trace_candidate_normals=19.253s`,
  `trace_current_sample=18.283s`.
- Batched field lookup result:
  `restarts=3`, `err/kvx=0.2`, `err/m=19.6 (38.2mm)`,
  `trace_wall_s=188.519`, `trace_cpu_s=725.442`.
  `field_sample_lookup` dropped to `23.484s` and `trace_current_sample`
  dropped to `14.502s`; new measured route component was
  `field_lookup_route=10.187s`.
- Carried endpoint-current beam state result:
  `restarts=3`, `err/kvx=0.2`, `err/m=19.6 (38.2mm)`,
  `trace_wall_s=178.775`, `trace_cpu_s=713.605`.
  `trace_current_sample` collapsed to `0.143s`, proving the duplicate
  current-point lookup was removed. `field_sample_lookup=12.470s` and
  `field_lookup_route=9.569s`.
- GPU resident block-bank route experiment:
  - First exact 3-vector match version: `trace_wall_s=181.388`.
  - Exact linear-key version: `trace_wall_s=181.033`.
  - Both kept `restarts=3`, but both were slower than the carried-state run.
    The extra resident prechecks and pointwise block-bank sampling outweighed
    the saved CPU route, so this path was removed rather than kept as dead or
    slower code.
- Parent-routed candidate lookup experiment:
  - Result: `restarts=3`, `trace_wall_s=244.104`.
  - Rejected and removed. It routed fewer CPU points, but duplicated block
    tensors per beam state; `field_sample_lookup` jumped to `139.970s`.
- GPU block-origin/group routing result:
  - Result: `restarts=3`, `err/kvx=0.2`, `err/m=19.6 (38.2mm)`,
    `trace_wall_s=105.795`, `trace_cpu_s=610.183`.
  - This keeps candidate points on torch tensors, computes block origins with
    torch, transfers only unique block origins to CPU for block inference, and
    performs per-block grouping with tensors. `field_lookup_route` dropped to
    `0.506s`; `field_lookup_origin=1.558s`; `field_sample_lookup=13.466s`.
- Reference-line model-block prefetch experiment:
  - Rejected and removed. It prefetched `763` blocks from `2336` densified
    reference-line points before tracing, but changed quality immediately
    (`restart` at segment 3 instead of segment 31). This indicates model block
    inference is not safe to reorder/batch differently for this checkpoint, so
    explicit broad prefetch is not kept.
- Torch-only beam prune selection:
  - Result: `restarts=3`, `err/kvx=0.2`, `err/m=19.6 (38.2mm)`,
    `trace_wall_s=104.255`, `trace_cpu_s=608.720`.
  - `trace_beam_prune` dropped from `3.345s` to `2.594s`; overall gain was
    small but positive, and behavior remained unchanged.
- Batched 3x3 eigensolve normal reconstruction:
  - Result: `restarts=3`, `err/kvx=0.2`, `err/m=19.6 (38.2mm)`,
    `trace_wall_s=101.806`, `trace_cpu_s=604.198`.
  - Kept by user direction. This does not interpolate raw compact `nx/ny`; it
    recovers the principal axis from the already-built sign-invariant local
    tensor and replaces the previous fixed power iteration.
- Cached inferred-block metadata tensors:
  - Result with the retained eigensolve normal path: `restarts=3`,
    `trace_wall_s=102.048`.
  - `field_sample_lookup` dropped from `13.300s` to `12.935s`, but total time
    was within noise. Kept provisionally because it is a direct field-lookup
    cleanup and does not change numerical semantics.
- Focused validation after the final code/test cleanup:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:lasagna:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
  passed with `159 passed, 2 skipped in 5.73s`.
- Syntax and whitespace validation:
  `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/trace2cp_tool.py`
  passed.
  `git diff --check -- ...` passed for the touched Trace2CP, test, and planning
  files.
- Test cleanup:
  - The principal-axis torch test now compares against the exact dominant
    eigenvector with the same hint sign handling, matching the current spec.
  - Fallback test caches without `sample_point_choices_torch` no longer carry
    endpoint samples from `sample_points_torch`; they resample current-point
    validity through the fallback path. Production multi-choice caches keep the
    carried endpoint-state optimization.
