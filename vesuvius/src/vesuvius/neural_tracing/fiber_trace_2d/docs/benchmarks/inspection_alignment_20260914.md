# Inspection cross-section alignment

## Cause and fix

The top/side ribbon rasters independently derived horizontal resolution from
their longest offset row. Curvature made their widths different; cross-sections
were placed using top coordinates, visibly drifting from the side strip above.
The shared renderer now derives one centerline-based count per path component
at two base voxels per pixel, explicitly reusing it for the side view. Component
gaps, panel crops and actual-point inverse mapping consequently agree.

A second issue mapped own-trace failure markers through reference arc, ambiguous
for reference-progress plateaus. Native own-trace bands now use the actual terminal
point, matching the breakoff section. Other trace overlays retain arc matching.
Reference and traced paths remain independent axes, since their lengths differ.

## Validation

- Approved workspace build: `bash /home/hendrik/villa2/vesuvius/scripts/run_fiber3d_native_build`.
- Regression: `bash /home/hendrik/villa2/vesuvius/scripts/run_fiber3d_native_replay_tests`.
- 23 native cases; 138 Python tests passed in44.35s. Curved multi-component
  synthetic test checks actual cyan band pixels against section centers in both
  traced views/modalities, including a plateau-reference case.
- Independent plan/code review passed; `git diff --check` clean.
- Real cases19 and60 visually inspected: both traced view restart bands align
  with the old-end and new-start sections. All180 trace JSON documents exactly
  match the preceding v1 run (including provenance).

## Regeneration

Same EMA99u5200 snapshot SHA
`ccf706ab68d51c76910a8134b6970c3289a1e2503320d74c8eb75fa1f74209a6`,
same inference namespace/model key and40/80/grace160 policy. No model or metric
changes. New output preserves old images for comparison:

`/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema99_u5200_vis_20260914_v2`

Reusable command (preceded by `freeze-snapshot` and `prepare` with same config):

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_eligible_ema99_u5200_vis_20260914_v2.json
```

Existing inference chunks are reused. A changed visualization sampling grid can
touch additional support chunks; these are inferred on demand, not a cache reset.

## Completed result

Exit0 in416.814738s:180/180 JPEG galleries,32,041 transverse sections and complete
50-fiber README/CSV/JSON index. All component top/side bounds identical; every
section center matches its strip column. Real case60 JPEG checks find actual cyan
R0-/R1- bands at section centers in CT/presence top/side rows.

18,346warm chunks reused;939tiles inferred and96new support chunks published.
All180 trace JSON documents and summary metrics unchanged:148reported restarts,
6867.627253mm directed length,mean46.091458mm,median8.202461mm,
P05.143040mm,P9588.437892mm. Old output remains unchanged for comparison.
