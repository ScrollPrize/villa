# Task: Trace2CP Metric Command-Line Tool

Implement the `trace2cp` metric from `planning/plan.md` as a command-line
inspection tool, similar to the existing `--line-trace-vis` runner tool.

This task is limited to a runner/export tool for now. It must not wire the
metric into training, TensorBoard, snapshots, or evaluation loops.

Required behavior:

- Add `--trace2cp-vis`.
- Required args are `config`, `--checkpoint <snapshot>`, `--export-dir <dir>`,
  and `--sample-index <idx>`.
- `--sample-index` resolves through the same deterministic sample ordering as
  training, prefetch, augment-vis, dir-vis, and line-trace-vis.
- The sampled CP is the start CP.
- The target CP defaults to the next control point in the same fiber.
- Add `--trace2cp-target-offset`, default `1`, and
  `--trace2cp-target-cp-index` for absolute target override.
- Reject out-of-range target CPs and target CPs equal to the start CP.
- Load a side-strip segment between the two CPs, with enough margin for tracing
  and visualization.
- Segment loading must use Lasagna manifest normals and VC3D-equivalent
  side-strip/segment coordinate semantics. It must not use the neural-tracing
  3D crop loader or fabricate normals.
- Run the checkpointed direction model on the strip and decode the Lasagna
  ambiguous two-cos-channel direction output.
- Trace from the first CP toward the second CP. The initial direction must be
  aligned toward the target CP.
- Stop when the trace reaches/crosses the target CP x-column, enters the
  receptive-field border, sees invalid direction/validity data, or hits a max
  step cap.
- Score by y-distance at the target CP x-column:
  - linearly interpolate trace y at the target x-column;
  - raw error is `abs(trace_y_at_target_x - target_cp_y)`;
  - normalized score is clamped to `0..1`;
  - `0.0` means exact hit;
  - `1.0` means usable strip-edge error or failure before the target column.
- Run deterministic geometric TTAs on the strip. Per `planning/plan.md`, leave
  out y-shift and scale for trace2cp because those are hard to handle for long
  strips.
- Each TTA trace must be scored only after inverse-mapping its trace points back
  into the reference segment strip.
- Reuse `--line-trace-step`, `--line-trace-rf-margin`, and
  `--line-trace-tta-count`.
- The visualization must show the reference trace and TTA flock in reference
  segment-strip coordinates; `--med-tta` should also show the median-TTA trace.
- Print a concise score line containing sample index, fiber path, start/target
  CP indices, score, raw y-error pixels, target column, and trace status.
- Export `trace2cp_vis.jpg` and `trace2cp_summary.txt`.
