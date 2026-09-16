# Current default refinement weights

The current tracing-selected refinement default is EMA999 update 22,000 from
`5scroll_json_eligible_refinement_128_sd1_16x8_ema_20260915_212757`.

It is the best refinement checkpoint measured so far on the established full
50-fiber benchmark: **99 restart episodes**, **0.014415 failures/mm**, and
**68.676 mm mean segment length** over 90 eligible CP-bounded/non-Lasagna spans,
180 directed traces, and 6867.627 directed mm. The checkpoint was selected by
tracing performance, not merely by its dense validation loss.

The machine-readable source of truth is
`fiber_trace_3d/configs/refinement_default_weights.json`. The canonical
refinement training config uses its `training_checkpoint` as
`training.initialize_from`, restoring both the selected model weights and the
sibling refined-fiber archive. This begins a new optimization run at update zero
with a fresh optimizer and fresh EMA tracks; it is deliberately distinct from
an exact `training.resume`, which requires a raw checkpoint. Inference should
use the immutable `inference_checkpoint`; both files have the same SHA256:

`67edd3a1bde162c7105ee37d849a0f0b60a9b840dd37fc7ff6e17e0c3a2e9742`

The associated refined-fiber state is `fibers_step_00022000.zip`, SHA256
`71421ef7b18b46ece142c87975993346d890628939ff2254c87d0137e62e9989`.
Refinement initialization requires the source checkpoint and this archive to
remain siblings; the frozen inference copy intentionally contains only model
weights.

The later run `20260916_062136` did not replace this default. Its best/current
EMA999 snapshots scored 114/109 restarts and 59.718/62.433 mm respectively.
See `docs/benchmarks/refinement_212757_ema999_best_current_20260916.md` and
`docs/benchmarks/refinement_062136_ema999_best_current_20260916.md`.
