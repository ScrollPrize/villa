# Refinement default selection and 062136 evaluation log

Committed preceding merge/split/fiberlet work as `54e869a50`.

Evaluated run `5scroll_json_eligible_refinement_128_sd1_16x8_ema_20260916_062136`
on the established full original-GT benchmark:50 fibers,90 eligible spans,
180 directions,6867.627253 directedmm, ordinary tracing/no embeddings/no render.

- Best EMA999u21300,2726400samples,SHA `6e2d7ff...`:114 restarts,
  59.718498mm mean,10.113452median,0.204171p05,90.134232p95.
- Current EMA999u32700,4185600samples,SHA `3c13616a...`:109 restarts,
  62.432975mm mean,10.282471median,0.241037p05,101.640998p95.

Both summaries completed all180 directions and runner provenance validation.
Inference/native times were341.550/297.995s and344.026/300.340s. Current beats
this run's validation-selected best, but neither beats the existing refinement
leader EMA999u22000 from run `20260915_212757`:99 restarts,68.676273mm mean.

Promoted that actual tracing leader as the canonical refinement default. Independent
review caught that an EMA artifact cannot be used as `training.resume`: it has no
optimizer or EMA history and the existing guard correctly rejects it. Added explicit
`training.initialize_from` semantics instead. They load only the selected model
weights, restore sibling `fibers_step_00022000.zip`, and start update/sample counters,
optimizer, and EMA tracks fresh. CLI/config resume remains exact continuation and
overrides initialization; simultaneous configured resume/initialize is rejected. Added
`refinement_default_weights.json` with source/frozen paths, SHA256 identities,
checkpoint metadata and full-evaluation result. Model SHA is `67edd3a1...`; state
archive SHA is `71421ef7...`. No binary checkpoint was copied into Git: this repo
has no model-artifact/LFS mechanism, and the manifest pins the existing immutable
211MiB artifact without bloating or making the Git repository unpushable.

Detailed reports: `docs/refinement_default_weights.md` and
`docs/benchmarks/refinement_062136_ema999_best_current_20260916.md`.

Validation:

- Approved evaluation runner `validate-full` passed for both new `062136`
  best/current configs.
- Manifest/config/checkpoint/archive audit passed: both model copies match SHA
  `67edd3a1...`; the geometry archive matches SHA `71421ef7...`, passes ZIP CRC,
  and checkpoint metadata matches update22000/2816000samples/EMA0.999.
- Approved cooperative test runner reached 17passed/2skipped; its sole failure
  was the existing distributed Gloo test because this sandbox prohibits creating
  its loopback TCP transport. The same three multiprocessing tests passed outside
  that restricted runner environment. Broader affected Python tests completed
  53passed/3skipped. Focused new initialization tests passed.
- Independent reviewer recomputed both 180-direction summaries/per-scroll values,
  loaded the selected weights into the current architecture, and approved the
  corrected initialization semantics. `git diff --check` passed.
