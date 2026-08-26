# Plan: periodic flow reset with frozen-accumulator baking

## Goal

Extreme deformations are hard to optimise because the live flow field must
represent the entire accumulated warp. Periodically freeze the smooth part of
the transform into a fixed "accumulator" stage and reset the live parameters
to identity, so optimisation always works on a near-identity problem. The
total map is unchanged at the moment of reset (restructuring is exact by
construction), and per-step cost stays flat in the number of resets.

## Design summary

Current forward chain (canonical spiral -> scroll), built in
`SpiralAndTransform.get_slice_to_spiral_transform` (transforms.py:525):

```
[gap_expander, flip?, flow diffeo(s), varying_linear, umbilicus]
```

New chain:

```
[gap_expander, flip?, flow diffeo(s), varying_linear, ACCUM]
```

- **Gap expander + dr_per_winding: permanently live, never reset, never
  baked.** They stay innermost, in canonical winding coordinates, at native
  logit resolution. (Baking them into a grid would smooth the radial kinks at
  winding boundaries — the whole content of the gap stage.)
- **Flow(s) + linear: live, periodically reset to zero.** With
  `model_num_flow_stages > 1`, all live stages reset together.
- **ACCUM: frozen displacement-grid stage** representing the composition of
  every previously frozen `[flows, linear]` epoch plus the original umbilicus
  shear. Two grids (forward and inverse), sampled trilinearly; gradients flow
  through the input only, never through grid values. Before the first bake,
  ACCUM is structurally absent — the chain ends with the plain
  `UmbilicusTransform` (no identity grid, no extra grid_sample).
- **Exact history on CPU is the source of truth.** Every reset snapshots the
  exact `[flow, linear]` state_dicts (CPU). ACCUM is rebuilt *from scratch*
  through the exact composition each reset — never by resampling the old grid
  through the new epoch (grid-through-grid compounds interpolation error
  linearly with resets). Final export composes the exact chains.

This is "compose-everything": constraints stay in scroll space; nothing is
baked into the point/volume inputs; scroll-metric hyperparameters
(`dt_target_floating_threshold`, `epsilon=6.0` in
`get_radial_normal_in_scroll_space`, satisfaction thresholds, ...) keep their
meaning.

## Components

### 1. `transforms.py`

- New `FrozenDisplacementTransform(pyro Transform)`: holds `grid_fwd`,
  `grid_inv` ([Z,Y,X,3] displacement over a stated bounding box), `_call` /
  `_inverse` = trilinear sample + add. Grids are buffers (no grad), fp32.
- `SpiralAndTransform` gains:
  - `frozen_epochs`: list of CPU snapshots (flow + linear state_dicts, plus
    construction metadata). Serialised in `state_dict` or alongside it.
  - `accum_transform`: the current `FrozenDisplacementTransform` or `None`.
  - `rebake_accumulator(lattice_spec)`: evaluate the exact composition
    `umbilicus ∘ linear_1 ∘ flows_1 ∘ ... ∘ linear_k ∘ flows_k` (application
    order: most recent epoch innermost) at lattice points, both directions,
    chunked, under `no_grad`; build a fresh ACCUM. Log a round-trip /
    vs-exact probe error.
  - `reset_live_smooth_params_()`: zero all flow params and `linear_logits`
    in place. Gap logits and `dr_per_winding_logit` untouched.
  - `get_slice_to_spiral_transform` / `_get_transform_parts`: replace the
    trailing `umbilicus` with `accum_transform` once it exists.
- `ray_specialized_spiral_to_scroll` (transforms.py:363): the chain matcher
  hard-codes `[gap, flip?, diffeo, linear, umbilicus]` (and already rejects
  `num_flow_stages > 1`). Extend it to accept a trailing
  `FrozenDisplacementTransform` in place of / after the umbilicus; otherwise
  phase-bundle losses silently fall back to the slow generic path.

### 2. `fit_spiral.py` — reset driver

New `FitContext.bake_and_reset(iteration)`, called from `run()` between steps
on a new config key (e.g. `model_bake_reset_interval`):

1. Snapshot exact live `[flows, linear]` to CPU; append to `frozen_epochs`.
2. `rebake_accumulator(...)` from the full exact history.
3. `reset_live_smooth_params_()`.
4. **Clear optimiser state** for the reset params (delete
   `optimiser.state[p]` for flow + linear params) — stale Adam moments
   re-deform immediately otherwise. LR scheduler keeps running.
5. Invalidate derived caches: `theta_crossing_map.force_refresh(...)` (and
   re-run `_enforce_theta_liftability`), reset `dt_target_cache_manager`,
   drop the per-instance transform caches by rebuilding
   `self.slice_to_spiral_transform`.
6. Log probe error + a `bake_reset` metric.

Grid lattice: cover the flow box (`flow_min/max_corner_spiral_zyx`) at HR-flow
resolution or finer; resolution as a config key.

### 3. Checkpointing (`_checkpoint_payload` fit_spiral.py:2966, `apply_checkpoint`, `inspect_checkpoint`)

- Payload gains `frozen_epochs` (CPU state_dicts + bake iterations + lattice
  spec). Bump/gate schema so old code refuses new checkpoints cleanly.
- On load: restore history, `rebake_accumulator()` (deterministic given the
  history — grids need not be stored, but storing them saves a rebake; either
  is fine, pick one).
- `inspect_checkpoint` compares model state shapes — frozen history changes
  neither param shapes nor `CHECKPOINT_MODEL_SHAPE_KEYS`, but verify.

### 4. Export / preview / final composition

- `export_preview` (fit_spiral.py:3387), `flatten_spiral_checkpoint.py`,
  satisfaction overlays, preview rendering: all consume
  `get_slice_to_spiral_transform()` and work unchanged *if* that returns the
  ACCUM-composed chain. For final export, provide
  `get_slice_to_spiral_transform(exact_frozen=True)` that materialises the
  exact epoch chains instead of the grid, and decide (deliberately, once)
  whether export uses grid (matches what was optimised) or exact chains
  (true map; differs by grid error). Recommend exact + report probe error.

## Footguns

- **Grid-through-grid folding.** Never update ACCUM by resampling itself;
  always rebake from the exact CPU history. This is the single biggest
  silent-quality-loss risk.
- **Optimiser moments.** Forgetting step 4 destroys the exact-match property
  one step after reset.
- **Detached-leaf plumbing in `step()`** (fit_spiral.py:4066): the shared
  transform tensors / leaf-gradient flush only covers dr, linear logits, gap
  logits. ACCUM has no trainable params so it needs no leaf, but confirm the
  frozen grid tensors don't `requires_grad` and don't end up in
  `dist_grad_params` / `broadcast_model_params` with mismatched shapes across
  a resume (they're buffers — decide whether they're broadcast or rebaked
  per-rank; rebake is deterministic, broadcast is simpler).
- **DDP determinism.** Bake at the same iteration on every rank; params are
  identical post-allreduce so rebake is deterministic — keep chunking
  rank-independent and don't consume RNG.
- **Warm-up truncation.** `truncate_at_step` interpolates the live stages
  toward identity; only bake after warm-up is finished.
- **`ray_specialized` fallback is silent.** Add an assert/log when the
  matcher rejects the chain, or the phase losses get quietly slower.
- **Influence machinery** (`influence.py`): grad masks and
  `apply_masked_gap_decay_` are defined on the live flow lattices; after a
  reset the lattice's material meaning shifts by the frozen warp. Gap decay
  is unaffected (gap stays live). Flow masks: document as approximate, or
  forbid bake while an influence run is active (simplest).
- **Interactive sessions** (`incorporate_interactive_inputs`): inputs stay in
  scroll space, so no ingestion change needed — one of the payoffs of
  compose-everything. But `apply_config` / `rebuild_model_state` must carry
  `frozen_epochs` across a model-stage rebuild (add to
  `_MODEL_STAGE_ATTRIBUTES` handling deliberately).
- **Flow-box coverage.** ACCUM's inverse grid domain is scroll space, whose
  bounding box is *not* the flow box (umbilicus shear + warp move it).
  Derive the inverse-grid bounds from the data extent + margin, not from
  `flow_min/max_corner_spiral_zyx`.
- **Border behaviour.** Points sampled outside a grid's box must extrapolate
  sanely (border-clamped displacement, like the flow fields' padding);
  regularisation samples (`get_symmetric_dirichlet_loss`) roam the whole
  canonical box.
- **Permanent approximation.** Grid error is outside the optimisation and
  cannot be compensated by the live model. Make lattice resolution a config
  knob, measure a probe-set error at every bake, and alert if it grows.

## Validation

1. **Exactness at reset:** run N steps, `bake_and_reset`, and assert every
   active loss is unchanged (up to grid interpolation error; bitwise-close
   with an exact-chain ACCUM stand-in) before the next optimiser step.
2. **Grid fidelity:** probe points through ACCUM vs exact composition; both
   directions; assert < threshold (sub-voxel in scroll space).
3. **End-to-end equivalence:** tiny synthetic fit, resets on vs off,
   comparable final constraint satisfaction; composed export ≈ no-reset
   export.
4. **Checkpoint round-trip** through ≥2 bakes, including a resume that
   rebakes, on both single-GPU and DDP.
5. **Perf:** step time flat in number of resets; `ray_specialized` still
   matching (assert in test).
