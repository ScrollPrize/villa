# Plan B: bake constraints into canonical space at each reset

Alternative to `reset_plan_a_frozen_accumulator.md` (compose-everything +
frozen accumulator).
Here nothing frozen runs during training: at each reset, every input is pushed
through the frozen inverse transform into (near-)canonical spiral space, the
whole live transform resets, and the frozen chains are kept only as CPU
snapshots, composed once at the end. SDT / phase-bundle losses are **out of
scope** for runs using resets.

## Design summary

At reset time, with current chain `F: canonical -> scroll` =
`Compose([gap, flip?, flows, linear, umbilicus])`:

1. Snapshot `F`'s exact state to CPU (state_dicts + `dr_per_winding` value +
   construction args); append to `frozen_epochs`.
2. Bake every constraint: `c' = F.inv(c)` (chunked, `no_grad`, identical on
   all ranks). Constraints now sit on/near the canonical spiral at uniform
   spacing `dr`.
3. Reset live params: flow fields -> 0, `linear_logits` -> 0, gap logits -> 0,
   umbilicus replaced by **identity** (the baked inputs are already
   z-axis-centred). **`dr_per_winding_logit` is preserved** — baked radii
   encode windings at the current dr; reinitialising it shifts every residual.
4. Continue fitting. Repeat. Final map = `F_1 ∘ F_2 ∘ ... ∘ F_live`
   (earliest epoch outermost / nearest scroll), materialised from the CPU
   snapshots for export.

Exactness: radius/winding losses compare `F.inv(c)` to the canonical spiral,
so baked constraints under the reset (identity) transform reproduce the
pre-reset residuals — up to the RK4 forward/inverse inconsistency, which is
committed **permanently into the constraints once per bake** and accumulates
across bakes. Log a probe round-trip error at every bake.

Metric caveat (accept deliberately or don't use this plan): all residuals,
thresholds and kernel widths expressed in scroll voxels
(`dt_target_floating_threshold`, huber deltas, satisfaction thresholds,
`epsilon=6.0` finite-difference steps, min-spacing targets) are silently
re-expressed in baked-space units, distorted by the local stretch of the
frozen map — most where deformation was most extreme.

## Input inventory and treatment

Point inputs — bake in place, keep a pristine scroll-space CPU copy for
visualisation/metrics (see footguns):

| Input | Where | Notes |
|---|---|---|
| Verified patches | `PatchAtlas` (fit_spiral.py:358): materialised `vertex_zyxs` + CPU payloads | Bake both copies; `rebuild_sampling_atlas()`; satisfaction atlas invalid |
| Unverified patches | `unverified_patch_atlas` | Same |
| Unattached PCL strips | `unattached_pcl_strips` + cached flat bundle (`get_or_build_unattached_pcl_flat`, fit_spiral.py:780) | Bake strips, drop the cached flat bundle |
| Cross-patch PCLs | `cross_patch_pcls` + `pcl_sampling_strata` | Bake points; strata are index-based, likely survive — verify |
| Tracks | `prepared_main_tracks` device tables **and** `flat_zyx_cpu` (fit_spiral.py:2839) | Bake all coordinate tensors coherently; `self.tracks` may already be released (`release_setup_only_tracks`) — bake the prepared tables, not the source |
| Umbilicus | `umbilicus_zyx` | Becomes the z-axis: replace with zeros in yx |
| Shell radius pool | `shell_valid_zyxs_gpu` | Bake |
| Shell polar map | `ShellPolarMap` (fit_spiral.py:240) | Rebuild from baked shell patch with an origin-centred (identity-umbilicus) polar frame |
| Influence anchors | `influence.py` | Bake anchors; see footguns for masks |

Derived caches — invalidate/rebuild after every bake: `theta_crossing_map`
(force_refresh + `_enforce_theta_liftability`), `dt_target_cache_manager`,
patch sampling/satisfaction atlases, PCL flat bundle,
`self.slice_to_spiral_transform`. Also clear optimiser state for all reset
params (flows, linear, gap; keep dr's or clear — decide once).

## Disproportionate-complexity assessment

Assessed each remaining input for whether "push it through the diffeo" is
proportionate:

**1. Lasagna volume (dense normals + spacing) — DISPROPORTIONATE; recommend
out of scope alongside SDT.** Three reasons, each sufficient:
- It is on-disk zarr stores (`normal_nx`/`normal_ny`/`grad_mag`,
  `lasagna_data.ensure_fit_sparse_stores`) with sidecar sparse-CUDA GPU
  caches. Warp-resampling means generating new zarr stores + sidecars +
  rebuilding the resident caches **per reset** — an offline-pipeline-scale
  job inside the training loop.
- The nx/ny channels are **direction components**, not scalars: warping them
  requires rotating by the local Jacobian (covariant transform), and
  `grad_mag` rescales by local stretch. This is per-voxel Jacobian
  evaluation over the whole store, and the anisotropic rescale has no exact
  scalar form.
- Interpolation blur compounds unless always resampled from the original
  through the full composed stack (k-chain evaluation over the whole volume
  per reset).
If dense losses must coexist with resets, use the hybrid instead: push the
per-step dense sample points through the frozen chains (spiral->scroll) at
eval time — i.e. plan A's mechanism for exactly this input.

**2. Winding-inference store — bakeable and proportionate.** Targets are
winding *differences* (dimensionless, bake-invariant), and the store is a
finite GPU-resident crossing list (`WindingInferenceStore`: points are
`origin + t·step` per crossing), not a volume. Bake by materialising all
crossing points once (`[num_crossings, 3]`, a small multiple of what is
already resident) and pushing them through `F.inv` at each reset like any
other point input; `_materialize` gathers baked points instead of
reconstructing from the ray parametrisation. The cost amortizes: one pass
over the store per reset vs. per-step pushes forever, and each reset only
applies the newest inverse to already-baked points (point composition is
exact). Caveats from losing the straight-ray form: the z-eligibility pruning
(winding_supervision.py:129) assumes z linear in t — keep computing it from
scroll-space coordinates before baking; and the loss's z-slab validity mask
(winding_supervision.py:248) will then filter on baked z, the same metric
drift as every other threshold in this plan.

**3. Interactive sessions — high-friction, recommend gating.**
- `incorporate_interactive_inputs` ingests scroll-space inputs mid-run: every
  ingestion must replay the full frozen-inverse stack, and metric radii used
  at ingestion (`track_exclusion_radius`, trusted-geometry masking distances
  via `trusted_geometry_tree`) are applied in a distorted space.
- Influence grad masks are derived from scroll-space regions on the flow
  lattice; after a bake the lattice is a different material space — masks
  must be re-derived through the bake or influence runs forbidden across a
  bake boundary (simplest: refuse to bake while `influence_state.active`).
- VC3D preview/export round-trips coordinates continuously.
Recommend: v1 supports headless fits only; interactive support is its own
follow-up.

**4. Everything else is proportionate.** Patches/tracks/PCLs/umbilicus/shell
are plain coordinate tensors plus rebuildable derived structures. The
min-spacing barrier and `sym_dirichlet` become incremental (post-reset-only)
regularisers — semantic change to accept and document, not a code problem.
(If barrier blindness to composed gap collapse matters, the fix is plan A's
"keep gap live" structure, which this plan trades away for full resets.)

## Footguns

- **Coherence across copies.** Constraint coordinates live in many tensors
  (GPU tables, CPU flats, payloads, KD-trees, caches). One central
  `bake_all_inputs(transform)` registry, with an explicit list mirroring
  `_MODEL_STAGE_ATTRIBUTES`' style, so a missed holder is a visible omission.
  Mixing spaces fails silently — losses just get quietly wrong.
- **Inversion error is permanent and compounding.** Unlike plan A (where the
  exact history is the source of truth), each bake here rewrites the
  constraints through an inexact RK4 inverse. Pin bake-time integration
  steps/solver to the training values (the discrete map is the fitted ground
  truth); probe and log per-bake error; cap total resets or alert on drift.
- **dr must survive the reset; gap logits must not.** Baked points land at
  uniform current-dr spacing: zero gap logits reproduce that exactly, a
  reinitialised dr does not.
- **Umbilicus reset to identity, not to the original shear** — the baked
  inputs are already centred.
- **Optimiser moments** for reset params must be cleared or the exact match
  dies one step after reset.
- **Visualisation/metrics/exports assume scroll space**
  (`overlay_patches_on_slices`, `save_overlay_and_print_satisfaction`,
  preview extents, tifxyz export). Keep pristine scroll-space originals on
  CPU (patch geometry is GB-scale — budget it), and materialise the composed
  frozen stack on GPU for export/preview moments. Satisfaction thresholds
  should be evaluated against originals + composed transform, not in baked
  space.
- **Checkpoint/resume must replay bakes.** Host inputs always load from disk
  in scroll space; `apply_checkpoint`/`_build_model_state` must re-run the
  bake stack over them deterministically (or persist baked tensors — bigger
  checkpoints, pick one). Every host-input reload path (`apply_config` with
  path changes, model-stage rebuilds) must re-bake — audit
  `load_host_inputs` consumers.
- **Flow-domain bounds:** baked z can drift via the 3-D flow; assert baked
  constraints stay inside `flow_min/max_corner_spiral_zyx` margins.
  yx radius shrinks toward canonical, so that side gets safer.
- **DDP:** bake at the same iteration on all ranks, rank-independent
  chunking, no RNG consumption; params are identical post-allreduce so the
  bake is deterministic.
- **Warm-up:** never bake while `truncate_at_step` warm-up is active.
- **`ray_specialized_spiral_to_scroll`** keeps matching (chain shape is
  unchanged — that's a genuine advantage of this plan), *unless* the hybrid
  per-step frozen push for winding-inference/dense samples is bolted onto the
  transform instead of the loss — keep it in the loss.

## Validation

1. Loss invariance across a bake for every in-scope loss (spiral-space losses
   bitwise-close; DT within inversion tolerance).
2. Per-bake probe: round-trip `F(F.inv(p))` error on a fixed probe set;
   assert sub-voxel, watch accumulation across ≥3 bakes.
3. End-to-end: tiny synthetic fit, resets on/off, comparable satisfaction;
   composed export ≈ no-reset export.
4. Checkpoint round-trip through ≥2 bakes with a re-bake on load; DDP parity.
5. Negative tests: bake refused during warm-up, influence runs, and when
   SDT/lasagna losses are enabled.
