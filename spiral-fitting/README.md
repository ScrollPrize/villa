# Spiral fitting

Code and helpers to fit a canonical Archimedean spiral to deformed scrolls.
`spiral_service.py` hosts one persistent interactive fit session over HTTP for
the VC3D Spiral workspace; `fit_spiral.py` is the underlying fitter.

## Scroll specification (spiral-scroll.json)

`fit_spiral.py` requires a `spiral-scroll.json` file in the dataset root.
This is not covered by scrollprize.org/tutorial_spiral. Required keys:

- `schema_version` — must equal `1`.
- `name`, `voxel_size_um` — required, no validation beyond presence.
- `spiral_outward_sense` — must be `"CW"` or `"ACW"` (case-insensitive).
  No automated method determines this; it is read off the CT data by a
  person in VC3D, or computed from an already-fitted spiral.

Optional `paths` object for per-input overrides when a dataset's file names
don't match the catalog's conventional defaults (e.g. `tracks_dbm`).

Also optional, and easy to get wrong silently: `normal_zarr_group`
(default `"4"`) and `lasagna_scale` (default `4`) select which OME-Zarr
pyramid level the `normal_x`/`normal_y` lasagna stores are read at.
**`lasagna_scale` must equal the actual downsample factor of whichever
group you pick for this specific scroll's lasagna store** — read it from
the store's own `.zattrs` multiscales metadata, do not assume it from
another scroll's example or from a generic recommendation. A mismatched
value either silently reads the wrong-resolution normal maps (no error) or
throws `RuntimeError: lasagna z-ROI [...] is empty` if the mismatch is
large enough to push the requested z-range outside the (wrongly-scaled)
store bounds.

Example (PHerc0826, where group `"2"` is a 4x downsample for this scroll
specifically):

```json
{
  "schema_version": 1,
  "name": "PHerc0826",
  "voxel_size_um": 9.362,
  "spiral_outward_sense": "CW",
  "normal_zarr_group": "2",
  "lasagna_scale": 4,
  "paths": {
    "tracks_dbm": "tracks/PHerc0826_20250821151701_surface_m7_L0_th0.2.dbm"
  }
}
```

## Sweep runner output

`runners/run_sweep.py` prefixes each active fit's live `PROGRESS` and
every-200-step loss lines with its configuration name. Optimization progress
includes the average iteration rate for the current stage (`it/s`). Complete
combined stdout/stderr for every attempt remains available under
`<output>/.sweep/logs/<config>.log`.



## Flattening a fitted checkpoint

`flatten_spiral_checkpoint.py` is a standalone, one-shot exporter. It
reconstructs the combined surface from a fitted checkpoint, launches a private
Lasagna service, flattens with `flatten_fast_nofilter.json`, writes the final
TIFXYZ directory, and tears the service down even if the job fails or is
interrupted:

```sh
python flatten_spiral_checkpoint.py \
    /path/to/checkpoint_fitted.ckpt \
    /path/to/output.tifxyz
```

The checkpoint format does not embed the fixed umbilicus curve. The script
looks for `umbilicus.json` in the checkpoint's ancestors, in
`$SPIRAL_DATASET`, and in the standard local s1 dataset location. For other
layouts, pass `--umbilicus /path/to/umbilicus.json`. Use `--lasagna-dir` if
the Lasagna repository is not in its standard sibling or `~/villa` location.
An existing output path is never overwritten.

## Flow-gradient conditioning

These optional settings change how the flow lattices are optimized, without
adding loss terms or changing the model parameterization. The smoothing,
lazy-moment and shared-second-moment switches default to off; gradient
clipping defaults to disabled. The `optimizer_flow_*` settings apply at run
boundaries and are read every step. Older checkpoints backfill these defaults.

The step order is: DDP gradient averaging, NaN/Inf detection and replacement
with zero, optional clipping, optional smoothing, influence masks, then the
optimizer update. Sanitizing before smoothing prevents a single invalid entry
from contaminating its neighborhood. Influence masks constrain the processed
gradient after smoothing.

- `optimizer_flow_grad_smoothing` Gaussian-smooths each flow lattice's
  gradient. `optimizer_flow_grad_smoothing_sigma_voxels` sets the standard
  deviation in scroll-voxel units of the flow coordinate frame. Cartesian
  smoothing is isotropic. Cylindrical smoothing runs along z and periodically
  around each ring, independently for the local z/radial/tangential components.
  `optimizer_flow_grad_smoothing_across_sigma_voxels` optionally smooths across
  rings at matching angles; its default of 0 disables this pass. These are
  lattice directions approximating along/across-sheet directions, not measured
  sheet tangents or winding boundaries. Kernels are truncated at three sigma
  and renormalized at nonperiodic borders to preserve constant gradients.
- `optimizer_flow_grad_smoothing_low_res_sigma_voxels` overrides the coarse
  lattice's along-sheet width; 0 uses the same width as the fine lattice. With
  the default fine spacing of 16 voxels and sixfold coarse spacing, a width of
  32 voxels is 2 fine cells but only about 0.33 coarse cells. Very small widths
  become identity kernels. The fitter reports effective widths at startup
  when smoothing is enabled. The across-ring width has no separate coarse
  override and is ignored for Cartesian fields.
- `optimizer_flow_lazy_moments` updates moments and applies the gradient step
  only to entries whose **processed gradient is nonzero**. Smoothing can make
  an entry active even without a direct sample there. Other entries retain
  their moments; configured AdamW weight decay still applies everywhere.
  This preserves gradient-scale history through unsampled steps and suppresses
  momentum-only movement there, but retains stale history too. It does not
  prevent large relative steps on a previously untouched smoothing tail.
- `optimizer_flow_shared_second_moment` uses one denominator across all cells
  and vector components of each lattice's leading slab, independently for
  each flow stage and for the coarse/fine lattices. Per-entry Adam scaling can
  make even a smooth gradient produce nearly sign-sized steps, particularly
  on first touch. A shared denominator preserves relative magnitudes and
  direction of the first moment before lazy masking and weight decay; it
  does not guarantee a smooth final displacement. The shared statistic is the
  mean of positive stored second moments, capped before averaging at
  `optimizer_flow_shared_second_moment_clip_quantile` (default 0.99). A value
  of 1 disables the cap. This limits outliers' effect on the common scale at
  the cost of local adaptivity. Full per-entry moments remain stored.
- `optimizer_flow_grad_clip_median_multiple` clips individual gradient
  components to this multiple of the median nonzero absolute component value,
  separately for each lattice slab. Zero disables it. Clipping before blur
  limits how far an extreme gradient can affect neighbors and optimizer
  moments, but can also suppress legitimate corrections and change vector
  direction. The median and second-moment cap are estimated from fixed-stride
  samples; the final averages and clipped fractions use the whole slab.
  Sampling is deterministic for identical inputs but may miss sparse support.
- `model_flow_field_low_res_lr_scale` multiplies the scheduled base learning
  rate for the coarse lattice independently of the fine lattice's existing
  ramp. It defaults to 1; 0 freezes coarse parameter values, including weight
  decay, although moments may still update. The fitter reads it every step,
  but the configuration catalog currently classifies it as a model-rebuild
  setting, like the other `model_` learning-rate controls.

`flow_grad_smoothing.py` provides Triton CUDA kernels and PyTorch reference/
fallback paths. `lazy_moment_adamw.LazyMomentAdamW` retains AdamW's state format
so its flags can be switched between runs without converting moment buffers.
For custom optimizer updates, diagnostics report the denominator, nonzero
update fraction, and per-component RMS over nonzero updates in voxel units.
These describe flow-parameter increments, excluding weight decay, not final
sheet displacement after integration. After both moment flags are disabled,
stored optimizer diagnostics can still describe the last custom step rather
than the current fused AdamW step. Clipping diagnostics report the bound
and fraction clipped. No full-scroll accuracy or throughput comparison is
established by the implementation tests.

### Rationale and references

Gaussian update smoothing has precedent in
[Vercauteren et al., *Diffeomorphic Demons*](https://www-sop.inria.fr/asclepios/Publications/Tom.Vercauteren/DiffeoDemons-NeuroImage08-Vercauteren.pdf).
Smooth velocity metrics are central to
[Beg et al., *Computing Large Deformation Metric Mappings*](https://www.cs.jhu.edu/~misha/ReadingSeminar/Papers/Beg05.pdf).
These motivate the preconditioning here; this discrete blur followed by Adam,
clipping and masking is not an implementation of classical Sobolev gradient
descent or LDDMM, and inherits no guarantee of the same fitted solution or
fold-free numerical integration. Direction-dependent smoothing has a related
motivation in [Pace et al.'s sliding-organ registration](https://pmc.ncbi.nlm.nih.gov/articles/PMC4112204/),
although this fitter uses cylindrical coordinates rather than detected sliding
interfaces.

Lazy moments follow the masked-update idea of
[PyTorch SparseAdam](https://docs.pytorch.org/docs/main/generated/torch.optim.SparseAdam.html),
using an explicit nonzero mask on dense gradients, a global per-parameter step
counter, AdamW's epsilon placement, and optional decoupled weight decay.
[Adam-mini](https://arxiv.org/abs/2406.16793) provides precedent for sharing
adaptive scales within parameter blocks; our grouping and winsorization are
custom choices, and retaining full moments does not provide its state-memory
saving. [Koloskova et al., *Revisiting Gradient Clipping*](https://proceedings.mlr.press/v202/koloskova23a/koloskova23a.pdf)
analyze clipping's stabilization and stochastic bias; their results do not
validate this particular sampled-median rule or its combination with Adam.

## Sobolev-damped Hessian-free flow step (prototype)

`optimizer_flow_sobolev_gn` replaces the AdamW update of the flow lattices
with a damped step `d` from `(H + lambda A) d = -g`, where `g` is the flow
gradient after DDP averaging, non-finite sanitizing and influence masks, `A`
is a Sobolev metric and `H` an optional Hessian-vector product (see
`sobolev_gauss_newton.py` and `SOBOLEV_GAUSS_NEWTON_PLAN.md`). Every other
parameter keeps its AdamW update. While enabled, the flow lattices bypass
gradient clipping, Gaussian smoothing, Adam moments and flow weight decay;
influence masks still hold masked cells exactly fixed. Off by default and
backfilled, so older checkpoints load unchanged; the step keeps no optimizer
state of its own.

- `A = I + l^2 (Dz^T Dz + Dy^T Dy + Dx^T Dx)` on Cartesian lattices: first
  differences with Neumann borders, with `l` =
  `optimizer_flow_sobolev_length_voxels` converted to each lattice's cells
  (coarse cells are wider). Cylindrical lattices use z edges and cyclic
  angular edges within each ring; there are **no radial edges**, and ring 0 is
  pinned. Influence-masked cells are eliminated from the operator (Dirichlet
  zero at the mask), not masked afterwards.
- `optimizer_flow_sobolev_curvature` `none` takes the Sobolev gradient step
  `-(1/lambda) A^-1 g`, with `optimizer_flow_sobolev_damping` as the step
  length. `finite_difference` solves the damped system by PCG with `A^-1` as
  preconditioner, using a forward finite difference of the gradient on the
  same batch as `H v` (the random state is restored before each
  re-evaluation, so every PCG iteration costs one extra forward/backward).
  PyTorch second derivatives are not available: the fitter's backward runs
  family by family through custom Triton kernels and frees each graph. PCG
  stops at `optimizer_flow_sobolev_pcg_iterations`, at the relative residual
  `optimizer_flow_sobolev_pcg_tolerance`, on non-finite arithmetic, or on
  non-positive curvature (the full Hessian may be indefinite), or a residual
  that grows more than twofold in one iteration (a sign the finite-difference
  operator is inconsistent). On a growing residual the previous iterate is
  kept; on non-positive curvature the iterate is moved along that direction
  to the trust boundary (CG-Steihaug), which is the configured Sobolev trust
  radius or else the Sobolev norm of the fallback step `-(1/lambda) A^-1 g`;
  with no valid iterate at all it falls back to that step.
- `optimizer_flow_sobolev_adapt_damping` (default on) adapts lambda across
  steps without retries: times `optimizer_flow_sobolev_damping_increase`
  after a poor solve (negative curvature, growing residual, non-finite
  values), divided by `optimizer_flow_sobolev_damping_decrease` after a
  clean one, never below the configured damping nor above it times
  `optimizer_flow_sobolev_damping_max_factor`. The adapted value is stored
  in checkpoints (`sobolev_damping`; older checkpoints restart from the
  configured value) and logged as `lambda before->next`.
- `A^-1` is applied by a fixed budget of inner CG iterations
  (`optimizer_flow_sobolev_inner_cg_iterations`, preconditioner `cg`).
  Preconditioner `gaussian` instead uses the Gaussian gradient smoother and
  its `optimizer_flow_grad_smoothing_*` widths as an approximate `A^-1`. That
  path is **heuristic**: the smoother has not been shown self-adjoint under
  the cell measure, so PCG with it is not a true PCG. It is the only
  prototype path that couples cylindrical rings.
- The joint step over all flow lattices is scaled by
  `optimizer_flow_sobolev_step_scale`, then capped so that no lattice's
  per-component RMS increment exceeds `optimizer_flow_sobolev_max_step_voxels`
  (default 2 voxels; 0 disables). Unlike Adam the raw step follows the
  gradient scale: on the golden workload the uncapped default step was
  thousands of voxels. `optimizer_flow_sobolev_trust_radius`, when positive,
  additionally scales the step down to that Sobolev norm. There is no Levenberg-Marquardt retry within a step, line
  search or acceptance test. `optimizer_flow_sobolev_evaluate_step` re-runs
  the loss on the same batch after the step, for logging only.

Every 200 steps the fitter logs the curvature mode, termination reason, PCG
and inner CG iteration counts, initial and final residual, Sobolev step norm,
solve wall time, per-lattice step RMS in voxels and a dimensionless roughness
(RMS of first differences over RMS of values), plus the same-batch loss before
and after when evaluation is enabled. VRAM: the solver holds several
lattice-sized work vectors per flow parameter (gradient copy, parameter
backup, PCG vectors), so expect a multiple of the lattice size in transient
memory. Curvature re-evaluations also add their forward/backward time to the
profiler's `fwd`/`bwd` buckets and the whole step to `sobolev`.

`tests/test_sobolev_gauss_newton.py` checks the operators against dense
matrices (symmetry, positive definiteness, borders, periodicity, eliminated
cells), PCG against a direct solve, the large-damping limit against the
Sobolev step, the curvature and non-finite fallbacks, exact parameter
restoration around re-evaluations, and a toy ablation of AdamW, smoothed
AdamW, shared-denominator AdamW, the Sobolev gradient step and the
Hessian-free step on a small synthetic sampling problem. That ablation is
evidence about update smoothness and same-batch loss reduction on a toy; no
full-scroll quality or wall-clock comparison has been run, and none is
claimed. The minimum production comparison is described in the plan.

Smoke evidence from the golden-run workload (201 steps, z 10000 to 11000,
cylindrical, curvature `finite_difference`, 3 PCG iterations, damping 1,
adaptive damping on): with a 1-voxel finite-difference epsilon the first
step's residual grew (the guard stopped it), damping climbed to about 9e3 by
step 200 where PCG then converged in 3 iterations (residual 4.1e3 to 3.3e2)
and the same-batch loss fell from 740 to 693. Epsilons of 4 and 16 voxels
were worse: the secant operator was inconsistent almost every step, damping
ran to its ceiling and training stalled (loss 1297 and 1646 at step 200
against 740). Keep the epsilon at 1 voxel, and keep the damping ceiling
moderate. About 2 s per step against 0.5 s for the AdamW path.

A matched cylindrical A/B with `optimizer_flow_sobolev_diagnostic_interval`
20 (same spec, curvature `none` against `finite_difference`) showed the
curvature path is not earning its cost in this form. Final losses at step 200
were 715 (none) and 720 (finite difference) for 0.5 s against 2 s per step.
The Hessian term was 1 to 5 percent of the predicted reduction at every
probe, so the damped system is essentially `lambda A`; the finite-difference
operator's relative asymmetry on random directions ranged from 0.01 to 1.7,
so it is not the symmetric operator PCG assumes; and `rho` fell from about
0.95 at step 0 to 0.05 to 0.35 for both variants, meaning the loss has large
real curvature that the model does not capture. The conclusion is the plan's
production path: a Gauss-Newton operator built from the residual losses, not a
secant of the full gradient.

### Gauss-Newton on residual losses

Curvature `gauss_newton` uses the positive semidefinite operator
`G v = Jᵀ W J v` of the residual-shaped loss terms (`gauss_newton_residuals.py`).
Each of those losses registers its residual tensor and unweighted scalar with
one call (patch, unverified-patch, unattached-PCL and track radius and DT
terms, umbilicus, shell outer, shell patch radius, absolute winding); every
other term stays first order and the fitter lists the two groups once at
startup. Because the penalties are hinges, L1 and Huber, whose residual-space
curvature is zero almost everywhere, the weights are the IRLS majorizer
`w = (∂L/∂r)/r` with `|r|` floored at `optimizer_flow_sobolev_irls_floor`.
That is exact Gauss-Newton for squared penalties and the classical
reweighting for the L1 family, read off generically with one small autograd
call per term, so no per-loss formula is maintained. `J v` is a forward
difference of the residuals at `p + eps v` (a no-grad capture pass), and
`Jᵀ (W J v)` is the ordinary family-by-family backward with the residuals'
upstream gradient replaced, so the product costs one forward plus one
forward/backward and respects the fitter's per-family graph release.

With `optimizer_flow_sobolev_evaluate_step` the fitter also applies the
plan's acceptance rule: `rho` (actual over predicted same-batch reduction)
drives the damping (`optimizer_flow_sobolev_rho_poor` / `_rho_good`), and a
step whose same-batch loss did not fall is undone bitwise from a saved copy
(`optimizer_flow_sobolev_rho_reject`). PCG's residual-growth guard is
configurable (`optimizer_flow_sobolev_residual_growth_limit`, default 10x)
because a Gauss-Newton operator confined to the sampled cells legitimately
spikes CG's residual norm.

Evidence, cylindrical golden workload, 201 steps, diagnostics every 20:

| Curvature | epsilon (voxels) | symmetry defect | typical rho | loss at 200 |
|---|---|---|---|---|
| none | - | - | 0.94 → 0.03 to 0.5 | 715 (767 with rho-LM) |
| finite difference | 1 | 0.01 to 1.7 | 0.05 to 0.35 | 720 |
| Gauss-Newton | 1 | 0.02 to 0.49 | 0.1 to 0.99 | 818 |
| Gauss-Newton | 0.25 | 0.006 to 0.11 | 0.08 to 1.02 | 780 (898 with rho-LM) |
| Gauss-Newton | 0.1 | 0.003 to 0.05 | 0.05 to 1.03 | 867 |

Reading: the Gauss-Newton product is a consistent symmetric PSD operator at
epsilon 0.25 or below (use that, not the gradient-difference epsilon of 1),
and when the model is trusted `rho` is 1.0 within a percent. But on this
workload its curvature term is only 1 to 5 percent of the predicted
reduction: the IRLS weights are `1/|r|`, and early in a fit the hinge and L1
residuals are tens to hundreds of voxels, so the reweighted curvature is weak
and the step is essentially the Sobolev gradient step at four times the cost.
The failures that remain (`rho < 0` on a third of the probes for every
variant) are overshoots across kinks of the loss, hinge activations and points
crossing lattice cells, which no quadratic model captures; the step-length
control (damping, voxel cap, rejection) decides progress, not curvature.
Single runs, GPU-nondeterministic, on one z window: differences of tens in
the step-200 loss are within run-to-run noise. Warm-started from a
1500-step AdamW checkpoint (satisfied area 45.6%), 200 further steps gave
47.0% for AdamW, 44.7% for Gauss-Newton with abs penalties and 40.6% with
squared ones, and 38.4% for the Sobolev gradient step with squared penalties:
even late in the fit the curvature term stays 1 to 5 percent of the model, and
the flow-lattice second-order step loses ground while AdamW gains. A zero step
reproduces the same-batch loss exactly, so the negative `rho` values are the
true jaggedness of the batch loss, not noise.

### Squared penalties (`loss_penalty_shape`)

`loss_penalty_shape = "square"` replaces the hinge and L1 penalties of the
radius, umbilicus, shell, absolute-winding and track-radius terms by
`magnitude² / (2 s)` with `s = loss_square_scale_voxels` (see
`spiral_helpers.penalty`), giving Gauss-Newton constant residual weights. It
was tried because the `1/|r|` weights above are the reason curvature carries
so little information, and it did not pay:

| Run (401 steps from scratch unless noted) | satisfied area | PCL points |
|---|---|---|
| AdamW, abs (baseline) | 25.1% | 37.1% |
| AdamW, square, s = 16 / 64 / 256 | 16.5% / 16.3% / 16.2% | 28.8% / 25.3% / 24.2% |
| AdamW, square, s = 64, rel-winding weight ×4 | 16.2% | 24.4% |
| AdamW, square, s = 64, rel-winding ×4, shell outer ×0.3 | 17.5% | 23.7% |
| AdamW, square, s = 64, rel-winding ×8, shell outer ×0.3 | 17.4% | 23.1% |
| warm 1500 → 1700, AdamW, abs | 47.0% | 45.5% |
| warm, AdamW, square, s = 16 | 45.0% | 49.1% |
| warm, AdamW, square, rebalanced (×4, ×0.3) / (×8, ×0.3) | 41.9% / 40.8% | 44.5% / 42.2% |

The scale `s` is irrelevant for AdamW (it rescales the squared families as a
block and Adam is scale-invariant), and rebalancing the two families whose
relative weight the squaring changed most (relative winding stays L1 and lost
about 7×; shell outer gained about 3×) moved the result by a point or two. The
gap is structural: the satisfaction metric counts points within a tolerance,
which is what median-like L1 and hinge penalties optimise, while a squared
penalty spends its effort pulling in the far outliers. Squared penalties with
Gauss-Newton were also worse than AdamW when warm-started (above). `abs`
remains the default; the switch is kept for experiments and is backfilled.

`loss_penalty_shape = "huber"` (quadratic below `loss_huber_delta_voxels`,
L1 beyond; default δ 6 voxels, near the satisfaction tolerance) was the
remaining candidate: robust to outliers, smooth and curvature-bearing exactly
where points are pulled inside the tolerance. Over a longer warm-start window
(1500 → 2500 steps from the same checkpoint, satisfied area 45.6% at the
start):

| Warm 1500 → 2500 | satisfied area | PCL points |
|---|---|---|
| AdamW, abs | 50.3% | 54.3% |
| AdamW, Huber δ 6 | 48.9% | 54.0% |
| Sobolev gradient step, Huber δ 6 | 45.2% | 44.2% |
| Gauss-Newton, Huber δ 6 | 45.4% | 44.2% |

Huber with AdamW is at parity with abs within run-to-run noise (a point or
two), so it is a safe alternative but not an improvement. With the flow
lattices stepped by the Sobolev or Gauss-Newton solver the fit did not
advance at all in 1000 steps, while AdamW gained 5 points. The curvature term
stayed at 1 to 2 percent of the model because most residuals that carry
gradient lie beyond δ, in the L1 regime, and the damping sat at its ceiling
because even small smoothed steps (Sobolev norm 0.2 to 0.3) raised the
same-batch loss by 100 to 600 on a third of the probes. The likely reading is
that this loss is reduced by cell-local flow changes, which per-cell Adam
scaling follows and a Sobolev-smoothed direction (length 32 voxels, two fine
cells) suppresses; a shorter length or the unsmoothed metric `A = I` was the
test of that, and it refuted it: over the same 1500 → 2500 window the Sobolev
gradient step with Huber penalties reached 42.8% at length 8 voxels and 43.1%
at length 0 (identity metric, a capped gradient step with rho acceptance),
against 45.2% at length 32 and 48.9% for AdamW. The smoothing prior is not
what holds the step back; the gradient direction itself, scaled by one step
length for the whole lattice, is. Gradient magnitudes differ by orders of
magnitude between densely sampled and barely sampled cells, so a global step
either overshoots the former (the negative `rho` probes) or freezes the
latter, and the reweighted curvature is too weak in the L1 regime to equalise
them. Per-cell normalisation, which is what AdamW provides, is the property
that matters on this loss; a diagonal (Jacobi) preconditioner built from the
Gauss-Newton operator would be the principled version of it and is the one
direction from this work not yet tried.

## Spiral service host setup

VC3D connects to a Spiral service in one of three modes, all speaking the same
authenticated HTTP protocol:

- **Localhost** — VC3D launches and owns the service on loopback. Nothing to
  set up beyond the Python environment; the dataset (plus optional output and
  cache roots) is chosen in the connection panel and VC3D launches the bound
  service with those values. Selecting a different dataset restarts the owned
  service — one service instance is bound to one dataset.
- **Remote (SSH)** — the supported internet flow. SSH access to the host is
  the only client-side prerequisite: VC3D opens and manages its own SSH
  tunnel, reads the service's auto-generated API key over SSH, and attaches to
  a persistent loopback service you start on the host. VC3D never starts the
  service on a remote host.
- **Remote (LAN)** — direct HTTP on a trusted network, authenticated with the
  service's auto-generated API key. No reverse proxies, VPNs, or manual
  tunnels are ever required.

In every mode the service — not the client — owns the base inputs: it is
launched with `--dataset` (inputs) and `--output` (all generated state),
resolves the dataset once at startup, and advertises the result through
`/dataset`. `--output` must resolve outside the dataset root; the optional
`--cache` (derived host caches) defaults to the documented user cache,
`$XDG_CACHE_HOME/vc3d/spiral` (`~/.cache/vc3d/spiral`). Clients can add
ephemeral inputs, commit them, and change run parameters, but cannot repoint
the session at different host paths.

### Creating the Spiral Python environment

The service host needs the Spiral environment (a CUDA-capable PyTorch plus the
dependencies in `pyproject.toml`, Python ≥ 3.14). With [uv](https://docs.astral.sh/uv/):

```sh
cd spiral-fitting
uv sync            # creates .venv from pyproject.toml
```

This also builds Spiral's native helpers as `vc_spiral.spiral_sampling`,
`vc_spiral.track_crossings`, `vc_spiral.track_store`, and
`vc_spiral.surface_index`. OpenMP is used when the toolchain provides it; the
same modules build with serial kernels when it does not.

or with conda/pip, install `torch` for your CUDA version and then
`pip install -e .` from `spiral-fitting/`.

### Resident sparse field pools

Normals, gradient magnitude, and surf-SDT samples are served by fully
resident device brick pools. Each store's occupied bricks are packed once
into a flat sidecar next to the source zarr by `pack_resident_pools.py`:

```sh
python pack_resident_pools.py /path/to/lasagna_inputs \
    --ct /path/to/<scroll>_ds2.zarr --ct-group 2 --verify 2000
```

`--ct` zeroes every voxel whose CT voxel reads 0 (the mask region around the
scroll) so those bricks drop out of the pool and sample as no-data. The
fitter loads the sidecars restricted to the configured z-ROI in one
sequential read per channel (for the full s1 ROI: ~33 GiB SDT + ~10 GiB
normals); after that every gather is pure device indexing with no I/O and no
eviction. When a required sidecar is missing, the fitter builds it before GPU
loading and reports chunk progress. In DDP runs only rank 0 builds it. Manual
prepacking with `--ct` remains useful because the CT mask can substantially
reduce the resident pool size.
Set `FIT_SPIRAL_RESIDENT_BOUNDS_CHECK=1` to enable per-gather bounds
assertions when debugging new sampling code.

### Internet flow (SSH attach)

Start a persistent loopback service on the GPU host with its dataset. Give
each independently operated service a stable session name, port, and GPU.
Nothing is exposed on the network; VC3D tunnels to it over SSH:

```sh
tmux new -s spiral-alice 'python spiral_service.py --port 8765 \
    --dataset /data/scrolls/s1 --output /data/spiral-output/s1 \
    --gpus 0 --session-name alice'

tmux new -s spiral-bob 'python spiral_service.py --port 8766 \
    --dataset /data/scrolls/s1 --output /data/spiral-output/s1 \
    --gpus 1 --session-name bob'
```

The service uses only physical CUDA device `0` by default. Select a different
device or enable distributed fitting across several GPUs with a comma-separated
host-side list:

```sh
python spiral_service.py --port 8765 \
    --dataset /data/scrolls/s1 --output /data/spiral-output/s1 --gpus 0,1,2,3
```

Multi-GPU sessions run one fitter rank per listed device and split the configured
per-step sample counts across those ranks by default. The device list is fixed for
the lifetime of the service; restart it to change the selection.

A named service writes autosaves, previews, artifacts, uploaded checkpoints,
Lasagna output, and ephemeral inputs beneath `<output>/<session-name>/`, held
under an exclusive lease: two live services cannot own the same
output/session-name pair. Launches without `--session-name` use `<output>/`
directly. Permanent dataset inputs and the shared user cache stay untouched —
nothing generated is ever written under the dataset root.

Every completed Spiral preview is flattened by the host's Lasagna service
before it becomes downloadable in VC3D. The published grid uses a fixed
20-voxel output step: each dimension is
`ceil(((source_points - 1) * source_step) / 20) + 1`. Winding membership,
loss-map overlays, and run differences are transferred through Lasagna's
output-to-source correspondence so they remain aligned when the output grid
dimensions differ from the Spiral grid. If flattening or artifact mapping
fails, the service reports the publication error and VC3D keeps displaying the
previous successfully published preview.

On first start the service generates a strong API key at
`~/.config/vc3d/spiral_api_key` (mode `0600`) and prints it to the console.
For an SSH profile you never copy it: VC3D reads that file over SSH.

In VC3D's Spiral workspace, add a *Remote (SSH)* profile with the
`[user@]host` destination (your `~/.ssh/config` aliases, agents, and jump
hosts work unchanged) and the service port (`8765` above), then Connect.
Non-interactive SSH authentication (keys or an agent) is required. If SSH does
not trust the host key yet, run `ssh <destination>` once in a terminal to
accept it — VC3D deliberately never auto-trusts host keys.

The fit survives viewer disconnects, laptop sleep, and network drops;
disconnecting or closing VC3D never terminates a service it did not launch.
The workspace reports the active loading, optimization, checkpoint, and
preview stage with elapsed time. Stages with a real work total also show a
counter and ETA; opaque native or CUDA operations deliberately use an
indeterminate bar instead of a guessed overall percentage. The same stage
updates are printed by standalone `fit_spiral.py`, with periodic elapsed-time
heartbeats when output is captured to a log.
While connected, the circular-arrow button beside the connection controls
restarts the remote service and reconnects automatically. The service replaces
its own process in place, so a containing `tmux` session remains alive and an
attached terminal is not disconnected.

### Trusted-LAN flow (direct HTTP)

```sh
python spiral_service.py --bind 0.0.0.0 --port 8765 \
    --dataset /data/scrolls/s1 --output /data/spiral-output/s1
```

Copy the API key printed at startup into the *Remote (LAN)* profile's API key
field (or export `SPIRAL_API_KEY` before starting VC3D). A non-loopback bind
always requires an API key (auto-generated when absent).

**Plaintext-HTTP risk note:** direct HTTP is not encrypted — on-path observers
can read the API key and the transferred data, so use it only on networks the
operator trusts. Over the internet, use an SSH profile instead. HTTPS
endpoints behind an existing TLS proxy also work; VC3D uses normal system CA
validation and never ignores certificate errors.

### API key file

- Location: `~/.config/vc3d/spiral_api_key` (respects `XDG_CONFIG_HOME`), or
  pass `--api-key-file PATH`.
- The key is created on first start (mode `0600`) and reused on later starts.
- To rotate it, delete the file and restart the service; reconnect clients
  with the new key. The key is never written to HTTP logs, responses, or the
  ready line — the console print at startup is the intended way to obtain it.
- `--nonce` is only for processes launched and owned by VC3D.

### Datasets, output, and cache

`--dataset` must point at a dataset root containing at least `umbilicus.json`
and `spiral-scroll.json`; the service refuses to start when either is missing.
Verified patches are required when their default-on input toggle is active,
but a patch-free fit can initialize with that source disabled. The dataset
holds inputs only.

`--output` is required and must resolve outside the dataset root. Every piece
of generated state — run directories, autosaves, previews, published
artifacts, ephemeral inputs, upload staging, and uploaded checkpoints — lives
under it (under `<output>/<session-name>` for a named service). Make sure its
filesystem has room for checkpoints and previews.

`--cache` holds derived host caches (content-addressed, shareable between
datasets). It defaults to `$XDG_CACHE_HOME/vc3d/spiral`
(`~/.cache/vc3d/spiral`) and must also resolve outside the dataset root. The
headless `fit_spiral.py` CLI accepts the same `--cache` with the same default
(`FIT_SPIRAL_CACHE_DIR` still overrides it for the CLI).

If the dataset root is read-only the fit still works, but *Commit current
inputs* is unavailable (committing writes inputs into the dataset).

### Connecting from VC3D

Open the Spiral workspace and pick the profile in the *Spiral Service*
section. For the local profile, set the dataset root (and optionally output
and cache roots) there — VC3D launches its owned service bound to those
values. Connection must succeed (an authenticated `/health` handshake and an
API-version check) before session controls enable. The base-input rows always
populate read-only from the service's advertised dataset resolution; run
parameters (z range, iterations, advanced config) stay editable and persist
per profile. Generated previews, geometry, and
checkpoints transfer through the artifact API into a local cache — no shared
filesystem is needed. Optional: set the profile's **Local dataset path** if
this machine mounts the same dataset, so input surface overlays
(verified/unverified/shell) can be displayed locally. It is assumed to
correspond to the dataset root the service advertises, which is the prefix
service paths are translated from; without it those overlays are simply marked
unavailable.

`spiral-scroll.json` in the dataset root is the only source of the scroll's
name and voxel resolution and of the Lasagna store layout (zarr groups,
coordinate scale). None of them are panel settings: the panel reports them
read-only, and the service rejects a session request that carries
`scroll_name`, `voxel_size_um`, `lasagna_group` or `lasagna_scale`.

Optional supervision sources have rebuild-scoped boolean switches in Advanced
config. Set an `input_use_*` key to `false` to skip validation, loading,
sampling, and losses for that source without changing its tuned weights or
sample counts. Available switches cover verified/unverified patches, tracks,
fibers, each PCL role (`absolute`, `relative`, `same_winding`, and
`drawn_control_points`), normals, surface SDT, gradient magnitude, winding
inference, and the outer shell. For example:

```json
{
  "input_use_tracks": false,
  "input_use_fibers": false,
  "input_use_pcl_drawn_control_points": false
}
```

Changing one requires a whole-fit rebuild. Disabling a prerequisite also
disables its dependent supervision: phase spacing needs normals and surface
SDT, while winding inference needs the outer shell.

While a session is active you can right-click a patch in the Surface panel or
a fiber in the Fibers panel and pick *Add to current spiral fit*. Added inputs
are uploaded into a session-scoped ephemeral folder, used from the next run
onward, and can be moved into the shared dataset with *Commit current inputs*.
Commits from multiple service processes are serialized; distinct inputs and
point collections are preserved, while an existing patch or fiber identifier
is reported as a conflict and is never overwritten.

Interactive influence settings are scoped to each **Run** request. The fitter
builds a fresh influence region from only the inputs pending for that run,
uses it for the requested iteration window, and discards it before autosaving.
Influence masks, limits, and controls are not checkpoint state. All
`interactive_influence_*` advanced settings can therefore change between runs
without reloading the resident session. The **Disable DT** percentage controls
how much of that run suppresses directional DT losses after incorporating its
pending inputs.

**Checkpoints** are one panel section, and loading one is one button. It lists
what the service advertises (checkpoints at the dataset root, and those under
the output directory such as the autosave) plus any **client-local `.ckpt`**
you browse for; a local file is uploaded to the service's
`<output>/uploaded-checkpoints/` directory on the way (the panel shows
progress and the transfer restarts if interrupted). Checkpoints are identified
by SHA-256, so choosing content the service already retains reuses it without
transferring the file again, and the service validates new archives and keeps
the newest few unique uploads.

Before the first fit is initialized, *Load* initializes it directly from the
selected checkpoint; it does not first construct a throwaway model. The
configuration profile becomes **Checkpoint** and displays the resolved
configuration carried by that checkpoint. *Initialize Fit* is the separate
from-scratch action.

With an existing fit, *Load* replaces the resident model's weights, optimiser
and RNG state in place. When the checkpoint does not match the live model the service refuses
it and says what a rebuild would have to replace: rebuilding the **model only**
keeps the loaded dataset inputs and everything already added to the fit, while
a **whole-fit** rebuild re-reads the dataset and discards added inputs that
were never committed. The panel reports the reasons and asks; a checkpoint no
rebuild can accept — one written against another dataset, or against a
configuration schema this service does not have — is reported and nothing is
offered. A checkpoint-backed session takes its durable configuration from the
checkpoint, so the local advanced-config profile does not override it.

The Iterations value on *Run* is a count added to the checkpoint's durable
iteration. The progress bar is local to that run and therefore starts at zero;
the session status line reports the global current and target iterations.

The section also holds *Save on Service* and *Download…*, and reports the
checkpoint the resident fit was actually built from. That report is read-only:
it is not a field, and a rebuild carries it forward by itself.

### Shutdown and logs

Stop the service with `Ctrl-C` or `SIGTERM` (`tmux kill-session -t spiral`);
it tears the fit session down at a safe boundary. Logs go to the service's
stdout/stderr on the host — for a `tmux` session, `tmux attach -t spiral`; for
an unowned service VC3D's Python-output dialog only reminds you of this. A
service started on an explicit port can be restarted immediately (the socket
uses `SO_REUSEADDR`). VC3D's remote restart control does not run
`tmux kill-session`; it gracefully closes the fit and re-executes the service
with the same interpreter, arguments, and process ID. Note that a large artifact
download during a running fit competes with the fitter for the Python
interpreter and can slow iterations somewhat.

### Optional systemd user unit

```ini
# ~/.config/systemd/user/spiral-service.service
[Unit]
Description=VC3D Spiral fitting service

[Service]
WorkingDirectory=%h/villa/spiral-fitting
ExecStart=%h/villa/spiral-fitting/.venv/bin/python \
    %h/villa/spiral-fitting/spiral_service.py \
    --port 8765 --dataset /data/scrolls/s1 \
    --output /data/spiral-output/s1 --gpus 0
Restart=on-failure

[Install]
WantedBy=default.target
```

```sh
systemctl --user daemon-reload
systemctl --user enable --now spiral-service
journalctl --user -u spiral-service -f     # logs (includes the API key print)
```

Direct command-line use remains fully supported; the unit is a convenience.

## Packing large track databases

Legacy track DBMs store a pickled list of NumPy arrays in every key. For large
datasets this spends minutes decoding millions of Python objects each time a
fit starts. Convert a DBM once to the adjacent packed format:

```sh
python convert_track_store.py \
    /data/tracks/2um_ds2_ps256_surf_v2.dbm
```

This writes `2um_ds2_ps256_surf_v2.dbm.vctracks/` atomically. The directory
contains contiguous coordinates, ragged offsets, source IDs, family codes,
Z bounds, arclengths, and tortuosities. `fit_spiral.py` automatically prefers
a current adjacent packed store while retaining the DBM as the authoritative
source and compatibility fallback. A source-file fingerprint prevents a stale
store from being used after the DBM changes; rerun with `--force` to replace it.

The native `vc_spiral.track_store` loader memory-maps the packed files, applies the Z
ROI from per-track metadata, and emits one compact float32 ragged array without
constructing per-track Python objects. The crossing builder also stages
directly from a current packed store, bypassing DBM and pickle decoding.

## Caching exact track crossings

Crossing-connected track sampling needs the exact shared voxels between the
horizontal and vertical track families. Build that index once as a CSR
sidecar instead of sorting every track point whenever a fit session loads:

```sh
python build_track_crossings.py \
    /data/tracks/2um_ds2_ps256_surf_v2.dbm \
    --z-min 4000 --z-max 17000 \
    --temp-dir /fast/disk/tmp
```

The optional Z range is half-open (`[z-min, z-max)`) and retains only tracks
entirely contained in that range. Omit both options to index the whole DBM.
The standalone builder uses a hybrid memory/disk index: it streams DBM tracks
into temporary coordinate and packed-voxel files, keeps the coordinates
memory-mapped, then loads and radix-sorts the packed keys in RAM. The native
`vc_spiral.track_crossings` kernel uses all requested workers for sorting,
exact-voxel discovery, arclength calculation, and pair consolidation. The
extension is built with the other Spiral native modules by `uv sync` from this
directory. A slower Python fallback remains available.

The builder needs roughly 20 bytes of temporary disk space per selected point.
The native radix sort temporarily holds about 32 RAM bytes per point; after the
sort, those arrays are released before the 8-byte-per-point arclength vector and
compact 16-byte crossing events are consolidated. This avoids retaining either
the selected track database or Python dictionaries of crossing pairs in RAM.
Temporary files are removed after the sidecar is written. Without `--temp-dir`,
the temporary workspace is created beside the tracks DBM rather than under the
system temporary directory.

The script writes
`/data/tracks/2um_ds2_ps256_surf_v2.dbm.crossings.npz` atomically.
`fit_spiral.py` finds it automatically from the configured tracks path. The
sidecar includes a fingerprint of every DBM backing file; a stale or malformed
file is ignored and the fitter falls back to its in-memory exact crossing
scan. Re-run the builder after changing the DBM (`--force` replaces a current
cache). A range-limited sidecar can serve the same or a narrower fitting Z
range; building another range replaces it. Point-level track exclusion also
uses the fallback because clipping a
track changes its crossing-local indices.

## Converting track DBMs to OME-Zarr

`tracks_to_ome_zarr.py` rasterizes the ZYX polylines produced by
`extract_surface_tracks.py` into a compressed `uint8` OME-Zarr. Value 0 is
background; values 1–255 are assigned with proximity-aware reuse and display
as categorical colors with VC3D's Glasbey colormap. Rasterization uses worker
processes, while independent Zarr chunks are compressed and written by a
thread pool using Zstandard level 3.

Use a paired OME-Zarr to copy the exact volume shape and physical geometry:

```sh
python tracks_to_ome_zarr.py \
    /data/tracks/2um_ds2_ps256_surf_v2.dbm \
    --out /data/tracks/2um_ds2_ps256_tracks.ome.zarr \
    --like /data/volumes/2um.ome.zarr \
    --like-group 0
```

Alternatively pass `--shape Z,Y,X`. If neither `--shape` nor `--like` is
given, the script first scans the DBM and uses the maximum track coordinate
plus one. The explicit forms avoid that extra pass for large databases.
`--resume` continues an interrupted conversion. Multiple positional DBMs are
combined into one output, so separate scrolls should be converted in separate
commands.

## Neural winding-inference losses

Set `dense_spacing_mode` to `winding_model` and provide the compact exported
crossing directory at the conventional `<dataset>/winding_inference` path or
override `paths.winding_inference` in `spiral-scroll.json`. Two vocabularies
deliberately coexist: `winding_model` names the fitting mode and its tunables
(`sample_count_winding_model_*`, `winding_model_relative_pair_delta`,
`winding_model_huber_delta`, the `dense_spacing_winding_model_*` losses),
while `winding_inference` names the exported artifact and everything tied to
its on-disk identity (the input path, the `winding_inference_crossings`
artifact type, and the checkpoint fingerprint field). The store is
checksum-verified and copied to each fitting GPU at startup; rays whose
crossings cannot intersect the configured z-range are excluded from sampling,
and optimisation then does no inference-store filesystem I/O. The default
24,000 samples per step are split evenly between long relative-winding pairs
(`sample_count_winding_model_relative_pairs`, index separation drawn from
`winding_model_relative_pair_delta`) and adjacent-passage density pairs
(`sample_count_winding_model_density_pairs`). In this mode surf-SDT is
neither loaded nor required, while the independent Lasagna normal and native
minimum-spacing losses remain available.

The compact store is created by the Vesuvius winding-model
`export_spiral_supervision.py` tool; see its `NATIVE_PHASE_CACHE.md` for the
exact export command and format.

For a headless fit, pass the dataset root with `--dataset` and select inference
mode (plus any independently disabled losses) through
`FIT_SPIRAL_CONFIG_OVERRIDES`. The dataset's `spiral-scroll.json` and the
declarative input catalog determine which conventional inputs are resolved.

## Fiber direction samples

The optional fiber-direction loss consumes one packed artifact extracted from a
remote Lasagna fiber prediction. Extraction downloads only chunks intersecting
the requested z ROI and keeps the highest-presence voxel in each fixed
prediction-space cell:

```bash
./.venv/bin/python fiber_direction_samples.py \
  https://example/fibers.lasagna.json \
  /path/to/dataset/fiber_directions.npz \
  --z-roi 10000,11000 --output-scale 4 \
  --presence-threshold 160 --cell-size 2
```

The z ROI is half-open and expressed in the output/fitter coordinate system;
`--output-scale 4` means one fitter `/2` coordinate is four base `/0` voxels. The
extractor always covers the fiber volume's complete XY extent.

Both `input_use_fiber_directions` and `loss_weight_fiber_directions` default
to off/zero; set the toggle true and the weight above zero to enable the
loss. The fitter then loads the conventional `fiber_directions.npz` artifact
and samples `sample_count_fiber_direction_points` observations per step.
Positions and directions constrain only local fitted-sheet orientation; they
do not attach a sample to a particular winding.
