# Pinned spiral spacing: implementation and test plan

Target: `villa/spiral-fitting`. Two stages, each independently codable, testable and
shippable, in order. Stage 2 is the core change, split into a correctness build (2a) and a
performance pass (2b); stage 3 changes the losses. (Numbering starts at 2 to match the
original discussion; there is no stage 1.)

## 0. Background and goal

### Current architecture (what this plan builds on)

`transforms.SpiralAndTransform.get_slice_to_spiral_transform()` returns the inverse of a
`ComposeTransform` whose forward (spiral -> scroll) order is:

1. `GapExpandingTransform` -- radial-only. For each ray (theta, z) it builds a table of
   winding radii `R[k, theta, z]` by cumulative-summing per-winding gaps
   (`get_transformed_winding_radii`); gaps are `lower_bounded_gap(logit, dr, min_gap, ...)`,
   i.e. `min_gap + (dr - min_gap) * softplus_ratio(logit)`. A point at canonical radius
   `(k + theta/2pi) * dr` is moved to `R[k]`, and points between windings are lerped.
   The logits live in `GapExpanderParams.logits`, laid out per winding along theta then
   concatenated (index `winding_first_logit_idx`), by z.
2. optional horizontal flip (`spiral_outward_sense`)
3. one or more `IntegratedFlowDiffeomorphism` (RK4-integrated flow fields, `flow_fields.py`)
4. `VaryingLinearTransform` (z-dependent 2x2 `expm`)
5. `UmbilicusTransform` (z-dependent shift to the umbilicus)

So in the scroll -> spiral direction the gap expander is the **last** stage: it acts on
the output of the flow chain (call this the *intermediate* space: umbilicus-centred,
roughly circularised). `dr_per_winding` is a learnable global scalar
(`dr_per_winding_logit`, `lower_bounded_dr`) and appears in ~190 places as the canonical
spacing: `sample_spiral.get_theta_and_radii` defines the shifted radius
`s = r - dr*theta/2pi`, windings are `s / dr`, the seam at theta=0 is handled by
`unwrap_shifted_radii` and `theta_crossing_map.ThetaCrossingMap`.

Hard constraints are currently *soft losses* in `losses.py`:

- `_patch_radius_and_dt_losses` (via `get_patch_and_umbilicus_losses`,
  `get_unverified_patch_losses`): variance of shifted radius over patch samples (radius
  loss), plus a DT loss that pulls samples to a snapped integer target winding
  (`dt_targets.patch_dt_target_in_sample_frame`), measured in scroll space via
  `slice_to_spiral_transform.inv`.
- `get_patch_rel_winding_loss`, `get_patch_abs_winding_loss`: PCL-annotated winding
  differences between patches.
- `get_unattached_pcl_strip_losses`: constant-radius-along-strip for PCL chains not on a
  patch.
- Constraint grouping already exists: `FitContext` merges PCLs, patches and fiber links
  into connected components (see comments around `fit_spiral.py` lines 1370-1450, 1690-1780,
  `link_points_to_patches`, `normalise_pcl_winding_annotations`).

### Note on `dr_per_winding`

The gap expander stays a transform throughout, and `dr_per_winding` stays as it is: the
learnable global canonical spacing, and the unit in which shifted radii, winding indices,
the theta=0 seam adjustment, pin targets, the density kernel and hinge margins are all
expressed. Nothing below needs it constant: pin targets `dr*(T_g + n_i + theta/2pi)` and the
free map both live in canonical units and scale together. Wherever `dr` appears below it
means `get_dr_per_winding()`.

### Goal

Replace the gap expander's free winding-radius table with one that is **pinned** so that
every hard-constraint point lands exactly on its target winding coordinate, for every
value of every other parameter, while the map stays strictly monotone along each ray
(invertible). The flow chain is untouched and stays free. What the pin *targets* are is
decided by one fractional scalar `T_g` per constraint component, optimised by the existing
(tent-shaped) DT loss.

Exactness is structural but **conditional**: a monotone radial map can only satisfy the
pins on a ray if the flow already places them in the same radial order as their targets.
Where two components are swapped along a ray the ordering guard (2a.4) fires and one of
them misses. So the scheme still needs the flow to be right *in ordering* -- which is what
the soft losses currently establish -- and the guard-violation count is the diagnostic
that says whether the flow is good enough. The flow does not need to be accurate in
*position*; that is what the pins absorb.

Vocabulary used below:

- **component** `g`: a connected component of the hard-constraint graph (patch(es) +
  PCLs + fiber links). One free scalar `T_g` each (fractional winding coordinate).
- **constraint point** `x_i`: a point in scroll space belonging to component `g(i)` with
  known integer offset `n_i` relative to the component's reference (from PCL annotations
  and theta=0 crossing bookkeeping). Its target *canonical radius* is
  `r_target_i = dr * (T_g + n_i + theta_i / 2pi)` where `theta_i` is its intermediate-space
  angle. Equivalently target shifted radius `dr * (T_g + n_i)`.
- **pin**: a constraint point after the flow chain, `(z_i, theta_i, r_i)`, together with its
  target. `z_i` is the *post-flow* z (the flow fields are 3D and move z). The pinned
  radial map on ray `(theta_i, z_i)` must send `r_i -> r_target_i`.
- **winding slot** `k_i = round(T_g + n_i)`: the integer winding a pin belongs to. This is
  the key the pin table is indexed by (2a.3); it matches the free table's winding axis.
- **free map** `s_free`: what the radial map would be without pins (the existing gap
  expander).

---

## Stage 2a. Pinned winding radii -- correctness build

This is the core change. Build it plainly in PyTorch first; optimise in 2b.

### 2a.0 One-ray reference implementation (do this first)

Before the registry, the training loop or any transform class, write `pins.py`'s core
as pure functions over one ray: inputs are a free table row (`R_free[k]`, `dr`, `theta`)
and a list of anchors `(R_j, S_j, w_j)`; outputs are `s(r)`, its inverse, and the
diagnostic counters. Everything in 2a.4 -- the guard semantics, zero-mass transparency,
coincident and crossing anchors, the DT gradient into `T`, and the uncapped table -- is
specified and unit-tested at this level (tests 1-5c, 7 below) before anything is
integrated. The later Triton kernel (2b) is checked against this reference.

### 2a.1 Constraint registry

New module `pins.py` with a `ConstraintRegistry` built once from `FitContext`:

- Enumerate every hard-constraint point: verified patch grid points (from `PatchAtlas`),
  cross-patch PCL points, unattached-strip PCL points, absolute-winding PCL points. Fiber
  *direction* samples (`fiber_direction_samples.py`) are orientation data, not pins; leave
  them to the fiber-direction loss.
- Assign each point a component id `g(i)` from the existing link components and an
  integer offset `n_i`. `n_i` = PCL winding annotation relative to the component reference
  point, plus theta=0 crossing offsets. Crossing offsets come from the same machinery the
  rel-winding loss uses (`_pcl_chain_seam_adjustments`, `ThetaCrossingMap` unwrap trees);
  for points within one patch use the patch's theta lift. Store `n_i` as `int32`.
  `n_i` is defined relative to the pin's intermediate-space angle, and that angle is
  recomputed post-flow every step, so a pin near theta=0 that crosses the seam between
  steps would otherwise see its target jump by a whole `dr`. Store each pin's angle at
  construction, `theta0_i`, and each step use
  `n_i(t) = n_i(0) + round((theta0_i - theta_i(t)) / 2pi)` (a pin never moves more than
  half a turn in one step). The slot-rebuild check in 2a.3 must watch this adjusted `n_i`,
  not the stored one; a seam crossing is a slot change.
- Store scroll-space `zyx_i` (float32), `g(i)`, `n_i`, and the per-pin footprint
  `(eps_theta_i, eps_z_i)` (float32, rule in 2a.3) as flat GPU tensors, plus an object-type
  tag (patch grid / chain / isolated) for logging.
- Register `T = nn.Parameter(num_components)` on `SpiralAndTransform`. Initialise
  `T_g` = median over the component's points of the *current* model's unwrapped shifted
  radius / `dr` minus `n_i` (i.e. the value the existing soft fit has already found).
  For a fresh model do not pin from step 0: run the existing unpinned soft fit for a
  short warm-up (config `model_pins_warmup_steps`) so the flow has the ordering roughly
  right, then build `T` from that state and switch pins on. Starting pinned from an
  identity flow would begin with many guard violations. Absolute-winding components:
  `T_g` fixed (buffer or masked gradient).
- Report the registry size (points per component, total) at construction. Every
  registry point is pushed through the RK4 flow forward and backward each step in 2a
  (2a.2). Measure this on a real dataset **before** starting 2a; if the count is in the
  millions, 2a is unusable even as a reference and 2b item 1 (subsampling) moves into 2a
  with a config default that keeps the full set available for export and for tests.

Also emit, once at construction, a consistency report: components whose points have
mutually inconsistent `n_i` (a cycle with non-zero sum) are logged with the offending
PCLs. Do not try to fix them here; just make them visible.

### 2a.2 Per-step pin computation

Add `SpiralAndTransform.compute_pins()`:

1. Push all registry points through the flow chain **excluding** the gap expander. There
   is already `get_flowbox_to_spiral_transform`; add a sibling that returns the compose of
   umbilicus^-1, linear^-1, diffeo^-1, flip only (i.e. `get_slice_to_intermediate_transform`).
   Output `(z_i, theta_i, r_i)` via `get_theta_and_radii(..., dr)`.
2. Target canonical radius `r_target_i = dr * (T[g(i)] + n_i) + dr * theta_i / 2pi`.
3. Keep the graph attached (gradients into flow, linear, `T`). For the shared-detached-leaf
   scheme in `get_shared_transform_tensors`, the pins tensor `(z, theta, r, r_target)`
   becomes an additional shared leaf: each loss family's backward accumulates into it, and
   the step propagates once through `compute_pins`.

For 2a, push **all** registry points every step. This is deliberately slow; 2b fixes it.

### 2a.3 Pin lookup per ray

Rays are `(theta, z)`. A ray crosses **every** winding, so in a well-annotated region a
ray carries one pin per covered winding -- with the default 130 windings that is dozens,
not a handful. And one component can legitimately put several pins on one ray (a patch
spanning more than one turn has points with different `n_i` on the same ray). So the
table is keyed by **winding slot** `k`, not by component, and is dense over windings:

- Rasterise pins into a `(k, theta_bin, z_bin)` table at the existing gap-logit
  resolution (`model_gap_expander_logit_resolution`), the same layout as the free table
  `R[k, theta, z]`. The table is **uncapped**: pins are sorted by cell id and stored in
  compressed-sparse-row form (one flat pin array plus per-cell offsets). A query ray
  gathers all pins in its neighbourhood cells and accumulates kernel sums with segment
  or `index_add` operations over (query, pin) pairs. No pin is ever dropped for capacity
  reasons, so the exactness claim at export is unconditional on density. A padded
  fixed-capacity layout is allowed only as an explicitly approximate *training* path in
  2b (see 2b item 4); it must expose a `pins_dropped` diagnostic, and export asserts it
  is running the uncapped path.
  Rebuild the slot assignment whenever `round(T_g + n_i)` changes for any pin (cheap
  check each step; a slot flip is rare once `T` has settled).
- The theta axis **wraps**: the neighbourhood gather at theta bins 0 and last must
  include each other, and theta differences are taken modulo 2pi.
- **Coincident-pin compatibility pass**, run after slot assignment and before
  rasterisation (and again on every rebuild). The model has exactly one radius per
  `(ray, slot)`, so two pins in the same slot on (nearly) the same ray are either the
  same physical sheet observed twice, a fold (same component and `n_i`), or an
  inconsistent annotation (different components). The singular kernel cannot separate
  them; IDW would silently average them and both would lose exactness. Rule: pins in the
  same slot whose `(theta, z)` distance, in each pin's own normalised footprint units, is
  below `model_pin_coincidence_frac` (~0.05) are compared radially. If
  `|r_a - r_b| < model_pin_conflict_tolerance` times the local free winding gap (~0.1)
  they are *compatible* and merged into one pin (mean radius, mean target, footprint of
  the larger). Otherwise they are a *conflict*: counted (`pin_conflicts`), listed with
  component ids, `n_i` and object types, tagged fold vs cross-component, and in 2a still
  averaged (demotion is out of scope). The export report carries the conflict list, so
  the exactness claim is precisely "every pin that is neither conflicted nor touched by
  the ordering guard".
- **Per-pin footprint.** Each pin carries its own kernel widths `(eps_theta_i, eps_z_i)`,
  stored in the registry alongside `zyx_i, g(i), n_i`. Theta is an angle, not a voxel
  distance (a voxel of arc is `1/r` radians, which varies by a factor of 100 across the
  scroll). The widths are set from the pin's **own object's spacing**, not from a global
  number: if the kernel is narrower than half the spacing to a pin's neighbours the map
  reverts to the free map between them and the canonical sheet gets a bump at every pin;
  if it is wider than the object's spacing a dense patch bleeds onto other components'
  pins in the same slot. The registry mixes 2D patch grids, 1D PCL chains and isolated
  points, whose spacings are unrelated, so one global width cannot serve all three.
  Rule, computed once at registry construction (and again on rebin in 2b):
  - `eps_i = model_pin_kernel_spacing_factor` (~1.5) times the distance, in intermediate
    `(theta, z)`, from pin `i` to its nearest neighbour **in the same component and slot**,
    taken separately in theta and z. For a patch grid point this is the grid stride in
    each direction; for a chain point the along-chain step projects onto both axes.
  - Each of `eps_theta_i`, `eps_z_i` is floored at `model_pin_kernel_min_theta_radians` /
    `model_pin_kernel_min_z_voxels` (~3 voxels of arc at the pin's radius, ~3 voxels) so
    an isolated point or a chain running purely along z still has a footprint in both
    directions, and capped at `model_pin_kernel_max_*` so a stray far-apart pair does not
    produce a footprint spanning many bins.
  - Nearest-neighbour distances are measured in the *current* intermediate space at
    construction. The flow changes them slowly; recomputing on every rebin (2b item 3)
    is enough. Log the distribution of `eps` per object type.
  The neighbourhood gather radius is sized from the largest `eps` in the registry (in
  bins), not a fixed 3x3.

For a query ray `(theta, z)` and each slot `k` with pins in the ray's neighbourhood cells,
compute

```
K_i   = K((theta - theta_i) / eps_theta_i, (z - z_i) / eps_z_i)   # each pin's own footprint
w_k   = min(1, sum_i K_i)                                          # kernel mass, saturating
R_k   = sum_i K_i * r_i        / sum_i K_i                         # IDW pin radius
S_k   = sum_i K_i * r_target_i / sum_i K_i                         # IDW pin target
```

with `d = sqrt(u^2 + v^2)` and

```
K(d) = (1 - d) / (d + tiny)     for d < 1
     = 0                        for d >= 1
```

`K` is singular at 0, so a query on a pin's own ray reproduces that pin exactly
(`w=1, R=r_i, S=r_target_i`), and it reaches **zero at the footprint edge**, so each
pin's contribution vanishes continuously and `w, R_k, S_k` (hence `s`) are continuous
in the query ray. (A plain truncated `1/d` does not do this: it jumps from ~1 to 0 at
`d=1`, and `min(1, sum K)` would then saturate over almost the whole footprint.) With this
kernel `w=1` for `d <= 1/2` from an isolated pin and decays to 0 at `d=1`. `tiny` is
small enough that exactness holds to fp32 tolerance and large enough not to produce inf.
Note `r_target_i` varies with `theta_i` through the `theta/2pi` term; IDW over that is
fine because the pins are theta-local.

**Accuracy off the pin's own ray.** On a neighbouring ray inside the footprint, `R_k`
is an IDW of the pins' own radii, but the physical sheet on that ray sits at a slightly
different intermediate radius (the sheet has radial slope `dr/2pi` per radian plus
whatever tilt the flow leaves in z). So the anchor is off by roughly the sheet's radial
gradient times the offset to the nearest pin. The footprint rule (1.5x own-object
spacing, floored at ~3 voxels) bounds this: for a grid the error is about half a grid
step times the radial gradient, for an isolated pin about 3 voxels times it. State this
bound explicitly; the tolerance in unit test 5 is derived from it, not asserted.

Result per ray: a dense list over slots `k = 0..K-1` of `(R_k, S_k, w_k, valid_k)`. Slots
serve only target computation and cell indexing; 2a.4 orders the anchors by their actual
`R_k` and checks that the targets `S_k` agree with that order.

### 2a.4 The pinned monotone radial map

Let `s_free(r)` be the existing (unpinned) gap-expander map on this ray (piecewise linear through
`(R_free[k], (k + theta/2pi) * dr)`) -- exactly `GapExpandingTransform._inverse`
today, i.e. intermediate radius -> canonical radius. Take the ray's **valid** slots --
those with kernel mass `sum_i K_i > 0`; slots with zero mass are dropped *before* any
division, so `R_k, S_k` are never evaluated as 0/0 -- and **sort them by `R_k`** (their
actual intermediate radius on this ray, not their slot index) to get
`R_1 <= R_2 <= ... <= R_J` with the matching `S_j, w_j`. If the flow has swapped or
folded windings on this ray, the sorted `S_j` are not increasing; that is detected and
repaired by the ordering guard below and counted (`pin_order_violations`, with the slot
pair). Coincident `R_j == R_{j-1}` (`free_rise_j = 0`) is handled by treating the
interval as empty: `s(R_j) = a_j` from the guard, no interior points. Define the pinned
map `s(r)` with anchor `A_0 = (R_0 = 0, s(R_0) = 0)`:

```
for j = 1..J:
    free_rise_j = s_free(R_j) - s_free(R_{j-1})
    a_j_free    = s(R_{j-1}) + free_rise_j                       # what free map would give
    a_j         = w_j * S_j + (1 - w_j) * a_j_free                 # attenuated pin
    a_j         = s(R_{j-1}) + max(a_j - s(R_{j-1}), min_rise_j)   # ordering guard (hard max)
    s(r) for r in (R_{j-1}, R_j] =
        s(R_{j-1}) + (a_j - s(R_{j-1})) * (s_free(r) - s_free(R_{j-1})) / free_rise_j
after last pin:
    s(r) = s(R_J) + s_free(r) - s_free(R_J)
```

The guard is a **hard `max`**, not a smooth floor. A smooth floor of the form
`m + softplus(x - m)` perturbs every anchor, including those far above the bound, and
would destroy exactness everywhere; the hard max is the identity whenever the guard is
inactive and its derivative kink only appears in the violation regime, where exactness
is already lost.

Properties, all of which become unit tests:

- **Exact**: on a pin's own ray `w=1`, `R_j=r_i`, so `s(r_i) = r_target_i` exactly (the
  hard-max guard is the identity when inactive), unless the ordering guard fires or the
  pin is a coincidence conflict (2a.3).
- **Monotone**: every interval rises by a positive amount times a monotone function of
  `s_free`, so `s` is strictly increasing for any parameters; invertible.
- **Free where unpinned**: `w=0` reproduces `s_free` exactly (no pins => identical to
  unpinned transform).
- **Zero-mass anchors are transparent.** For an anchor with `w_j = 0`, `a_j = a_j_free`,
  and the interval rescale then reproduces the shape of `s_free`, so `s` on
  `(R_{j-1}, R_j]` equals `s(R_{j-1}) + s_free(r) - s_free(R_{j-1})`, and `s(R_j)` is
  exactly what the merged interval `(R_{j-1}, R_{j+1}]` would have given. The anchor's
  position `R_j` therefore has no effect on the map or on any downstream anchor, for any
  finite `R_j`. This needs the hard-max guard: with `min_gap < dr`,
  `min_rise_j = min_gap * free_rise_j / dr < free_rise_j`, so the guard is inactive on a
  free-shaped rise. More generally, an anchor's deviation from the free map is
  `w_j * (S_j - a_j_free)`, and `R_j` is a convex combination of pin radii, so the
  deviation is bounded and goes to zero linearly with `w_j`. A corollary: weak anchors
  cannot trigger the ordering guard, because `a_j` is close to the free value.
- **Continuous in (theta, z) wherever the guard is inactive.** `w` and `R, S` are
  continuous in the query ray (the kernel vanishes at the footprint edge), zero-mass
  anchors are transparent, so anchors enter and leave the sorted sequence continuously
  at footprint edges, wherever their `R` falls relative to the others. Sorting by `R` is
  continuous as long as anchors that cross in `R` have targets that agree with the new
  order. **Where the guard fires, `s` is not continuous.** If two anchors cross in `R`
  with targets out of order (or with equal targets but unequal weights), no strictly
  increasing map can honour both at the crossing radius, and `s(R_j)` jumps by about
  `S_a - S_b` (resp. by a weight-dependent amount) as they cross. This is unavoidable in
  any formulation that keeps the pins exact and the map strictly monotone; it is exactly
  the regime the warm-up (2a.1) is meant to remove, and every such crossing is a counted
  guard violation. A soft-merge policy (blend the two targets toward their weighted mean
  as `R_j - R_{j-1}` falls below a band) is recorded as a follow-up, to be tried only if
  the violation count is non-negligible after warm-up; do not build it in 2a.
- **Ordering guard**: `min_rise_j = min_gap * free_rise_j / dr` (scale the min gap by the
  free interval's winding count). When the guard is active exactness fails for that pin;
  log count and location per step (`pin_order_violations`), split into "targets out of
  order after sorting by `R`" (flow swapped or folded windings) and "targets in order
  but rise below `min_rise`" (pins closer in canonical space than the min gap allows).
- **Min gap is not preserved inside a rescaled interval.** The interval's rise is floored
  but the per-winding effective gap inside it is `free_gap * rise / free_rise`, which can
  fall below `min_gap` when `rise < free_rise`. The existing min-gap barrier acts on the
  free gaps and will not see this. Log the minimum effective gap per step
  (`pin_min_effective_gap`); do not act on it in 2a.

Inverse (canonical -> intermediate), needed by `slice_to_spiral_transform.inv` and the
DT loss: locate the interval by `searchsorted` over the anchors `s(R_j)`, invert the
affine relation to get `s_free(r)`, then invert `s_free` with the existing bracket search.

Implementation: new `PinnedGapExpandingTransform(GapExpandingTransform)` overriding
`_call`/`_inverse`; the Triton path (`gap_triton`) is bypassed in 2a (force
`_use_triton -> False` when pins are present). Pins are supplied per instance like
`_pinned_scaled_logits` is today. Keep `GapExpandingTransform` as the no-pin case so
the unpinned behaviour is recoverable with a config flag `model_pins_enabled`.

Also update `spiral_helpers` surface reconstruction: it evaluates the canonical sheet
through `.inv`; nothing changes except that `compute_pins()` must be run (with the full
registry) before export. Add an assertion in the export path.

### 2a.5 Losses in this stage: keep the originals

Do **not** change the loss functions yet. Expected behaviour, which is itself the
integration test:

- Patch radius loss, rel-winding, abs-winding, unattached-strip losses: **small, not
  zero**. These losses sample random points on each patch (`_sample_patch_tracks`), not
  the grid points that are pinned. Between grid pins the pinned map interpolates in ray
  space, which is not the same surface as the bilinear patch pushed through the flow, so
  the residual is of order patch curvature times grid spacing squared. Set an explicit
  tolerance for the integration test (a small fraction of the hinge margin) and log the
  residuals. If they are larger than that, either the registry disagrees with the loss's
  notion of a constraint (fix the registry), the kernel is too narrow (2a.3), or the
  ordering guard fired. PCL-only losses whose points *are* in the registry should be at fp
  noise.
- DT loss (`_patch_radius_and_dt_losses` DT branch and the track/unattached DT losses):
  the intended driver of `T_g`. The formulation, stated once so it is not misread:
  - pin target (what the pinned map sends `r_i` to): `dr * (T_g + n_i + theta_i/2pi)`;
  - DT target (what the DT loss pulls the sample toward): the **snapped** coordinate
    `dr * round(T_g + n_i)` shifted radius, built in spiral space and **detached** (as
    the loss already does with `target_spiral_zyxs.detach()`);
  - gradient into `T_g`: only through `slice_to_spiral_transform.inv` of that detached
    target, whose pinned anchors depend on `T_g`. Not through the rounding.

  The residual is then nonzero exactly when `T_g` is fractional: the inverse of the
  snapped point on a pinned ray sits at `r_i + dr * (round(T_g) - T_g) / s'(r_i)`, so the
  scroll-space distance is tent-shaped in `T_g` with its minimum at the integer and the
  gradient sign pushes toward it. Test 7b demonstrates this analytically on one pin.
  For pinned patches, the *input* to `dt_target_cache`'s whole-object selection
  (`select_whole_object_target`) is `T_g + n_i` instead of the cached unwrapped median;
  the cache still stores the rounded winding and the loss still snaps. (The earlier
  wording "use `T_g + n_i` directly" did not mean an unsnapped DT target: an unsnapped
  target would be inverted back onto the pin itself and carry no gradient into `T_g`.)
  Keep the median path for unverified patches and tracks, which stay soft. Verify that
  `T_g` drifts toward integers on a smoke fit.
- `T_g` is **not** moved by DT alone. Every loss that evaluates the transform near a pin
  (lasagna, normals, spacing, Dirichlet) has gradient into `T` through the pinned anchors,
  so image evidence can shift a whole component by a fraction of a winding. That is
  desirable but it competes with DT's pull to the nearest integer. Log the per-step
  gradient on `T` split by loss family so the balance is visible, and size the `T`
  learning rate for the total, not for DT alone.
- All lasagna/normal/spacing/fiber-direction/Dirichlet losses: unchanged in form.
- Optimiser: `T` gets its own param group; learning rate such that one step moves `T_g`
  by ~0.01 windings at most.

### Tests

Unit (pure functions, CPU ok):

1. `test_pinned_map_exact_on_pins`: random `s_free` tables, random consistent pins with
   `w=1`; `|s(r_i) - r_target_i| < 1e-5`.
2. `test_pinned_map_monotone`: random parameters including adversarial `S_j` ordering and
   `R_j` out of slot order; `s` strictly increasing on a dense radial grid; guard
   counters match hand-counted violations (split by cause); pins not touched by the
   guard are exact; the hard-max guard leaves every non-violating anchor bit-identical.
2a. `test_anchor_crossing_matrix`: pairs of anchors whose `R` cross as a parameter is
   swept, for each of: (i) targets in the new order, equal weights -> `s` continuous
   through the crossing, both exact after; (ii) targets out of order -> guard fires on
   the far side, `s(R)` jumps by about `S_a - S_b` at the crossing, one violation counted
   with the right cause; (iii) equal targets, unequal weights -> the jump is bounded by
   `|w_a - w_b| * |S - a_free|`, counted; (iv) exactly coincident `R` (`free_rise = 0`) ->
   empty interval, `s` finite and monotone, no NaN.
2b. `test_pin_table_multi_turn_patch`: one component whose points cover more than one turn
   puts two pins with different slots on the same ray; both are exact. Also a cell with
   several hundred pins (dense grid, wide footprint) is represented completely by the CSR
   table and every one of its pins is exact.
2c. `test_pin_coincidence_pass`: two pins in one slot on the same ray with radii within
   the conflict tolerance are merged and exact; with radii outside it they are reported
   as a conflict (fold vs cross-component tag correct) and neither is claimed exact.
3. `test_pinned_map_no_pins_is_free`: `w=0` for all pins -> bitwise-close to the unpinned transform.
4. `test_pinned_map_inverse_roundtrip`: `inv(s(r)) == r` to 1e-4.
5. `test_pin_lookup_continuity`: `s(r)` at rays straddling a cell boundary differs by
   `O(delta_theta)`; the same across the theta=0 seam and across a footprint edge
   approached from both sides (the kernel is zero there, so the *complete map*, not just
   `S_k`, matches the free map on both sides to fp tolerance), including the case where
   the appearing anchor's `R` lies strictly between two existing anchors' `R`; zero-mass
   slots are never divided (no NaN/inf under `torch.autograd.detect_anomaly`); outside
   every pin's own footprint, equals the free map. With pins on a regular grid
   and the default spacing factor, the map between pins stays within the off-ray bound
   from 2a.3 (sheet radial gradient times half a grid step) of the pin targets (no
   reversion to the free map between neighbouring pins).
5c. `test_pin_seam_crossing`: a pin whose post-flow angle moves from `+0.01` to `-0.01`
   radians between two steps keeps the same target radius to fp tolerance (`n_i(t)`
   adjusts by one) and triggers a slot rebuild.
5b. `test_pin_footprints`: a registry with one dense patch grid, one sparse PCL chain
   and one isolated point in the same slot. Each pin's `eps` equals the spacing rule
   (grid stride for the grid, chain step for the chain, the floor for the isolated point);
   the dense grid's pins do not contribute to rays more than one grid step outside the
   patch; the chain's pins do bridge the chain's own gaps.
6. `test_registry_offsets`: synthetic patches + PCLs with known winding annotations and a
   theta=0 crossing; `n_i` matches hand-computed values; inconsistent cycle is reported.
7. Gradient check (`torch.autograd.gradcheck`, double) of `s(r)` w.r.t. `T`, pin radii and
   logits on a tiny instance.
7b. `test_dt_gradient_on_T_one_pin`: one pin on one ray, free map identity, `T` swept
   through `k - 0.4 .. k + 0.4`. The DT residual (scroll distance between the pin and
   the inverse of the detached snapped target) equals `dr * |round(T) - T| / s'(r_i)` to
   fp tolerance, is zero at the integer, and `d(residual)/dT` has sign `-(round(T) - T)`
   on both sides. Also confirm the residual is identically zero and has no gradient when
   the target is (wrongly) left unsnapped.

Integration:

8. Synthetic scroll: an Archimedean spiral with a known smooth deformation, sample
   "patches" from it, add PCLs with correct offsets. Run the fit with pins from a flow
   that has the winding *ordering* right but is off in position (e.g. the true
   deformation scaled by 0.5, or plus smooth noise smaller than half a winding): the
   guard counter is zero, the constraint losses are within tolerance from step 0, and the
   reconstructed sheet passes through every registry point to < 0.1 voxel at export. Then
   the negative case: a flow that swaps two windings along some rays; the guard counter is
   nonzero exactly there, and the export report names those pins.
9. Real dataset smoke fit from an existing checkpoint: constraint losses within tolerance,
   guard count reported, `T_g` moving toward integers under DT, lasagna losses not worse
   than control after equal wall-clock (they will be slower per step -- record it, along
   with the registry size and the per-step cost of `compute_pins`).

### Done when

Exactness and monotonicity tests pass; export goes through every constraint point not
flagged by the guard or the coincidence pass; original losses report within tolerance on
the constrained sets; the guard count, conflict count and registry size are logged.

---

## Stage 2b. Performance

Only after 2a is correct. Each item independently checkable against 2a as a reference
implementation (same inputs, identical outputs to fp tolerance).

1. **Pin subsampling.** Push a random subset of registry points per step (config
   `sample_count_pins`), stratified by component so every component keeps some pins.
   Exactness then holds for the sampled subset during training and for the full set at
   export (`compute_pins(full=True)`). Test: export with full pins is exact; training-time
   pin count controls step time roughly linearly.
2. **Ray-specialised path.** Loss samples that share `(theta, z)` (phase-bundle rays,
   registration targets; `ray_specialized_spiral_to_scroll`) evaluate the pinned map once
   per ray. Extend that path to build the anchor list once per ray and gather per sample.
3. **Amortised binning.** Rebuild the pin table's cell membership and the per-pin
   footprints every `k` steps (config `model_pin_rebin_interval`), recompute only
   `(R, S, w)` from fresh pin positions each step. Slot changes (`round(T_g + n_i)`
   flipping) still force a rebuild. Test: identical to per-step rebuild whenever no pin
   crossed a bin margin or changed slot, and footprints drift by less than a stated
   fraction between rebuilds on a real fit.
4. **Triton kernel.** Port 2a.4 to `gap_triton` alongside `gap_bracketing_radii` /
   `gap_search_radii`: per sample, gather the ray's anchor list and run the
   forward/inverse interval logic. If the kernel needs a padded fixed-capacity table
   (`model_pin_max_per_cell`), that is an explicitly approximate training-only path: it
   reports `pins_dropped` per step, is off by default until measured, and export always
   uses the uncapped CSR path (asserted). Test: eager vs Triton agree to fp-association
   tolerance on the 2a test suite, forward and backward, whenever no pin was dropped.
5. **Memory.** Pins are a shared leaf; make sure `retain_graph` is not needed (same
   pattern as `_pinned_scaled_logits`). Profile steady-state VRAM vs the unpinned model.

Target: step time within ~1.3x of the unpinned model at the same sample counts.

### Status (2026-09-17)

Items 1-3 and 5 are implemented (`sample_pin_subset` / `compute_pins(full=)`,
`pinned_map_inverse_rays` in `ray_specialized_spiral_to_scroll`, the layout cache
in `build_pin_table` with `refresh_pin_footprints`, and the pins shared leaf).
The sampled subset is redrawn only when the layout is rebuilt, since a fresh
subset would otherwise force a rebuild every step. Item 4 is implemented as three
uncapped Triton kernels in `gap_triton.py`, each with an eager fallback and a
fused-vs-eager test in `tests/test_pins_stage2b.py`:

- `pinned_affine_map`: the per-ray piecewise-linear radius deformation (the
  interval search plus affine evaluation), forward and backward.
- `anchor_sums`: the CSR pin walk that accumulates the singular kernel into the
  per-(ray, slot) sums, with a recompute backward (no (query, pin) pair list is
  ever materialised, so no pin is dropped and there is no `pins_dropped` path).
- `tridiagonal_solve`: the batched Thomas solve of the radius blend, with the
  adjoint solve as backward (replaces a Python loop over anchor columns).

The remaining eager work per chunk is the `[chunk, anchors]` merge/guard
arithmetic, which is launch-bound; the chunk default is therefore 65536.

Measured on the synthetic production-scale benchmark
(`tests/benchmark_pins_scale.py`: 130 windings, 225k registry pins in 9000
patch components, 100k pins sampled per step, RTX 3090 shared with another
job, `--points 98304 --rays 4000 --rebin-interval 8`, forward + backward,
medians of 8):

| path                         | unpinned | pinned  | ratio |
|------------------------------|----------|---------|-------|
| transform build per step     | 0.15 ms  | 10.8 ms |       |
| generic inverse, 98k points  | 39.5 ms  | 153 ms  | 3.9x  |
| generic forward, 98k points  | 38.1 ms  | 165 ms  | 4.3x  |
| ray bundles, 4000 x 32       | 32.3 ms  | 58 ms   | 1.8x  |

Before this pass the same pinned paths took ~2.1 s (a per-pin device sync in
the coincidence conflict report) and, once that was removed, ~12x unpinned with
3x the memory. The 1.3x target is **not** met for the generic paths: the
per-chunk cost is now the fused anchor walk (~22M kernel evaluations per 16k
rays at this pin density, GPU-bound) plus the eager anchor merge. Further
gains need either fewer sampled pins per step (`sample_count_pins`, the cost is
roughly linear in it) or fusing the anchor merge/guard into the anchor kernel.
Retained memory is ~13 KB per query point (2.9 GB at 98k points).

**Ordering guard made local (2026-09-18).** The 2a guard added every lift to all
outer anchors (a cumulative sum), so on real rays carrying ~70 anchors with a
fraction of them out of order, almost every pin ended up inexact and the pinned
map missed its targets by far more than the free map did (600-step smoke fit:
median miss 11.4 windings pinned vs 1.06 free; 96% of pins inexact with 18% of
anchors guard-active). `build_pinned_ray_map` now applies
`resolved_j = max(solved_j, resolved_{j-1} + min_rise_j)` (a running max), re-solves
the blend with lifted anchors' radii as their desired radii so weak anchors follow
them (zero-mass anchors stay transparent), and guards once more. Only anchors whose
own target sits below the floor are inexact, as the plan text always intended.

Note on fp32 gradients: gradients into ray angles and pin positions pass through
the singular kernel's derivative and are resolved by fp32 only to ~1e-2 relative
in *either* implementation (both differ from a float64 reference by O(1) on
rays close to a pin); the eager and fused paths agree with each other to that
tolerance, and to fp-association tolerance on every other gradient.

---

## Stage 3. Strain losses in place of the constraint losses

### Rationale

After stage 2 the constraint losses are identically ~0 and carry no gradient. They used
to do a second job: pull the flow toward configurations where the constraints hold
*without* correction, which is what keeps the coordinate system smooth. Restore that
signal as a loss on the correction each pin has to absorb.

### Change

New `get_pin_strain_loss` in `losses.py`, evaluated on the pins already computed this step:

```
strain_i = s_free(r_i; ray_i) - r_target_i          # canonical-radius units
loss     = mean_i  relu(|strain_i| - margin)          # margin in units of dr, config loss_margin_pin_strain
```

Optionally a scroll-space variant mirroring `radius_loss_inv`: map `r_target_i` back
through the *free* inverse and take the scroll-space distance to `x_i`. Start with the
spiral-space form; it is one gather per pin.

Strain is zero exactly when the pin is not needed. Its gradient flows into the flow
chain (via `r_i`), the gap logits (via `s_free`) and `T` (via `r_target_i`). Note: on `T`
it is a *centring* force (pulls `T_g` to where the free map already puts the component),
which competes with the DT loss's pull to the nearest integer; keep the strain weight on
`T` lower than DT or detach `r_target_i` in the strain term (config
`loss_pin_strain_detach_targets`, default true -- strain then shapes only the flow and
gaps, DT alone shapes `T`).

Then, by config flags defaulting to the new behaviour:

- `loss_weight_patch_radius`, `loss_weight_rel_winding`, `loss_weight_abs_winding`,
  `loss_weight_unattached_pcl` -> 0 for pinned inputs. Their functions stay for
  unverified inputs, which remain soft.
- `loss_weight_pin_strain` new, roughly the old patch-radius weight.
- DT losses on pinned inputs: keep, tent-shaped as now, acting on `T` (and, through
  `.inv`, weakly on everything else). DT on unpinned inputs (unverified patches, tracks):
  unchanged.
- Normals, spacing, fiber-direction, Dirichlet: unchanged. They are the only orientation
  signal for 1D/0D constraint sets and for everything unannotated; do not lower their
  weights in this stage.

### Tests

1. `test_strain_zero_iff_free_map_satisfies`: construct a case where `s_free` already
   passes through the pins -> strain 0; perturb one logit -> strain > 0 with gradient of
   the right sign on that logit.
2. `test_strain_matches_old_radius_loss_in_limit`: for a single patch with `T` fixed,
   strain equals (up to hinge) the old radius loss evaluated on the *free* transform.
3. Gradient check of the strain loss w.r.t. flow-field parameters on a tiny model.
4. A/B fits on the synthetic scroll and one real dataset: stage 2 (old losses, ~0) vs
   stage 3 (strain). Compare composite-map Dirichlet energy and lasagna losses at equal
   steps; expect stage 3 to be smoother around constrained regions with no loss of
   exactness (which is structural).
5. Regression: `T_g` still converges toward integers under DT with strain enabled.

### Done when

Constraint losses are removed from the pinned path, strain gives a measurable smoothness
gain on the A/B, exactness and monotonicity tests still pass.

### Status (2026-09-21)

Implemented: `losses.get_pin_strain_loss` (spiral-space form, hinge at
`loss_margin_pin_strain` windings, `loss_pin_strain_detach_targets` default true),
evaluated in the step on the pins leaf with the step's own gap stage, as its own loss
family with `loss_weight_pin_strain`. While pins are active and
`loss_pins_replace_constraint_losses` is on (default), the patch-radius, rel/abs-winding
and unattached-strip radius weights are zero; their DT terms and the unverified-input
losses are untouched; the unweighted patch-radius value is still logged
(`patch_radius_unweighted`) along with `pin_strain_median/p90/frac_over_margin`. Tests in
`tests/test_pin_strain.py` (zero iff the free map satisfies the pins, descent direction
on the gap logits, agreement with the free-map winding deviation, detached targets, a
float64 gradcheck). The "T converges to integers" regression is deliberately not built:
integer-ness is not a goal at this point.

Registry bug found by the strain diagnostics (fixed 2026-09-21): cross-patch PCL and
fiber-chain points outside the flow z domain (whole-scroll inputs extend far beyond a
z-range fit: 18.8k of 23k chain pins here) were pins. The transform is undefined there,
so they landed on clamped z bins with targets a median 16 windings from the free map,
were pinned on every step, and dominated the strain. `build_pin_graph(z_range=...)` now
drops such points (cutting chains there) and `finalize_registry` drops patch quad
centres outside the domain.

Prerequisite found on the way (committed separately): the training-time pin subset must
be the pins of the patches and PCLs the step's losses sample (all of them, at registry
footprints), not a thin uniform subsample with widened footprints; the latter left the
constraint losses unchanged on the training transform even though the full-registry
export was aligned. Note that the strain loss's hinge is essential: at margin 0 every
satisfied pin sits on the |strain| kink and, since gaps accumulate outward, any change
gives them nonzero strain.

---

### Where the pinned fit stands (2026-09-22)

Measured on the real z 10000-11000 fit (5000 unpinned warm-up, then pinned with strain),
relative to each patch's own median shifted winding (no integer snapping), over all
quad centres of the pinned verified patches:

| | quad centres within 0.5 winding | patches with every quad within 0.5 |
|---|---|---|
| free stage | 91% | 31% |
| pinned, all patches | 96% | 75% |
| pinned, patches with free-map spread > 1 winding not pinned (240 of 9309) | 97.3% | 81% |
| pinned, spread > 0.5 not pinned (590) | 97.9% | 84% |

3000 pinned steps give the same numbers as 300: the residual is structural, not a matter
of training time, and quadrupling the strain weight barely moves the strain plateau
(median ~0.13 winding). The residual is the ordering guard: on a ray the radial pinned
map cannot swap points, so two components whose free-map radial order disagrees with
their targets cost one of them its exactness. Half of those conflicts are whole-winding
target disagreements between components (independent per-component estimates; overlap
linking reduces them but adds inconsistent cycles), half are components the free map
bends across a neighbour at their edges. 9% of patches hold 57% of the inexact pins;
excluding them (`model_pin_max_patch_spread_windings`) is the cheapest lever so far.
Integer-snapped frozen targets (`model_pin_targets_integer`) lift the integer-snapping
satisfaction metric from 58% to 89% of area but are slightly worse on the fractional
yardstick. A joint integer assignment (`model_pin_targets_joint`: coincident cross-
component pin pairs give integer relations, solved as robust potentials over the
component graph) changes 826 of 9194 components and removes a fifth of the same-sheet
conflicts, but pin inexactness stays at ~11% (95.6% vs 95.3% of quads): the swapped-
order class dominates once same-sheet disagreements shrink.

The flow field matters more. Same warm-up/pinning protocol, fractional targets, no
exclusion, quads within 0.5 winding of own patch median / patches fully within:
cartesian lattice 16: 96.0% / 75%; B-spline lattice 16: 96.4% / 80%; B-spline lattice
24: 96.9% / 82% (free stage 42% of patches fully flat vs 31%); B-spline lattice 32:
97.1% / 82% (plateau). The C2 B-spline with a coarser lattice represents the sheet-scale
bends the pins need better than the trilinear field does, and it is the one change that
helps without discarding inputs. Combined with the spread filter at 1.0 winding (152
patches, 2.3% of quads, not pinned and given no soft loss): 98.0% / 87%, with the best
free stage too (49% of patches fully flat). Keeping the soft radius loss on the
spread-excluded patches instead drags the free stage down to 28% while leaving the
pinned numbers unchanged, so they get neither pins nor soft loss.

The spread cap was a proxy: a lone patch is always satisfiable by the radial map, and
what breaks pins is *pairwise* inconsistency, two constraints disagreeing about their
relative winding on shared rays. `model_pin_demote_conflicting_patches` demotes on that
directly (cross-component pin pairs within 30 voxels, compared the way the ray map
treats slots; greedy demotion of the patch with the highest conflicting-to-agreeing pair
ratio). With `model_pin_demote_conflict_ratio` 0.5 (disagrees more than it agrees) on the
B-spline 24 fit it demotes 346 patches (4.3% of pins) and gives 98.5% / 88% (spread cap:
98.0% / 87% with 2.3% excluded); with ratio 0 it demotes 1701 patches (19% of pins) and
the kept set is essentially exact (99.8% / 93%), which is the trade-off curve. The
demoted set is recorded in the registry (`excluded_patches`) and gets no soft loss.
Only 35 of the 152 spread-capped patches are among the 346 demoted, so the spread cap
was not even a good proxy; it has been retired (`model_pin_max_patch_spread_windings`
is a retired key). Demotion is on by default with ratio 0.5, is re-checked against the
current free map every `model_pin_demote_recheck_interval` steps (reinstating patches
that no longer conflict; the full registry is kept from finalisation or rebuilt on
resume), and every decision is appended to `pin_demotion.jsonl` in the run directory
with each demoted patch's id, pair counts and most-conflicting neighbours, as a review
queue for the annotations.

## Out of scope for these stages (recorded so they are not lost)

- Discrete optimisation of `T` (integer search, difference-constraint feasibility). `T`
  stays fractional under the tent-shaped DT loss throughout.
- Demoting individual constraints from pinned to soft (a per-point flag plus a registry
  rebuild path); stage 2a only reports inconsistencies.
- Acting on fold detection. A fold is the same `(g, n_i)` -- i.e. the same slot from the
  same component -- hitting a ray twice after the flow. (A component hitting a ray twice
  with *different* `n_i` is a legitimate multi-turn patch, not a fold.) After sorting by
  `R` it shows up as an ordering-guard violation in 2a.4 and is logged, not acted on.
- Enforcing `min_gap` inside rescaled intervals (logged only, see 2a.4).
- Soft-merging anchors that cross in `R` with conflicting targets (see 2a.4 continuity);
  only if guard violations persist after warm-up.
- Demoting or resolving coincidence conflicts (2a.3); they are averaged and reported.
- Restating the model as a single sheet `R(u, z)` with unwrapped angle `u`; the pinned
  table is equivalent, and the existing seam/unwrap machinery is reused as is.

## File touch list

- `transforms.py`: stage 2 (`PinnedGapExpandingTransform`,
  `compute_pins`, `get_slice_to_intermediate_transform`, `T` parameter, shared leaves).
- `pins.py` (new): registry, binning, ray lookup, pinned map forward/inverse.
- `gap_triton.py`: stage 2b kernel.
- `losses.py`: stage 3 `get_pin_strain_loss`; stage 2 DT target simplification.
- `fit_spiral.py` (`FitContext`): build registry, call `compute_pins` per step and at
  export, param groups, logging of guard violations and constraint-loss residuals.
- `config.py`: `model_pins_enabled`, `model_pins_warmup_steps`,
  `model_pin_kernel_spacing_factor`, `model_pin_kernel_min_theta_radians`,
  `model_pin_kernel_min_z_voxels`, `model_pin_kernel_max_theta_radians`,
  `model_pin_kernel_max_z_voxels`, `model_pin_coincidence_frac`,
  `model_pin_conflict_tolerance`, `model_pin_max_per_cell` (2b, approximate path only),
  `sample_count_pins`,
  `model_pin_rebin_interval`, `loss_weight_pin_strain`, `loss_margin_pin_strain`,
  `loss_pin_strain_detach_targets`, `optimizer_lr_pin_targets`.
- `spiral_helpers.py`, `flatten_spiral_checkpoint.py`: ensure full pins at export.
- `tests/`: as listed per stage.
