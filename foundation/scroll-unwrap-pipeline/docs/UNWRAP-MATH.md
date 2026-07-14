# Unwrap Kinematics Specification (binding contract)

The unwrap animates a wrap mesh from its 3D wound pose to its de-normalized,
isometric UV domain embedded as a plane in 3D.

## De-normalization (3a)
Least-squares fit of `‖Δp‖² ≈ (s_u Δu)² + (s_v Δv)²` over mesh edges recovers
the physical UV scales (the charts are SLIM output in voxel units, normalized
at texturing time). Anisotropy and relative residual are reported per mesh and
feed the gate thresholds.

## Target embedding (3b)
The flat target V1 places material at its de-normalized UV coordinates on a
plane whose v-direction is the scroll axis. The anchor strip (the winding's
outer, free end — selected radially) is rigidly registered to its source pose
so the sheet peels open from a fixed edge. A chirality check (`mirror_check`)
asserts the embedding preserves UV↔3D orientation; reading direction on screen
is a camera decision.

## Production kinematics: rolling-contact unroll — `scrollkit.unwrap.rolling`
- One continuous motion: the rigid field is defined on c ∈ [−c_pre, S+band];
  the negative range is the tip-onto-plane lean-in (slerp Identity → entry
  pose, arc-proportional), smoothed together with the contact field and driven
  by a single linear map of the eased timeline. Two-phase settle/roll designs
  meet at a zero-velocity junction and expose raw-pose jitter — avoided here.
- Landing coordinate `s` is the vertex's payout coordinate IN THE FLAT
  EMBEDDING (projection of V1 on the regression direction of V1 against the
  UV arc), NOT linear-in-u. Landing order must equal flat-space order: under
  chart warp the linear-in-u arc can disagree with the true flat position by
  thousands of voxels, letting late-landing patches travel through already
  settled sheet. With `s` from V1, a vertex flattens exactly when the contact
  line sweeps its own landing slot, making settled-behind / rolled-ahead exact
  and pierce-through impossible by construction.
- Material behind the contact lies flat (V1); material beyond keeps its EXACT
  source geometry under the per-contact rigid transform; a curvature-adaptive
  band (clip(0.5/κ, 4·med-edge, band), band = max(0.8% arc, 4·med-edge))
  blends between them.
- Rigid field: per-band Kabsch of source onto flat targets, full Gaussian
  smoothing (σ≈21), unified re-smooth (σ=5) after the lean-in prepend, then
  frame-0 exactness restored by a TAPERED correction over the first ~2σ grid
  samples (a hard re-pin of sample 0 alone leaves a field step that reads as a
  morph-start snap at large tip angles).
- Plane-clearance barrier (closed form): the rigid roll's only possible
  collider is the unrolling plane, so the whole body rides at
  `lift(c) = max(0, ε − min height of behind-contact points)`, ε = half the
  median edge. The smoothed lift takes the UPPER envelope with the raw
  requirement (plain smoothing can shave spiky needs and let pre-landing lobes
  dig through the plane). No end taper: forcing lift → 0 while the core is
  still wound pushes it into the flat sheet (coincident surfaces z-fight
  through the alpha cutout). No long ramp behind the contact: a raised sheet
  chases the very lift it exists to avoid, and a v-uniform ramp cannot thread
  between lobes at different v.
- Paper bridge: a SHORT raise of the sheet confined to the landing flap
  (≤ pad_arc, the zone the clearance certificate exempts as single material by
  construction), following a spike-free `lift_vis ≤ lift`, so the sheet meets
  the roll's tangent like paper off a roller without inheriting lift spikes.
- Chart sanitation before embedding: (1) `enforce_chart_injectivity` — a chart
  must be injective; duplicated trace patches (two layers mapping to the same
  UV within ~4 texels but 3D-distant) double-print the same texture content
  with an offset and flicker as the layers cross while landing; the layer
  deviating from the local UV-neighbour consensus is dropped. (2)
  `junk_tail_trim` — data-driven, no-op on healthy charts.

## Clearance certificate — `scrollkit.metrics.clearance`
Per frame, no rolled vertex may sit within 0.05 × median-edge of (or below)
the settled sheet at the same plan cell (landing flap exempt). Rolling paths
certify in EXACT-STATE mode: settled ⇔ s < c − pad_arc, rolled ⇔ s > c +
pad_arc, with state persisted in `frames.npz` (`s_arc`, `c_arc`, `pad_arc`).
Quantile gates are provably blind to sub-percent vertex populations, and
state inferred from geometry misclassifies near the contact — the certificate
therefore uses the kinematics' own state and is a hard, non-waivable gate.

## Fallback: deformation-gradient interpolation
Per-face deformation gradients (Sumner's construction) with the det(R)<0 SVD
fix, branch-consistent rotation logs via winding decomposition (scalar graph
unwrap of the axis angle), junk-weighted Poisson reconstruction with one
factorization reused across frames, and an escalation ladder over junk
suppression knobs. Used only when rolling fails its gates; the best-attempt
ranking is clearance-dominant so an interpenetrating fallback never ships in
favour of a clean rolling path.

## Easing & timeline (30 fps)
240 frames = hold rolled (20) / quintic-eased unwrap (195) / hold flat (25).
Reveal variant appends a tail (see RENDER-STYLE).
