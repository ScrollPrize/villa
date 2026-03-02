> **No backwards compatibility**: The 3D model is a complete rewrite. No backwards compatibility with the 2D model (archived in `old_2d/`). The 2D UNet inference is reused, but fitting code is independent.

# 3d Lasagna model specs

- the 3d lasagna
    - models a wound up and distorted papyrus as a 3d tensor where:
        - width in the model is the 2d width of one sheet/winding (max) - maximum width corresponds to one winding but for now we only use arcs not a full winding
        - height is the height of the scroll - and the height of single columne
        - depth is the winding dimensions - we have one model sample of depth per winding that we model
    - pyramid: the model tensor is presented as a scale-space pyramid to ease optimization
    - note that a 2d slice of the model represents one 2d surface (sheet/winding) as a quadmesh grid
    - conn vectors - compared to the 2d model we still want the conn vectors between neighboring (+- 1 z in the model tensor) to point to a point on the next/previous winding and be orthongonal to the model surface (quadmesh surface). We process and model this the same as in the 2d model but with the 3d model this is of course a 2d point that intersects a quad - we regard the quad as a bilinear quad so we can calculate that intersection quickly on the quad level
- model saving should:
    - save the whole pyramid
    - save absolute volume positions
    - save the model extent bbox used (max min of xyz coords)
- model data: give a breif but detailed descriptions of the modalities we currently use
- data sizing and scaling, model data
    - mode arguments should always be relative to full scale (e.g. mesh-step is in full voxel coords)
    - model data should be in full voxel coords - and then scaled / shifted for grid sample or other applications
    - mesh-lr/hr same as in the 2d model
- sizing and sampling
    - the model shall be able to accept arbitrary data scaling and sub-sampling, losses should simply be sampled at the appropriate interpolated support points (e.g. we interopolate with grad sample in the 3d volumes)
- global transform & scales
    - the global transform should be an "arc" like transform + winding-step, so on init we want the model to follow a part of a circle
- visualization - lets leave that out for now.
- preprocessing
    - we want to continue to use a scaledown, but then process the model the same in xyz (all do just scaledown as stepping)
    - normal should continue to be fused separately (we need the three plane directions) but for cos and grad density:
        - use the 3 plane intersection normal estimate to fit a 3d normal
        - use the normal to
            - 1. calculate the weight how much each direction should influence the output - a slice tangential to the surface should obviousl not influence the estimate, while a slice orthogonal is best
            - 2. normalize the grad magnitude with the normal - as the 2d network observed along its slicing plane the distance can be stretched depending on the angle (and hence the gradient magniutde reduced proportionally)
            - this norm must _not_ be used for the data term (as fractional distances should stay the mostly independend of the angle)
- losses - relevant losses (leave out description and later also implementation of the other lossses)
    - dir_v - call it just dir - it should just enforce the model folloing the tangentials of the data (respectively orthogonal to the data normal)
    - step - mesh-step distance between quadmesh corners points
    - gradmag - same as in the 2d model - measured along the conn vectors!
    - data - same as 2d
    - data_plain - same as 2d
    for now these are the only losses we want to use!

---

# Technical details from 2D model analysis

Adapted as a spec for the 3D model. Only the relevant losses (dir, step, gradmag, data, data_plain, pred_dt) are described. All other 2D losses (dir_conn, data_grad, contr, mod_smooth_y, smooth_x/y, meshoff_sy, conn_sy, angle, y/z_straight, z_normal, corr_winding, min_dist, mean_pos) are omitted.

## Data modalities

Most data originates from a UNet applied to 2D slices of the CT volume along three orthogonal axes (z, y, x). The UNet outputs 4 channels per slice. Additional channels (pred_dt) may be derived from post-processing.

### cos (cosine) — periodic layer signal
- UNet channel 0. Oscillates between 0 and 1 as the slice crosses papyrus layers.
- Stored as uint8 in preprocessed zarr, decoded: `float32 / 255.0` → range [0, 1].
- In the 3D model: preprocessed from three axis volumes. The cos values from each axis-slice are **weighted by normal alignment** (see preprocessing section). Not affected by the angle normalization (division by cos(angle)).

### grad_mag (gradient magnitude) — sheet density
- UNet channel 1. Encodes the **density of papyrus sheets** = `|∇(fractional_winding_position)|`.
- The UNet predicts a fractional winding position field (`frac_pos`) derived from distance transforms to sheet skeletons → monotone normalization → iterative weighted averaging (see `gen_post_data.py:compute_label_supervision()`, `train_unet.py:compute_frac_mag_dir()`). `grad_mag` is the magnitude of the spatial gradient of this field — it measures how fast winding position changes spatially. High grad_mag = sheets closely packed; low = sheets far apart.
- Stored as uint8 with `grad_mag_encode_scale = 1000.0`, decoded: `float32 / 1000.0`.
- Optionally Gaussian blurred (configurable `grad_mag_blur_sigma`, default 4.0 in preprocessing).
- In the 3D model: each axis's grad_mag is both **weighted by normal alignment** AND **normalized by dividing by `cos(angle)`** between the slicing plane and the fitted 3D surface normal (see preprocessing section). The normalization corrects for the stretching effect of oblique slicing.

### dir0, dir1 — 2D direction encoding (180°-symmetric)
- UNet channels 2 and 3. Encode the local layer normal direction in the slicing plane using a double-angle representation:
  ```
  r² = gx² + gy² + ε
  cos2θ = (gx² - gy²) / r²
  sin2θ = 2·gx·gy / r²
  dir0 = 0.5 + 0.5·cos2θ           ∈ [0, 1]
  dir1 = 0.5 + 0.5·(cos2θ - sin2θ)/√2  ∈ [0, 1]
  ```
- This encoding is 180°-symmetric (no preferred sign for the normal).
- Stored as uint8, decoded: `float32 / 255.0`. Optionally blurred (configurable `dir_blur_sigma`, default 2.0).
- In the 3D model: all three axis dir0/dir1 pairs are **stored separately** in preprocessing (same as in the existing 2D model). There is no fusion into a single 3D normal — the dir loss compares the model's quad normal against each axis's direction encoding independently, weighted by slicing plane alignment.

### valid — data validity mask
- Binary mask (uint8, >0 → 1.0). Indicates where UNet inference produced valid output.
- Used for masking all losses: mesh points outside valid regions receive zero gradient.

### pred_dt — predicted distance transform (optional)
- Not a UNet output. Euclidean distance (in model-pixel voxels) to the nearest skeletonized predicted surface, stored as uint8 clamped to 255.
- Decoded as raw uint8 → float32 (no division — values stay as distance 0–255).
- May be absent from the preprocessed zarr; losses handle this gracefully (zero loss).

## Model representation

### Mesh pyramid (3D adaptation)

In the 2D model, the mesh is stored as `mesh_ms`: a multiscale residual pyramid with shape `(N, 2, Hm, Wm)` at the finest level. `N` = number of z-slices, `2` = (x, y) pixel coords, `(Hm, Wm)` = mesh grid size.

For the 3D model:
- The mesh tensor stores **3D positions** `(x, y, z)` → 3 coords per vertex.
- Shape at finest level: `(D, Hm, Wm, 3)` where D = depth (winding count).
- The pyramid has 5 levels by default. Each coarser level halves **all three** spatial dimensions D, H, W (min 2 each). Pyramid level i has shape `(D_i, Hm_i, Wm_i, 3)`.
- Reconstruction: integrate from coarsest to finest: `result = ms[-1]; for i in reversed(range(len(ms)-1)): result = upsample_crop(result) + ms[i]`. Upsampling is now 3D (trilinear) instead of 2D bilinear.
- All mesh positions are stored in **fullres voxel coordinates**. They are converted to grid_sample coordinates ([-1, 1]) only at sampling time by dividing by the volume dimensions.

### Connection offset pyramid

- Same pyramid structure as mesh (D, H, W all pyramided), shape `(D, Hm, Wm, 2)` — 2 channels for **(prev_winding_offset, next_winding_offset)**, i.e. ±D direction.
- These fractional row offsets tell the orthogonal intersection algorithm which row-segment in the **neighbor depth slice** (D±1) to target.
- Offsets are relative to the base row index: `target_row = base_row + offset`.
- The connection ray is cast along the quad surface normal and intersects a bilinear quad in the neighbor depth slice, as described in the top-level spec.

### Amplitude and bias (modulation parameters)

- `amp`: `(D, Hm, Wm, 1)`, clamped to [0.1, 1.0]. Scales the cosine target.
- `bias`: `(D, Hm, Wm, 1)`, clamped to [0.0, 0.45]. Shifts the cosine target.
- `target_plain = 0.5 + 0.5·cos(2π·x_index / periods)` — a periodic pattern with one full period per winding column.
- `target_mod = bias + amp·(target_plain - 0.5)`, clamped to [0, 1].
- These modulate the expected cosine appearance to account for local contrast/brightness variations.

## Coordinate system and grid sampling

### Coordinate spaces
- **Fullres**: raw CT voxel coordinates (x, y, z integers).
- **Model pixel space** (2D model): fullres / scaledown. The 2D model stores mesh coords in model pixels.
- **3D model**: mesh coords are stored in **fullres voxel coords**. Scaledown is applied only when converting to grid_sample coords or when comparing to downscaled data.

### Grid sampling
- Data volumes are 3D tensors indexed as `(C, Z, Y, X)`.
- To sample at mesh positions: convert fullres (x, y, z) to normalized [-1, 1] coords: `x_norm = (x / (W-1)) * 2 - 1`, etc. Then use `F.grid_sample(..., mode='trilinear', align_corners=True)`.
- The 3D model uses **trilinear** interpolation in 3D volumes (vs bilinear in 2D).

### Mesh LR/HR
- **LR (low resolution)**: the base quadmesh grid `(D, Hm, Wm, 3)`.
- **HR (high resolution)**: bilinear upsampled from LR by `subsample_mesh × subsample_winding` factors → `(D, He, We, 3)`. Used for the data loss (finer sampling → smoother loss landscape).

## Forward pass

1. **Reconstruct mesh**: integrate pyramid → coarse mesh `(D, Hm, Wm, 3)`.
2. **Apply global transform**: arc + winding-step transform (see init section).
3. **xy_lr**: the LR mesh `(D, Hm, Wm, 3)` — one 3D position per mesh vertex.
4. **xy_hr**: bilinear upsample of xy_lr to `(D, He, We, 3)` where `He = (Hm-1)*subsample_mesh+1`, `We = (Wm-1)*subsample_winding+1`.
5. **xy_conn**: for each mesh vertex, compute connection points to neighbor depth slices (see below).
6. **Sample data**: use grid_sample at xy_hr positions → sampled cos, grad_mag, dir channels.
7. **Compute targets**: target_plain and target_mod from amp/bias at HR resolution.
8. **Compute masks**: sample valid channel at LR/HR/conn positions; multiply by any stage masks.

## Connection vectors (3D adaptation)

### 2D model behavior
For each mesh vertex at column `c`:
1. Compute the vertical (v) direction via central differences along height (forward/backward at edges).
2. Rotate 90° to get orthogonal direction `d = (-vy, vx)`.
3. Cast ray from vertex in direction d.
4. Intersect with line segments in neighbor columns `c-1` (left) and `c+1` (right).
5. The target segment is selected by `conn_offset_ms` (fractional row offset from current row).
6. Returns `xy_conn (N, Hm, Wm, 3, 2)`: [left_conn, self, right_conn].

### 3D model adaptation
- Connections are in the **±D direction** (prev/next winding), not left/right.
- The mesh surface is a quadmesh in 3D: each cell is a **bilinear quad** defined by 4 corner positions.
- The orthogonal direction d is perpendicular to the quad surface (the quad normal).
- The intersection target is a quad in the **neighbor depth slice** (D-1 or D+1), not a neighbor column.
- **Intersection algorithm**: cast a ray from vertex along the quad normal and intersect with the bilinear quad in the neighbor depth slice. This is a ray-bilinear-patch intersection (can be solved as a quadratic in parameter space).
- Connection offsets specify which row in the neighbor depth slice to target, and the intersection is computed on the quad formed by `[row, row+1] × [height_i, height_i+1]`.
- Returns `xy_conn (D, Hm, Wm, 3, 3)`: [prev_winding_conn, self, next_winding_conn], each a 3D point.

### Post-step offset update
After each optimizer step, `update_conn_offsets()` adjusts the offsets to track the actual intersection parameter. This ensures the offsets stay aligned with the evolving mesh geometry.

## Losses (relevant subset)

All losses follow the pattern: compute a per-vertex loss map `lm` and a binary mask, return `(loss, (lm,), (mask,))` where `loss = masked_mean(lm, mask)`.

### dir — direction alignment loss

Enforces that the model surface tangent/normal aligns with the UNet-predicted direction. Evaluated at **LR** (base mesh) resolution.

**2D behavior**: For each LR mesh vertex, compute the vertical edge vector via **forward difference** along height (`p[j+1] - p[j]`, last row copies the previous row's vector). Rotate 90° to get the predicted surface normal direction. Encode via `_encode_dir()` (double-angle). UNet dir0/dir1 data is **bilinearly downsampled** to LR grid size. MSE:
```
loss = 0.5 * ((pred_dir0 - data_dir0)² + (pred_dir1 - data_dir1)²)
```
Mask: requires both the current vertex and its forward neighbor (j and j+1) to be valid. Last row copies the mask of the previous row.

**3D adaptation**: The model surface is a quad in 3D. Compute the **quad face normal** from cross product of the two quad edge directions. Compare the model normal against each axis's dir0/dir1 encoding separately (all three pairs are stored in preprocessing, no fusion). Each axis comparison is weighted by how aligned the slicing plane is with the model surface.

### step — mesh step distance

Penalizes vertical mesh edge lengths deviating from the configured step size. Evaluated at **LR** mesh positions only.

**Exact formula** (forward difference along height, j ∈ [0, Hm-2]):
```
len_v = ||p[j+1] - p[j]||     (Euclidean distance between adjacent mesh rows)
rel = (len_v - mesh_step_px) / mesh_step_px
loss = rel²
```
No mask (all vertices contribute). Output shape: `(D, 1, Hm-1, Wm)`.

**3D adaptation**: Same formula but using 3D Euclidean distance. The step is measured in fullres voxels (since mesh coords are fullres). The configured mesh_step is also in fullres voxels.

### gradmag — sheet-density period-sum loss

Integrates the sheet density (grad_mag) along the connection direction (±depth / winding). The integral over one winding period should equal 1.0 (exactly one winding traversed).

**Detailed algorithm (2D model)**:
1. Get connection points from `xy_conn` at LR: prev, center, next (in 2D: left, self, right).
2. **Upsample** connection endpoints to HR resolution (bilinear, by subsample_mesh factor).
3. Create **strip samples**: `strip_samples = subsample_mesh + 1` linearly interpolated points between each pair (prev→center) and (center→next).
4. **Sample grad_mag** at all HR strip points via `grid_sample`.
5. Compute physical strip length: `len = ||endpoint_diff|| × scaledown` (model pixels → fullres distance).
6. Compute normalized integral: `mag_normalized = grad_mag × ds × strip_samples` where `ds = len / strip_samples`.
7. Loss = `(mag_normalized - 1.0)²`, averaged over the two strips.
8. Masked by: connection validity (both endpoints valid) AND image-space validity of ALL strip sample points (amin across strip dim).

**3D adaptation**: The strip direction is **±depth (winding direction)**, running from the current vertex to its neighbor in the D±1 depth slice along the quad surface normal. Each strip is effectively a **mini-model**: three quadmesh surfaces (prev, center, next winding) with bilinearly interpolated points between them — the strip samples live on these interpolated surfaces. Sample grad_mag from the 3D volume at these 3D strip points via **trilinear** interpolation. Length computation uses 3D Euclidean distance. The scaledown factor is uniform in xyz. The integral of sheet density along one winding connection should equal 1.0.

### data — cosine MSE loss

MSE between the sampled cosine and the modulated target. Evaluated at **HR** (upsampled mesh) resolution.

**Exact formula**:
```
lm = (sampled_cos - target_mod)²
```
where `target_mod = bias + amp·(target_plain - 0.5)`, `target_plain = 0.5 + 0.5·cos(2π·x_index/periods)`.

Cos data is sampled via `grid_sample_px(xy_px=xy_hr)` at HR mesh positions. A `max_pool2d(kernel=(1,3))` is applied along the winding (W) direction to both the loss map (max-pool) and the mask (min-pool), making the loss robust to slight phase shifts.

Masked by `xy_img_mask` at HR positions, which combines the validity mask with any stage-scheduled masks.

**3D adaptation**: Same formula. Grid sample the 3D cos volume at HR mesh positions via trilinear interpolation. The max-pool along winding direction serves the same purpose.

### data_plain — unmodulated cosine MSE loss

Same as data loss but uses `target_plain` instead of `target_mod`:
```
lm = (sampled_cos - target_plain)²
```
No amp/bias modulation. Same HR sampling, pooling and masking as data loss.

### pred_dt — predicted distance transform loss

Penalizes mesh vertices that are far from predicted sheet surfaces. Evaluated at **LR** (base mesh) resolution.

- The preprocessed zarr may contain a `pred_dt` channel: Euclidean distance (in model-pixel voxels) to the nearest skeletonized predicted surface, stored as uint8 clamped to 255.
- Sample `pred_dt` at LR mesh positions via `grid_sample`. The sampled value is the loss map directly (higher distance = higher loss).
- Masked by `mask_lr`.
- If `pred_dt` channel is absent from data, the loss is zero (no compute).
- Default weight 0.0 — must be explicitly enabled in config.

**3D adaptation**: Same — sample `pred_dt` from the 3D volume at LR mesh positions via trilinear interpolation.

## Optimization structure (3D adaptation)

### Stage-based optimization
- Optimization is configured via a JSON config with `base` (default loss weights) and `stages` (list of optimization stages).
- Each stage has: `name`, optional `grow` spec, optional `masks`, `global_opt` settings, optional `local_opt` settings.
- OptSettings include: `steps`, `lr` (scalar or per-scale list), `params` (which parameter groups to optimize), `min_scaledown` (skip finest N pyramid levels), `w_fac` (per-loss weight multipliers), `default_mul` (global weight multiplier).

### Loss weight system
- `lambda_global`: dict of all loss names → base weights.
- `base` in config JSON overrides these defaults.
- Per-stage `w_fac` multiplies specific weights: `eff[name] = base[name] × w_fac[name]`.
- `default_mul` multiplies all weights not explicitly in w_fac.
- `_need_term()` returns 0.0 for disabled losses → they are skipped entirely (no compute).

### Parameter groups
- `mesh_ms`: multiscale 3D mesh pyramid levels (D×H×W all pyramided). Each level gets its own learning rate from the lr list (last = finest).
- `conn_offset_ms`: same 3D pyramid structure for connection offsets (D×H×W pyramided).
- `amp`, `bias`: modulation parameters, use the finest LR.
- Global transform parameters: arc center, radius, angular extent (only if global transform is enabled).

### Optimizer
- Adam optimizer with per-parameter-group learning rates.
- After each step: `model.update_conn_offsets()` re-aligns connection offsets.
- `model.update_ema()` updates exponential moving averages of mesh positions (used by mask scheduling).

### Growing (3D adaptation)
- Model can grow by adding mesh vertices in **6 directions**: ±height (H), ±width (W), ±depth (D).
- Since D is pyramided, growing in the D direction extends all pyramid levels.
- H/W directions: **linear extrapolation** for mesh positions, **edge copy** for conn_offsets/amp/bias.
- D (±depth) directions: **edge copy** of the boundary depth slice for all parameters.
- After grow: local optimization freezes the old region via `const_mask_lr` (gradient masking) and optimizes only the newly added region.
- For D grows: new data may be loaded (preprocessed zarr reload or UNet inference on new volume region).

### Mask scheduling
Two mask types (not needed for initial 3D model, but noted for completeness):
- **central_winding_pie_ema**: Triangle/wedge mask expanding from mesh center, using EMA-smoothed mesh positions to determine the wedge geometry.
- **constant_velocity_dilation**: 3D morphological dilation from a seed voxel at the mesh center, growing at a configured rate in fullres voxels per iteration. Separate XY/Z accumulators handle anisotropic grid spacing.

### const_mask_lr (gradient masking for grow)
- A `(D, Hm, Wm, 1)` binary mask applied as a gradient hook on mesh_ms[0] and conn_offset_ms[0].
- `1.0` = frozen (gradients zeroed), `0.0` = free to optimize.
- An `opt_window` parameter creates a transition zone at the boundary between old and new regions.

## Init and global transform

### 2D model init
- Mesh initialized as a regular grid centered on the crop region, covering `init_size_frac` of the crop extent.
- Grid spacing = `mesh_step_px` (height) × `winding_step_px` (width).
- Global transform: rotation `theta` + `winding_scale` around the mesh center.

### 3D model init
- The global transform is an **arc transform + winding-step** (no rotation/scale parameters like the 2D model).
- On init, the mesh follows a **circular arc** (partial winding) with configurable radius and angular extent.
- Arc parameterization: given center `(cx, cy)`, radius `r`, angle range `[θ_start, θ_end]`, each mesh column (W) maps to an angular position, and rows (H) map to height positions. Position: `(x, y) = (cx + r·cos(θ), cy + r·sin(θ))`, `z = height`.
- Each depth slice (D) is offset **radially** by `winding_step` in fullres voxels (next winding = same arc at radius ± winding_step).
- Optimizable parameters: arc center, radius, angular extent (or a subset thereof).

## Preprocessing (3D adaptation)

### Current 2D pipeline
1. Run UNet on z-slices → `(cos, grad_mag, dir0_z, dir1_z, valid)` per slice.
2. Optionally run on y-slices and x-slices → `(dir0_y, dir1_y)` and `(dir0_x, dir1_x)`.
3. Gaussian pyramid downscale by `scaledown` factor.
4. Blur grad_mag and dir channels.
5. Store as uint8 zarr with CZYX layout.

### 3D model preprocessing
- **Uniform scaledown**: apply `scaledown` in all three dimensions (z included), so 1 model voxel = scaledown fullres voxels in every direction.
- **Dir channels stored separately per axis**: all three axis-specific dir0/dir1 pairs are stored in the preprocessed zarr (no fusion). Same as the existing 2D model.

#### Normal estimation algorithm

Each axis's UNet outputs dir0/dir1 encoding the 2D gradient direction in the slicing plane via the double-angle representation. To decode the angle θ:

```
cos2θ = 2·dir0 - 1
sin2θ = cos2θ - √2·(2·dir1 - 1)
θ = atan2(sin2θ, cos2θ) / 2        # θ ∈ (-π/2, π/2]
```

Each axis constrains the 3D surface normal (nx, ny, nz) via a linear equation (the gradient direction in the slicing plane must be perpendicular to the surface normal projected onto that plane):
- z-slices (XY plane, gx→X, gy→Y): `nx·sin(θ_z) - ny·cos(θ_z) = 0`
- y-slices (XZ plane, gx→X, gy→Z): `nx·sin(θ_y) - nz·cos(θ_y) = 0`
- x-slices (YZ plane, gx→Y, gy→Z): `ny·sin(θ_x) - nz·cos(θ_x) = 0`

**Least-squares solve via cross products**: with 3 constraint rows (r₁, r₂, r₃), compute three cross products of row pairs to get candidate normals:
- n₁ = r₁×r₂ = (cos θ_z · cos θ_y,  sin θ_z · cos θ_y,  cos θ_z · sin θ_y)
- n₂ = r₁×r₃ = (cos θ_z · cos θ_x,  sin θ_z · cos θ_x,  sin θ_z · sin θ_x)
- n₃ = r₂×r₃ = (cos θ_y · sin θ_x,  sin θ_y · cos θ_x,  sin θ_y · sin θ_x)

Align signs so dot(nᵢ, n₁) ≥ 0, then sum: `n_avg = n₁ + n₂ + n₃`. Normalize to unit length. Cross product magnitude naturally weights by reliability — when two constraint planes are nearly parallel their cross product is small.

Output weights: `(w_z, w_y, w_x) = (|nz_component|, |ny_component|, |nx_component|)` of the normalized normal. These represent how orthogonal each slicing axis is to the surface.

#### Fusion formulas

The UNet observes grad_mag stretched by the slicing angle: `gm_observed = G_true · cos(α)` where `cos(α) = w_axis` (the weight for that axis). Angle normalization recovers the true gradient: `G_est = gm_observed / cos(α) = gm_observed / w_axis`.

**cos fusion** (weighted average, no angle normalization):
```
cos_fused = (w_z·cos_z + w_y·cos_y + w_x·cos_x) / (w_z + w_y + w_x)
```

**grad_mag fusion** (weight and normalization cancel):
```
gm_fused = Σ(w · G_est) / Σ(w)
         = Σ(w · gm/w) / Σ(w)
         = Σ(gm) / Σ(w)
         = (gm_z + gm_y + gm_x) / (w_z + w_y + w_x)
```

Equivalently: `gm_fused = sqrt(gm_z² + gm_y² + gm_x²)` (L2 norm, since `gm_axis = G·|n_component|` and `|n|=1`).
