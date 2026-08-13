# Correction Point Winding Loss (`--corr-mode winding`)

## Overview

The winding corr loss observes the continuous winding position of each correction point via grad_mag strip integrals, averages within collections (superconductor coupling), then builds proxy targets on the correct model quads to push/pull the mesh into place. It replaces the older snap and legacy modes as the default.

Select mode via CLI:

```
--corr-mode winding   # default
--corr-mode snap      # previous default (signed distance along conn normal)
--corr-mode legacy    # bracket + collection-coupled winding observation
--corr-snap 1         # backward compat alias for snap
```

## Concepts

### Correction points

Each corr point is placed in VC3D at a known 3D position `P` with a relative winding annotation `winda` (from `d.tif`). Points are grouped into **collections** — all points in a collection should agree on absolute winding (up to their relative `winda` offsets). The loss acts as a superconductor: it equalizes winding within each collection, then corrects the mesh.

### Four correspondences per point

For each corr point, the loss maintains up to 4 model quad anchors:

| Index | Name | Purpose |
|-------|------|---------|
| 0 | closest_low | Lower bracketing surface (signed distance flips sign) |
| 1 | closest_up | Upper bracketing surface |
| 2 | avg_low | Layer `floor(target_winding)` — correction target |
| 3 | avg_up | Layer `floor(target_winding) + 1` — correction target |

The closest pair is used for **winding observation** only (detached). The avg pair is used for **proxy correction** (gradients flow through model quad corners).

### Single-sided points

When a point is outside all model layers (no bracket), only the nearest surface is used. The winding estimate is `d_nearest +/- integral` and is included in the collection average only if the integral is < 1.0 winding. The point still participates in the correction loss as long as a valid avg pair quad exists.

## Algorithm

### Step 1: Winding observation (no_grad)

**Bracketed case**: project P onto closest_low and closest_up surfaces. Compute strip integrals (grad_mag) from each surface to P:

```
integral_low = strip_len(Q_low, P) * mean_grad_mag(Q_low -> P)
integral_up  = strip_len(P, Q_up)  * mean_grad_mag(P -> Q_up)
winding = d_low + integral_low / (integral_low + integral_up)
```

**Single-sided case**: only one surface. Winding estimate valid if integral < 1.0.

### Step 2: Collection averaging (no_grad)

Same +/- coupling as the legacy mode. For each collection:

```
avg_pos = mean(winding - winda)      target_pos = avg_pos + winda
avg_neg = mean(winding + winda)      target_neg = avg_neg - winda
```

Pick the coupling mode (pos or neg) with lower MSE. Points excluded from observation (bad strips, integral >= 1.0 for single-sided) still get a target from the collection average if other points in the collection are valid.

### Step 3: Proxy correction (with gradients)

For each avg surface (low and high), following the ext_offset_loss pattern:

1. Gather 4 model quad corners from `xyz_lr` **(WITH gradients)**
2. Project P onto quad -> detached `(u, v)` for bilinear weights
3. Compute bilinear model point Q (detached)
4. Strip integral from P to Q -> signed winding
5. Target offset: `-frac` for avg_low, `+(1-frac)` for avg_up
6. Error: `err = signed_winding - target_offset`
7. Proxy: `M_corner_det - gt_normal * err` (4 proxies per surface)
8. Loss: `sum(bilinear_weight * (M_corner - proxy)^2)`

The GT normal is sampled at the corr point position from the lasagna volume (hemisphere-encoded nx/ny).

### Sign convention

- avg_low (below P): target offset is negative. If surface is too far below, the proxy pulls it upward toward P.
- avg_up (above P): target offset is positive. If surface is too far above, the proxy pulls it downward toward P.

## Persistent anchor state

Anchors `(d, h, w)` per correspondence persist across optimization steps (module-level globals). `(u, v)` are recomputed each step via bilinear projection.

- **Init**: brute-force nearest-quad search across all layers, bracket finding, collection averaging, avg pair quad finding.
- **Update**: unclamped bilinear re-projection -> integer shift -> clamped re-project. Closest pair re-checks bracket validity. Avg pair re-initializes if the target layer changes.

## Files

| File | What changed |
|------|--------------|
| `opt_loss_corr.py` | New `_corr_winding_loss`, helpers, anchor state |
| `cli_opt.py` | `--corr-mode` arg (replaces `--corr-snap`) |
| `fit.py` | Calls `set_corr_mode()` instead of `set_snap_mode()` |
| `optimizer.py` | Debug print uses `_corr_mode` |

## Key functions in opt_loss_corr.py

- `_wind_brute_force_init()` — full initialization (bracket search, strip integrals, collection avg, avg pair quads)
- `_wind_update_anchors()` — per-step local anchor update with bracket re-check
- `_wind_strip_integral()` — vectorized signed/unsigned winding integral between two point sets
- `_wind_collection_average()` — +/- coupling collection averaging
- `_wind_nearest_quad_on_layer()` — per-point nearest quad on a given depth layer
- `_corr_winding_loss()` — main loss function: observation -> averaging -> proxy correction
