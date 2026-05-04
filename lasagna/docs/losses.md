\# Losses: adding a new term (checklist)

## Rule: FitResult-only

- Loss code must only consume [`model.FitResult`](../model.py:23).
- Do not re-sample the image or recompute model grids inside the loss.

## 1) Implement the loss

- Create a new module like [`opt_loss_gradmag.py`](../opt_loss_gradmag.py:1) (or add to an existing `opt_loss_*.py`).
- Prefer a `*_loss_map()` function returning `(lm, mask)` and a scalar wrapper using masked mean.

Conventions:

- `lm` is a per-sample loss map (usually `(N,1,H,W)` or `(N,1,H,P)`), float.
- `mask` must match `lm` spatial shape and be derived via `min(...)` of all participating sample masks (never interpolate).

## 2) Wire it into optimization

- Add import in [`optimizer.py`](../optimizer.py:1).
- Add a new key to `lambda_global` in [`optimizer.load_stages()`](../optimizer.py:61).
- Add the term to the `terms` dict in [`optimizer.optimize()`](../optimizer.py:110), mapping name → loss function.

The term name string must match the stage config key.

## 3) Add a default weight in the stages JSON

- Add the term to the `base` map in [`stages_scalespace.json`](../stages_scalespace.json:1).

Notes:

- Some terms may need stage-initialization context computed once (e.g. `mean_pos` captures the current mean position at stage start and penalizes drift).

Optimizer stages can also optionally perform mesh growth + local optimization (see [`docs/model.md`](model.md:1)).

### Optional `pred_dt` dense-flow gate

`pred_dt` snapping can be gated per stage via `global_opt.args`:

```json
{
  "name": "snap",
  "steps": 1000,
  "params": ["mesh_ms"],
  "args": {
    "pred_dt_flow_gate": {
      "enabled": true,
      "flow_zero": 50.0,
      "flow_one": 300.0,
      "backtrack_distance": 10.0,
      "anticipatory_pull": {
        "enabled": true,
        "samples": 8,
        "inlier_zero": 80.0,
        "inlier_one": 120.0,
        "loss_weight": 1.0
      },
      "debug": true
    }
  }
}
```

When enabled, the current single-winding `pred_dt` render is median-filtered
with radius 1, thresholded at `110`, routed through `dense_batch_min_cut`, and
sampled at the exact model grid corners. The resulting gate is linearly mapped
from `flow_zero -> 0` to `flow_one -> 1` and multiplies the `pred_dt` loss map.
The loss denominator remains the original validity-mask sum; the gate is
intentionally not renormalized.

`backtrack_distance` is measured in the rendered `pred_dt` image pixel units.
It is passed through to the dense grid flow routing and matches the C++ debug
CLI option:

```bash
./dense_batch_preprocess -i pred.tif --source 240,240 --grid-step 4 --backtrack-distance 10
```

`anticipatory_pull` is optional and only runs with active flow gating. It scores
all one-step straight LR neighbor lines before flow weights are known, using
subsampled `pred_dt` values along each line. After flow returns, each candidate
whose root gate is higher/nonzero and whose tip gate is below 1 contributes an
independent straight pull to the tip corner, weighted by root gate and prefix
inlier score. The pull is not winner-take-all; multiple neighbor lines may
contribute to the same tip.

## 4) Add visualization output (loss map)

- Import the new loss module in [`vis.py`](../vis.py:1).
- Compute the loss map once in [`vis.save()`](../vis.py:154) (next to the other maps).
- Add an entry to `loss_maps` so a `res_loss_<suffix>_<postfix>.tif` gets written.

## 5) Sanity check

- Run a syntax check:
	- `python -m py_compile <changed_files>`
