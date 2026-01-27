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

## 4) Add visualization output (loss map)

- Import the new loss module in [`vis.py`](../vis.py:1).
- Compute the loss map once in [`vis.save()`](../vis.py:154) (next to the other maps).
- Add an entry to `loss_maps` so a `res_loss_<suffix>_<postfix>.tif` gets written.

## 5) Sanity check

- Run a syntax check:
	- `python -m py_compile <changed_files>`
