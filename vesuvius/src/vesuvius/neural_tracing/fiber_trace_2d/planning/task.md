# VC3D Requested-Level Blocking Coordinate Sampling

Fix native 3D Trace2CP rendering and inference-block loading so blocking VC3D
coordinate sampling really means strict requested-level sampling.

Requirements:

- `blocking=True` coordinate sampling must wait for every required requested
  zarr-level chunk to be fetched and decoded before image sampling begins.
- Required chunk decoded data must stay referenced/pinned for the duration of
  the sampling call so it cannot be evicted from the decoded chunk cache while
  rendering.
- Scale fallback must be disabled in blocking mode. A requested-level sample
  must never silently use a coarser zarr level.
- A chunk that is genuinely absent from the requested level may render as black
  fill. That is the only allowed black fallback.
- I/O or decode errors must fail loudly; they must not render as valid black or
  coarse data.
- Native 3D Trace2CP raw volume panels and prediction/presence panels must use
  the same fixed blocking semantics.
- Avoid using the returned sampler `valid_mask` as evidence of requested-level
  data. It is only a geometry/sample-coverage mask.
