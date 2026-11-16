"""Registry of external (Python) backprojectors: backprojector "python(fn=NAME, ...)".

    register(name, fn, filtered=True, host=True)
    fn(rows, weights, ctx, **params) -> (y1 - y0, x1 - x0) float32 slice of ctx["roi"]
      rows: (n_angles, n_det) ramp-filtered x balanced weights (filtered=True) or the -log rows (filtered=False);
      weights: balanced weights; ctx: angles (u = x cos a + y sin a + rot_center), rot_center, scale, center,
      grid_size, roi, clip_outer_circle, clip_radius, view_height, device. On CUDA, host=True passes NumPy copies,
      host=False the device arrays (DLPack).
The built-in methods (ghbp, fbp, sirt, mlem, gridrec) are C++/CUDA; this is for experiments only.
"""
from types import SimpleNamespace

_registry: dict = {}


def register(name, fn, filtered=True, host=True):
    _registry[name] = SimpleNamespace(fn=fn, filtered=bool(filtered), host=bool(host))
