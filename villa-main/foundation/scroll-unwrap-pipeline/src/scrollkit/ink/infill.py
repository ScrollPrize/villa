"""Papyrus infill for render textures.

The composites' alpha-0 regions mix two things: voids CONNECTED to
the chart boundary (the sheet's outside plus genuine edge bites — physically
missing papyrus, KEPT as holes) and INTERIOR-ENCLOSED dropout (compositor had no
texture data there; the papyrus wall is physically continuous). Rendering interior
dropout as cutouts perforates the sheet — and on the WOUND roll it opens windows
into the scroll interior: the camera sees the far wall's recto ink through the
near wall (a see-through roll showing interior ink). A real carbonized scroll
is a closed wall, so interior dropout is filled up to a size cap; only large
TRUE lacunae and boundary-connected bites remain holes.

Fill is a single-shot Voronoi fill (nearest valid texel via EDT) — completes any
region size, unlike bounded ring dilation, whose 64-iteration cap silently left
big regions' RGB unfilled while alpha was still forced opaque — followed by a
blur on filled pixels only plus matched papyrus grain so large fills don't read
as smooth plastic.

Originals are never modified — these are presentation textures for animation
rendering and frame-OBJ exports only (conversion deliverables keep exact source
textures).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def classify_holes(alpha: np.ndarray, *, alpha_min: int = 16,
                   max_fill_frac: float = 0.0005) -> dict:
    """Label invalid regions. Interior-enclosed components up to max_fill_frac of
    the papyrus area are dropout (fill); LARGER interior components are TRUE
    lacunae and stay holes. The size distribution typically shows a clean gap
    (a handful of components >= 0.1% of papyrus vs tens of thousands of dropout
    specks below 0.025%), which the cap sits inside. Components touching the
    canvas border are the chart's outside / edge bites (keep)."""
    valid = alpha > alpha_min
    lab, n = ndimage.label(~valid)
    if n == 0:
        return {"fill_mask": np.zeros_like(valid), "n_filled_components": 0,
                "n_kept_lacunae": 0, "filled_frac_of_papyrus": 0.0}
    border_ids = set(np.unique(np.concatenate(
        [lab[0], lab[-1], lab[:, 0], lab[:, -1]])).tolist())
    border_ids.discard(0)
    papyrus = float(valid.sum())
    sizes = ndimage.sum_labels(np.ones(lab.shape, np.float64), lab, np.arange(1, n + 1))
    big = set((np.flatnonzero(sizes > max_fill_frac * papyrus) + 1).tolist())
    keep = border_ids | big
    fillable_ids = np.array([i for i in range(1, n + 1) if i not in keep],
                            dtype=np.int64)
    fill_mask = np.isin(lab, fillable_ids)
    return {
        "fill_mask": fill_mask,
        "n_filled_components": int(len(fillable_ids)),
        "n_kept_lacunae": int(n - len(fillable_ids)),
        "filled_frac_of_papyrus": float(fill_mask.sum() / max(papyrus, 1.0)),
    }


def voronoi_fill(rgb: np.ndarray, valid: np.ndarray, fill_mask: np.ndarray) -> np.ndarray:
    """Fill fill_mask pixels with their nearest valid texel (EDT indices) — exact,
    single pass, any region size."""
    out = rgb.astype(np.float32).copy()
    _, (iy, ix) = ndimage.distance_transform_edt(~valid, return_indices=True)
    sel = fill_mask
    out[sel] = rgb[iy[sel], ix[sel]].astype(np.float32)
    return out


def dilate_inpaint(rgb: np.ndarray, valid: np.ndarray, fill_mask: np.ndarray,
                   *, max_iter: int = 64) -> np.ndarray:
    """Legacy ring-dilation fill (kept for tests/back-compat); prefer voronoi_fill."""
    out = rgb.astype(np.float32).copy()
    known = valid.copy()
    todo = fill_mask & ~known
    k = np.ones((3, 3), dtype=np.float32)
    for _ in range(max_iter):
        if not todo.any():
            break
        ksum = ndimage.convolve(known.astype(np.float32), k, mode="nearest")
        ring = todo & (ksum > 0)
        if not ring.any():
            break
        for c in range(out.shape[2]):
            csum = ndimage.convolve(np.where(known, out[..., c], 0.0), k, mode="nearest")
            out[..., c][ring] = csum[ring] / ksum[ring]
        known |= ring
        todo &= ~ring
    return out


def build_render_texture(rgba: np.ndarray, *, alpha_min: int = 16,
                         max_fill_frac: float = 0.0005,
                         blend_sigma: float = 2.5,
                         grain: float = 0.5,
                         seed: int = 0) -> tuple[np.ndarray, dict]:
    """RGBA uint8 → (infilled RGBA uint8, stats). Valid pixels untouched bit-for-bit;
    interior dropout gets Voronoi papyrus + blur + matched grain, alpha=255."""
    rgb = rgba[..., :3]
    alpha = rgba[..., 3]
    info = classify_holes(alpha, alpha_min=alpha_min, max_fill_frac=max_fill_frac)
    fill = info["fill_mask"]
    if not fill.any():
        return rgba.copy(), info
    valid = alpha > alpha_min
    filled = voronoi_fill(rgb, valid, fill)
    if blend_sigma > 0:
        sm = np.stack([ndimage.gaussian_filter(filled[..., c], blend_sigma)
                       for c in range(3)], -1)
        filled = np.where(fill[..., None], sm, filled)
    if grain > 0:
        # matched papyrus grain: replicate the valid region's high-frequency energy
        # so large smooth fills don't read as plastic
        g = rgb.astype(np.float32).mean(-1)
        hf = g - ndimage.gaussian_filter(g, 3.0)
        hf_std = float(hf[valid].std()) if valid.any() else 0.0
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, grain * hf_std, size=fill.shape).astype(np.float32)
        noise = ndimage.gaussian_filter(noise, 0.7)
        filled = filled + np.where(fill, noise, 0.0)[..., None]
    out = rgba.copy()
    out[..., :3] = np.where(fill[..., None],
                            np.clip(np.rint(filled), 0, 255).astype(np.uint8), rgb)
    a = out[..., 3].copy()
    a[fill] = 255
    out[..., 3] = a
    return out, info
