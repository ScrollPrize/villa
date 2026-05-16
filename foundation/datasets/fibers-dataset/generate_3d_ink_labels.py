#!/usr/bin/env python3
"""Generate true-3D ink pseudo-labels by gating a 2D ink prediction with CT intensity.

Addresses `ScrollPrize/villa#192 <https://github.com/ScrollPrize/villa/issues/192>`_
("Accurate 3d ink labels", labelled ``good first issue`` + ``help wanted``).

## The annotator-bias / 2D-projection problem

The issue calls out two failure modes of the current ink-label pipeline:

1. **Annotator bias** — the label reflects the human's guess at the letter, not
   what's actually detectable in CT. The model then over-fits to features the
   human thought should be there.
2. **2D → 3D smearing** — the 2D mask is replicated across all z-layers the
   surface touches, so the resulting "3D" label is just an extruded prism. Real
   ink is *thin and localised* in z.

## Approach

The ink model's 2D probability already encodes "where on the surface does ink
appear, as the model sees it." We use that as the spatial gate. Then we
intersect it with the **CT intensity at the corresponding (z, y, x) voxel** so
only voxels that are *actually radiopaque* (i.e., contain ink material) get
labelled.

::

    for each voxel (z, y, x) in bbox:
        ink_2d = ink_pred[y, x]          # 2D surface probability (model)
        ct_z   = ct[z, y, x]              # CT intensity in 3D
        label  = (ink_2d  >= ink_threshold) AND
                 (ct_z    >= ct_percentile_threshold)

The CT threshold is computed **per-column** by default (each (y, x) gets its
own quantile of the CT z-stack), so we adapt to local papyrus density rather
than imposing a global threshold. ``--global-ct-threshold`` switches to a
single scalar threshold computed from the whole bbox.

## Three refinements

- ``--surface-window N``: restrict labels to voxels within +/- N z-voxels of
  the per-column CT-intensity peak. Real ink lives in a thin shell on the
  papyrus surface, not in the bulk. This is the dominant refinement.
- ``--close R``: morphological closing radius. Fills 1-voxel holes in
  letterforms.
- ``--min-component-voxels K``: drop 26-connected components below K voxels
  as noise.

## Status

First cut. Not validated against ground-truth ink labels; threshold choices
are uncalibrated; CT intensity is a coarse proxy for actual ink material.
The script is intended to start the conversation on #192, not close it.
Several obvious next-cuts are documented in the PR description.

Usage::

    python generate_3d_ink_labels.py \\
        --ct scroll.zarr \\
        --ink-pred ink_pred.zarr \\
        --output ink_labels.zarr \\
        --bbox z0 z1 y0 y1 x0 x1 \\
        --ink-threshold 0.5 \\
        --ct-percentile 85 \\
        [--surface-window 4] \\
        [--close 1] \\
        [--min-component-voxels 16] \\
        [--debug-png out.png]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def _load_zarr_window(zarr_path: str, z0: int, z1: int, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    """Read a (z1-z0, y1-y0, x1-x0) window from a zarr; uint8 inputs scaled to float32 [0, 1]."""
    import zarr

    store = zarr.open(zarr_path, mode="r")
    arr = store if isinstance(store, zarr.core.Array) else store["0"]
    z1 = min(z1, arr.shape[0])
    y1 = min(y1, arr.shape[1])
    x1 = min(x1, arr.shape[2])
    if z0 >= z1 or y0 >= y1 or x0 >= x1:
        raise ValueError(f"empty bbox vs shape {arr.shape}")
    raw = arr[z0:z1, y0:y1, x0:x1]
    if raw.dtype == np.uint8:
        return raw.astype(np.float32) / 255.0
    return raw.astype(np.float32)


def _load_2d_surface_prediction(zarr_path: str, expected_hw: tuple[int, int]) -> np.ndarray:
    """Read the 2D ink-surface probability map as a (H, W) float32 in [0, 1].

    Prediction zarrs may sit under a ``/0`` subgroup with shape ``(1, H, W)``.
    We try the subgroup first and fall back to the root. Coordinates are LOCAL
    to the prediction window; ``expected_hw`` (from the bbox) is checked for
    a shape match.
    """
    import zarr

    try:
        arr = zarr.open(f"{zarr_path}/0", mode="r")
    except Exception:
        store = zarr.open(zarr_path, mode="r")
        arr = store if isinstance(store, zarr.core.Array) else store["0"]
    if arr.ndim == 3:
        raw = np.asarray(arr[0])
    elif arr.ndim == 2:
        raw = np.asarray(arr)
    else:
        raise ValueError(f"unexpected ink prediction zarr ndim {arr.ndim}")
    if raw.shape != expected_hw:
        raise ValueError(
            f"ink prediction shape {raw.shape} != expected (h, w) from bbox {expected_hw}"
        )
    if raw.dtype == np.uint8:
        return raw.astype(np.float32) / 255.0
    return raw.astype(np.float32)


def _ensure_output_zarr(output_path: str, shape: tuple[int, int, int], chunks=(128, 128, 128)):
    import zarr

    if Path(output_path).exists():
        return zarr.open(output_path, mode="a")
    return zarr.open(
        output_path,
        mode="w",
        shape=shape,
        chunks=chunks,
        dtype="uint8",
        fill_value=0,
    )


def _infer_full_shape(ct_path: str) -> tuple[int, int, int]:
    import zarr

    store = zarr.open(ct_path, mode="r")
    arr = store if isinstance(store, zarr.core.Array) else store["0"]
    return tuple(int(s) for s in arr.shape[:3])


def _apply_surface_manifold_restriction(
    label_volume: np.ndarray, ct: np.ndarray, surface_window: int
) -> tuple[np.ndarray, int]:
    """Keep labels only within +/- surface_window voxels of the per-column CT peak."""
    if surface_window <= 0:
        return label_volume, 0
    surface_z = np.argmax(ct, axis=0).astype(np.int32)  # (H, W)
    z_indices = np.arange(ct.shape[0], dtype=np.int32)[:, None, None]  # (D, 1, 1)
    within_window = np.abs(z_indices - surface_z[None, :, :]) <= surface_window
    before = int(label_volume.sum())
    restricted = (label_volume.astype(bool) & within_window).astype(np.uint8)
    return restricted, before - int(restricted.sum())


def _filter_small_components(
    label_volume: np.ndarray, min_voxels: int
) -> tuple[np.ndarray, int, int]:
    """Drop 26-connected components below ``min_voxels``. Returns (filtered, total, kept)."""
    if min_voxels <= 1:
        return label_volume, 0, 0
    from scipy.ndimage import label as cc_label

    structure = np.ones((3, 3, 3), dtype=bool)
    labeled, num_components = cc_label(label_volume, structure=structure)
    if num_components == 0:
        return label_volume, 0, 0
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    keep_mask = counts >= min_voxels
    keep_mask[0] = False
    remap = np.zeros(counts.shape[0], dtype=np.uint8)
    remap[keep_mask] = 1
    return remap[labeled].astype(np.uint8), int(num_components), int(keep_mask.sum())


def _write_debug_png(
    png_path: str,
    ct: np.ndarray,
    ink_2d: np.ndarray,
    label_volume: np.ndarray,
    params_label: str,
    num_slices: int = 4,
) -> None:
    """Contact-sheet PNG: top row CT slices, bottom row CT + red label overlay + cyan ink contour."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    depth = ct.shape[0]
    z_indices = np.linspace(0, depth - 1, num=num_slices, dtype=int)
    fig, axes = plt.subplots(2, num_slices, figsize=(2.4 * num_slices, 5.4), squeeze=False)
    fig.suptitle(params_label, fontsize=9)
    ct_lo, ct_hi = np.percentile(ct, [1, 99])
    label_cmap = ListedColormap([(0, 0, 0, 0), (1, 0.1, 0.1, 0.55)])
    for col, z in enumerate(z_indices):
        ax_top = axes[0, col]
        ax_top.imshow(ct[z], cmap="gray", vmin=ct_lo, vmax=ct_hi, interpolation="nearest")
        ax_top.set_title(f"CT z={z}", fontsize=8)
        ax_top.axis("off")
        ax_bot = axes[1, col]
        ax_bot.imshow(ct[z], cmap="gray", vmin=ct_lo, vmax=ct_hi, interpolation="nearest")
        ax_bot.imshow(label_volume[z], cmap=label_cmap, vmin=0, vmax=1, interpolation="nearest")
        ax_bot.contour(ink_2d, levels=[0.1], colors="cyan", linewidths=0.6, alpha=0.7)
        ax_bot.set_title(f"label @z={z} ({int(label_volume[z].sum())}px)", fontsize=8)
        ax_bot.axis("off")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def generate_3d_ink_labels(
    ct_path: str,
    ink_pred_path: str,
    output_path: str,
    z0: int,
    z1: int,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    ink_threshold: float = 0.5,
    ct_percentile: float = 85.0,
    global_ct_threshold: bool = False,
    morphological_close: int = 0,
    surface_window: int = 0,
    min_component_voxels: int = 0,
    debug_png: str | None = None,
) -> dict:
    """CT-gated 3D ink labelling on a bbox. See module docstring for the algorithm.

    Returns a metadata dict (bbox, threshold settings, voxel counts, timing).
    """
    start = time.perf_counter()
    ct = _load_zarr_window(ct_path, z0, z1, y0, y1, x0, x1)
    ink_2d = _load_2d_surface_prediction(ink_pred_path, expected_hw=(y1 - y0, x1 - x0))

    if ct.shape[1:] != ink_2d.shape:
        raise ValueError(f"CT yx shape {ct.shape[1:]} != ink prediction shape {ink_2d.shape}")

    ink_mask_2d = ink_2d >= ink_threshold

    if global_ct_threshold:
        threshold_scalar = float(
            np.percentile(ct[:, ink_mask_2d] if ink_mask_2d.any() else ct, ct_percentile)
        )
        high_intensity = ct >= threshold_scalar
    else:
        per_col_thresh = np.full(ink_2d.shape, np.inf, dtype=np.float32)
        if ink_mask_2d.any():
            cols_ct = ct[:, ink_mask_2d]
            quantile_vals = np.percentile(cols_ct, ct_percentile, axis=0)
            per_col_thresh[ink_mask_2d] = quantile_vals
        high_intensity = ct >= per_col_thresh[None, :, :]

    ink_mask_3d_surface = np.broadcast_to(ink_mask_2d[None, :, :], ct.shape)
    label_volume = (ink_mask_3d_surface & high_intensity).astype(np.uint8)
    pre_refinement = int(label_volume.sum())

    label_volume, dropped_off_surface = _apply_surface_manifold_restriction(
        label_volume, ct, surface_window
    )

    if morphological_close > 0:
        from scipy.ndimage import binary_closing

        structure = np.ones((morphological_close,) * 3, dtype=bool)
        label_volume = binary_closing(label_volume, structure=structure).astype(np.uint8)

    label_volume, cc_total, cc_kept = _filter_small_components(label_volume, min_component_voxels)

    out_arr = _ensure_output_zarr(output_path, _infer_full_shape(ct_path))
    out_arr[z0:z1, y0:y1, x0:x1] = label_volume

    if debug_png is not None:
        params_label = (
            f"ink>={ink_threshold} ct%={ct_percentile} "
            f"surf_win={surface_window} cc>={min_component_voxels} close={morphological_close}"
        )
        _write_debug_png(debug_png, ct, ink_2d, label_volume, params_label)

    elapsed = time.perf_counter() - start
    return {
        "bbox": (z0, z1, y0, y1, x0, x1),
        "ink_threshold": ink_threshold,
        "ct_percentile": ct_percentile,
        "global_ct_threshold": global_ct_threshold,
        "morphological_close": morphological_close,
        "surface_window": surface_window,
        "min_component_voxels": min_component_voxels,
        "voxels": int(label_volume.size),
        "labeled_voxels": int(label_volume.sum()),
        "labeled_voxels_pre_refinement": pre_refinement,
        "label_fraction": float(label_volume.mean()),
        "voxels_dropped_off_surface": dropped_off_surface,
        "connected_components_total": cc_total,
        "connected_components_kept": cc_kept,
        "active_surface_columns": int(ink_mask_2d.sum()),
        "surface_active_fraction": float(ink_mask_2d.mean()),
        "elapsed_s": elapsed,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ct", required=True, help="Path to the input CT zarr.")
    parser.add_argument("--ink-pred", required=True, help="Path to the 2D ink-probability zarr.")
    parser.add_argument("--output", required=True, help="Path to the output 3D ink-label zarr (uint8).")
    parser.add_argument(
        "--bbox",
        type=int,
        nargs=6,
        required=True,
        metavar=("Z0", "Z1", "Y0", "Y1", "X0", "X1"),
    )
    parser.add_argument("--ink-threshold", type=float, default=0.5)
    parser.add_argument("--ct-percentile", type=float, default=85.0)
    parser.add_argument("--global-ct-threshold", action="store_true")
    parser.add_argument("--close", type=int, default=0)
    parser.add_argument("--surface-window", type=int, default=0)
    parser.add_argument("--min-component-voxels", type=int, default=0)
    parser.add_argument("--debug-png", type=str, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    z0, z1, y0, y1, x0, x1 = args.bbox
    result = generate_3d_ink_labels(
        args.ct,
        args.ink_pred,
        args.output,
        z0,
        z1,
        y0,
        y1,
        x0,
        x1,
        ink_threshold=args.ink_threshold,
        ct_percentile=args.ct_percentile,
        global_ct_threshold=args.global_ct_threshold,
        morphological_close=args.close,
        surface_window=args.surface_window,
        min_component_voxels=args.min_component_voxels,
        debug_png=args.debug_png,
    )
    print(
        f"# OK voxels={result['voxels']} labeled={result['labeled_voxels']} "
        f"(pre_refinement={result['labeled_voxels_pre_refinement']}) "
        f"label_fraction={result['label_fraction']:.6f} "
        f"off_surface_dropped={result['voxels_dropped_off_surface']} "
        f"cc_total={result['connected_components_total']} cc_kept={result['connected_components_kept']} "
        f"elapsed={result['elapsed_s']:.2f}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
