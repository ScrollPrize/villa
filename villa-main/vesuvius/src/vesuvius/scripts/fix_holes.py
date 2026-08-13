#!/usr/bin/env python3
"""
Fix holes in label volumes using a hole_mask produced by detect_holes.py.

Default method: 2D hole fill per Z slice, applied only where hole_mask is true.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import tifffile as tiff
from scipy import ndimage as ndi
from tqdm import tqdm

try:
    import cc3d
    _HAS_CC3D = True
except Exception:
    cc3d = None
    _HAS_CC3D = False
try:
    import alphashape
    from shapely.geometry import Polygon, MultiPolygon
    _HAS_ALPHASHAPE = True
except Exception:
    alphashape = None
    Polygon = None
    MultiPolygon = None
    _HAS_ALPHASHAPE = False
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    cv2 = None
    _HAS_CV2 = False

TIFF_EXTS = {".tif", ".tiff"}


def _find_label_file(folder: Path, pattern: str) -> Optional[Path]:
    matches = sorted(folder.glob(pattern))
    if not matches:
        return None
    if len(matches) > 1:
        print(f"[WARN] Multiple label matches in {folder}: {matches}. Using {matches[0].name}")
    return matches[0]


def _fix_2d_hole_fill(
    label: np.ndarray,
    hole_mask: np.ndarray,
    *,
    fg_value: int,
    ignore_value: Optional[int],
    mask_dilate: int,
) -> np.ndarray:
    if label.ndim != 3:
        raise ValueError(f"Expected 3D label, got shape {label.shape}")
    if hole_mask.shape != label.shape:
        raise ValueError(f"hole_mask shape {hole_mask.shape} does not match label {label.shape}")

    ignore_mask = (label == ignore_value) if ignore_value is not None else np.zeros(label.shape, dtype=bool)
    label_mask = (label == fg_value) & (~ignore_mask)
    hole_mask = hole_mask.astype(bool)
    if mask_dilate and mask_dilate > 0:
        struct = ndi.generate_binary_structure(label.ndim, 1)
        hole_mask = ndi.binary_dilation(hole_mask, structure=struct, iterations=int(mask_dilate))

    filled = np.zeros_like(label_mask, dtype=bool)
    for z in range(label_mask.shape[0]):
        filled[z] = ndi.binary_fill_holes(label_mask[z])

    to_add = filled & hole_mask & (~ignore_mask)

    out = label.copy()
    out[to_add] = fg_value
    # Restore ignore label explicitly to avoid any accidental overwrites.
    if ignore_value is not None:
        out[ignore_mask] = ignore_value
    return out


def _fix_morph_close(
    label: np.ndarray,
    hole_mask: np.ndarray,
    *,
    fg_value: int,
    ignore_value: Optional[int],
    kernel: int,
    mask_dilate: int,
) -> np.ndarray:
    if label.ndim != 3:
        raise ValueError(f"Expected 3D label, got shape {label.shape}")
    if hole_mask.shape != label.shape:
        raise ValueError(f"hole_mask shape {hole_mask.shape} does not match label {label.shape}")
    if kernel < 1:
        raise ValueError("--closing-kernel must be >= 1")

    if not _HAS_CC3D:
        raise ImportError("cc3d is required for component-wise closing. Install with: pip install cc3d")

    ignore_mask = (label == ignore_value) if ignore_value is not None else np.zeros(label.shape, dtype=bool)
    label_mask = (label == fg_value) & (~ignore_mask)
    struct = ndi.generate_binary_structure(label.ndim, 1)
    hole_mask = hole_mask.astype(bool)
    if mask_dilate and mask_dilate > 0:
        hole_mask = ndi.binary_dilation(hole_mask, structure=struct, iterations=int(mask_dilate))
    if kernel > 1:
        struct = ndi.iterate_structure(struct, kernel)
    struct_conn = ndi.generate_binary_structure(label.ndim, label.ndim)  # 26-connectivity in 3D

    labels_cc, n_cc = cc3d.connected_components(label_mask.astype(np.uint8), connectivity=26, return_N=True)
    if n_cc == 0:
        return label.copy()
    slices = ndi.find_objects(labels_cc)

    out = label.copy()

    for comp_id, slc in enumerate(slices, start=1):
        if slc is None:
            continue
        comp = (labels_cc[slc] == comp_id)
        if not comp.any():
            continue

        region = hole_mask[slc] & (~ignore_mask[slc])
        if not region.any():
            continue

        closed = ndi.binary_closing(comp, structure=struct)
        # Only add voxels (do not remove original component)
        proposed = comp | (closed & region)
        added = proposed & (~comp)

        if not added.any():
            continue

        # Skip if this would merge with other components (26-neighborhood adjacency).
        others = label_mask[slc] & (~comp)
        if others.any():
            if np.any(added & ndi.binary_dilation(others, structure=struct_conn)):
                print(f"[WARN] Skipping close on component {comp_id}: would merge with neighbor")
                continue

        # Skip if closing would change component connectivity (should remain 1).
        _, n_after = cc3d.connected_components(proposed.astype(np.uint8), connectivity=26, return_N=True)
        if n_after != 1:
            print(f"[WARN] Skipping close on component {comp_id}: connectivity {n_after}")
            continue

        out_slice = out[slc]
        out_slice[added] = fg_value
        out[slc] = out_slice

    if ignore_value is not None:
        out[ignore_mask] = ignore_value
    return out


def _alpha_shape_fill_mask(
    points: np.ndarray,
    shape: Tuple[int, int],
    *,
    alpha: float,
) -> Optional[np.ndarray]:
    if not _HAS_ALPHASHAPE:
        raise ImportError("alphashape is required for alpha-wrap. Install with: pip install alphashape shapely")
    if not _HAS_CV2:
        raise ImportError("opencv-python is required for alpha-wrap rasterization. Install with: pip install opencv-python")
    if points.shape[0] < 4:
        return None
    # Avoid Qhull errors on degenerate (colinear) point sets.
    if np.linalg.matrix_rank(points - points[0]) < 2:
        return None
    try:
        alpha_shape = alphashape.alphashape(points, alpha)
    except Exception as exc:
        print(f"[WARN] Alpha shape computation failed: {exc}")
        return None
    if alpha_shape is None or alpha_shape.is_empty:
        return None

    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)

    def _clip_coords(coords: np.ndarray) -> np.ndarray:
        coords[:, 0] = np.clip(coords[:, 0], 0, w - 1)
        coords[:, 1] = np.clip(coords[:, 1], 0, h - 1)
        return coords.astype(np.int32)

    def _fill_polygon(geom: Polygon) -> None:
        exterior_coords = _clip_coords(np.array(geom.exterior.coords))
        cv2.fillPoly(mask, [exterior_coords], 1)
        for interior in geom.interiors:
            interior_coords = _clip_coords(np.array(interior.coords))
            cv2.fillPoly(mask, [interior_coords], 0)

    if isinstance(alpha_shape, Polygon):
        _fill_polygon(alpha_shape)
    elif isinstance(alpha_shape, MultiPolygon):
        for poly in alpha_shape.geoms:
            _fill_polygon(poly)
    else:
        print(f"[WARN] Unexpected alpha shape geometry: {type(alpha_shape)}")
        return None

    return mask.astype(bool)


def _fix_alpha_wrap(
    label: np.ndarray,
    hole_mask: np.ndarray,
    *,
    fg_value: int,
    ignore_value: Optional[int],
    alpha: float,
    mask_dilate: int,
) -> np.ndarray:
    if label.ndim != 3:
        raise ValueError(f"Expected 3D label, got shape {label.shape}")
    if hole_mask.shape != label.shape:
        raise ValueError(f"hole_mask shape {hole_mask.shape} does not match label {label.shape}")
    if alpha <= 0:
        raise ValueError("--alpha must be > 0")
    if not _HAS_CC3D:
        raise ImportError("cc3d is required for component-wise alpha wrap. Install with: pip install cc3d")

    ignore_mask = (label == ignore_value) if ignore_value is not None else np.zeros(label.shape, dtype=bool)
    label_mask = (label == fg_value) & (~ignore_mask)
    hole_mask = hole_mask.astype(bool)
    if mask_dilate and mask_dilate > 0:
        struct = ndi.generate_binary_structure(label.ndim, 1)
        hole_mask = ndi.binary_dilation(hole_mask, structure=struct, iterations=int(mask_dilate))
    struct_conn = ndi.generate_binary_structure(label.ndim, label.ndim)  # 26-connectivity in 3D

    labels_cc, n_cc = cc3d.connected_components(label_mask.astype(np.uint8), connectivity=26, return_N=True)
    if n_cc == 0:
        return label.copy()
    slices = ndi.find_objects(labels_cc)

    out = label.copy()

    for comp_id, slc in enumerate(slices, start=1):
        if slc is None:
            continue
        comp = (labels_cc[slc] == comp_id)
        if not comp.any():
            continue

        region = hole_mask[slc] & (~ignore_mask[slc])
        if not np.any(comp & region):
            continue

        added = np.zeros_like(comp, dtype=bool)
        for z in range(comp.shape[0]):
            comp_slice = comp[z]
            if not comp_slice.any():
                continue
            region_slice = region[z]
            points_mask = comp_slice & region_slice
            if points_mask.sum() < 4:
                continue
            y_coords, x_coords = np.where(points_mask)
            points = np.column_stack([x_coords, y_coords])
            alpha_mask = _alpha_shape_fill_mask(points, comp_slice.shape, alpha=alpha)
            if alpha_mask is None:
                continue
            add_slice = alpha_mask & region_slice & (~comp_slice)
            if add_slice.any():
                added[z] = add_slice

        if not added.any():
            continue

        proposed = comp | added

        # Skip if this would merge with other components (26-neighborhood adjacency).
        others = label_mask[slc] & (~comp)
        if others.any():
            if np.any(added & ndi.binary_dilation(others, structure=struct_conn)):
                print(f"[WARN] Skipping alpha wrap on component {comp_id}: would merge with neighbor")
                continue

        # Skip if alpha wrap would change component connectivity (should remain 1).
        _, n_after = cc3d.connected_components(proposed.astype(np.uint8), connectivity=26, return_N=True)
        if n_after != 1:
            print(f"[WARN] Skipping alpha wrap on component {comp_id}: connectivity {n_after}")
            continue

        out_slice = out[slc]
        out_slice[added] = fg_value
        out[slc] = out_slice

    if ignore_value is not None:
        out[ignore_mask] = ignore_value
    return out


def _count_components_26(
    label: np.ndarray,
    *,
    fg_value: int,
    ignore_value: Optional[int],
) -> int:
    if not _HAS_CC3D:
        raise ImportError("cc3d is required for component-count checks. Install with: pip install cc3d")
    label_cc = label.copy()
    if ignore_value is not None:
        label_cc[label_cc == ignore_value] = 0
    mask = (label_cc == fg_value).astype(np.uint8)
    _labels, n = cc3d.connected_components(mask, connectivity=26, return_N=True)
    return int(n)


def _list_subfolders(root: Path) -> List[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir()])


def _process_folder(
    folder: Path,
    *,
    hole_mask_name: str,
    label_glob: str,
    fg_value: int,
    ignore_value: Optional[int],
    suffix: str,
    methods: List[str],
    closing_kernel: int,
    alpha: float,
    iters: int,
    mask_dilate: int,
) -> Tuple[bool, str]:
    hole_path = folder / hole_mask_name
    if not hole_path.exists():
        return False, f"[WARN] Missing hole mask in {folder}"

    label_path = _find_label_file(folder, label_glob)
    if label_path is None or label_path.suffix.lower() not in TIFF_EXTS:
        return False, f"[WARN] Missing label in {folder} (glob={label_glob})"

    label = tiff.imread(str(label_path))
    hole_mask = tiff.imread(str(hole_path))

    if label.shape != hole_mask.shape:
        return False, f"[WARN] Shape mismatch in {folder}: label {label.shape} vs hole_mask {hole_mask.shape}"

    before_n = _count_components_26(
        label,
        fg_value=int(fg_value),
        ignore_value=None if ignore_value is None else int(ignore_value),
    )

    fixed = label
    for _ in range(max(1, int(iters))):
        for method in methods:
            if method == "2d_hole_fill":
                fixed = _fix_2d_hole_fill(
                    fixed,
                    hole_mask,
                    fg_value=int(fg_value),
                    ignore_value=None if ignore_value is None else int(ignore_value),
                    mask_dilate=int(mask_dilate),
                )
            elif method == "close":
                fixed = _fix_morph_close(
                    fixed,
                    hole_mask,
                    fg_value=int(fg_value),
                    ignore_value=None if ignore_value is None else int(ignore_value),
                    kernel=int(closing_kernel),
                    mask_dilate=int(mask_dilate),
                )
            elif method == "alpha_wrap":
                fixed = _fix_alpha_wrap(
                    fixed,
                    hole_mask,
                    fg_value=int(fg_value),
                    ignore_value=None if ignore_value is None else int(ignore_value),
                    alpha=float(alpha),
                    mask_dilate=int(mask_dilate),
                )
            else:
                return False, f"[WARN] Unknown method '{method}' in {folder}"

    after_n = _count_components_26(
        fixed,
        fg_value=int(fg_value),
        ignore_value=None if ignore_value is None else int(ignore_value),
    )

    msg = ""
    if before_n != after_n:
        msg = f"[WARN] Component count changed in {folder.name}: {before_n} -> {after_n}"

    out_name = f"{label_path.stem}_{suffix}{label_path.suffix}"
    out_path = folder / out_name
    tiff.imwrite(str(out_path), fixed.astype(label.dtype))
    ok_msg = f"[OK] Wrote {out_path}"
    if msg:
        ok_msg = msg + " | " + ok_msg
    return True, ok_msg


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Fix holes in label volumes using hole_mask.tif in each subfolder."
    )
    ap.add_argument("--in-dir", type=Path, required=True,
                    help="Root folder containing per-sample subfolders.")
    ap.add_argument("--hole-mask-name", type=str, default="hole_mask.tif",
                    help="Filename of the hole mask inside each subfolder.")
    ap.add_argument("--label-glob", type=str, default="*_surf.tif",
                    help="Glob for label file inside each subfolder (default: *_surf.tif).")
    ap.add_argument("--fg-value", type=int, default=1,
                    help="Foreground label value to fill (default: 1).")
    ap.add_argument("--ignore-value", type=int, default=None,
                    help="Label value to ignore (never filled).")
    ap.add_argument("--method", type=str, action="append", default=None,
                    help=("Fix method(s). Repeatable and order-preserving. "
                          "Choices: 2d_hole_fill, close, alpha_wrap. "
                          "You can also pass a comma-separated list."))
    ap.add_argument("--closing-kernel", type=int, default=2,
                    help="Kernel radius for morphological closing (method=close).")
    ap.add_argument("--alpha", type=float, default=0.005,
                    help="Alpha parameter for alpha shape (method=alpha_wrap).")
    ap.add_argument("--mask-dilate", type=int, default=0,
                    help="Dilate hole_mask by this many iterations before applying fixes.")
    ap.add_argument("--iters", type=int, default=1,
                    help="Number of iterations of the selected method (default: 1).")
    ap.add_argument("--suffix", type=str, default=None,
                    help="Suffix for output file (default depends on method).")
    ap.add_argument("--workers", type=int, default=0,
                    help="Number of parallel workers (0=sequential).")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    root = args.in_dir
    if not root.exists():
        raise FileNotFoundError(root)
    if not _HAS_CC3D:
        raise ImportError("cc3d is required for component-count checks. Install with: pip install cc3d")

    subfolders = _list_subfolders(root)
    if not subfolders:
        print(f"No subfolders found under {root}")
        return

    allowed_methods = {"2d_hole_fill", "close", "alpha_wrap"}
    raw_methods = args.method if args.method is not None else ["2d_hole_fill"]
    methods: List[str] = []
    for entry in raw_methods:
        parts = [p.strip() for p in str(entry).split(",") if p.strip()]
        methods.extend(parts)
    if not methods:
        raise ValueError("No methods provided. Use --method 2d_hole_fill (or repeat --method).")
    invalid = sorted({m for m in methods if m not in allowed_methods})
    if invalid:
        raise ValueError(f"Unknown method(s): {', '.join(invalid)}")

    suffix = args.suffix
    if suffix is None:
        tags: List[str] = []
        for method in methods:
            if method == "close":
                tags.append(f"closed_k{int(args.closing_kernel)}")
            elif method == "alpha_wrap":
                alpha_tag = str(args.alpha).replace(".", "p")
                tags.append(f"alpha_wrap_a{alpha_tag}")
            else:
                tags.append("2d_hole_fill")
        suffix = "_".join(tags)
        if int(args.mask_dilate) > 0:
            suffix = f"{suffix}_md{int(args.mask_dilate)}"
        if int(args.iters) > 1:
            suffix = f"{suffix}_x{int(args.iters)}"

    if args.workers and args.workers > 0:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [
                ex.submit(
                    _process_folder,
                    folder,
                    hole_mask_name=str(args.hole_mask_name),
                    label_glob=str(args.label_glob),
                    fg_value=int(args.fg_value),
                    ignore_value=None if args.ignore_value is None else int(args.ignore_value),
                    suffix=str(suffix),
                    methods=list(methods),
                    closing_kernel=int(args.closing_kernel),
                    alpha=float(args.alpha),
                    iters=int(args.iters),
                    mask_dilate=int(args.mask_dilate),
                )
                for folder in subfolders
            ]
            for fut in tqdm(futures, total=len(futures), desc="Fixing"):
                ok, msg = fut.result()
                if msg:
                    print(msg)
    else:
        for folder in tqdm(subfolders, desc="Fixing"):
            ok, msg = _process_folder(
                folder,
                hole_mask_name=str(args.hole_mask_name),
                label_glob=str(args.label_glob),
                fg_value=int(args.fg_value),
                ignore_value=None if args.ignore_value is None else int(args.ignore_value),
                suffix=str(suffix),
                methods=list(methods),
                closing_kernel=int(args.closing_kernel),
                alpha=float(args.alpha),
                iters=int(args.iters),
                mask_dilate=int(args.mask_dilate),
            )
            if msg:
                print(msg)


if __name__ == "__main__":
    main()
