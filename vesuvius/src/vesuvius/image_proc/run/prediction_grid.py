"""
Build a grid TIF per image showing overlays of:
- Top-left: original image + GT overlay (with title)
- Remaining tiles: original image + each model's predictions (with model name)

Features:
- Preserves original image size for each tile; adds a top text pad.
- Up to --max-cols tiles per row (default: 4).
- Supports 2D images and 3D stacks (overlay per-slice; writes multi-page RGB TIF).
- Renders a stats table below the grid: rows are models, columns are metrics.
- Uses tqdm progress and multiprocessing for speed.

Usage
-----
python -m vesuvius.image_proc.prediction_grid \
  --eval-root /path/to/eval_offline_outputs \
  --images /path/to/images \
  --output /path/to/output \
  [--models modelA,modelB] [--max-cols 4]

Notes
-----
- Eval root must contain subfolders (one per model). Each model folder contains per-image subfolders with files:
  - label.tif, pred.tif, stats.json (as written by eval_offline in directory mode).
- Labels/predictions are treated as binary masks: nonzero is foreground; floats in [0,1] use --threshold.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import tifffile

# Optional text rendering backends
try:  # pragma: no cover - optional dependency
    from PIL import Image, ImageDraw, ImageFont  # type: ignore
    _HAS_PIL = True
except Exception:  # pragma: no cover
    Image = None  # type: ignore
    ImageDraw = None  # type: ignore
    ImageFont = None  # type: ignore
    _HAS_PIL = False

try:  # pragma: no cover - optional dependency
    import cv2  # type: ignore
    _HAS_CV2 = True
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore
    _HAS_CV2 = False

try:  # pragma: no cover - optional dependency
    from tqdm import tqdm as _tqdm  # type: ignore
    def tqdm(iterable, **kwargs):
        return _tqdm(iterable, **kwargs)
except Exception:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable

from concurrent.futures import ProcessPoolExecutor, as_completed


def _discover_tifs(folder: Path) -> List[Path]:
    return sorted(list(folder.glob("*.tif")) + list(folder.glob("*.tiff")))


def _parse_color(s: str) -> Tuple[int, int, int]:
    s = s.strip()
    if s.startswith("#"):
        s = s[1:]
    if len(s) == 6:
        return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
    named = {
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "blue": (0, 128, 255),
        "cyan": (0, 255, 255),
        "magenta": (255, 0, 255),
        "yellow": (255, 255, 0),
        "white": (255, 255, 255),
        "black": (0, 0, 0),
    }
    if s.lower() in named:
        return named[s.lower()]
    raise ValueError(f"Unrecognized color: {s}")


def _to_uint8_grayscale(arr: np.ndarray, rescale: bool = True) -> np.ndarray:
    """Convert array to uint8 grayscale for display.

    - If already uint8 and range within [0,255], returns as-is (copy).
    - If rescale=True: percentile-based scaling to [0,255].
    - Else: linear clip to [0,255] if possible, then cast.
    """
    a = np.asarray(arr)
    # Remove any singleton channel dim
    if a.ndim >= 3 and a.shape[-1] == 1:
        a = a[..., 0]
    if a.dtype == np.uint8:
        return a.copy()
    if rescale:
        # Compute global percentiles for stability
        lo, hi = np.percentile(a.astype(np.float32), [1.0, 99.0])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.min(a)), float(np.max(a))
            if hi <= lo:
                return np.zeros_like(a, dtype=np.uint8)
        a32 = np.clip((a.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
        return (a32 * 255.0 + 0.5).astype(np.uint8)
    # Best-effort linear clip
    minv, maxv = float(np.min(a)), float(np.max(a))
    if maxv <= minv:
        return np.zeros_like(a, dtype=np.uint8)
    a32 = np.clip((a.astype(np.float32) - minv) / (maxv - minv), 0.0, 1.0)
    return (a32 * 255.0 + 0.5).astype(np.uint8)


def _ensure_rgb(img_u8: np.ndarray) -> np.ndarray:
    """Ensure HxW or ZxHxW uint8 becomes RGB: HxWx3 or ZxHxWx3.

    Rules:
    - If last dim is 3, assume HxWx3 and return as-is.
    - If 2D, replicate to 3 channels.
    - If 3D and last dim != 3, assume ZxHxW and replicate along a new last channel dim.
    - If 4D and last dim != 3, assume ...x1 and replicate to ...x3.
    """
    if img_u8.ndim == 2:
        return np.stack([img_u8, img_u8, img_u8], axis=-1)
    if img_u8.ndim == 3:
        if img_u8.shape[-1] == 3:
            return img_u8
        # Treat as Z,H,W
        return np.repeat(img_u8[..., None], 3, axis=-1)
    if img_u8.ndim == 4:
        if img_u8.shape[-1] == 3:
            return img_u8
        # Assume last is singleton channel
        return np.repeat(img_u8, 3, axis=-1)
    raise ValueError(f"Unsupported image shape for RGB conversion: {img_u8.shape}")


def _binarize_mask(arr: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    a = np.asarray(arr)
    if a.dtype.kind in ("f",):
        # probabilities or logits; if outside [0,1], treat >0 as foreground
        if np.nanmax(a) <= 1.0 and np.nanmin(a) >= 0.0:
            return a >= float(threshold)
        return a > 0
    # integer/boolean
    return a != 0


def _overlay_color(rgb: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float) -> np.ndarray:
    """Overlay a solid color on rgb where mask is True, with blending alpha.

    Supports 2D (H,W,3) and 3D stacks (Z,H,W,3). Returns same shape/dtype uint8.
    """
    out = rgb.astype(np.float32).copy()
    if out.ndim == 3:
        # H,W,3 and mask H,W or H,W,1
        m = mask.astype(bool)
        if m.ndim == 3 and m.shape[-1] == 1:
            m = m[..., 0]
        if m.ndim != 2:
            raise ValueError("Mask must be 2D for 2D overlay")
        for c in range(3):
            out[..., c] = np.where(m, (1.0 - alpha) * out[..., c] + alpha * float(color[c]), out[..., c])
        return np.clip(out, 0, 255).astype(np.uint8)
    elif out.ndim == 4:
        # Z,H,W,3 and mask Z,H,W or Z,H,W,1
        m = mask.astype(bool)
        if m.ndim == 4 and m.shape[-1] == 1:
            m = m[..., 0]
        if m.ndim == 3 and m.shape[0] == out.shape[0]:
            pass
        elif m.ndim == 2:
            # Broadcast 2D mask along Z
            m = np.repeat(m[None, ...], out.shape[0], axis=0)
        else:
            raise ValueError("Unsupported mask shape for 3D overlay")
        for c in range(3):
            out[..., c] = np.where(m, (1.0 - alpha) * out[..., c] + alpha * float(color[c]), out[..., c])
        return np.clip(out, 0, 255).astype(np.uint8)
    else:
        raise ValueError(f"Unsupported RGB shape {out.shape}")


def _draw_text_pad(width: int, pad_px: int, text: str, font_color: Tuple[int, int, int], margin_x: int = 6) -> np.ndarray:
    """Create an RGB pad of height pad_px and given width with left-aligned text.

    Falls back to plain pad if no text backend is available.
    """
    pad = np.zeros((pad_px, width, 3), dtype=np.uint8)
    if not text:
        return pad
    # Prefer PIL for better text rendering
    if _HAS_PIL:
        img = Image.fromarray(pad)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        # Text with stroke for legibility
        try:
            draw.text((margin_x, 2), text, fill=tuple(font_color), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
        except TypeError:
            draw.text((margin_x, 2), text, fill=tuple(font_color), font=font)
        return np.asarray(img, dtype=np.uint8)
    if _HAS_CV2:
        # OpenCV uses BGR; convert font_color
        bgr = (int(font_color[2]), int(font_color[1]), int(font_color[0]))
        cv2.putText(pad, text, (max(1, margin_x), max(14, pad_px - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 1, cv2.LINE_AA)
        return pad
    # Fallback: colored bar without text
    pad[...] = (32, 32, 32)
    return pad


def _ensure_4d_rgb(img: np.ndarray) -> np.ndarray:
    """Ensure image is either HxWx3 (2D) or ZxHxWx3 (3D)."""
    if img.ndim == 3 and img.shape[-1] == 3:
        return img
    if img.ndim == 2:
        return _ensure_rgb(img)
    if img.ndim == 3:
        # assume Z,H,W
        return _ensure_rgb(img)
    if img.ndim == 4 and img.shape[-1] == 3:
        return img
    raise ValueError(f"Unsupported image shape: {img.shape}")


def _pad_and_title_tile(tile_rgb: np.ndarray, title: str, pad_px: int, title_color: Tuple[int, int, int]) -> np.ndarray:
    # tile_rgb: HxWx3 or ZxHxWx3
    if tile_rgb.ndim == 3:
        h, w, _ = tile_rgb.shape
        pad = _draw_text_pad(w, pad_px, title, title_color)
        return np.vstack([pad, tile_rgb])
    elif tile_rgb.ndim == 4:
        z, h, w, _ = tile_rgb.shape
        pad = _draw_text_pad(w, pad_px, title, title_color)
        # Repeat pad across Z slices
        pad_stack = np.repeat(pad[None, ...], z, axis=0)
        return np.concatenate([pad_stack, tile_rgb], axis=1)
    else:
        raise ValueError("tile_rgb must be HxWx3 or ZxHxWx3")


def _assemble_grid(tiles: List[np.ndarray], max_cols: int, hpad: int = 4, vpad: int = 0) -> np.ndarray:
    """Assemble list of tiles into a grid.

    All tiles must share shape HxWx3 or ZxHxWx3 with the same H,W (and Z if 3D).
    Returns same type of array.
    """
    if not tiles:
        raise ValueError("No tiles to assemble")
    first = tiles[0]
    is_3d = (first.ndim == 4)
    if is_3d:
        z, H, W, C = first.shape
    else:
        H, W, C = first.shape
    # Validate and infer rows/cols
    for t in tiles:
        if t.shape[-1] != C:
            raise ValueError("Mismatched channels among tiles")
        if is_3d and t.ndim != 4:
            raise ValueError("Mix of 2D and 3D tiles")
        if not is_3d and t.ndim != 3:
            raise ValueError("Mix of 2D and 3D tiles")
        if is_3d and (t.shape[0] != z or t.shape[1] != H or t.shape[2] != W):
            raise ValueError("Tile shapes must match (Z,H,W)")
        if not is_3d and (t.shape[0] != H or t.shape[1] != W):
            raise ValueError("Tile shapes must match (H,W)")

    n = len(tiles)
    cols = min(max_cols, n)
    rows = (n + cols - 1) // cols

    if is_3d:
        total_h = rows * H + (rows - 1) * vpad
        total_w = cols * W + (cols - 1) * hpad
        out = np.zeros((z, total_h, total_w, C), dtype=tiles[0].dtype)
        for i, t in enumerate(tiles):
            r = i // cols
            c = i % cols
            y0 = r * (H + vpad)
            x0 = c * (W + hpad)
            out[:, y0:y0 + H, x0:x0 + W, :] = t
        return out
    else:
        total_h = rows * H + (rows - 1) * vpad
        total_w = cols * W + (cols - 1) * hpad
        out = np.zeros((total_h, total_w, C), dtype=tiles[0].dtype)
        for i, t in enumerate(tiles):
            r = i // cols
            c = i % cols
            y0 = r * (H + vpad)
            x0 = c * (W + hpad)
            out[y0:y0 + H, x0:x0 + W, :] = t
        return out


@dataclass
class Task:
    image_path: Path
    # One label read is sufficient; if None we'll try to find from first model at runtime
    label_path: Optional[Path]
    # For each model, the per-image directory (expected to contain pred.tif and stats.json)
    model_items: List[Tuple[str, Path]]
    out_path: Path


def _find_label_for_image(labels_dir: Path, stem: str, label_suffix: Optional[str]) -> Optional[Path]:
    candidates = []
    # Exact stem match first
    for ext in (".tif", ".tiff"):
        p = labels_dir / f"{stem}{ext}"
        if p.exists():
            return p
    if label_suffix:
        for ext in (".tif", ".tiff"):
            p = labels_dir / f"{stem}{label_suffix}{ext}"
            if p.exists():
                return p
    # Fallback: any that startswith stem (unique)
    cand = sorted(list(labels_dir.glob(f"{stem}*.tif")) + list(labels_dir.glob(f"{stem}*.tiff")))
    if len(cand) == 1:
        return cand[0]
    return None


def _find_pred_for_image(model_dir: Path, stem: str, pred_suffix: Optional[str]) -> Optional[Path]:
    # Exact stem match
    for ext in (".tif", ".tiff"):
        p = model_dir / f"{stem}{ext}"
        if p.exists():
            return p
    if pred_suffix:
        for ext in (".tif", ".tiff"):
            p = model_dir / f"{stem}{pred_suffix}{ext}"
            if p.exists():
                return p
    cand = sorted(list(model_dir.glob(f"{stem}*.tif")) + list(model_dir.glob(f"{stem}*.tiff")))
    # Prefer exact prefix match with underscore
    if cand:
        return cand[0]
    return None


def _format_required_stats(stats: Dict[str, float]) -> List[Tuple[str, str]]:
    """Extract and format the required stat fields from stats.json content."""
    # Mapping: (label, key)
    wanted: List[Tuple[str, str]] = [
        ("critical total", "critical components total"),
        ("branch absdiff c1", "branch_points_absdiff_class_1"),
        ("cc diff c1", "connected_components_difference_class_1"),
        ("dice c1", "dice_class_1"),
        ("f1 c1", "f1_class_1"),
        ("hd95 c1", "hausdorff_distance_95_class_1"),
        ("iou c1", "iou_class_1"),
        ("map c1", "map_class_1"),
        ("recall c1", "recall_class_1"),
        ("precision c1", "precision_class_1"),
    ]
    out: List[Tuple[str, str]] = []
    for label, key in wanted:
        if key not in stats:
            out.append((label, "-"))
            continue
        v = stats[key]
        if v is None:
            out.append((label, "-"))
            continue
        # Formatting: if nearly integer, show as int; otherwise 4 decimals
        try:
            if isinstance(v, (int, np.integer)):
                out.append((label, str(int(v))))
            else:
                vf = float(v)
                if np.isfinite(vf) and abs(vf - round(vf)) < 1e-6 and abs(vf) > 1.0:
                    out.append((label, str(int(round(vf)))))
                else:
                    out.append((label, f"{vf:.4f}"))
        except Exception:
            out.append((label, str(v)))
    return out


def _draw_bottom_stats_table(width: int,
                             models_stats: List[Tuple[str, Dict[str, float]]],
                             line_h: int = 20,
                             header_color: Tuple[int, int, int] = (255, 255, 255),
                             value_color: Tuple[int, int, int] = (210, 210, 210),
                             stripe_color: Tuple[int, int, int] = (28, 28, 28),
                             side_pad: int = 12,
                             model_col_pad: int = 24) -> np.ndarray:
    """Render a tabular stats panel (models as rows, metrics as columns) to place below the grid.

    Returns an RGB uint8 array of shape [H_table, width, 3].
    """
    headers = [
        "model",
        "critical total",
        "branches c1",
        "cc diff c1",
        "dice c1",
        "f1 c1",
        "hd95 c1",
        "iou c1",
        "map c1",
        "recall c1",
        "precision c1",
    ]

    # Prepare rows: list of strings for each column per model
    rows: List[List[str]] = []
    for model_name, stats in models_stats:
        values = {k: v for k, v in _format_required_stats(stats)}
        row = [
            model_name,
            values.get("critical total", "-"),
            values.get("branch absdiff c1", "-"),
            values.get("cc diff c1", "-"),
            values.get("dice c1", "-"),
            values.get("f1 c1", "-"),
            values.get("hd95 c1", "-"),
            values.get("iou c1", "-"),
            values.get("map c1", "-"),
            values.get("recall c1", "-"),
            values.get("precision c1", "-"),
        ]
        rows.append(row)

    n_cols = len(headers)
    # Compute height: header + rows
    header_h = line_h + 6
    table_h = header_h + line_h * max(1, len(rows)) + 6
    img = np.zeros((table_h, width, 3), dtype=np.uint8)

    # Draw alternating stripes for legibility
    for i in range(len(rows)):
        if i % 2 == 0:
            y0 = header_h + i * line_h
            img[y0:y0 + line_h, :, :] = np.array(stripe_color, dtype=np.uint8)

    # Draw with PIL if available
    if _HAS_PIL:
        pil = Image.fromarray(img)
        draw = ImageDraw.Draw(pil)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        # Column positions spread across available width
        # Ensure some left/right margins to avoid edge overlap
        sp = max(0, min(side_pad, width // 10))
        avail_w = max(1, width - 2 * sp)
        col_x = [sp + int(round(j * (avail_w / n_cols))) for j in range(n_cols)]
        # Header row
        for j, h in enumerate(headers):
            # Add RIGHT padding after the model column: shift subsequent columns
            x = col_x[j] + (6 if j == 0 else model_col_pad + 6)
            try:
                draw.text((x, 4), h, fill=tuple(header_color), font=font)
            except TypeError:
                draw.text((x, 4), h, fill=tuple(header_color), font=font)
        # Data rows
        for i, row in enumerate(rows):
            y = header_h + i * line_h + 2
            for j, cell in enumerate(row):
                x = col_x[j] + (6 if j == 0 else model_col_pad + 6)
                draw.text((x, y), str(cell), fill=tuple(value_color), font=font)
        return np.asarray(pil, dtype=np.uint8)

    # Fallback to OpenCV if available
    if _HAS_CV2:
        sp = max(0, min(side_pad, width // 10))
        avail_w = max(1, width - 2 * sp)
        col_x = [sp + int(round(j * (avail_w / n_cols))) for j in range(n_cols)]
        # Header
        for j, h in enumerate(headers):
            x = col_x[j] + (6 if j == 0 else model_col_pad + 6)
            cv2.putText(img, h, (x, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        # Rows
        for i, row in enumerate(rows):
            y = header_h + i * line_h + 14
            for j, cell in enumerate(row):
                x = col_x[j] + (6 if j == 0 else model_col_pad + 6)
                cv2.putText(img, str(cell), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1, cv2.LINE_AA)
        return img

    return img


def _choose_best_model(models_stats: List[Tuple[str, Dict[str, float]]]) -> Optional[str]:
    """Pick best model by minimizing topology/structure errors and maximizing detection.

    Criteria (equal weight):
    - minimize connected_components_difference_class_1
    - minimize branch_points_absdiff_class_1
    - minimize hausdorff_distance_95_class_1
    - maximize precision_class_1
    - maximize recall_class_1
    """
    if not models_stats:
        return None

    names: List[str] = []
    cc_vals: List[float] = []
    bp_vals: List[float] = []
    hd_vals: List[float] = []
    pr_vals: List[float] = []
    rc_vals: List[float] = []

    def _safe(stats: Dict[str, float], key: str, want_max: bool = False) -> float:
        v = stats.get(key)
        try:
            vf = float(v)
            if not np.isfinite(vf):
                return np.inf if not want_max else -np.inf
            return vf
        except Exception:
            return np.inf if not want_max else -np.inf

    for name, stats in models_stats:
        names.append(name)
        cc_vals.append(_safe(stats, "connected_components_difference_class_1", want_max=False))
        bp_vals.append(_safe(stats, "branch_points_absdiff_class_1", want_max=False))
        hd_vals.append(_safe(stats, "hausdorff_distance_95_class_1", want_max=False))
        pr_vals.append(_safe(stats, "precision_class_1", want_max=True))
        rc_vals.append(_safe(stats, "recall_class_1", want_max=True))

    def _norm_min(vals: List[float]) -> List[float]:
        arr = np.asarray(vals, dtype=np.float64)
        if not np.isfinite(arr).all():
            finite = arr[np.isfinite(arr)]
            repl = (finite.max() if finite.size else 1.0) * 10.0
            arr[~np.isfinite(arr)] = repl
        lo = float(np.min(arr))
        hi = float(np.max(arr))
        rng = hi - lo
        if rng <= 1e-12:
            return [0.0] * len(arr)
        return [(float(v) - lo) / rng for v in arr]

    def _norm_max(vals: List[float]) -> List[float]:
        arr = np.asarray(vals, dtype=np.float64)
        if not np.isfinite(arr).all():
            finite = arr[np.isfinite(arr)]
            repl = (finite.min() if finite.size else 0.0) - 10.0
            arr[~np.isfinite(arr)] = repl
        lo = float(np.min(arr))
        hi = float(np.max(arr))
        rng = hi - lo
        if rng <= 1e-12:
            return [0.0] * len(arr)
        return [(hi - float(v)) / rng for v in arr]

    cc_n = _norm_min(cc_vals)
    bp_n = _norm_min(bp_vals)
    hd_n = _norm_min(hd_vals)
    pr_n = _norm_max(pr_vals)
    rc_n = _norm_max(rc_vals)

    totals = [cc_n[i] + bp_n[i] + hd_n[i] + pr_n[i] + rc_n[i] for i in range(len(names))]

    best_idx = min(
        range(len(names)),
        key=lambda i: (totals[i], -rc_vals[i], -pr_vals[i], hd_vals[i], bp_vals[i], cc_vals[i])
    )
    return names[best_idx]


def _rank_models(models_stats: List[Tuple[str, Dict[str, float]]]) -> List[str]:
    """Return model names ranked best→worst using the same scoring and tie-breakers as _choose_best_model."""
    if not models_stats:
        return []

    names: List[str] = []
    cc_vals: List[float] = []
    bp_vals: List[float] = []
    hd_vals: List[float] = []
    pr_vals: List[float] = []
    rc_vals: List[float] = []

    def _safe(stats: Dict[str, float], key: str, want_max: bool = False) -> float:
        v = stats.get(key)
        try:
            vf = float(v)
            if not np.isfinite(vf):
                return np.inf if not want_max else -np.inf
            return vf
        except Exception:
            return np.inf if not want_max else -np.inf

    for name, stats in models_stats:
        names.append(name)
        cc_vals.append(_safe(stats, "connected_components_difference_class_1", want_max=False))
        bp_vals.append(_safe(stats, "branch_points_absdiff_class_1", want_max=False))
        hd_vals.append(_safe(stats, "hausdorff_distance_95_class_1", want_max=False))
        pr_vals.append(_safe(stats, "precision_class_1", want_max=True))
        rc_vals.append(_safe(stats, "recall_class_1", want_max=True))

    def _norm_min(vals: List[float]) -> List[float]:
        arr = np.asarray(vals, dtype=np.float64)
        if not np.isfinite(arr).all():
            finite = arr[np.isfinite(arr)]
            repl = (finite.max() if finite.size else 1.0) * 10.0
            arr[~np.isfinite(arr)] = repl
        lo = float(np.min(arr))
        hi = float(np.max(arr))
        rng = hi - lo
        if rng <= 1e-12:
            return [0.0] * len(arr)
        return [(float(v) - lo) / rng for v in arr]

    def _norm_max(vals: List[float]) -> List[float]:
        arr = np.asarray(vals, dtype=np.float64)
        if not np.isfinite(arr).all():
            finite = arr[np.isfinite(arr)]
            repl = (finite.min() if finite.size else 0.0) - 10.0
            arr[~np.isfinite(arr)] = repl
        lo = float(np.min(arr))
        hi = float(np.max(arr))
        rng = hi - lo
        if rng <= 1e-12:
            return [0.0] * len(arr)
        return [(hi - float(v)) / rng for v in arr]

    cc_n = _norm_min(cc_vals)
    bp_n = _norm_min(bp_vals)
    hd_n = _norm_min(hd_vals)
    pr_n = _norm_max(pr_vals)
    rc_n = _norm_max(rc_vals)

    totals = [cc_n[i] + bp_n[i] + hd_n[i] + pr_n[i] + rc_n[i] for i in range(len(names))]

    ranked = sorted(
        range(len(names)),
        key=lambda i: (totals[i], -rc_vals[i], -pr_vals[i], hd_vals[i], bp_vals[i], cc_vals[i])
    )
    return [names[i] for i in ranked]


def _draw_ranking_pad(width: int, names: List[str], line_h: int = 20, side_pad: int = 12,
                      header_color: Tuple[int, int, int] = (255, 255, 0),
                      value_color: Tuple[int, int, int] = (230, 230, 230)) -> np.ndarray:
    """Render a padded banner with the ranked model names, wrapped to fit width.

    Returns an RGB uint8 array of shape [H_pad, width, 3].
    """
    if not names:
        return np.zeros((max(20, line_h), width, 3), dtype=np.uint8)

    sp = max(0, min(side_pad, width // 10))
    # Rough character width estimate for default font
    chars_per_line = max(16, (width - 2 * sp) // 9)

    tokens = [f"{i+1}) {n}" for i, n in enumerate(names)]
    lines: List[str] = []
    current = "Ranking: "
    for tok in tokens:
        candidate = (current + ("  " if current.strip() else "") + tok).strip()
        if len(candidate) > chars_per_line and current.strip():
            lines.append(current)
            current = tok
        else:
            current = candidate
    if current:
        lines.append(current)

    pad_h = 6 + line_h * len(lines) + 6
    img = np.zeros((pad_h, width, 3), dtype=np.uint8)

    if _HAS_PIL:
        pil = Image.fromarray(img)
        draw = ImageDraw.Draw(pil)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        y = 4
        for i, line in enumerate(lines):
            color = header_color if i == 0 else value_color
            try:
                draw.text((sp + 6, y), line, fill=tuple(color), font=font)
            except TypeError:
                draw.text((sp + 6, y), line, fill=tuple(color), font=font)
            y += line_h
        return np.asarray(pil, dtype=np.uint8)

    if _HAS_CV2:
        y = 14
        for i, line in enumerate(lines):
            color = header_color if i == 0 else value_color
            bgr = (int(color[2]), int(color[1]), int(color[0]))
            cv2.putText(img, line, (sp + 6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 1, cv2.LINE_AA)
            y += line_h
        return img

    return img


def _draw_right_ranking_panel(height: int, width: int, names: List[str], line_h: int = 20,
                              side_pad: int = 12, text_color: Tuple[int, int, int] = (255, 255, 0)) -> np.ndarray:
    """Render a right-side vertical ranking panel with yellow text. One model per line.

    Returns an RGB uint8 array of shape [height, width, 3].
    """
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    if not names:
        return panel
    sp = max(0, min(side_pad, width // 10))
    if _HAS_PIL:
        pil = Image.fromarray(panel)
        draw = ImageDraw.Draw(pil)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        y = 6
        # Header
        try:
            draw.text((sp + 4, y), "Ranking", fill=tuple(text_color), font=font)
        except TypeError:
            draw.text((sp + 4, y), "Ranking", fill=tuple(text_color), font=font)
        y += line_h
        # Items
        for i, name in enumerate(names, start=1):
            line = f"{i}) {name}"
            draw.text((sp + 4, y), line, fill=tuple(text_color), font=font)
            y += line_h
            if y >= height - line_h:
                break
        return np.asarray(pil, dtype=np.uint8)
    if _HAS_CV2:
        y = 16
        cv2.putText(panel, "Ranking", (sp + 4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (int(text_color[2]), int(text_color[1]), int(text_color[0])), 1, cv2.LINE_AA)
        y += line_h
        for i, name in enumerate(names, start=1):
            line = f"{i}) {name}"
            cv2.putText(panel, line, (sp + 4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (int(text_color[2]), int(text_color[1]), int(text_color[0])), 1, cv2.LINE_AA)
            y += line_h
            if y >= height - line_h:
                break
        return panel
    return panel


def _collect_aggregated_model_stats(eval_root: Path, images_dir: Path,
                                    only_models: Optional[Sequence[str]] = None) -> Tuple[List[Tuple[str, Dict[str, float]]], int]:
    """Read per-image stats.json for each model and aggregate the metrics used for ranking.

    Returns (models_stats_list, num_images_considered), where models_stats_list is a list of (model_name, stats_means).
    The stats_means dict contains means of the five keys used in ranking when available.
    """
    # Determine models
    model_dirs = [d for d in sorted(eval_root.iterdir()) if d.is_dir()]
    if only_models:
        sel = set(only_models)
        model_dirs = [d for d in model_dirs if d.name in sel]

    image_files = _discover_tifs(images_dir)
    stems = [p.stem for p in image_files]

    keys_min = [
        "connected_components_difference_class_1",
        "branch_points_absdiff_class_1",
        "hausdorff_distance_95_class_1",
    ]
    keys_max = [
        "precision_class_1",
        "recall_class_1",
    ]
    all_keys = keys_min + keys_max

    # Accumulators per model
    sums: Dict[str, Dict[str, float]] = {}
    counts: Dict[str, Dict[str, int]] = {}

    for md in model_dirs:
        sums[md.name] = {k: 0.0 for k in all_keys}
        counts[md.name] = {k: 0 for k in all_keys}

    for stem in stems:
        for md in model_dirs:
            stats_path = md / stem / "stats.json"
            if not stats_path.exists():
                continue
            try:
                import json
                with stats_path.open("r") as f:
                    data = json.load(f)
                if not isinstance(data, dict):
                    continue
            except Exception:
                continue
            # Accumulate available numeric values
            for k in all_keys:
                v = data.get(k, None)
                try:
                    vf = float(v)
                except Exception:
                    continue
                if not np.isfinite(vf):
                    continue
                sums[md.name][k] += vf
                counts[md.name][k] += 1

    # Compute means
    models_stats: List[Tuple[str, Dict[str, float]]] = []
    for md in model_dirs:
        name = md.name
        means: Dict[str, float] = {}
        for k in all_keys:
            c = counts[name][k]
            if c > 0:
                means[k] = sums[name][k] / c
            else:
                # Missing stays absent; ranking safety functions will treat missing as worst
                pass
        models_stats.append((name, means))

    return models_stats, len(stems)


def _print_overall_ranking(eval_root: Path, images_dir: Path, only_models: Optional[Sequence[str]] = None,
                           stats_line_h: int = 20, stats_side: int = 12) -> None:
    models_stats, num_images = _collect_aggregated_model_stats(eval_root, images_dir, only_models)
    ranking = _rank_models(models_stats)
    print(f"\nOverall ranking across {num_images} images:")
    if not ranking:
        print("(no models found)")
        return
    # Build lookup for pretty print
    stats_map = {name: stats for name, stats in models_stats}
    print("rank  model                             cc_diff  branch_abs  hd95    prec    recall")
    print("----  --------------------------------  -------  ----------  -----  ------  ------")
    for i, name in enumerate(ranking, start=1):
        s = stats_map.get(name, {})
        def fmt(k, width):
            v = s.get(k, None)
            try:
                if v is None:
                    return f"{'-':>{width}}"
                vf = float(v)
                if abs(vf - round(vf)) < 1e-6 and abs(vf) > 1.0:
                    return f"{int(round(vf)):>{width}}"
                return f"{vf:>{width}.4f}"
            except Exception:
                return f"{'-':>{width}}"
        print(
            f"{i:>4}  {name:<32}  {fmt('connected_components_difference_class_1',7)}  "
            f"{fmt('branch_points_absdiff_class_1',10)}  {fmt('hausdorff_distance_95_class_1',5)}  "
            f"{fmt('precision_class_1',6)}  {fmt('recall_class_1',6)}"
        )
    print(f"\nBest overall: {ranking[0]}")


def _process_task(task: Task, *, alpha: float, pred_alpha: float, threshold: float, pad_px: int,
                  max_cols: int, hpad: int, label_color: Tuple[int, int, int], pred_color: Tuple[int, int, int],
                  rescale: bool, stats_line_h: int, stats_side: int) -> Tuple[str, bool, str]:
    """Process one image -> write mosaic. Returns (stem, success, message)."""
    stem = task.image_path.stem
    try:
        base = tifffile.imread(str(task.image_path))
    except Exception as e:
        return stem, False, f"Failed to read image: {e}"

    # Convert base to uint8 RGB
    base_u8 = _to_uint8_grayscale(base, rescale=rescale)
    base_rgb = _ensure_4d_rgb(base_u8)  # HxWx3 or ZxHxWx3

    tiles: List[np.ndarray] = []

    # Determine label path (if not already provided)
    label_path = task.label_path
    if label_path is None:
        for _, per_img_dir in task.model_items:
            cand = per_img_dir / "label.tif"
            if cand.exists():
                label_path = cand
                break

    # GT tile and reuseable GT mask for later overlays
    title_color = (255, 255, 255)
    gt_mask_b_for_tiles = None
    if label_path and label_path.exists():
        try:
            gt = tifffile.imread(str(label_path))
        except Exception as e:
            gt = None
        if gt is not None:
            gt_mask = _binarize_mask(gt, threshold)
            # Shape check and basic broadcasting for Z if needed
            # Expect gt_mask to match base spatial dims; if 2D mask and base is 3D, broadcast along Z
            try:
                if base_rgb.ndim == 4 and gt_mask.ndim == 2:
                    gt_mask_b = np.repeat(gt_mask[None, ...], base_rgb.shape[0], axis=0)
                else:
                    gt_mask_b = gt_mask
            except Exception:
                gt_mask_b = gt_mask
            gt_mask_b_for_tiles = gt_mask_b
            try:
                gt_overlay = _overlay_color(base_rgb, gt_mask_b, label_color, alpha)
            except Exception:
                gt_overlay = _ensure_4d_rgb(base_u8)
            gt_titled = _pad_and_title_tile(gt_overlay, "GT", pad_px, title_color)
            tiles.append(gt_titled)
        else:
            # Fallback: no overlay, just base with title
            tiles.append(_pad_and_title_tile(base_rgb, "GT (missing)", pad_px, title_color))
    else:
        tiles.append(_pad_and_title_tile(base_rgb, "GT (missing)", pad_px, title_color))

    # Pred tiles and collect stats
    models_stats: List[Tuple[str, Dict[str, float]]] = []
    for model_name, per_img_dir in task.model_items:
        pred_path = per_img_dir / "pred.tif"
        stats_path = per_img_dir / "stats.json"

        # Stats (best-effort)
        stats_dict: Dict[str, float] = {}
        if stats_path.exists():
            try:
                import json
                with stats_path.open("r") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    stats_dict = {k: float(v) if isinstance(v, (int, float)) else v for k, v in data.items()}
            except Exception:
                pass
        models_stats.append((model_name, stats_dict))

        if not pred_path.exists():
            # Show GT overlay if available
            if gt_mask_b_for_tiles is not None:
                try:
                    base_plus_gt = _overlay_color(base_rgb, gt_mask_b_for_tiles, label_color, alpha)
                except Exception:
                    base_plus_gt = base_rgb
                tiles.append(_pad_and_title_tile(base_plus_gt, f"{model_name} (missing)", pad_px, title_color))
            else:
                tiles.append(_pad_and_title_tile(base_rgb, f"{model_name} (missing)", pad_px, title_color))
            continue
        try:
            pred = tifffile.imread(str(pred_path))
        except Exception:
            if gt_mask_b_for_tiles is not None:
                try:
                    base_plus_gt = _overlay_color(base_rgb, gt_mask_b_for_tiles, label_color, alpha)
                except Exception:
                    base_plus_gt = base_rgb
                tiles.append(_pad_and_title_tile(base_plus_gt, f"{model_name} (read error)", pad_px, title_color))
            else:
                tiles.append(_pad_and_title_tile(base_rgb, f"{model_name} (read error)", pad_px, title_color))
            continue
        pred_mask = _binarize_mask(pred, threshold)
        try:
            if base_rgb.ndim == 4 and pred_mask.ndim == 2:
                pred_mask_b = np.repeat(pred_mask[None, ...], base_rgb.shape[0], axis=0)
            else:
                pred_mask_b = pred_mask
        except Exception:
            pred_mask_b = pred_mask
        # Overlay order with exclusivity: draw GT only where pred is absent; draw pred on its mask.
        try:
            working = base_rgb
            if gt_mask_b_for_tiles is not None:
                # Exclude overlap so GT doesn't show under prediction
                if isinstance(gt_mask_b_for_tiles, np.ndarray) and isinstance(pred_mask_b, np.ndarray):
                    gt_only = np.logical_and(gt_mask_b_for_tiles.astype(bool), np.logical_not(pred_mask_b.astype(bool)))
                else:
                    gt_only = gt_mask_b_for_tiles
                working = _overlay_color(working, gt_only, label_color, alpha)
            # Apply prediction overlay with stronger alpha
            pred_overlay = _overlay_color(working, pred_mask_b, pred_color, pred_alpha)
        except Exception:
            pred_overlay = _ensure_4d_rgb(base_u8)
        tiles.append(_pad_and_title_tile(pred_overlay, model_name, pad_px, title_color))

    # Normalize tile types and assemble grid
    try:
        grid = _assemble_grid(tiles, max_cols=max_cols, hpad=hpad)
    except Exception as e:
        return stem, False, f"Grid assembly failed: {e}"

    # Append right ranking panel, then bottom stats table
    if grid.ndim == 3:
        H, W, _ = grid.shape
        ranking = _rank_models(models_stats)
        rank_panel = _draw_right_ranking_panel(H, max(200, stats_side * 8), ranking, line_h=stats_line_h, side_pad=stats_side)
        grid = np.concatenate([grid, rank_panel], axis=1)
        # Now draw bottom stats under the wider grid
        H2, W2, _ = grid.shape
        stats_table = _draw_bottom_stats_table(W2, models_stats, line_h=stats_line_h, side_pad=stats_side,
                                               model_col_pad=max(24, stats_side * 2))
        grid = np.vstack([grid, stats_table])
    else:
        Z, H, W, _ = grid.shape
        ranking = _rank_models(models_stats)
        rank_panel_2d = _draw_right_ranking_panel(H, max(200, stats_side * 8), ranking, line_h=stats_line_h, side_pad=stats_side)
        rank_stack = np.repeat(rank_panel_2d[None, ...], Z, axis=0)
        grid = np.concatenate([grid, rank_stack], axis=2)
        # Bottom stats under the widened Z-stack
        Z, H2, W2, _ = grid.shape
        stats_table_2d = _draw_bottom_stats_table(W2, models_stats, line_h=stats_line_h, side_pad=stats_side,
                                                  model_col_pad=max(24, stats_side * 2))
        stats_stack = np.repeat(stats_table_2d[None, ...], Z, axis=0)
        grid = np.concatenate([grid, stats_stack], axis=1)

    # Write TIF (use single imwrite; tifffile handles ZxHxWx3 as multi-page RGB)
    try:
        task.out_path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(str(task.out_path), grid, photometric='rgb', compression='lzw')
    except Exception as e:
        return stem, False, f"Failed to write output: {e}"

    return stem, True, f"Wrote {task.out_path.name}"


def build_tasks(images_dir: Path, eval_root: Path, output_dir: Path,
                only_models: Optional[Sequence[str]] = None,
                overwrite: bool = False) -> List[Task]:
    image_files = _discover_tifs(images_dir)
    model_dirs = [d for d in sorted(eval_root.iterdir()) if d.is_dir()]
    if only_models:
        sel = set(only_models)
        model_dirs = [d for d in model_dirs if d.name in sel]

    tasks: List[Task] = []
    for img_path in image_files:
        stem = img_path.stem
        model_items: List[Tuple[str, Path]] = []
        for md in model_dirs:
            per_img = md / stem
            model_items.append((md.name, per_img))
        # Find label once (first model that has it)
        label_path: Optional[Path] = None
        for _, per_img in model_items:
            cand = per_img / "label.tif"
            if cand.exists():
                label_path = cand
                break
        out_path = output_dir / f"{stem}_grid.tif"
        if out_path.exists() and not overwrite:
            continue
        tasks.append(Task(image_path=img_path, label_path=label_path, model_items=model_items, out_path=out_path))
    return tasks


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Build grid TIFs showing GT and model prediction overlays.")
    p.add_argument("--eval-root", required=True, type=Path, help="Root dir of eval_offline outputs (contains per-model folders)")
    p.add_argument("--images", required=True, type=Path, help="Directory of source image TIFs")
    p.add_argument("--output", required=True, type=Path, help="Directory to write grid TIFs")

    p.add_argument("--models", default=None, help="Comma-separated subset of model folder names to include")

    p.add_argument("--alpha", type=float, default=0.4, help="GT overlay alpha in [0,1]")
    p.add_argument("--pred-alpha", type=float, default=0.8, help="Prediction overlay alpha in [0,1]")
    p.add_argument("--threshold", type=float, default=0.5, help="Threshold for float masks")
    p.add_argument("--max-cols", type=int, default=4, help="Max columns per grid row")
    p.add_argument("--hpad", type=int, default=4, help="Horizontal padding in pixels between tiles in the grid")
    p.add_argument("--pad", type=int, default=24, help="Pad height above each tile for text label (pixels)")
    p.add_argument("--label-color", default="#ff0000", help="GT overlay color (hex or name)")
    p.add_argument("--pred-color", default="blue", help="Prediction overlay color (hex or name)")
    p.add_argument("--no-rescale", action="store_true", help="Disable percentile rescaling of base image to 8-bit")
    p.add_argument("--stats-line", type=int, default=20, help="Row height (pixels) for stats table below the grid")
    p.add_argument("--stats-side", type=int, default=12, help="Left/right padding inside stats table and best banner")
    p.add_argument("--workers", type=int, default=0, help="Number of worker processes (0 -> CPU count)")
    p.add_argument("--summary-only", action="store_true", help="Only print overall ranking from stats.json; do not create grids")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")

    args = p.parse_args(argv)

    eval_root = args.eval_root
    images_dir = args.images
    output_dir = args.output
    only_models = [s.strip() for s in args.models.split(",")] if args.models else None

    label_color = _parse_color(args.label_color)
    pred_color = _parse_color(args.pred_color)

    # If summary-only, just compute and print ranking and exit
    if args.summary_only:
        _print_overall_ranking(eval_root, images_dir, only_models)
        return 0

    tasks = build_tasks(
        images_dir=images_dir,
        eval_root=eval_root,
        output_dir=output_dir,
        only_models=only_models,
        overwrite=args.overwrite,
    )

    if not tasks:
        print("No tasks to process (maybe outputs already exist or inputs missing)")
        # Still print ranking if possible
        _print_overall_ranking(eval_root, images_dir, only_models)
        return 0

    n_workers = int(args.workers or 0)
    if n_workers <= 0:
        try:
            import os
            n_workers = max(1, len(os.sched_getaffinity(0)))
        except Exception:
            import multiprocessing as mp
            n_workers = max(1, mp.cpu_count())

    futures = []
    successes = 0
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for t in tasks:
            futures.append(ex.submit(
                _process_task,
                t,
                alpha=float(args.alpha),
                pred_alpha=float(args.pred_alpha),
                threshold=float(args.threshold),
                pad_px=int(args.pad),
                max_cols=int(args.max_cols),
                hpad=int(args.hpad),
                label_color=label_color,
                pred_color=pred_color,
                rescale=not args.no_rescale,
                stats_line_h=int(args.stats_line),
                stats_side=int(args.stats_side),
            ))

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Grids"):
            stem, ok, msg = fut.result()
            if ok:
                successes += 1
            else:
                print(f"[{stem}] Warning: {msg}")

    print(f"Completed {successes}/{len(tasks)} grids -> {output_dir}")
    # Print overall ranking after building
    _print_overall_ranking(eval_root, images_dir, only_models)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
