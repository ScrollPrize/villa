import argparse
import colorsys
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import tifffile
from scipy.spatial import cKDTree

from vesuvius.tifxyz import list_tifxyz, read_tifxyz

SIDE_FRONT = "front"
SIDE_BACK = "back"
SIDES = (SIDE_FRONT, SIDE_BACK)


@dataclass(frozen=True)
class DetectorConfig:
    edge_depth_rows: Optional[int] = None
    max_band_rows: Optional[int] = None
    band_inward_margin_rows: int = 24
    row_point_stride: int = 4
    min_row_points: int = 32
    min_band_rows: int = 8
    score_threshold: float = 1.5


@dataclass(frozen=True)
class WrapInfo:
    path: Path
    name: str
    wrap_ids: Tuple[int, ...]


@dataclass
class LoadedWrap:
    info: WrapInfo
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    valid: np.ndarray

    @property
    def shape(self) -> Tuple[int, int]:
        return self.x.shape


@dataclass(frozen=True)
class BandDetection:
    source_name: str
    target_name: str
    source_side: str
    target_side: str
    score: float
    last_selected_col: int
    band_col_mask: np.ndarray


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    def _opt_pos_int(value: str) -> Optional[int]:
        text = str(value).strip().lower()
        if text in {"none", "auto", "unbounded"}:
            return None
        parsed = int(text)
        if parsed < 1:
            raise argparse.ArgumentTypeError("value must be >= 1 or 'none'")
        return parsed

    parser = argparse.ArgumentParser(
        description=(
            "Detect contiguous wrap-overlap bands in tifxyz UV space and write "
            "overlap_mask.tif for each wrap."
        )
    )
    parser.add_argument("--paths-dir", type=Path, required=True, help="Directory containing wrap tifxyz folders.")
    parser.add_argument(
        "--pair-mode",
        type=str,
        default="consecutive_self",
        choices=("consecutive_self", "consecutive", "self_only", "all_pairs_self"),
        help="Which wrap comparisons to run.",
    )
    parser.add_argument(
        "--edge-depth-rows",
        type=_opt_pos_int,
        default=None,
        help="Columns sampled from each edge inward. Use 'none' (default) to scan full width.",
    )
    parser.add_argument(
        "--max-band-rows",
        type=_opt_pos_int,
        default=None,
        help="Maximum overlap band depth from an edge. Use 'none' (default) for no hard cap.",
    )
    parser.add_argument("--row-point-stride", type=int, default=4, help="Row stride for column point sampling.")
    parser.add_argument("--min-row-points", type=int, default=32, help="Minimum valid points per column candidate.")
    parser.add_argument("--min-band-rows", type=int, default=8, help="Minimum accepted overlap-band column count.")
    parser.add_argument("--score-threshold", type=float, default=1.5, help="Minimum edge-jump detection score.")
    parser.add_argument(
        "--band-inward-margin-rows",
        type=int,
        default=24,
        help=(
            "Expand each detected edge band inward by this many columns before writing masks. "
            "This reduces brittle misses from small edge-cutoff shifts."
        ),
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing overlap_mask.tif files.")
    parser.add_argument("--napari", action="store_true", help="Display wrap and overlap points in napari.")
    parser.add_argument("--napari-downsample", type=int, default=8, help="Point downsample stride for napari.")
    parser.add_argument("--napari-point-size", type=float, default=3.0, help="Point size for napari point layers.")
    parser.add_argument(
        "--napari-zrange",
        type=float,
        nargs=2,
        default=None,
        metavar=("Z_MIN", "Z_MAX"),
        help="Optional z-range filter for napari points (inclusive).",
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-pair detection details.")
    args = parser.parse_args(argv)

    if args.row_point_stride < 1:
        parser.error("--row-point-stride must be >= 1")
    if args.min_row_points < 1:
        parser.error("--min-row-points must be >= 1")
    if args.min_band_rows < 1:
        parser.error("--min-band-rows must be >= 1")
    if args.band_inward_margin_rows < 0:
        parser.error("--band-inward-margin-rows must be >= 0")
    if args.max_band_rows is not None and args.max_band_rows < args.min_band_rows:
        parser.error("--max-band-rows must be >= --min-band-rows")
    if args.napari_downsample < 1:
        parser.error("--napari-downsample must be >= 1")
    if args.napari_point_size <= 0:
        parser.error("--napari-point-size must be > 0")
    if args.napari_zrange is not None and float(args.napari_zrange[0]) > float(args.napari_zrange[1]):
        parser.error("--napari-zrange requires Z_MIN <= Z_MAX")
    return args


def _wrap_infos(paths_dir: Path) -> List[WrapInfo]:
    required = ("x.tif", "y.tif", "z.tif", "meta.json")
    if paths_dir.is_dir() and all((paths_dir / name).exists() for name in required):
        name = paths_dir.name
        wrap_ids = _extract_wrap_ids(name)
        if not wrap_ids:
            return []
        return [WrapInfo(path=paths_dir, name=name, wrap_ids=wrap_ids)]

    infos: List[WrapInfo] = []
    for item in list_tifxyz(paths_dir, recursive=False):
        name = item.path.name
        wrap_ids = _extract_wrap_ids(name)
        if not wrap_ids:
            continue
        infos.append(WrapInfo(path=item.path, name=name, wrap_ids=wrap_ids))
    infos.sort(key=lambda x: (x.wrap_ids[0], x.wrap_ids[-1], x.name))
    return infos


def _extract_wrap_ids(name: str) -> Tuple[int, ...]:
    wrap_ids = sorted({int(m.group(1)) for m in re.finditer(r"w(\d+)", name)})
    return tuple(wrap_ids)


def _has_consecutive_wrap_ids(left_ids: Tuple[int, ...], right_ids: Tuple[int, ...]) -> bool:
    right_set = set(right_ids)
    for wrap_id in left_ids:
        if (wrap_id - 1) in right_set or (wrap_id + 1) in right_set:
            return True
    return False


def _make_pair_lists(
    infos: List[WrapInfo],
    pair_mode: str,
) -> Tuple[List[Tuple[WrapInfo, WrapInfo]], List[WrapInfo]]:
    inter_pairs: List[Tuple[WrapInfo, WrapInfo]] = []
    self_items: List[WrapInfo] = []

    if pair_mode in ("consecutive_self", "consecutive"):
        for i, left in enumerate(infos):
            for right in infos[i + 1:]:
                if _has_consecutive_wrap_ids(left.wrap_ids, right.wrap_ids):
                    inter_pairs.append((left, right))
    elif pair_mode == "all_pairs_self":
        for i, left in enumerate(infos):
            for right in infos[i + 1:]:
                inter_pairs.append((left, right))

    if pair_mode in ("consecutive_self", "self_only", "all_pairs_self"):
        self_items = list(infos)

    return inter_pairs, self_items


def _median_smooth_1d(values: np.ndarray, window: int = 5) -> np.ndarray:
    if values.size == 0 or window <= 1:
        return values.copy()
    radius = window // 2
    out = np.empty_like(values, dtype=np.float64)
    for idx in range(values.size):
        lo = max(0, idx - radius)
        hi = min(values.size, idx + radius + 1)
        out[idx] = float(np.median(values[lo:hi]))
    return out


def _robust_z(values: np.ndarray) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float64)
    finite = np.isfinite(values)
    if finite.sum() < 3:
        return out
    vals = values[finite].astype(np.float64, copy=False)
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    if mad < 1e-9:
        std = float(vals.std())
        if std < 1e-9:
            return out
        out[finite] = (vals - med) / std
        return out
    scale = max(1e-9, 1.4826 * mad)
    out[finite] = (vals - med) / scale
    return out


def _fill_nans_with_nearest(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64).copy()
    if out.size == 0:
        return out
    finite_idx = np.flatnonzero(np.isfinite(out))
    if finite_idx.size == 0:
        return np.zeros_like(out)
    first = int(finite_idx[0])
    last = int(finite_idx[-1])
    out[:first] = out[first]
    out[last + 1:] = out[last]
    for i0, i1 in zip(finite_idx[:-1], finite_idx[1:]):
        if i1 - i0 > 1:
            out[i0 + 1:i1] = out[i0]
    return out


def _positive_outlier_score(values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(vals)
    out = np.full(vals.shape, -np.inf, dtype=np.float64)
    if finite.sum() < 3:
        return out
    v = vals[finite]
    center = float(np.median(v))
    mad = float(np.median(np.abs(v - center)))
    if mad < 1e-9:
        spread = float(v.std())
        if spread < 1e-9:
            out[finite] = 0.0
            return out
        out[finite] = (v - center) / spread
        return out
    spread = 1.4826 * mad
    out[finite] = (v - center) / max(spread, 1e-9)
    return out


def _prefix_objective(prefix_scores: np.ndarray) -> float:
    return float(prefix_scores.mean() - 0.25 * prefix_scores.std())


def _pick_best_prefix_length(
    scores: np.ndarray,
    min_rows: int,
) -> Tuple[Optional[int], Optional[float]]:
    n_rows = int(scores.size)
    if n_rows < min_rows:
        return None, None

    best_len: Optional[int] = None
    best_score: Optional[float] = None
    for length in range(min_rows, n_rows + 1):
        val = _prefix_objective(scores[:length])
        if best_score is None or val > best_score:
            best_len = length
            best_score = val
    return best_len, best_score


def _band_cols_from_edge(width: int, side: str, last_col: int, inward_margin: int = 0) -> np.ndarray:
    col_mask = np.zeros((width,), dtype=bool)
    inward_margin = max(0, int(inward_margin))
    if side == SIDE_FRONT:
        hi = min(width - 1, int(last_col) + inward_margin)
        col_mask[: hi + 1] = True
    elif side == SIDE_BACK:
        lo = max(0, int(last_col) - inward_margin)
        col_mask[lo:] = True
    else:
        raise ValueError(f"Unknown side: {side}")
    return col_mask


class OverlapDetector:
    def __init__(self, wraps: Sequence[LoadedWrap], cfg: DetectorConfig):
        self.wraps = list(wraps)
        self.cfg = cfg
        self._row_cache: Dict[Tuple[Path, str], np.ndarray] = {}
        self._edge_points_cache: Dict[Tuple[Path, str], Tuple[np.ndarray, np.ndarray]] = {}

    def _candidate_rows(self, wrap: LoadedWrap, side: str) -> np.ndarray:
        key = (wrap.info.path, side)
        cached = self._row_cache.get(key)
        if cached is not None:
            return cached

        width = wrap.shape[1]
        row_order = np.arange(width, dtype=np.int32)
        if side == SIDE_BACK:
            row_order = row_order[::-1]
        elif side != SIDE_FRONT:
            raise ValueError(f"Unknown side: {side}")

        depth_limit = width if self.cfg.edge_depth_rows is None else min(width, int(self.cfg.edge_depth_rows))

        rows: List[int] = []
        for row in row_order:
            if int(wrap.valid[:, row].sum()) >= self.cfg.min_row_points:
                rows.append(int(row))
            if len(rows) >= depth_limit:
                break
        result = np.asarray(rows, dtype=np.int32)
        self._row_cache[key] = result
        return result

    def _sample_row_points(self, wrap: LoadedWrap, row: int) -> np.ndarray:
        valid_rows = np.flatnonzero(wrap.valid[:, row])
        if valid_rows.size == 0:
            return np.zeros((0, 3), dtype=np.float32)
        sampled_rows = valid_rows[:: self.cfg.row_point_stride]
        if sampled_rows.size == 0:
            return np.zeros((0, 3), dtype=np.float32)

        points = np.stack(
            [
                wrap.z[sampled_rows, row],
                wrap.y[sampled_rows, row],
                wrap.x[sampled_rows, row],
            ],
            axis=-1,
        ).astype(np.float32, copy=False)
        finite = np.isfinite(points).all(axis=1)
        return points[finite]

    def _edge_points(self, wrap: LoadedWrap, side: str) -> Tuple[np.ndarray, np.ndarray]:
        key = (wrap.info.path, side)
        cached = self._edge_points_cache.get(key)
        if cached is not None:
            return cached

        rows = self._candidate_rows(wrap, side)
        points_parts: List[np.ndarray] = []
        row_ids_parts: List[np.ndarray] = []
        for row in rows:
            points = self._sample_row_points(wrap, int(row))
            if points.shape[0] == 0:
                continue
            points_parts.append(points)
            row_ids_parts.append(np.full((points.shape[0],), int(row), dtype=np.int32))

        if not points_parts:
            result = (np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.int32))
            self._edge_points_cache[key] = result
            return result

        edge_points = np.concatenate(points_parts, axis=0)
        edge_rows = np.concatenate(row_ids_parts, axis=0)
        result = (edge_points, edge_rows)
        self._edge_points_cache[key] = result
        return result

    def detect_band(
        self,
        source: LoadedWrap,
        target: LoadedWrap,
        source_side: str,
        target_side: str,
    ) -> Optional[BandDetection]:
        src_rows = self._candidate_rows(source, source_side)
        if src_rows.size < self.cfg.min_band_rows:
            return None

        tgt_points, tgt_point_rows = self._edge_points(target, target_side)
        if tgt_points.shape[0] < self.cfg.min_row_points:
            return None

        tree = cKDTree(tgt_points)
        min_query_points = max(6, self.cfg.min_row_points // max(1, self.cfg.row_point_stride))

        medians = np.full((src_rows.size,), np.nan, dtype=np.float64)
        q80s = np.full((src_rows.size,), np.nan, dtype=np.float64)
        line_iqrs = np.full((src_rows.size,), np.nan, dtype=np.float64)

        for idx, row in enumerate(src_rows):
            points = self._sample_row_points(source, int(row))
            if points.shape[0] < min_query_points:
                continue
            dists, nearest_idx = tree.query(points, k=1)
            dists = np.asarray(dists, dtype=np.float64)
            nearest_idx = np.asarray(nearest_idx, dtype=np.int64)
            if dists.size == 0:
                continue

            matched_rows = tgt_point_rows[nearest_idx].astype(np.float64, copy=False)
            medians[idx] = float(np.median(dists))
            q80s[idx] = float(np.quantile(dists, 0.80))
            line_iqrs[idx] = float(np.quantile(matched_rows, 0.75) - np.quantile(matched_rows, 0.25))

        finite_mask = np.isfinite(medians) & np.isfinite(q80s) & np.isfinite(line_iqrs)
        if int(finite_mask.sum()) < self.cfg.min_band_rows:
            return None

        medians_filled = _fill_nans_with_nearest(medians)
        iqrs_filled = _fill_nans_with_nearest(line_iqrs)
        medians_smooth = _median_smooth_1d(medians_filled, window=5)
        iqrs_smooth = _median_smooth_1d(iqrs_filled, window=5)

        med_jumps = np.diff(medians_smooth)
        iqr_jumps = np.diff(iqrs_smooth)
        if med_jumps.size == 0:
            return None

        jump_scores = _positive_outlier_score(med_jumps) + 0.35 * _positive_outlier_score(iqr_jumps)
        min_index = self.cfg.min_band_rows - 1
        if min_index >= jump_scores.size:
            return None
        jump_scores[:min_index] = -np.inf

        min_required = float(self.cfg.score_threshold)
        if self.cfg.max_band_rows is None:
            max_index = jump_scores.size - 1
        else:
            max_index = min(jump_scores.size - 1, int(self.cfg.max_band_rows) - 1)
        candidate_idx = np.flatnonzero(jump_scores[: max_index + 1] >= min_required)
        if candidate_idx.size == 0:
            return None
        best_jump_idx = int(candidate_idx[0])
        best_jump_score = float(jump_scores[best_jump_idx])

        best_len = best_jump_idx + 1
        if best_len >= medians_smooth.size:
            return None

        prefix_med = float(np.mean(medians_smooth[:best_len]))
        suffix_med = float(np.mean(medians_smooth[best_len:]))
        if suffix_med <= prefix_med:
            return None

        last_col = int(src_rows[best_len - 1])
        band_col_mask = _band_cols_from_edge(
            source.shape[1],
            source_side,
            last_col,
            inward_margin=self.cfg.band_inward_margin_rows,
        )
        if not bool((band_col_mask[None, :] & source.valid).any()):
            return None

        return BandDetection(
            source_name=source.info.name,
            target_name=target.info.name,
            source_side=source_side,
            target_side=target_side,
            score=best_jump_score,
            last_selected_col=last_col,
            band_col_mask=band_col_mask,
        )

    def detect_best_inter(self, source: LoadedWrap, target: LoadedWrap) -> Optional[BandDetection]:
        best: Optional[BandDetection] = None
        for source_side in SIDES:
            for target_side in SIDES:
                detection = self.detect_band(source, target, source_side, target_side)
                if detection is None:
                    continue
                if best is None or detection.score > best.score:
                    best = detection
        return best

    def detect_self(self, wrap: LoadedWrap) -> List[BandDetection]:
        detections: List[BandDetection] = []
        for source_side, target_side in ((SIDE_FRONT, SIDE_BACK), (SIDE_BACK, SIDE_FRONT)):
            detection = self.detect_band(wrap, wrap, source_side, target_side)
            if detection is not None:
                detections.append(detection)
        return detections


def _load_wrap(info: WrapInfo) -> LoadedWrap:
    surface = read_tifxyz(info.path, load_mask=True, validate=True)
    return LoadedWrap(
        info=info,
        x=np.asarray(surface._x, dtype=np.float32),
        y=np.asarray(surface._y, dtype=np.float32),
        z=np.asarray(surface._z, dtype=np.float32),
        valid=np.asarray(surface._valid_mask, dtype=bool),
    )


def _write_overlap_mask(mask: np.ndarray, wrap_dir: Path, overwrite: bool) -> Path:
    out_path = wrap_dir / "overlap_mask.tif"
    if out_path.exists() and not overwrite:
        raise FileExistsError(
            f"{out_path} already exists. Re-run with --overwrite to replace it."
        )
    tifffile.imwrite(str(out_path), (mask.astype(np.uint8) * 255), compression="zlib")
    return out_path


def _show_napari(
    wraps: Sequence[LoadedWrap],
    overlap_masks: Dict[Path, np.ndarray],
    downsample: int,
    point_size: float,
    zrange: Optional[Tuple[float, float]],
) -> None:
    try:
        import napari
    except Exception as exc:
        raise RuntimeError("--napari was set, but napari is not available.") from exc

    if not wraps:
        print("No wraps to display in napari.")
        return

    viewer = napari.Viewer(ndisplay=3)

    total_wraps = len(wraps)
    for idx, wrap in enumerate(wraps):
        wrap_rgb = np.asarray(
            colorsys.hsv_to_rgb(idx / max(total_wraps, 1), 1.0, 1.0),
            dtype=np.float32,
        )

        wrap_points = np.stack(
            [wrap.z[wrap.valid], wrap.y[wrap.valid], wrap.x[wrap.valid]],
            axis=-1,
        ).astype(np.float32, copy=False)
        wrap_points = _filter_points_by_zrange(wrap_points, zrange)
        if downsample > 1:
            wrap_points = wrap_points[::downsample]
        if wrap_points.shape[0] == 0:
            continue
        layer = viewer.add_points(
            wrap_points,
            name=f"wrap_points::{wrap.info.name}",
            size=point_size,
            face_color=wrap_rgb.tolist(),
            opacity=0.35,
        )
        layer.visible = False

        overlap_mask = overlap_masks[wrap.info.path]
        overlap_points = np.stack(
            [wrap.z[overlap_mask], wrap.y[overlap_mask], wrap.x[overlap_mask]],
            axis=-1,
        ).astype(np.float32, copy=False)
        overlap_points = _filter_points_by_zrange(overlap_points, zrange)
        if downsample > 1:
            overlap_points = overlap_points[::downsample]
        if overlap_points.shape[0] == 0:
            continue
        viewer.add_points(
            overlap_points,
            name=f"overlap_points::{wrap.info.name}",
            size=point_size,
            face_color=wrap_rgb.tolist(),
            opacity=0.95,
        )

    napari.run()


def _filter_points_by_zrange(
    points: np.ndarray,
    zrange: Optional[Tuple[float, float]],
) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
        return pts
    if zrange is None:
        return pts
    z_min, z_max = float(zrange[0]), float(zrange[1])
    keep = (pts[:, 0] >= z_min) & (pts[:, 0] <= z_max)
    return pts[keep]


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    wrap_infos = _wrap_infos(args.paths_dir)
    if not wrap_infos:
        raise RuntimeError(f"No tifxyz wraps found in: {args.paths_dir}")

    loaded_wraps = [_load_wrap(info) for info in wrap_infos]
    wraps_by_name = {wrap.info.name: wrap for wrap in loaded_wraps}
    inter_pairs, self_items = _make_pair_lists(wrap_infos, args.pair_mode)

    cfg = DetectorConfig(
        edge_depth_rows=args.edge_depth_rows,
        max_band_rows=args.max_band_rows,
        band_inward_margin_rows=int(args.band_inward_margin_rows),
        row_point_stride=args.row_point_stride,
        min_row_points=args.min_row_points,
        min_band_rows=args.min_band_rows,
        score_threshold=args.score_threshold,
    )
    detector = OverlapDetector(loaded_wraps, cfg)
    overlap_masks: Dict[Path, np.ndarray] = {
        wrap.info.path: np.zeros(wrap.shape, dtype=bool) for wrap in loaded_wraps
    }

    inter_hits = 0
    for left_info, right_info in inter_pairs:
        left = wraps_by_name[left_info.name]
        right = wraps_by_name[right_info.name]

        left_detection = detector.detect_best_inter(left, right)
        if left_detection is not None:
            overlap_masks[left.info.path] |= left_detection.band_col_mask[None, :] & left.valid
            inter_hits += 1
            if args.verbose:
                print(
                    f"[inter] {left_detection.source_name} ({left_detection.source_side}) "
                    f"<- {left_detection.target_name} ({left_detection.target_side}) "
                    f"score={left_detection.score:.3f}"
                )

        right_detection = detector.detect_best_inter(right, left)
        if right_detection is not None:
            overlap_masks[right.info.path] |= right_detection.band_col_mask[None, :] & right.valid
            inter_hits += 1
            if args.verbose:
                print(
                    f"[inter] {right_detection.source_name} ({right_detection.source_side}) "
                    f"<- {right_detection.target_name} ({right_detection.target_side}) "
                    f"score={right_detection.score:.3f}"
                )

    self_hits = 0
    for info in self_items:
        wrap = wraps_by_name[info.name]
        detections = detector.detect_self(wrap)
        for detection in detections:
            overlap_masks[wrap.info.path] |= detection.band_col_mask[None, :] & wrap.valid
            self_hits += 1
            if args.verbose:
                print(
                    f"[self] {detection.source_name} ({detection.source_side}) "
                    f"<- ({detection.target_side}) score={detection.score:.3f}"
                )

    for wrap in loaded_wraps:
        mask = overlap_masks[wrap.info.path]
        out_path = _write_overlap_mask(mask, wrap.info.path, overwrite=bool(args.overwrite))
        n_cols = int(mask.any(axis=0).sum())
        n_points = int(mask.sum())
        n_valid = int(wrap.valid.sum())
        masked_valid_pct = 0.0 if n_valid <= 0 else (100.0 * float(n_points) / float(n_valid))
        print(
            f"{wrap.info.name}: cols={n_cols} points={n_points} "
            f"masked_valid_pct={masked_valid_pct:.2f}% -> {out_path}"
        )

    print(
        f"Completed overlap detection: wraps={len(loaded_wraps)} "
        f"inter_hits={inter_hits} self_hits={self_hits}"
    )

    if args.napari:
        _show_napari(
            loaded_wraps,
            overlap_masks,
            downsample=int(args.napari_downsample),
            point_size=float(args.napari_point_size),
            zrange=None if args.napari_zrange is None else (float(args.napari_zrange[0]), float(args.napari_zrange[1])),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
