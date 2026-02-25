from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from vesuvius.neural_tracing.datasets import detect_wrap_overlap_masks as overlap_module
from vesuvius.tifxyz import read_tifxyz


PHERC0139_ROOT = Path(__file__).resolve().parent / "test_tifxyzs" / "PHerc0139"
PARENT_WRAP_NAMES = (
    "w018_20260130192417321",
    "w019_20260203030108447",
    "w020_20260219135139420",
)


@dataclass(frozen=True)
class ProbeStats:
    hit_count: int
    point_count: int
    hit_fraction: float
    median_distance: float
    p95_distance: float
    max_distance: float


def _resolve_existing_name(candidates: tuple[str, ...]) -> str:
    for name in candidates:
        if (PHERC0139_ROOT / name).is_dir():
            return name
    raise FileNotFoundError(
        f"None of the expected cutout directories exist under {PHERC0139_ROOT}: {candidates}"
    )


@lru_cache(maxsize=1)
def cutout_name_map() -> dict[str, str]:
    return {
        "w018_no_overlap": _resolve_existing_name(("w018_no_overlap",)),
        "w019_no_overlap": _resolve_existing_name(("w019_no_overlap",)),
        "w018_w019_overlap": _resolve_existing_name(("w018_w019_overlap", "w019_w018_overlap")),
        "w019_w020_overlap": _resolve_existing_name(("w019_w020_overlap", "w020_w019_overlap")),
    }


@lru_cache(maxsize=1)
def cutout_wrap_names() -> tuple[str, ...]:
    mapping = cutout_name_map()
    return (
        mapping["w018_no_overlap"],
        mapping["w019_no_overlap"],
        mapping["w018_w019_overlap"],
        mapping["w019_w020_overlap"],
    )


def _load_wrap_by_name(name: str) -> overlap_module.LoadedWrap:
    infos = overlap_module._wrap_infos(PHERC0139_ROOT / name)
    if len(infos) != 1:
        raise RuntimeError(f"Expected exactly one tifxyz wrap in {PHERC0139_ROOT / name}, got {len(infos)}")
    return overlap_module._load_wrap(infos[0])


@lru_cache(maxsize=1)
def loaded_wraps_by_name() -> dict[str, overlap_module.LoadedWrap]:
    names = PARENT_WRAP_NAMES + cutout_wrap_names()
    return {name: _load_wrap_by_name(name) for name in names}


@lru_cache(maxsize=1)
def parent_point_index():
    index = {}
    wraps = loaded_wraps_by_name()
    for parent_name in PARENT_WRAP_NAMES:
        wrap = wraps[parent_name]
        rc = np.argwhere(wrap.valid)
        points = np.stack([wrap.z[wrap.valid], wrap.y[wrap.valid], wrap.x[wrap.valid]], axis=1).astype(np.float32)
        index[parent_name] = {
            "rows": rc[:, 0],
            "cols": rc[:, 1],
            "tree": cKDTree(points),
        }
    return index


def cutout_points(cutout_name: str) -> np.ndarray:
    wrap = loaded_wraps_by_name()[cutout_name]
    return np.stack([wrap.z[wrap.valid], wrap.y[wrap.valid], wrap.x[wrap.valid]], axis=1).astype(np.float32)


def map_cutout_to_parent(cutout_name: str, parent_name: str):
    points = cutout_points(cutout_name)
    parent_idx = parent_point_index()[parent_name]
    dists, nearest_idx = parent_idx["tree"].query(points, k=1)
    rows = parent_idx["rows"][nearest_idx]
    cols = parent_idx["cols"][nearest_idx]
    return rows, cols, np.asarray(dists, dtype=np.float64)


def projected_bbox_2d(cutout_name: str, parent_name: str) -> tuple[int, int, int, int]:
    rows, cols, _ = map_cutout_to_parent(cutout_name, parent_name)
    return (int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max()))


def probe_mask(mask: np.ndarray, cutout_name: str, parent_name: str) -> ProbeStats:
    rows, cols, dists = map_cutout_to_parent(cutout_name, parent_name)
    mask_arr = np.asarray(mask) > 0
    hits = mask_arr[rows, cols]
    hit_count = int(hits.sum())
    point_count = int(hits.size)
    hit_fraction = 0.0 if point_count == 0 else float(hit_count) / float(point_count)
    return ProbeStats(
        hit_count=hit_count,
        point_count=point_count,
        hit_fraction=hit_fraction,
        median_distance=float(np.median(dists)),
        p95_distance=float(np.quantile(dists, 0.95)),
        max_distance=float(np.max(dists)),
    )


def detect_parent_overlap_masks(
    cfg: overlap_module.DetectorConfig | None = None,
) -> dict[str, np.ndarray]:
    wraps_by_name = {name: loaded_wraps_by_name()[name] for name in PARENT_WRAP_NAMES}
    wraps = [wraps_by_name[name] for name in PARENT_WRAP_NAMES]
    detector = overlap_module.OverlapDetector(wraps, cfg or overlap_module.DetectorConfig())
    infos = [wrap.info for wrap in wraps]
    inter_pairs, self_items = overlap_module._make_pair_lists(infos, "consecutive_self")

    masks = {name: np.zeros(wraps_by_name[name].shape, dtype=bool) for name in PARENT_WRAP_NAMES}
    for left_info, right_info in inter_pairs:
        left = wraps_by_name[left_info.name]
        right = wraps_by_name[right_info.name]

        left_detection = detector.detect_best_inter(left, right)
        if left_detection is not None:
            masks[left.info.name] |= left_detection.band_col_mask[None, :] & left.valid

        right_detection = detector.detect_best_inter(right, left)
        if right_detection is not None:
            masks[right.info.name] |= right_detection.band_col_mask[None, :] & right.valid

    for info in self_items:
        wrap = wraps_by_name[info.name]
        for detection in detector.detect_self(wrap):
            masks[wrap.info.name] |= detection.band_col_mask[None, :] & wrap.valid

    return masks


@lru_cache(maxsize=1)
def load_parent_tifxyz_segments():
    segments = {}
    for parent_name in PARENT_WRAP_NAMES:
        segments[parent_name] = read_tifxyz(PHERC0139_ROOT / parent_name, load_mask=True, validate=True)
    return segments
