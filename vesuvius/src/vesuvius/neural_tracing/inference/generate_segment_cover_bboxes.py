import argparse
import json
from pathlib import Path
import colorsys
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm.auto import tqdm

from vc_reader import tifxyz


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Generate fixed-size world-coordinate bboxes that cover all tifxyz points, "
            "then optimize per-z-band lateral shifts while preserving neighbor face-touch "
            "contact and full point coverage."
        )
    )
    parser.add_argument("--tifxyz-path", type=str, required=True)
    parser.add_argument("--crop-size", type=int, nargs=3, required=True, metavar=("Z", "Y", "X"))
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.0,
        help="Initial chunk overlap fraction in [0, 1). Example: 0.25 means 25%% overlap.",
    )
    parser.add_argument("--napari", action="store_true")
    parser.add_argument("--napari-downsample", type=int, default=8)
    parser.add_argument("--napari-point-size", type=float, default=1.0)
    parser.add_argument(
        "--num-bands",
        type=int,
        default=None,
        help=(
            "When --napari is set, only visualize this many contiguous z-bands, "
            "starting from the lowest occupied z-band."
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--prune-bboxes",
        action="store_true",
        help=(
            "After initial optimization, try removing low-support bboxes per z-band. "
            "A removal is accepted only if re-optimizing remaining bboxes still covers all band points."
        ),
    )
    parser.add_argument(
        "--prune-max-remove-per-band",
        type=int,
        default=None,
        help=(
            "Optional cap on number of bboxes removed per z-band when --prune-bboxes is enabled."
        ),
    )
    parser.add_argument(
        "--band-workers",
        type=int,
        default=1,
        help="Number of worker processes used to optimize z-bands in parallel.",
    )
    args = parser.parse_args(argv)

    crop = np.asarray(args.crop_size, dtype=np.int64)
    if crop.shape != (3,) or np.any(crop <= 0):
        parser.error("--crop-size must be 3 positive integers: Z Y X")
    if not np.isfinite(float(args.overlap)) or float(args.overlap) < 0.0 or float(args.overlap) >= 1.0:
        parser.error("--overlap must be a finite float in [0, 1)")
    if args.napari_downsample < 1:
        parser.error("--napari-downsample must be >= 1")
    if args.napari_point_size <= 0:
        parser.error("--napari-point-size must be > 0")
    if args.num_bands is not None and int(args.num_bands) < 1:
        parser.error("--num-bands must be >= 1 when provided")
    if args.prune_max_remove_per_band is not None and int(args.prune_max_remove_per_band) < 1:
        parser.error("--prune-max-remove-per-band must be >= 1 when provided")
    if int(args.band_workers) < 1:
        parser.error("--band-workers must be >= 1")
    return args


def _load_valid_points_zyx(tifxyz_path):
    tifxyz_path = Path(tifxyz_path)
    if not tifxyz_path.exists():
        raise FileNotFoundError(f"tifxyz path not found: {tifxyz_path}")
    if not tifxyz_path.is_dir():
        raise NotADirectoryError(f"tifxyz path must be a directory: {tifxyz_path}")

    surface = tifxyz.read(tifxyz_path)
    points_zyx = surface.xyz[surface.valid][:, ::-1]
    if points_zyx.size == 0:
        raise RuntimeError("No valid tifxyz points found.")
    return np.asarray(points_zyx, dtype=np.float64)


def _compute_world_bbox(points_zyx):
    mins = np.floor(points_zyx.min(axis=0)).astype(np.int64)
    maxs = np.ceil(points_zyx.max(axis=0)).astype(np.int64)
    return mins, maxs


def _group_point_indices_by_grid(grid_indices):
    if grid_indices.shape[0] == 0:
        return {}

    order = np.lexsort((grid_indices[:, 2], grid_indices[:, 1], grid_indices[:, 0]))
    sorted_idx = grid_indices[order]
    changed = np.any(np.diff(sorted_idx, axis=0) != 0, axis=1)
    starts = np.concatenate(([0], np.where(changed)[0] + 1))
    ends = np.concatenate((starts[1:], [len(order)]))

    grouped = {}
    for s, e in zip(starts, ends):
        key = tuple(int(v) for v in sorted_idx[s])
        grouped[key] = order[s:e]
    return grouped


def _assign_points_to_chunks(points_zyx, world_min_zyx, world_max_zyx, crop_size_zyx):
    crop_size_zyx = np.asarray(crop_size_zyx, dtype=np.int64)
    world_min_zyx = np.asarray(world_min_zyx, dtype=np.int64)
    world_max_zyx = np.asarray(world_max_zyx, dtype=np.int64)
    grid_shape = ((world_max_zyx - world_min_zyx) // crop_size_zyx) + 1

    relative = points_zyx - world_min_zyx[None, :]
    grid_indices = np.floor_divide(relative, crop_size_zyx[None, :]).astype(np.int64)
    grid_indices = np.clip(grid_indices, 0, grid_shape[None, :] - 1)

    grouped = _group_point_indices_by_grid(grid_indices)
    return grouped, grid_shape, crop_size_zyx.copy()


def _compute_stride_from_overlap(crop_size_zyx, overlap_frac):
    crop_size_zyx = np.asarray(crop_size_zyx, dtype=np.int64)
    overlap_frac = float(overlap_frac)
    step_float = crop_size_zyx.astype(np.float64) * (1.0 - overlap_frac)
    step = np.round(step_float).astype(np.int64)
    step = np.maximum(step, 1)
    return step


def _compute_overlapped_grid_shape(world_min_zyx, world_max_zyx, crop_size_zyx, stride_zyx):
    world_min_zyx = np.asarray(world_min_zyx, dtype=np.int64)
    world_max_zyx = np.asarray(world_max_zyx, dtype=np.int64)
    crop_size_zyx = np.asarray(crop_size_zyx, dtype=np.int64)
    stride_zyx = np.asarray(stride_zyx, dtype=np.int64)

    extent = world_max_zyx - world_min_zyx + 1
    shape = np.ones((3,), dtype=np.int64)
    for i in range(3):
        if extent[i] <= crop_size_zyx[i]:
            shape[i] = 1
        else:
            remainder = int(extent[i] - crop_size_zyx[i])
            shape[i] = int((remainder + stride_zyx[i] - 1) // stride_zyx[i]) + 1
    return shape


def _axis_chunk_index_range(point_axis, axis_origin, axis_crop, axis_stride, axis_n):
    p_rel = float(point_axis) - float(axis_origin)
    low = int(np.floor((p_rel - float(axis_crop)) / float(axis_stride))) + 1
    high = int(np.floor(p_rel / float(axis_stride)))
    low = max(0, low)
    high = min(int(axis_n) - 1, high)
    if low > high:
        return None
    return low, high


def _assign_points_to_chunks_overlapped(points_zyx, world_min_zyx, world_max_zyx, crop_size_zyx, overlap_frac):
    crop_size_zyx = np.asarray(crop_size_zyx, dtype=np.int64)
    stride_zyx = _compute_stride_from_overlap(crop_size_zyx, overlap_frac)
    grid_shape = _compute_overlapped_grid_shape(world_min_zyx, world_max_zyx, crop_size_zyx, stride_zyx)

    world_min_zyx = np.asarray(world_min_zyx, dtype=np.float64)
    rel = np.asarray(points_zyx, dtype=np.float64) - world_min_zyx[None, :]  # (N, 3)
    stride_f = stride_zyx.astype(np.float64)
    crop_f = crop_size_zyx.astype(np.float64)
    gs = grid_shape.astype(np.int64)

    # For each point, find the range [low, high] of grid cell indices that cover it.
    # Cell g covers [world_min + g*stride, world_min + g*stride + crop - 1], so
    # point p (rel = p - world_min) is covered by g iff:
    #   floor((rel - crop) / stride) + 1 <= g <= floor(rel / stride)
    low = (np.floor((rel - crop_f[None, :]) / stride_f[None, :]) + 1).astype(np.int64)  # (N, 3)
    high = np.floor(rel / stride_f[None, :]).astype(np.int64)                            # (N, 3)
    low = np.clip(low, 0, gs[None, :] - 1)
    high = np.clip(high, 0, gs[None, :] - 1)

    # Max number of extra cells per axis beyond the first (for any point).
    # Typically 1 for 25% overlap (2 cells per axis -> 8 total offset combos).
    delta = high - low  # (N, 3)
    max_dz = int(np.maximum(0, delta[:, 0]).max()) if len(delta) else 0
    max_dy = int(np.maximum(0, delta[:, 1]).max()) if len(delta) else 0
    max_dx = int(np.maximum(0, delta[:, 2]).max()) if len(delta) else 0

    N = points_zyx.shape[0]
    point_indices = np.arange(N, dtype=np.int64)

    # Outer loop is O(max_cells_per_axis^3), typically 8 iterations.
    # Inner operations are fully vectorized over N points.
    all_assignments = []
    for dz in range(max_dz + 1):
        for dy in range(max_dy + 1):
            for dx in range(max_dx + 1):
                gz = low[:, 0] + dz
                gy = low[:, 1] + dy
                gx = low[:, 2] + dx
                mask = (gz <= high[:, 0]) & (gy <= high[:, 1]) & (gx <= high[:, 2])
                if not np.any(mask):
                    continue
                all_assignments.append(np.stack(
                    [gz[mask], gy[mask], gx[mask], point_indices[mask]], axis=1
                ))

    if not all_assignments:
        return {}, grid_shape, stride_zyx

    all_asgn = np.concatenate(all_assignments, axis=0)  # (M, 4): gz, gy, gx, point_idx

    # Sort by (gz, gy, gx) and extract groups.
    order = np.lexsort((all_asgn[:, 2], all_asgn[:, 1], all_asgn[:, 0]))
    sorted_asgn = all_asgn[order]
    changed = np.any(np.diff(sorted_asgn[:, :3], axis=0) != 0, axis=1)
    starts = np.concatenate(([0], np.where(changed)[0] + 1))
    ends = np.concatenate((starts[1:], [len(order)]))

    grouped = {}
    for s, e in zip(starts, ends):
        key = (int(sorted_asgn[s, 0]), int(sorted_asgn[s, 1]), int(sorted_asgn[s, 2]))
        grouped[key] = sorted_asgn[s:e, 3]

    return grouped, grid_shape, stride_zyx


def _initial_bbox_from_grid_index(world_min_zyx, crop_size_zyx, grid_index_zyx, stride_zyx=None):
    world_min_zyx = np.asarray(world_min_zyx, dtype=np.int64)
    crop_size_zyx = np.asarray(crop_size_zyx, dtype=np.int64)
    grid_index_zyx = np.asarray(grid_index_zyx, dtype=np.int64)
    if stride_zyx is None:
        stride_zyx = crop_size_zyx
    stride_zyx = np.asarray(stride_zyx, dtype=np.int64)

    min_corner = world_min_zyx + grid_index_zyx * stride_zyx
    max_corner = min_corner + crop_size_zyx - 1
    return (
        int(min_corner[0]), int(max_corner[0]),
        int(min_corner[1]), int(max_corner[1]),
        int(min_corner[2]), int(max_corner[2]),
    )


def _bbox_from_state(initial_bbox, axis, shift):
    z_min, z_max, y_min, y_max, x_min, x_max = initial_bbox
    shift = int(shift)
    if axis == "y":
        return (z_min, z_max, y_min + shift, y_max + shift, x_min, x_max)
    if axis == "x":
        return (z_min, z_max, y_min, y_max, x_min + shift, x_max + shift)
    raise ValueError(f"Unknown axis '{axis}'")


def _points_within_bbox(points_zyx, bbox):
    z_min, z_max, y_min, y_max, x_min, x_max = bbox
    z_hi_exclusive = float(z_max) + 1.0
    y_hi_exclusive = float(y_max) + 1.0
    x_hi_exclusive = float(x_max) + 1.0
    return (
        (points_zyx[:, 0] >= float(z_min))
        & (points_zyx[:, 0] < z_hi_exclusive)
        & (points_zyx[:, 1] >= float(y_min))
        & (points_zyx[:, 1] < y_hi_exclusive)
        & (points_zyx[:, 2] >= float(x_min))
        & (points_zyx[:, 2] < x_hi_exclusive)
    )


def _build_band_neighbor_slots(cells):
    # Per record, the indices of its (up, down, left, right) grid neighbors, -1 where there is none.
    by_cell = {cell: i for i, cell in enumerate(cells)}
    neighbor_slots = [(-1, -1, -1, -1)] * len(cells)
    for (gy, gx), idx in by_cell.items():
        neighbor_slots[idx] = (
            by_cell.get((gy - 1, gx), -1),
            by_cell.get((gy + 1, gx), -1),
            by_cell.get((gy, gx - 1), -1),
            by_cell.get((gy, gx + 1), -1),
        )
    return neighbor_slots


def _intervals_touch_or_overlap(min_a, max_a, min_b, max_b):
    return max(float(min_a), float(min_b)) <= (min(float(max_a), float(max_b)) + 1.0)


def _intervals_overlap(min_a, max_a, min_b, max_b):
    return max(float(min_a), float(min_b)) <= min(float(max_a), float(max_b))


def _touching_relation_holds(bbox_a, bbox_b, relation, allow_overlap=False, require_overlap=False):
    if relation == "right":
        if require_overlap:
            return _intervals_overlap(bbox_a[4], bbox_a[5], bbox_b[4], bbox_b[5])
        if allow_overlap:
            return _intervals_touch_or_overlap(bbox_a[4], bbox_a[5], bbox_b[4], bbox_b[5])
        return int(bbox_a[5]) + 1 == int(bbox_b[4])
    if relation == "left":
        if require_overlap:
            return _intervals_overlap(bbox_a[4], bbox_a[5], bbox_b[4], bbox_b[5])
        if allow_overlap:
            return _intervals_touch_or_overlap(bbox_a[4], bbox_a[5], bbox_b[4], bbox_b[5])
        return int(bbox_b[5]) + 1 == int(bbox_a[4])
    if relation == "down":
        if require_overlap:
            return _intervals_overlap(bbox_a[2], bbox_a[3], bbox_b[2], bbox_b[3])
        if allow_overlap:
            return _intervals_touch_or_overlap(bbox_a[2], bbox_a[3], bbox_b[2], bbox_b[3])
        return int(bbox_a[3]) + 1 == int(bbox_b[2])
    if relation == "up":
        if require_overlap:
            return _intervals_overlap(bbox_a[2], bbox_a[3], bbox_b[2], bbox_b[3])
        if allow_overlap:
            return _intervals_touch_or_overlap(bbox_a[2], bbox_a[3], bbox_b[2], bbox_b[3])
        return int(bbox_b[3]) + 1 == int(bbox_a[2])
    raise ValueError(f"Unknown relation '{relation}'")


def _axis_min_and_size(initial_bbox, axis):
    if axis == "y":
        base_min, base_max = int(initial_bbox[2]), int(initial_bbox[3])
    elif axis == "x":
        base_min, base_max = int(initial_bbox[4]), int(initial_bbox[5])
    else:
        raise ValueError(f"Unknown axis '{axis}'")
    size = int(base_max - base_min + 1)
    center = float(base_min) + (float(size) / 2.0)
    return base_min, size, center


def _axis_shift_bounds_from_points(points_zyx, initial_bbox, axis):
    axis_idx = 1 if axis == "y" else 2
    base_min, size, _ = _axis_min_and_size(initial_bbox, axis)
    pts = np.asarray(points_zyx[:, axis_idx], dtype=np.float64)
    min_pts = float(np.min(pts))
    max_pts = float(np.max(pts))

    # Half-open containment along this axis:
    #   base_min + shift <= p < base_min + shift + size
    # gives:
    #   shift <= p_min - base_min
    #   shift > p_max - (base_min + size)
    # lower integer bound is floor(rhs) + 1.
    low = int(np.floor(max_pts - float(base_min + size))) + 1
    high = int(np.floor(min_pts - float(base_min)))
    return low, high


def _axis_objective(points_zyx, bbox, axis):
    axis_idx = 1 if axis == "y" else 2
    if axis == "y":
        center = float(bbox[2]) + (float(bbox[3] - bbox[2] + 1) / 2.0)
    elif axis == "x":
        center = float(bbox[4]) + (float(bbox[5] - bbox[4] + 1) / 2.0)
    else:
        raise ValueError(f"Unknown axis '{axis}'")
    return float(np.mean(np.abs(points_zyx[:, axis_idx] - center)))


def _generate_shift_candidates(low, high, ideal_shift, current_shift):
    if low > high:
        return []

    def _clamp(v):
        return int(min(high, max(low, int(v))))

    vals = {
        int(low),
        int(high),
        _clamp(int(round(ideal_shift))),
        _clamp(int(current_shift)),
        _clamp((int(low) + int(high)) // 2),
    }
    base_ideal = int(round(ideal_shift))
    base_current = int(current_shift)
    for base in (base_ideal, base_current):
        for d in (-3, -2, -1, 1, 2, 3):
            vals.add(_clamp(base + d))

    return sorted(vals)


def _is_candidate_better(obj, axis, shift, best_obj, best_axis, best_shift):
    eps = 1e-12
    if obj < best_obj - eps:
        return True
    if abs(obj - best_obj) > eps:
        return False

    cand_abs = abs(int(shift))
    best_abs = abs(int(best_shift))
    if cand_abs != best_abs:
        return cand_abs < best_abs
    if axis != best_axis:
        return axis == "y"
    return int(shift) < int(best_shift)


# Pruning re-optimizes a whole z-band for every removal attempt, and almost all of
# that work repeats: the same record, in the same state, beside the same neighbor
# bboxes. The helpers below memoize those pure sub-results for one band at a time.
#
# Every table is least-recently-used with a hard cap fixed when the band starts: a
# few entries per record of the band, or per band point for the tables that hold
# point indices. A band's caches therefore stay proportional to the records and
# points the band already holds, however many removal attempts it needs, and they
# are dropped together with the band's _BandCache.
_STEP_MEMO_ENTRIES_PER_RECORD = 4
_COVER_CACHE_ENTRIES_PER_RECORD = 2
_COVER_CACHE_ELEMENTS_PER_POINT = 6
_VARIANT_CACHE_ENTRIES_PER_RECORD = 1
_VARIANT_CACHE_ELEMENTS_PER_POINT = 1
_OBJECTIVE_MEMO_MAX_ENTRIES = 256
_MIN_CACHE_ENTRIES = 64
_NEAREST_CENTER_CHUNK_ELEMENTS = 1 << 15


class _LruCache:
    """Least-recently-used table with hard caps on its entry count and on the summed weight of its entries."""

    __slots__ = ("_entries", "_max_entries", "_max_weight", "_weigh", "_weight")

    def __init__(self, max_entries, max_weight, weigh):
        self._entries = OrderedDict()
        self._max_entries = int(max_entries)
        self._max_weight = int(max_weight)
        self._weigh = weigh
        self._weight = 0

    def __len__(self):
        return len(self._entries)

    @property
    def weight(self):
        return self._weight

    def get(self, key):
        value = self._entries.get(key)
        if value is not None:
            self._entries.move_to_end(key)
        return value

    def put(self, key, value):
        """Store a key that is not in the table yet. An entry heavier than the weight cap is not stored."""
        weight = self._weigh(key, value)
        if weight > self._max_weight:
            return
        self._entries[key] = value
        self._weight += weight
        while len(self._entries) > self._max_entries or self._weight > self._max_weight:
            self._weight -= self._weigh(*self._entries.popitem(last=False))


class _BandItem:
    """One record of a band, with the values that depend only on its points and initial bbox."""

    __slots__ = (
        "serial",
        "record",
        "cell",
        "initial_bbox",
        "points",
        "extra_idx",
        "base_points",
        "_axis_stats",
        "_limits",
        "_initial",
    )

    def __init__(self, serial, record, points, extra_idx=None):
        self.serial = serial
        self.record = record
        self.cell = (int(record["grid_index"][1]), int(record["grid_index"][2]))
        self.initial_bbox = record["initial_bbox"]
        # ``points`` is None for a prune-trial variant until something needs the array:
        # it is then rebuilt as base_points followed by the band points ``extra_idx``.
        self.points = points
        self.extra_idx = extra_idx
        self.base_points = None
        self._axis_stats = [None, None]
        self._limits = None
        self._initial = None

    def get_points(self, band):
        points = self.points
        if points is None:
            points = np.asarray(self.base_points, dtype=np.float64)
            if self.extra_idx.shape[0] > 0:
                points = np.concatenate([points, band.points[self.extra_idx]], axis=0)
            self.points = points
        return points

    def shift_range(self, band, axis):
        """(low, high, ideal) shift along ``axis``; ideal is None when no shift fits (low > high)."""
        slot = 0 if axis == "y" else 1
        stats = self._axis_stats[slot]
        if stats is None:
            points = self.get_points(band)
            low, high = _axis_shift_bounds_from_points(points, self.initial_bbox, axis)
            ideal = None
            if low <= high:
                _, _, base_center = _axis_min_and_size(self.initial_bbox, axis)
                axis_idx = 1 if axis == "y" else 2
                ideal = float(np.median(points[:, axis_idx])) - base_center
            stats = self._axis_stats[slot] = (low, high, ideal)
        return stats

    def contains(self, band, bbox):
        # Same half-open test as np.all(_points_within_bbox(points, bbox)), on the
        # per-axis extremes. np.min/np.max return NaN when any coordinate is NaN, and
        # every comparison with NaN is False, exactly as in the per-point test.
        limits = self._limits
        if limits is None:
            points = self.get_points(band)
            limits = self._limits = tuple(
                float(reduce(points[:, axis_idx])) for axis_idx in range(3) for reduce in (np.min, np.max)
            )
        return (
            limits[0] >= float(bbox[0])
            and limits[1] < float(bbox[1]) + 1.0
            and limits[2] >= float(bbox[2])
            and limits[3] < float(bbox[3]) + 1.0
            and limits[4] >= float(bbox[4])
            and limits[5] < float(bbox[5]) + 1.0
        )

    def objective(self, band, axis, shift, bbox):
        key = (self.serial, axis, shift)
        memo = band.objectives
        value = memo.get(key)
        if value is None:
            value = _axis_objective(self.get_points(band), bbox, axis)
            memo[key] = value
            if len(memo) > _OBJECTIVE_MEMO_MAX_ENTRIES:
                memo.popitem(last=False)
        else:
            memo.move_to_end(key)
        return value

    def initial(self, band):
        """The optimizer's start: no shift, on the axis with the smaller objective.

        Returns (state, bbox, y_span, x_span), where a state is (axis, shift, objective).
        """
        initial = self._initial
        if initial is None:
            obj_y = self.objective(band, "y", 0, self.initial_bbox)
            obj_x = self.objective(band, "x", 0, self.initial_bbox)
            state = ("y", 0, obj_y) if obj_y <= obj_x else ("x", 0, obj_x)
            bbox = _bbox_from_state(self.initial_bbox, state[0], 0)
            initial = self._initial = (state, bbox, bbox[2:4], bbox[4:6])
        return initial


def _cover_weight(bbox, index):
    return index.shape[0]


def _variant_weight(key, item):
    return item.extra_idx.shape[0]


class _BandCache:
    """Bounded memo tables for one z-band. Create one per band and let it go out of scope with the band."""

    __slots__ = (
        "points",
        "steps",
        "max_steps",
        "objectives",
        "_cover",
        "_variants",
        "_trial_variants",
        "_serial",
        "_index_dtype",
        "_delta",
    )

    def __init__(self, points_zyx_band, record_count):
        self.points = np.asarray(points_zyx_band, dtype=np.float64)
        records = max(1, int(record_count))
        points = max(1, int(self.points.shape[0]))
        # Step results, keyed by everything a step reads (see _solve_band_states).
        self.steps = OrderedDict()
        self.max_steps = max(_MIN_CACHE_ENTRIES, _STEP_MEMO_ENTRIES_PER_RECORD * records)
        # (item serial, axis, shift) -> objective.
        self.objectives = OrderedDict()
        # bbox -> indices of the band points inside it, weighed by the number of indices.
        self._cover = _LruCache(
            max(_MIN_CACHE_ENTRIES, _COVER_CACHE_ENTRIES_PER_RECORD * records),
            _COVER_CACHE_ELEMENTS_PER_POINT * points,
            _cover_weight,
        )
        # (base item serial, bytes of the extra point indices) -> variant item, weighed by the extra points.
        self._variants = _LruCache(
            max(_MIN_CACHE_ENTRIES, _VARIANT_CACHE_ENTRIES_PER_RECORD * records),
            _VARIANT_CACHE_ELEMENTS_PER_POINT * points,
            _variant_weight,
        )
        self._trial_variants = []
        self._serial = 0
        # Smallest unsigned type that can index the band, so cached index lists stay small.
        self._index_dtype = np.min_scalar_type(max(int(self.points.shape[0]) - 1, 0))
        self._delta = None

    def item(self, record, points):
        self._serial += 1
        return _BandItem(self._serial, record, points)

    def variant(self, base, extra_idx):
        """``base`` with the band points ``extra_idx`` appended, as a prune trial augments it."""
        key = (base.serial, extra_idx.tobytes())
        item = self._variants.get(key)
        if item is None:
            self._serial += 1
            item = _BandItem(self._serial, base.record, None, extra_idx)
            self._variants.put(key, item)
        item.base_points = base.points
        self._trial_variants.append(item)
        return item

    def end_trial(self):
        """Drop the point arrays this trial's variants built. An accepted variant's array lives on in its record."""
        for item in self._trial_variants:
            item.points = None
            item.base_points = None
        self._trial_variants.clear()

    def cover_index(self, bbox):
        """Indices of the band points inside ``bbox``."""
        index = self._cover.get(bbox)
        if index is None:
            index = np.flatnonzero(_points_within_bbox(self.points, bbox)).astype(self._index_dtype, copy=False)
            self._cover.put(bbox, index)
        return index

    def cover_count(self, bboxes):
        """How many of ``bboxes`` contain each band point."""
        count = np.zeros((self.points.shape[0],), dtype=np.int32)
        for bbox in bboxes:
            count[self.cover_index(bbox)] += 1
        return count

    def never_covered(self, count):
        """Band points that ``count`` leaves without any bbox, in ascending order."""
        return np.flatnonzero(count == 0).astype(self._index_dtype, copy=False)

    def uncovered_without(self, count, never_covered, removed_cover):
        """Band points, in ascending order, that no bbox covers once the bbox behind ``removed_cover`` is gone."""
        uncovered = removed_cover[count[removed_cover] == 1]
        if never_covered.shape[0] > 0:
            uncovered = np.concatenate([never_covered, uncovered])
        return np.sort(uncovered)

    def still_covered(self, count, never_covered, removed_cover, old_bboxes, new_bboxes):
        """Whether every band point is covered once ``removed_cover`` is gone and old_bboxes become new_bboxes.

        ``count`` is cover_count() of the bboxes before the change and ``never_covered``
        the points it leaves at zero. Only those points, and points that lose a bbox,
        can end up uncovered, so only they are checked.
        """
        if self._delta is None:
            self._delta = np.zeros((self.points.shape[0],), dtype=np.int32)
        delta = self._delta
        lost = [removed_cover]
        gained = []
        for old_bbox, new_bbox in zip(old_bboxes, new_bboxes):
            if new_bbox != old_bbox:
                lost.append(self.cover_index(old_bbox))
                gained.append(self.cover_index(new_bbox))
        for index in gained:
            delta[index] += 1
        for index in lost:
            delta[index] -= 1
        covered = all(bool(np.all(count[index] + delta[index] > 0)) for index in lost) and bool(
            np.all(delta[never_covered] > 0)
        )
        for index in gained:
            delta[index] = 0
        for index in lost:
            delta[index] = 0
        return covered


def _item_cell(item):
    return item.cell


def _best_item_state(band, item, current, neighbor_bboxes, allow_overlap, require_overlap):
    """One optimizer step for one record: its best feasible (axis, shift, objective) beside the given neighbors."""
    init_bbox = item.initial_bbox
    best_axis, best_shift, best_obj = current

    for axis in ("y", "x"):
        low, high, ideal_shift = item.shift_range(band, axis)
        if low > high:
            continue

        cur_shift_for_axis = current[1] if axis == current[0] else 0
        candidates = _generate_shift_candidates(low, high, ideal_shift, cur_shift_for_axis)

        for shift in candidates:
            cand_bbox = _bbox_from_state(init_bbox, axis, shift)
            if not item.contains(band, cand_bbox):
                continue

            ok_touch = True
            for nbr_bbox, relation in neighbor_bboxes:
                if not _touching_relation_holds(
                    cand_bbox,
                    nbr_bbox,
                    relation,
                    allow_overlap=allow_overlap,
                    require_overlap=require_overlap,
                ):
                    ok_touch = False
                    break
            if not ok_touch:
                continue

            cand_obj = item.objective(band, axis, shift, cand_bbox)
            if _is_candidate_better(cand_obj, axis, shift, best_obj, best_axis, best_shift):
                best_obj = cand_obj
                best_axis = axis
                best_shift = int(shift)

    if best_axis != current[0] or best_shift != current[1]:
        return (best_axis, best_shift, float(best_obj))
    return current


_NEIGHBOR_RELATIONS = ("up", "down", "left", "right")


def _solve_band_states(band, items, neighbor_slots, max_passes, allow_overlap, require_overlap):
    """Run the optimizer passes over ``items`` (sorted by cell); return their final states and bboxes."""
    allow_overlap = bool(allow_overlap)
    require_overlap = bool(require_overlap)
    # _touching_relation_holds tests require_overlap first, then allow_overlap.
    contact_mode = 2 if require_overlap else (1 if allow_overlap else 0)
    steps = band.steps
    max_steps = band.max_steps

    states = []
    bboxes = []
    y_spans = []
    x_spans = []
    for item in items:
        state, bbox, y_span, x_span = item.initial(band)
        states.append(state)
        bboxes.append(bbox)
        y_spans.append(y_span)
        x_spans.append(x_span)
    # Slot -1 stands for "no neighbor on that side".
    y_spans.append(None)
    x_spans.append(None)

    for _ in range(int(max_passes)):
        changed = False
        for idx, item in enumerate(items):
            current = states[idx]
            up, down, left, right = neighbor_slots[idx]
            # A step reads the record, its state, the contact mode, the y-span of the
            # up/down neighbors and the x-span of the left/right neighbors. Nothing else.
            key = (
                item.serial,
                contact_mode,
                current[0],
                current[1],
                y_spans[up],
                y_spans[down],
                x_spans[left],
                x_spans[right],
            )
            best = steps.get(key)
            if best is None:
                best = _best_item_state(
                    band,
                    item,
                    current,
                    [
                        (bboxes[nbr_idx], relation)
                        for nbr_idx, relation in zip(neighbor_slots[idx], _NEIGHBOR_RELATIONS)
                        if nbr_idx >= 0
                    ],
                    allow_overlap,
                    require_overlap,
                )
                steps[key] = best
                if len(steps) > max_steps:
                    steps.popitem(last=False)
            else:
                steps.move_to_end(key)

            if best[0] != current[0] or best[1] != current[1]:
                states[idx] = best
                bbox = bboxes[idx] = _bbox_from_state(item.initial_bbox, best[0], best[1])
                y_spans[idx] = bbox[2:4]
                x_spans[idx] = bbox[4:6]
                changed = True

        if not changed:
            break

    return states, bboxes


def _emit_band_records(band, items, states, bboxes):
    out = []
    for item, state, bbox in zip(items, states, bboxes):
        out_item = dict(item.record)
        out_item["points"] = item.get_points(band)
        out_item["shift_axis"] = state[0]
        out_item["shift_voxels"] = int(state[1])
        out_item["bbox"] = bbox
        out.append(out_item)
    return out


def _optimize_band_items(band, items, max_passes, allow_neighbor_overlap, require_neighbor_overlap):
    if not items:
        return [], []
    items = sorted(items, key=_item_cell)
    neighbor_slots = _build_band_neighbor_slots([item.cell for item in items])
    states, bboxes = _solve_band_states(
        band, items, neighbor_slots, max_passes, allow_neighbor_overlap, require_neighbor_overlap
    )
    if not np.all(band.cover_count(bboxes)):
        raise RuntimeError("Band optimization violated full point coverage.")
    return _emit_band_records(band, items, states, bboxes), items


def _optimize_band_bboxes(
    records,
    points_zyx_band,
    max_passes=10,
    allow_neighbor_overlap=False,
    require_neighbor_overlap=False,
):
    if not records:
        return records

    band = _BandCache(points_zyx_band, len(records))
    optimized, _ = _optimize_band_items(
        band,
        [band.item(rec, rec["points"]) for rec in records],
        max_passes,
        allow_neighbor_overlap,
        require_neighbor_overlap,
    )
    return optimized


def _nearest_center_indices(centers_y, centers_x, points_zyx):
    """For each point, the index of the center nearest in |dy| + |dx|; ties go to the first center."""
    nearest = np.empty((points_zyx.shape[0],), dtype=np.intp)
    rows = max(1, _NEAREST_CENTER_CHUNK_ELEMENTS // max(1, centers_y.shape[0]))
    for start in range(0, points_zyx.shape[0], rows):
        chunk = points_zyx[start:start + rows]
        dist = np.abs(centers_y[None, :] - chunk[:, 1:2]) + np.abs(centers_x[None, :] - chunk[:, 2:3])
        nearest[start:start + rows] = np.argmin(dist, axis=1)
    return nearest


def _is_float64_array(points):
    return isinstance(points, np.ndarray) and points.dtype == np.float64


_NO_EXTRA_POINTS = np.zeros((0,), dtype=np.intp)


def _prune_band_items(
    band,
    kept,
    kept_items,
    max_passes,
    max_remove,
    allow_neighbor_overlap,
    require_neighbor_overlap,
):
    initial = len(kept)
    if not kept:
        return kept, {"initial": 0, "removed": 0, "final": 0, "attempts": 0}

    if bool(require_neighbor_overlap):
        contact_modes = ((True, True),)
    elif bool(allow_neighbor_overlap):
        contact_modes = ((True, False), (False, False))
    else:
        contact_modes = ((False, False), (True, False))

    removed = 0
    attempts = 0

    while len(kept) > 1:
        if max_remove is not None and removed >= int(max_remove):
            break

        for item, rec in zip(kept_items, kept):
            item.record = rec
            item.points = rec["points"]

        # Try smallest support boxes first. Ties resolve deterministically by grid index.
        candidate_order = sorted(
            range(len(kept)),
            key=lambda i: (
                int(kept[i]["points"].shape[0]),
                int(kept[i]["grid_index"][1]),
                int(kept[i]["grid_index"][2]),
            ),
        )

        # Prefer current optimized bbox if available; otherwise use initial bbox.
        bboxes = [tuple(int(v) for v in (rec["bbox"] if "bbox" in rec else rec["initial_bbox"])) for rec in kept]
        count = band.cover_count(bboxes)
        never_covered = band.never_covered(count)
        centers_y = np.asarray([0.5 * (float(bbox[2]) + float(bbox[3])) for bbox in bboxes], dtype=np.float64)
        centers_x = np.asarray([0.5 * (float(bbox[4]) + float(bbox[5])) for bbox in bboxes], dtype=np.float64)
        needs_float64 = [i for i, item in enumerate(kept_items) if not _is_float64_array(item.points)]

        # The optimizer works on records sorted by cell. Sort once per round; a trial
        # is this order with one record left out.
        order = sorted(range(len(kept)), key=lambda i: kept_items[i].cell)
        rank = [0] * len(order)
        for position, i in enumerate(order):
            rank[i] = position
        sorted_items = [kept_items[i] for i in order]
        sorted_bboxes = [bboxes[i] for i in order]

        removed_one = False
        for idx in candidate_order:
            attempts += 1
            removed_rank = rank[idx]
            trial_items = sorted_items[:removed_rank] + sorted_items[removed_rank + 1:]
            trial_bboxes = sorted_bboxes[:removed_rank] + sorted_bboxes[removed_rank + 1:]

            # Points only the removed bbox covered go to the remaining record whose
            # bbox center is nearest, in ascending point order.
            removed_cover = band.cover_index(bboxes[idx])
            uncovered = band.uncovered_without(count, never_covered, removed_cover)
            if uncovered.shape[0] > 0:
                nearest = _nearest_center_indices(
                    np.concatenate([centers_y[:idx], centers_y[idx + 1:]]),
                    np.concatenate([centers_x[:idx], centers_x[idx + 1:]]),
                    band.points[uncovered],
                )
                extra = {i: _NO_EXTRA_POINTS for i in needs_float64 if i != idx}
                for target in np.unique(nearest).tolist():
                    extra[target if target < idx else target + 1] = uncovered[nearest == target]
                for i, extra_idx in extra.items():
                    position = rank[i] - (1 if rank[i] > removed_rank else 0)
                    trial_items[position] = band.variant(kept_items[i], extra_idx)

            neighbor_slots = _build_band_neighbor_slots([item.cell for item in trial_items])
            trial_states = None
            # Try contact modes in order; accept any full-coverage solution.
            for allow_overlap, require_overlap in contact_modes:
                states, new_bboxes = _solve_band_states(
                    band, trial_items, neighbor_slots, max_passes, allow_overlap, require_overlap
                )
                if band.still_covered(count, never_covered, removed_cover, trial_bboxes, new_bboxes):
                    trial_states = states
                    break
            if trial_states is None:
                band.end_trial()
                continue

            kept = _emit_band_records(band, trial_items, trial_states, new_bboxes)
            band.end_trial()
            kept_items = trial_items
            removed += 1
            removed_one = True
            break

        if not removed_one:
            break

    return kept, {
        "initial": int(initial),
        "removed": int(removed),
        "final": int(len(kept)),
        "attempts": int(attempts),
    }


def _prune_band_bboxes_via_reopt(
    optimized_records,
    points_zyx_band,
    max_passes=5,
    max_remove=None,
    allow_neighbor_overlap=False,
    require_neighbor_overlap=False,
):
    if not optimized_records:
        return optimized_records, {"initial": 0, "removed": 0, "final": 0, "attempts": 0}

    band = _BandCache(points_zyx_band, len(optimized_records))
    kept = [dict(item) for item in optimized_records]
    return _prune_band_items(
        band,
        kept,
        [band.item(rec, rec["points"]) for rec in kept],
        max_passes,
        max_remove,
        allow_neighbor_overlap,
        require_neighbor_overlap,
    )


def _bbox_wireframe_segments(bbox):
    z_min, z_max, y_min, y_max, x_min, x_max = bbox
    corners = np.asarray(
        [
            [z_min, y_min, x_min],
            [z_min, y_min, x_max],
            [z_min, y_max, x_min],
            [z_min, y_max, x_max],
            [z_max, y_min, x_min],
            [z_max, y_min, x_max],
            [z_max, y_max, x_min],
            [z_max, y_max, x_max],
        ],
        dtype=np.float32,
    )
    edge_pairs = (
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7),
        (6, 7),
    )
    return [corners[[i0, i1]] for (i0, i1) in edge_pairs]


def _select_napari_bands(bbox_records, num_bands=None):
    bands = sorted({int(item["z_band"]) for item in bbox_records})
    if num_bands is None:
        return bands
    limit = int(num_bands)
    if limit <= 0:
        return bands
    start = bands[0] if bands else 0
    selected = set(range(int(start), int(start) + limit))
    return [band for band in bands if band in selected]


def _show_napari(points_zyx, bbox_records, downsample=8, point_size=1.0, num_bands=None):
    try:
        import napari
    except Exception as exc:
        raise RuntimeError("--napari was set, but napari is not available.") from exc

    viewer = napari.Viewer(ndisplay=3)
    downsample = max(1, int(downsample))
    point_size = float(point_size)

    pts = np.asarray(points_zyx, dtype=np.float32)
    if downsample > 1 and pts.shape[0] > 0:
        pts = pts[::downsample]
    if pts.shape[0] > 0:
        viewer.add_points(pts, name="segment_points", size=point_size, face_color=[0.0, 1.0, 0.0])

    bands = _select_napari_bands(bbox_records, num_bands=num_bands)
    selected_band_set = set(bands)
    bbox_records = [item for item in bbox_records if int(item["z_band"]) in selected_band_set]

    if len(bbox_records) == 0:
        print("No bboxes in selected --num-bands range for Napari visualization.")
        napari.run()
        return

    band_to_rgb = {}
    n_band = max(1, len(bands))
    for i, band in enumerate(bands):
        band_to_rgb[band] = colorsys.hsv_to_rgb((i / n_band) % 1.0, 1.0, 1.0)

    # Batch wireframes by z-band so Napari only creates a small number of layers.
    # Creating one Shapes layer per bbox can make window startup very slow.
    segments_by_band = {int(band): [] for band in bands}
    for item in bbox_records:
        z_band = int(item["z_band"])
        segments_by_band[z_band].extend(_bbox_wireframe_segments(item["bbox"]))

    for z_band in bands:
        segments = segments_by_band[z_band]
        if not segments:
            continue
        rgb = band_to_rgb[z_band]
        viewer.add_shapes(
            segments,
            shape_type="path",
            edge_color=[*rgb, 0.9],
            edge_width=1,
            face_color="transparent",
            name=f"bboxes_band_{z_band:03d}",
            opacity=0.9,
        )

    napari.run()


def _serialize_bbox_record(item):
    return {
        "bbox_id": int(item["bbox_id"]),
        "z_band": int(item["z_band"]),
        "grid_index": [int(v) for v in item["grid_index"]],
        "shift_axis": str(item["shift_axis"]),
        "shift_voxels": int(item["shift_voxels"]),
        "bbox": [int(v) for v in item["bbox"]],
    }


def _build_band_records(points_zyx, grouped_chunk_indices, world_min_zyx, crop_size_zyx, stride_zyx):
    by_band = {}
    for grid_idx, point_indices in grouped_chunk_indices.items():
        gz, gy, gx = (int(grid_idx[0]), int(grid_idx[1]), int(grid_idx[2]))
        rec = {
            "grid_index": (gz, gy, gx),
            "initial_bbox": _initial_bbox_from_grid_index(world_min_zyx, crop_size_zyx, (gz, gy, gx), stride_zyx=stride_zyx),
            "point_indices": np.asarray(point_indices, dtype=np.int64),
            "points": np.asarray(points_zyx[point_indices], dtype=np.float64),
        }
        by_band.setdefault(gz, []).append(rec)
    return by_band


def _write_output_json(output_json_path, payload):
    output_json_path = Path(output_json_path)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with output_json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _validate_generation_inputs(crop_size, overlap, prune_max_remove_per_band, band_workers):
    crop_size_zyx = np.asarray(crop_size, dtype=np.int64)
    if crop_size_zyx.shape != (3,) or np.any(crop_size_zyx <= 0):
        raise ValueError("crop_size must be 3 positive integers: Z Y X")

    overlap = float(overlap)
    if not np.isfinite(overlap) or overlap < 0.0 or overlap >= 1.0:
        raise ValueError("overlap must be a finite float in [0, 1)")

    if prune_max_remove_per_band is not None and int(prune_max_remove_per_band) < 1:
        raise ValueError("prune_max_remove_per_band must be >= 1 when provided")
    band_workers = int(band_workers)
    if band_workers < 1:
        raise ValueError("band_workers must be >= 1")
    return crop_size_zyx, overlap, band_workers


def _optimize_single_band(
    z_band,
    band_records,
    band_points,
    prune_bboxes,
    prune_max_remove_per_band,
    overlap,
):
    overlap_positive = bool(float(overlap) > 0.0)
    # One cache per band: pruning reuses what the first optimization computed, and
    # all of it is released when this call returns.
    band = _BandCache(band_points, len(band_records))
    optimized, items = _optimize_band_items(
        band,
        [band.item(rec, rec["points"]) for rec in band_records],
        max_passes=5,
        allow_neighbor_overlap=overlap_positive,
        require_neighbor_overlap=overlap_positive,
    )
    prune_stats = None
    if prune_bboxes:
        optimized, prune_stats = _prune_band_items(
            band,
            optimized,
            items,
            max_passes=5,
            max_remove=prune_max_remove_per_band,
            allow_neighbor_overlap=overlap_positive,
            require_neighbor_overlap=overlap_positive,
        )
    optimized_sorted = sorted(optimized, key=lambda r: (int(r["grid_index"][1]), int(r["grid_index"][2])))
    return int(z_band), optimized_sorted, prune_stats


def _generate_segment_cover_records(
    points_zyx,
    crop_size_zyx,
    overlap=0.0,
    prune_bboxes=False,
    prune_max_remove_per_band=None,
    band_workers=1,
    show_progress=False,
    progress_desc=None,
):
    points_zyx = np.asarray(points_zyx, dtype=np.float64)
    crop_size_zyx, overlap, band_workers = _validate_generation_inputs(
        crop_size_zyx,
        overlap,
        prune_max_remove_per_band,
        band_workers,
    )

    world_min_zyx, world_max_zyx = _compute_world_bbox(points_zyx)
    if overlap <= 0.0:
        grouped, grid_shape, stride_zyx = _assign_points_to_chunks(points_zyx, world_min_zyx, world_max_zyx, crop_size_zyx)
    else:
        grouped, grid_shape, stride_zyx = _assign_points_to_chunks_overlapped(
            points_zyx,
            world_min_zyx,
            world_max_zyx,
            crop_size_zyx,
            overlap_frac=overlap,
        )

    by_band = _build_band_records(points_zyx, grouped, world_min_zyx, crop_size_zyx, stride_zyx=stride_zyx)

    def band_jobs():
        # One band at a time: a band's point array is needed only while that band is
        # optimized or waits for a worker, so the arrays of all bands are never held at once.
        for z_band in sorted(by_band.keys()):
            band_records = by_band[z_band]
            band_point_indices = np.unique(np.concatenate([item["point_indices"] for item in band_records]))
            yield int(z_band), band_records, points_zyx[band_point_indices]

    band_count = len(by_band)
    progress_enabled = bool(show_progress) and band_count > 0
    progress_desc = str(progress_desc or "Optimizing z-bands")

    if band_workers == 1 or band_count <= 1:
        band_iter = band_jobs()
        if progress_enabled:
            band_iter = tqdm(
                band_iter,
                total=band_count,
                desc=progress_desc,
                unit="band",
                leave=False,
            )
        band_results = []
        for z_band, band_records, band_points in band_iter:
            band_results.append(
                _optimize_single_band(
                    z_band,
                    band_records,
                    band_points,
                    prune_bboxes,
                    prune_max_remove_per_band,
                    overlap,
                )
            )
    else:
        with ProcessPoolExecutor(max_workers=band_workers) as executor:
            futures = [
                executor.submit(
                    _optimize_single_band,
                    z_band,
                    band_records,
                    band_points,
                    prune_bboxes,
                    prune_max_remove_per_band,
                    overlap,
                )
                for z_band, band_records, band_points in band_jobs()
            ]
            if progress_enabled:
                band_results = []
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=progress_desc,
                    unit="band",
                    leave=False,
                ):
                    band_results.append(future.result())
            else:
                band_results = [future.result() for future in futures]
    band_results = sorted(band_results, key=lambda item: int(item[0]))

    final_records = []
    bbox_id = 0
    prune_stats_by_band = {}
    total_prune_removed = 0
    total_prune_attempts = 0
    for z_band, optimized, prune_stats in band_results:
        if prune_stats is not None:
            prune_stats_by_band[int(z_band)] = prune_stats
            total_prune_removed += int(prune_stats["removed"])
            total_prune_attempts += int(prune_stats["attempts"])
        for item in optimized:
            out_item = dict(item)
            out_item["bbox_id"] = int(bbox_id)
            out_item["z_band"] = int(z_band)
            final_records.append(out_item)
            bbox_id += 1

    initial_chunk_count = int(np.prod(grid_shape, dtype=np.int64))
    occupied_count = int(len(grouped))
    final_bbox_count = int(len(final_records))
    return {
        "points_zyx": points_zyx,
        "crop_size_zyx": crop_size_zyx,
        "overlap": overlap,
        "world_min_zyx": world_min_zyx,
        "world_max_zyx": world_max_zyx,
        "grouped": grouped,
        "grid_shape": grid_shape,
        "stride_zyx": stride_zyx,
        "by_band": by_band,
        "final_records": final_records,
        "initial_chunk_count": initial_chunk_count,
        "occupied_chunk_count": occupied_count,
        "final_bbox_count": final_bbox_count,
        "prune_stats_by_band": prune_stats_by_band,
        "prune_removed_total": int(total_prune_removed),
        "prune_attempts_total": int(total_prune_attempts),
        "band_workers": int(band_workers),
    }


def generate_segment_cover_bboxes_list(
    tifxyz_path,
    crop_size,
    overlap=0.0,
    prune_bboxes=False,
    prune_max_remove_per_band=None,
    band_workers=1,
):
    points_zyx = _load_valid_points_zyx(tifxyz_path)
    result = _generate_segment_cover_records(
        points_zyx,
        crop_size,
        overlap=overlap,
        prune_bboxes=prune_bboxes,
        prune_max_remove_per_band=prune_max_remove_per_band,
        band_workers=band_workers,
    )
    tifxyz_uuid = Path(tifxyz_path).resolve().name
    return [
        {
            "tifxyz_uuid": tifxyz_uuid,
            **_serialize_bbox_record(item),
        }
        for item in result["final_records"]
    ]


def main(argv=None):
    args = parse_args(argv)
    points_zyx = _load_valid_points_zyx(args.tifxyz_path)
    result = _generate_segment_cover_records(
        points_zyx,
        args.crop_size,
        overlap=args.overlap,
        prune_bboxes=args.prune_bboxes,
        prune_max_remove_per_band=args.prune_max_remove_per_band,
        band_workers=args.band_workers,
    )
    crop_size_zyx = result["crop_size_zyx"]
    world_min_zyx = result["world_min_zyx"]
    world_max_zyx = result["world_max_zyx"]
    stride_zyx = result["stride_zyx"]
    by_band = result["by_band"]
    final_records = result["final_records"]
    initial_chunk_count = result["initial_chunk_count"]
    occupied_count = result["occupied_chunk_count"]
    final_bbox_count = result["final_bbox_count"]
    prune_stats_by_band = result["prune_stats_by_band"]
    total_prune_removed = result["prune_removed_total"]
    total_prune_attempts = result["prune_attempts_total"]

    payload = {
        "meta": {
            "tifxyz_path": str(Path(args.tifxyz_path).resolve()),
            "crop_size_zyx": [int(v) for v in crop_size_zyx],
            "overlap": float(args.overlap),
            "chunk_stride_zyx": [int(v) for v in stride_zyx],
            "prune_bboxes": bool(args.prune_bboxes),
            "prune_max_remove_per_band": (
                None if args.prune_max_remove_per_band is None else int(args.prune_max_remove_per_band)
            ),
            "prune_removed_total": int(total_prune_removed),
            "prune_attempts_total": int(total_prune_attempts),
            "band_workers": int(result["band_workers"]),
            "input_valid_points": int(points_zyx.shape[0]),
            "initial_chunk_count": initial_chunk_count,
            "occupied_chunk_count": occupied_count,
            "z_band_count": int(len(by_band)),
            "final_bbox_count": final_bbox_count,
        },
        "world_bbox": {
            "min_zyx": [int(v) for v in world_min_zyx],
            "max_zyx": [int(v) for v in world_max_zyx],
        },
        "bboxes": [_serialize_bbox_record(item) for item in final_records],
        "prune_stats_by_band": prune_stats_by_band,
    }
    _write_output_json(args.output_json, payload)

    print(f"input_valid_points: {int(points_zyx.shape[0])}")
    print(f"world_bbox_min_zyx: {[int(v) for v in world_min_zyx]}")
    print(f"world_bbox_max_zyx: {[int(v) for v in world_max_zyx]}")
    print(f"overlap: {float(args.overlap)}")
    print(f"band_workers: {int(result['band_workers'])}")
    print(f"chunk_stride_zyx: {[int(v) for v in stride_zyx]}")
    print(f"initial_chunk_count: {initial_chunk_count}")
    print(f"occupied_chunk_count: {occupied_count}")
    print(f"z_band_count: {int(len(by_band))}")
    print(f"final_bbox_count: {final_bbox_count}")
    if args.prune_bboxes:
        print(f"prune_removed_total: {int(total_prune_removed)}")
        print(f"prune_attempts_total: {int(total_prune_attempts)}")
    print(f"output_json: {Path(args.output_json).resolve()}")

    if args.napari:
        bands_all = sorted({int(item["z_band"]) for item in final_records})
        bands_show = _select_napari_bands(final_records, num_bands=args.num_bands)
        print(f"napari_total_bands: {len(bands_all)}")
        print(f"napari_showing_bands: {bands_show}")
        _show_napari(
            points_zyx,
            final_records,
            downsample=args.napari_downsample,
            point_size=args.napari_point_size,
            num_bands=args.num_bands,
        )


if __name__ == "__main__":
    main()
