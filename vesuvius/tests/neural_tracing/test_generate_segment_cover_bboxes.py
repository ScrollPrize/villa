"""Tests for the memoized per-band optimizer and pruner in generate_segment_cover_bboxes.

The production module memoizes pure sub-results of the band optimizer so that pruning
does not recompute them for every removal attempt. Its output must not change at all,
so most tests here run it beside ``_reference_implementation()``, a verbatim copy of
the optimizer and pruner as they were before memoization, and require identical
records: same order, same keys, same shifts and bboxes, same augmented point arrays.

The module imports ``vc_reader.tifxyz`` only to load a segment from disk, which no test
here does. Where the compiled ``vc_reader`` package is not installed, the ``cover``
fixture puts a stand-in on ``sys.path`` for the duration of this file, so the tests run
either way.
"""

import gc
import importlib
import io
import json
import sys
import warnings
import weakref
from contextlib import redirect_stdout
from types import SimpleNamespace

import numpy as np
import pytest


MODULE_NAME = "vesuvius.neural_tracing.inference.generate_segment_cover_bboxes"

_STAND_IN_TIFXYZ = '''"""Stand-in for vc_reader.tifxyz, used only when the compiled package is not installed."""


def read(path):
    raise RuntimeError("the stand-in vc_reader.tifxyz cannot read surfaces")
'''


def _vc_reader_tifxyz_importable():
    try:
        from vc_reader import tifxyz  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.fixture(scope="module")
def cover(tmp_path_factory):
    """The production module, importable in this process and in the worker processes it starts."""
    if _vc_reader_tifxyz_importable():
        yield importlib.import_module(MODULE_NAME)
        return

    # No compiled vc_reader: provide ``vc_reader.tifxyz`` from a temporary directory on
    # sys.path. Worker processes inherit sys.path, so they can import the module too.
    # Everything touched here is put back afterwards, so other test files are unaffected.
    stand_in_root = tmp_path_factory.mktemp("vc_reader_stand_in")
    package = stand_in_root / "vc_reader"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "tifxyz.py").write_text(_STAND_IN_TIFXYZ)

    shadowed = {
        name: sys.modules.pop(name)
        for name in list(sys.modules)
        if name == "vc_reader" or name.startswith("vc_reader.") or name == MODULE_NAME
    }
    sys.path.insert(0, str(stand_in_root))
    importlib.invalidate_caches()
    try:
        yield importlib.import_module(MODULE_NAME)
    finally:
        sys.path.remove(str(stand_in_root))
        for name in list(sys.modules):
            if name == "vc_reader" or name.startswith("vc_reader.") or name == MODULE_NAME:
                del sys.modules[name]
        sys.modules.update(shadowed)
        importlib.invalidate_caches()


# ---------------------------------------------------------------------------
# Reference: the optimizer and pruner before memoization
# ---------------------------------------------------------------------------


def _reference_implementation():
    """Verbatim copy of the band optimizer, the pruner and their helpers from before memoization.

    Kept inside a function so the names cannot be confused with the production ones.
    Do not "fix" or modernize this code: it is the oracle the production module is
    compared against.
    """

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


    def _validate_band_coverage(points_zyx_band, bboxes):
        if points_zyx_band.shape[0] == 0:
            return True
        covered = np.zeros((points_zyx_band.shape[0],), dtype=bool)
        for bbox in bboxes:
            covered |= _points_within_bbox(points_zyx_band, bbox)
            if covered.all():
                return True
        return bool(covered.all())


    def _build_band_neighbor_map(records):
        by_cell = {(int(r["grid_index"][1]), int(r["grid_index"][2])): i for i, r in enumerate(records)}
        neighbor_map = [[] for _ in range(len(records))]
        for (gy, gx), idx in by_cell.items():
            up = (gy - 1, gx)
            down = (gy + 1, gx)
            left = (gy, gx - 1)
            right = (gy, gx + 1)
            if up in by_cell:
                neighbor_map[idx].append((by_cell[up], "up"))
            if down in by_cell:
                neighbor_map[idx].append((by_cell[down], "down"))
            if left in by_cell:
                neighbor_map[idx].append((by_cell[left], "left"))
            if right in by_cell:
                neighbor_map[idx].append((by_cell[right], "right"))
        return neighbor_map


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


    def _optimize_band_bboxes(
        records,
        points_zyx_band,
        max_passes=10,
        allow_neighbor_overlap=False,
        require_neighbor_overlap=False,
    ):
        if not records:
            return records

        records = sorted(records, key=lambda r: (int(r["grid_index"][1]), int(r["grid_index"][2])))
        neighbor_map = _build_band_neighbor_map(records)

        states = []
        for rec in records:
            bbox = rec["initial_bbox"]
            pts = rec["points"]
            obj_y = _axis_objective(pts, bbox, "y")
            obj_x = _axis_objective(pts, bbox, "x")
            if obj_y <= obj_x:
                states.append({"axis": "y", "shift": 0, "objective": obj_y})
            else:
                states.append({"axis": "x", "shift": 0, "objective": obj_x})

        for _ in range(int(max_passes)):
            changed = False
            for idx, rec in enumerate(records):
                pts = rec["points"]
                init_bbox = rec["initial_bbox"]

                current = states[idx]
                best_axis = current["axis"]
                best_shift = int(current["shift"])
                best_obj = float(current["objective"])

                for axis in ("y", "x"):
                    low, high = _axis_shift_bounds_from_points(pts, init_bbox, axis)
                    if low > high:
                        continue

                    _, _, base_center = _axis_min_and_size(init_bbox, axis)
                    axis_idx = 1 if axis == "y" else 2
                    median_pt = float(np.median(pts[:, axis_idx]))
                    ideal_shift = median_pt - base_center
                    cur_shift_for_axis = int(current["shift"]) if axis == current["axis"] else 0
                    candidates = _generate_shift_candidates(low, high, ideal_shift, cur_shift_for_axis)

                    for shift in candidates:
                        cand_bbox = _bbox_from_state(init_bbox, axis, shift)
                        if not np.all(_points_within_bbox(pts, cand_bbox)):
                            continue

                        ok_touch = True
                        for nbr_idx, relation in neighbor_map[idx]:
                            nbr_state = states[nbr_idx]
                            nbr_bbox = _bbox_from_state(
                                records[nbr_idx]["initial_bbox"],
                                nbr_state["axis"],
                                nbr_state["shift"],
                            )
                            if not _touching_relation_holds(
                                cand_bbox,
                                nbr_bbox,
                                relation,
                                allow_overlap=bool(allow_neighbor_overlap),
                                require_overlap=bool(require_neighbor_overlap),
                            ):
                                ok_touch = False
                                break
                        if not ok_touch:
                            continue

                        cand_obj = _axis_objective(pts, cand_bbox, axis)
                        if _is_candidate_better(cand_obj, axis, shift, best_obj, best_axis, best_shift):
                            best_obj = cand_obj
                            best_axis = axis
                            best_shift = int(shift)

                if best_axis != current["axis"] or int(best_shift) != int(current["shift"]):
                    states[idx] = {"axis": best_axis, "shift": int(best_shift), "objective": float(best_obj)}
                    changed = True

            if not changed:
                break

        out = []
        final_bboxes = []
        for rec, state in zip(records, states):
            final_bbox = _bbox_from_state(rec["initial_bbox"], state["axis"], state["shift"])
            out_item = dict(rec)
            out_item["shift_axis"] = state["axis"]
            out_item["shift_voxels"] = int(state["shift"])
            out_item["bbox"] = final_bbox
            out.append(out_item)
            final_bboxes.append(final_bbox)

        if not _validate_band_coverage(points_zyx_band, final_bboxes):
            raise RuntimeError("Band optimization violated full point coverage.")

        return out


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

        kept = [dict(item) for item in optimized_records]
        removed = 0
        attempts = 0

        while len(kept) > 1:
            if max_remove is not None and removed >= int(max_remove):
                break

            # Try smallest support boxes first. Ties resolve deterministically by grid index.
            candidate_order = sorted(
                range(len(kept)),
                key=lambda i: (
                    int(kept[i]["points"].shape[0]),
                    int(kept[i]["grid_index"][1]),
                    int(kept[i]["grid_index"][2]),
                ),
            )

            removed_one = False
            for idx in candidate_order:
                trial_records = [kept[j] for j in range(len(kept)) if j != idx]
                attempts += 1
                trial_records_seeded = _augment_records_with_uncovered_points(trial_records, points_zyx_band)
                trial_optimized = None
                if bool(require_neighbor_overlap):
                    contact_modes = ((True, True),)
                elif bool(allow_neighbor_overlap):
                    contact_modes = ((True, False), (False, False))
                else:
                    contact_modes = ((False, False), (True, False))
                # Try contact modes in order; accept any full-coverage solution.
                for allow_overlap, require_overlap in contact_modes:
                    try:
                        trial_optimized = _optimize_band_bboxes(
                            trial_records_seeded,
                            points_zyx_band,
                            max_passes=max_passes,
                            allow_neighbor_overlap=allow_overlap,
                            require_neighbor_overlap=require_overlap,
                        )
                        break
                    except RuntimeError:
                        trial_optimized = None
                if trial_optimized is None:
                    continue

                kept = trial_optimized
                removed += 1
                removed_one = True
                break

            if not removed_one:
                break

        return kept, {
            "initial": int(len(optimized_records)),
            "removed": int(removed),
            "final": int(len(kept)),
            "attempts": int(attempts),
        }


    def _augment_records_with_uncovered_points(records, points_zyx_band):
        if not records:
            return []

        points_zyx_band = np.asarray(points_zyx_band, dtype=np.float64)
        if points_zyx_band.shape[0] == 0:
            return [dict(item) for item in records]

        # Prefer current optimized bbox if available; otherwise use initial bbox.
        bboxes = []
        for rec in records:
            if "bbox" in rec:
                bboxes.append(tuple(int(v) for v in rec["bbox"]))
            else:
                bboxes.append(tuple(int(v) for v in rec["initial_bbox"]))

        covered = np.zeros((points_zyx_band.shape[0],), dtype=bool)
        for bbox in bboxes:
            covered |= _points_within_bbox(points_zyx_band, bbox)
        uncovered_idx = np.where(~covered)[0]
        if uncovered_idx.size == 0:
            return [dict(item) for item in records]

        centers = np.zeros((len(records), 2), dtype=np.float64)
        for i, bbox in enumerate(bboxes):
            centers[i, 0] = 0.5 * (float(bbox[2]) + float(bbox[3]))  # y
            centers[i, 1] = 0.5 * (float(bbox[4]) + float(bbox[5]))  # x

        assigned_points = [[] for _ in range(len(records))]
        for p_idx in uncovered_idx:
            p = points_zyx_band[p_idx]
            d = np.abs(centers[:, 0] - float(p[1])) + np.abs(centers[:, 1] - float(p[2]))
            tgt = int(np.argmin(d))
            assigned_points[tgt].append(p)

        out = []
        for i, rec in enumerate(records):
            rec_out = dict(rec)
            base_points = np.asarray(rec["points"], dtype=np.float64)
            extra = np.asarray(assigned_points[i], dtype=np.float64) if assigned_points[i] else np.zeros((0, 3), dtype=np.float64)
            if extra.shape[0] > 0:
                rec_out["points"] = np.concatenate([base_points, extra], axis=0)
            else:
                rec_out["points"] = base_points
            out.append(rec_out)
        return out


    def _optimize_single_band(
        z_band,
        band_records,
        band_points,
        prune_bboxes,
        prune_max_remove_per_band,
        overlap,
    ):
        overlap_positive = bool(float(overlap) > 0.0)
        optimized = _optimize_band_bboxes(
            band_records,
            band_points,
            max_passes=5,
            allow_neighbor_overlap=overlap_positive,
            require_neighbor_overlap=overlap_positive,
        )
        prune_stats = None
        if prune_bboxes:
            optimized, prune_stats = _prune_band_bboxes_via_reopt(
                optimized,
                band_points,
                max_passes=5,
                max_remove=prune_max_remove_per_band,
                allow_neighbor_overlap=overlap_positive,
                require_neighbor_overlap=overlap_positive,
            )
        optimized_sorted = sorted(optimized, key=lambda r: (int(r["grid_index"][1]), int(r["grid_index"][2])))
        return int(z_band), optimized_sorted, prune_stats

    return SimpleNamespace(
        _optimize_band_bboxes=_optimize_band_bboxes,
        _prune_band_bboxes_via_reopt=_prune_band_bboxes_via_reopt,
        _optimize_single_band=_optimize_single_band,
        _points_within_bbox=_points_within_bbox,
    )


REFERENCE = _reference_implementation()


# ---------------------------------------------------------------------------
# Inputs and comparison helpers
# ---------------------------------------------------------------------------

CROP_ZYX = (40, 48, 40)

# Vertices along the arc of the test sheet, per overlap. The reference pruner re-optimizes
# a whole band for every removal attempt, so bands are kept to 15-20 records: each
# reference run then takes about a second.
SHEET_COLS = {0.0: 60, 0.25: 36, 0.5: 20}
SHEET_ROWS = 12


def _curved_sheet(cols, rows=SHEET_ROWS, radius=160.0, row_step=9.0, col_step=9.0):
    """Vertices (z, y, x) of a rolled sheet, like one wrap of a scroll segment.

    ``cols`` vertices follow an arc in the y/x plane and ``rows`` copies of it are stacked
    along z. The arc drifts and widens from row to row, its radius ripples, and the
    vertices are jittered, so every z-band cuts the crop grid differently. Everything is
    deterministic.
    """
    col = np.arange(cols, dtype=np.float64)[None, :]
    row = np.arange(rows, dtype=np.float64)[:, None]
    theta = 0.35 + col * (col_step / radius) + 0.015 * row
    r = radius + 6.0 * np.sin(3.0 * theta) + 1.5 * row
    z = 40.0 + row_step * row + 0.4 * np.sin(1.7 * col + 0.3)
    y = 300.0 + r * np.sin(theta) + 0.4 * np.sin(2.3 * row + 0.9 * col)
    x = 500.0 + r * np.cos(theta) + 0.4 * np.cos(1.1 * row + 1.3 * col)
    points = np.stack([np.broadcast_to(z, y.shape), y, x], axis=-1).reshape(-1, 3)
    return np.ascontiguousarray(points, dtype=np.float64)


def _band_jobs(cover, points_zyx, crop_zyx, overlap):
    """(z_band, records, band_points) per z-band, built the way _generate_segment_cover_records builds them."""
    crop = np.asarray(crop_zyx, dtype=np.int64)
    world_min, world_max = cover._compute_world_bbox(points_zyx)
    if overlap <= 0.0:
        grouped, _, stride = cover._assign_points_to_chunks(points_zyx, world_min, world_max, crop)
    else:
        grouped, _, stride = cover._assign_points_to_chunks_overlapped(
            points_zyx, world_min, world_max, crop, overlap_frac=overlap
        )
    by_band = cover._build_band_records(points_zyx, grouped, world_min, crop, stride_zyx=stride)
    jobs = []
    for z_band in sorted(by_band):
        records = by_band[z_band]
        point_indices = np.unique(np.concatenate([item["point_indices"] for item in records]))
        jobs.append((int(z_band), records, points_zyx[point_indices]))
    return jobs


def _reference_bands(cover, points_zyx, overlap, prune, max_remove=None):
    """(z_band, optimized records, prune stats) per z-band, from the reference optimizer and pruner."""
    return [
        REFERENCE._optimize_single_band(z_band, records, band_points, prune, max_remove, overlap)
        for z_band, records, band_points in _band_jobs(cover, points_zyx, CROP_ZYX, overlap)
    ]


def _final_records(bands):
    """Number the per-band results the way _generate_segment_cover_records does."""
    final_records = []
    prune_stats_by_band = {}
    for z_band, optimized, prune_stats in bands:
        if prune_stats is not None:
            prune_stats_by_band[z_band] = prune_stats
        for item in optimized:
            out_item = dict(item)
            out_item["bbox_id"] = len(final_records)
            out_item["z_band"] = z_band
            final_records.append(out_item)
    return final_records, prune_stats_by_band


_SHEETS = {}
_SHEET_REFERENCE = {}


def _sheet(overlap):
    """The test sheet for ``overlap``. Shared between tests, so it is read-only."""
    if overlap not in _SHEETS:
        points = _curved_sheet(SHEET_COLS[overlap])
        points.flags.writeable = False
        _SHEETS[overlap] = points
    return _SHEETS[overlap]


def _sheet_reference(cover, overlap, prune, max_remove=None):
    """_reference_bands() of the test sheet, computed once per configuration."""
    key = (overlap, prune, max_remove)
    if key not in _SHEET_REFERENCE:
        _SHEET_REFERENCE[key] = _reference_bands(cover, _sheet(overlap), overlap, prune, max_remove)
    return _SHEET_REFERENCE[key]


def _assert_same_value(actual, expected, where):
    if isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray), where
        assert actual.dtype == expected.dtype, where
        assert actual.shape == expected.shape, where
        assert np.array_equal(actual, expected, equal_nan=expected.dtype.kind == "f"), where
    else:
        assert type(actual) is type(expected), where
        assert actual == expected, where


def _assert_same_records(actual, expected):
    assert len(actual) == len(expected)
    for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
        assert list(actual_item.keys()) == list(expected_item.keys()), f"record {index}"
        for key, expected_value in expected_item.items():
            _assert_same_value(actual_item[key], expected_value, f"record {index}, key {key!r}")


def _outcome(function, *args, **kwargs):
    """("ok", result) or ("error", exception type, message), with numpy's warnings about NaN and empty input muted."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            return ("ok", function(*args, **kwargs))
        except Exception as exc:  # the comparison below is what makes this meaningful
            return ("error", type(exc), str(exc))


def _assert_same_outcome(actual, expected):
    assert actual[0] == expected[0], (actual, expected)
    if expected[0] == "error":
        assert actual[1:] == expected[1:]
    elif isinstance(expected[1], tuple):  # the pruner returns (records, stats)
        _assert_same_records(actual[1][0], expected[1][0])
        assert actual[1][1] == expected[1][1]
    else:
        _assert_same_records(actual[1], expected[1])


def _copy_records(records):
    return [{key: (value.copy() if isinstance(value, np.ndarray) else value) for key, value in item.items()} for item in records]


# (allow_neighbor_overlap, require_neighbor_overlap)
STRICT, ALLOW, REQUIRE = (False, False), (True, False), (True, True)
CONTACT_MODES = (STRICT, ALLOW, REQUIRE)


def _band_outcomes(impl, records, band_points, contact_mode=STRICT, max_passes=5):
    """Outcomes of optimizing a band with ``impl``, then pruning both the raw records and the optimized ones.

    The raw records have no "bbox" yet and are in arbitrary order, which the pruner accepts.
    """
    kwargs = dict(allow_neighbor_overlap=contact_mode[0], require_neighbor_overlap=contact_mode[1])
    optimized = _outcome(
        impl._optimize_band_bboxes, _copy_records(records), band_points.copy(), max_passes=max_passes, **kwargs
    )
    outcomes = [optimized]
    prune_inputs = [records]
    if optimized[0] == "ok":
        prune_inputs.append(optimized[1])
    for prune_input in prune_inputs:
        for max_remove in (None, 1):
            outcomes.append(
                _outcome(
                    impl._prune_band_bboxes_via_reopt,
                    _copy_records(prune_input),
                    band_points.copy(),
                    max_passes=max(1, max_passes),
                    max_remove=max_remove,
                    **kwargs,
                )
            )
    return outcomes


def _assert_band_matches_reference(cover, records, band_points, contact_mode=STRICT, max_passes=5):
    expected = _band_outcomes(REFERENCE, records, band_points, contact_mode, max_passes)
    actual = _band_outcomes(cover, records, band_points, contact_mode, max_passes)
    assert len(actual) == len(expected)
    for actual_outcome, expected_outcome in zip(actual, expected):
        _assert_same_outcome(actual_outcome, expected_outcome)
    return expected


class _Lcg:
    """A tiny linear congruential generator.

    The random bands below must be the same on every platform and numpy version: some
    seeds were picked because they expose a particular way of getting the memoization
    wrong (see RANDOM_BAND_CASES).
    """

    def __init__(self, seed):
        self.state = (int(seed) * 2654435761 + 1013904223) % (1 << 32)

    def random(self):
        self.state = (self.state * 1664525 + 1013904223) % (1 << 32)
        return self.state / float(1 << 32)

    def uniform(self, low, high):
        return low + (high - low) * self.random()

    def integer(self, low, high):
        """An integer in [low, high)."""
        return low + int(self.random() * (high - low))

    def shuffled(self, items):
        items = list(items)
        for i in range(len(items) - 1, 0, -1):
            j = self.integer(0, i + 1)
            items[i], items[j] = items[j], items[i]
        return items


def _random_band(
    seed,
    crop_yx=(40, 36),
    stride_yx=(20, 18),
    grid_yx=(3, 4),
    record_range=(3, 9),
    cluster=8.0,
    stray_points=0,
    float32=False,
    duplicate_cells=False,
):
    """A small band of records on an (overlapping) grid, each holding a tight cluster of points.

    Tight clusters leave every bbox room to shift along both axes, which is what makes
    the optimizer's choices depend on the neighbors. About half the records also get one
    far point, so that the median and the mean of their coordinates differ. Records come
    back in shuffled order. ``stray_points`` adds band points that no record holds.
    """
    rng = _Lcg(seed)
    crop_z = 16
    cells = [(gy, gx) for gy in range(grid_yx[0]) for gx in range(grid_yx[1])]
    chosen = sorted(rng.shuffled(cells)[: rng.integer(*record_range)])
    if duplicate_cells:
        chosen.append(chosen[rng.integer(0, len(chosen))])

    records = []
    offset = 0
    for gy, gx in chosen:
        y_min = 100 + gy * stride_yx[0]
        x_min = 200 + gx * stride_yx[1]
        count = rng.integer(2, 7)
        size_y = rng.uniform(1.0, cluster)
        size_x = rng.uniform(1.0, cluster)
        start_y = y_min + rng.uniform(0.0, crop_yx[0] - size_y)
        start_x = x_min + rng.uniform(0.0, crop_yx[1] - size_x)
        points = np.asarray(
            [
                [rng.uniform(50.0, 50.0 + crop_z), rng.uniform(start_y, start_y + size_y), rng.uniform(start_x, start_x + size_x)]
                for _ in range(count)
            ],
            dtype=np.float64,
        )
        if rng.random() < 0.5:
            points[0, 1] = rng.uniform(y_min, y_min + crop_yx[0])
            points[0, 2] = rng.uniform(x_min, x_min + crop_yx[1])
        records.append(
            {
                "grid_index": (0, gy, gx),
                "initial_bbox": (50, 50 + crop_z - 1, y_min, y_min + crop_yx[0] - 1, x_min, x_min + crop_yx[1] - 1),
                "point_indices": np.arange(offset, offset + count, dtype=np.int64),
                "points": points.astype(np.float32) if float32 else points,
            }
        )
        offset += count

    band_points = np.concatenate([item["points"] for item in records]).astype(np.float64)
    if stray_points:
        # Near existing points, so that a bbox can still reach them once they are handed to a record.
        picked = [rng.integer(0, band_points.shape[0]) for _ in range(stray_points)]
        offsets = [[0.0, rng.uniform(-6.0, 6.0), rng.uniform(-6.0, 6.0)] for _ in range(stray_points)]
        band_points = np.concatenate([band_points, band_points[picked] + np.asarray(offsets)])
    return rng.shuffled(records), band_points


def _track_band_caches(cover, monkeypatch, keep_alive):
    """Make the module build _BandCache subclasses that can be observed; returns the list they are collected in."""
    created = []

    class TrackedBandCache(cover._BandCache):
        __slots__ = ("__weakref__",)

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self if keep_alive else weakref.ref(self))

    monkeypatch.setattr(cover, "_BandCache", TrackedBandCache)
    return created


# ---------------------------------------------------------------------------
# Whole bands of a curved sheet, against the reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("prune", [False, True])
@pytest.mark.parametrize("overlap", [0.0, 0.25, 0.5])
def test_single_band_matches_reference_on_curved_sheet(cover, overlap, prune):
    jobs = _band_jobs(cover, _sheet(overlap), CROP_ZYX, overlap)
    reference = _sheet_reference(cover, overlap, prune)
    assert len(jobs) == len(reference) >= 2

    for (z_band, records, band_points), expected in zip(jobs, reference):
        actual = cover._optimize_single_band(z_band, records, band_points, prune, None, overlap)
        assert actual[0] == expected[0]
        _assert_same_records(actual[1], expected[1])
        assert actual[2] == expected[2]

    if prune:
        # The sheet must really exercise pruning: removals, and attempts that fail and are retried.
        removed = sum(stats["removed"] for _, _, stats in reference)
        attempts = sum(stats["attempts"] for _, _, stats in reference)
        assert removed >= 2 * len(reference) and attempts > 2 * removed


RESULT_KEYS = [
    "points_zyx",
    "crop_size_zyx",
    "overlap",
    "world_min_zyx",
    "world_max_zyx",
    "grouped",
    "grid_shape",
    "stride_zyx",
    "by_band",
    "final_records",
    "initial_chunk_count",
    "occupied_chunk_count",
    "final_bbox_count",
    "prune_stats_by_band",
    "prune_removed_total",
    "prune_attempts_total",
    "band_workers",
]


@pytest.mark.parametrize("overlap", [0.0, 0.25, 0.5])
def test_generate_segment_cover_records_matches_reference(cover, overlap):
    expected_records, expected_stats = _final_records(_sheet_reference(cover, overlap, True))

    result = cover._generate_segment_cover_records(_sheet(overlap), CROP_ZYX, overlap=overlap, prune_bboxes=True)

    # deprecated/ink-detection/tifxyz_dataset/patch_finding.py reads this dict; its keys must stay as they are.
    assert list(result.keys()) == RESULT_KEYS
    _assert_same_records(result["final_records"], expected_records)
    assert result["prune_stats_by_band"] == expected_stats
    assert result["final_bbox_count"] == len(expected_records)
    assert result["prune_removed_total"] == sum(stats["removed"] for stats in expected_stats.values())
    assert result["prune_attempts_total"] == sum(stats["attempts"] for stats in expected_stats.values())
    assert result["band_workers"] == 1


def test_generate_segment_cover_records_without_pruning_matches_reference(cover):
    expected_records, expected_stats = _final_records(_sheet_reference(cover, 0.25, False))

    result = cover._generate_segment_cover_records(_sheet(0.25), CROP_ZYX, overlap=0.25)

    assert list(result.keys()) == RESULT_KEYS
    _assert_same_records(result["final_records"], expected_records)
    assert expected_stats == {} and result["prune_stats_by_band"] == {}
    assert result["prune_removed_total"] == 0 and result["prune_attempts_total"] == 0


@pytest.mark.parametrize("max_remove", [1, 2])
def test_prune_max_remove_per_band_matches_reference(cover, max_remove):
    expected_records, expected_stats = _final_records(_sheet_reference(cover, 0.25, True, max_remove))
    unlimited_records, _ = _final_records(_sheet_reference(cover, 0.25, True))
    assert len(expected_records) > len(unlimited_records), "the cap must be what stops pruning"

    result = cover._generate_segment_cover_records(
        _sheet(0.25), CROP_ZYX, overlap=0.25, prune_bboxes=True, prune_max_remove_per_band=max_remove
    )

    _assert_same_records(result["final_records"], expected_records)
    assert result["prune_stats_by_band"] == expected_stats
    assert all(stats["removed"] == max_remove for stats in result["prune_stats_by_band"].values())


@pytest.mark.slow
@pytest.mark.parametrize("overlap", [0.0, 0.25, 0.5])
def test_larger_sheet_matches_reference(cover, overlap):
    # Bands of 30-50 records: many more removal rounds, and failed attempts retried across them.
    # The reference needs up to a minute for this, hence the marker.
    points = _curved_sheet(int(SHEET_COLS[overlap] * 2.5), rows=2 * SHEET_ROWS)
    expected_records, expected_stats = _final_records(_reference_bands(cover, points, overlap, True))

    result = cover._generate_segment_cover_records(points, CROP_ZYX, overlap=overlap, prune_bboxes=True)

    _assert_same_records(result["final_records"], expected_records)
    assert result["prune_stats_by_band"] == expected_stats


def test_single_record_bands_match_reference(cover):
    # A strip narrower than one crop: every z-band holds exactly one record, so there is nothing to prune.
    row = np.arange(12.0)[:, None]
    col = np.arange(3.0)[None, :]
    points = np.stack(
        [np.broadcast_to(40.0 + 9.0 * row, (12, 3)), 300.0 + 8.0 * col + 0.3 * row, 500.0 + 5.0 * col - 0.2 * row],
        axis=-1,
    ).reshape(-1, 3)
    jobs = _band_jobs(cover, points, CROP_ZYX, 0.0)
    assert len(jobs) == 3 and all(len(records) == 1 for _, records, _ in jobs)

    for z_band, records, band_points in jobs:
        expected = REFERENCE._optimize_single_band(z_band, records, band_points, True, None, 0.0)
        actual = cover._optimize_single_band(z_band, records, band_points, True, None, 0.0)
        _assert_same_records(actual[1], expected[1])
        assert actual[2] == expected[2] == {"initial": 1, "removed": 0, "final": 1, "attempts": 0}


def test_non_contiguous_points_match_reference(cover):
    # The real loader hands over ``surface.xyz[valid][:, ::-1]``, a negatively strided view.
    xyz = np.ascontiguousarray(_sheet(0.25)[:, ::-1])
    points = xyz[:, ::-1]
    assert not points.flags.c_contiguous
    expected_records, expected_stats = _final_records(_sheet_reference(cover, 0.25, True))

    result = cover._generate_segment_cover_records(points, CROP_ZYX, overlap=0.25, prune_bboxes=True)

    _assert_same_records(result["final_records"], expected_records)
    assert result["prune_stats_by_band"] == expected_stats

    # The band functions themselves also take strided record and band arrays.
    z_band, records, band_points = _band_jobs(cover, _sheet(0.25), CROP_ZYX, 0.25)[0]
    strided_records = []
    for item in records:
        padded = np.zeros((item["points"].shape[0] * 2, 6), dtype=np.float64)
        padded[::2, ::2] = item["points"]
        strided_records.append({**item, "points": padded[::2, ::2]})
    padded_band = np.zeros((band_points.shape[0], 6), dtype=np.float64)
    padded_band[:, ::2] = band_points
    strided_band = padded_band[:, ::2]
    assert not strided_band.flags.c_contiguous and not strided_records[0]["points"].flags.c_contiguous
    expected = _sheet_reference(cover, 0.25, True)[0]
    actual = cover._optimize_single_band(z_band, strided_records, strided_band, True, None, 0.25)
    _assert_same_records(actual[1], expected[1])
    assert actual[2] == expected[2]


# ---------------------------------------------------------------------------
# The band functions called directly: contact modes, pass counts, odd inputs
# ---------------------------------------------------------------------------

def _cases(name, seeds, **kwargs):
    return [dict(id=f"{name}-{seed}", seed=seed, **kwargs) for seed in seeds]


# Bands for the test below. A stride below the crop size makes the initial bboxes overlap,
# as --overlap does. The first group is a plain sweep. The seeds of the second group were
# picked by search: each of them makes at least one deliberately broken variant of the
# memoization return something else than the reference, for example a step key that leaves
# out one neighbor span, a span that is not refreshed after a shift, or a variant looked up
# without its extra points. Together they tell each of those apart from the real thing.
RANDOM_BAND_CASES = (
    _cases("overlap50-require", range(4), contact_mode=REQUIRE)
    + _cases("overlap50-allow", range(4, 6), contact_mode=ALLOW)
    + _cases("overlap25-require", range(6, 8), contact_mode=REQUIRE, stride_yx=(30, 27))
    + _cases("no-overlap-strict", range(8, 12), contact_mode=STRICT, stride_yx=(40, 36))
    + _cases("no-overlap-strict-1-pass", range(12, 14), contact_mode=STRICT, stride_yx=(40, 36), max_passes=1)
    + _cases("overlap50-require-10-passes", range(14, 16), contact_mode=REQUIRE, max_passes=10)
    # Picked by search, see above.
    + _cases("no-overlap-strict-duplicate-cell", (3, 15, 18, 24), contact_mode=STRICT, stride_yx=(40, 36), duplicate_cells=True)
    + _cases("overlap50-require-float32", (13, 25), contact_mode=REQUIRE, float32=True)
    + _cases("no-overlap-allow", (1, 21), contact_mode=ALLOW, stride_yx=(40, 36))
    + _cases("overlap50-require-stray-points", (1, 6), contact_mode=REQUIRE, stray_points=2)
    + _cases("overlap25-require", (13, 25), contact_mode=REQUIRE, stride_yx=(30, 27))
    + _cases("overlap25-allow-stray-points", (2, 4, 13, 33), contact_mode=ALLOW, stride_yx=(30, 27), stray_points=3)
)


@pytest.mark.parametrize("case", RANDOM_BAND_CASES, ids=lambda case: case["id"])
def test_band_functions_match_reference_on_random_bands(cover, case):
    kwargs = {key: value for key, value in case.items() if key not in ("id", "contact_mode", "max_passes")}
    records, band_points = _random_band(**kwargs)
    _assert_band_matches_reference(cover, records, band_points, case["contact_mode"], case.get("max_passes", 5))


def test_random_bands_exercise_pruning(cover):
    # Guards the cases above against becoming trivial: across them bboxes are removed,
    # removal attempts fail, and removed bboxes leave points that a survivor takes over.
    removed = attempts = grown = 0
    for case in RANDOM_BAND_CASES:
        kwargs = {key: value for key, value in case.items() if key not in ("id", "contact_mode", "max_passes")}
        records, band_points = _random_band(**kwargs)
        initial_sizes = sorted(item["points"].shape[0] for item in records)
        for outcome in _band_outcomes(cover, records, band_points, case["contact_mode"], case.get("max_passes", 5))[1:]:
            if outcome[0] == "ok":
                kept, stats = outcome[1]
                removed += stats["removed"]
                attempts += stats["attempts"]
                grown += max(item["points"].shape[0] for item in kept) > initial_sizes[-1]
    assert removed > len(RANDOM_BAND_CASES)
    assert attempts > 2 * removed
    assert grown > 0


def test_points_no_bbox_covered_are_handed_over_in_point_order(cover):
    # Two records side by side, without a "bbox" yet. Band point 2 lies left of both initial
    # bboxes, so nothing covers it. Removing the left record leaves its point 0 uncovered as
    # well; the right record takes both and must list them in ascending band order: 0, then 2.
    left = {
        "grid_index": (0, 0, 0),
        "initial_bbox": (50, 65, 100, 139, 200, 239),
        "point_indices": np.asarray([0], dtype=np.int64),
        "points": np.asarray([[55.0, 110.0, 205.0]]),
    }
    right = {
        "grid_index": (0, 0, 1),
        "initial_bbox": (50, 65, 100, 139, 220, 259),
        "point_indices": np.asarray([1], dtype=np.int64),
        "points": np.asarray([[55.0, 110.0, 225.0]]),
    }
    band_points = np.asarray([[55.0, 110.0, 205.0], [55.0, 110.0, 225.0], [55.0, 110.0, 198.5]])

    for contact_mode in CONTACT_MODES:
        expected = _assert_band_matches_reference(cover, [left, right], band_points, contact_mode)
        assert expected[0][:2] == ("error", RuntimeError)  # the optimizer alone cannot cover point 2
        kept, stats = expected[1][1]
        assert stats == {"initial": 2, "removed": 1, "final": 1, "attempts": 1}
        assert kept[0]["grid_index"] == (0, 0, 1)
        assert kept[0]["points"][:, 2].tolist() == [225.0, 205.0, 198.5]


def test_axis_tie_is_resolved_like_reference(cover):
    # One point, equally far from the bbox center along y and along x: both objectives are
    # exactly 7.0, and the optimizer must start on "y" as before.
    records = [
        {
            "grid_index": (0, 0, 0),
            "initial_bbox": (50, 65, 100, 123, 200, 219),
            "point_indices": np.asarray([0], dtype=np.int64),
            "points": np.asarray([[55.0, 105.0, 203.0]]),
        },
        {
            "grid_index": (0, 0, 1),
            "initial_bbox": (50, 65, 100, 123, 220, 239),
            "point_indices": np.asarray([1], dtype=np.int64),
            "points": np.asarray([[55.0, 119.0, 237.0]]),
        },
    ]
    band_points = np.concatenate([item["points"] for item in records])
    for contact_mode in CONTACT_MODES:
        for max_passes in (0, 5):
            expected = _assert_band_matches_reference(cover, records, band_points, contact_mode, max_passes)
            if max_passes == 0:
                assert [item["shift_axis"] for item in expected[0][1]] == ["y", "y"]


def test_empty_inputs_match_reference(cover):
    band_points = np.zeros((0, 3), dtype=np.float64)
    empty = []
    assert cover._optimize_band_bboxes(empty, band_points) is empty
    assert REFERENCE._optimize_band_bboxes(empty, band_points) is empty
    actual = cover._prune_band_bboxes_via_reopt(empty, band_points)
    assert actual[0] is empty and actual[1] == REFERENCE._prune_band_bboxes_via_reopt(empty, band_points)[1]
    assert cover._optimize_single_band(3, [], band_points, True, None, 0.5) == REFERENCE._optimize_single_band(
        3, [], band_points, True, None, 0.5
    )

    # Records, but no band points to cover: any state is a full cover.
    records, _ = _random_band(3)
    for contact_mode in CONTACT_MODES:
        _assert_band_matches_reference(cover, records, band_points, contact_mode)

    # A record without points: the optimizer has always refused it with numpy's ValueError.
    records, band_points = _random_band(4)
    records[0]["points"] = np.zeros((0, 3), dtype=np.float64)
    expected = _assert_band_matches_reference(cover, records, band_points)
    assert expected[0][:2] == ("error", ValueError)


# ---------------------------------------------------------------------------
# NaN vertices
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("nan_axis", [0, 1, 2])
@pytest.mark.parametrize("contact_mode", CONTACT_MODES)
def test_nan_in_record_points_matches_reference(cover, nan_axis, contact_mode):
    records, band_points = _random_band(11)
    records[1]["points"][0, nan_axis] = np.nan

    # A NaN in y or x makes the shift bounds NaN, which the optimizer refuses with a
    # ValueError. A NaN in z leaves the record where it is: no candidate bbox contains it.
    expected = _assert_band_matches_reference(cover, records, band_points, contact_mode)
    if nan_axis in (1, 2):
        assert expected[0][:2] == ("error", ValueError)
    else:
        assert expected[0][0] == "ok"

    # With max_passes=0 the optimizer never looks at the shift bounds, so nothing raises there.
    expected_idle = _assert_band_matches_reference(cover, records, band_points, contact_mode, max_passes=0)
    assert expected_idle[0][0] == "ok"


@pytest.mark.parametrize("nan_axis", [0, 1, 2])
@pytest.mark.parametrize("contact_mode", CONTACT_MODES)
def test_nan_in_band_points_matches_reference(cover, nan_axis, contact_mode):
    records, band_points = _random_band(12)
    nan_point = band_points[:1].copy()
    nan_point[0, nan_axis] = np.nan
    band_points = np.concatenate([band_points, nan_point])

    # No bbox can contain a NaN point, so the optimizer reports the missing coverage. The
    # pruner hands the NaN point to a record, with whatever that leads to.
    expected = _assert_band_matches_reference(cover, records, band_points, contact_mode)
    assert expected[0][:2] == ("error", RuntimeError)
    assert len(expected) == 3


def test_containment_on_extremes_equals_per_point_test(cover):
    # _BandItem.contains replaces np.all(_points_within_bbox(...)) with comparisons on the
    # per-axis minimum and maximum. Check them against each other on edge values.
    rng = np.random.default_rng(5)
    band = cover._BandCache(np.zeros((0, 3)), 1)
    special = [np.nan, np.inf, -np.inf, 10.0, 19.999999999999996, 20.0, 9.999999999999998]
    checked = {True: 0, False: 0}
    for _ in range(300):
        points = rng.uniform(10.0, 20.0, size=(int(rng.integers(1, 5)), 3))
        if rng.random() < 0.6:
            points[rng.integers(0, points.shape[0]), rng.integers(0, 3)] = special[int(rng.integers(0, len(special)))]
        record = {"grid_index": (0, 0, 0), "initial_bbox": (10, 19, 10, 19, 10, 19), "points": points}
        item = band.item(record, points)
        for bbox in ((10, 19, 10, 19, 10, 19), (10, 19, 11, 20, 10, 19), (10, 19, 10, 19, 9, 18), (11, 19, 10, 19, 10, 19)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                expected = bool(np.all(REFERENCE._points_within_bbox(points, bbox)))
                assert item.contains(band, bbox) is expected
            checked[expected] += 1
    assert checked[True] > 0 and checked[False] > 0


def test_nearest_center_indices_equals_per_point_argmin(cover, monkeypatch):
    rng = np.random.default_rng(9)
    centers_y = np.asarray([0.0, 10.0, 0.0, 10.0, 5.0])
    centers_x = np.asarray([0.0, 0.0, 10.0, 10.0, 5.0])
    points = rng.uniform(-2.0, 12.0, size=(64, 3))
    points[0, 1:] = (5.0, 0.0)  # equally far from centers 0 and 1 (and 4): the first one wins
    points[1, 1:] = (np.nan, 3.0)  # NaN distance to every center: argmin picks the first
    points[2, 1:] = (0.0, 5.0)
    expected = [
        int(np.argmin(np.abs(centers_y - float(p[1])) + np.abs(centers_x - float(p[2])))) for p in points
    ]
    assert expected[0] == 0 and expected[1] == 0

    for chunk_elements in (1 << 15, 12, 1):  # one chunk, several rows per chunk, one row per chunk
        monkeypatch.setattr(cover, "_NEAREST_CENTER_CHUNK_ELEMENTS", chunk_elements)
        assert cover._nearest_center_indices(centers_y, centers_x, points).tolist() == expected


# ---------------------------------------------------------------------------
# Cache bounds and lifetime
# ---------------------------------------------------------------------------


def test_lru_cache_enforces_entry_and_weight_caps(cover):
    cache = cover._LruCache(3, 10, lambda key, value: value)
    cache.put("a", 4)
    cache.put("b", 4)
    assert cache.get("a") == 4  # "a" becomes the most recently used
    cache.put("c", 4)  # weight 12 > 10: the least recently used entry, "b", goes
    assert cache.get("b") is None and len(cache) == 2 and cache.weight == 8
    cache.put("too heavy", 11)  # could never fit: not stored, nothing evicted for it
    assert cache.get("too heavy") is None and len(cache) == 2 and cache.weight == 8
    cache.put("d", 1)
    cache.put("e", 1)  # four entries > 3: "a" goes
    assert cache.get("a") is None and len(cache) == 3 and cache.weight == 6
    assert [cache.get(key) for key in ("c", "d", "e")] == [4, 1, 1]


def test_band_cache_caps_scale_with_the_band(cover):
    small = cover._BandCache(np.zeros((10, 3)), 2)
    large = cover._BandCache(np.zeros((5000, 3)), 400)
    assert small.max_steps == cover._MIN_CACHE_ENTRIES
    assert large.max_steps == cover._STEP_MEMO_ENTRIES_PER_RECORD * 400
    assert large._cover._max_entries == cover._COVER_CACHE_ENTRIES_PER_RECORD * 400
    assert large._cover._max_weight == cover._COVER_CACHE_ELEMENTS_PER_POINT * 5000
    assert large._variants._max_entries == cover._VARIANT_CACHE_ENTRIES_PER_RECORD * 400
    assert large._variants._max_weight == cover._VARIANT_CACHE_ELEMENTS_PER_POINT * 5000
    # Point indices are stored in the smallest type that can address the band.
    assert small.cover_index((0, 0, 0, 0, 0, 0)).dtype == np.uint8
    assert large.cover_index((0, 0, 0, 0, 0, 0)).dtype == np.uint16


@pytest.mark.parametrize("overlap", [0.0, 0.5])
def test_caches_stay_within_their_caps_while_a_band_is_pruned(cover, monkeypatch, overlap):
    points = _sheet(overlap)
    bands = _track_band_caches(cover, monkeypatch, keep_alive=True)
    peaks = {"steps": 0, "objectives": 0, "cover": 0, "cover_weight": 0, "variants": 0, "variant_weight": 0}

    solve = cover._solve_band_states

    def solve_and_measure(band, *args, **kwargs):
        result = solve(band, *args, **kwargs)
        peaks["steps"] = max(peaks["steps"], len(band.steps) - band.max_steps)
        peaks["objectives"] = max(peaks["objectives"], len(band.objectives) - cover._OBJECTIVE_MEMO_MAX_ENTRIES)
        peaks["cover"] = max(peaks["cover"], len(band._cover) - band._cover._max_entries)
        peaks["cover_weight"] = max(peaks["cover_weight"], band._cover.weight - band._cover._max_weight)
        peaks["variants"] = max(peaks["variants"], len(band._variants) - band._variants._max_entries)
        peaks["variant_weight"] = max(peaks["variant_weight"], band._variants.weight - band._variants._max_weight)
        return result

    monkeypatch.setattr(cover, "_solve_band_states", solve_and_measure)
    result = cover._generate_segment_cover_records(points, CROP_ZYX, overlap=overlap, prune_bboxes=True)

    assert result["prune_removed_total"] > 0 and len(bands) == len(result["by_band"])
    # Checked after every optimizer run of every removal attempt: no table was ever over its cap.
    assert all(excess <= 0 for excess in peaks.values()), peaks
    for band in bands:
        assert 0 < len(band.steps) <= band.max_steps
        assert len(band.objectives) <= cover._OBJECTIVE_MEMO_MAX_ENTRIES


@pytest.mark.parametrize(
    "caps",
    [
        # One entry per table, no room for any point index: everything is recomputed.
        dict(_MIN_CACHE_ENTRIES=1, _OBJECTIVE_MEMO_MAX_ENTRIES=1, per_record=0, per_point=0),
        # A few entries: constant eviction while the tables are in use.
        dict(_MIN_CACHE_ENTRIES=3, _OBJECTIVE_MEMO_MAX_ENTRIES=5, per_record=0, per_point=1),
    ],
)
@pytest.mark.parametrize("overlap", [0.0, 0.5])
def test_output_is_unchanged_when_caches_are_starved(cover, monkeypatch, caps, overlap):
    monkeypatch.setattr(cover, "_MIN_CACHE_ENTRIES", caps["_MIN_CACHE_ENTRIES"])
    monkeypatch.setattr(cover, "_OBJECTIVE_MEMO_MAX_ENTRIES", caps["_OBJECTIVE_MEMO_MAX_ENTRIES"])
    for name in ("_STEP_MEMO_ENTRIES_PER_RECORD", "_COVER_CACHE_ENTRIES_PER_RECORD", "_VARIANT_CACHE_ENTRIES_PER_RECORD"):
        monkeypatch.setattr(cover, name, caps["per_record"])
    for name in ("_COVER_CACHE_ELEMENTS_PER_POINT", "_VARIANT_CACHE_ELEMENTS_PER_POINT"):
        monkeypatch.setattr(cover, name, caps["per_point"])
    bands = _track_band_caches(cover, monkeypatch, keep_alive=True)

    expected_records, expected_stats = _final_records(_sheet_reference(cover, overlap, True))
    result = cover._generate_segment_cover_records(_sheet(overlap), CROP_ZYX, overlap=overlap, prune_bboxes=True)

    _assert_same_records(result["final_records"], expected_records)
    assert result["prune_stats_by_band"] == expected_stats
    for band in bands:
        assert len(band.steps) <= band.max_steps == caps["_MIN_CACHE_ENTRIES"]
        assert len(band.objectives) <= caps["_OBJECTIVE_MEMO_MAX_ENTRIES"]
        assert len(band._cover) <= caps["_MIN_CACHE_ENTRIES"]
        assert len(band._variants) <= caps["_MIN_CACHE_ENTRIES"]


def test_band_caches_are_released_as_soon_as_each_band_returns(cover, monkeypatch):
    points = _sheet(0.25)
    bands = _track_band_caches(cover, monkeypatch, keep_alive=False)

    # With the cyclic collector off, only reference counting can free a cache: this fails
    # if a band's tables are reachable from anything that outlives the band, or form a cycle.
    gc.collect()
    gc.disable()
    try:
        for z_band, records, band_points in _band_jobs(cover, points, CROP_ZYX, 0.25):
            before = len(bands)
            _, optimized, prune_stats = cover._optimize_single_band(z_band, records, band_points, True, None, 0.25)
            assert len(bands) == before + 1
            assert bands[-1]() is None
            assert prune_stats["attempts"] > 0
    finally:
        gc.enable()

    # The public band functions build and drop their own cache the same way.
    z_band, records, band_points = _band_jobs(cover, points, CROP_ZYX, 0.25)[0]
    optimized = cover._optimize_band_bboxes(records, band_points, allow_neighbor_overlap=True, require_neighbor_overlap=True)
    cover._prune_band_bboxes_via_reopt(optimized, band_points, allow_neighbor_overlap=True, require_neighbor_overlap=True)
    assert len(bands) >= 4 and all(ref() is None for ref in bands)


# ---------------------------------------------------------------------------
# Command line and worker processes
# ---------------------------------------------------------------------------


def _run_cli(cover, tmp_path, name, extra_args):
    output_json = tmp_path / f"{name}.json"
    argv = [
        "--tifxyz-path", str(tmp_path / "segment.tifxyz"),
        "--crop-size", *[str(v) for v in CROP_ZYX],
        "--output-json", str(output_json),
        *extra_args,
    ]
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        cover.main(argv)
    return json.loads(output_json.read_text()), stdout.getvalue()


def test_cli_output_is_the_same_for_one_and_two_band_workers(cover, monkeypatch, tmp_path):
    points = _sheet(0.25)
    # main() loads the segment in the parent process only; the workers receive plain arrays.
    monkeypatch.setattr(cover, "_load_valid_points_zyx", lambda tifxyz_path: points)
    expected_records, expected_stats = _final_records(_sheet_reference(cover, 0.25, True))
    assert len(expected_stats) >= 2, "band workers are only used when there are several bands"

    payloads = {}
    for workers in (1, 2):
        payload, stdout = _run_cli(
            cover, tmp_path, f"workers{workers}", ["--overlap", "0.25", "--prune-bboxes", "--band-workers", str(workers)]
        )
        assert f"band_workers: {workers}" in stdout
        assert payload["meta"].pop("band_workers") == workers
        payloads[workers] = payload

    assert payloads[1] == payloads[2]
    assert payloads[1]["bboxes"] == [cover._serialize_bbox_record(item) for item in expected_records]
    assert payloads[1]["prune_stats_by_band"] == {str(z_band): stats for z_band, stats in expected_stats.items()}
    assert payloads[1]["meta"]["prune_removed_total"] == sum(stats["removed"] for stats in expected_stats.values())
    assert payloads[1]["meta"]["final_bbox_count"] == len(expected_records)


def test_generate_segment_cover_records_with_band_workers_matches_reference(cover):
    expected_records, expected_stats = _final_records(_sheet_reference(cover, 0.5, True))

    result = cover._generate_segment_cover_records(
        _sheet(0.5), CROP_ZYX, overlap=0.5, prune_bboxes=True, band_workers=2
    )

    _assert_same_records(result["final_records"], expected_records)
    assert result["prune_stats_by_band"] == expected_stats
    assert result["band_workers"] == 2


def test_cli_parses_its_arguments_and_prints_help(cover, capsys):
    # The --overlap help text contains a percent sign; argparse must be given it escaped.
    args = cover.parse_args(["--tifxyz-path", "s.tifxyz", "--crop-size", "8", "8", "8", "--output-json", "o.json"])
    assert args.overlap == 0.0 and args.band_workers == 1 and args.prune_bboxes is False

    with pytest.raises(SystemExit) as excinfo:
        cover.parse_args(["--help"])
    assert excinfo.value.code == 0
    assert "0.25 means 25% overlap" in " ".join(capsys.readouterr().out.split())
