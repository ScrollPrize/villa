import argparse
import json
from pathlib import Path
import colorsys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from vesuvius.tifxyz import read_tifxyz


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
        help="Initial chunk overlap fraction in [0, 1). Example: 0.25 means 25% overlap.",
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

    surface = read_tifxyz(tifxyz_path)
    surface.use_stored_resolution()
    x, y, z, valid = surface[:]
    points_zyx = np.stack([z, y, x], axis=-1)
    points_zyx = points_zyx[np.asarray(valid, dtype=bool)]
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

    grouped = {}
    for point_idx, point in enumerate(points_zyx):
        zr = _axis_chunk_index_range(point[0], world_min_zyx[0], crop_size_zyx[0], stride_zyx[0], grid_shape[0])
        yr = _axis_chunk_index_range(point[1], world_min_zyx[1], crop_size_zyx[1], stride_zyx[1], grid_shape[1])
        xr = _axis_chunk_index_range(point[2], world_min_zyx[2], crop_size_zyx[2], stride_zyx[2], grid_shape[2])
        if zr is None or yr is None or xr is None:
            continue

        for gz in range(zr[0], zr[1] + 1):
            for gy in range(yr[0], yr[1] + 1):
                for gx in range(xr[0], xr[1] + 1):
                    grouped.setdefault((int(gz), int(gy), int(gx)), []).append(int(point_idx))

    grouped_np = {k: np.asarray(v, dtype=np.int64) for k, v in grouped.items()}
    return grouped_np, grid_shape, stride_zyx


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


def _touching_relation_holds(bbox_a, bbox_b, relation, allow_overlap=False):
    if relation == "right":
        if allow_overlap:
            return _intervals_touch_or_overlap(bbox_a[4], bbox_a[5], bbox_b[4], bbox_b[5])
        return int(bbox_a[5]) + 1 == int(bbox_b[4])
    if relation == "left":
        if allow_overlap:
            return _intervals_touch_or_overlap(bbox_a[4], bbox_a[5], bbox_b[4], bbox_b[5])
        return int(bbox_b[5]) + 1 == int(bbox_a[4])
    if relation == "down":
        if allow_overlap:
            return _intervals_touch_or_overlap(bbox_a[2], bbox_a[3], bbox_b[2], bbox_b[3])
        return int(bbox_a[3]) + 1 == int(bbox_b[2])
    if relation == "up":
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


def _optimize_band_bboxes(records, points_zyx_band, max_passes=10, allow_neighbor_overlap=False):
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
            # Try both contact modes; accept any full-coverage solution.
            for allow_overlap in (True, False):
                try:
                    trial_optimized = _optimize_band_bboxes(
                        trial_records_seeded,
                        points_zyx_band,
                        max_passes=max_passes,
                        allow_neighbor_overlap=allow_overlap,
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


def _optimize_single_band(z_band, band_records, band_points, prune_bboxes, prune_max_remove_per_band):
    optimized = _optimize_band_bboxes(band_records, band_points, max_passes=5)
    prune_stats = None
    if prune_bboxes:
        optimized, prune_stats = _prune_band_bboxes_via_reopt(
            optimized,
            band_points,
            max_passes=5,
            max_remove=prune_max_remove_per_band,
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

    band_jobs = []
    for z_band in sorted(by_band.keys()):
        band_records = by_band[z_band]
        band_point_indices = np.unique(np.concatenate([item["point_indices"] for item in band_records]))
        band_points = points_zyx[band_point_indices]
        band_jobs.append((int(z_band), band_records, band_points))

    if band_workers == 1 or len(band_jobs) <= 1:
        band_results = [
            _optimize_single_band(z_band, band_records, band_points, prune_bboxes, prune_max_remove_per_band)
            for z_band, band_records, band_points in band_jobs
        ]
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
                )
                for z_band, band_records, band_points in band_jobs
            ]
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
