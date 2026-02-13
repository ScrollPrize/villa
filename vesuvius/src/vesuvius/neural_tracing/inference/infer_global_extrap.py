import argparse
import colorsys

import numpy as np
import zarr

from vesuvius.image_proc.intensity.normalization import normalize_zscore
from vesuvius.neural_tracing.inference.common import (
    _aggregate_pred_samples_to_uv_grid,
    _resolve_extrapolation_settings,
)
from vesuvius.neural_tracing.inference.infer_rowcol_split import (
    _bbox_to_min_corner_and_bounds_array,
    _build_uv_grid,
    _build_uv_query_from_cond_points,
    _crop_volume_from_min_corner,
    _get_growth_context,
    _grid_in_bounds_mask,
    _initialize_window_state,
    _load_optional_json,
    _resolve_segment_volume,
    _stored_to_full_bounds,
    compute_edge_one_shot_extrapolation,
    compute_extrapolation_infer,
    compute_window_and_split,
    get_cond_edge_bboxes,
    load_checkpoint_config,
    setup_segment,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare per-bbox extrapolation vs one-shot edge extrapolation."
    )
    parser.add_argument("--tifxyz-path", type=str, required=True)
    parser.add_argument("--volume-path", type=str, required=True)
    parser.add_argument("--volume-scale", type=int, default=1)
    parser.add_argument("--grow-direction", type=str, required=True, choices=["left", "right", "up", "down"])
    parser.add_argument("--cond-pct", type=float, default=0.50)
    parser.add_argument("--crop-size", type=int, nargs=3, default=[128, 128, 128])
    parser.add_argument("--window-pad", type=int, default=10)
    parser.add_argument("--bbox-overlap-frac", type=float, default=0.15)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--config-path", type=str, default=None)
    parser.add_argument("--extrapolation-method", type=str, default=None)
    parser.add_argument(
        "--edge-input-rowscols",
        type=int,
        required=True,
        help="Number of edge rows/cols from conditioning region to use in one-shot extrapolation.",
    )
    parser.add_argument("--napari", action="store_true", help="Visualize bbox/full-edge conditioning and extrapolation.")
    args = parser.parse_args()

    if args.edge_input_rowscols < 1:
        parser.error("--edge-input-rowscols must be >= 1")
    if args.bbox_overlap_frac < 0.0 or args.bbox_overlap_frac >= 1.0:
        parser.error("--bbox-overlap-frac must be in [0, 1)")
    return args


def _resolve_settings(args):
    runtime_config = {}
    if args.checkpoint_path:
        model_config, _ = load_checkpoint_config(args.checkpoint_path)
        runtime_config.update(model_config)
    runtime_config.update(_load_optional_json(args.config_path))
    return _resolve_extrapolation_settings(args, runtime_config)


def _finite_uv_world(uv, world):
    if uv is None or world is None:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0, 3), dtype=np.float32)
    uv = np.asarray(uv)
    world = np.asarray(world)
    if uv.size == 0 or world.size == 0:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0, 3), dtype=np.float32)
    keep = np.isfinite(world).all(axis=1)
    if not keep.any():
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0, 3), dtype=np.float32)
    return uv[keep].astype(np.float64, copy=False), world[keep].astype(np.float32, copy=False)


def _build_bbox_crops(
    bboxes,
    tgt_segment,
    volume_scale,
    cond_zyxs,
    cond_valid,
    uv_cond,
    grow_direction,
    crop_size,
    extrapolation_settings,
    cond_pct,
):
    cond_direction, _ = _get_growth_context(grow_direction)
    cond_valid_base = np.asarray(cond_valid, dtype=bool)
    crop_size = tuple(int(v) for v in crop_size)
    volume_for_crops = _resolve_segment_volume(tgt_segment, volume_scale=volume_scale)

    uv_global_ids = {}
    next_global_id = 0
    bbox_crops = []

    def _assign_global_ids(uv_flat):
        nonlocal next_global_id
        out = np.zeros((uv_flat.shape[0],), dtype=np.int64)
        for i in range(uv_flat.shape[0]):
            key = (int(uv_flat[i, 0]), int(uv_flat[i, 1]))
            gid = uv_global_ids.get(key)
            if gid is None:
                gid = next_global_id
                uv_global_ids[key] = gid
                next_global_id += 1
            out[i] = gid
        return out

    for bbox_idx, bbox in enumerate(bboxes):
        min_corner, bbox_bounds = _bbox_to_min_corner_and_bounds_array(bbox)
        vol_crop = _crop_volume_from_min_corner(volume_for_crops, min_corner, crop_size)
        vol_crop = normalize_zscore(vol_crop)

        cond_grid_local = cond_zyxs.astype(np.float64, copy=False) - min_corner[None, None, :]
        cond_grid_valid = cond_valid_base.copy()
        cond_grid_valid &= _grid_in_bounds_mask(cond_grid_local, crop_size)

        cond_uv = uv_cond[cond_grid_valid].astype(np.float64, copy=False)
        cond_world = cond_zyxs[cond_grid_valid].astype(np.float32, copy=False)
        cond_local = cond_grid_local[cond_grid_valid].astype(np.float32, copy=False)
        uv_query = _build_uv_query_from_cond_points(cond_uv, grow_direction, cond_pct)
        uv_query_flat = uv_query.reshape(-1, 2).astype(np.float64, copy=False)
        query_global_ids = _assign_global_ids(uv_query_flat)

        extrap_local = np.zeros((0, 3), dtype=np.float32)
        extrap_uv = np.zeros((0, 2), dtype=np.float64)
        extrap_world = np.zeros((0, 3), dtype=np.float32)
        extrap_global_ids = np.zeros((0,), dtype=np.int64)
        extrap_valid_mask = np.zeros((uv_query_flat.shape[0],), dtype=bool)
        if uv_query.size > 0 and len(cond_uv) > 0:
            extrap_result = compute_extrapolation_infer(
                uv_cond=cond_uv,
                zyx_cond=cond_world,
                uv_query=uv_query,
                min_corner=min_corner,
                crop_size=crop_size,
                method=extrapolation_settings["method"],
                cond_direction=cond_direction,
                degrade_prob=extrapolation_settings["degrade_prob"],
                degrade_curvature_range=extrapolation_settings["degrade_curvature_range"],
                degrade_gradient_range=extrapolation_settings["degrade_gradient_range"],
                skip_bounds_check=True,
                **extrapolation_settings["method_kwargs"],
            )
            if extrap_result is not None:
                extrap_local_full = np.asarray(extrap_result["extrap_coords_local"], dtype=np.float32)
                if extrap_local_full.shape[0] != uv_query_flat.shape[0]:
                    raise ValueError(
                        f"bbox {bbox_idx} extrapolation count mismatch: "
                        f"{extrap_local_full.shape[0]} coords for {uv_query_flat.shape[0]} UV queries"
                    )
                extrap_world_full = extrap_local_full + min_corner[None, :].astype(np.float32)
                extrap_valid_mask = np.isfinite(extrap_world_full).all(axis=1)
                extrap_local = extrap_local_full[extrap_valid_mask].astype(np.float32, copy=False)
                extrap_uv = uv_query_flat[extrap_valid_mask].astype(np.float64, copy=False)
                extrap_world = extrap_world_full[extrap_valid_mask].astype(np.float32, copy=False)
                extrap_global_ids = query_global_ids[extrap_valid_mask]

        bbox_crops.append(
            {
                "bbox_idx": bbox_idx,
                "bbox": bbox,
                "bbox_bounds": bbox_bounds,
                "min_corner": min_corner.astype(np.int64, copy=False),
                "crop_size": np.asarray(crop_size, dtype=np.int64),
                "volume": vol_crop,
                "cond_grid_valid": cond_grid_valid,
                "cond_uv_rc": cond_uv,
                "cond_world_zyx": cond_world,
                "cond_local_zyx": cond_local,
                "query_uv_rc": uv_query_flat,
                "query_global_ids": query_global_ids,
                "extrap_valid_mask": extrap_valid_mask,
                "extrap_uv_rc": extrap_uv,
                "extrap_global_ids": extrap_global_ids,
                "extrap_world_zyx": extrap_world,
                "extrap_local_zyx": extrap_local,
            }
        )

    return bbox_crops


def _build_bbox_results_from_crops(bbox_crops):
    bbox_results = []
    pred_samples = []
    for crop in bbox_crops:
        extrap_uv = crop["extrap_uv_rc"]
        extrap_world = crop["extrap_world_zyx"]
        if len(extrap_uv) > 0:
            pred_samples.append((extrap_uv, extrap_world))
        bbox_results.append(
            {
                "bbox_idx": crop["bbox_idx"],
                "bbox": crop["bbox"],
                "cond_world": crop["cond_world_zyx"],
                "cond_uv": crop["cond_uv_rc"],
                "query_uv": crop["query_uv_rc"],
                "extrap_world": extrap_world,
                "extrap_uv": extrap_uv,
            }
        )
    return bbox_results, pred_samples


def _print_bbox_crop_debug_table(bbox_crops):
    if not bbox_crops:
        print("== BBox Crop Debug ==")
        print("No bbox crops.")
        return

    global_id_counts = {}
    for crop in bbox_crops:
        gids = np.asarray(crop.get("query_global_ids", np.zeros((0,), dtype=np.int64)), dtype=np.int64)
        for gid in gids:
            key = int(gid)
            global_id_counts[key] = global_id_counts.get(key, 0) + 1

    headers = ("bbox", "n_cond", "n_query", "n_extrap", "n_nonfinite", "qids_shared")
    rows = []
    for crop in bbox_crops:
        bbox_idx = int(crop["bbox_idx"])
        n_cond = int(np.asarray(crop.get("cond_uv_rc", np.zeros((0, 2)))).shape[0])
        query_ids = np.asarray(crop.get("query_global_ids", np.zeros((0,), dtype=np.int64)), dtype=np.int64)
        n_query = int(query_ids.shape[0])
        extrap_mask = np.asarray(crop.get("extrap_valid_mask", np.zeros((0,), dtype=bool)), dtype=bool)
        n_extrap = int(np.asarray(crop.get("extrap_uv_rc", np.zeros((0, 2)))).shape[0])
        n_nonfinite = int(max(n_query - int(extrap_mask.sum()), 0))
        n_shared = int(sum(1 for gid in query_ids if global_id_counts.get(int(gid), 0) > 1))
        rows.append((bbox_idx, n_cond, n_query, n_extrap, n_nonfinite, n_shared))

    widths = []
    for i, header in enumerate(headers):
        cell_width = max(len(header), *(len(str(row[i])) for row in rows))
        widths.append(cell_width)

    def _fmt(row):
        return " | ".join(str(row[i]).rjust(widths[i]) for i in range(len(headers)))

    print("== BBox Crop Debug ==")
    print(_fmt(headers))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(_fmt(row))

    total_q = int(sum(row[2] for row in rows))
    total_shared = int(sum(row[5] for row in rows))
    unique_qids = int(len(global_id_counts))
    shared_qids = int(sum(1 for c in global_id_counts.values() if c > 1))
    print(
        "totals: "
        f"queries={total_q}, unique_query_ids={unique_qids}, "
        f"shared_query_entries={total_shared}, shared_query_ids={shared_qids}"
    )


def _grid_to_uv_world_dict(grid, valid, offset):
    rows, cols = np.where(valid)
    if rows.size == 0:
        return {}
    rows_abs = rows.astype(np.int64) + int(offset[0])
    cols_abs = cols.astype(np.int64) + int(offset[1])
    pts = grid[rows, cols]
    out = {}
    for i in range(rows_abs.shape[0]):
        if np.isfinite(pts[i]).all():
            out[(int(rows_abs[i]), int(cols_abs[i]))] = pts[i].astype(np.float32, copy=False)
    return out


def _slice_one_shot_map_by_bbox_intersection(one_map, bbox_results):
    if not one_map:
        return {}

    one_pts = np.asarray(list(one_map.values()), dtype=np.float32)
    finite = np.isfinite(one_pts).all(axis=1)

    sliced = {}
    for item in bbox_results:
        idx = int(item["bbox_idx"])
        z_min, z_max, y_min, y_max, x_min, x_max = item["bbox"]
        in_bbox = (
            finite &
            (one_pts[:, 0] >= z_min) & (one_pts[:, 0] <= z_max) &
            (one_pts[:, 1] >= y_min) & (one_pts[:, 1] <= y_max) &
            (one_pts[:, 2] >= x_min) & (one_pts[:, 2] <= x_max)
        )
        if in_bbox.any():
            sliced[idx] = one_pts[in_bbox]
    return sliced


def _compute_safe_band(bbox_results, one_shot, one_map, grow_direction):
    one_uv = np.asarray(one_shot.get("uv_query_flat", np.zeros((0, 2), dtype=np.float64)))
    if one_uv.size == 0:
        return {
            "axis_name": None,
            "total_lines": 0,
            "safe_lines": 0,
            "safe_uv_set": set(),
            "near_axis_value": None,
            "farthest_safe_axis_value": None,
            "first_unsafe_axis_value": None,
            "first_unsafe_covered": 0,
            "first_unsafe_total": 0,
        }

    one_uv_i = one_uv.astype(np.int64, copy=False)

    bbox_bounds = []
    for item in bbox_results:
        z_min, z_max, y_min, y_max, x_min, x_max = item["bbox"]
        bbox_bounds.append(
            (
                float(z_min), float(z_max),
                float(y_min), float(y_max),
                float(x_min), float(x_max),
            )
        )
    bbox_bounds = tuple(bbox_bounds)

    def _in_bbox_union(pt):
        z, y, x = float(pt[0]), float(pt[1]), float(pt[2])
        for z_min, z_max, y_min, y_max, x_min, x_max in bbox_bounds:
            if (
                z >= z_min and z <= z_max and
                y >= y_min and y <= y_max and
                x >= x_min and x <= x_max
            ):
                return True
        return False

    # UVs that have finite one-shot predictions inside the 3D union of bbox bounds.
    uv_supported = set()
    for uv_key, pt in one_map.items():
        if np.isfinite(pt).all() and _in_bbox_union(pt):
            uv_supported.add((int(uv_key[0]), int(uv_key[1])))

    axis_idx = 1 if grow_direction in {"left", "right"} else 0
    axis_name = "col" if axis_idx == 1 else "row"
    near_to_far_desc = grow_direction in {"left", "up"}

    line_orth_values = {}
    for row, col in one_uv_i:
        axis_val = int(col) if axis_idx == 1 else int(row)
        orth_val = int(row) if axis_idx == 1 else int(col)
        line_orth_values.setdefault(axis_val, set()).add(orth_val)

    axis_values = sorted(line_orth_values.keys(), reverse=near_to_far_desc)
    if len(axis_values) == 0:
        return {
            "axis_name": axis_name,
            "total_lines": 0,
            "safe_lines": 0,
            "safe_uv_set": set(),
            "near_axis_value": None,
            "farthest_safe_axis_value": None,
            "first_unsafe_axis_value": None,
            "first_unsafe_covered": 0,
            "first_unsafe_total": 0,
        }

    safe_axis_values = []
    first_unsafe_axis_value = None
    first_unsafe_covered = 0
    first_unsafe_total = 0

    for axis_val in axis_values:
        orth_values = line_orth_values[axis_val]
        covered = 0
        for orth_val in orth_values:
            uv_key = (orth_val, axis_val) if axis_idx == 1 else (axis_val, orth_val)
            if uv_key in uv_supported:
                covered += 1
        if covered == len(orth_values):
            safe_axis_values.append(axis_val)
        else:
            first_unsafe_axis_value = axis_val
            first_unsafe_covered = int(covered)
            first_unsafe_total = int(len(orth_values))
            break

    safe_uv_set = set()
    for axis_val in safe_axis_values:
        for orth_val in line_orth_values[axis_val]:
            uv_key = (orth_val, axis_val) if axis_idx == 1 else (axis_val, orth_val)
            safe_uv_set.add(uv_key)

    return {
        "axis_name": axis_name,
        "total_lines": int(len(axis_values)),
        "safe_lines": int(len(safe_axis_values)),
        "safe_uv_set": safe_uv_set,
        "near_axis_value": int(axis_values[0]),
        "farthest_safe_axis_value": None if len(safe_axis_values) == 0 else int(safe_axis_values[-1]),
        "first_unsafe_axis_value": None if first_unsafe_axis_value is None else int(first_unsafe_axis_value),
        "first_unsafe_covered": first_unsafe_covered,
        "first_unsafe_total": first_unsafe_total,
    }


def _print_comparison(bbox_results, one_shot, bbox_map, one_map, safe_band):
    bbox_uv = set(bbox_map.keys())
    one_uv = set(one_map.keys())
    common_uv = sorted(bbox_uv & one_uv)

    print("== Compare Extrapolation ==")
    print(f"bboxes: {len(bbox_results)}")
    print(f"bbox extrap uv count (aggregated): {len(bbox_uv)}")
    print(f"one-shot edge-input uv count: {len(one_shot.get('edge_uv', []))}")
    print(f"one-shot extrap uv count (aggregated): {len(one_uv)}")
    print(f"uv overlap count: {len(common_uv)}")
    print("== Safe Band ==")
    print(f"safe axis: {safe_band.get('axis_name')}")
    print(f"safe lines (near->far contiguous): {safe_band.get('safe_lines', 0)}/{safe_band.get('total_lines', 0)}")
    print(f"safe uv count: {len(safe_band.get('safe_uv_set', set()))}")
    near_axis = safe_band.get("near_axis_value")
    far_axis = safe_band.get("farthest_safe_axis_value")
    if near_axis is not None:
        print(f"safe axis range near->far: {near_axis} -> {far_axis}")
    first_unsafe = safe_band.get("first_unsafe_axis_value")
    if first_unsafe is not None:
        covered = int(safe_band.get("first_unsafe_covered", 0))
        total = int(safe_band.get("first_unsafe_total", 0))
        print(f"first unsafe axis value: {first_unsafe} ({covered}/{total} covered)")

    if len(common_uv) == 0:
        print("No overlapping UVs between methods; cannot compute coordinate deltas.")
        return

    deltas = np.asarray(
        [
            np.linalg.norm(bbox_map[uv] - one_map[uv])
            for uv in common_uv
        ],
        dtype=np.float64,
    )
    print(f"delta L2 mean: {deltas.mean():.4f}")
    print(f"delta L2 median: {np.median(deltas):.4f}")
    print(f"delta L2 p95: {np.percentile(deltas, 95):.4f}")
    print(f"delta L2 max: {deltas.max():.4f}")


def _show_napari(cond_zyxs, cond_valid, bbox_results, one_shot, bbox_map, one_map, safe_band):
    try:
        import napari
    except Exception as exc:
        raise RuntimeError("--napari was set, but napari is not available.") from exc

    viewer = napari.Viewer(ndisplay=3)
    one_shot_by_bbox = _slice_one_shot_map_by_bbox_intersection(one_map, bbox_results)

    cond_full = cond_zyxs[cond_valid]
    if cond_full.size > 0:
        viewer.add_points(cond_full, name="cond_full", size=1, face_color=[0.7, 0.7, 0.7], opacity=0.2)

    n_bbox = max(len(bbox_results), 1)
    for item in bbox_results:
        idx = int(item["bbox_idx"])
        rgb = colorsys.hsv_to_rgb((idx / n_bbox) % 1.0, 1.0, 1.0)
        z_min, z_max, y_min, y_max, x_min, x_max = item["bbox"]
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
        # Draw 12 cuboid edges as 3D path segments so bbox extents are visible in napari.
        edge_pairs = (
            (0, 1), (0, 2), (0, 4),
            (1, 3), (1, 5),
            (2, 3), (2, 6),
            (3, 7),
            (4, 5), (4, 6),
            (5, 7),
            (6, 7),
        )
        bbox_edges = [corners[[i0, i1]] for (i0, i1) in edge_pairs]
        viewer.add_shapes(
            bbox_edges,
            shape_type="path",
            edge_color=[*rgb, 0.9],
            edge_width=1,
            face_color="transparent",
            name=f"bbox_{idx:03d}_wire",
            opacity=0.9,
        )
        if item["cond_world"].size > 0:
            viewer.add_points(
                item["cond_world"],
                name=f"bbox_{idx:03d}_cond",
                size=1,
                face_color=list(rgb),
                opacity=0.6,
            )
        if item["extrap_world"].size > 0:
            viewer.add_points(
                item["extrap_world"],
                name=f"bbox_{idx:03d}_extrap",
                size=1,
                face_color=list(rgb),
                symbol="ring",
                opacity=0.9,
            )
        one_bbox_pts = one_shot_by_bbox.get(idx)
        if one_bbox_pts is not None and len(one_bbox_pts) > 0:
            viewer.add_points(
                one_bbox_pts,
                name=f"bbox_{idx:03d}_one_shot_slice",
                size=1,
                face_color=list(rgb),
                opacity=0.9,
            )

    edge_cond = one_shot.get("edge_zyx") if one_shot is not None else None
    if edge_cond is not None and len(edge_cond) > 0:
        viewer.add_points(
            edge_cond,
            name="one_shot_edge_cond",
            size=2,
            face_color=[1.0, 1.0, 0.0],
            opacity=0.9,
        )

    if bbox_map:
        bbox_pts = np.asarray(list(bbox_map.values()), dtype=np.float32)
        viewer.add_points(
            bbox_pts,
            name="bbox_agg_extrap",
            size=2,
            face_color=[0.2, 1.0, 0.2],
            opacity=0.6,
        )

    if one_map:
        one_pts = np.asarray(list(one_map.values()), dtype=np.float32)
        viewer.add_points(
            one_pts,
            name="one_shot_agg_extrap",
            size=2,
            face_color=[1.0, 0.4, 0.0],
            opacity=0.6,
        )
    safe_uv_set = safe_band.get("safe_uv_set", set()) if isinstance(safe_band, dict) else set()
    if one_map and safe_uv_set:
        safe_pts = np.asarray(
            [one_map[uv] for uv in safe_uv_set if uv in one_map and np.isfinite(one_map[uv]).all()],
            dtype=np.float32,
        )
        if safe_pts.size > 0:
            viewer.add_points(
                safe_pts,
                name="one_shot_safe_band",
                size=2,
                face_color=[0.0, 1.0, 1.0],
                opacity=0.8,
            )

    def _default_visible(layer_name):
        if layer_name in {"cond_full", "one_shot_edge_cond", "one_shot_safe_band"}:
            return True
        if layer_name.startswith("bbox_") and (
            layer_name.endswith("_wire") or layer_name.endswith("_one_shot_slice")
        ):
            return True
        return False

    # Keep all layers available but show only the requested subset by default.
    for layer in viewer.layers:
        layer.visible = _default_visible(layer.name)

    napari.run()


def main():
    args = parse_args()
    crop_size = tuple(int(v) for v in args.crop_size)

    extrapolation_settings = _resolve_settings(args)

    volume = zarr.open_group(args.volume_path, mode="r")
    tgt_segment, stored_zyxs, valid_s, grow_direction, h_s, w_s = setup_segment(args, volume)
    cond_direction, _ = _get_growth_context(grow_direction)

    r0_s, r1_s, c0_s, c1_s = compute_window_and_split(
        args, stored_zyxs, valid_s, grow_direction, h_s, w_s, crop_size
    )
    full_bounds = _stored_to_full_bounds(tgt_segment, (r0_s, r1_s, c0_s, c1_s))
    cond_zyxs, cond_valid, cond_uv_offset = _initialize_window_state(tgt_segment, full_bounds)
    uv_cond = _build_uv_grid(cond_uv_offset, cond_zyxs.shape[:2])

    bboxes, _ = get_cond_edge_bboxes(
        cond_zyxs,
        cond_direction,
        crop_size,
        overlap_frac=args.bbox_overlap_frac,
    )

    bbox_crops = _build_bbox_crops(
        bboxes=bboxes,
        tgt_segment=tgt_segment,
        volume_scale=args.volume_scale,
        cond_zyxs=cond_zyxs,
        cond_valid=cond_valid,
        uv_cond=uv_cond,
        grow_direction=grow_direction,
        crop_size=crop_size,
        extrapolation_settings=extrapolation_settings,
        cond_pct=args.cond_pct,
    )
    _print_bbox_crop_debug_table(bbox_crops)
    bbox_results, bbox_samples = _build_bbox_results_from_crops(bbox_crops)

    one_shot = compute_edge_one_shot_extrapolation(
        cond_zyxs=cond_zyxs,
        cond_valid=cond_valid,
        uv_cond=uv_cond,
        grow_direction=grow_direction,
        edge_input_rowscols=args.edge_input_rowscols,
        cond_pct=args.cond_pct,
        method=extrapolation_settings["method"],
        min_corner=np.zeros(3, dtype=np.float64),
        crop_size=crop_size,
        degrade_prob=extrapolation_settings["degrade_prob"],
        degrade_curvature_range=extrapolation_settings["degrade_curvature_range"],
        degrade_gradient_range=extrapolation_settings["degrade_gradient_range"],
        skip_bounds_check=True,
        **extrapolation_settings["method_kwargs"],
    )
    if one_shot is None:
        one_shot = {
            "edge_uv": np.zeros((0, 2), dtype=np.float64),
            "edge_zyx": np.zeros((0, 3), dtype=np.float32),
            "uv_query_flat": np.zeros((0, 2), dtype=np.float64),
            "extrap_coords_world": np.zeros((0, 3), dtype=np.float32),
        }

    one_uv, one_world = _finite_uv_world(one_shot.get("uv_query_flat"), one_shot.get("extrap_coords_world"))
    one_samples = [(one_uv, one_world)] if len(one_uv) > 0 else []

    bbox_grid, bbox_valid, bbox_offset = _aggregate_pred_samples_to_uv_grid(bbox_samples)
    one_grid, one_valid, one_offset = _aggregate_pred_samples_to_uv_grid(one_samples)

    bbox_map = _grid_to_uv_world_dict(bbox_grid, bbox_valid, bbox_offset)
    one_map = _grid_to_uv_world_dict(one_grid, one_valid, one_offset)
    safe_band = _compute_safe_band(bbox_results, one_shot, one_map, grow_direction)

    _print_comparison(bbox_results, one_shot, bbox_map, one_map, safe_band)

    if args.napari:
        _show_napari(cond_zyxs, cond_valid, bbox_results, one_shot, bbox_map, one_map, safe_band)


if __name__ == "__main__":
    main()
