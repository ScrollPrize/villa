import json
import os

import numpy as np

from vesuvius.neural_tracing.tifxyz import save_tifxyz


def _resolve_extrapolation_settings(args, runtime_config):
    cfg = runtime_config or {}

    method = args.extrapolation_method
    if method is None:
        method = str(cfg.get("extrapolation_method", "rbf"))

    degrade_prob = float(cfg.get("extrap_degrade_prob", 0.0))

    def _pair_from_cfg(key, default_pair):
        val = cfg.get(key, default_pair)
        if isinstance(val, (list, tuple)) and len(val) == 2:
            return (float(val[0]), float(val[1]))
        return (float(default_pair[0]), float(default_pair[1]))

    degrade_curvature_range = _pair_from_cfg("extrap_degrade_curvature_range", (0.001, 0.01))
    degrade_gradient_range = _pair_from_cfg("extrap_degrade_gradient_range", (0.05, 0.2))

    method_kwargs = {}
    if method in {"rbf", "rbf_edge_only"}:
        rbf_downsample = int(cfg.get("rbf_downsample_factor", 2))
        edge_downsample_cfg = cfg.get("rbf_edge_downsample_factor", None)
        edge_downsample = rbf_downsample if edge_downsample_cfg is None else int(edge_downsample_cfg)
        method_kwargs["downsample_factor"] = edge_downsample if method == "rbf_edge_only" else rbf_downsample
        method_kwargs["rbf_max_points"] = cfg.get("rbf_max_points")

    if method == "rbf_edge_only":
        method_kwargs["edge_band_frac"] = float(cfg.get("rbf_edge_band_frac", 0.10))
        method_kwargs["edge_band_cells"] = cfg.get("rbf_edge_band_cells")
        method_kwargs["edge_min_points"] = int(cfg.get("rbf_edge_min_points", 128))

    # Keep args aligned with resolved settings for existing debug prints/metadata.
    args.extrapolation_method = method

    return {
        "method": method,
        "degrade_prob": float(degrade_prob),
        "degrade_curvature_range": degrade_curvature_range,
        "degrade_gradient_range": degrade_gradient_range,
        "method_kwargs": method_kwargs,
    }


def resolve_tifxyz_params(args, model_config, volume_scale):
    tifxyz_step_size = args.tifxyz_step_size
    tifxyz_voxel_size_um = args.tifxyz_voxel_size_um

    if tifxyz_step_size is None:
        if model_config is not None:
            tifxyz_step_size = model_config.get(
                "step_size",
                model_config.get("heatmap_step_size", 10),
            )
        else:
            tifxyz_step_size = 10
    if tifxyz_voxel_size_um is None:
        meta_path = os.path.join(args.volume_path, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "rt") as meta_fp:
                tifxyz_voxel_size_um = json.load(meta_fp).get("voxelsize", None)
    if tifxyz_voxel_size_um is None:
        tifxyz_voxel_size_um = 8.24
    tifxyz_step_size = int(round(float(tifxyz_step_size) * (2 ** volume_scale)))
    return tifxyz_step_size, tifxyz_voxel_size_um


def _validate_named_method(merge_method, valid_methods, method_label):
    method = str(merge_method).strip().lower()
    if method not in valid_methods:
        raise ValueError(
            f"Unknown {method_label} '{merge_method}'. "
            f"Supported methods: {list(valid_methods)}"
        )
    return method


def save_tifxyz_output(
    args,
    tgt_segment,
    pred_samples,
    tifxyz_uuid,
    tifxyz_step_size,
    tifxyz_voxel_size_um,
    checkpoint_path,
    cond_direction,
    grow_direction,
    volume_scale,
    aggregate_pred_samples_to_uv_grid,
    overlap_merge_method="mean",
):
    tgt_segment.use_full_resolution()
    full_zyxs = tgt_segment.get_zyxs(stored_resolution=False)

    h_full, w_full = full_zyxs.shape[:2]
    pred_grid, pred_valid, (uv_r_min, uv_c_min) = aggregate_pred_samples_to_uv_grid(
        pred_samples,
        base_uv_bounds=(0, 0, h_full - 1, w_full - 1),
        overlap_merge_method=overlap_merge_method,
    )
    full_pred_zyxs = np.full_like(pred_grid, -1.0, dtype=np.float32)
    r_off = -uv_r_min
    c_off = -uv_c_min
    full_pred_zyxs[r_off:r_off + h_full, c_off:c_off + w_full] = full_zyxs
    full_pred_zyxs[pred_valid] = pred_grid[pred_valid]

    scale_factor = 2 ** volume_scale
    if scale_factor != 1:
        full_pred_zyxs_out = np.where(
            (full_pred_zyxs == -1).all(axis=-1, keepdims=True),
            -1.0,
            full_pred_zyxs * scale_factor,
        )
    else:
        full_pred_zyxs_out = full_pred_zyxs

    # Downsample grid to the correct tifxyz density (step_size spacing in full-res UV).
    current_step_y = int(round(2 ** volume_scale))
    current_step_x = int(round(2 ** volume_scale))

    stride_y = int(round(float(tifxyz_step_size) / max(1, current_step_y)))
    stride_x = int(round(float(tifxyz_step_size) / max(1, current_step_x)))
    if stride_y > 1 or stride_x > 1:
        full_pred_zyxs_out = full_pred_zyxs_out[::max(1, stride_y), ::max(1, stride_x)]

    save_tifxyz(
        full_pred_zyxs_out,
        args.tifxyz_out_dir,
        tifxyz_uuid,
        step_size=tifxyz_step_size,
        voxel_size_um=tifxyz_voxel_size_um,
        source=str(checkpoint_path),
        additional_metadata={
            "grow_direction": grow_direction,
            "cond_direction": cond_direction,
            "extrapolation_method": args.extrapolation_method,
            "refine_steps": None if args.refine is None else int(args.refine) + 1,
        }
    )
    print(f"Saved tifxyz to {os.path.join(args.tifxyz_out_dir, tifxyz_uuid)}")
