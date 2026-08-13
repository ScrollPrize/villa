from typing import Any, MutableMapping

import numpy as np

from vesuvius.neural_tracing.datasets.growth_direction import (
    growth_direction_channel_count,
)


def setdefault_rowcol_cond_dataset_config(config: MutableMapping[str, Any]) -> None:
    """Populate default config values for the row/col conditioning dataset."""
    config.setdefault("training_mode", "rowcol_hidden")
    config.setdefault("cond_percent", [0.1, 0.5])
    config.setdefault("lambda_velocity_dir", 0.1)
    config.setdefault("trace_target_dilation_radius", 1.0)
    config.setdefault("trace_surface_attract_radius", 0.0)
    config.setdefault("lambda_trace_validity", 0.0)
    config.setdefault("trace_validity_positive_radius", 2.0)
    config.setdefault("trace_validity_negative_radius", 3.0)
    config.setdefault("trace_validity_margin", 3.0)
    config.setdefault("trace_validity_background_weight", 0.25)
    config.setdefault("trace_validity_pos_weight", 1.0)
    config.setdefault("force_recompute_patches", False)

    config.setdefault("validate_result_tensors", False)
    config.setdefault("profile_create_split_masks", False)

    # Dense front/back displacement defaults for training_mode="copy_neighbors".
    config.setdefault("use_triplet_direction_priors", True)
    config.setdefault("triplet_direction_prior_mask", "cond")
    config.setdefault("triplet_random_channel_swap_prob", 0.5)
    config.setdefault("triplet_dense_weight_mode", "all")
    config.setdefault("triplet_gt_vector_dilation_radius", 1.0)
    config.setdefault("triplet_band_padding_voxels", 0.0)
    config.setdefault("triplet_band_distance_percentile", 100.0)
    config.setdefault("triplet_band_boost_weight", 2.0)
    config.setdefault("triplet_edt_bbox_padding_voxels", 16.0)
    config.setdefault("displacement_loss_type", "vector_huber_per_branch")
    config.setdefault("displacement_huber_beta", 3.0)
    config.setdefault("lambda_smooth", 0.0)
    config.setdefault("triplet_min_disp_vox", 1.0)
    config.setdefault("lambda_triplet_min_disp", 0.0)
    config.setdefault("lambda_displaced_source_edt", 0.0)
    config.setdefault("displaced_source_edt_loss_type", "huber")
    config.setdefault("displaced_source_edt_beta", 1.0)
    config.setdefault("displaced_source_edt_max_points", 4096)
    config.setdefault("displaced_source_edt_oob_weight", 1.0)

    # Chunk-first patch-finding defaults.
    config.setdefault("overlap_fraction", 0.0)
    config.setdefault("min_points_per_wrap", 100)
    config.setdefault("scale_normalize_patch_counts", True)
    config.setdefault("patch_count_reference_scale", 0)
    config.setdefault("bbox_pad_2d", 0)
    config.setdefault("terminal_chunk_guard_voxels", 20.0)

    cond_local_perturb = dict(config.get("cond_local_perturb") or {})
    cond_local_perturb.setdefault("enabled", True)
    cond_local_perturb.setdefault("probability", 0.35)
    cond_local_perturb.setdefault("num_blobs", [1, 3])
    cond_local_perturb.setdefault("points_affected", 10)
    cond_local_perturb.setdefault("sigma_fraction_range", [0.04, 0.10])
    cond_local_perturb.setdefault("amplitude_range", [0.25, 1.25])
    cond_local_perturb.setdefault("radius_sigma_mult", 2.5)
    cond_local_perturb.setdefault("max_total_displacement", 6.0)
    config["cond_local_perturb"] = cond_local_perturb


def setdefault_rowcol_cond_trainer_config(config: MutableMapping[str, Any]) -> None:
    """Populate default config values owned by the training loop."""
    config.setdefault("num_iterations", 250000)
    config.setdefault("log_frequency", 100)
    config.setdefault("ckpt_frequency", 5000)
    config.setdefault("grad_clip", 5)
    config.setdefault("learning_rate", 0.01)
    config.setdefault("weight_decay", 3e-5)
    config.setdefault("batch_size", 4)
    config.setdefault("num_workers", 4)
    config.setdefault("val_num_workers", 0)
    config.setdefault("pin_memory", True)
    config.setdefault("non_blocking", True)
    config.setdefault("persistent_workers", False)
    config.setdefault("prefetch_factor", 1)
    config.setdefault("val_prefetch_factor", 1)
    config.setdefault("mixed_precision", "no")
    config.setdefault("grad_acc_steps", 1)
    config.setdefault("val_fraction", 0.1)
    config.setdefault("seed", 0)
    config.setdefault("lambda_velocity_smooth", 0.0)
    config.setdefault("velocity_smooth_normalize", True)
    config.setdefault("lambda_trace_integration", 0.0)
    config.setdefault("trace_integration_steps", 2)
    config.setdefault("trace_integration_step_size", 1.0)
    config.setdefault("trace_integration_max_points", 2048)
    config.setdefault("trace_integration_min_weight", 0.5)
    config.setdefault("trace_integration_detach_steps", False)
    config.setdefault("surface_attract_huber_beta", 5.0)
    config.setdefault("val_batches_per_log", 4)
    config.setdefault("log_at_step_zero", False)
    config.setdefault("ckpt_at_step_zero", False)
    config.setdefault("wandb_resume", False)
    config.setdefault("wandb_resume_mode", "allow")
    config.setdefault("compile_model", True)
    config.setdefault("load_weights_only", False)
    config.setdefault("allow_partial_weight_load", False)
    config.setdefault("verbose", False)


def setdefault_rowcol_cond_model_config(config: MutableMapping[str, Any]) -> None:
    """Populate model input and output defaults for trace-ODE training."""
    config.setdefault("step_count", 1)  # Required by make_model.
    training_mode = str(config.get("training_mode", "rowcol_hidden"))
    if training_mode == "copy_neighbors":
        config["in_channels"] = 8 if bool(config.get("use_triplet_direction_priors", True)) else 2
        config["targets"] = {
            "displacement": {"out_channels": 6, "activation": "none"},
        }
    else:
        config["in_channels"] = 2 + growth_direction_channel_count()
        config["targets"] = {
            "velocity_dir": {"out_channels": 3, "activation": "none"},
            "surface_attract": {"out_channels": 3, "activation": "none"},
            "trace_validity": {"out_channels": 1, "activation": "none"},
        }


def prepare_rowcol_cond_train_config(config: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """Apply defaults and validation for the row/col conditioned trainer path."""
    setdefault_rowcol_cond_dataset_config(config)
    setdefault_rowcol_cond_trainer_config(config)
    setdefault_rowcol_cond_model_config(config)
    resolve_rowcol_cond_scheduler_config(config)
    resolve_rowcol_cond_optimizer_config(config)
    validate_rowcol_cond_dataset_config(config)
    return config


def resolve_rowcol_cond_scheduler_config(
    config: MutableMapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Normalize scheduler config without constructing scheduler objects."""
    scheduler_type = config.setdefault("scheduler", "diffusers_cosine_warmup")
    scheduler_kwargs = dict(config.setdefault("scheduler_kwargs", {}) or {})
    if scheduler_type in {"diffusers_cosine_warmup", "warmup_poly", "cosine_warmup"}:
        scheduler_kwargs.setdefault("warmup_steps", config.get("warmup_steps", 5000))
    config["scheduler_kwargs"] = scheduler_kwargs
    return scheduler_type, scheduler_kwargs


def resolve_rowcol_cond_optimizer_config(
    config: MutableMapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Normalize optimizer config without constructing optimizer objects."""
    optimizer_config = config.setdefault("optimizer", "adamw")
    if isinstance(optimizer_config, dict):
        optimizer_type = optimizer_config.get("name", "adamw")
        optimizer_kwargs = dict(optimizer_config)
        optimizer_kwargs.pop("name", None)
    else:
        optimizer_type = optimizer_config
        optimizer_kwargs = dict(config.setdefault("optimizer_kwargs", {}) or {})
    optimizer_kwargs.setdefault("learning_rate", config.get("learning_rate", 1e-3))
    optimizer_kwargs.setdefault("weight_decay", config.get("weight_decay", 1e-4))
    config["optimizer_kwargs"] = optimizer_kwargs
    return optimizer_type, optimizer_kwargs


def rowcol_cond_training_summary_lines(
    config: MutableMapping[str, Any],
    *,
    loss_config: Any,
    optimizer_type: Any,
    optimizer_kwargs: dict[str, Any],
    scheduler_type: Any,
    scheduler_kwargs: dict[str, Any],
    num_train: int,
    num_val: int,
) -> list[str]:
    """Return the active trace-ODE training configuration summary."""
    if str(config.get("training_mode", "rowcol_hidden")) == "copy_neighbors":
        lines = [
            "\n=== Copy Neighbor Dense Displacement Training Configuration ===",
            f"Input channels: {config['in_channels']}",
            "Output: displacement (6ch, lower/behind + upper/front)",
            f"Direction priors: {bool(config.get('use_triplet_direction_priors', True))}",
            f"Dense weight mode: {config.get('triplet_dense_weight_mode')}",
            (
                f"Displacement loss: {config.get('displacement_loss_type')} "
                f"(beta={config.get('displacement_huber_beta')})"
            ),
            (
                f"Displaced source EDT loss: lambda={config.get('lambda_displaced_source_edt')}, "
                f"type={config.get('displaced_source_edt_loss_type')}, "
                f"beta={config.get('displaced_source_edt_beta')}, "
                f"max_points={config.get('displaced_source_edt_max_points')}"
            ),
            (
                f"Optimizer: {optimizer_type} "
                f"(lr={optimizer_kwargs['learning_rate']}, "
                f"weight_decay={optimizer_kwargs.get('weight_decay', 0)})"
            ),
        ]
        scheduler_details = ", ".join(f"{k}={v}" for k, v in scheduler_kwargs.items())
        scheduler_summary = f"Scheduler: {scheduler_type}"
        if scheduler_details:
            scheduler_summary = f"{scheduler_summary} ({scheduler_details})"
        lines.extend(
            [
                scheduler_summary,
                f"Train samples: {num_train}, Val samples: {num_val}",
                "=======================================================\n",
            ]
        )
        return lines

    lines = [
        "\n=== Trace ODE Training Configuration ===",
        f"Input channels: {config['in_channels']}",
        "Growth direction channels: True",
        "Output: velocity_dir (3ch) + surface_attract (3ch) + trace_validity (1ch)",
        (
            f"Velocity direction loss: lambda={loss_config.lambda_velocity_dir}, "
            f"dilation={config.get('trace_target_dilation_radius')}"
        ),
    ]

    if loss_config.lambda_velocity_smooth > 0.0:
        lines.append(
            f"Velocity smoothness loss: lambda={loss_config.lambda_velocity_smooth}, "
            f"normalize={loss_config.velocity_smooth_normalize}"
        )

    if loss_config.lambda_trace_integration > 0.0:
        lines.append(
            f"Trace integration loss: lambda={loss_config.lambda_trace_integration}, "
            f"steps={loss_config.trace_integration_steps}, "
            f"step_size={loss_config.trace_integration_step_size}, "
            f"max_points={loss_config.trace_integration_max_points}, "
            f"detach_steps={loss_config.trace_integration_detach_steps}"
        )

    lines.extend(
        [
            (
                f"Trace ODE losses: lambda_attract={loss_config.lambda_surface_attract}, "
                f"lambda_validity={loss_config.lambda_trace_validity}, "
                f"dilation={config.get('trace_target_dilation_radius')}, "
                f"attract_mode=trace_band, "
                f"attract_radius={config.get('trace_surface_attract_radius')}"
            ),
            "Trace validity EDT in trainer: True",
            (
                f"Optimizer: {optimizer_type} "
                f"(lr={optimizer_kwargs['learning_rate']}, "
                f"weight_decay={optimizer_kwargs.get('weight_decay', 0)})"
            ),
        ]
    )

    scheduler_details = ", ".join(f"{k}={v}" for k, v in scheduler_kwargs.items())
    scheduler_summary = f"Scheduler: {scheduler_type}"
    if scheduler_details:
        scheduler_summary = f"{scheduler_summary} ({scheduler_details})"
    lines.extend(
        [
            scheduler_summary,
            f"Train samples: {num_train}, Val samples: {num_val}",
            "=================================================\n",
        ]
    )
    return lines


def print_rowcol_cond_training_summary(print_fn, *args, **kwargs) -> None:
    """Print the row/col conditioned trace-ODE training configuration summary."""
    for line in rowcol_cond_training_summary_lines(*args, **kwargs):
        print_fn(line)


def _require_finite(name: str, value: float) -> None:
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")


def _require_finite_range(
    name: str,
    value: float,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> None:
    _require_finite(name, value)
    if min_value is not None and value < min_value:
        raise ValueError(f"{name} must satisfy {min_value} <= value, got {value!r}")
    if max_value is not None and value > max_value:
        raise ValueError(f"{name} must satisfy value <= {max_value}, got {value!r}")


def validate_rowcol_cond_dataset_config(config: MutableMapping[str, Any]) -> None:
    """Validate row/col conditioning dataset config invariants."""
    training_mode = str(config.get("training_mode", "rowcol_hidden"))
    if training_mode not in {"rowcol_hidden", "copy_neighbors"}:
        raise ValueError(f"training_mode must be 'rowcol_hidden' or 'copy_neighbors', got {training_mode!r}")

    trace_target_dilation_radius = float(config.get("trace_target_dilation_radius", 1.0))
    _require_finite_range("trace_target_dilation_radius", trace_target_dilation_radius, min_value=0.0)

    trace_surface_attract_radius = float(config.get("trace_surface_attract_radius", 0.0))
    _require_finite_range("trace_surface_attract_radius", trace_surface_attract_radius, min_value=0.0)

    trace_validity_positive_radius = float(config.get("trace_validity_positive_radius", 2.0))
    _require_finite_range("trace_validity_positive_radius", trace_validity_positive_radius, min_value=0.0)

    trace_validity_negative_radius = float(config.get("trace_validity_negative_radius", 3.0))
    _require_finite_range("trace_validity_negative_radius", trace_validity_negative_radius, min_value=0.0)

    trace_validity_margin = float(config.get("trace_validity_margin", 3.0))
    _require_finite_range("trace_validity_margin", trace_validity_margin, min_value=0.0)

    trace_validity_background_weight = float(config.get("trace_validity_background_weight", 0.25))
    _require_finite_range("trace_validity_background_weight", trace_validity_background_weight, min_value=0.0)

    trace_validity_pos_weight = float(config.get("trace_validity_pos_weight", 1.0))
    _require_finite_range("trace_validity_pos_weight", trace_validity_pos_weight, min_value=0.0)

    if training_mode == "copy_neighbors":
        if str(config.get("triplet_dense_weight_mode", "all")).lower() not in {"all", "band", "all_band_boost"}:
            raise ValueError("triplet_dense_weight_mode must be one of all, band, or all_band_boost")
        if str(config.get("triplet_direction_prior_mask", "cond")).lower() not in {"cond", "full"}:
            raise ValueError("triplet_direction_prior_mask must be 'cond' or 'full'")
        allowed_displacement_losses = {
            "vector_l2",
            "vector_huber",
            "vector_huber_per_branch",
            "component_huber",
        }
        if str(config.get("displacement_loss_type", "vector_huber_per_branch")) not in allowed_displacement_losses:
            raise ValueError(
                "displacement_loss_type must be accepted by dense_displacement_loss, got "
                f"{config.get('displacement_loss_type')!r}"
            )
        for key in (
            "triplet_random_channel_swap_prob",
            "triplet_gt_vector_dilation_radius",
            "triplet_band_padding_voxels",
            "triplet_band_distance_percentile",
            "triplet_band_boost_weight",
            "triplet_edt_bbox_padding_voxels",
            "displacement_huber_beta",
            "lambda_smooth",
            "triplet_min_disp_vox",
            "lambda_triplet_min_disp",
            "lambda_displaced_source_edt",
            "displaced_source_edt_beta",
            "displaced_source_edt_oob_weight",
        ):
            _require_finite_range(key, float(config.get(key, 0.0)), min_value=0.0)
        if str(config.get("displaced_source_edt_loss_type", "huber")).lower() not in {"huber", "l1", "l2"}:
            raise ValueError("displaced_source_edt_loss_type must be one of huber, l1, or l2")
        max_points = int(config.get("displaced_source_edt_max_points", 4096))
        if max_points < 0:
            raise ValueError("displaced_source_edt_max_points must be >= 0")
        swap_prob = float(config.get("triplet_random_channel_swap_prob", 0.0))
        if swap_prob > 1.0:
            raise ValueError("triplet_random_channel_swap_prob must satisfy 0 <= p <= 1")
        if float(config.get("triplet_band_distance_percentile", 100.0)) > 100.0:
            raise ValueError("triplet_band_distance_percentile must be <= 100")
