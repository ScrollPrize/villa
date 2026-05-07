from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping

import torch

from vesuvius.neural_tracing.datasets.growth_direction import make_growth_direction_tensor


def _torch_to_cupy(tensor: torch.Tensor):
    import cupy as cp

    return cp.from_dlpack(torch.utils.dlpack.to_dlpack(tensor.contiguous()))


def _cupy_to_torch(array):
    return torch.utils.dlpack.from_dlpack(array)


def _cupy_device(tensor: torch.Tensor):
    import cupy as cp

    device_index = tensor.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    return cp.cuda.Device(int(device_index))


def _cupyx_edt(
    surface_bool: torch.Tensor,
    *,
    return_distances: bool,
    return_indices: bool,
):
    """Run the single supported EDT implementation on a 3D CUDA mask."""
    import cupy as cp
    from cupyx.scipy import ndimage as cndimage

    if surface_bool.ndim != 3:
        raise ValueError(f"surface mask must be 3D, got shape {tuple(surface_bool.shape)}")
    if not surface_bool.is_cuda:
        raise RuntimeError("rowcol_cond EDT targets require CUDA tensors and cupyx.scipy.ndimage")

    with _cupy_device(surface_bool):
        surface_cp = _torch_to_cupy(surface_bool)
        return cndimage.distance_transform_edt(
            ~surface_cp,
            return_distances=bool(return_distances),
            return_indices=bool(return_indices),
            float64_distances=False,
        )


def _distance_transform_distances_cupy(surface_mask: torch.Tensor):
    import cupy as cp

    surface_bool = (surface_mask > 0.5).contiguous()
    if surface_bool.ndim != 3:
        raise ValueError(f"surface mask must be 3D, got shape {tuple(surface_bool.shape)}")
    if not bool(surface_bool.any().item()):
        return None
    with _cupy_device(surface_bool):
        distances_cp = _cupyx_edt(surface_bool, return_distances=True, return_indices=False)
        return cp.ascontiguousarray(distances_cp.astype(cp.float32, copy=False))


@lru_cache(maxsize=None)
def _trace_validity_from_edt_kernel():
    import cupy as cp

    return cp.ElementwiseKernel(
        (
            "float32 d_target, float32 d_neighbor, float32 positive_radius, "
            "float32 negative_radius, float32 margin_threshold, float32 background_weight"
        ),
        "float32 label, float32 weight",
        """
        const float margin = d_neighbor - d_target;
        const bool pos = (d_target <= positive_radius) && (margin >= margin_threshold);
        const bool neg = (d_neighbor <= negative_radius) && (margin < margin_threshold);
        const bool far_neg = d_target >= negative_radius;
        const bool hard_neg = neg && !pos;
        const bool background_neg = far_neg && !pos && !hard_neg;
        label = pos ? 1.0f : 0.0f;
        if (pos || hard_neg) {
            weight = 1.0f;
        } else if (background_weight > 0.0f && background_neg) {
            weight = background_weight;
        } else {
            weight = 0.0f;
        }
        """,
        "trace_validity_from_edt",
    )


def _trace_validity_from_edt(
    d_target_cp,
    d_neighbor_cp,
    *,
    positive_radius: float,
    negative_radius: float,
    margin_threshold: float,
    background_weight: float,
):
    import cupy as cp

    label_cp = cp.empty_like(d_target_cp, dtype=cp.float32)
    weight_cp = cp.empty_like(d_target_cp, dtype=cp.float32)
    _trace_validity_from_edt_kernel()(
        d_target_cp,
        d_neighbor_cp,
        float(positive_radius),
        float(negative_radius),
        float(margin_threshold),
        float(background_weight),
        label_cp,
        weight_cp,
    )
    return label_cp, weight_cp


@lru_cache(maxsize=None)
def _dilate_and_attract_from_edt_kernel():
    import cupy as cp

    return cp.ElementwiseKernel(
        (
            "raw float32 vel_z, raw float32 vel_y, raw float32 vel_x, "
            "float32 dist, I nearest_z, I nearest_y, I nearest_x, "
            "int64 h, int64 w, float32 dilation_radius, float32 attract_radius"
        ),
        (
            "float32 out_vel_z, float32 out_vel_y, float32 out_vel_x, float32 dilated_weight, "
            "float32 attract_z, float32 attract_y, float32 attract_x, float32 attract_weight"
        ),
        """
        const long long linear = i;
        const long long x = linear % w;
        const long long y = (linear / w) % h;
        const long long z = linear / (w * h);
        const long long src_linear =
            static_cast<long long>(nearest_z) * h * w +
            static_cast<long long>(nearest_y) * w +
            static_cast<long long>(nearest_x);
        const bool finite = isfinite(dist);
        const bool in_dilation_band = finite && dist <= dilation_radius;
        const bool in_attract_band = finite && dist <= attract_radius;

        dilated_weight = in_dilation_band ? 1.0f : 0.0f;
        out_vel_z = in_dilation_band ? vel_z[src_linear] : vel_z[linear];
        out_vel_y = in_dilation_band ? vel_y[src_linear] : vel_y[linear];
        out_vel_x = in_dilation_band ? vel_x[src_linear] : vel_x[linear];

        attract_weight = in_attract_band ? 1.0f : 0.0f;
        attract_z = in_attract_band ? static_cast<float>(nearest_z) - static_cast<float>(z) : 0.0f;
        attract_y = in_attract_band ? static_cast<float>(nearest_y) - static_cast<float>(y) : 0.0f;
        attract_x = in_attract_band ? static_cast<float>(nearest_x) - static_cast<float>(x) : 0.0f;
        """,
        "dilate_and_attract_from_edt",
    )


def _dilate_and_attract_from_edt(
    velocity_cp,
    nearest_dist_cp,
    nearest_idx_cp,
    shape,
    dilation_radius: float,
    attract_radius: float,
):
    import cupy as cp

    _, h, w = shape
    dilated_velocity_cp = cp.empty_like(velocity_cp, dtype=cp.float32)
    dilated_weight_cp = cp.empty((1, *shape), dtype=cp.float32)
    attract_cp = cp.empty((3, *shape), dtype=cp.float32)
    attract_weight_cp = cp.empty((1, *shape), dtype=cp.float32)
    _dilate_and_attract_from_edt_kernel()(
        velocity_cp[0],
        velocity_cp[1],
        velocity_cp[2],
        nearest_dist_cp,
        nearest_idx_cp[0],
        nearest_idx_cp[1],
        nearest_idx_cp[2],
        int(h),
        int(w),
        float(dilation_radius),
        float(attract_radius),
        dilated_velocity_cp[0],
        dilated_velocity_cp[1],
        dilated_velocity_cp[2],
        dilated_weight_cp[0],
        attract_cp[0],
        attract_cp[1],
        attract_cp[2],
        attract_weight_cp[0],
    )
    return dilated_velocity_cp, dilated_weight_cp, attract_cp, attract_weight_cp


def _dilate_trace_targets_torch(
    velocity_dir: torch.Tensor,
    velocity_weight: torch.Tensor,
    radius_voxels: float,
    trace_loss_weight: torch.Tensor | None = None,
    surface_attract_radius: float = 0.0,
):
    if float(radius_voxels) <= 0.0 and float(surface_attract_radius) <= 0.0:
        return velocity_dir, velocity_weight, trace_loss_weight, None, None

    if velocity_dir.ndim != 5 or velocity_dir.shape[1] != 3:
        raise ValueError(f"velocity_dir must have shape [B, 3, D, H, W], got {tuple(velocity_dir.shape)}")
    if velocity_weight.ndim != 5 or velocity_weight.shape[1] != 1:
        raise ValueError(
            f"velocity_weight must have shape [B, 1, D, H, W], got {tuple(velocity_weight.shape)}"
        )

    source_weight = trace_loss_weight if trace_loss_weight is not None else velocity_weight
    dilated_dirs = []
    dilated_weights = []
    surface_attracts = [] if float(surface_attract_radius) > 0.0 else None
    surface_attract_weights = [] if float(surface_attract_radius) > 0.0 else None

    for b in range(velocity_dir.shape[0]):
        if not bool((source_weight[b, 0] > 0.5).any().item()):
            dilated_dirs.append(velocity_dir[b])
            dilated_weights.append(velocity_weight[b])
            if surface_attracts is not None:
                surface_attracts.append(torch.zeros_like(velocity_dir[b]))
                surface_attract_weights.append(torch.zeros_like(velocity_weight[b]))
            continue

        with _cupy_device(velocity_dir):
            velocity_b = velocity_dir[b].contiguous()
            velocity_cp = _torch_to_cupy(velocity_b)
            nearest_dist_cp, nearest_idx_cp = _cupyx_edt(
                (source_weight[b, 0] > 0.5).contiguous(),
                return_distances=True,
                return_indices=True,
            )

            attract_b = None
            attract_weight_b = None
            attract_radius = float(surface_attract_radius)
            if attract_radius > 0.0:
                (
                    dilated_velocity_cp,
                    dilated_weight_cp,
                    attract_cp,
                    attract_weight_cp,
                ) = _dilate_and_attract_from_edt(
                    velocity_cp,
                    nearest_dist_cp,
                    nearest_idx_cp,
                    source_weight.shape[2:],
                    float(radius_voxels),
                    attract_radius,
                )
                dilated_dir = _cupy_to_torch(dilated_velocity_cp)
                dilated_weight = _cupy_to_torch(dilated_weight_cp)
                attract_b = _cupy_to_torch(attract_cp)
                attract_weight_b = _cupy_to_torch(attract_weight_cp)
            else:
                import cupy as cp

                band_cp = cp.isfinite(nearest_dist_cp) & (nearest_dist_cp <= float(radius_voxels))
                src_z = nearest_idx_cp[0][band_cp]
                src_y = nearest_idx_cp[1][band_cp]
                src_x = nearest_idx_cp[2][band_cp]
                velocity_cp[:, band_cp] = velocity_cp[:, src_z, src_y, src_x]
                dilated_dir = velocity_b
                dilated_weight = _cupy_to_torch(
                    cp.ascontiguousarray(band_cp[None].astype(cp.float32, copy=False))
                )
        dilated_dirs.append(dilated_dir)
        dilated_weights.append(dilated_weight)
        if surface_attracts is not None:
            surface_attracts.append(attract_b)
            surface_attract_weights.append(attract_weight_b)

    velocity_dir = torch.stack(dilated_dirs, dim=0)
    velocity_weight = torch.stack(dilated_weights, dim=0)
    if trace_loss_weight is not None:
        trace_loss_weight = velocity_weight
    surface_attract = torch.stack(surface_attracts, dim=0) if surface_attracts is not None else None
    surface_attract_weight = (
        torch.stack(surface_attract_weights, dim=0) if surface_attract_weights is not None else None
    )
    return (
        velocity_dir,
        velocity_weight,
        trace_loss_weight,
        surface_attract,
        surface_attract_weight,
    )


@dataclass(frozen=True)
class RowColTargets:
    """Model inputs and supervised targets for one row/col conditioned batch."""

    inputs: torch.Tensor
    velocity_dir: torch.Tensor
    velocity_loss_weight: torch.Tensor
    trace_loss_weight: torch.Tensor
    trace_validity: torch.Tensor
    trace_validity_weight: torch.Tensor
    surface_attract: torch.Tensor
    surface_attract_weight: torch.Tensor

    @classmethod
    def from_batch(cls, batch: Mapping[str, object], config: Mapping[str, object]) -> "RowColTargets":
        required_keys = (
            "vol",
            "cond",
            "cond_direction",
            "velocity_dir",
            "velocity_loss_weight",
            "trace_loss_weight",
            "cond_gt",
            "masked_seg",
            "neighbor_seg",
        )
        missing = [key for key in required_keys if key not in batch]
        if missing:
            raise ValueError(f"Batch is missing required row/col target keys: {missing}")

        trace_validity, trace_validity_weight = cls._build_trace_validity_targets(batch, config)
        (
            velocity_dir,
            velocity_loss_weight,
            trace_loss_weight,
            surface_attract,
            surface_attract_weight,
        ) = cls._build_dilated_trace_targets(batch, config)

        vol = batch["vol"].unsqueeze(1)  # [B, 1, D, H, W]
        cond = batch["cond"].unsqueeze(1)  # [B, 1, D, H, W]
        inputs = torch.cat(
            [
                vol,
                cond,
                make_growth_direction_tensor(
                    batch["cond_direction"],
                    vol.shape[2:],
                    device=vol.device,
                    dtype=vol.dtype,
                ),
            ],
            dim=1,
        )

        return cls(
            inputs=inputs,
            velocity_dir=velocity_dir,
            velocity_loss_weight=velocity_loss_weight,
            trace_loss_weight=trace_loss_weight,
            trace_validity=trace_validity,
            trace_validity_weight=trace_validity_weight,
            surface_attract=surface_attract,
            surface_attract_weight=surface_attract_weight,
        )

    @staticmethod
    def _build_trace_validity_targets(batch: Mapping[str, object], config: Mapping[str, object]):
        labels = []
        weights = []
        target_mask = torch.maximum(batch["cond_gt"], batch["masked_seg"]) > 0.5
        positive_radius = float(config.get("trace_validity_positive_radius", 2.0))
        negative_radius = float(config.get("trace_validity_negative_radius", 3.0))
        margin_threshold = float(config.get("trace_validity_margin", 3.0))
        background_weight = float(config.get("trace_validity_background_weight", 0.25))

        for b in range(target_mask.shape[0]):
            d_target_cp = _distance_transform_distances_cupy(target_mask[b])
            d_neighbor_cp = _distance_transform_distances_cupy(batch["neighbor_seg"][b])
            if d_target_cp is None or d_neighbor_cp is None:
                labels.append(torch.zeros_like(target_mask[b], dtype=torch.float32))
                weights.append(torch.zeros_like(target_mask[b], dtype=torch.float32))
                continue

            label_cp, weight_cp = _trace_validity_from_edt(
                d_target_cp,
                d_neighbor_cp,
                positive_radius=positive_radius,
                negative_radius=negative_radius,
                margin_threshold=margin_threshold,
                background_weight=background_weight,
            )
            labels.append(_cupy_to_torch(label_cp))
            weights.append(_cupy_to_torch(weight_cp))

        return torch.stack(labels, dim=0).unsqueeze(1), torch.stack(weights, dim=0).unsqueeze(1)

    @staticmethod
    def _build_dilated_trace_targets(batch: Mapping[str, object], config: Mapping[str, object]):
        trace_target_dilation_radius = float(config.get("trace_target_dilation_radius", 0.0))
        trace_surface_attract_radius = float(config.get("trace_surface_attract_radius", 0.0))
        (
            velocity_dir,
            velocity_loss_weight,
            trace_loss_weight,
            surface_attract,
            surface_attract_weight,
        ) = _dilate_trace_targets_torch(
            batch["velocity_dir"],
            batch["velocity_loss_weight"],
            trace_target_dilation_radius,
            trace_loss_weight=batch.get("trace_loss_weight", None),
            surface_attract_radius=trace_surface_attract_radius,
        )
        if surface_attract is None or surface_attract_weight is None:
            raise ValueError(
                "RowColTargets requires surface attraction targets; "
                "set trace_surface_attract_radius > 0 for the active trace-ODE path"
            )
        return velocity_dir, velocity_loss_weight, trace_loss_weight, surface_attract, surface_attract_weight
