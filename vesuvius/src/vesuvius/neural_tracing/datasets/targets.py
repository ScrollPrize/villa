from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
import vc_reader.transforms as A

from vesuvius.neural_tracing.datasets.common import create_band_mask
from vesuvius.neural_tracing.datasets.growth_direction import make_growth_direction_tensor


_SURFACE_EDT = A.DistanceTransform(mode="outside", indices="_nearest", implicit_background=True)


def _surface_edt(
    surface_bool: torch.Tensor,
    *,
    return_distances: bool,
    return_indices: bool,
):
    """Exact EDT to the voxels of a 3D CUDA mask (vc_reader), as tensors on its device.

    Distances are float32; indices are the (3, D, H, W) int32 nearest surface
    voxel, ties resolved as scipy.ndimage.distance_transform_edt resolves them.
    """
    if surface_bool.ndim != 3:
        raise ValueError(f"surface mask must be 3D, got shape {tuple(surface_bool.shape)}")
    if not surface_bool.is_cuda:
        raise RuntimeError("rowcol_cond EDT targets require CUDA tensors")

    out = _SURFACE_EDT({"mask": surface_bool[None].to(torch.uint8)})
    results = []
    if return_distances:
        results.append(out["mask_sdt"][0])
    if return_indices:
        results.append(out["mask_nearest"])
    return results[0] if len(results) == 1 else tuple(results)


def _distance_transform_distances(surface_mask: torch.Tensor):
    surface_bool = (surface_mask > 0.5).contiguous()
    if surface_bool.ndim != 3:
        raise ValueError(f"surface mask must be 3D, got shape {tuple(surface_bool.shape)}")
    if not bool(surface_bool.any().item()):
        return None
    return _surface_edt(surface_bool, return_distances=True, return_indices=False)


def _dilate_binary_mask(surface_mask: torch.Tensor, radius_voxels: float):
    radius = float(radius_voxels)
    surface_bool = (surface_mask > 0.5).contiguous()
    if radius <= 0.0 or not bool(surface_bool.any().item()):
        return surface_bool
    return _surface_edt(surface_bool, return_distances=True, return_indices=False) <= radius


def _voxel_coordinates(shape, device):
    """(3, D, H, W) float32 z, y and x coordinates."""
    return torch.stack(torch.meshgrid(*(torch.arange(n, device=device, dtype=torch.float32) for n in shape), indexing="ij"))


def _dense_displacement_from_mask(surface_mask: torch.Tensor):
    surface_bool = (surface_mask > 0.5).contiguous()
    if surface_bool.ndim != 3:
        raise ValueError(f"surface mask must be 3D, got shape {tuple(surface_bool.shape)}")
    if not bool(surface_bool.any().item()):
        return None, None
    distances, nearest = _surface_edt(surface_bool, return_distances=True, return_indices=True)
    return nearest.float() - _voxel_coordinates(surface_bool.shape, surface_bool.device), distances


def _swap_triplet_torch(dense_gt: torch.Tensor, dir_priors: torch.Tensor | None):
    dense_gt = torch.cat([dense_gt[3:6], dense_gt[0:3]], dim=0)
    if dir_priors is not None:
        dir_priors = torch.cat([dir_priors[3:6], dir_priors[0:3]], dim=0)
    return dense_gt, dir_priors


def _swap_triplet_scalar_torch(branch_values: torch.Tensor):
    return torch.cat([branch_values[1:2], branch_values[0:1]], dim=0)


def _align_triplet_to_priors_torch(
    dense_gt: torch.Tensor,
    dir_priors: torch.Tensor | None,
    cond_mask: torch.Tensor,
):
    order = torch.tensor([0, 1], device=dense_gt.device, dtype=torch.long)
    if dir_priors is None:
        return dense_gt, dir_priors, order

    mask = cond_mask > 0.5
    if not bool(mask.any().item()):
        return dense_gt, dir_priors, order

    prior_vecs = dir_priors[0:3, mask].transpose(0, 1)
    finite = torch.isfinite(prior_vecs).all(dim=1)
    prior_vecs = prior_vecs[finite]
    if prior_vecs.numel() == 0:
        return dense_gt, dir_priors, order
    prior_norm = prior_vecs.norm(dim=1)
    prior_vecs = prior_vecs[prior_norm > 1e-6]
    if prior_vecs.numel() == 0:
        return dense_gt, dir_priors, order
    unit_prior = prior_vecs / prior_vecs.norm(dim=1, keepdim=True).clamp(min=1e-6)
    normal = unit_prior.mean(dim=0)
    normal_norm = normal.norm()
    if not bool(torch.isfinite(normal_norm).item()) or float(normal_norm.item()) <= 1e-6:
        return dense_gt, dir_priors, order
    normal = normal / normal_norm.clamp(min=1e-6)

    def _median_signed_projection(field: torch.Tensor):
        vecs = field[:, mask].transpose(0, 1)
        finite_vecs = torch.isfinite(vecs).all(dim=1)
        vecs = vecs[finite_vecs]
        if vecs.numel() == 0:
            return None
        vecs = vecs[vecs.norm(dim=1) > 1e-6]
        if vecs.numel() == 0:
            return None
        return torch.median(vecs @ normal)

    s0 = _median_signed_projection(dense_gt[0:3])
    s1 = _median_signed_projection(dense_gt[3:6])
    if s0 is None or s1 is None:
        return dense_gt, dir_priors, order
    if bool((s1 > s0).item()):
        dense_gt = torch.cat([dense_gt[3:6], dense_gt[0:3]], dim=0)
        order = torch.tensor([1, 0], device=dense_gt.device, dtype=torch.long)
    return dense_gt, dir_priors, order


def _copy_neighbor_dense_weight(
    cond_mask: torch.Tensor,
    front_dist: torch.Tensor,
    behind_dist: torch.Tensor,
    front_disp: torch.Tensor,
    behind_disp: torch.Tensor,
    config: Mapping[str, object],
):
    weight_mode = str(config["triplet_dense_weight_mode"]).lower()
    if weight_mode == "all":
        return torch.ones((1, *cond_mask.shape), device=cond_mask.device, dtype=torch.float32)

    if weight_mode not in {"band", "all_band_boost"}:
        raise ValueError(
            "triplet_dense_weight_mode must be one of {'all', 'band', 'all_band_boost'}, "
            f"got {weight_mode!r}"
        )

    dense_band = create_band_mask(
        cond_bin_full=(cond_mask > 0.5).detach().cpu().numpy(),
        d_front_work=front_dist.detach().cpu().numpy(),
        d_behind_work=behind_dist.detach().cpu().numpy(),
        front_disp_work=front_disp.detach().cpu().numpy(),
        behind_disp_work=behind_disp.detach().cpu().numpy(),
        band_pct=min(100.0, max(1.0, float(config["triplet_band_distance_percentile"]))),
        band_padding=max(0.0, float(config["triplet_band_padding_voxels"])),
    )
    if dense_band is None:
        if weight_mode == "all_band_boost":
            return torch.ones((1, *cond_mask.shape), device=cond_mask.device, dtype=torch.float32)
        return torch.zeros((1, *cond_mask.shape), device=cond_mask.device, dtype=torch.float32)

    dense_band_t = torch.as_tensor(dense_band, device=cond_mask.device, dtype=torch.float32).unsqueeze(0)
    if weight_mode == "all_band_boost":
        band_boost_weight = float(config.get("triplet_band_boost_weight", 2.0))
        return torch.ones_like(dense_band_t) + (band_boost_weight - 1.0) * dense_band_t
    return dense_band_t


def _trace_validity_from_edt(
    d_target: torch.Tensor,
    d_neighbor: torch.Tensor,
    *,
    positive_radius: float,
    negative_radius: float,
    margin_threshold: float,
    background_weight: float,
):
    """float32 label and weight: positives near the target and clearly nearer it
    than the neighbour; hard negatives near the neighbour; far background at
    background_weight."""
    margin = d_neighbor - d_target
    pos = (d_target <= positive_radius) & (margin >= margin_threshold)
    neg = (d_neighbor <= negative_radius) & (margin < margin_threshold)
    hard_neg = neg & ~pos
    background_neg = (d_target >= negative_radius) & ~pos & ~hard_neg
    label = pos.to(torch.float32)
    weight = (pos | hard_neg).to(torch.float32)
    if background_weight > 0.0:
        weight = torch.where(background_neg, torch.tensor(background_weight, dtype=torch.float32, device=weight.device), weight)
    return label, weight


def _dilate_and_attract_from_edt(
    velocity: torch.Tensor,
    nearest_dist: torch.Tensor,
    nearest_idx: torch.Tensor,
    shape,
    dilation_radius: float,
    attract_radius: float,
):
    """Within dilation_radius of the source, velocity copied from the nearest source
    voxel; within attract_radius, the offset to it. Weights mark both bands."""
    finite = torch.isfinite(nearest_dist)
    in_dilation = finite & (nearest_dist <= dilation_radius)
    in_attract = finite & (nearest_dist <= attract_radius)
    _, h, w = shape
    source = (nearest_idx[0].long() * h + nearest_idx[1].long()) * w + nearest_idx[2].long()
    gathered = velocity.reshape(3, -1)[:, source.reshape(-1)].reshape(velocity.shape)
    dilated_velocity = torch.where(in_dilation, gathered, velocity).to(torch.float32)
    offset = nearest_idx.float() - _voxel_coordinates(shape, velocity.device)
    attract = torch.where(in_attract, offset, torch.zeros_like(offset))
    return dilated_velocity, in_dilation[None].to(torch.float32), attract, in_attract[None].to(torch.float32)


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

        velocity_b = velocity_dir[b].contiguous()
        nearest_dist, nearest_idx = _surface_edt(
            (source_weight[b, 0] > 0.5).contiguous(),
            return_distances=True,
            return_indices=True,
        )

        attract_b = None
        attract_weight_b = None
        attract_radius = float(surface_attract_radius)
        if attract_radius > 0.0:
            dilated_dir, dilated_weight, attract_b, attract_weight_b = _dilate_and_attract_from_edt(
                velocity_b,
                nearest_dist,
                nearest_idx,
                source_weight.shape[2:],
                float(radius_voxels),
                attract_radius,
            )
        else:
            # In place, as before: velocity_b shares velocity_dir's storage when contiguous.
            band = torch.isfinite(nearest_dist) & (nearest_dist <= float(radius_voxels))
            source = nearest_idx[:, band].long()
            velocity_b[:, band] = velocity_b[:, source[0], source[1], source[2]]
            dilated_dir = velocity_b
            dilated_weight = band[None].to(torch.float32)
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
            d_target = _distance_transform_distances(target_mask[b])
            d_neighbor = _distance_transform_distances(batch["neighbor_seg"][b])
            if d_target is None or d_neighbor is None:
                labels.append(torch.zeros_like(target_mask[b], dtype=torch.float32))
                weights.append(torch.zeros_like(target_mask[b], dtype=torch.float32))
                continue

            label, weight = _trace_validity_from_edt(
                d_target,
                d_neighbor,
                positive_radius=positive_radius,
                negative_radius=negative_radius,
                margin_threshold=margin_threshold,
                background_weight=background_weight,
            )
            labels.append(label)
            weights.append(weight)

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


@dataclass(frozen=True)
class CopyNeighborTargets:
    """Model inputs and dense displacement targets for one copy-neighbor batch."""

    inputs: torch.Tensor
    dense_gt_displacement: torch.Tensor
    dense_loss_weight: torch.Tensor
    cond: torch.Tensor
    cond_gt: torch.Tensor | None = None
    behind_seg: torch.Tensor | None = None
    front_seg: torch.Tensor | None = None
    branch_target_edt: torch.Tensor | None = None
    dir_priors: torch.Tensor | None = None
    triplet_channel_order: torch.Tensor | None = None

    @classmethod
    def from_batch(cls, batch: Mapping[str, object], config: Mapping[str, object]) -> "CopyNeighborTargets":
        required_keys = (
            "vol",
            "cond",
            "cond_gt",
            "behind_seg",
            "front_seg",
        )
        missing = [key for key in required_keys if key not in batch]
        if missing:
            raise ValueError(f"Batch is missing required copy-neighbor target keys: {missing}")

        vol = batch["vol"].unsqueeze(1)
        cond = batch["cond"].unsqueeze(1)
        dir_priors = batch.get("dir_priors", None)
        if bool(config.get("use_triplet_direction_priors", False)):
            if dir_priors is None:
                raise ValueError("use_triplet_direction_priors=True but batch has no dir_priors")
            if dir_priors.ndim != 5 or dir_priors.shape[1] != 6:
                raise ValueError(f"dir_priors must have shape [B, 6, D, H, W], got {tuple(dir_priors.shape)}")

        (
            dense_gt_displacement,
            dense_loss_weight,
            branch_target_edt,
            dir_priors,
            triplet_channel_order,
        ) = cls._build_dense_targets(
            batch,
            config,
            dir_priors=dir_priors,
        )
        input_parts = [vol, cond]
        if bool(config.get("use_triplet_direction_priors", False)):
            input_parts.append(dir_priors.to(device=vol.device, dtype=vol.dtype))
        inputs = torch.cat(input_parts, dim=1)

        return cls(
            inputs=inputs,
            dense_gt_displacement=dense_gt_displacement,
            dense_loss_weight=dense_loss_weight,
            cond=cond,
            cond_gt=batch.get("cond_gt", None),
            behind_seg=batch.get("behind_seg", None),
            front_seg=batch.get("front_seg", None),
            branch_target_edt=branch_target_edt,
            dir_priors=dir_priors,
            triplet_channel_order=triplet_channel_order,
        )

    @staticmethod
    def _build_dense_targets(
        batch: Mapping[str, object],
        config: Mapping[str, object],
        *,
        dir_priors: torch.Tensor | None,
    ):
        dense_targets = []
        dense_weights = []
        branch_target_edts = []
        aligned_priors = [] if dir_priors is not None else None
        channel_orders = []

        dilation_radius = max(0.0, float(config["triplet_gt_vector_dilation_radius"]))
        swap_prob = float(config.get("triplet_random_channel_swap_prob", 0.0))
        swap_enabled = batch.get("triplet_swap_enabled", None)

        for b in range(batch["behind_seg"].shape[0]):
            behind_mask = _dilate_binary_mask(batch["behind_seg"][b], dilation_radius)
            front_mask = _dilate_binary_mask(batch["front_seg"][b], dilation_radius)
            behind_disp, behind_dist = _dense_displacement_from_mask(behind_mask)
            front_disp, front_dist = _dense_displacement_from_mask(front_mask)
            if behind_disp is None or front_disp is None:
                shape = tuple(int(v) for v in batch["behind_seg"].shape[1:])
                dense_gt = torch.zeros((6, *shape), device=batch["behind_seg"].device, dtype=torch.float32)
                dense_weight = torch.zeros((1, *shape), device=batch["behind_seg"].device, dtype=torch.float32)
                branch_edt = torch.zeros((2, *shape), device=batch["behind_seg"].device, dtype=torch.float32)
            else:
                dense_gt = torch.cat([behind_disp, front_disp], dim=0).to(dtype=torch.float32)
                branch_edt = torch.stack([behind_dist, front_dist], dim=0).to(dtype=torch.float32)
                dense_weight = _copy_neighbor_dense_weight(
                    batch["cond_gt"][b],
                    front_dist,
                    behind_dist,
                    front_disp,
                    behind_disp,
                    config,
                )

            priors_b = dir_priors[b].to(device=dense_gt.device, dtype=dense_gt.dtype) if dir_priors is not None else None
            dense_gt, priors_b, channel_order = _align_triplet_to_priors_torch(
                dense_gt,
                priors_b,
                batch["cond_gt"][b],
            )
            branch_edt = branch_edt.index_select(0, channel_order)

            if swap_enabled is None:
                do_swap = False
            else:
                do_swap = bool(swap_enabled[b].item())
            if do_swap and swap_prob > 0.0:
                if not np.isfinite(swap_prob) or swap_prob < 0.0 or swap_prob > 1.0:
                    raise ValueError(f"triplet_random_channel_swap_prob must satisfy 0 <= p <= 1, got {swap_prob!r}")
                if bool((torch.rand((), device=dense_gt.device) < swap_prob).item()):
                    dense_gt, priors_b = _swap_triplet_torch(dense_gt, priors_b)
                    branch_edt = _swap_triplet_scalar_torch(branch_edt)
                    channel_order = channel_order.flip(0)

            dense_targets.append(dense_gt)
            dense_weights.append(dense_weight)
            branch_target_edts.append(branch_edt)
            if aligned_priors is not None:
                aligned_priors.append(priors_b)
            channel_orders.append(channel_order)

        dense_gt_displacement = torch.stack(dense_targets, dim=0)
        dense_loss_weight = torch.stack(dense_weights, dim=0)
        branch_target_edt = torch.stack(branch_target_edts, dim=0)
        if aligned_priors is not None:
            dir_priors = torch.stack(aligned_priors, dim=0)
        triplet_channel_order = torch.stack(channel_orders, dim=0)
        return dense_gt_displacement, dense_loss_weight, branch_target_edt, dir_priors, triplet_channel_order
