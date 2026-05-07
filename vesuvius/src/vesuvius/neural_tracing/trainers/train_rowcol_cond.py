#!/usr/bin/env python
"""
Trainer for row/col conditioned displacement field prediction.

Trains a model to predict dense 3D displacement fields from conditioned surfaces,
with optional SDT (Signed Distance Transform) prediction.
"""
import os
import json
import sys
import click
import torch
import wandb
import copy
import random
import accelerate
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import time
from scipy import ndimage

from vesuvius.neural_tracing.datasets.dataset_rowcol_cond import EdtSegDataset
from vesuvius.neural_tracing.datasets.common import create_band_mask
from vesuvius.neural_tracing.datasets.dataset_defaults import (
    setdefault_rowcol_cond_dataset_config,
    validate_rowcol_cond_dataset_config,
)
from vesuvius.neural_tracing.datasets.growth_direction import (
    growth_direction_channel_count,
    make_growth_direction_tensor,
)
from vesuvius.neural_tracing.loss.displacement_losses import (
    dense_displacement_loss,
    surface_sampled_loss,
    surface_sampled_normal_loss,
    smoothness_loss,
    triplet_min_displacement_loss,
    velocity_streamline_integration_loss,
    weighted_vector_smoothness_loss,
)
from vesuvius.models.training.loss.nnunet_losses import DC_and_BCE_loss
from vesuvius.models.training.loss.skeleton_recall import DC_SkelREC_and_CE_loss
from vesuvius.models.training.optimizers import create_optimizer
from vesuvius.models.training.lr_schedulers import get_scheduler
from vesuvius.neural_tracing.nets.models import make_model
from vesuvius.neural_tracing.trainers.rowcol_cond_visualization import (
    make_dense_visualization,
    make_visualization,
)
from accelerate.utils import TorchDynamoPlugin

import multiprocessing


def seed_worker(worker_id):
    """Seed worker for reproducibility."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _zero_loss_from_output(output):
    for tensor in output.values():
        return tensor.new_zeros(())
    raise ValueError("Model returned no output tensors")


def _torch_to_cupy(tensor: torch.Tensor):
    import cupy as cp

    return cp.from_dlpack(torch.utils.dlpack.to_dlpack(tensor.contiguous()))


def _cupy_to_torch(array):
    return torch.utils.dlpack.from_dlpack(array)


def _cupyx_distance_transform_edt(surface_bool: torch.Tensor, return_distances: bool):
    """Run cupyx EDT on the CUDA device backing ``surface_bool``."""
    import cupy as cp
    from cupyx.scipy import ndimage as cndimage

    device_index = surface_bool.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()

    with cp.cuda.Device(int(device_index)):
        surface_cp = _torch_to_cupy(surface_bool)
        out = cndimage.distance_transform_edt(
            ~surface_cp,
            return_distances=bool(return_distances),
            return_indices=True,
            float64_distances=False,
        )
        if return_distances:
            distances_cp, nearest_idx_cp = out
            distances_cp = cp.ascontiguousarray(distances_cp.astype(cp.float32, copy=False))
        else:
            nearest_idx_cp = out
            distances_cp = None

        disp_cp = nearest_idx_cp.astype(cp.float32, copy=False)
        d, h, w = surface_bool.shape
        disp_cp[0] -= cp.arange(d, dtype=cp.float32)[:, None, None]
        disp_cp[1] -= cp.arange(h, dtype=cp.float32)[None, :, None]
        disp_cp[2] -= cp.arange(w, dtype=cp.float32)[None, None, :]
        disp_cp = cp.ascontiguousarray(disp_cp)

        disp = _cupy_to_torch(disp_cp)
        if return_distances:
            distances = _cupy_to_torch(distances_cp)
            return disp, distances
        return disp, None


def _scipy_distance_transform_edt(surface_bool: torch.Tensor, return_distances: bool):
    surface_np = surface_bool.detach().cpu().numpy().astype(bool, copy=False)
    if return_distances:
        distances_np, nearest_idx_np = ndimage.distance_transform_edt(
            ~surface_np,
            return_distances=True,
            return_indices=True,
        )
        distances = torch.from_numpy(distances_np.astype(np.float32, copy=False))
    else:
        nearest_idx_np = ndimage.distance_transform_edt(
            ~surface_np,
            return_distances=False,
            return_indices=True,
        )
        distances = None

    disp_np = nearest_idx_np.astype(np.float32, copy=False)
    d, h, w = surface_np.shape
    disp_np[0] -= np.arange(d, dtype=np.float32)[:, None, None]
    disp_np[1] -= np.arange(h, dtype=np.float32)[None, :, None]
    disp_np[2] -= np.arange(w, dtype=np.float32)[None, None, :]
    return torch.from_numpy(disp_np), distances


def _compute_dense_displacement_field_torch(
    surface_mask: torch.Tensor,
    *,
    return_weights: bool = True,
    return_distances: bool = False,
):
    surface_bool = (surface_mask > 0.5).contiguous()
    if surface_bool.ndim != 3:
        raise ValueError(f"surface mask must be 3D, got shape {tuple(surface_bool.shape)}")
    if not bool(surface_bool.any().item()):
        if return_distances:
            return None, None, None
        return None, None

    if surface_bool.is_cuda:
        disp, distances = _cupyx_distance_transform_edt(surface_bool, return_distances=return_distances)
    else:
        disp, distances = _scipy_distance_transform_edt(surface_bool, return_distances=return_distances)
        disp = disp.to(device=surface_mask.device, non_blocking=True)
        if distances is not None:
            distances = distances.to(device=surface_mask.device, non_blocking=True)

    disp = disp.to(dtype=torch.float32)
    weights = (
        torch.ones((1, *surface_bool.shape), device=surface_mask.device, dtype=torch.float32)
        if return_weights else None
    )
    if return_distances:
        return disp, weights, distances.to(dtype=torch.float32)
    return disp, weights


def _cupyx_dilate_velocity_target(
    velocity_dir: torch.Tensor,
    velocity_weight: torch.Tensor,
    radius_voxels: float,
):
    import cupy as cp
    from cupyx.scipy import ndimage as cndimage

    device_index = velocity_dir.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()

    with cp.cuda.Device(int(device_index)):
        velocity_cp = _torch_to_cupy(velocity_dir)
        valid_cp = _torch_to_cupy((velocity_weight[0] > 0.5).contiguous())
        nearest_dist_cp, nearest_idx_cp = cndimage.distance_transform_edt(
            ~valid_cp,
            return_distances=True,
            return_indices=True,
            float64_distances=False,
        )
        band_cp = cp.isfinite(nearest_dist_cp) & (nearest_dist_cp <= float(radius_voxels))
        if bool(cp.any(band_cp).item()):
            velocity_cp[:, band_cp] = velocity_cp[
                :,
                nearest_idx_cp[0][band_cp],
                nearest_idx_cp[1][band_cp],
                nearest_idx_cp[2][band_cp],
            ]
        velocity_weight_cp = cp.ascontiguousarray(band_cp[None].astype(cp.float32, copy=False))
        return velocity_dir, _cupy_to_torch(velocity_weight_cp)


def _scipy_dilate_velocity_target(
    velocity_dir: torch.Tensor,
    velocity_weight: torch.Tensor,
    radius_voxels: float,
):
    velocity_np = velocity_dir.detach().cpu().numpy().astype(np.float32, copy=True)
    valid_np = velocity_weight[0].detach().cpu().numpy() > 0.5
    nearest_dist_np, nearest_idx_np = ndimage.distance_transform_edt(
        ~valid_np,
        return_distances=True,
        return_indices=True,
    )
    band_np = np.isfinite(nearest_dist_np) & (nearest_dist_np <= float(radius_voxels))
    dilated_np = np.zeros_like(velocity_np, dtype=np.float32)
    if band_np.any():
        dilated_np[:, band_np] = velocity_np[
            :,
            nearest_idx_np[0][band_np],
            nearest_idx_np[1][band_np],
            nearest_idx_np[2][band_np],
        ]
    dilated = torch.from_numpy(dilated_np).to(device=velocity_dir.device, non_blocking=True)
    weight = torch.from_numpy(band_np[None].astype(np.float32, copy=False)).to(
        device=velocity_weight.device,
        non_blocking=True,
    )
    return dilated, weight


def _dilate_velocity_targets_torch(
    velocity_dir: torch.Tensor,
    velocity_weight: torch.Tensor,
    radius_voxels: float,
):
    if float(radius_voxels) <= 0.0:
        return velocity_dir, velocity_weight

    if velocity_dir.ndim != 5 or velocity_dir.shape[1] != 3:
        raise ValueError(f"velocity_dir must have shape [B, 3, D, H, W], got {tuple(velocity_dir.shape)}")
    if velocity_weight.ndim != 5 or velocity_weight.shape[1] != 1:
        raise ValueError(
            f"velocity_weight must have shape [B, 1, D, H, W], got {tuple(velocity_weight.shape)}"
        )

    dilated_dirs = []
    dilated_weights = []
    for b in range(velocity_dir.shape[0]):
        if not bool((velocity_weight[b, 0] > 0.5).any().item()):
            dilated_dirs.append(velocity_dir[b])
            dilated_weights.append(velocity_weight[b])
            continue
        if velocity_dir.is_cuda:
            dilated_dir, dilated_weight = _cupyx_dilate_velocity_target(
                velocity_dir[b].contiguous(),
                velocity_weight[b].contiguous(),
                radius_voxels,
            )
        else:
            dilated_dir, dilated_weight = _scipy_dilate_velocity_target(
                velocity_dir[b],
                velocity_weight[b],
                radius_voxels,
            )
        dilated_dirs.append(dilated_dir)
        dilated_weights.append(dilated_weight)

    return torch.stack(dilated_dirs, dim=0), torch.stack(dilated_weights, dim=0)


def _cupyx_distance_transform_distances(surface_bool: torch.Tensor):
    """Run cupyx EDT distances on the CUDA device backing ``surface_bool``."""
    import cupy as cp
    from cupyx.scipy import ndimage as cndimage

    device_index = surface_bool.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()

    with cp.cuda.Device(int(device_index)):
        surface_cp = _torch_to_cupy(surface_bool)
        distances_cp = cndimage.distance_transform_edt(
            ~surface_cp,
            return_distances=True,
            return_indices=False,
            float64_distances=False,
        )
        distances_cp = cp.ascontiguousarray(distances_cp.astype(cp.float32, copy=False))
        return _cupy_to_torch(distances_cp)


def _scipy_distance_transform_distances(surface_bool: torch.Tensor):
    surface_np = surface_bool.detach().cpu().numpy().astype(bool, copy=False)
    distances_np = ndimage.distance_transform_edt(
        ~surface_np,
        return_distances=True,
        return_indices=False,
    )
    return torch.from_numpy(distances_np.astype(np.float32, copy=False)).to(
        device=surface_bool.device,
        non_blocking=True,
    )


def _distance_transform_distances_torch(surface_mask: torch.Tensor):
    surface_bool = (surface_mask > 0.5).contiguous()
    if surface_bool.ndim != 3:
        raise ValueError(f"surface mask must be 3D, got shape {tuple(surface_bool.shape)}")
    if not bool(surface_bool.any().item()):
        return None
    if surface_bool.is_cuda:
        distances = _cupyx_distance_transform_distances(surface_bool)
    else:
        distances = _scipy_distance_transform_distances(surface_bool)
    return distances.to(dtype=torch.float32)


def _cupyx_dilate_trace_targets(
    velocity_dir: torch.Tensor,
    source_weight: torch.Tensor,
    radius_voxels: float,
    trace_dist: torch.Tensor | None = None,
    trace_stop: torch.Tensor | None = None,
    surface_attract_radius: float = 0.0,
):
    import cupy as cp
    from cupyx.scipy import ndimage as cndimage

    device_index = velocity_dir.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()

    with cp.cuda.Device(int(device_index)):
        velocity_cp = _torch_to_cupy(velocity_dir)
        valid_cp = _torch_to_cupy((source_weight[0] > 0.5).contiguous())
        nearest_dist_cp, nearest_idx_cp = cndimage.distance_transform_edt(
            ~valid_cp,
            return_distances=True,
            return_indices=True,
            float64_distances=False,
        )
        band_cp = cp.isfinite(nearest_dist_cp) & (nearest_dist_cp <= float(radius_voxels))
        if bool(cp.any(band_cp).item()):
            src_z = nearest_idx_cp[0][band_cp]
            src_y = nearest_idx_cp[1][band_cp]
            src_x = nearest_idx_cp[2][band_cp]
            velocity_cp[:, band_cp] = velocity_cp[:, src_z, src_y, src_x]
            if trace_dist is not None:
                trace_dist_cp = _torch_to_cupy(trace_dist)
                trace_dist_cp[0, band_cp] = trace_dist_cp[0, src_z, src_y, src_x]
            if trace_stop is not None:
                trace_stop_cp = _torch_to_cupy(trace_stop)
                trace_stop_cp[0, band_cp] = trace_stop_cp[0, src_z, src_y, src_x]

        surface_attract = None
        surface_attract_weight = None
        attract_radius = float(surface_attract_radius)
        if attract_radius > 0.0:
            attract_band_cp = cp.isfinite(nearest_dist_cp) & (nearest_dist_cp <= attract_radius)
            d, h, w = source_weight.shape[1:]
            attract_cp = cp.zeros((3, d, h, w), dtype=cp.float32)
            if bool(cp.any(attract_band_cp).item()):
                z, y, x = cp.nonzero(attract_band_cp)
                attract_cp[0, z, y, x] = nearest_idx_cp[0, z, y, x].astype(cp.float32, copy=False) - z.astype(cp.float32)
                attract_cp[1, z, y, x] = nearest_idx_cp[1, z, y, x].astype(cp.float32, copy=False) - y.astype(cp.float32)
                attract_cp[2, z, y, x] = nearest_idx_cp[2, z, y, x].astype(cp.float32, copy=False) - x.astype(cp.float32)
            surface_attract = _cupy_to_torch(cp.ascontiguousarray(attract_cp))
            surface_attract_weight = _cupy_to_torch(
                cp.ascontiguousarray(attract_band_cp[None].astype(cp.float32, copy=False))
            )

        dilated_weight = _cupy_to_torch(cp.ascontiguousarray(band_cp[None].astype(cp.float32, copy=False)))
        return velocity_dir, dilated_weight, trace_dist, trace_stop, surface_attract, surface_attract_weight


def _scipy_dilate_trace_targets(
    velocity_dir: torch.Tensor,
    source_weight: torch.Tensor,
    radius_voxels: float,
    trace_dist: torch.Tensor | None = None,
    trace_stop: torch.Tensor | None = None,
    surface_attract_radius: float = 0.0,
):
    velocity_np = velocity_dir.detach().cpu().numpy().astype(np.float32, copy=True)
    valid_np = source_weight[0].detach().cpu().numpy() > 0.5
    nearest_dist_np, nearest_idx_np = ndimage.distance_transform_edt(
        ~valid_np,
        return_distances=True,
        return_indices=True,
    )
    band_np = np.isfinite(nearest_dist_np) & (nearest_dist_np <= float(radius_voxels))
    if band_np.any():
        src = (
            nearest_idx_np[0][band_np],
            nearest_idx_np[1][band_np],
            nearest_idx_np[2][band_np],
        )
        velocity_np[:, band_np] = velocity_np[:, src[0], src[1], src[2]]

    velocity = torch.from_numpy(velocity_np).to(device=velocity_dir.device, non_blocking=True)
    weight = torch.from_numpy(band_np[None].astype(np.float32, copy=False)).to(
        device=source_weight.device,
        non_blocking=True,
    )

    if trace_dist is not None:
        trace_dist_np = trace_dist.detach().cpu().numpy().astype(np.float32, copy=True)
        if band_np.any():
            trace_dist_np[0, band_np] = trace_dist_np[0, src[0], src[1], src[2]]
        trace_dist = torch.from_numpy(trace_dist_np).to(device=trace_dist.device, non_blocking=True)

    if trace_stop is not None:
        trace_stop_np = trace_stop.detach().cpu().numpy().astype(np.float32, copy=True)
        if band_np.any():
            trace_stop_np[0, band_np] = trace_stop_np[0, src[0], src[1], src[2]]
        trace_stop = torch.from_numpy(trace_stop_np).to(device=trace_stop.device, non_blocking=True)

    surface_attract = None
    surface_attract_weight = None
    attract_radius = float(surface_attract_radius)
    if attract_radius > 0.0:
        attract_band_np = np.isfinite(nearest_dist_np) & (nearest_dist_np <= attract_radius)
        attract_np = np.zeros_like(velocity_np, dtype=np.float32)
        if attract_band_np.any():
            z, y, x = np.nonzero(attract_band_np)
            attract_np[0, z, y, x] = nearest_idx_np[0, z, y, x].astype(np.float32, copy=False) - z.astype(np.float32)
            attract_np[1, z, y, x] = nearest_idx_np[1, z, y, x].astype(np.float32, copy=False) - y.astype(np.float32)
            attract_np[2, z, y, x] = nearest_idx_np[2, z, y, x].astype(np.float32, copy=False) - x.astype(np.float32)
        surface_attract = torch.from_numpy(attract_np).to(device=velocity_dir.device, non_blocking=True)
        surface_attract_weight = torch.from_numpy(attract_band_np[None].astype(np.float32, copy=False)).to(
            device=source_weight.device,
            non_blocking=True,
        )

    return velocity, weight, trace_dist, trace_stop, surface_attract, surface_attract_weight


def _dilate_trace_targets_torch(
    velocity_dir: torch.Tensor,
    velocity_weight: torch.Tensor,
    radius_voxels: float,
    trace_dist: torch.Tensor | None = None,
    trace_stop: torch.Tensor | None = None,
    trace_loss_weight: torch.Tensor | None = None,
    surface_attract_radius: float = 0.0,
):
    if float(radius_voxels) <= 0.0 and float(surface_attract_radius) <= 0.0:
        return velocity_dir, velocity_weight, trace_dist, trace_stop, trace_loss_weight, None, None

    if velocity_dir.ndim != 5 or velocity_dir.shape[1] != 3:
        raise ValueError(f"velocity_dir must have shape [B, 3, D, H, W], got {tuple(velocity_dir.shape)}")
    if velocity_weight.ndim != 5 or velocity_weight.shape[1] != 1:
        raise ValueError(
            f"velocity_weight must have shape [B, 1, D, H, W], got {tuple(velocity_weight.shape)}"
        )

    source_weight = trace_loss_weight if trace_loss_weight is not None else velocity_weight
    dilated_dirs = []
    dilated_weights = []
    dilated_trace_dists = [] if trace_dist is not None else None
    dilated_trace_stops = [] if trace_stop is not None else None
    surface_attracts = [] if float(surface_attract_radius) > 0.0 else None
    surface_attract_weights = [] if float(surface_attract_radius) > 0.0 else None

    for b in range(velocity_dir.shape[0]):
        if not bool((source_weight[b, 0] > 0.5).any().item()):
            dilated_dirs.append(velocity_dir[b])
            dilated_weights.append(velocity_weight[b])
            if trace_dist is not None:
                dilated_trace_dists.append(trace_dist[b])
            if trace_stop is not None:
                dilated_trace_stops.append(trace_stop[b])
            if surface_attracts is not None:
                surface_attracts.append(torch.zeros_like(velocity_dir[b]))
                surface_attract_weights.append(torch.zeros_like(velocity_weight[b]))
            continue

        dist_b = trace_dist[b].contiguous() if trace_dist is not None else None
        stop_b = trace_stop[b].contiguous() if trace_stop is not None else None
        if velocity_dir.is_cuda:
            (
                dilated_dir,
                dilated_weight,
                dist_b,
                stop_b,
                attract_b,
                attract_weight_b,
            ) = _cupyx_dilate_trace_targets(
                velocity_dir[b].contiguous(),
                source_weight[b].contiguous(),
                radius_voxels,
                trace_dist=dist_b,
                trace_stop=stop_b,
                surface_attract_radius=surface_attract_radius,
            )
        else:
            (
                dilated_dir,
                dilated_weight,
                dist_b,
                stop_b,
                attract_b,
                attract_weight_b,
            ) = _scipy_dilate_trace_targets(
                velocity_dir[b],
                source_weight[b],
                radius_voxels,
                trace_dist=dist_b,
                trace_stop=stop_b,
                surface_attract_radius=surface_attract_radius,
            )
        dilated_dirs.append(dilated_dir)
        dilated_weights.append(dilated_weight)
        if trace_dist is not None:
            dilated_trace_dists.append(dist_b)
        if trace_stop is not None:
            dilated_trace_stops.append(stop_b)
        if surface_attracts is not None:
            surface_attracts.append(attract_b)
            surface_attract_weights.append(attract_weight_b)

    velocity_dir = torch.stack(dilated_dirs, dim=0)
    velocity_weight = torch.stack(dilated_weights, dim=0)
    if trace_dist is not None:
        trace_dist = torch.stack(dilated_trace_dists, dim=0)
    if trace_stop is not None:
        trace_stop = torch.stack(dilated_trace_stops, dim=0)
    if trace_loss_weight is not None:
        trace_loss_weight = velocity_weight
    surface_attract = torch.stack(surface_attracts, dim=0) if surface_attracts is not None else None
    surface_attract_weight = (
        torch.stack(surface_attract_weights, dim=0) if surface_attract_weights is not None else None
    )
    return (
        velocity_dir,
        velocity_weight,
        trace_dist,
        trace_stop,
        trace_loss_weight,
        surface_attract,
        surface_attract_weight,
    )


def _fraction_disp_within_distance_torch(disp: torch.Tensor, source_mask: torch.Tensor, max_distance_voxels: float):
    mask = source_mask > 0.5
    if not bool(mask.any().item()):
        return 0.0
    vecs = disp[:, mask].T
    if vecs.numel() == 0:
        return 0.0
    finite = torch.isfinite(vecs).all(dim=1)
    vecs = vecs[finite]
    if vecs.numel() == 0:
        return 0.0
    mags = torch.linalg.vector_norm(vecs, dim=1)
    return float((mags <= float(max_distance_voxels)).float().mean().item())


def _triplet_close_contact_fractions_torch(
    cond_mask: torch.Tensor,
    behind_mask: torch.Tensor,
    front_mask: torch.Tensor,
    behind_disp: torch.Tensor,
    front_disp: torch.Tensor,
    max_distance_voxels: float,
):
    cond_behind = _fraction_disp_within_distance_torch(behind_disp, cond_mask, max_distance_voxels)
    cond_front = _fraction_disp_within_distance_torch(front_disp, cond_mask, max_distance_voxels)
    behind_to_front = _fraction_disp_within_distance_torch(front_disp, behind_mask, max_distance_voxels)
    front_to_behind = _fraction_disp_within_distance_torch(behind_disp, front_mask, max_distance_voxels)
    return cond_behind, cond_front, max(behind_to_front, front_to_behind)


def _median_signed_projection_torch(disp_field: torch.Tensor, mask: torch.Tensor, direction: torch.Tensor):
    vecs = disp_field[:, mask].T
    if vecs.numel() == 0:
        return None
    finite = torch.isfinite(vecs).all(dim=1)
    vecs = vecs[finite]
    if vecs.numel() == 0:
        return None
    mags = torch.linalg.vector_norm(vecs, dim=1)
    vecs = vecs[mags > 1e-6]
    if vecs.numel() == 0:
        return None
    return torch.median(vecs @ direction)


def _align_triplet_targets_to_priors(dense_gt: torch.Tensor, dir_priors: torch.Tensor, cond_gt: torch.Tensor):
    channel_order = torch.tensor(
        [[0, 1]] * dense_gt.shape[0],
        device=dense_gt.device,
        dtype=torch.int64,
    )
    aligned_dense = dense_gt.clone()

    for b in range(dense_gt.shape[0]):
        mask = cond_gt[b] > 0.5
        if not bool(mask.any().item()):
            continue

        prior_vecs = dir_priors[b, 0:3, mask].T
        if prior_vecs.numel() == 0:
            continue
        finite = torch.isfinite(prior_vecs).all(dim=1)
        prior_vecs = prior_vecs[finite]
        if prior_vecs.numel() == 0:
            continue
        n = prior_vecs.mean(dim=0)
        n_norm = torch.linalg.vector_norm(n)
        if not bool(torch.isfinite(n_norm).item()) or float(n_norm.item()) <= 1e-6:
            continue
        n = n / n_norm

        s0 = _median_signed_projection_torch(aligned_dense[b, 0:3], mask, n)
        s1 = _median_signed_projection_torch(aligned_dense[b, 3:6], mask, n)
        if s0 is None or s1 is None:
            continue
        if bool((s1 > s0).item()):
            aligned_dense[b] = torch.cat([aligned_dense[b, 3:6].clone(), aligned_dense[b, 0:3].clone()], dim=0)
            channel_order[b] = torch.tensor([1, 0], device=dense_gt.device, dtype=torch.int64)

    return aligned_dense, dir_priors, channel_order


def _maybe_swap_triplet_targets(
    dense_gt: torch.Tensor,
    dir_priors: torch.Tensor | None,
    channel_order: torch.Tensor,
    swap_prob: float,
):
    p = float(swap_prob)
    if p <= 0.0:
        return dense_gt, dir_priors, channel_order
    swap_mask = torch.rand((dense_gt.shape[0],), device=dense_gt.device) < p
    if not bool(swap_mask.any().item()):
        return dense_gt, dir_priors, channel_order

    swapped_dense = dense_gt.clone()
    swapped_dense[swap_mask] = torch.cat(
        [dense_gt[swap_mask, 3:6], dense_gt[swap_mask, 0:3]],
        dim=1,
    )
    swapped_priors = dir_priors
    if dir_priors is not None:
        swapped_priors = dir_priors.clone()
        swapped_priors[swap_mask] = torch.cat(
            [dir_priors[swap_mask, 3:6], dir_priors[swap_mask, 0:3]],
            dim=1,
        )
    swapped_order = channel_order.clone()
    swapped_order[swap_mask] = channel_order[swap_mask][:, [1, 0]]
    return swapped_dense, swapped_priors, swapped_order


def resolve_dataloader_context(config):
    """Resolve multiprocessing context for DataLoader workers."""
    context_name = str(config.get('dataloader_multiprocessing_context', 'auto')).lower()
    if context_name == 'auto':
        context_name = 'fork' if sys.platform.startswith('linux') else 'spawn'
    if context_name not in {'fork', 'spawn', 'forkserver'}:
        raise ValueError(
            "dataloader_multiprocessing_context must be one of "
            "'auto', 'fork', 'spawn', 'forkserver'"
        )
    try:
        return multiprocessing.get_context(context_name)
    except ValueError as exc:
        raise ValueError(
            f"Multiprocessing context {context_name!r} is not available on this platform"
        ) from exc


def collate_with_padding(batch):
    """Collate batch with optional sparse point padding and dense targets."""
    # Stack fixed-size tensors normally
    vol = torch.stack([b['vol'] for b in batch])
    cond = torch.stack([b['cond'] for b in batch])
    result = {
        'vol': vol,
        'cond': cond,
    }
    if 'cond_direction' in batch[0]:
        result['cond_direction'] = [b['cond_direction'] for b in batch]

    has_extrap_surface = 'extrap_surface' in batch[0]
    if has_extrap_surface:
        result['extrap_surface'] = torch.stack([b['extrap_surface'] for b in batch])

    has_sparse_supervision = 'extrap_coords' in batch[0] and 'gt_displacement' in batch[0]
    if has_sparse_supervision:
        coords_list = [b['extrap_coords'] for b in batch]
        disp_list = [b['gt_displacement'] for b in batch]
        weight_list = [
            b['point_weights'] if 'point_weights' in b else torch.ones(len(b['extrap_coords']), dtype=torch.float32)
            for b in batch
        ]
        max_points = max(len(c) for c in coords_list)

        B = len(batch)
        padded_coords = torch.zeros(B, max_points, 3)
        padded_disp = torch.zeros(B, max_points, 3)
        valid_mask = torch.zeros(B, max_points)
        padded_point_weights = torch.zeros(B, max_points)
        padded_point_normals = torch.zeros(B, max_points, 3)
        has_point_normals = 'point_normals' in batch[0]

        for i, (c, d, w) in enumerate(zip(coords_list, disp_list, weight_list)):
            n = len(c)
            padded_coords[i, :n] = c
            padded_disp[i, :n] = d
            valid_mask[i, :n] = 1.0
            padded_point_weights[i, :n] = w
            if has_point_normals:
                padded_point_normals[i, :n] = batch[i]['point_normals']

        result['extrap_coords'] = padded_coords
        result['gt_displacement'] = padded_disp
        result['valid_mask'] = valid_mask
        result['point_weights'] = padded_point_weights
        if has_point_normals:
            result['point_normals'] = padded_point_normals

    if 'dense_gt_displacement' in batch[0]:
        result['dense_gt_displacement'] = torch.stack([b['dense_gt_displacement'] for b in batch])
        if 'dense_loss_weight' in batch[0]:
            result['dense_loss_weight'] = torch.stack([b['dense_loss_weight'] for b in batch])
    if 'flow_dir' in batch[0]:
        result['flow_dir'] = torch.stack([b['flow_dir'] for b in batch])
    if 'flow_dist' in batch[0]:
        result['flow_dist'] = torch.stack([b['flow_dist'] for b in batch])
    if 'velocity_dir' in batch[0]:
        result['velocity_dir'] = torch.stack([b['velocity_dir'] for b in batch])
    if 'velocity_loss_weight' in batch[0]:
        result['velocity_loss_weight'] = torch.stack([b['velocity_loss_weight'] for b in batch])
    if 'trace_dist' in batch[0]:
        result['trace_dist'] = torch.stack([b['trace_dist'] for b in batch])
    if 'trace_stop' in batch[0]:
        result['trace_stop'] = torch.stack([b['trace_stop'] for b in batch])
    if 'trace_loss_weight' in batch[0]:
        result['trace_loss_weight'] = torch.stack([b['trace_loss_weight'] for b in batch])
    if 'trace_validity' in batch[0]:
        result['trace_validity'] = torch.stack([b['trace_validity'] for b in batch])
    if 'trace_validity_weight' in batch[0]:
        result['trace_validity_weight'] = torch.stack([b['trace_validity_weight'] for b in batch])
    if 'neighbor_sheet_present' in batch[0]:
        result['neighbor_sheet_present'] = torch.stack([b['neighbor_sheet_present'] for b in batch])
    if 'surface_attract' in batch[0]:
        result['surface_attract'] = torch.stack([b['surface_attract'] for b in batch])
    if 'surface_attract_weight' in batch[0]:
        result['surface_attract_weight'] = torch.stack([b['surface_attract_weight'] for b in batch])
    if 'dir_priors' in batch[0]:
        result['dir_priors'] = torch.stack([b['dir_priors'] for b in batch])
    if 'triplet_channel_order' in batch[0]:
        result['triplet_channel_order'] = torch.stack([b['triplet_channel_order'] for b in batch])

    # Source masks used by trainer-side dense target generation. Keeping dense
    # EDT out of dataloader workers avoids cupyx defaulting every worker to GPU 0.
    for key in ('cond_gt', 'masked_seg', 'behind_seg', 'front_seg', 'neighbor_seg'):
        if key in batch[0]:
            result[key] = torch.stack([b[key] for b in batch])

    # Optional SDT
    if 'sdt' in batch[0]:
        result['sdt'] = torch.stack([b['sdt'] for b in batch])

    # Optional heatmap target
    if 'heatmap_target' in batch[0]:
        result['heatmap_target'] = torch.stack([b['heatmap_target'] for b in batch])

    # Optional segmentation target (full segmentation + skeleton)
    if 'segmentation' in batch[0]:
        result['segmentation'] = torch.stack([b['segmentation'] for b in batch])
        result['segmentation_skel'] = torch.stack([b['segmentation_skel'] for b in batch])

    # Optional other_wraps
    if 'other_wraps' in batch[0]:
        result['other_wraps'] = torch.stack([b['other_wraps'] for b in batch])

    return result


def prepare_batch(
    batch,
    use_sdt=False,
    use_heatmap=False,
    use_segmentation=False,
    use_growth_direction_channels=False,
    dense_target_builder=None,
    velocity_target_builder=None,
):
    """Prepare batch tensors for training."""
    if dense_target_builder is not None:
        dense_target_builder(batch)
    if velocity_target_builder is not None:
        velocity_target_builder(batch)

    vol = batch['vol'].unsqueeze(1)  # [B, 1, D, H, W]
    cond = batch['cond'].unsqueeze(1)  # [B, 1, D, H, W]

    input_list = [vol, cond]
    if use_growth_direction_channels:
        if 'cond_direction' not in batch:
            raise ValueError("use_growth_direction_channels=True but batch is missing cond_direction")
        input_list.append(
            make_growth_direction_tensor(
                batch['cond_direction'],
                vol.shape[2:],
                device=vol.device,
                dtype=vol.dtype,
            )
        )
    if 'dir_priors' in batch:
        input_list.append(batch['dir_priors'])  # [B, 6, D, H, W]
    if 'extrap_surface' in batch:
        extrap_surf = batch['extrap_surface'].unsqueeze(1)  # [B, 1, D, H, W]
        input_list.append(extrap_surf)
    if 'other_wraps' in batch:
        other_wraps = batch['other_wraps'].unsqueeze(1)  # [B, 1, D, H, W]
        input_list.append(other_wraps)

    inputs = torch.cat(input_list, dim=1)

    extrap_coords = batch.get('extrap_coords', None)
    gt_displacement = batch.get('gt_displacement', None)
    valid_mask = batch.get('valid_mask', None)
    point_weights = None
    if valid_mask is not None:
        point_weights = batch['point_weights'] if 'point_weights' in batch else torch.ones_like(valid_mask)
    point_normals = batch.get('point_normals', None)

    dense_gt_displacement = batch.get('dense_gt_displacement', None)  # [B, C, D, H, W]
    dense_loss_weight = batch.get('dense_loss_weight', None)  # [B, 1, D, H, W]
    flow_dir_target = batch.get('flow_dir', None)  # [B, C, D, H, W]
    flow_dist_target = batch.get('flow_dist', None)  # [B, C//3, D, H, W]
    velocity_dir_target = batch.get('velocity_dir', None)  # [B, 3, D, H, W]
    velocity_loss_weight = batch.get('velocity_loss_weight', None)  # [B, 1, D, H, W]
    trace_dist_target = batch.get('trace_dist', None)  # [B, 1, D, H, W]
    trace_stop_target = batch.get('trace_stop', None)  # [B, 1, D, H, W]
    trace_loss_weight = batch.get('trace_loss_weight', None)  # [B, 1, D, H, W]
    trace_validity_target = batch.get('trace_validity', None)  # [B, 1, D, H, W]
    trace_validity_weight = batch.get('trace_validity_weight', None)  # [B, 1, D, H, W]
    surface_attract_target = batch.get('surface_attract', None)  # [B, 3, D, H, W]
    surface_attract_weight = batch.get('surface_attract_weight', None)  # [B, 1, D, H, W]

    sdt_target = batch['sdt'].unsqueeze(1) if use_sdt and 'sdt' in batch else None  # [B, 1, D, H, W]
    heatmap_target = batch['heatmap_target'].unsqueeze(1) if use_heatmap and 'heatmap_target' in batch else None  # [B, 1, D, H, W]

    seg_target = None
    seg_skel = None
    if use_segmentation and 'segmentation' in batch:
        seg_target = batch['segmentation'].unsqueeze(1)  # [B, 1, D, H, W]
        seg_skel = batch['segmentation_skel'].unsqueeze(1)  # [B, 1, D, H, W]

    return (
        inputs,
        extrap_coords,
        gt_displacement,
        valid_mask,
        point_weights,
        point_normals,
        dense_gt_displacement,
        dense_loss_weight,
        flow_dir_target,
        flow_dist_target,
        velocity_dir_target,
        velocity_loss_weight,
        trace_dist_target,
        trace_stop_target,
        trace_loss_weight,
        trace_validity_target,
        trace_validity_weight,
        surface_attract_target,
        surface_attract_weight,
        sdt_target,
        heatmap_target,
        seg_target,
        seg_skel,
    )


@click.command()
@click.argument('config_path', type=click.Path(exists=True))
def train(config_path):
    """Train a displacement field prediction model with optional SDT."""

    with open(config_path, 'r') as f:
        config = json.load(f)

    user_set_triplet_direction_priors = 'use_triplet_direction_priors' in config
    user_set_defer_velocity_dilation = 'defer_velocity_dilation_to_trainer' in config
    user_set_defer_trace_dilation = 'defer_trace_dilation_to_trainer' in config
    user_set_defer_trace_validity = 'defer_trace_validity_to_trainer' in config
    setdefault_rowcol_cond_dataset_config(config)
    config.setdefault('defer_dense_targets_to_trainer', True)
    if not user_set_defer_velocity_dilation:
        config['defer_velocity_dilation_to_trainer'] = True
    if not user_set_defer_trace_dilation:
        config['defer_trace_dilation_to_trainer'] = True
    if not user_set_defer_trace_validity:
        config['defer_trace_validity_to_trainer'] = True

    # Defaults
    triplet_mode = bool(config.get('use_triplet_wrap_displacement', False))
    if not user_set_triplet_direction_priors:
        config['use_triplet_direction_priors'] = triplet_mode
    validate_rowcol_cond_dataset_config(config)

    use_triplet_direction_priors = bool(config.get('use_triplet_direction_priors', False))
    if triplet_mode and use_triplet_direction_priors:
        default_in_channels = 8
    else:
        default_in_channels = (
            2
            + int(config.get('use_other_wrap_cond', False))
            + (
                growth_direction_channel_count()
                if bool(config.get('use_growth_direction_channels', False))
                else 0
            )
        )
    config.setdefault('in_channels', default_in_channels)
    if int(config['in_channels']) != default_in_channels:
        if triplet_mode and use_triplet_direction_priors:
            raise ValueError(
                f"in_channels={config['in_channels']} does not match configured inputs "
                "(expected 8 from triplet mode with direction priors: vol+cond+6 direction channels)"
            )
        raise ValueError(
            f"in_channels={config['in_channels']} does not match configured inputs "
            f"(expected {default_in_channels} from use_other_wrap_cond={config.get('use_other_wrap_cond', False)}, "
            f"use_growth_direction_channels={config.get('use_growth_direction_channels', False)})"
        )
    config.setdefault('step_count', 1)  # Required by make_model
    config.setdefault('num_iterations', 250000)
    config.setdefault('log_frequency', 100)
    config.setdefault('ckpt_frequency', 5000)
    config.setdefault('grad_clip', 5)
    config.setdefault('learning_rate', 0.01)
    config.setdefault('weight_decay', 3e-5)
    config.setdefault('batch_size', 4)
    config.setdefault('num_workers', 4)
    config.setdefault('val_num_workers', 1)
    config.setdefault('pin_memory', True)
    config.setdefault('non_blocking', True)
    config.setdefault('persistent_workers', True)
    config.setdefault('prefetch_factor', 1)
    config.setdefault('val_prefetch_factor', 1)
    config.setdefault('dataloader_multiprocessing_context', 'auto')
    config.setdefault('seed', 0)
    config.setdefault('use_sdt', False)
    config.setdefault('lambda_sdt', 1.0)
    config.setdefault('use_heatmap_targets', False)
    config.setdefault('lambda_heatmap', 1.0)
    config.setdefault('use_segmentation', False)
    config.setdefault('lambda_segmentation', 1.0)
    config.setdefault('segmentation_loss', {})
    config.setdefault('supervise_conditioning', False)
    config.setdefault('cond_supervision_weight', 0.1)
    config.setdefault('lambda_cond_disp', 0.0)
    config.setdefault('triplet_min_disp_vox', 1.0)
    config.setdefault('lambda_triplet_min_disp', 0.0)
    config.setdefault('use_flow_refinement_targets', False)
    config.setdefault('lambda_flow_dir', 0.1)
    config.setdefault('lambda_flow_dist', 0.1)
    config.setdefault('flow_dist_huber_beta', config.get('displacement_huber_beta', 5.0))
    config.setdefault('use_velocity_targets', False)
    config.setdefault('lambda_velocity_dir', 0.1)
    config.setdefault('velocity_target_mode', 'away_from_conditioning')
    config.setdefault('velocity_target_dilation_radius', 1.0)
    config.setdefault('velocity_target_region', 'full')
    config.setdefault('use_trace_ode_targets', False)
    config.setdefault('lambda_displacement', 1.0)
    config.setdefault('lambda_trace_dist', 0.1)
    config.setdefault('lambda_trace_stop', 0.05)
    config.setdefault('lambda_surface_attract', 0.1)
    config.setdefault('use_trace_validity_targets', False)
    config.setdefault('lambda_trace_validity', 0.0)
    config.setdefault('trace_validity_pos_weight', 1.0)
    config.setdefault('lambda_velocity_smooth', 0.0)
    config.setdefault('velocity_smooth_normalize', True)
    config.setdefault('lambda_trace_integration', 0.0)
    config.setdefault('trace_integration_steps', 2)
    config.setdefault('trace_integration_step_size', 1.0)
    config.setdefault('trace_integration_max_points', 2048)
    config.setdefault('trace_integration_min_weight', 0.5)
    config.setdefault('trace_integration_detach_steps', False)
    config.setdefault('trace_dist_huber_beta', config.get('displacement_huber_beta', 5.0))
    config.setdefault('trace_stop_pos_weight', 4.0)
    config.setdefault('surface_attract_huber_beta', config.get('displacement_huber_beta', 5.0))
    config.setdefault('trace_target_mode', 'away_from_conditioning')
    config.setdefault('trace_target_region', 'full')
    config.setdefault('trace_target_dilation_radius', config.get('velocity_target_dilation_radius', 1.0))
    config.setdefault('defer_trace_dilation_to_trainer', True)
    config.setdefault('defer_trace_validity_to_trainer', True)
    config.setdefault('trace_stop_radius', 1.0)
    config.setdefault('displacement_supervision', 'vector')  # 'vector' or 'normal_scalar'
    config.setdefault('displacement_loss_type', 'vector_l2')
    config.setdefault('displacement_huber_beta', 5.0)
    config.setdefault('normal_loss_type', 'normal_huber')
    config.setdefault('normal_loss_beta', config.get('displacement_huber_beta', 5.0))
    config.setdefault('lambda_smooth', 0.0)
    config.setdefault('eval_perturbed_val', False)
    config.setdefault('log_perturbed_val_images', False)
    config.setdefault('val_batches_per_log', 4)
    config.setdefault('log_at_step_zero', False)
    config.setdefault('ckpt_at_step_zero', False)
    config.setdefault('use_accelerate_dynamo', False)
    config.setdefault('wandb_resume', False)
    config.setdefault('wandb_resume_mode', 'allow')
    config.setdefault('profile_data_time', False)
    config.setdefault('profile_step_time', False)
    config.setdefault('profile_first_iteration', True)
    config.setdefault('profile_log_every', 100)
    config.setdefault('compile_model', True)
    config.setdefault('separate_eager_eval_for_logging', True)

    displacement_out_channels = 6 if triplet_mode else 3
    use_displacement_head = (
        float(config.get('lambda_displacement', 1.0)) > 0.0
        or float(config.get('lambda_smooth', 0.0)) > 0.0
        or float(config.get('lambda_cond_disp', 0.0)) > 0.0
        or float(config.get('lambda_triplet_min_disp', 0.0)) > 0.0
    )

    # Build targets dict based on config. The displacement head is omitted for
    # trace/aux-only runs where no configured loss consumes it.
    targets = {}
    if use_displacement_head:
        targets['displacement'] = {'out_channels': displacement_out_channels, 'activation': 'none'}
    use_flow_refinement_targets = bool(config.get('use_flow_refinement_targets', False))
    if use_flow_refinement_targets:
        targets['flow_dir'] = {'out_channels': displacement_out_channels, 'activation': 'none'}
        targets['flow_dist'] = {'out_channels': displacement_out_channels // 3, 'activation': 'none'}
    use_trace_ode_targets = bool(config.get('use_trace_ode_targets', False))
    use_velocity_targets = bool(config.get('use_velocity_targets', False)) or use_trace_ode_targets
    use_trace_dist_head = use_trace_ode_targets and float(config.get('lambda_trace_dist', 0.0)) > 0.0
    use_trace_stop_head = use_trace_ode_targets and float(config.get('lambda_trace_stop', 0.0)) > 0.0
    use_surface_attract_head = use_trace_ode_targets and float(config.get('lambda_surface_attract', 0.0)) > 0.0
    use_trace_validity_head = (
        use_trace_ode_targets
        and (
            bool(config.get('use_trace_validity_targets', False))
            or float(config.get('lambda_trace_validity', 0.0)) > 0.0
        )
    )
    if use_velocity_targets:
        targets['velocity_dir'] = {'out_channels': 3, 'activation': 'none'}
    if use_trace_dist_head:
        targets['trace_dist'] = {'out_channels': 1, 'activation': 'none'}
    if use_trace_stop_head:
        targets['trace_stop'] = {'out_channels': 1, 'activation': 'none'}
    if use_surface_attract_head:
        targets['surface_attract'] = {'out_channels': 3, 'activation': 'none'}
    if use_trace_validity_head:
        targets['trace_validity'] = {'out_channels': 1, 'activation': 'none'}
    use_sdt = config.get('use_sdt', False)
    if use_sdt:
        targets['sdt'] = {'out_channels': 1, 'activation': 'none'}
    use_heatmap = config.get('use_heatmap_targets', False)
    if use_heatmap:
        targets['heatmap'] = {'out_channels': 1, 'activation': 'none'}
    use_segmentation = config.get('use_segmentation', False)
    if use_segmentation:
        targets['segmentation'] = {'out_channels': 2, 'activation': 'none'}
    if not targets:
        raise ValueError("No model output heads are enabled; set at least one supervised target/loss")
    config['targets'] = targets

    out_dir = config['out_dir']
    os.makedirs(out_dir, exist_ok=True)

    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])

    dynamo_plugin = None
    if config.get('use_accelerate_dynamo', False):
        dynamo_plugin = TorchDynamoPlugin(
            backend="inductor",
            mode="default",
            fullgraph=False,
            dynamic=False,
            use_regional_compilation=False,
        )

    dataloader_config = accelerate.DataLoaderConfiguration(
        non_blocking=bool(config.get('non_blocking', True))
    )

    accelerator = accelerate.Accelerator(
        mixed_precision=config.get('mixed_precision', 'no'),
        gradient_accumulation_steps=config.get('grad_acc_steps', 1),
        dynamo_plugin=dynamo_plugin,
        dataloader_config=dataloader_config,
    )

    preloaded_ckpt = None
    wandb_resume = bool(config.get('wandb_resume', False))
    wandb_run_id = config.get('wandb_run_id')
    if wandb_resume and wandb_run_id is None and 'load_ckpt' in config:
        preloaded_ckpt = torch.load(config['load_ckpt'], map_location='cpu', weights_only=False)
        wandb_run_id = preloaded_ckpt.get('wandb_run_id')
        if wandb_run_id is None:
            ckpt_config = preloaded_ckpt.get('config', {})
            if isinstance(ckpt_config, dict):
                wandb_run_id = ckpt_config.get('wandb_run_id')
        if wandb_run_id is not None:
            config['wandb_run_id'] = wandb_run_id

    if 'wandb_project' in config and accelerator.is_main_process:
        wandb_kwargs = {
            'project': config['wandb_project'],
            'entity': config.get('wandb_entity', None),
            'config': config,
        }
        if wandb_resume:
            wandb_kwargs['resume'] = config.get('wandb_resume_mode', 'allow')
            if wandb_run_id is not None:
                wandb_kwargs['id'] = wandb_run_id
        wandb.init(**wandb_kwargs)
        if wandb.run is not None:
            config['wandb_run_id'] = wandb.run.id

    # Setup SDT loss if enabled
    sdt_loss_fn = None
    if use_sdt:
        from vesuvius.models.training.loss.losses import SignedDistanceLoss
        sdt_loss_fn = SignedDistanceLoss(
            beta=config.get('sdt_beta', 1.0),
            eikonal=config.get('sdt_eikonal', True),
            eikonal_weight=config.get('sdt_eikonal_weight', 0.01),
            laplacian=config.get('sdt_laplacian', True),
            laplacian_weight=config.get('sdt_laplacian_weight', 0.01),
            surface_sigma=config.get('sdt_surface_sigma', 3.0),
            reduction='mean',
        )

    lambda_sdt = config.get('lambda_sdt', 1.0)
    lambda_heatmap = config.get('lambda_heatmap', 1.0)
    lambda_segmentation = config.get('lambda_segmentation', 1.0)
    lambda_cond_disp = config.get('lambda_cond_disp', 0.0)
    triplet_min_disp_vox = float(config.get('triplet_min_disp_vox', 1.0))
    lambda_triplet_min_disp = float(config.get('lambda_triplet_min_disp', 0.0))
    lambda_flow_dir = float(config.get('lambda_flow_dir', 0.0))
    lambda_flow_dist = float(config.get('lambda_flow_dist', 0.0))
    lambda_velocity_dir = float(config.get('lambda_velocity_dir', 0.0))
    lambda_displacement = float(config.get('lambda_displacement', 1.0))
    lambda_trace_dist = float(config.get('lambda_trace_dist', 0.0))
    lambda_trace_stop = float(config.get('lambda_trace_stop', 0.0))
    lambda_surface_attract = float(config.get('lambda_surface_attract', 0.0))
    lambda_trace_validity = float(config.get('lambda_trace_validity', 0.0))
    trace_validity_pos_weight = float(config.get('trace_validity_pos_weight', 1.0))
    lambda_velocity_smooth = float(config.get('lambda_velocity_smooth', 0.0))
    lambda_trace_integration = float(config.get('lambda_trace_integration', 0.0))
    trace_integration_steps = int(config.get('trace_integration_steps', 2))
    trace_integration_step_size = float(config.get('trace_integration_step_size', 1.0))
    trace_integration_max_points = int(config.get('trace_integration_max_points', 2048))
    trace_integration_min_weight = float(config.get('trace_integration_min_weight', 0.5))
    trace_integration_detach_steps = bool(config.get('trace_integration_detach_steps', False))
    surface_attract_target_mode = str(config.get('surface_attract_target_mode', 'dense_edt')).lower()
    trace_surface_attract_radius = float(config.get('trace_surface_attract_radius', 0.0))
    needs_dense_targets = (
        lambda_displacement > 0.0
        or (
            use_flow_refinement_targets
            and (lambda_flow_dir > 0.0 or lambda_flow_dist > 0.0)
        )
        or (
            use_trace_ode_targets
            and lambda_surface_attract > 0.0
            and surface_attract_target_mode == 'dense_edt'
        )
    )
    velocity_target_dilation_radius = float(config.get('velocity_target_dilation_radius', 0.0))
    defer_velocity_dilation_to_trainer = bool(config.get('defer_velocity_dilation_to_trainer', False))
    trace_target_dilation_radius = float(config.get('trace_target_dilation_radius', velocity_target_dilation_radius))
    defer_trace_dilation_to_trainer = bool(config.get('defer_trace_dilation_to_trainer', False))
    defer_trace_validity_to_trainer = bool(config.get('defer_trace_validity_to_trainer', False))
    trace_validity_positive_radius = float(config.get('trace_validity_positive_radius', 2.0))
    trace_validity_negative_radius = float(config.get('trace_validity_negative_radius', 3.0))
    trace_validity_margin = float(config.get('trace_validity_margin', 3.0))
    trace_validity_background_weight = float(config.get('trace_validity_background_weight', 0.25))
    trace_dist_huber_beta = float(config.get('trace_dist_huber_beta', config.get('displacement_huber_beta', 5.0)))
    trace_stop_pos_weight = float(config.get('trace_stop_pos_weight', 1.0))
    surface_attract_huber_beta = float(config.get('surface_attract_huber_beta', config.get('displacement_huber_beta', 5.0)))
    lambda_smooth = config.get('lambda_smooth', 0.0)
    if config.get('supervise_conditioning', False) and lambda_cond_disp > 0.0:
        raise ValueError(
            "supervise_conditioning=True adds nonzero conditioning displacement targets, "
            "which conflicts with lambda_cond_disp > 0 (zero-displacement penalty on conditioning voxels). "
            "Set lambda_cond_disp to 0 when supervise_conditioning is enabled."
        )
    mask_cond_from_seg_loss = config.get('mask_cond_from_seg_loss', False)
    use_dense_displacement = bool(config.get('use_dense_displacement', False))
    triplet_direction_prior_mask = str(config.get('triplet_direction_prior_mask', 'cond')).lower()
    triplet_random_channel_swap_prob = float(config.get('triplet_random_channel_swap_prob', 0.5))
    if triplet_min_disp_vox < 0:
        raise ValueError(f"triplet_min_disp_vox must be >= 0, got {triplet_min_disp_vox}")
    if lambda_triplet_min_disp > 0.0 and not triplet_mode:
        raise ValueError("lambda_triplet_min_disp > 0 requires use_triplet_wrap_displacement=True")
    if use_flow_refinement_targets and triplet_mode:
        raise ValueError("use_flow_refinement_targets=True is currently supported only for regular single-wrap dense mode")
    if use_velocity_targets and triplet_mode:
        raise ValueError("velocity/trace targets are currently supported only for regular row/col split mode")
    if lambda_velocity_smooth > 0.0 and not use_velocity_targets:
        raise ValueError("lambda_velocity_smooth > 0 requires velocity/trace targets")
    if lambda_trace_integration > 0.0 and not use_velocity_targets:
        raise ValueError("lambda_trace_integration > 0 requires velocity/trace targets")
    if trace_integration_steps < 0:
        raise ValueError(f"trace_integration_steps must be >= 0, got {trace_integration_steps}")
    if trace_integration_step_size < 0.0:
        raise ValueError(f"trace_integration_step_size must be >= 0, got {trace_integration_step_size}")
    if trace_integration_max_points < 0:
        raise ValueError(f"trace_integration_max_points must be >= 0, got {trace_integration_max_points}")
    disp_supervision = str(config.get('displacement_supervision', 'vector')).lower()
    if not use_dense_displacement:
        raise ValueError(
            "rowcol_cond training now requires use_dense_displacement=True."
        )
    if disp_supervision == 'normal_scalar':
        raise ValueError("displacement_supervision='normal_scalar' is not supported in dense-only rowcol_cond training")
    disp_loss_type = config.get('displacement_loss_type', 'vector_l2')
    disp_huber_beta = config.get('displacement_huber_beta', 5.0)
    flow_dist_huber_beta = float(config.get('flow_dist_huber_beta', disp_huber_beta))
    normal_loss_type = str(config.get('normal_loss_type', 'normal_huber')).lower()
    normal_loss_beta = float(config.get('normal_loss_beta', disp_huber_beta))
    if normal_loss_type in {'huber', 'l2', 'l1'}:
        normal_loss_type = f'normal_{normal_loss_type}'

    dense_target_warning_counts = {
        'triplet_close_zeroed': 0,
        'triplet_band_zeroed': 0,
    }

    def _warn_dense_target_once(key, message):
        dense_target_warning_counts[key] += 1
        if dense_target_warning_counts[key] <= 5 and accelerator.is_local_main_process:
            accelerator.print(message)

    def build_dense_targets_on_device(batch):
        """Create dense displacement targets on the current batch device."""
        if 'dense_gt_displacement' in batch:
            if use_trace_ode_targets and 'surface_attract' not in batch:
                batch['surface_attract'] = batch['dense_gt_displacement']
            return

        if triplet_mode:
            required = ('cond_gt', 'behind_seg', 'front_seg')
            missing = [k for k in required if k not in batch]
            if missing:
                raise ValueError(f"Missing deferred triplet target source masks: {missing}")

            cond_gt = batch['cond_gt']
            behind_seg = batch['behind_seg']
            front_seg = batch['front_seg']
            dense_fields = []
            dense_weights = []
            need_band = str(config.get('triplet_dense_weight_mode', 'band')).lower() in {'band', 'all_band_boost'}
            weight_mode = str(config.get('triplet_dense_weight_mode', 'band')).lower()

            for b in range(cond_gt.shape[0]):
                if need_band:
                    behind_disp, _, d_behind = _compute_dense_displacement_field_torch(
                        behind_seg[b],
                        return_weights=False,
                        return_distances=True,
                    )
                    front_disp, _, d_front = _compute_dense_displacement_field_torch(
                        front_seg[b],
                        return_weights=False,
                        return_distances=True,
                    )
                else:
                    behind_disp, _ = _compute_dense_displacement_field_torch(
                        behind_seg[b],
                        return_weights=False,
                    )
                    front_disp, _ = _compute_dense_displacement_field_torch(
                        front_seg[b],
                        return_weights=False,
                    )
                    d_behind = None
                    d_front = None

                if behind_disp is None or front_disp is None:
                    raise RuntimeError("Deferred triplet dense displacement field unavailable")

                dense_weight = torch.ones(
                    (1, *behind_seg.shape[1:]),
                    device=behind_seg.device,
                    dtype=torch.float32,
                )

                if bool(config.get('triplet_close_check_enabled', True)):
                    close_fracs = _triplet_close_contact_fractions_torch(
                        cond_gt[b],
                        behind_seg[b],
                        front_seg[b],
                        behind_disp,
                        front_disp,
                        float(config.get('triplet_close_distance_voxels', 1.0)),
                    )
                    if max(close_fracs) > float(config.get('triplet_close_fraction_threshold', 0.05)):
                        dense_weight.zero_()
                        _warn_dense_target_once(
                            'triplet_close_zeroed',
                            "Deferred triplet target was close-contact rejected; "
                            "zeroing its dense loss weight because dataloader resampling has already completed.",
                        )

                if need_band:
                    if d_behind is None or d_front is None:
                        if weight_mode == 'band':
                            dense_weight.zero_()
                    else:
                        band_np = create_band_mask(
                            cond_bin_full=(cond_gt[b].detach().cpu().numpy() > 0.5),
                            d_front_work=d_front.detach().cpu().numpy(),
                            d_behind_work=d_behind.detach().cpu().numpy(),
                            front_disp_work=front_disp.detach().cpu().numpy(),
                            behind_disp_work=behind_disp.detach().cpu().numpy(),
                            band_pct=min(100.0, max(1.0, float(config.get('triplet_band_distance_percentile', 95.0)))),
                            band_padding=max(0.0, float(config.get('triplet_band_padding_voxels', 4.0))),
                            cc_structure_26=np.ones((3, 3, 3), dtype=np.uint8),
                            closing_structure_3=np.ones((3, 3, 3), dtype=bool),
                        )
                        if band_np is None:
                            if weight_mode == 'band':
                                dense_weight.zero_()
                                _warn_dense_target_once(
                                    'triplet_band_zeroed',
                                    "Deferred triplet band mask was unavailable; zeroing that sample's dense loss weight.",
                                )
                        elif weight_mode == 'all_band_boost':
                            band = torch.from_numpy(band_np).to(device=behind_seg.device, dtype=torch.float32)
                            boost = float(config.get('triplet_band_boost_weight', 2.0))
                            dense_weight = dense_weight + (boost - 1.0) * band.unsqueeze(0)
                        else:
                            dense_weight = torch.from_numpy(band_np[None]).to(
                                device=behind_seg.device,
                                dtype=torch.float32,
                            )

                dense_fields.append(torch.cat([behind_disp, front_disp], dim=0))
                dense_weights.append(dense_weight)

            dense_gt = torch.stack(dense_fields, dim=0)
            dense_loss_weight = torch.stack(dense_weights, dim=0)

            dir_priors = batch.get('dir_priors', None)
            if dir_priors is not None:
                dense_gt, dir_priors, channel_order = _align_triplet_targets_to_priors(
                    dense_gt,
                    dir_priors,
                    cond_gt,
                )
            else:
                channel_order = torch.tensor(
                    [[0, 1]] * dense_gt.shape[0],
                    device=dense_gt.device,
                    dtype=torch.int64,
                )

            dense_gt, dir_priors, channel_order = _maybe_swap_triplet_targets(
                dense_gt,
                dir_priors,
                channel_order,
                triplet_random_channel_swap_prob,
            )
            batch['dense_gt_displacement'] = dense_gt
            batch['dense_loss_weight'] = dense_loss_weight
            batch['triplet_channel_order'] = channel_order
            if dir_priors is not None:
                batch['dir_priors'] = dir_priors
            return

        if 'cond_gt' not in batch or 'masked_seg' not in batch:
            raise ValueError("Missing deferred split target source masks: ['cond_gt', 'masked_seg']")

        full_dense_surface = torch.maximum(batch['masked_seg'], batch['cond_gt'])
        dense_fields = []
        dense_weights = []
        dense_distances = []
        for b in range(full_dense_surface.shape[0]):
            if use_flow_refinement_targets:
                dense_disp, dense_weight, dense_dist = _compute_dense_displacement_field_torch(
                    full_dense_surface[b],
                    return_distances=True,
                )
                dense_distances.append(dense_dist)
            else:
                dense_disp, dense_weight = _compute_dense_displacement_field_torch(
                    full_dense_surface[b],
                    return_weights=False,
                )
            if dense_disp is None:
                raise RuntimeError("Deferred split dense displacement field unavailable")
            dense_fields.append(dense_disp)
            if dense_weight is not None:
                dense_weights.append(dense_weight)

        dense_gt = torch.stack(dense_fields, dim=0)
        batch['dense_gt_displacement'] = dense_gt
        if dense_weights:
            batch['dense_loss_weight'] = torch.stack(dense_weights, dim=0)
        if use_flow_refinement_targets:
            dense_dist = torch.stack(dense_distances, dim=0)
            disp_norm = torch.linalg.vector_norm(dense_gt, dim=1, keepdim=True)
            batch['flow_dir'] = dense_gt / disp_norm.clamp(min=1e-6)
            batch['flow_dist'] = dense_dist.unsqueeze(1)
        if use_trace_ode_targets:
            batch['surface_attract'] = dense_gt

    def build_velocity_targets_on_device(batch):
        """Build EDT-derived velocity/trace targets on the current batch device."""
        if (
            use_trace_validity_head
            and defer_trace_validity_to_trainer
            and 'trace_validity' not in batch
        ):
            if 'cond_gt' not in batch or 'masked_seg' not in batch:
                raise ValueError("Trace validity deferral requires cond_gt and masked_seg in the batch")
            labels = []
            weights = []
            neighbor_present_values = []
            neighbor_seg = batch.get('neighbor_seg', None)
            target_mask = torch.maximum(batch['cond_gt'], batch['masked_seg']) > 0.5
            for b in range(target_mask.shape[0]):
                d_target = _distance_transform_distances_torch(target_mask[b])
                if d_target is None:
                    labels.append(torch.zeros_like(target_mask[b], dtype=torch.float32))
                    weights.append(torch.zeros_like(target_mask[b], dtype=torch.float32))
                    neighbor_present_values.append(torch.tensor(0.0, device=target_mask.device, dtype=torch.float32))
                    continue

                label = torch.zeros_like(d_target, dtype=torch.float32)
                weight = torch.zeros_like(d_target, dtype=torch.float32)
                pos = d_target <= trace_validity_positive_radius

                has_neighbor = (
                    neighbor_seg is not None
                    and bool((neighbor_seg[b] > 0.5).any().item())
                )
                if has_neighbor:
                    d_neighbor = _distance_transform_distances_torch(neighbor_seg[b])
                    if d_neighbor is None:
                        has_neighbor = False

                if has_neighbor:
                    margin = d_neighbor - d_target
                    pos = pos & (margin >= trace_validity_margin)
                    neg = (d_neighbor <= trace_validity_negative_radius) & (margin < trace_validity_margin)
                    far_neg = d_target >= trace_validity_negative_radius

                    label[pos] = 1.0
                    weight[pos] = 1.0
                    hard_neg = neg & ~pos
                    weight[hard_neg] = 1.0
                    background_neg = far_neg & ~pos & ~hard_neg
                    if trace_validity_background_weight > 0.0:
                        weight[background_neg] = trace_validity_background_weight
                else:
                    neg = d_target >= trace_validity_negative_radius
                    label[pos] = 1.0
                    weight[pos] = 1.0
                    if trace_validity_background_weight > 0.0:
                        weight[neg & ~pos] = trace_validity_background_weight

                labels.append(label)
                weights.append(weight)
                neighbor_present_values.append(
                    torch.tensor(float(has_neighbor), device=target_mask.device, dtype=torch.float32)
                )

            batch['trace_validity'] = torch.stack(labels, dim=0).unsqueeze(1)
            batch['trace_validity_weight'] = torch.stack(weights, dim=0).unsqueeze(1)
            batch['neighbor_sheet_present'] = torch.stack(neighbor_present_values, dim=0)

        if not use_velocity_targets:
            return
        if 'velocity_dir' not in batch or 'velocity_loss_weight' not in batch:
            raise ValueError("Velocity targets are enabled but missing from the batch")

        if use_trace_ode_targets:
            trace_band_attract_radius = (
                trace_surface_attract_radius
                if (
                    use_surface_attract_head
                    and surface_attract_target_mode == 'trace_band'
                    and 'surface_attract' not in batch
                )
                else 0.0
            )
            if (
                not defer_trace_dilation_to_trainer
                or (trace_target_dilation_radius <= 0.0 and trace_band_attract_radius <= 0.0)
            ):
                return
            (
                velocity_dir,
                velocity_weight,
                trace_dist,
                trace_stop,
                trace_loss_weight,
                surface_attract,
                surface_attract_weight,
            ) = _dilate_trace_targets_torch(
                batch['velocity_dir'],
                batch['velocity_loss_weight'],
                trace_target_dilation_radius,
                trace_dist=batch.get('trace_dist', None),
                trace_stop=batch.get('trace_stop', None),
                trace_loss_weight=batch.get('trace_loss_weight', None),
                surface_attract_radius=trace_band_attract_radius,
            )
            batch['velocity_dir'] = velocity_dir
            batch['velocity_loss_weight'] = velocity_weight
            if trace_dist is not None:
                batch['trace_dist'] = trace_dist
            if trace_stop is not None:
                batch['trace_stop'] = trace_stop
            if trace_loss_weight is not None:
                batch['trace_loss_weight'] = trace_loss_weight
            if surface_attract is not None:
                batch['surface_attract'] = surface_attract
                batch['surface_attract_weight'] = surface_attract_weight
            return

        if not defer_velocity_dilation_to_trainer:
            return
        if velocity_target_dilation_radius <= 0.0:
            return

        velocity_dir, velocity_weight = _dilate_velocity_targets_torch(
            batch['velocity_dir'],
            batch['velocity_loss_weight'],
            velocity_target_dilation_radius,
        )
        batch['velocity_dir'] = velocity_dir
        batch['velocity_loss_weight'] = velocity_weight

    # Setup heatmap loss if enabled (BCE + Dice)
    heatmap_loss_fn = None
    if use_heatmap:
        heatmap_loss_fn = DC_and_BCE_loss(
            bce_kwargs={},
            soft_dice_kwargs={'batch_dice': False, 'ddp': False},
            weight_ce=1.0,
            weight_dice=1.0
        )

    # Setup segmentation loss if enabled (MedialSurfaceRecall)
    seg_loss_fn = None
    if use_segmentation:
        seg_loss_cfg = config.get('segmentation_loss', {})
        soft_dice_kwargs = {
            'batch_dice': seg_loss_cfg.get('batch_dice', False),
            'smooth': seg_loss_cfg.get('smooth', 1e-5),
            'do_bg': seg_loss_cfg.get('do_bg', False),
            'ddp': seg_loss_cfg.get('ddp', False),
        }
        if 'soft_dice_kwargs' in seg_loss_cfg:
            soft_dice_kwargs.update(seg_loss_cfg['soft_dice_kwargs'])
        seg_loss_fn = DC_SkelREC_and_CE_loss(
            soft_dice_kwargs=soft_dice_kwargs,
            soft_skelrec_kwargs={
                'batch_dice': soft_dice_kwargs.get('batch_dice'),
                'smooth': soft_dice_kwargs.get('smooth'),
                'do_bg': soft_dice_kwargs.get('do_bg'),
                'ddp': soft_dice_kwargs.get('ddp'),
            },
            ce_kwargs=seg_loss_cfg.get('ce_kwargs', {}),
            weight_ce=seg_loss_cfg.get('weight_ce', 1),
            weight_dice=seg_loss_cfg.get('weight_dice', 1),
            weight_srec=seg_loss_cfg.get('weight_srec', 1),
            ignore_label=seg_loss_cfg.get('ignore_label', None),
        )

    def compute_displacement_loss(
        disp_pred,
        extrap_coords,
        gt_displacement,
        valid_mask,
        point_weights,
        point_normals,
        dense_gt_displacement,
        dense_loss_weight,
    ):
        if dense_gt_displacement is not None:
            return dense_displacement_loss(
                disp_pred,
                dense_gt_displacement,
                sample_weights=dense_loss_weight,
                loss_type=disp_loss_type,
                beta=disp_huber_beta,
            )

        if extrap_coords is None or gt_displacement is None or valid_mask is None:
            raise ValueError("Sparse displacement supervision expected but sparse batch tensors are missing")

        if disp_supervision == 'normal_scalar':
            if point_normals is None:
                raise ValueError("point_normals missing while displacement_supervision='normal_scalar'")
            return surface_sampled_normal_loss(
                disp_pred, extrap_coords, gt_displacement, point_normals, valid_mask,
                loss_type=normal_loss_type, beta=normal_loss_beta, sample_weights=point_weights
            )

        return surface_sampled_loss(
            disp_pred, extrap_coords, gt_displacement, valid_mask,
            loss_type=disp_loss_type, beta=disp_huber_beta, sample_weights=point_weights
        )

    def compute_flow_refinement_loss(flow_dir_pred, flow_dist_pred, flow_dir_target, flow_dist_target, dense_loss_weight):
        if flow_dir_target is None or flow_dist_target is None:
            raise ValueError("Flow refinement targets are enabled but missing from the batch")
        if dense_loss_weight is None:
            dense_loss_weight = torch.ones_like(flow_dist_target[:, :1])

        pred_dir = F.normalize(flow_dir_pred.float(), dim=1, eps=1e-6)
        target_dir = F.normalize(flow_dir_target.float(), dim=1, eps=1e-6)
        dir_diff = 1.0 - (pred_dir * target_dir).sum(dim=1, keepdim=True).clamp(min=-1.0, max=1.0)
        dir_loss = (dir_diff * dense_loss_weight).sum() / dense_loss_weight.sum().clamp(min=1.0)

        dist_pred = flow_dist_pred.float()
        dist_target = flow_dist_target.float()
        dist_weight = dense_loss_weight
        if dist_weight.shape[1] != dist_pred.shape[1]:
            dist_weight = dist_weight.expand(-1, dist_pred.shape[1], -1, -1, -1)
        dist_diff = F.smooth_l1_loss(
            dist_pred,
            dist_target,
            beta=flow_dist_huber_beta,
            reduction='none',
        )
        dist_loss = (dist_diff * dist_weight).sum() / dist_weight.sum().clamp(min=1.0)
        return dir_loss, dist_loss

    def compute_velocity_dir_loss(velocity_dir_pred, velocity_dir_target, velocity_loss_weight):
        if velocity_dir_target is None or velocity_loss_weight is None:
            raise ValueError("Velocity targets are enabled but missing from the batch")
        pred = F.normalize(velocity_dir_pred.float(), dim=1, eps=1e-6)
        target = F.normalize(velocity_dir_target.float(), dim=1, eps=1e-6)
        dir_diff = 1.0 - (pred * target).sum(dim=1, keepdim=True).clamp(min=-1.0, max=1.0)
        weight = velocity_loss_weight.float()
        return (dir_diff * weight).sum() / weight.sum().clamp(min=1.0)

    def compute_trace_ode_losses(
        trace_dist_pred,
        trace_stop_pred,
        surface_attract_pred,
        trace_dist_target,
        trace_stop_target,
        trace_loss_weight,
        surface_attract_target,
        surface_attract_weight,
    ):
        ref_tensor = next(
            (
                tensor for tensor in (trace_dist_pred, trace_stop_pred, surface_attract_pred)
                if tensor is not None
            ),
            None,
        )
        if ref_tensor is None:
            raise ValueError("Trace ODE loss requested but no enabled trace output head is present")
        zero = ref_tensor.new_zeros(())
        dist_loss = zero
        stop_loss = zero
        attract_loss = zero

        if lambda_trace_dist > 0.0:
            if trace_dist_pred is None or trace_dist_target is None or trace_loss_weight is None:
                raise ValueError("Trace distance loss is enabled but trace distance tensors are missing")
            weight = trace_loss_weight.float()
            dist_pred = F.softplus(trace_dist_pred.float())
            dist_target = trace_dist_target.float()
            dist_diff = F.smooth_l1_loss(
                dist_pred,
                dist_target,
                beta=trace_dist_huber_beta,
                reduction='none',
            )
            dist_loss = (dist_diff * weight).sum() / weight.sum().clamp(min=1.0)

        if lambda_trace_stop > 0.0:
            if trace_stop_pred is None or trace_stop_target is None or trace_loss_weight is None:
                raise ValueError("Trace stop loss is enabled but trace stop tensors are missing")
            weight = trace_loss_weight.float()
            stop_target = trace_stop_target.float().clamp(min=0.0, max=1.0)
            pos_weight = torch.tensor(
                max(trace_stop_pos_weight, 1e-6),
                device=trace_stop_pred.device,
                dtype=torch.float32,
            )
            stop_diff = F.binary_cross_entropy_with_logits(
                trace_stop_pred.float(),
                stop_target,
                pos_weight=pos_weight,
                reduction='none',
            )
            stop_loss = (stop_diff * weight).sum() / weight.sum().clamp(min=1.0)

        if lambda_surface_attract > 0.0:
            if surface_attract_pred is None or surface_attract_target is None:
                raise ValueError("Surface attraction loss is enabled but surface attraction tensors are missing")
            attract_loss = dense_displacement_loss(
                surface_attract_pred,
                surface_attract_target,
                sample_weights=surface_attract_weight,
                loss_type='vector_huber',
                beta=surface_attract_huber_beta,
            )
        return dist_loss, stop_loss, attract_loss

    def compute_trace_validity_loss(trace_validity_pred, trace_validity_target, trace_validity_weight):
        if trace_validity_pred is None or trace_validity_target is None or trace_validity_weight is None:
            raise ValueError("Trace validity loss is enabled but validity tensors are missing")
        target = trace_validity_target.float().clamp(min=0.0, max=1.0)
        weight = trace_validity_weight.float()
        pos_weight = torch.tensor(
            max(trace_validity_pos_weight, 1e-6),
            device=trace_validity_pred.device,
            dtype=torch.float32,
        )
        diff = F.binary_cross_entropy_with_logits(
            trace_validity_pred.float(),
            target,
            pos_weight=pos_weight,
            reduction='none',
        )
        return (diff * weight).sum() / weight.sum().clamp(min=1.0)

    def make_generator(offset=0):
        gen = torch.Generator()
        gen.manual_seed(config['seed'] + accelerator.process_index * 1000 + offset)
        return gen

    # If requested, recompute patch caches exactly once on the main process.
    # Then disable force_recompute so train/val dataset construction just reads cache.
    if config.get('force_recompute_patches', False):
        if accelerator.is_main_process:
            accelerator.print("force_recompute_patches=True: recomputing patch cache once on main process...")
            _recompute_ds = EdtSegDataset(config, apply_augmentation=False, apply_perturbation=False)
            del _recompute_ds
            accelerator.print("Patch cache recompute complete.")
        accelerator.wait_for_everyone()
        config = dict(config)
        config['force_recompute_patches'] = False

    # Train with augmentation, val without
    train_dataset = EdtSegDataset(config, apply_augmentation=True, apply_perturbation=True)
    patch_metadata = train_dataset.export_patch_metadata()
    val_dataset = EdtSegDataset(
        config,
        apply_augmentation=False,
        apply_perturbation=False,
        patch_metadata=patch_metadata,
    )
    val_pert_dataset = None
    if config.get('eval_perturbed_val', False):
        val_pert_config = copy.deepcopy(config)
        val_pert_cfg = dict(val_pert_config.get('cond_local_perturb') or {})
        val_pert_cfg['enabled'] = True
        val_pert_config['cond_local_perturb'] = val_pert_cfg
        val_pert_dataset = EdtSegDataset(
            val_pert_config,
            apply_augmentation=False,
            apply_perturbation=True,
            patch_metadata=patch_metadata,
        )

    # Train/val split by indices
    num_patches = len(train_dataset)
    num_val = max(1, int(num_patches * config.get('val_fraction', 0.1)))
    num_train = num_patches - num_val

    indices = torch.randperm(num_patches, generator=torch.Generator().manual_seed(config['seed'])).tolist()
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]

    def _restrict_dataset_samples(dataset, selected_indices):
        # Subset wrappers break dataset-internal resampling guarantees because
        # EdtSegDataset may resample via self[random_idx] when a sample is invalid.
        dataset.sample_index = [dataset.sample_index[i] for i in selected_indices]
        return dataset

    train_dataset = _restrict_dataset_samples(train_dataset, train_indices)
    val_dataset = _restrict_dataset_samples(val_dataset, val_indices)
    if val_pert_dataset is not None:
        val_pert_dataset = _restrict_dataset_samples(val_pert_dataset, val_indices)

    train_num_workers = max(0, int(config.get('num_workers', 0)))
    val_num_workers = max(0, int(config.get('val_num_workers', 1)))
    pin_memory = bool(config.get('pin_memory', True))
    train_prefetch_factor = max(1, int(config.get('prefetch_factor', 1)))
    val_prefetch_factor = max(1, int(config.get('val_prefetch_factor', train_prefetch_factor)))
    train_persistent_workers = bool(config.get('persistent_workers', True)) and train_num_workers > 0
    val_persistent_workers = bool(config.get('persistent_workers', True)) and val_num_workers > 0

    dataloader_context = None
    if max(train_num_workers, val_num_workers) > 0:
        dataloader_context = resolve_dataloader_context(config)

    train_dataloader_kwargs = dict(
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=train_num_workers,
        worker_init_fn=seed_worker,
        generator=make_generator(0),
        drop_last=True,
        collate_fn=collate_with_padding,
        pin_memory=pin_memory,
    )
    if train_num_workers > 0:
        train_dataloader_kwargs['persistent_workers'] = train_persistent_workers
        train_dataloader_kwargs['prefetch_factor'] = train_prefetch_factor
        train_dataloader_kwargs['multiprocessing_context'] = dataloader_context
    train_dataloader = torch.utils.data.DataLoader(train_dataset, **train_dataloader_kwargs)

    val_dataloader_kwargs = dict(
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=val_num_workers,
        worker_init_fn=seed_worker,
        generator=make_generator(1),
        collate_fn=collate_with_padding,
        pin_memory=pin_memory,
    )
    if val_num_workers > 0:
        val_dataloader_kwargs['persistent_workers'] = val_persistent_workers
        val_dataloader_kwargs['prefetch_factor'] = val_prefetch_factor
        val_dataloader_kwargs['multiprocessing_context'] = dataloader_context
    val_dataloader = torch.utils.data.DataLoader(val_dataset, **val_dataloader_kwargs)

    val_pert_dataloader = None
    if val_pert_dataset is not None:
        val_pert_dataloader_kwargs = dict(
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=val_num_workers,
            worker_init_fn=seed_worker,
            generator=make_generator(2),
            collate_fn=collate_with_padding,
            pin_memory=pin_memory,
        )
        if val_num_workers > 0:
            val_pert_dataloader_kwargs['persistent_workers'] = val_persistent_workers
            val_pert_dataloader_kwargs['prefetch_factor'] = val_prefetch_factor
            val_pert_dataloader_kwargs['multiprocessing_context'] = dataloader_context
        val_pert_dataloader = torch.utils.data.DataLoader(val_pert_dataset, **val_pert_dataloader_kwargs)

    model = make_model(config)

    if config.get('compile_model', True):
        model = torch.compile(model)
        if accelerator.is_main_process:
            accelerator.print("Model compiled with torch.compile")

    scheduler_type = config.setdefault('scheduler', 'diffusers_cosine_warmup')
    scheduler_kwargs = dict(config.setdefault('scheduler_kwargs', {}) or {})
    if scheduler_type in {'diffusers_cosine_warmup', 'warmup_poly', 'cosine_warmup'}:
        scheduler_kwargs.setdefault('warmup_steps', config.get('warmup_steps', 5000))
    config['scheduler_kwargs'] = scheduler_kwargs

    optimizer_config = config.setdefault('optimizer', 'adamw')
    # Handle optimizer being either a string or a dict
    if isinstance(optimizer_config, dict):
        optimizer_type = optimizer_config.get('name', 'adamw')
        optimizer_kwargs = dict(optimizer_config)
        optimizer_kwargs.pop('name', None)
    else:
        optimizer_type = optimizer_config
        optimizer_kwargs = dict(config.setdefault('optimizer_kwargs', {}) or {})
    optimizer_kwargs.setdefault('learning_rate', config.get('learning_rate', 1e-3))
    optimizer_kwargs.setdefault('weight_decay', config.get('weight_decay', 1e-4))
    config['optimizer_kwargs'] = optimizer_kwargs
    optimizer = create_optimizer({'name': optimizer_type, **optimizer_kwargs}, model)

    lr_scheduler = get_scheduler(
        scheduler_type=scheduler_type,
        optimizer=optimizer,
        initial_lr=optimizer_kwargs['learning_rate'],
        max_steps=config['num_iterations'],
        **scheduler_kwargs,
    )

    start_iteration = 0
    if 'load_ckpt' in config:
        accelerator.print(f'Loading checkpoint {config["load_ckpt"]}')
        ckpt = preloaded_ckpt if preloaded_ckpt is not None else torch.load(
            config['load_ckpt'], map_location='cpu', weights_only=False
        )
        state_dict = ckpt['model']
        # Handle compiled model state dict
        model_keys = set(model.state_dict().keys())
        ckpt_has_compile_prefix = any(k.startswith('_orig_mod.') for k in state_dict.keys())
        model_has_compile_prefix = any(k.startswith('_orig_mod.') for k in model_keys)
        if ckpt_has_compile_prefix and not model_has_compile_prefix:
            state_dict = {k.replace('_orig_mod.', '', 1): v for k, v in state_dict.items()}
            accelerator.print('Stripped _orig_mod. prefix from checkpoint state dict')
        elif model_has_compile_prefix and not ckpt_has_compile_prefix:
            state_dict = {f'_orig_mod.{k}': v for k, v in state_dict.items()}
            accelerator.print('Added _orig_mod. prefix to checkpoint state dict')
        if config.get('allow_partial_weight_load', False):
            current_state = model.state_dict()
            compatible_state = {
                k: v for k, v in state_dict.items()
                if k in current_state and tuple(v.shape) == tuple(current_state[k].shape)
            }
            skipped = sorted(set(state_dict.keys()) - set(compatible_state.keys()))
            missing, unexpected = model.load_state_dict(compatible_state, strict=False)
            accelerator.print(
                f"Loaded {len(compatible_state)}/{len(current_state)} compatible checkpoint tensors "
                f"(missing={len(missing)}, unexpected={len(unexpected)}, skipped={len(skipped)})"
            )
        else:
            model.load_state_dict(state_dict)

        if not config.get('load_weights_only', False):
            start_iteration = ckpt.get('step', 0)
            # Load optimizer state if optimizer type matches (SGD vs Adam check via betas)
            ckpt_optim_type = type(ckpt['optimizer']['param_groups'][0].get('betas', None))
            curr_optim_type = type(optimizer.param_groups[0].get('betas', None))
            if ckpt_optim_type == curr_optim_type:
                optimizer.load_state_dict(ckpt['optimizer'])
                accelerator.print('Loaded optimizer state (momentum preserved)')
            else:
                accelerator.print('Skipping optimizer state load (optimizer type changed)')

            if wandb_resume:
                if 'lr_scheduler' in ckpt:
                    lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
                    accelerator.print('Loaded lr scheduler state (resume enabled)')
                else:
                    accelerator.print('Resume enabled but checkpoint missing lr_scheduler state; using fresh scheduler')

    if val_pert_dataloader is not None:
        model, optimizer, train_dataloader, val_dataloader, val_pert_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, val_dataloader, val_pert_dataloader, lr_scheduler
        )
    else:
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, val_dataloader, lr_scheduler
        )

    def _state_dict_for_eager_eval():
        state_dict = accelerator.unwrap_model(model).state_dict()
        if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            state_dict = {k.replace('_orig_mod.', '', 1): v for k, v in state_dict.items()}
        return state_dict

    use_separate_eager_eval_for_logging = bool(config.get('separate_eager_eval_for_logging', False))
    eval_model = None
    if use_separate_eager_eval_for_logging and accelerator.is_main_process:
        eval_model = make_model(config).to(accelerator.device)
        eval_model.eval()
        eval_model.load_state_dict(_state_dict_for_eager_eval())

    if accelerator.is_main_process:
        accelerator.print("\n=== Displacement Field Training Configuration ===")
        accelerator.print(f"Input channels: {config['in_channels']}")
        accelerator.print(f"Growth direction channels: {config.get('use_growth_direction_channels', False)}")
        if not use_displacement_head:
            accelerator.print("Displacement head: disabled (no displacement-consuming loss is enabled)")
        elif use_dense_displacement:
            accelerator.print(f"Displacement supervision: dense ({disp_loss_type}, beta={disp_huber_beta})")
        elif disp_supervision == 'normal_scalar':
            accelerator.print(f"Displacement supervision: {disp_supervision} ({normal_loss_type}, beta={normal_loss_beta})")
        else:
            accelerator.print(f"Displacement supervision: {disp_supervision} ({disp_loss_type}, beta={disp_huber_beta})")
        output_heads = []
        if use_displacement_head:
            output_heads.append(f"displacement ({displacement_out_channels}ch)")
        if use_flow_refinement_targets:
            output_heads.append(f"flow_dir ({displacement_out_channels}ch)")
            output_heads.append(f"flow_dist ({displacement_out_channels // 3}ch)")
        if use_velocity_targets:
            output_heads.append("velocity_dir (3ch)")
        if use_trace_dist_head:
            output_heads.append("trace_dist (1ch)")
        if use_trace_stop_head:
            output_heads.append("trace_stop (1ch)")
        if use_surface_attract_head:
            output_heads.append("surface_attract (3ch)")
        if use_trace_validity_head:
            output_heads.append("trace_validity (1ch)")
        if use_sdt:
            output_heads.append("SDT (1ch)")
        if use_heatmap:
            output_heads.append("heatmap (1ch)")
        if use_segmentation:
            output_heads.append("segmentation (2ch)")
        accelerator.print("Output: " + " + ".join(output_heads))
        if use_sdt:
            accelerator.print(f"Lambda SDT: {lambda_sdt}")
        if use_heatmap:
            accelerator.print(f"Lambda heatmap: {lambda_heatmap}")
        if use_segmentation:
            accelerator.print(f"Lambda segmentation: {lambda_segmentation}")
        if lambda_cond_disp > 0.0:
            accelerator.print(f"Lambda cond disp: {lambda_cond_disp}")
        if lambda_triplet_min_disp > 0.0:
            accelerator.print(
                f"Lambda triplet min disp: {lambda_triplet_min_disp} (min={triplet_min_disp_vox} vx)"
            )
        if use_flow_refinement_targets:
            accelerator.print(
                f"Flow refinement losses: lambda_dir={lambda_flow_dir}, "
                f"lambda_dist={lambda_flow_dist}, dist_beta={flow_dist_huber_beta}"
            )
        if use_velocity_targets:
            accelerator.print(
                f"Velocity direction loss: lambda={lambda_velocity_dir}, "
                f"mode={config.get('velocity_target_mode')}, "
                f"region={config.get('velocity_target_region')}, "
                f"dilation={config.get('velocity_target_dilation_radius')}"
            )
            if lambda_velocity_smooth > 0.0:
                accelerator.print(
                    f"Velocity smoothness loss: lambda={lambda_velocity_smooth}, "
                    f"normalize={config.get('velocity_smooth_normalize')}"
                )
            if lambda_trace_integration > 0.0:
                accelerator.print(
                    f"Trace integration loss: lambda={lambda_trace_integration}, "
                    f"steps={trace_integration_steps}, step_size={trace_integration_step_size}, "
                    f"max_points={trace_integration_max_points}, "
                    f"detach_steps={trace_integration_detach_steps}"
                )
        if use_trace_ode_targets:
            accelerator.print(
                f"Trace ODE losses: lambda_dist={lambda_trace_dist}, "
                f"lambda_stop={lambda_trace_stop}, lambda_attract={lambda_surface_attract}, "
                f"lambda_validity={lambda_trace_validity}, "
                f"region={config.get('trace_target_region')}, "
                f"dilation={config.get('trace_target_dilation_radius')}, "
                f"defer_dilation_to_trainer={defer_trace_dilation_to_trainer}, "
                f"stop_radius={config.get('trace_stop_radius')}, "
                f"attract_mode={surface_attract_target_mode}, "
                f"attract_radius={config.get('trace_surface_attract_radius')}"
            )
            if use_trace_validity_head:
                accelerator.print(f"Trace validity EDT in trainer: {defer_trace_validity_to_trainer}")
        accelerator.print(f"Build dense EDT targets: {needs_dense_targets}")
        if triplet_mode:
            accelerator.print(
                f"Triplet direction priors: enabled={use_triplet_direction_priors}"
            )
            if use_triplet_direction_priors:
                accelerator.print(
                    f"Triplet prior mask={triplet_direction_prior_mask}, "
                    f"random_swap_prob={triplet_random_channel_swap_prob}"
                )
        accelerator.print(f"Supervise conditioning: {config.get('supervise_conditioning', False)}")
        if config.get('supervise_conditioning', False):
            accelerator.print(f"Cond supervision weight: {config.get('cond_supervision_weight', 0.1)}")
        optimizer_summary = f"Optimizer: {optimizer_type} (lr={optimizer_kwargs['learning_rate']}, weight_decay={optimizer_kwargs.get('weight_decay', 0)})"
        scheduler_details = ", ".join(f"{k}={v}" for k, v in scheduler_kwargs.items())
        scheduler_summary = f"Scheduler: {scheduler_type}" + (f" ({scheduler_details})" if scheduler_details else "")
        accelerator.print(optimizer_summary)
        accelerator.print(scheduler_summary)
        accelerator.print(f"Train samples: {num_train}, Val samples: {num_val}")
        accelerator.print(f"Eval perturbed val: {config.get('eval_perturbed_val', False)}")
        accelerator.print("=================================================\n")

    if config['verbose']:
        accelerator.print("creating iterators...")
    val_iterator = iter(val_dataloader)
    val_pert_iterator = iter(val_pert_dataloader) if val_pert_dataloader is not None else None
    train_iterator = iter(train_dataloader)
    grad_clip = config['grad_clip']
    profile_data_time = bool(config.get('profile_data_time', False))
    profile_step_time = bool(config.get('profile_step_time', False))
    profile_first_iteration = bool(config.get('profile_first_iteration', True))
    profile_log_every = max(1, int(config.get('profile_log_every', 100)))

    progress_bar = tqdm(
        total=config['num_iterations'],
        initial=start_iteration,
        disable=not accelerator.is_local_main_process
    )

    for iteration in range(start_iteration, config['num_iterations']):
        if config['verbose']:
            accelerator.print(f"starting iteration {iteration}")
        first_iter_profile = profile_first_iteration and iteration == start_iteration
        iter_profile_t0 = time.perf_counter()
        last_profile_t = iter_profile_t0

        def _mark_first_iter(stage):
            nonlocal last_profile_t
            if not first_iter_profile:
                return
            if accelerator.device.type == 'cuda':
                torch.cuda.synchronize(accelerator.device)
            now = time.perf_counter()
            print(
                f"[rank {accelerator.process_index}/{accelerator.num_processes} "
                f"device={accelerator.device}] "
                f"[iter {iteration} timing] {stage}: "
                f"+{now - last_profile_t:.3f}s, total={now - iter_profile_t0:.3f}s",
                flush=True,
            )
            last_profile_t = now

        should_log_this_iteration = (
            (iteration > 0 or config.get('log_at_step_zero', False))
            and iteration % config['log_frequency'] == 0
        )
        data_wait_start = time.perf_counter()
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_dataloader)
            batch = next(train_iterator)
        data_wait_time = time.perf_counter() - data_wait_start
        _mark_first_iter("data batch ready")
        step_start_time = time.perf_counter()

        if config['verbose']:
            accelerator.print(f"got batch, keys: {batch.keys()}")

        inputs, extrap_coords, gt_displacement, valid_mask, point_weights, point_normals, dense_gt_displacement, dense_loss_weight, flow_dir_target, flow_dist_target, velocity_dir_target, velocity_loss_weight, trace_dist_target, trace_stop_target, trace_loss_weight, trace_validity_target, trace_validity_weight, surface_attract_target, surface_attract_weight, sdt_target, heatmap_target, seg_target, seg_skel = prepare_batch(
            batch,
            use_sdt,
            use_heatmap,
            use_segmentation,
            use_growth_direction_channels=bool(config.get('use_growth_direction_channels', False)),
            dense_target_builder=build_dense_targets_on_device if needs_dense_targets else None,
            velocity_target_builder=build_velocity_targets_on_device,
        )
        _mark_first_iter("batch prepared")

        wandb_log = {}

        with accelerator.accumulate(model):
            # Forward pass
            output = model(inputs)
            _mark_first_iter("forward complete")
            disp_pred = output.get('displacement')  # [B, C, D, H, W]
            if use_displacement_head and disp_pred is None:
                raise ValueError("Displacement head is enabled but missing from model outputs")
            grad_norm = None

            if lambda_displacement > 0.0:
                surf_loss = compute_displacement_loss(
                    disp_pred,
                    extrap_coords,
                    gt_displacement,
                    valid_mask,
                    point_weights,
                    point_normals,
                    dense_gt_displacement,
                    dense_loss_weight,
                )
                weighted_surf_loss = lambda_displacement * surf_loss
            else:
                surf_loss = _zero_loss_from_output(output)
                weighted_surf_loss = surf_loss
            total_loss = weighted_surf_loss

            wandb_log['surf_loss'] = surf_loss.detach().item()
            if lambda_displacement != 1.0:
                wandb_log['weighted_surf_loss'] = weighted_surf_loss.detach().item()

            if use_flow_refinement_targets and (lambda_flow_dir > 0.0 or lambda_flow_dist > 0.0):
                flow_dir_loss, flow_dist_loss = compute_flow_refinement_loss(
                    output['flow_dir'],
                    output['flow_dist'],
                    flow_dir_target,
                    flow_dist_target,
                    dense_loss_weight,
                )
                weighted_flow_dir_loss = lambda_flow_dir * flow_dir_loss
                weighted_flow_dist_loss = lambda_flow_dist * flow_dist_loss
                total_loss = total_loss + weighted_flow_dir_loss + weighted_flow_dist_loss
                wandb_log['flow_dir_loss'] = weighted_flow_dir_loss.detach().item()
                wandb_log['flow_dist_loss'] = weighted_flow_dist_loss.detach().item()

            if use_velocity_targets and lambda_velocity_dir > 0.0:
                velocity_dir_loss = compute_velocity_dir_loss(
                    output['velocity_dir'],
                    velocity_dir_target,
                    velocity_loss_weight,
                )
                weighted_velocity_dir_loss = lambda_velocity_dir * velocity_dir_loss
                total_loss = total_loss + weighted_velocity_dir_loss
                wandb_log['velocity_dir_loss'] = weighted_velocity_dir_loss.detach().item()

            if use_velocity_targets and lambda_velocity_smooth > 0.0:
                velocity_smooth_loss = weighted_vector_smoothness_loss(
                    output['velocity_dir'],
                    sample_weights=velocity_loss_weight,
                    normalize_vectors=bool(config.get('velocity_smooth_normalize', True)),
                )
                weighted_velocity_smooth_loss = lambda_velocity_smooth * velocity_smooth_loss
                total_loss = total_loss + weighted_velocity_smooth_loss
                wandb_log['velocity_smooth_loss'] = weighted_velocity_smooth_loss.detach().item()

            if use_velocity_targets and lambda_trace_integration > 0.0:
                trace_integration_loss = velocity_streamline_integration_loss(
                    output['velocity_dir'],
                    velocity_dir_target,
                    velocity_loss_weight,
                    steps=trace_integration_steps,
                    step_size=trace_integration_step_size,
                    max_points=trace_integration_max_points,
                    min_weight=trace_integration_min_weight,
                    detach_steps=trace_integration_detach_steps,
                    random_sample=True,
                )
                weighted_trace_integration_loss = lambda_trace_integration * trace_integration_loss
                total_loss = total_loss + weighted_trace_integration_loss
                wandb_log['trace_integration_loss'] = weighted_trace_integration_loss.detach().item()

            if use_trace_ode_targets and (
                lambda_trace_dist > 0.0 or lambda_trace_stop > 0.0 or lambda_surface_attract > 0.0
            ):
                trace_dist_loss, trace_stop_loss, surface_attract_loss = compute_trace_ode_losses(
                    output.get('trace_dist'),
                    output.get('trace_stop'),
                    output.get('surface_attract'),
                    trace_dist_target,
                    trace_stop_target,
                    trace_loss_weight,
                    surface_attract_target,
                    surface_attract_weight,
                )
                weighted_trace_dist_loss = lambda_trace_dist * trace_dist_loss
                weighted_trace_stop_loss = lambda_trace_stop * trace_stop_loss
                weighted_surface_attract_loss = lambda_surface_attract * surface_attract_loss
                total_loss = (
                    total_loss
                    + weighted_trace_dist_loss
                    + weighted_trace_stop_loss
                    + weighted_surface_attract_loss
                )
                wandb_log['trace_dist_loss'] = weighted_trace_dist_loss.detach().item()
                wandb_log['trace_stop_loss'] = weighted_trace_stop_loss.detach().item()
                wandb_log['surface_attract_loss'] = weighted_surface_attract_loss.detach().item()
            if use_trace_ode_targets and lambda_trace_validity > 0.0:
                trace_validity_loss = compute_trace_validity_loss(
                    output.get('trace_validity'),
                    trace_validity_target,
                    trace_validity_weight,
                )
                weighted_trace_validity_loss = lambda_trace_validity * trace_validity_loss
                total_loss = total_loss + weighted_trace_validity_loss
                wandb_log['trace_validity_loss'] = weighted_trace_validity_loss.detach().item()
                if 'neighbor_sheet_present' in batch:
                    wandb_log['neighbor_context_fraction'] = batch['neighbor_sheet_present'].float().mean().detach().item()

            # Smoothness loss on displacement field
            if lambda_smooth > 0:
                if disp_pred is None:
                    raise ValueError("lambda_smooth > 0 requires the displacement output head")
                smooth_loss = smoothness_loss(disp_pred)
                weighted_smooth_loss = lambda_smooth * smooth_loss
                total_loss = total_loss + weighted_smooth_loss
                wandb_log['smooth_loss'] = weighted_smooth_loss.detach().item()

            # Optional SDT loss
            if use_sdt:
                sdt_pred = output['sdt']  # [B, 1, D, H, W]
                sdt_loss = sdt_loss_fn(sdt_pred, sdt_target)
                weighted_sdt_loss = lambda_sdt * sdt_loss
                total_loss = total_loss + weighted_sdt_loss
                wandb_log['sdt_loss'] = weighted_sdt_loss.detach().item()

            # Optional heatmap loss (BCE + Dice)
            heatmap_pred = None
            if use_heatmap:
                heatmap_pred = output['heatmap']  # [B, 1, D, H, W]
                heatmap_target_binary = (heatmap_target > 0.5).float()
                heatmap_loss = heatmap_loss_fn(heatmap_pred, heatmap_target_binary)
                weighted_heatmap_loss = lambda_heatmap * heatmap_loss
                total_loss = total_loss + weighted_heatmap_loss
                wandb_log['heatmap_loss'] = weighted_heatmap_loss.detach().item()

            # Optional segmentation loss (MedialSurfaceRecall)
            if use_segmentation:
                seg_pred = output['segmentation']  # [B, 2, D, H, W]

                # Optionally mask out conditioning region from seg loss
                seg_loss_mask = None
                if mask_cond_from_seg_loss:
                    cond_mask_seg = (inputs[:, 1:2] > 0.5).float()  # [B, 1, D, H, W]
                    seg_loss_mask = (cond_mask_seg < 0.5).float()   # 1 everywhere except cond

                seg_loss = seg_loss_fn(seg_pred, seg_target.long(), seg_skel.long(), loss_mask=seg_loss_mask)
                weighted_seg_loss = lambda_segmentation * seg_loss
                total_loss = total_loss + weighted_seg_loss
                wandb_log['seg_loss'] = weighted_seg_loss.detach().item()

            if lambda_cond_disp > 0.0:
                if disp_pred is None:
                    raise ValueError("lambda_cond_disp > 0 requires the displacement output head")
                cond_mask = (inputs[:, 1:2] > 0.5).float()
                disp_mag_sq = (disp_pred ** 2).sum(dim=1, keepdim=True)
                cond_loss = (disp_mag_sq * cond_mask).sum() / cond_mask.sum().clamp(min=1.0)
                weighted_cond_loss = lambda_cond_disp * cond_loss
                total_loss = total_loss + weighted_cond_loss
                wandb_log['cond_disp_loss'] = weighted_cond_loss.detach().item()

            if lambda_triplet_min_disp > 0.0:
                if disp_pred is None:
                    raise ValueError("lambda_triplet_min_disp > 0 requires the displacement output head")
                cond_mask = (inputs[:, 1:2] > 0.5).float()
                triplet_min_loss = triplet_min_displacement_loss(
                    disp_pred,
                    cond_mask,
                    min_magnitude=triplet_min_disp_vox,
                    loss_type='squared_hinge',
                )
                weighted_triplet_min_loss = lambda_triplet_min_disp * triplet_min_loss
                total_loss = total_loss + weighted_triplet_min_loss
                wandb_log['triplet_min_disp_loss'] = weighted_triplet_min_loss.detach().item()

            if torch.isnan(total_loss).any():
                raise ValueError('loss is NaN')

            do_optimizer_step = True
            accelerator.backward(total_loss)
            _mark_first_iter("backward complete")
            if accelerator.sync_gradients:
                grad_norm = accelerator.clip_grad_norm_(model.parameters(), grad_clip)
                grad_norm_value = float(grad_norm.detach().item() if torch.is_tensor(grad_norm) else grad_norm)
                if not np.isfinite(grad_norm_value):
                    do_optimizer_step = False
                    if accelerator.is_main_process:
                        accelerator.print(
                            f"Warning: non-finite grad norm at iteration {iteration}; skipping optimizer step"
                        )
                    wandb_log['skipped_step_nonfinite_grad'] = 1.0
            if do_optimizer_step:
                optimizer.step()
                lr_scheduler.step()
            optimizer.zero_grad()
            _mark_first_iter("optimizer step complete")
        step_compute_time = time.perf_counter() - step_start_time

        wandb_log['loss'] = total_loss.detach().item()
        wandb_log['current_lr'] = optimizer.param_groups[0]['lr']
        if grad_norm is not None:
            wandb_log['grad_norm'] = float(grad_norm.detach().item() if torch.is_tensor(grad_norm) else grad_norm)
        if (iteration % profile_log_every) == 0:
            if profile_data_time:
                wandb_log['data_wait_time'] = data_wait_time
            if profile_step_time:
                wandb_log['step_compute_time'] = step_compute_time

        postfix = {
            'loss': f"{wandb_log['loss']:.4f}",
            'surf': f"{wandb_log['surf_loss']:.4f}",
        }
        if lambda_smooth > 0:
            postfix['smooth'] = f"{wandb_log['smooth_loss']:.4f}"
        if use_sdt:
            postfix['sdt'] = f"{wandb_log['sdt_loss']:.4f}"
        if use_heatmap:
            postfix['hm'] = f"{wandb_log['heatmap_loss']:.4f}"
        if use_segmentation:
            postfix['seg'] = f"{wandb_log['seg_loss']:.4f}"
        if use_flow_refinement_targets and lambda_flow_dir > 0.0:
            postfix['flow_dir'] = f"{wandb_log['flow_dir_loss']:.4f}"
        if use_flow_refinement_targets and lambda_flow_dist > 0.0:
            postfix['flow_dist'] = f"{wandb_log['flow_dist_loss']:.4f}"
        if use_velocity_targets and lambda_velocity_dir > 0.0:
            postfix['vel_dir'] = f"{wandb_log['velocity_dir_loss']:.4f}"
        if use_velocity_targets and lambda_velocity_smooth > 0.0:
            postfix['vel_smooth'] = f"{wandb_log['velocity_smooth_loss']:.4f}"
        if use_velocity_targets and lambda_trace_integration > 0.0:
            postfix['trace_int'] = f"{wandb_log['trace_integration_loss']:.4f}"
        if use_trace_ode_targets and lambda_trace_dist > 0.0:
            postfix['trace_dist'] = f"{wandb_log['trace_dist_loss']:.4f}"
        if use_trace_ode_targets and lambda_trace_stop > 0.0:
            postfix['trace_stop'] = f"{wandb_log['trace_stop_loss']:.4f}"
        if use_trace_ode_targets and lambda_surface_attract > 0.0:
            postfix['attract'] = f"{wandb_log['surface_attract_loss']:.4f}"
        if use_trace_ode_targets and lambda_trace_validity > 0.0:
            postfix['valid'] = f"{wandb_log['trace_validity_loss']:.4f}"
        if lambda_cond_disp > 0.0:
            postfix['cond'] = f"{wandb_log['cond_disp_loss']:.4f}"
        if lambda_triplet_min_disp > 0.0:
            postfix['min1vx'] = f"{wandb_log['triplet_min_disp_loss']:.4f}"
        progress_bar.set_postfix(postfix)
        progress_bar.update(1)

        if should_log_this_iteration and accelerator.is_main_process:
            with torch.no_grad():
                eval_forward_model = model
                if eval_model is not None:
                    eval_model.load_state_dict(_state_dict_for_eager_eval())
                    eval_model.eval()
                    eval_forward_model = eval_model
                else:
                    model.eval()

                val_batches_per_log = max(1, int(config.get('val_batches_per_log', 4)))
                val_metric_sums = {
                    'val_surf_loss': 0.0,
                    'val_loss': 0.0,
                }
                if lambda_smooth > 0:
                    val_metric_sums['val_smooth_loss'] = 0.0
                if use_sdt:
                    val_metric_sums['val_sdt_loss'] = 0.0
                if use_heatmap:
                    val_metric_sums['val_heatmap_loss'] = 0.0
                if use_segmentation:
                    val_metric_sums['val_seg_loss'] = 0.0
                if lambda_cond_disp > 0.0:
                    val_metric_sums['val_cond_disp_loss'] = 0.0
                if lambda_triplet_min_disp > 0.0:
                    val_metric_sums['val_triplet_min_disp_loss'] = 0.0
                if use_flow_refinement_targets and lambda_flow_dir > 0.0:
                    val_metric_sums['val_flow_dir_loss'] = 0.0
                if use_flow_refinement_targets and lambda_flow_dist > 0.0:
                    val_metric_sums['val_flow_dist_loss'] = 0.0
                if use_velocity_targets and lambda_velocity_dir > 0.0:
                    val_metric_sums['val_velocity_dir_loss'] = 0.0
                if use_velocity_targets and lambda_velocity_smooth > 0.0:
                    val_metric_sums['val_velocity_smooth_loss'] = 0.0
                if use_velocity_targets and lambda_trace_integration > 0.0:
                    val_metric_sums['val_trace_integration_loss'] = 0.0
                if use_trace_ode_targets:
                    if lambda_trace_dist > 0.0:
                        val_metric_sums['val_trace_dist_loss'] = 0.0
                    if lambda_trace_stop > 0.0:
                        val_metric_sums['val_trace_stop_loss'] = 0.0
                    if lambda_surface_attract > 0.0:
                        val_metric_sums['val_surface_attract_loss'] = 0.0
                    if lambda_trace_validity > 0.0:
                        val_metric_sums['val_trace_validity_loss'] = 0.0

                first_val_vis = None
                for val_batch_idx in range(val_batches_per_log):
                    try:
                        val_batch = next(val_iterator)
                    except StopIteration:
                        val_iterator = iter(val_dataloader)
                        val_batch = next(val_iterator)

                    val_inputs, val_extrap_coords, val_gt_displacement, val_valid_mask, val_point_weights, val_point_normals, val_dense_gt_displacement, val_dense_loss_weight, val_flow_dir_target, val_flow_dist_target, val_velocity_dir_target, val_velocity_loss_weight, val_trace_dist_target, val_trace_stop_target, val_trace_loss_weight, val_trace_validity_target, val_trace_validity_weight, val_surface_attract_target, val_surface_attract_weight, val_sdt_target, val_heatmap_target, val_seg_target, val_seg_skel = prepare_batch(
                        val_batch,
                        use_sdt,
                        use_heatmap,
                        use_segmentation,
                        use_growth_direction_channels=bool(config.get('use_growth_direction_channels', False)),
                        dense_target_builder=build_dense_targets_on_device if needs_dense_targets else None,
                        velocity_target_builder=build_velocity_targets_on_device,
                    )

                    with accelerator.autocast():
                        val_output = eval_forward_model(val_inputs)
                    val_disp_pred = val_output.get('displacement')
                    if use_displacement_head and val_disp_pred is None:
                        raise ValueError("Displacement head is enabled but missing from validation model outputs")

                    if lambda_displacement > 0.0:
                        val_surf_loss = compute_displacement_loss(
                            val_disp_pred,
                            val_extrap_coords,
                            val_gt_displacement,
                            val_valid_mask,
                            val_point_weights,
                            val_point_normals,
                            val_dense_gt_displacement,
                            val_dense_loss_weight,
                        )
                        val_weighted_surf_loss = lambda_displacement * val_surf_loss
                    else:
                        val_surf_loss = _zero_loss_from_output(val_output)
                        val_weighted_surf_loss = val_surf_loss
                    val_total_loss = val_weighted_surf_loss
                    val_metric_sums['val_surf_loss'] += val_surf_loss.item()

                    val_sdt_pred = None
                    if use_flow_refinement_targets and (lambda_flow_dir > 0.0 or lambda_flow_dist > 0.0):
                        val_flow_dir_loss, val_flow_dist_loss = compute_flow_refinement_loss(
                            val_output['flow_dir'],
                            val_output['flow_dist'],
                            val_flow_dir_target,
                            val_flow_dist_target,
                            val_dense_loss_weight,
                        )
                        val_weighted_flow_dir_loss = lambda_flow_dir * val_flow_dir_loss
                        val_weighted_flow_dist_loss = lambda_flow_dist * val_flow_dist_loss
                        val_total_loss = val_total_loss + val_weighted_flow_dir_loss + val_weighted_flow_dist_loss
                        if lambda_flow_dir > 0.0:
                            val_metric_sums['val_flow_dir_loss'] += val_weighted_flow_dir_loss.item()
                        if lambda_flow_dist > 0.0:
                            val_metric_sums['val_flow_dist_loss'] += val_weighted_flow_dist_loss.item()

                    if use_velocity_targets and lambda_velocity_dir > 0.0:
                        val_velocity_dir_loss = compute_velocity_dir_loss(
                            val_output['velocity_dir'],
                            val_velocity_dir_target,
                            val_velocity_loss_weight,
                        )
                        val_weighted_velocity_dir_loss = lambda_velocity_dir * val_velocity_dir_loss
                        val_total_loss = val_total_loss + val_weighted_velocity_dir_loss
                        val_metric_sums['val_velocity_dir_loss'] += val_weighted_velocity_dir_loss.item()

                    if use_velocity_targets and lambda_velocity_smooth > 0.0:
                        val_velocity_smooth_loss = weighted_vector_smoothness_loss(
                            val_output['velocity_dir'],
                            sample_weights=val_velocity_loss_weight,
                            normalize_vectors=bool(config.get('velocity_smooth_normalize', True)),
                        )
                        val_weighted_velocity_smooth_loss = lambda_velocity_smooth * val_velocity_smooth_loss
                        val_total_loss = val_total_loss + val_weighted_velocity_smooth_loss
                        val_metric_sums['val_velocity_smooth_loss'] += val_weighted_velocity_smooth_loss.item()

                    if use_velocity_targets and lambda_trace_integration > 0.0:
                        val_trace_integration_loss = velocity_streamline_integration_loss(
                            val_output['velocity_dir'],
                            val_velocity_dir_target,
                            val_velocity_loss_weight,
                            steps=trace_integration_steps,
                            step_size=trace_integration_step_size,
                            max_points=trace_integration_max_points,
                            min_weight=trace_integration_min_weight,
                            detach_steps=trace_integration_detach_steps,
                            random_sample=False,
                        )
                        val_weighted_trace_integration_loss = lambda_trace_integration * val_trace_integration_loss
                        val_total_loss = val_total_loss + val_weighted_trace_integration_loss
                        val_metric_sums['val_trace_integration_loss'] += val_weighted_trace_integration_loss.item()

                    if use_trace_ode_targets and (
                        lambda_trace_dist > 0.0 or lambda_trace_stop > 0.0 or lambda_surface_attract > 0.0
                    ):
                        val_trace_dist_loss, val_trace_stop_loss, val_surface_attract_loss = compute_trace_ode_losses(
                            val_output.get('trace_dist'),
                            val_output.get('trace_stop'),
                            val_output.get('surface_attract'),
                            val_trace_dist_target,
                            val_trace_stop_target,
                            val_trace_loss_weight,
                            val_surface_attract_target,
                            val_surface_attract_weight,
                        )
                        val_weighted_trace_dist_loss = lambda_trace_dist * val_trace_dist_loss
                        val_weighted_trace_stop_loss = lambda_trace_stop * val_trace_stop_loss
                        val_weighted_surface_attract_loss = lambda_surface_attract * val_surface_attract_loss
                        val_total_loss = (
                            val_total_loss
                            + val_weighted_trace_dist_loss
                            + val_weighted_trace_stop_loss
                            + val_weighted_surface_attract_loss
                        )
                        if lambda_trace_dist > 0.0:
                            val_metric_sums['val_trace_dist_loss'] += val_weighted_trace_dist_loss.item()
                        if lambda_trace_stop > 0.0:
                            val_metric_sums['val_trace_stop_loss'] += val_weighted_trace_stop_loss.item()
                        if lambda_surface_attract > 0.0:
                            val_metric_sums['val_surface_attract_loss'] += val_weighted_surface_attract_loss.item()
                    if use_trace_ode_targets and lambda_trace_validity > 0.0:
                        val_trace_validity_loss = compute_trace_validity_loss(
                            val_output.get('trace_validity'),
                            val_trace_validity_target,
                            val_trace_validity_weight,
                        )
                        val_weighted_trace_validity_loss = lambda_trace_validity * val_trace_validity_loss
                        val_total_loss = val_total_loss + val_weighted_trace_validity_loss
                        val_metric_sums['val_trace_validity_loss'] += val_weighted_trace_validity_loss.item()

                    if lambda_smooth > 0:
                        if val_disp_pred is None:
                            raise ValueError("lambda_smooth > 0 requires the displacement output head")
                        val_smooth_loss = smoothness_loss(val_disp_pred)
                        val_weighted_smooth_loss = lambda_smooth * val_smooth_loss
                        val_total_loss = val_total_loss + val_weighted_smooth_loss
                        val_metric_sums['val_smooth_loss'] += val_weighted_smooth_loss.item()

                    if use_sdt:
                        val_sdt_pred = val_output['sdt']
                        val_sdt_loss = sdt_loss_fn(val_sdt_pred, val_sdt_target)
                        val_weighted_sdt_loss = lambda_sdt * val_sdt_loss
                        val_total_loss = val_total_loss + val_weighted_sdt_loss
                        val_metric_sums['val_sdt_loss'] += val_weighted_sdt_loss.item()

                    val_heatmap_pred = None
                    if use_heatmap:
                        val_heatmap_pred = val_output['heatmap']
                        val_heatmap_target_binary = (val_heatmap_target > 0.5).float()
                        val_heatmap_loss = heatmap_loss_fn(val_heatmap_pred, val_heatmap_target_binary)
                        val_weighted_heatmap_loss = lambda_heatmap * val_heatmap_loss
                        val_total_loss = val_total_loss + val_weighted_heatmap_loss
                        val_metric_sums['val_heatmap_loss'] += val_weighted_heatmap_loss.item()

                    if use_segmentation:
                        val_seg_pred = val_output['segmentation']
                        val_seg_loss_mask = None
                        if mask_cond_from_seg_loss:
                            val_cond_mask_seg = (val_inputs[:, 1:2] > 0.5).float()
                            val_seg_loss_mask = (val_cond_mask_seg < 0.5).float()
                        val_seg_loss = seg_loss_fn(
                            val_seg_pred, val_seg_target.long(), val_seg_skel.long(), loss_mask=val_seg_loss_mask
                        )
                        val_weighted_seg_loss = lambda_segmentation * val_seg_loss
                        val_total_loss = val_total_loss + val_weighted_seg_loss
                        val_metric_sums['val_seg_loss'] += val_weighted_seg_loss.item()

                    if lambda_cond_disp > 0.0:
                        if val_disp_pred is None:
                            raise ValueError("lambda_cond_disp > 0 requires the displacement output head")
                        val_cond_mask = (val_inputs[:, 1:2] > 0.5).float()
                        val_disp_mag_sq = (val_disp_pred ** 2).sum(dim=1, keepdim=True)
                        val_cond_loss = (val_disp_mag_sq * val_cond_mask).sum() / val_cond_mask.sum().clamp(min=1.0)
                        val_weighted_cond_loss = lambda_cond_disp * val_cond_loss
                        val_total_loss = val_total_loss + val_weighted_cond_loss
                        val_metric_sums['val_cond_disp_loss'] += val_weighted_cond_loss.item()

                    if lambda_triplet_min_disp > 0.0:
                        if val_disp_pred is None:
                            raise ValueError("lambda_triplet_min_disp > 0 requires the displacement output head")
                        val_cond_mask = (val_inputs[:, 1:2] > 0.5).float()
                        val_triplet_min_loss = triplet_min_displacement_loss(
                            val_disp_pred,
                            val_cond_mask,
                            min_magnitude=triplet_min_disp_vox,
                            loss_type='squared_hinge',
                        )
                        val_weighted_triplet_min_loss = lambda_triplet_min_disp * val_triplet_min_loss
                        val_total_loss = val_total_loss + val_weighted_triplet_min_loss
                        val_metric_sums['val_triplet_min_disp_loss'] += val_weighted_triplet_min_loss.item()

                    val_metric_sums['val_loss'] += val_total_loss.item()

                    if val_batch_idx == 0:
                        first_val_vis = {
                            'inputs': val_inputs,
                            'disp_pred': val_disp_pred,
                            'extrap_coords': val_extrap_coords,
                            'gt_displacement': val_gt_displacement,
                            'valid_mask': val_valid_mask,
                            'dense_gt_displacement': val_dense_gt_displacement,
                            'dense_loss_weight': val_dense_loss_weight,
                            'triplet_channel_order': val_batch.get('triplet_channel_order', None),
                            'sdt_pred': val_sdt_pred,
                            'sdt_target': val_sdt_target,
                            'heatmap_pred': val_heatmap_pred,
                            'heatmap_target': val_heatmap_target,
                            'seg_pred': val_output.get('segmentation') if use_segmentation else None,
                            'seg_target': val_seg_target if use_segmentation else None,
                            'velocity_dir_pred': val_output.get('velocity_dir') if use_velocity_targets else None,
                            'velocity_dir_target': val_velocity_dir_target,
                            'velocity_loss_weight': val_velocity_loss_weight,
                            'trace_dist_pred': val_output.get('trace_dist') if use_trace_ode_targets else None,
                            'trace_dist_target': val_trace_dist_target,
                            'trace_stop_pred': val_output.get('trace_stop') if use_trace_ode_targets else None,
                            'trace_stop_target': val_trace_stop_target,
                            'trace_loss_weight': val_trace_loss_weight,
                            'trace_validity_pred': val_output.get('trace_validity') if use_trace_ode_targets else None,
                            'trace_validity_target': val_trace_validity_target,
                            'trace_validity_weight': val_trace_validity_weight,
                            'surface_attract_pred': val_output.get('surface_attract') if use_trace_ode_targets else None,
                            'surface_attract_target': val_surface_attract_target,
                            'surface_attract_weight': val_surface_attract_weight,
                            'can_visualize_sparse': (
                                val_disp_pred is not None and
                                val_extrap_coords is not None and
                                val_gt_displacement is not None and
                                val_valid_mask is not None
                            ),
                            'can_visualize_dense': (
                                val_disp_pred is not None and val_dense_gt_displacement is not None
                            ),
                            'can_visualize_trace': (
                                use_trace_ode_targets and (
                                    val_velocity_dir_target is not None
                                    or val_trace_dist_target is not None
                                    or val_trace_stop_target is not None
                                    or val_trace_validity_target is not None
                                    or val_surface_attract_target is not None
                                )
                            ),
                        }

                for key, value in val_metric_sums.items():
                    wandb_log[key] = value / val_batches_per_log

                # Create visualization
                train_img_path = f'{out_dir}/{iteration:06}_train.png'
                val_img_path = f'{out_dir}/{iteration:06}_val.png'

                train_sdt_pred = output.get('sdt') if use_sdt else None
                train_heatmap_pred = heatmap_pred if use_heatmap else None
                train_seg_pred = output.get('segmentation') if use_segmentation else None
                train_velocity_dir_pred = output.get('velocity_dir') if use_velocity_targets else None
                train_trace_dist_pred = output.get('trace_dist') if use_trace_ode_targets else None
                train_trace_stop_pred = output.get('trace_stop') if use_trace_ode_targets else None
                train_trace_validity_pred = output.get('trace_validity') if use_trace_ode_targets else None
                train_surface_attract_pred = output.get('surface_attract') if use_trace_ode_targets else None
                train_can_visualize_sparse = (
                    disp_pred is not None
                    and extrap_coords is not None and gt_displacement is not None and valid_mask is not None
                )
                train_can_visualize_dense = disp_pred is not None and dense_gt_displacement is not None
                train_can_visualize_trace = use_trace_ode_targets and (
                    velocity_dir_target is not None
                    or trace_dist_target is not None
                    or trace_stop_target is not None
                    or trace_validity_target is not None
                    or surface_attract_target is not None
                )
                if train_can_visualize_sparse and first_val_vis is not None and first_val_vis.get('can_visualize_sparse', False):
                    make_visualization(
                        inputs, disp_pred, extrap_coords, gt_displacement, valid_mask,
                        sdt_pred=train_sdt_pred, sdt_target=sdt_target,
                        heatmap_pred=train_heatmap_pred, heatmap_target=heatmap_target,
                        seg_pred=train_seg_pred, seg_target=seg_target if use_segmentation else None,
                        save_path=train_img_path
                    )
                    make_visualization(
                        first_val_vis['inputs'], first_val_vis['disp_pred'],
                        first_val_vis['extrap_coords'], first_val_vis['gt_displacement'],
                        first_val_vis['valid_mask'],
                        sdt_pred=first_val_vis['sdt_pred'], sdt_target=first_val_vis['sdt_target'],
                        heatmap_pred=first_val_vis['heatmap_pred'], heatmap_target=first_val_vis['heatmap_target'],
                        seg_pred=first_val_vis['seg_pred'],
                        seg_target=first_val_vis['seg_target'],
                        save_path=val_img_path
                    )

                    if wandb.run is not None:
                        wandb_log['train_image'] = wandb.Image(train_img_path)
                        wandb_log['val_image'] = wandb.Image(val_img_path)
                elif (
                    first_val_vis is not None
                    and (
                        (train_can_visualize_dense and first_val_vis.get('can_visualize_dense', False))
                        or (train_can_visualize_trace and first_val_vis.get('can_visualize_trace', False))
                    )
                ):
                    make_dense_visualization(
                        inputs, disp_pred, dense_gt_displacement, dense_loss_weight,
                        triplet_channel_order=batch.get('triplet_channel_order', None),
                        sdt_pred=train_sdt_pred, sdt_target=sdt_target,
                        heatmap_pred=train_heatmap_pred, heatmap_target=heatmap_target,
                        seg_pred=train_seg_pred, seg_target=seg_target if use_segmentation else None,
                        velocity_dir_pred=train_velocity_dir_pred,
                        velocity_dir_target=velocity_dir_target,
                        velocity_loss_weight=velocity_loss_weight,
                        trace_dist_pred=train_trace_dist_pred,
                        trace_dist_target=trace_dist_target,
                        trace_stop_pred=train_trace_stop_pred,
                        trace_stop_target=trace_stop_target,
                        trace_loss_weight=trace_loss_weight,
                        trace_validity_pred=train_trace_validity_pred,
                        trace_validity_target=trace_validity_target,
                        trace_validity_weight=trace_validity_weight,
                        surface_attract_pred=train_surface_attract_pred,
                        surface_attract_target=surface_attract_target,
                        surface_attract_weight=surface_attract_weight,
                        save_path=train_img_path
                    )
                    make_dense_visualization(
                        first_val_vis['inputs'], first_val_vis['disp_pred'],
                        first_val_vis['dense_gt_displacement'], first_val_vis['dense_loss_weight'],
                        triplet_channel_order=first_val_vis['triplet_channel_order'],
                        sdt_pred=first_val_vis['sdt_pred'], sdt_target=first_val_vis['sdt_target'],
                        heatmap_pred=first_val_vis['heatmap_pred'], heatmap_target=first_val_vis['heatmap_target'],
                        seg_pred=first_val_vis['seg_pred'],
                        seg_target=first_val_vis['seg_target'],
                        velocity_dir_pred=first_val_vis['velocity_dir_pred'],
                        velocity_dir_target=first_val_vis['velocity_dir_target'],
                        velocity_loss_weight=first_val_vis['velocity_loss_weight'],
                        trace_dist_pred=first_val_vis['trace_dist_pred'],
                        trace_dist_target=first_val_vis['trace_dist_target'],
                        trace_stop_pred=first_val_vis['trace_stop_pred'],
                        trace_stop_target=first_val_vis['trace_stop_target'],
                        trace_loss_weight=first_val_vis['trace_loss_weight'],
                        trace_validity_pred=first_val_vis['trace_validity_pred'],
                        trace_validity_target=first_val_vis['trace_validity_target'],
                        trace_validity_weight=first_val_vis['trace_validity_weight'],
                        surface_attract_pred=first_val_vis['surface_attract_pred'],
                        surface_attract_target=first_val_vis['surface_attract_target'],
                        surface_attract_weight=first_val_vis['surface_attract_weight'],
                        save_path=val_img_path
                    )

                    if wandb.run is not None:
                        wandb_log['train_image'] = wandb.Image(train_img_path)
                        wandb_log['val_image'] = wandb.Image(val_img_path)

                if val_pert_dataloader is not None:
                    try:
                        val_pert_batch = next(val_pert_iterator)
                    except StopIteration:
                        val_pert_iterator = iter(val_pert_dataloader)
                        val_pert_batch = next(val_pert_iterator)

                    val_pert_inputs, val_pert_extrap_coords, val_pert_gt_displacement, val_pert_valid_mask, val_pert_point_weights, val_pert_point_normals, val_pert_dense_gt_displacement, val_pert_dense_loss_weight, val_pert_flow_dir_target, val_pert_flow_dist_target, val_pert_velocity_dir_target, val_pert_velocity_loss_weight, val_pert_trace_dist_target, val_pert_trace_stop_target, val_pert_trace_loss_weight, val_pert_trace_validity_target, val_pert_trace_validity_weight, val_pert_surface_attract_target, val_pert_surface_attract_weight, val_pert_sdt_target, val_pert_heatmap_target, val_pert_seg_target, val_pert_seg_skel = prepare_batch(
                        val_pert_batch,
                        use_sdt,
                        use_heatmap,
                        use_segmentation,
                        use_growth_direction_channels=bool(config.get('use_growth_direction_channels', False)),
                        dense_target_builder=build_dense_targets_on_device if needs_dense_targets else None,
                        velocity_target_builder=build_velocity_targets_on_device,
                    )

                    with accelerator.autocast():
                        val_pert_output = eval_forward_model(val_pert_inputs)
                    val_pert_disp_pred = val_pert_output.get('displacement')
                    if use_displacement_head and val_pert_disp_pred is None:
                        raise ValueError("Displacement head is enabled but missing from perturbed validation outputs")

                    if lambda_displacement > 0.0:
                        val_pert_surf_loss = compute_displacement_loss(
                            val_pert_disp_pred,
                            val_pert_extrap_coords,
                            val_pert_gt_displacement,
                            val_pert_valid_mask,
                            val_pert_point_weights,
                            val_pert_point_normals,
                            val_pert_dense_gt_displacement,
                            val_pert_dense_loss_weight,
                        )
                        val_pert_weighted_surf_loss = lambda_displacement * val_pert_surf_loss
                    else:
                        val_pert_surf_loss = _zero_loss_from_output(val_pert_output)
                        val_pert_weighted_surf_loss = val_pert_surf_loss
                    val_pert_total_loss = val_pert_weighted_surf_loss
                    wandb_log['val_pert_surf_loss'] = val_pert_surf_loss.item()

                    val_pert_sdt_pred = None
                    if use_flow_refinement_targets and (lambda_flow_dir > 0.0 or lambda_flow_dist > 0.0):
                        val_pert_flow_dir_loss, val_pert_flow_dist_loss = compute_flow_refinement_loss(
                            val_pert_output['flow_dir'],
                            val_pert_output['flow_dist'],
                            val_pert_flow_dir_target,
                            val_pert_flow_dist_target,
                            val_pert_dense_loss_weight,
                        )
                        val_pert_weighted_flow_dir_loss = lambda_flow_dir * val_pert_flow_dir_loss
                        val_pert_weighted_flow_dist_loss = lambda_flow_dist * val_pert_flow_dist_loss
                        val_pert_total_loss = (
                            val_pert_total_loss +
                            val_pert_weighted_flow_dir_loss +
                            val_pert_weighted_flow_dist_loss
                        )
                        if lambda_flow_dir > 0.0:
                            wandb_log['val_pert_flow_dir_loss'] = val_pert_weighted_flow_dir_loss.item()
                        if lambda_flow_dist > 0.0:
                            wandb_log['val_pert_flow_dist_loss'] = val_pert_weighted_flow_dist_loss.item()

                    if use_velocity_targets and lambda_velocity_dir > 0.0:
                        val_pert_velocity_dir_loss = compute_velocity_dir_loss(
                            val_pert_output['velocity_dir'],
                            val_pert_velocity_dir_target,
                            val_pert_velocity_loss_weight,
                        )
                        val_pert_weighted_velocity_dir_loss = lambda_velocity_dir * val_pert_velocity_dir_loss
                        val_pert_total_loss = val_pert_total_loss + val_pert_weighted_velocity_dir_loss
                        wandb_log['val_pert_velocity_dir_loss'] = val_pert_weighted_velocity_dir_loss.item()

                    if use_velocity_targets and lambda_velocity_smooth > 0.0:
                        val_pert_velocity_smooth_loss = weighted_vector_smoothness_loss(
                            val_pert_output['velocity_dir'],
                            sample_weights=val_pert_velocity_loss_weight,
                            normalize_vectors=bool(config.get('velocity_smooth_normalize', True)),
                        )
                        val_pert_weighted_velocity_smooth_loss = (
                            lambda_velocity_smooth * val_pert_velocity_smooth_loss
                        )
                        val_pert_total_loss = val_pert_total_loss + val_pert_weighted_velocity_smooth_loss
                        wandb_log['val_pert_velocity_smooth_loss'] = (
                            val_pert_weighted_velocity_smooth_loss.item()
                        )

                    if use_velocity_targets and lambda_trace_integration > 0.0:
                        val_pert_trace_integration_loss = velocity_streamline_integration_loss(
                            val_pert_output['velocity_dir'],
                            val_pert_velocity_dir_target,
                            val_pert_velocity_loss_weight,
                            steps=trace_integration_steps,
                            step_size=trace_integration_step_size,
                            max_points=trace_integration_max_points,
                            min_weight=trace_integration_min_weight,
                            detach_steps=trace_integration_detach_steps,
                            random_sample=False,
                        )
                        val_pert_weighted_trace_integration_loss = (
                            lambda_trace_integration * val_pert_trace_integration_loss
                        )
                        val_pert_total_loss = val_pert_total_loss + val_pert_weighted_trace_integration_loss
                        wandb_log['val_pert_trace_integration_loss'] = (
                            val_pert_weighted_trace_integration_loss.item()
                        )

                    if use_trace_ode_targets and (
                        lambda_trace_dist > 0.0 or lambda_trace_stop > 0.0 or lambda_surface_attract > 0.0
                    ):
                        (
                            val_pert_trace_dist_loss,
                            val_pert_trace_stop_loss,
                            val_pert_surface_attract_loss,
                        ) = compute_trace_ode_losses(
                            val_pert_output.get('trace_dist'),
                            val_pert_output.get('trace_stop'),
                            val_pert_output.get('surface_attract'),
                            val_pert_trace_dist_target,
                            val_pert_trace_stop_target,
                            val_pert_trace_loss_weight,
                            val_pert_surface_attract_target,
                            val_pert_surface_attract_weight,
                        )
                        val_pert_weighted_trace_dist_loss = lambda_trace_dist * val_pert_trace_dist_loss
                        val_pert_weighted_trace_stop_loss = lambda_trace_stop * val_pert_trace_stop_loss
                        val_pert_weighted_surface_attract_loss = (
                            lambda_surface_attract * val_pert_surface_attract_loss
                        )
                        val_pert_total_loss = (
                            val_pert_total_loss
                            + val_pert_weighted_trace_dist_loss
                            + val_pert_weighted_trace_stop_loss
                            + val_pert_weighted_surface_attract_loss
                        )
                        if lambda_trace_dist > 0.0:
                            wandb_log['val_pert_trace_dist_loss'] = val_pert_weighted_trace_dist_loss.item()
                        if lambda_trace_stop > 0.0:
                            wandb_log['val_pert_trace_stop_loss'] = val_pert_weighted_trace_stop_loss.item()
                        if lambda_surface_attract > 0.0:
                            wandb_log['val_pert_surface_attract_loss'] = (
                                val_pert_weighted_surface_attract_loss.item()
                            )
                    if use_trace_ode_targets and lambda_trace_validity > 0.0:
                        val_pert_trace_validity_loss = compute_trace_validity_loss(
                            val_pert_output.get('trace_validity'),
                            val_pert_trace_validity_target,
                            val_pert_trace_validity_weight,
                        )
                        val_pert_weighted_trace_validity_loss = lambda_trace_validity * val_pert_trace_validity_loss
                        val_pert_total_loss = val_pert_total_loss + val_pert_weighted_trace_validity_loss
                        wandb_log['val_pert_trace_validity_loss'] = val_pert_weighted_trace_validity_loss.item()

                    if lambda_smooth > 0:
                        if val_pert_disp_pred is None:
                            raise ValueError("lambda_smooth > 0 requires the displacement output head")
                        val_pert_smooth_loss = smoothness_loss(val_pert_disp_pred)
                        val_pert_weighted_smooth_loss = lambda_smooth * val_pert_smooth_loss
                        val_pert_total_loss = val_pert_total_loss + val_pert_weighted_smooth_loss
                        wandb_log['val_pert_smooth_loss'] = val_pert_weighted_smooth_loss.item()

                    if use_sdt:
                        val_pert_sdt_pred = val_pert_output['sdt']
                        val_pert_sdt_loss = sdt_loss_fn(val_pert_sdt_pred, val_pert_sdt_target)
                        val_pert_weighted_sdt_loss = lambda_sdt * val_pert_sdt_loss
                        val_pert_total_loss = val_pert_total_loss + val_pert_weighted_sdt_loss
                        wandb_log['val_pert_sdt_loss'] = val_pert_weighted_sdt_loss.item()

                    val_pert_heatmap_pred = None
                    if use_heatmap:
                        val_pert_heatmap_pred = val_pert_output['heatmap']
                        val_pert_heatmap_target_binary = (val_pert_heatmap_target > 0.5).float()
                        val_pert_heatmap_loss = heatmap_loss_fn(val_pert_heatmap_pred, val_pert_heatmap_target_binary)
                        val_pert_weighted_heatmap_loss = lambda_heatmap * val_pert_heatmap_loss
                        val_pert_total_loss = val_pert_total_loss + val_pert_weighted_heatmap_loss
                        wandb_log['val_pert_heatmap_loss'] = val_pert_weighted_heatmap_loss.item()

                    if use_segmentation:
                        val_pert_seg_pred = val_pert_output['segmentation']
                        val_pert_seg_loss_mask = None
                        if mask_cond_from_seg_loss:
                            val_pert_cond_mask_seg = (val_pert_inputs[:, 1:2] > 0.5).float()
                            val_pert_seg_loss_mask = (val_pert_cond_mask_seg < 0.5).float()
                        val_pert_seg_loss = seg_loss_fn(
                            val_pert_seg_pred, val_pert_seg_target.long(), val_pert_seg_skel.long(), loss_mask=val_pert_seg_loss_mask
                        )
                        val_pert_weighted_seg_loss = lambda_segmentation * val_pert_seg_loss
                        val_pert_total_loss = val_pert_total_loss + val_pert_weighted_seg_loss
                        wandb_log['val_pert_seg_loss'] = val_pert_weighted_seg_loss.item()

                    if lambda_cond_disp > 0.0:
                        if val_pert_disp_pred is None:
                            raise ValueError("lambda_cond_disp > 0 requires the displacement output head")
                        val_pert_cond_mask = (val_pert_inputs[:, 1:2] > 0.5).float()
                        val_pert_disp_mag_sq = (val_pert_disp_pred ** 2).sum(dim=1, keepdim=True)
                        val_pert_cond_loss = (val_pert_disp_mag_sq * val_pert_cond_mask).sum() / val_pert_cond_mask.sum().clamp(min=1.0)
                        val_pert_weighted_cond_loss = lambda_cond_disp * val_pert_cond_loss
                        val_pert_total_loss = val_pert_total_loss + val_pert_weighted_cond_loss
                        wandb_log['val_pert_cond_disp_loss'] = val_pert_weighted_cond_loss.item()

                    if lambda_triplet_min_disp > 0.0:
                        if val_pert_disp_pred is None:
                            raise ValueError("lambda_triplet_min_disp > 0 requires the displacement output head")
                        val_pert_cond_mask = (val_pert_inputs[:, 1:2] > 0.5).float()
                        val_pert_triplet_min_loss = triplet_min_displacement_loss(
                            val_pert_disp_pred,
                            val_pert_cond_mask,
                            min_magnitude=triplet_min_disp_vox,
                            loss_type='squared_hinge',
                        )
                        val_pert_weighted_triplet_min_loss = lambda_triplet_min_disp * val_pert_triplet_min_loss
                        val_pert_total_loss = val_pert_total_loss + val_pert_weighted_triplet_min_loss
                        wandb_log['val_pert_triplet_min_disp_loss'] = val_pert_weighted_triplet_min_loss.item()

                    wandb_log['val_pert_loss'] = val_pert_total_loss.item()

                    if config.get('log_perturbed_val_images', False):
                        if (
                            val_pert_disp_pred is not None and
                            val_pert_extrap_coords is not None and
                            val_pert_gt_displacement is not None and
                            val_pert_valid_mask is not None
                        ):
                            val_pert_img_path = f'{out_dir}/{iteration:06}_val_pert.png'
                            make_visualization(
                                val_pert_inputs, val_pert_disp_pred, val_pert_extrap_coords, val_pert_gt_displacement, val_pert_valid_mask,
                                sdt_pred=val_pert_sdt_pred, sdt_target=val_pert_sdt_target,
                                heatmap_pred=val_pert_heatmap_pred, heatmap_target=val_pert_heatmap_target,
                                seg_pred=val_pert_output.get('segmentation') if use_segmentation else None,
                                seg_target=val_pert_seg_target if use_segmentation else None,
                                velocity_dir_pred=val_pert_output.get('velocity_dir') if use_velocity_targets else None,
                                velocity_dir_target=val_pert_velocity_dir_target,
                                trace_dist_pred=val_pert_output.get('trace_dist') if use_trace_ode_targets else None,
                                trace_dist_target=val_pert_trace_dist_target,
                                trace_stop_pred=val_pert_output.get('trace_stop') if use_trace_ode_targets else None,
                                trace_stop_target=val_pert_trace_stop_target,
                                trace_validity_pred=val_pert_output.get('trace_validity') if use_trace_ode_targets else None,
                                trace_validity_target=val_pert_trace_validity_target,
                                trace_validity_weight=val_pert_trace_validity_weight,
                                surface_attract_pred=val_pert_output.get('surface_attract') if use_trace_ode_targets else None,
                                surface_attract_target=val_pert_surface_attract_target,
                                save_path=val_pert_img_path
                            )
                            if wandb.run is not None:
                                wandb_log['val_pert_image'] = wandb.Image(val_pert_img_path)
                        elif val_pert_dense_gt_displacement is not None or (
                            use_trace_ode_targets and (
                                val_pert_velocity_dir_target is not None
                                or val_pert_trace_dist_target is not None
                                or val_pert_trace_stop_target is not None
                                or val_pert_trace_validity_target is not None
                                or val_pert_surface_attract_target is not None
                            )
                        ):
                            val_pert_img_path = f'{out_dir}/{iteration:06}_val_pert.png'
                            make_dense_visualization(
                                val_pert_inputs, val_pert_disp_pred, val_pert_dense_gt_displacement, val_pert_dense_loss_weight,
                                triplet_channel_order=val_pert_batch.get('triplet_channel_order', None),
                                sdt_pred=val_pert_sdt_pred, sdt_target=val_pert_sdt_target,
                                heatmap_pred=val_pert_heatmap_pred, heatmap_target=val_pert_heatmap_target,
                                seg_pred=val_pert_output.get('segmentation') if use_segmentation else None,
                                seg_target=val_pert_seg_target if use_segmentation else None,
                                velocity_dir_pred=val_pert_output.get('velocity_dir') if use_velocity_targets else None,
                                velocity_dir_target=val_pert_velocity_dir_target,
                                velocity_loss_weight=val_pert_velocity_loss_weight,
                                trace_dist_pred=val_pert_output.get('trace_dist') if use_trace_ode_targets else None,
                                trace_dist_target=val_pert_trace_dist_target,
                                trace_stop_pred=val_pert_output.get('trace_stop') if use_trace_ode_targets else None,
                                trace_stop_target=val_pert_trace_stop_target,
                                trace_loss_weight=val_pert_trace_loss_weight,
                                trace_validity_pred=val_pert_output.get('trace_validity') if use_trace_ode_targets else None,
                                trace_validity_target=val_pert_trace_validity_target,
                                trace_validity_weight=val_pert_trace_validity_weight,
                                surface_attract_pred=val_pert_output.get('surface_attract') if use_trace_ode_targets else None,
                                surface_attract_target=val_pert_surface_attract_target,
                                surface_attract_weight=val_pert_surface_attract_weight,
                                save_path=val_pert_img_path
                            )
                            if wandb.run is not None:
                                wandb_log['val_pert_image'] = wandb.Image(val_pert_img_path)

                if eval_model is None:
                    model.train()

        if (
            (iteration > 0 or config.get('ckpt_at_step_zero', False))
            and iteration % config['ckpt_frequency'] == 0
            and accelerator.is_main_process
        ):
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
                'step': iteration,
                'wandb_run_id': wandb.run.id if wandb.run is not None else config.get('wandb_run_id'),
            }, f'{out_dir}/ckpt_{iteration:06}.pth')

        if wandb.run is not None and accelerator.is_main_process:
            wandb.log(wandb_log)

    progress_bar.close()

    if accelerator.is_main_process:
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'config': config,
            'step': config['num_iterations'],
            'wandb_run_id': wandb.run.id if wandb.run is not None else config.get('wandb_run_id'),
        }, f'{out_dir}/ckpt_final.pth')


if __name__ == '__main__':
    train()
