#!/usr/bin/env python
"""
Trainer for row/col conditioned trace-ODE target prediction.

Trains velocity, trace validity, and trace-band surface attraction heads from
conditioned surfaces.
"""
import os
import json
import sys
import click
import torch
import wandb
import random
import accelerate
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import time

from vesuvius.neural_tracing.datasets.dataset_rowcol_cond import EdtSegDataset
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
    velocity_streamline_integration_loss,
    weighted_vector_smoothness_loss,
)
from vesuvius.models.training.optimizers import create_optimizer
from vesuvius.models.training.lr_schedulers import get_scheduler
from vesuvius.neural_tracing.nets.models import make_model
from vesuvius.neural_tracing.trainers.rowcol_cond_visualization import (
    make_dense_visualization,
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


def _indices_to_displacement(nearest_idx_cp, shape):
    import cupy as cp

    d, h, w = shape
    disp_cp = nearest_idx_cp.astype(cp.float32, copy=False)
    disp_cp[0] -= cp.arange(d, dtype=cp.float32)[:, None, None]
    disp_cp[1] -= cp.arange(h, dtype=cp.float32)[None, :, None]
    disp_cp[2] -= cp.arange(w, dtype=cp.float32)[None, None, :]
    return cp.ascontiguousarray(disp_cp)


def _distance_transform_distances_torch(surface_mask: torch.Tensor):
    surface_bool = (surface_mask > 0.5).contiguous()
    if surface_bool.ndim != 3:
        raise ValueError(f"surface mask must be 3D, got shape {tuple(surface_bool.shape)}")
    if not bool(surface_bool.any().item()):
        return None
    with _cupy_device(surface_bool):
        import cupy as cp

        distances_cp = _cupyx_edt(surface_bool, return_distances=True, return_indices=False)
        distances_cp = cp.ascontiguousarray(distances_cp.astype(cp.float32, copy=False))
        return _cupy_to_torch(distances_cp).to(dtype=torch.float32)


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

    import cupy as cp

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
            band_cp = cp.isfinite(nearest_dist_cp) & (nearest_dist_cp <= float(radius_voxels))
            if bool(cp.any(band_cp).item()):
                src_z = nearest_idx_cp[0][band_cp]
                src_y = nearest_idx_cp[1][band_cp]
                src_x = nearest_idx_cp[2][band_cp]
                velocity_cp[:, band_cp] = velocity_cp[:, src_z, src_y, src_x]

            dilated_dir = velocity_b
            dilated_weight = _cupy_to_torch(
                cp.ascontiguousarray(band_cp[None].astype(cp.float32, copy=False))
            )

            attract_b = None
            attract_weight_b = None
            attract_radius = float(surface_attract_radius)
            if attract_radius > 0.0:
                attract_band_cp = cp.isfinite(nearest_dist_cp) & (nearest_dist_cp <= attract_radius)
                attract_cp = _indices_to_displacement(nearest_idx_cp, source_weight.shape[2:])
                attract_cp[:, ~attract_band_cp] = 0.0
                attract_b = _cupy_to_torch(attract_cp)
                attract_weight_b = _cupy_to_torch(
                    cp.ascontiguousarray(attract_band_cp[None].astype(cp.float32, copy=False))
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
    """Collate batch tensors for the active trace-ODE training path."""
    # Stack fixed-size tensors normally
    vol = torch.stack([b['vol'] for b in batch])
    cond = torch.stack([b['cond'] for b in batch])
    result = {
        'vol': vol,
        'cond': cond,
    }
    if 'cond_direction' in batch[0]:
        result['cond_direction'] = [b['cond_direction'] for b in batch]

    if 'velocity_dir' in batch[0]:
        result['velocity_dir'] = torch.stack([b['velocity_dir'] for b in batch])
    if 'velocity_loss_weight' in batch[0]:
        result['velocity_loss_weight'] = torch.stack([b['velocity_loss_weight'] for b in batch])
    if 'trace_loss_weight' in batch[0]:
        result['trace_loss_weight'] = torch.stack([b['trace_loss_weight'] for b in batch])
    if 'trace_validity' in batch[0]:
        result['trace_validity'] = torch.stack([b['trace_validity'] for b in batch])
    if 'trace_validity_weight' in batch[0]:
        result['trace_validity_weight'] = torch.stack([b['trace_validity_weight'] for b in batch])
    # Source masks used by trainer-side EDT target generation. Keeping EDT out
    # of dataloader workers avoids cupyx defaulting every worker to GPU 0.
    for key in ('cond_gt', 'masked_seg', 'neighbor_seg'):
        if key in batch[0]:
            result[key] = torch.stack([b[key] for b in batch])

    return result


def prepare_batch(
    batch,
    use_growth_direction_channels=False,
    velocity_target_builder=None,
):
    """Prepare batch tensors for training."""
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
    inputs = torch.cat(input_list, dim=1)

    velocity_dir_target = batch.get('velocity_dir', None)  # [B, 3, D, H, W]
    velocity_loss_weight = batch.get('velocity_loss_weight', None)  # [B, 1, D, H, W]
    trace_loss_weight = batch.get('trace_loss_weight', None)  # [B, 1, D, H, W]
    trace_validity_target = batch.get('trace_validity', None)  # [B, 1, D, H, W]
    trace_validity_weight = batch.get('trace_validity_weight', None)  # [B, 1, D, H, W]
    surface_attract_target = batch.get('surface_attract', None)  # [B, 3, D, H, W]
    surface_attract_weight = batch.get('surface_attract_weight', None)  # [B, 1, D, H, W]

    return (
        inputs,
        velocity_dir_target,
        velocity_loss_weight,
        trace_loss_weight,
        trace_validity_target,
        trace_validity_weight,
        surface_attract_target,
        surface_attract_weight,
    )


@click.command()
@click.argument('config_path', type=click.Path(exists=True))
def train(config_path):
    """Train a row/col conditioned trace-ODE model."""

    with open(config_path, 'r') as f:
        config = json.load(f)

    setdefault_rowcol_cond_dataset_config(config)

    validate_rowcol_cond_dataset_config(config)

    default_in_channels = (
        2
        + (
            growth_direction_channel_count()
            if bool(config.get('use_growth_direction_channels', False))
            else 0
        )
    )
    config.setdefault('in_channels', default_in_channels)
    if int(config['in_channels']) != default_in_channels:
        raise ValueError(
            f"in_channels={config['in_channels']} does not match configured inputs "
            f"(expected {default_in_channels} from "
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
    config.setdefault('lambda_velocity_smooth', 0.0)
    config.setdefault('velocity_smooth_normalize', True)
    config.setdefault('lambda_trace_integration', 0.0)
    config.setdefault('trace_integration_steps', 2)
    config.setdefault('trace_integration_step_size', 1.0)
    config.setdefault('trace_integration_max_points', 2048)
    config.setdefault('trace_integration_min_weight', 0.5)
    config.setdefault('trace_integration_detach_steps', False)
    config.setdefault('surface_attract_huber_beta', 5.0)
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

    use_surface_attract_head = float(config.get('lambda_surface_attract', 0.0)) > 0.0
    use_trace_validity_head = (
        bool(config.get('use_trace_validity_targets', False))
        or float(config.get('lambda_trace_validity', 0.0)) > 0.0
    )
    targets = {'velocity_dir': {'out_channels': 3, 'activation': 'none'}}
    if use_surface_attract_head:
        targets['surface_attract'] = {'out_channels': 3, 'activation': 'none'}
    if use_trace_validity_head:
        targets['trace_validity'] = {'out_channels': 1, 'activation': 'none'}
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

    unsupported_displacement_losses = {
        'lambda_displacement': float(config.get('lambda_displacement', 0.0)),
        'lambda_smooth': float(config.get('lambda_smooth', 0.0)),
        'lambda_cond_disp': float(config.get('lambda_cond_disp', 0.0)),
    }
    enabled_displacement_losses = [
        name for name, value in unsupported_displacement_losses.items() if value > 0.0
    ]
    if enabled_displacement_losses:
        raise ValueError(
            "train_rowcol_cond is trace-ODE-only; displacement-head losses are no longer supported: "
            f"{enabled_displacement_losses}"
        )
    lambda_velocity_dir = float(config.get('lambda_velocity_dir', 0.0))
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
    trace_surface_attract_radius = float(config.get('trace_surface_attract_radius', 0.0))
    trace_target_dilation_radius = float(config.get('trace_target_dilation_radius', 0.0))
    trace_validity_positive_radius = float(config.get('trace_validity_positive_radius', 2.0))
    trace_validity_negative_radius = float(config.get('trace_validity_negative_radius', 3.0))
    trace_validity_margin = float(config.get('trace_validity_margin', 3.0))
    trace_validity_background_weight = float(config.get('trace_validity_background_weight', 0.25))
    surface_attract_huber_beta = float(config.get('surface_attract_huber_beta', 5.0))
    if trace_integration_steps < 0:
        raise ValueError(f"trace_integration_steps must be >= 0, got {trace_integration_steps}")
    if trace_integration_step_size < 0.0:
        raise ValueError(f"trace_integration_step_size must be >= 0, got {trace_integration_step_size}")
    if trace_integration_max_points < 0:
        raise ValueError(f"trace_integration_max_points must be >= 0, got {trace_integration_max_points}")

    def build_velocity_targets_on_device(batch):
        """Build EDT-derived velocity/trace targets on the current batch device."""
        if (
            use_trace_validity_head
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

        if 'velocity_dir' not in batch or 'velocity_loss_weight' not in batch:
            raise ValueError("Velocity targets are enabled but missing from the batch")

        trace_band_attract_radius = (
            trace_surface_attract_radius
            if use_surface_attract_head and 'surface_attract' not in batch
            else 0.0
        )
        if (
            trace_target_dilation_radius <= 0.0
            and trace_band_attract_radius <= 0.0
        ):
            return
        (
            velocity_dir,
            velocity_weight,
            trace_loss_weight,
            surface_attract,
            surface_attract_weight,
        ) = _dilate_trace_targets_torch(
            batch['velocity_dir'],
            batch['velocity_loss_weight'],
            trace_target_dilation_radius,
            trace_loss_weight=batch.get('trace_loss_weight', None),
            surface_attract_radius=trace_band_attract_radius,
        )
        batch['velocity_dir'] = velocity_dir
        batch['velocity_loss_weight'] = velocity_weight
        if trace_loss_weight is not None:
            batch['trace_loss_weight'] = trace_loss_weight
        if surface_attract is not None:
            batch['surface_attract'] = surface_attract
            batch['surface_attract_weight'] = surface_attract_weight
        return

    def compute_velocity_dir_loss(velocity_dir_pred, velocity_dir_target, velocity_loss_weight):
        if velocity_dir_target is None or velocity_loss_weight is None:
            raise ValueError("Velocity targets are enabled but missing from the batch")
        pred = F.normalize(velocity_dir_pred.float(), dim=1, eps=1e-6)
        target = F.normalize(velocity_dir_target.float(), dim=1, eps=1e-6)
        dir_diff = 1.0 - (pred * target).sum(dim=1, keepdim=True).clamp(min=-1.0, max=1.0)
        weight = velocity_loss_weight.float()
        return (dir_diff * weight).sum() / weight.sum().clamp(min=1.0)

    def compute_surface_attract_loss(surface_attract_pred, surface_attract_target, surface_attract_weight):
        if surface_attract_pred is None or surface_attract_target is None:
            raise ValueError("Surface attraction loss is enabled but surface attraction tensors are missing")
        return dense_displacement_loss(
            surface_attract_pred,
            surface_attract_target,
            sample_weights=surface_attract_weight,
            loss_type='vector_huber',
            beta=surface_attract_huber_beta,
        )

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
        accelerator.print("\n=== Trace ODE Training Configuration ===")
        accelerator.print(f"Input channels: {config['in_channels']}")
        accelerator.print(f"Growth direction channels: {config.get('use_growth_direction_channels', False)}")
        output_heads = []
        output_heads.append("velocity_dir (3ch)")
        if use_surface_attract_head:
            output_heads.append("surface_attract (3ch)")
        if use_trace_validity_head:
            output_heads.append("trace_validity (1ch)")
        accelerator.print("Output: " + " + ".join(output_heads))
        accelerator.print(
            f"Velocity direction loss: lambda={lambda_velocity_dir}, "
            f"dilation={config.get('trace_target_dilation_radius')}"
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
        accelerator.print(
            f"Trace ODE losses: lambda_attract={lambda_surface_attract}, "
            f"lambda_validity={lambda_trace_validity}, "
            f"dilation={config.get('trace_target_dilation_radius')}, "
            f"attract_mode=trace_band, "
            f"attract_radius={config.get('trace_surface_attract_radius')}"
        )
        if use_trace_validity_head:
            accelerator.print("Trace validity EDT in trainer: True")
        optimizer_summary = f"Optimizer: {optimizer_type} (lr={optimizer_kwargs['learning_rate']}, weight_decay={optimizer_kwargs.get('weight_decay', 0)})"
        scheduler_details = ", ".join(f"{k}={v}" for k, v in scheduler_kwargs.items())
        scheduler_summary = f"Scheduler: {scheduler_type}" + (f" ({scheduler_details})" if scheduler_details else "")
        accelerator.print(optimizer_summary)
        accelerator.print(scheduler_summary)
        accelerator.print(f"Train samples: {num_train}, Val samples: {num_val}")
        accelerator.print("=================================================\n")

    if config['verbose']:
        accelerator.print("creating iterators...")
    val_iterator = iter(val_dataloader)
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

        inputs, velocity_dir_target, velocity_loss_weight, trace_loss_weight, trace_validity_target, trace_validity_weight, surface_attract_target, surface_attract_weight = prepare_batch(
            batch,
            use_growth_direction_channels=bool(config.get('use_growth_direction_channels', False)),
            velocity_target_builder=build_velocity_targets_on_device,
        )
        _mark_first_iter("batch prepared")

        wandb_log = {}

        with accelerator.accumulate(model):
            # Forward pass
            output = model(inputs)
            _mark_first_iter("forward complete")
            grad_norm = None
            total_loss = _zero_loss_from_output(output)

            if lambda_velocity_dir > 0.0:
                velocity_dir_loss = compute_velocity_dir_loss(
                    output['velocity_dir'],
                    velocity_dir_target,
                    velocity_loss_weight,
                )
                weighted_velocity_dir_loss = lambda_velocity_dir * velocity_dir_loss
                total_loss = total_loss + weighted_velocity_dir_loss
                wandb_log['velocity_dir_loss'] = weighted_velocity_dir_loss.detach().item()

            if lambda_velocity_smooth > 0.0:
                velocity_smooth_loss = weighted_vector_smoothness_loss(
                    output['velocity_dir'],
                    sample_weights=velocity_loss_weight,
                    normalize_vectors=bool(config.get('velocity_smooth_normalize', True)),
                )
                weighted_velocity_smooth_loss = lambda_velocity_smooth * velocity_smooth_loss
                total_loss = total_loss + weighted_velocity_smooth_loss
                wandb_log['velocity_smooth_loss'] = weighted_velocity_smooth_loss.detach().item()

            if lambda_trace_integration > 0.0:
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

            if lambda_surface_attract > 0.0:
                surface_attract_loss = compute_surface_attract_loss(
                    output.get('surface_attract'),
                    surface_attract_target,
                    surface_attract_weight,
                )
                weighted_surface_attract_loss = lambda_surface_attract * surface_attract_loss
                total_loss = total_loss + weighted_surface_attract_loss
                wandb_log['surface_attract_loss'] = weighted_surface_attract_loss.detach().item()
            if lambda_trace_validity > 0.0:
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
        }
        if lambda_velocity_dir > 0.0:
            postfix['vel_dir'] = f"{wandb_log['velocity_dir_loss']:.4f}"
        if lambda_velocity_smooth > 0.0:
            postfix['vel_smooth'] = f"{wandb_log['velocity_smooth_loss']:.4f}"
        if lambda_trace_integration > 0.0:
            postfix['trace_int'] = f"{wandb_log['trace_integration_loss']:.4f}"
        if lambda_surface_attract > 0.0:
            postfix['attract'] = f"{wandb_log['surface_attract_loss']:.4f}"
        if lambda_trace_validity > 0.0:
            postfix['valid'] = f"{wandb_log['trace_validity_loss']:.4f}"
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
                    'val_loss': 0.0,
                }
                if lambda_velocity_dir > 0.0:
                    val_metric_sums['val_velocity_dir_loss'] = 0.0
                if lambda_velocity_smooth > 0.0:
                    val_metric_sums['val_velocity_smooth_loss'] = 0.0
                if lambda_trace_integration > 0.0:
                    val_metric_sums['val_trace_integration_loss'] = 0.0
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

                    val_inputs, val_velocity_dir_target, val_velocity_loss_weight, val_trace_loss_weight, val_trace_validity_target, val_trace_validity_weight, val_surface_attract_target, val_surface_attract_weight = prepare_batch(
                        val_batch,
                        use_growth_direction_channels=bool(config.get('use_growth_direction_channels', False)),
                        velocity_target_builder=build_velocity_targets_on_device,
                    )

                    with accelerator.autocast():
                        val_output = eval_forward_model(val_inputs)
                    val_total_loss = _zero_loss_from_output(val_output)

                    if lambda_velocity_dir > 0.0:
                        val_velocity_dir_loss = compute_velocity_dir_loss(
                            val_output['velocity_dir'],
                            val_velocity_dir_target,
                            val_velocity_loss_weight,
                        )
                        val_weighted_velocity_dir_loss = lambda_velocity_dir * val_velocity_dir_loss
                        val_total_loss = val_total_loss + val_weighted_velocity_dir_loss
                        val_metric_sums['val_velocity_dir_loss'] += val_weighted_velocity_dir_loss.item()

                    if lambda_velocity_smooth > 0.0:
                        val_velocity_smooth_loss = weighted_vector_smoothness_loss(
                            val_output['velocity_dir'],
                            sample_weights=val_velocity_loss_weight,
                            normalize_vectors=bool(config.get('velocity_smooth_normalize', True)),
                        )
                        val_weighted_velocity_smooth_loss = lambda_velocity_smooth * val_velocity_smooth_loss
                        val_total_loss = val_total_loss + val_weighted_velocity_smooth_loss
                        val_metric_sums['val_velocity_smooth_loss'] += val_weighted_velocity_smooth_loss.item()

                    if lambda_trace_integration > 0.0:
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

                    if lambda_surface_attract > 0.0:
                        val_surface_attract_loss = compute_surface_attract_loss(
                            val_output.get('surface_attract'),
                            val_surface_attract_target,
                            val_surface_attract_weight,
                        )
                        val_weighted_surface_attract_loss = lambda_surface_attract * val_surface_attract_loss
                        val_total_loss = val_total_loss + val_weighted_surface_attract_loss
                        val_metric_sums['val_surface_attract_loss'] += val_weighted_surface_attract_loss.item()
                    if lambda_trace_validity > 0.0:
                        val_trace_validity_loss = compute_trace_validity_loss(
                            val_output.get('trace_validity'),
                            val_trace_validity_target,
                            val_trace_validity_weight,
                        )
                        val_weighted_trace_validity_loss = lambda_trace_validity * val_trace_validity_loss
                        val_total_loss = val_total_loss + val_weighted_trace_validity_loss
                        val_metric_sums['val_trace_validity_loss'] += val_weighted_trace_validity_loss.item()

                    val_metric_sums['val_loss'] += val_total_loss.item()

                    if val_batch_idx == 0:
                        first_val_vis = {
                            'inputs': val_inputs,
                            'velocity_dir_pred': val_output.get('velocity_dir'),
                            'velocity_dir_target': val_velocity_dir_target,
                            'velocity_loss_weight': val_velocity_loss_weight,
                            'trace_loss_weight': val_trace_loss_weight,
                            'trace_validity_pred': val_output.get('trace_validity'),
                            'trace_validity_target': val_trace_validity_target,
                            'trace_validity_weight': val_trace_validity_weight,
                            'surface_attract_pred': val_output.get('surface_attract'),
                            'surface_attract_target': val_surface_attract_target,
                            'surface_attract_weight': val_surface_attract_weight,
                            'can_visualize_trace': (
                                val_velocity_dir_target is not None
                                or val_trace_validity_target is not None
                                or val_surface_attract_target is not None
                            ),
                        }

                for key, value in val_metric_sums.items():
                    wandb_log[key] = value / val_batches_per_log

                # Create visualization
                train_img_path = f'{out_dir}/{iteration:06}_train.png'
                val_img_path = f'{out_dir}/{iteration:06}_val.png'

                train_velocity_dir_pred = output.get('velocity_dir')
                train_trace_validity_pred = output.get('trace_validity')
                train_surface_attract_pred = output.get('surface_attract')
                train_can_visualize_trace = (
                    velocity_dir_target is not None
                    or trace_validity_target is not None
                    or surface_attract_target is not None
                )
                if (
                    first_val_vis is not None
                    and train_can_visualize_trace
                    and first_val_vis.get('can_visualize_trace', False)
                ):
                    make_dense_visualization(
                        inputs, None, None, None,
                        velocity_dir_pred=train_velocity_dir_pred,
                        velocity_dir_target=velocity_dir_target,
                        velocity_loss_weight=velocity_loss_weight,
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
                        first_val_vis['inputs'], None, None, None,
                        velocity_dir_pred=first_val_vis['velocity_dir_pred'],
                        velocity_dir_target=first_val_vis['velocity_dir_target'],
                        velocity_loss_weight=first_val_vis['velocity_loss_weight'],
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
