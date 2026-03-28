import json
import math
import os
from collections import defaultdict
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
import wandb
import numpy as np 
import random
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import accelerate
from accelerate.utils import GradientAccumulationPlugin, set_seed
import click
from vesuvius.models.utils import InitWeights_He
from vesuvius.models.training.optimizers import (
    OptimizerParamGroupTarget,
    create_optimizer,
)
from vesuvius.models.training.lr_schedulers import get_scheduler
from koine_machines.models.make_model import make_model
from koine_machines.data.ink_dataset import InkDataset
from koine_machines.models.load_checkpoint import load_training_checkpoint_from_config, restore_training_state
from koine_machines.evaluation.metrics.balanced_accuracy import BalancedAccuracy
from koine_machines.evaluation.metrics.confusion import Confusion, ConfusionCounts
from koine_machines.training.normal_pooling import (
    RayFeatureHead,
    collate_normal_pooled_batch,
    pool_logits_along_normals,
)
from koine_machines.training.profiling import (
    NormalPoolingProfiler,
    _BackwardProfileMarkers,
)
from koine_machines.training.deep_supervision import (
    DeepSupervisionWrapper,
    _compute_ds_weights,
    build_deep_supervision_targets,
)
from koine_machines.training.visualization import PreviewAccumulator, build_validation_preview_log
from koine_machines.training.loss.losses import create_loss_from_config
from koine_machines.training.stitching import resolve_model_and_loader_patch_sizes, run_model_forward


@dataclass
class ValidationMetricBatch:
    logits: torch.Tensor
    targets: torch.Tensor
    valid_mask: torch.Tensor | None = None

    def require_targets(self):
        return self.targets


_NATIVE_3D_TRAINING_MODES = {"normal_pooled_3d", "full_3d"}


def _is_native_3d_training_mode(mode):
    return str(mode).strip().lower() in _NATIVE_3D_TRAINING_MODES


def _disable_z_projection_for_3d_training_modes(config):
    if not _is_native_3d_training_mode(config.get('mode', 'flat')):
        return

    model_config = config.setdefault('model_config', {})
    model_config['z_projection_mode'] = 'none'

    targets = config.get('targets') or {}
    for target_info in targets.values():
        if not isinstance(target_info, dict):
            continue
        target_info['z_projection_mode'] = 'none'
        if isinstance(target_info.get('z_projection'), dict):
            target_info['z_projection']['mode'] = 'none'


def _disable_z_projection_for_normal_pooled_3d(config):
    _disable_z_projection_for_3d_training_modes(config)


def _select_full_3d_preview_z_index(supervision_mask: torch.Tensor) -> int:
    if supervision_mask.ndim == 4:
        supervision_mask = supervision_mask.unsqueeze(1)
    elif supervision_mask.ndim != 5:
        raise ValueError(
            f"Expected supervision_mask with shape [B, 1, Z, Y, X] or [B, Z, Y, X], got {tuple(supervision_mask.shape)}"
        )

    if int(supervision_mask.shape[2]) <= 0:
        raise ValueError("supervision_mask must have positive depth for preview selection")

    per_slice_supervision = supervision_mask.float().sum(dim=(0, 1, 3, 4))
    best_index = int(torch.argmax(per_slice_supervision).item())
    if float(per_slice_supervision[best_index].item()) <= 0.0:
        return int(supervision_mask.shape[2] // 2)
    return best_index


def _build_full_3d_preview_batch(
    batch: dict,
    preds: torch.Tensor,
    targets: torch.Tensor,
    ignore_mask: torch.Tensor,
) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor]:
    if preds.ndim != 5 or targets.ndim != 5 or ignore_mask.ndim != 5:
        raise ValueError(
            "full_3d previews expect preds, targets, and ignore_mask with shape [B, 1, Z, Y, X]"
        )

    z_index = _select_full_3d_preview_z_index(batch['supervision_mask'])

    preview_batch = dict(batch)
    image = batch['image']
    if image.ndim == 4:
        image = image.unsqueeze(1)
    if image.ndim != 5:
        raise ValueError(f"Expected image with shape [B, 1, Z, Y, X] or [B, Z, Y, X], got {tuple(image.shape)}")
    preview_batch['image'] = image[:, :, z_index:z_index + 1]

    surface_mask = batch.get('surface_mask')
    if isinstance(surface_mask, torch.Tensor):
        if surface_mask.ndim == 4:
            surface_mask = surface_mask.unsqueeze(1)
        if surface_mask.ndim != 5:
            raise ValueError(
                f"Expected surface_mask with shape [B, 1, Z, Y, X] or [B, Z, Y, X], got {tuple(surface_mask.shape)}"
            )
        preview_batch['surface_mask'] = surface_mask[:, :, z_index:z_index + 1]

    return (
        preview_batch,
        preds[:, :, z_index],
        targets[:, :, z_index],
        ignore_mask[:, :, z_index],
    )


def _resolve_training_step_window(
    start_step: int,
    configured_num_iterations: int,
    profile_steps: int | None,
) -> tuple[int, int, int, int]:
    if profile_steps is None:
        return start_step, configured_num_iterations, configured_num_iterations, start_step
    return start_step, start_step + profile_steps, profile_steps, 0


def _record_dataset_profile_batch(profiler, batch):
    if profiler is None or not profiler.enabled:
        return

    profile_timings = batch.get('profile_timings')
    if not isinstance(profile_timings, list) or not profile_timings:
        return

    grouped = defaultdict(list)
    for sample_timings in profile_timings:
        if not isinstance(sample_timings, dict):
            continue
        for name, duration_seconds in sample_timings.items():
            grouped[str(name)].append(float(duration_seconds))

    for name, durations in grouped.items():
        if durations:
            profiler.add_duration(name, sum(durations) / len(durations))


def _compute_normal_pooled_3d_background_loss(
    volume_logits: torch.Tensor,
    surface_mask: torch.Tensor,
    *,
    outside_threshold: float = 0.0,
    loss_weight: float = 1.0,
) -> torch.Tensor:
    if volume_logits.ndim != 5:
        raise ValueError(
            f"Expected volume_logits with shape [B, 1, Z, Y, X], got {tuple(volume_logits.shape)}"
        )
    if surface_mask.ndim == 4:
        surface_mask = surface_mask.unsqueeze(1)
    elif surface_mask.ndim != 5:
        raise ValueError(
            f"Expected surface_mask with shape [B, 1, Z, Y, X] or [B, Z, Y, X], got {tuple(surface_mask.shape)}"
        )
    if tuple(int(v) for v in surface_mask.shape) != tuple(int(v) for v in volume_logits.shape):
        raise ValueError(
            f"surface_mask must match volume_logits shape, got {tuple(surface_mask.shape)} vs {tuple(volume_logits.shape)}"
        )

    loss_weight = float(loss_weight)
    if loss_weight <= 0.0:
        return volume_logits.new_zeros(())

    outside_mask = surface_mask <= float(outside_threshold)
    if not torch.any(outside_mask):
        return volume_logits.new_zeros(())

    outside_bg_loss = F.softplus(volume_logits.float())
    outside_bg_loss = outside_bg_loss.masked_select(outside_mask).mean()
    return outside_bg_loss * loss_weight


def _forward_model_with_optional_shared_features(
    model,
    image: torch.Tensor,
    model_crop_size,
    *,
    capture_shared_features: bool,
    stitched: bool = True,
    use_gradient_checkpointing: bool = True,
):
    if capture_shared_features:
        model_output = model(image)
        return model_output['ink'], model_output.get('shared_features')

    return (
        run_model_forward(
            model,
            image,
            model_crop_size,
            stitched=stitched,
            use_gradient_checkpointing=use_gradient_checkpointing,
        ),
        None,
    )


def _primary_supervision_output(preds):
    if isinstance(preds, (list, tuple)):
        if len(preds) == 0:
            raise ValueError("Deep supervision output list must not be empty")
        return preds[0]
    return preds


def _float_supervision_tree(value):
    if isinstance(value, (list, tuple)):
        return type(value)(_float_supervision_tree(v) for v in value)
    return value.float()


def _build_targets_with_ignore_for_loss(loss_preds, targets, ignore_mask):
    ds_targets = build_deep_supervision_targets(targets, loss_preds, mode='nearest')
    ds_ignore = build_deep_supervision_targets(ignore_mask, loss_preds, mode='nearest')
    if isinstance(ds_targets, (list, tuple)):
        return type(ds_targets)(
            torch.cat([target_level, ignore_level], dim=1)
            for target_level, ignore_level in zip(ds_targets, ds_ignore)
        )
    return torch.cat([ds_targets, ds_ignore], dim=1)

@click.command()
@click.argument('config_path', type=click.Path(exists=True))
@click.option(
    '--profile',
    'profile_steps',
    type=click.IntRange(min=1),
    default=None,
    help='Run only N training steps and print granular normal-pooling timings.',
)
def train(config_path, profile_steps):
    with open(config_path, 'r') as f:
        config = json.load(f)

    checkpoint_path, checkpoint, weights_only = load_training_checkpoint_from_config(config, config_path)
    resume_full_state = checkpoint is not None and not weights_only
    start_step = 0

    if str(config.get('model_type', '')).strip().lower() == 'dinov2':
        model_config = config.setdefault('model_config', {})
        for key in ('pretrained_backbone', 'pretrained_decoder_type'):
            if key in config:
                model_config.setdefault(key, config[key])
        if not model_config.get('pretrained_backbone'):
            raise ValueError(
                "model_type='dinov2' requires model_config.pretrained_backbone "
                "or a top-level pretrained_backbone entry"
            )
        config['model_type'] = 'vesuvius_unet'

    ema_config = config.get('ema') or {}
    ema_enabled = bool(ema_config.get('enabled', False))
    ema_decay = float(ema_config.get('decay', 0.999))
    ema_start_step = int(ema_config.get('start_step', 0))
    ema_update_every_steps = int(ema_config.get('update_every_steps', 1))
    ema_validate = bool(ema_config.get('validate', ema_enabled))
    ema_save_in_checkpoint = bool(ema_config.get('save_in_checkpoint', ema_enabled))
    
    config['ema'] = {
        'enabled': ema_enabled,
        'decay': ema_decay,
        'start_step': ema_start_step,
        'update_every_steps': ema_update_every_steps,
        'validate': ema_validate,
        'save_in_checkpoint': ema_save_in_checkpoint,
    }

    mode = str(config.get('mode', 'flat')).strip().lower()
    native_3d_mode = _is_native_3d_training_mode(mode)
    normal_pooled_mode = mode == 'normal_pooled_3d'
    pooling_config = config.get('normal_pooling') or {}
    feature_head_config = pooling_config.get('feature_head') or {}
    use_feature_head = normal_pooled_mode and bool(feature_head_config.get('enabled', False))
    deep_supervision_enabled = bool(config.get('enable_deep_supervision', False))
    model_type = str(config.get('model_type', '')).strip().lower()
    if normal_pooled_mode and model_type.startswith('resnet3d'):
        raise ValueError("normal_pooled_3d is currently only supported with the vesuvius_unet model path")
    if native_3d_mode:
        _disable_z_projection_for_3d_training_modes(config)
        config['in_channels'] = 2

    config.setdefault('volume_auth_json', None)
    requested_stitch_factor = int(config.get('stitch_factor', 1))
    use_stitched_forward = bool(config.get('use_stitched_forward', requested_stitch_factor > 1))
    if native_3d_mode:
        use_stitched_forward = False
    stitched_gradient_checkpointing = bool(config.get('stitched_gradient_checkpointing', True))
    model_crop_size = tuple(config['patch_size'])
    loader_patch_size = model_crop_size
    stitch_factor = 1
    if use_stitched_forward:
        model_crop_size, loader_patch_size, stitch_factor = resolve_model_and_loader_patch_sizes(config)
    config['crop_size'] = list(model_crop_size)
    config['patch_size'] = list(model_crop_size)
    config['stitch_factor'] = stitch_factor
    config['use_stitched_forward'] = use_stitched_forward
    config['stitched_gradient_checkpointing'] = stitched_gradient_checkpointing
    config['targets']['ink']['out_channels'] = 1
    config['targets']['ink']['activation'] = 'none'
    learning_rate = config.get('learning_rate', 0.01)
    grad_acc_steps = int(config.get('grad_acc_steps', 1))
    grad_clip = config.get('grad_clip')
    max_steps = config.get('max_steps', math.ceil(config['num_iterations'] / grad_acc_steps))
    val_every = int(config.get('val_every', 500))
    save_every = int(config.get('save_every', val_every))
    log_every = config.get('log_every', 1)
    val_preview_batches = config.get('val_preview_batches', 3)

    dataloader_config = accelerate.DataLoaderConfiguration(non_blocking = True)

    # The training loop reuses the dataloader indefinitely, so keep accumulation boundaries independent of dataloader exhaustion.
    gradient_accumulation_plugin = GradientAccumulationPlugin(num_steps=grad_acc_steps, sync_with_dataloader=False)
    accelerator = accelerate.Accelerator(mixed_precision=config.get('mixed_precision', "fp16"), gradient_accumulation_plugin=gradient_accumulation_plugin, dataloader_config=dataloader_config)
    normal_pooling_profiler = NormalPoolingProfiler(
        enabled=profile_steps is not None and normal_pooled_mode
    )

    if 'wandb_project' in config and accelerator.is_main_process:
        wandb_kwargs = { 'project' : config['wandb_project'], 'entity'  : config['wandb_entity'], 'config'  : config}
        if config.get('wandb_resume', False):
            wandb_run_id = config.get('wandb_run_id')
            if not wandb_run_id and checkpoint is not None:
                wandb_run_id = checkpoint.get('wandb_run_id')
            if not wandb_run_id:
                raise ValueError(
                    "wandb_resume=true requires wandb_run_id in config or checkpoint"
                )
            wandb_kwargs['id'] = wandb_run_id
            wandb_kwargs['resume'] = 'must'
            
        wandb.init(**wandb_kwargs)

    out_dir = config['out_dir']
    os.makedirs(out_dir, exist_ok=True)
    train_preview_dir = os.path.join(out_dir, 'train_previews')
    os.makedirs(train_preview_dir, exist_ok=True)
    val_preview_dir = os.path.join(out_dir, 'val_previews')
    os.makedirs(val_preview_dir, exist_ok=True)
    set_seed(config['seed'])
    dataset_config = deepcopy(config)
    dataset_config['patch_size'] = list(loader_patch_size)
    dataset_config['profile_dataset'] = profile_steps is not None and normal_pooled_mode

    shared_ds = InkDataset(dataset_config, do_augmentations=False)
    if len(shared_ds.training_patches) == 0:
        raise ValueError("FlatInkDataset produced no training patches after applying supervision masking")
    train_ds = InkDataset(dataset_config, do_augmentations=True, patches=shared_ds.training_patches)
    val_ds = InkDataset(dataset_config, do_augmentations=False, patches=shared_ds.validation_patches)
    train_subset = train_ds
    val_subset = val_ds

    dataloader_workers = int(config.get('dataloader_workers', 0))
    dataloader_kwargs = {
        'pin_memory': bool(config.get('pin_memory', accelerator.device.type == 'cuda')),
    }
    if dataloader_workers > 0:
        dataloader_kwargs['multiprocessing_context'] = 'spawn'
        dataloader_kwargs['persistent_workers'] = True
    if normal_pooled_mode:
        dataloader_kwargs['collate_fn'] = collate_normal_pooled_batch

    train_dl = DataLoader(
        train_subset,
        batch_size=config['batch_size'],
        shuffle=True,
        generator=torch.Generator().manual_seed(config['seed']),
        num_workers=dataloader_workers,
        **dataloader_kwargs,
    )
    # Validation only consumes a capped number of batches (`val_steps`), so
    # shuffle to sample a different deterministic subset on each pass.
    val_dl = DataLoader(
        val_subset,
        batch_size=config['batch_size'],
        shuffle=len(val_subset) > 0,
        generator=torch.Generator().manual_seed(config['seed'] + 1),
        num_workers=dataloader_workers,
        **dataloader_kwargs,
    )

    model = make_model(config)
    if deep_supervision_enabled and use_feature_head:
        raise ValueError(
            "enable_deep_supervision is not supported with normal_pooling.feature_head because "
            "deep supervision forces separate decoders and shared decoder features are unavailable"
        )
    if use_feature_head:
        if not hasattr(model, 'shared_encoder'):
            raise ValueError("normal_pooling.feature_head requires a model with shared_encoder output channels")
        feature_in_channels = int(model.shared_encoder.output_channels[0])
        model.normal_pooling_ray_head = RayFeatureHead(
            in_channels=feature_in_channels,
            bottleneck_channels=int(feature_head_config.get('bottleneck_channels', 16)),
            hidden_channels=int(feature_head_config.get('hidden_channels', 32)),
            pool_mode=str(feature_head_config.get('pool_mode', 'max')),
            pool_temperature=float(feature_head_config.get('pool_temperature', 1.0)),
        )
        model.return_shared_features = True
    optimizer_target = model
    pretrained_backbone = (config.get('model_config') or {}).get('pretrained_backbone')
    freeze_encoder = False
    if pretrained_backbone:
        freeze_encoder = bool(config.get('freeze_encoder', False))
        encoder_lr_mult = float(config.get('encoder_lr_mult', 1.0))

        encoder_params = list(model.shared_encoder.parameters())
        if freeze_encoder:
            for param in encoder_params:
                param.requires_grad = False

        if freeze_encoder or encoder_lr_mult != 1.0:
            encoder_param_ids = {id(param) for param in encoder_params}
            other_params = [
                param for param in model.parameters()
                if param.requires_grad and id(param) not in encoder_param_ids
            ]
            optimizer_target = []
            if other_params:
                optimizer_target.append({'params': other_params})
            if not freeze_encoder and encoder_params:
                optimizer_target.append({
                    'params': encoder_params,
                    'lr': learning_rate * encoder_lr_mult,
                })
            if not optimizer_target:
                raise ValueError("No trainable parameters remain after applying freeze_encoder")

    optimizer = create_optimizer({
                'name': config.get('optimizer', 'sgd'),
                'learning_rate': learning_rate,
                'weight_decay': config.get('weight_decay', 3e-5),
                }, OptimizerParamGroupTarget(optimizer_target) if isinstance(optimizer_target, list) else optimizer_target)

    lr_scheduler = get_scheduler(
        'diffusers_cosine_warmup',
        optimizer,
        initial_lr=learning_rate,
        max_steps=max_steps,
        warmup_steps=config.get('warmup_steps', 1000),
    )

    if not (config.get('model_config') or {}).get('pretrained_backbone'):
        model.apply(InitWeights_He(neg_slope=0.2))

    loss = create_loss_from_config(config)
    if deep_supervision_enabled:
        task_decoders = getattr(model, 'task_decoders', None)
        if not task_decoders:
            raise ValueError(
                "enable_deep_supervision requires per-task decoders, but the current model did not build any"
            )
        first_decoder = next(iter(task_decoders.values()))
        ds_output_count = len(getattr(first_decoder, 'stages', ()))
        ds_weights = _compute_ds_weights(ds_output_count)
        if ds_weights is None:
            raise ValueError(
                "enable_deep_supervision requires at least one decoder supervision stage"
            )
        loss = DeepSupervisionWrapper(loss, ds_weights)

    model, optimizer, train_dl, val_dl, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dl, val_dl, lr_scheduler
        )
    unwrapped_model = accelerator.unwrap_model(model)
    frozen_encoder = unwrapped_model.shared_encoder if pretrained_backbone and freeze_encoder else None
    ema_model = deepcopy(unwrapped_model) if ema_enabled else None
    if ema_model is not None:
        ema_model.eval()
        for parameter in ema_model.parameters():
            parameter.requires_grad_(False)

    optimizer_step = 0
    if checkpoint is not None:
        start_step, optimizer_step = restore_training_state(
            model,
            optimizer,
            lr_scheduler,
            checkpoint,
            checkpoint_path,
            load_weights_only=weights_only,
            ema_model=ema_model,
        )

        accelerator.print(f"Loaded checkpoint '{checkpoint_path}'"+ (f" and resuming from step {start_step}" if resume_full_state else " (weights only)"))
        
    train_iterator = iter(train_dl)
    latest_val_loss = None
    latest_ema_val_loss = None
    validation_confusion_metric = Confusion()
    validation_balanced_accuracy_metric = BalancedAccuracy()
    
    loop_start, loop_stop, progress_total, progress_initial = _resolve_training_step_window(
        start_step,
        config['num_iterations'],
        profile_steps,
    )
    progress_bar = tqdm(
        range(loop_start, loop_stop),
        total=progress_total,
        initial=progress_initial,
        disable=not accelerator.is_main_process,
        dynamic_ncols=True,
    )

    def get_model_input(batch):
        image = batch['image'].float()
        if image.ndim == 4:
            image = image.unsqueeze(1)
        if image.ndim != 5:
            raise ValueError(
                f"Expected 'image' batch tensor with 4 or 5 dims, got shape {tuple(image.shape)}"
            )
        if native_3d_mode:
            surface_mask = batch['surface_mask'].float()
            if surface_mask.ndim == 4:
                surface_mask = surface_mask.unsqueeze(1)
            if surface_mask.shape != image.shape:
                raise ValueError(
                    f"Expected 'surface_mask' to match 'image' shape, got {tuple(surface_mask.shape)} "
                    f"vs {tuple(image.shape)}"
                )
            return torch.cat([image, surface_mask], dim=1)
        return image

    def prepare_loss_inputs(preds, batch, profiler=None, feature_volume=None):
        if isinstance(preds, (list, tuple)):
            loss_preds = []
            targets = None
            ignore_mask = None
            outside_bg_loss = None
            for idx, pred_level in enumerate(preds):
                current_loss_preds, current_targets, current_ignore_mask, current_outside_bg_loss = prepare_loss_inputs(
                    pred_level,
                    batch,
                    profiler=profiler if idx == 0 else None,
                    feature_volume=feature_volume if idx == 0 else None,
                )
                loss_preds.append(current_loss_preds)
                if idx == 0:
                    targets = current_targets
                    ignore_mask = current_ignore_mask
                    outside_bg_loss = current_outside_bg_loss
            return type(preds)(loss_preds), targets, ignore_mask, outside_bg_loss

        if normal_pooled_mode:
            total_section = profiler.section('normal_pooling/total', preds.device) if profiler is not None else nullcontext()
            with total_section:
                crop_shape = tuple(int(v) for v in batch['image'].shape[-3:])
                if tuple(int(v) for v in preds.shape[-3:]) != crop_shape:
                    interpolate_section = profiler.section('normal_pooling/interpolate', preds.device) if profiler is not None else nullcontext()
                    with interpolate_section:
                        preds = F.interpolate(
                            preds,
                            size=crop_shape,
                            mode='trilinear',
                            align_corners=True,
                        )
                outside_bg_loss = _compute_normal_pooled_3d_background_loss(
                    preds,
                    batch['surface_mask'].to(device=preds.device, dtype=preds.dtype),
                    outside_threshold=float(pooling_config.get('outside_bg_threshold', 0.0)),
                    loss_weight=float(pooling_config.get('outside_bg_loss_weight', 1.0)),
                )
                pool_section = profiler.section('normal_pooling/pool_logits_along_normals', preds.device) if profiler is not None else nullcontext()
                with pool_section:
                    pooling_dtype = preds.dtype
                    flat_points = batch['flat_points_local_zyx'].to(dtype=pooling_dtype)
                    flat_normals = batch['flat_normals_local_zyx'].to(dtype=pooling_dtype)
                    if use_feature_head:
                        if feature_volume is None:
                            raise ValueError("normal_pooling.feature_head is enabled but feature_volume was not returned by the model")
                        pooled_logits, pooled_valid = model.normal_pooling_ray_head(
                            feature_volume.to(dtype=pooling_dtype),
                            flat_points,
                            flat_normals,
                            batch['flat_valid'],
                            neg_dist=float(pooling_config.get('neg_dist', 10.0)),
                            pos_dist=float(pooling_config.get('pos_dist', 10.0)),
                            sample_step=float(pooling_config.get('sample_step', 0.5)),
                            align_corners=True,
                            timer=(profiler.section if profiler is not None else None),
                        )
                    else:
                        pooled_logits, pooled_valid = pool_logits_along_normals(
                            preds,
                            flat_points,
                            flat_normals,
                            batch['flat_valid'],
                            neg_dist=float(pooling_config.get('neg_dist', 10.0)),
                            pos_dist=float(pooling_config.get('pos_dist', 10.0)),
                            sample_step=float(pooling_config.get('sample_step', 0.5)),
                            align_corners=True,
                            timer=(profiler.section if profiler is not None else None),
                        )
                targets_section = profiler.section('normal_pooling/build_targets_and_ignore_mask', preds.device) if profiler is not None else nullcontext()
                with targets_section:
                    targets = batch['flat_target']
                    ignore_mask = (
                        (batch['flat_supervision'] <= 0)
                        | (batch['flat_valid'] <= 0)
                        | (pooled_valid <= 0)
                    ).to(dtype=targets.dtype)
                return pooled_logits, targets, ignore_mask, outside_bg_loss

        if mode == 'full_3d':
            crop_shape = tuple(int(v) for v in batch['image'].shape[-3:])
            if preds.ndim != 5:
                raise ValueError(
                    f"full_3d mode expects volumetric logits with shape [B, 1, Z, Y, X], got {tuple(preds.shape)}"
                )
            if tuple(int(v) for v in preds.shape[-3:]) != crop_shape:
                preds = F.interpolate(
                    preds,
                    size=crop_shape,
                    mode='trilinear',
                    align_corners=True,
                )
            targets = batch['inklabels']
            ignore_mask = (batch['supervision_mask'] <= 0).to(dtype=targets.dtype)
            return preds, targets, ignore_mask, preds.new_zeros(())

        targets = (torch.amax(batch['inklabels'], dim=2) > 0).to(dtype=batch['inklabels'].dtype)
        supervision_mask = torch.amax(batch['supervision_mask'], dim=2)
        ignore_mask = (supervision_mask <= 0).to(dtype=targets.dtype)
        return preds, targets, ignore_mask, preds.new_zeros(())

    def refresh_progress_bar(current_train_loss):
        if not accelerator.is_main_process:
            return

        postfix = {'loss': f'{current_train_loss:.4f}'}
        if latest_val_loss is not None:
            postfix['val_loss'] = f'{latest_val_loss:.4f}'
        if latest_ema_val_loss is not None:
            postfix['ema_val_loss'] = f'{latest_ema_val_loss:.4f}'
        postfix['lr'] = f"{optimizer.param_groups[0]['lr']:.2e}"
        progress_bar.set_postfix(postfix, refresh=False)
        progress_bar.update(0)

    for step in progress_bar:
        model.train()
        if frozen_encoder is not None:
            frozen_encoder.eval()
        normal_pooling_profiler.start_step()

        next_batch_section = normal_pooling_profiler.section('dataloader/next_batch', accelerator.device) if normal_pooling_profiler.enabled else nullcontext()
        with next_batch_section:
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_dl)
                batch = next(train_iterator)
        _record_dataset_profile_batch(
            normal_pooling_profiler if normal_pooling_profiler.enabled else None,
            batch,
        )

        with accelerator.accumulate(model):
            forward_section = normal_pooling_profiler.section('model/forward', accelerator.device) if normal_pooling_profiler.enabled else nullcontext()
            with forward_section:
                with accelerator.autocast():
                    preds, feature_volume = _forward_model_with_optional_shared_features(
                        model,
                        get_model_input(batch),
                        model_crop_size,
                        capture_shared_features=use_feature_head,
                        stitched=use_stitched_forward,
                        use_gradient_checkpointing=stitched_gradient_checkpointing,
                    )
            loss_preds, targets, ignore_mask, outside_bg_loss = prepare_loss_inputs(
                preds,
                batch,
                profiler=normal_pooling_profiler if normal_pooling_profiler.enabled else None,
                feature_volume=feature_volume,
            )
            primary_loss_preds = _primary_supervision_output(loss_preds)
            loss_section = normal_pooling_profiler.section('normal_pooling/loss', primary_loss_preds.device) if normal_pooling_profiler.enabled else nullcontext()
            with loss_section:
                targets_with_ignore = _build_targets_with_ignore_for_loss(loss_preds, targets, ignore_mask)
                l = loss(
                    _float_supervision_tree(loss_preds),
                    _float_supervision_tree(targets_with_ignore),
                )
                l = l + outside_bg_loss
            if not torch.isfinite(l):
                raise RuntimeError(f"Non-finite loss at step {step}")
            backward_markers = _BackwardProfileMarkers.create(
                normal_pooling_profiler if normal_pooling_profiler.enabled else None,
                preds=_primary_supervision_output(preds),
                loss_preds=primary_loss_preds,
            )
            if backward_markers is None:
                backward_section = normal_pooling_profiler.section('train/backward_total', preds.device) if normal_pooling_profiler.enabled else nullcontext()
                with backward_section:
                    accelerator.backward(l)
            else:
                backward_markers.start()
                try:
                    accelerator.backward(l)
                finally:
                    backward_markers.finish()
            optimizer_section = normal_pooling_profiler.section('train/optimizer_step', accelerator.device) if normal_pooling_profiler.enabled else nullcontext()
            with optimizer_section:
                if grad_clip is not None and grad_clip > 0 and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                if accelerator.sync_gradients:
                    optimizer_step += 1
                    if ema_model is not None and optimizer_step >= ema_start_step:
                        if (optimizer_step - ema_start_step) % ema_update_every_steps == 0:
                            ema_state = ema_model.state_dict()
                            for name, model_value in unwrapped_model.state_dict().items():
                                ema_value = ema_state[name]
                                model_value = model_value.detach()
                                if torch.is_floating_point(ema_value):
                                    ema_value.lerp_(model_value.to(dtype=ema_value.dtype), 1.0 - ema_decay)
                                else:
                                    ema_value.copy_(model_value)

        train_loss = l.item()
        if accelerator.is_main_process:
            refresh_progress_bar(train_loss)

        if accelerator.is_main_process and step % log_every == 0:
            log_dict = {
                'train/loss': train_loss,
                'train/lr': optimizer.param_groups[0]['lr'],
                'step': step,
            }
            if normal_pooled_mode:
                log_dict['train/outside_bg_loss'] = float(outside_bg_loss.detach().item())
            latest_loss_metrics = getattr(loss, 'latest_metrics', None)
            if isinstance(latest_loss_metrics, dict):
                log_dict.update(latest_loss_metrics)
            if wandb.run is not None:
                wandb.log(log_dict, step=step)

        if step % val_every == 0 and step > 0:
            train_preview = PreviewAccumulator(
                accelerator=accelerator,
                get_model_input=get_model_input,
                normal_pooling_config=(config.get('normal_pooling') if normal_pooled_mode else None),
            )
            train_preview_batch = batch
            train_preview_preds = primary_loss_preds.detach()
            train_preview_targets = targets.detach()
            train_preview_ignore_mask = ignore_mask.detach()
            train_preview_volume_logits = _primary_supervision_output(preds).detach() if normal_pooled_mode else None
            if mode == 'full_3d':
                (
                    train_preview_batch,
                    train_preview_preds,
                    train_preview_targets,
                    train_preview_ignore_mask,
                ) = _build_full_3d_preview_batch(
                    batch,
                    train_preview_preds,
                    train_preview_targets,
                    train_preview_ignore_mask,
                )
            train_preview.add_batch(
                train_preview_batch,
                train_preview_preds,
                train_preview_targets,
                train_preview_ignore_mask,
                volume_logits=train_preview_volume_logits,
            )
            model.eval()
            val_losses = []
            ema_val_losses = []
            validation_counts = Confusion.zero_counts(device=accelerator.device)
            val_preview = PreviewAccumulator(
                accelerator=accelerator,
                get_model_input=get_model_input,
                normal_pooling_config=(config.get('normal_pooling') if normal_pooled_mode else None),
            )
            num_val_batches = min(len(val_dl), config.get('val_steps', 10))
            if num_val_batches == 0:
                if accelerator.is_main_process:
                    latest_val_loss = None
                    latest_ema_val_loss = None
                    refresh_progress_bar(train_loss)
                continue
            val_iterator = iter(val_dl)
            preview_batch_indices = set(
                random.sample(range(num_val_batches), k=min(val_preview_batches, num_val_batches))
            )
            with torch.no_grad():
                for val_batch_idx in range(num_val_batches):
                    val_batch = next(val_iterator)
                    with accelerator.autocast():
                        val_preds, val_feature_volume = _forward_model_with_optional_shared_features(
                            model,
                            get_model_input(val_batch),
                            model_crop_size,
                            capture_shared_features=use_feature_head,
                            stitched=use_stitched_forward,
                            use_gradient_checkpointing=stitched_gradient_checkpointing,
                        )
                    val_loss_preds, val_targets, val_ignore_mask, val_outside_bg_loss = prepare_loss_inputs(
                        val_preds,
                        val_batch,
                        feature_volume=val_feature_volume,
                    )
                    primary_val_loss_preds = _primary_supervision_output(val_loss_preds)
                    preview_preds = primary_val_loss_preds
                    preview_volume_preds = _primary_supervision_output(val_preds)
                    val_targets_with_ignore = _build_targets_with_ignore_for_loss(
                        val_loss_preds,
                        val_targets,
                        val_ignore_mask,
                    )
                    val_l = loss(
                        _float_supervision_tree(val_loss_preds),
                        _float_supervision_tree(val_targets_with_ignore),
                    ) + val_outside_bg_loss
                    val_losses.append(val_l.item())
                    batch_counts = validation_confusion_metric.compute_batch(
                        ValidationMetricBatch(
                            logits=primary_val_loss_preds.detach(),
                            targets=val_targets.detach(),
                            valid_mask=(val_ignore_mask <= 0).detach(),
                        )
                    )
                    gathered_batch_counts = accelerator.gather_for_metrics(
                        torch.stack(
                            (batch_counts.tp, batch_counts.fp, batch_counts.fn, batch_counts.tn)
                        ).unsqueeze(0)
                    )
                    validation_counts = Confusion.add_counts(
                        validation_counts,
                        ConfusionCounts(
                            tp=gathered_batch_counts[:, 0].sum(),
                            fp=gathered_batch_counts[:, 1].sum(),
                            fn=gathered_batch_counts[:, 2].sum(),
                            tn=gathered_batch_counts[:, 3].sum(),
                        ),
                    )
                    if ema_model is not None and ema_validate:
                        with accelerator.autocast():
                            ema_val_preds, ema_val_feature_volume = _forward_model_with_optional_shared_features(
                                ema_model,
                                get_model_input(val_batch),
                                model_crop_size,
                                capture_shared_features=use_feature_head,
                                stitched=use_stitched_forward,
                                use_gradient_checkpointing=stitched_gradient_checkpointing,
                            )
                        ema_val_loss_preds, _, _, ema_val_outside_bg_loss = prepare_loss_inputs(
                            ema_val_preds,
                            val_batch,
                            feature_volume=ema_val_feature_volume,
                        )
                        ema_val_targets_with_ignore = _build_targets_with_ignore_for_loss(
                            ema_val_loss_preds,
                            val_targets,
                            val_ignore_mask,
                        )
                        ema_val_l = loss(
                            _float_supervision_tree(ema_val_loss_preds),
                            _float_supervision_tree(ema_val_targets_with_ignore),
                        ) + ema_val_outside_bg_loss
                        ema_val_losses.append(ema_val_l.item())
                        preview_preds = _primary_supervision_output(ema_val_loss_preds)
                        preview_volume_preds = _primary_supervision_output(ema_val_preds)

                    if val_batch_idx in preview_batch_indices:
                        val_preview_batch = val_batch
                        val_preview_preds = preview_preds.detach()
                        val_preview_targets = val_targets.detach()
                        val_preview_ignore = val_ignore_mask.detach()
                        val_preview_volume_logits = preview_volume_preds.detach() if normal_pooled_mode else None
                        if mode == 'full_3d':
                            (
                                val_preview_batch,
                                val_preview_preds,
                                val_preview_targets,
                                val_preview_ignore,
                            ) = _build_full_3d_preview_batch(
                                val_batch,
                                val_preview_preds,
                                val_preview_targets,
                                val_preview_ignore,
                            )
                        val_preview.add_batch(
                            val_preview_batch,
                            val_preview_preds,
                            val_preview_targets,
                            val_preview_ignore,
                            volume_logits=val_preview_volume_logits,
                        )

            mean_val_loss = np.mean(val_losses)
            mean_ema_val_loss = np.mean(ema_val_losses) if ema_val_losses else None
            if accelerator.is_main_process:
                latest_val_loss = float(mean_val_loss)
                latest_ema_val_loss = (float(mean_ema_val_loss) if mean_ema_val_loss is not None else None)
                refresh_progress_bar(train_loss)
                balanced_accuracy = float(
                    validation_balanced_accuracy_metric._from_counts(validation_counts).item()
                )
                log_dict = build_validation_preview_log(
                    step=step,
                    train_preview=train_preview,
                    val_preview=val_preview,
                    train_preview_dir=train_preview_dir,
                    val_preview_dir=val_preview_dir,
                    mean_val_loss=mean_val_loss,
                    mean_ema_val_loss=mean_ema_val_loss,
                    include_wandb_images=wandb.run is not None,
                )
                log_dict.update(
                    {
                        'val/balanced_accuracy': balanced_accuracy,
                        'val/tp': float(validation_counts.tp.item()),
                        'val/fp': float(validation_counts.fp.item()),
                        'val/fn': float(validation_counts.fn.item()),
                        'val/tn': float(validation_counts.tn.item()),
                    }
                )
                if wandb.run is not None:
                    wandb.log(log_dict, step=step)

        if accelerator.is_main_process and step % save_every == 0 and step > 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
                'step': step,
                'wandb_run_id': wandb.run.id if wandb.run is not None else config.get('wandb_run_id'),
            }
            if ema_model is not None and ema_save_in_checkpoint:
                checkpoint['ema_model'] = ema_model.state_dict()
                checkpoint['ema_optimizer_step'] = optimizer_step
            torch.save(checkpoint, f'{out_dir}/ckpt_{step:06}.pth')

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        for line in normal_pooling_profiler.summary_lines():
            accelerator.print(line)

if __name__ == '__main__':
    train()
