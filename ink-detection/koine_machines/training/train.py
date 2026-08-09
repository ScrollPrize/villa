import json
import math
import os
import time
from copy import deepcopy
from dataclasses import dataclass
import wandb
import random
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
import accelerate
from accelerate.utils import (
    DistributedDataParallelKwargs,
    GradientAccumulationPlugin,
    set_seed,
)
import click
from vesuvius.models.utils import InitWeights_He
from vesuvius.models.training.optimizers import (
    OptimizerParamGroupTarget,
    create_optimizer,
)
from vesuvius.models.training.lr_schedulers import get_scheduler
from koine_machines.models.make_model import make_model
from koine_machines.data.ink_dataset import InkDataset
from koine_machines.data.normal_pooled_sample import native_volume_downsample_factor
from koine_machines.models.load_checkpoint import load_training_checkpoint_from_config, restore_training_state
from koine_machines.evaluation.metrics.balanced_accuracy import BalancedAccuracy
from koine_machines.evaluation.metrics.confusion import Confusion, ConfusionCounts
from koine_machines.training.normal_pooling import (
    collate_normal_pooled_batch,
    pool_logits_along_normals,
)
from koine_machines.training.deep_supervision import (
    DeepSupervisionWrapper,
    _compute_ds_weights,
    build_deep_supervision_targets,
)
from koine_machines.training.visualization import PreviewAccumulator, build_validation_preview_log
from koine_machines.training.loss.losses import create_loss_from_config
from koine_machines.training.stitching import resolve_model_and_loader_patch_sizes, run_model_forward
from koine_machines.training.samplers import (
    FixedScrollPriorStratifiedBatchSampler,
    hierarchical_scroll_segment_weights,
)
from koine_machines.models.input_padding import (
    center_pad_input_depth,
    configured_input_pad_depth,
)


@dataclass
class ValidationMetricBatch:
    logits: torch.Tensor
    targets: torch.Tensor
    valid_mask: torch.Tensor | None = None

    def require_targets(self):
        return self.targets


_FULL_3D_SINGLE_WRAP_MODE = "full_3d_single_wrap"
_FULL_3D_TRAINING_MODES = {"full_3d", _FULL_3D_SINGLE_WRAP_MODE}
_NATIVE_3D_TRAINING_MODES = {"normal_pooled_3d", *_FULL_3D_TRAINING_MODES}


def _is_native_3d_training_mode(mode):
    return str(mode).strip().lower() in _NATIVE_3D_TRAINING_MODES


def _uses_surface_mask_channel(mode):
    return str(mode).strip().lower() in {"normal_pooled_3d", _FULL_3D_SINGLE_WRAP_MODE}


def _disable_z_projection_for_normal_pooled_3d(config):
    config.setdefault('model_config', {})['z_projection_mode'] = 'none'
    for target_info in (config.get('targets') or {}).values():
        if isinstance(target_info, dict):
            target_info['z_projection_mode'] = 'none'
            if isinstance(target_info.get('z_projection'), dict):
                target_info['z_projection']['mode'] = 'none'


def _full_3d_dilation_distances_for_level(config):
    """Dilation distances are configured in level-0 voxels (like the projection
    half-thicknesses); convert them to voxels of the native volume level."""
    full_3d_config = config.get('full_3d') or {}
    label_dilation = float(full_3d_config.get('label_dilation_distance', 0.0))
    supervision_dilation = float(full_3d_config.get('supervision_dilation_distance', 0.0))
    if label_dilation <= 0.0 and supervision_dilation <= 0.0:
        return label_dilation, supervision_dilation
    factors = {
        native_volume_downsample_factor(ds['volume_scale'])
        for ds in config['datasets']
    }
    if len(factors) != 1:
        raise ValueError(
            "full_3d dilation distances require a single volume_scale across datasets, "
            f"got downsample factors {sorted(factors)!r}"
        )
    factor = float(factors.pop())
    return label_dilation / factor, supervision_dilation / factor


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

    z_index = int(batch['supervision_mask'].shape[-3] // 2)

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

def _apply_cucim_label_dilation(batch, label_dilation_distance, supervision_dilation_distance):
    """Apply GPU-accelerated dilation to inklabels and supervision_mask via cucim."""
    from vesuvius.models.augmentation.transforms.cucim_dilate import dilate_label_batch_with_cucim

    inklabels = batch['inklabels']
    supervision_mask = batch['supervision_mask']
    ones = torch.ones(inklabels.shape[0], 1, *inklabels.shape[2:], device=inklabels.device, dtype=inklabels.dtype)
    if label_dilation_distance > 0.0:
        inklabels = dilate_label_batch_with_cucim(inklabels, ones, label_dilation_distance)
    if supervision_dilation_distance > 0.0:
        bg = ((supervision_mask > 0) & (inklabels <= 0)).to(dtype=inklabels.dtype)
        bg = dilate_label_batch_with_cucim(bg, ones, supervision_dilation_distance)
        bg = bg * (inklabels <= 0).to(dtype=bg.dtype)
        supervision_mask = ((inklabels > 0) | (bg > 0)).to(dtype=supervision_mask.dtype)
    elif label_dilation_distance > 0.0:
        supervision_mask = ((inklabels > 0) | (supervision_mask > 0)).to(dtype=supervision_mask.dtype)
    batch['inklabels'] = inklabels
    batch['supervision_mask'] = supervision_mask


def _distributed_mean_scalar(accelerator, value):
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
    else:
        tensor = torch.tensor(float(value), device=accelerator.device)
    tensor = tensor.to(device=accelerator.device, dtype=torch.float32).reshape(())
    return accelerator.reduce(tensor, reduction='mean')


def _masked_unsmoothed_bce_with_logits(logits, targets, ignore_mask):
    """Plain BCE over valid pixels, independent of configured loss smoothing."""
    valid_mask = (ignore_mask <= 0).to(dtype=torch.float32)
    elements = F.binary_cross_entropy_with_logits(
        logits.detach().float(),
        targets.detach().float(),
        reduction='none',
    )
    return (elements * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)


def _distributed_mean_metrics(accelerator, metrics):
    if not isinstance(metrics, dict):
        return {}
    return {
        str(name): float(_distributed_mean_scalar(accelerator, metric_value).item())
        for name, metric_value in metrics.items()
    }


def _append_jsonl(path, record):
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def _benchmark_summary(*, elapsed_seconds, measured_steps, batch_size,
                       world_size, data_wait_seconds, peak_allocated_bytes,
                       peak_reserved_bytes):
    """Build the machine-readable summary used by short production benchmarks."""
    elapsed_seconds = float(elapsed_seconds)
    measured_steps = int(measured_steps)
    if elapsed_seconds <= 0 or measured_steps <= 0:
        raise ValueError("benchmark duration and measured steps must be positive")
    examples = measured_steps * int(batch_size) * int(world_size)
    return {
        'elapsed_seconds': elapsed_seconds,
        'measured_steps': measured_steps,
        'batch_size_per_process': int(batch_size),
        'world_size': int(world_size),
        'examples': examples,
        'steps_per_second': measured_steps / elapsed_seconds,
        'examples_per_second': examples / elapsed_seconds,
        'data_wait_seconds': float(data_wait_seconds),
        'data_wait_seconds_per_step': float(data_wait_seconds) / measured_steps,
        'peak_allocated_bytes': int(peak_allocated_bytes),
        'peak_reserved_bytes': int(peak_reserved_bytes),
    }


def _forward_model(
    model,
    image: torch.Tensor,
    model_crop_size,
    *,
    stitched: bool = True,
    use_gradient_checkpointing: bool = True,
):
    return run_model_forward(
        model,
        image,
        model_crop_size,
        stitched=stitched,
        use_gradient_checkpointing=use_gradient_checkpointing,
    )

@click.command()
@click.argument('config_path', type=click.Path(exists=True))
def train(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)

    benchmark_config = dict(config.get('benchmark') or {})
    benchmark_enabled = bool(benchmark_config.get('enabled', False))
    benchmark_warmup_steps = int(benchmark_config.get('warmup_steps', 0))
    if benchmark_enabled and not 0 <= benchmark_warmup_steps < int(config['num_iterations']):
        raise ValueError("benchmark warmup_steps must be in [0, num_iterations)")

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
    surface_mask_channel = _uses_surface_mask_channel(mode)
    pooling_config = config.get('normal_pooling') or {}
    deep_supervision_enabled = bool(config.get('enable_deep_supervision', False))
    model_type = str(config.get('model_type', '')).strip().lower()
    input_pad_depth_to = configured_input_pad_depth(config)

    if normal_pooled_mode and model_type.startswith('resnet3d'):
        raise ValueError("normal_pooled_3d is currently only supported with the vesuvius_unet model path")
    if native_3d_mode:
        _disable_z_projection_for_normal_pooled_3d(config)
        config['in_channels'] = 1 + int(surface_mask_channel)

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
    log_every = config.get('log_every', 50)
    val_preview_batches = config.get('val_preview_batches', 3)
    best_checkpoint_metric = config.get('best_checkpoint_metric')
    if best_checkpoint_metric not in (None, 'val_loss', 'val_balanced_accuracy'):
        raise ValueError(
            "best_checkpoint_metric must be one of "
            "{null, 'val_loss', 'val_balanced_accuracy'}"
        )
    best_checkpoint_value = None

    dataloader_config = accelerate.DataLoaderConfiguration(non_blocking = True)
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=bool(config.get('ddp_find_unused_parameters', False)),
        broadcast_buffers=bool(config.get('ddp_broadcast_buffers', False)),
    )

    # The training loop reuses the dataloader indefinitely, so keep accumulation boundaries independent of dataloader exhaustion.
    gradient_accumulation_plugin = GradientAccumulationPlugin(num_steps=grad_acc_steps, sync_with_dataloader=False)
    accelerator = accelerate.Accelerator(
        mixed_precision=config.get('mixed_precision', "fp16"),
        gradient_accumulation_plugin=gradient_accumulation_plugin,
        dataloader_config=dataloader_config,
        kwargs_handlers=[ddp_kwargs],
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

    full_3d_label_dilation = 0.0
    full_3d_supervision_dilation = 0.0
    if mode in _FULL_3D_TRAINING_MODES:
        full_3d_label_dilation, full_3d_supervision_dilation = (
            _full_3d_dilation_distances_for_level(config)
        )
        if full_3d_label_dilation > 0.0 or full_3d_supervision_dilation > 0.0:
            ds_full_3d = dataset_config.setdefault('full_3d', {})
            ds_full_3d['label_dilation_distance'] = 0.0
            ds_full_3d['supervision_dilation_distance'] = 0.0

    shared_ds = InkDataset(dataset_config, do_augmentations=False)
    if len(shared_ds.training_patches) == 0:
        raise ValueError("FlatInkDataset produced no training patches after applying supervision masking")
    train_ds = InkDataset(
        dataset_config,
        do_augmentations=True,
        patches=shared_ds.training_patches,
        segments=shared_ds.segments,
    )
    val_ds = InkDataset(
        dataset_config,
        do_augmentations=False,
        patches=shared_ds.validation_patches,
        segments=shared_ds.segments,
    )
    train_subset = train_ds
    val_subset = val_ds

    dataloader_workers = int(config.get('dataloader_workers', 0))
    dataloader_kwargs = {
        'pin_memory': bool(config.get('pin_memory', accelerator.device.type == 'cuda')),
    }
    if dataloader_workers > 0:
        dataloader_kwargs['multiprocessing_context'] = 'spawn'
        dataloader_kwargs['persistent_workers'] = True
        dataloader_kwargs['prefetch_factor'] = int(config.get('prefetch_factor', 2))
    if normal_pooled_mode:
        dataloader_kwargs['collate_fn'] = collate_normal_pooled_batch

    sampling_strategy = str(config.get('sampling_strategy', 'uniform')).strip().lower()
    train_generator = torch.Generator().manual_seed(config['seed'])
    train_sampler = None
    fixed_batch_sampler = None
    if sampling_strategy == 'uniform':
        train_shuffle = True
        sampling_audit = {
            'strategy': 'uniform',
            'patches': len(train_subset),
        }
    elif sampling_strategy == 'scroll_segment_balanced':
        sample_weights, sampling_audit = hierarchical_scroll_segment_weights(
            shared_ds.training_patches,
            config['datasets'],
        )
        train_sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(train_subset),
            replacement=True,
            generator=train_generator,
        )
        train_shuffle = False
    elif sampling_strategy == 'fixed_scroll_prior_stratified':
        fixed_prior = dict(config.get('fixed_scroll_prior') or {})
        if int(fixed_prior.get('seed', -1)) != int(config['seed']):
            raise ValueError(
                "fixed_scroll_prior.seed must match the training seed: "
                f"{fixed_prior.get('seed')!r} vs {config['seed']!r}"
            )
        fixed_batch_sampler = FixedScrollPriorStratifiedBatchSampler(
            shared_ds.training_patches,
            config['datasets'],
            batch_quotas=fixed_prior.get('target_batch_counts') or {},
            batch_size=config['batch_size'],
            seed=config['seed'],
        )
        sampling_audit = fixed_batch_sampler.definition_audit()
        train_shuffle = False
    else:
        raise ValueError(
            "sampling_strategy must be 'uniform', 'scroll_segment_balanced', "
            "or 'fixed_scroll_prior_stratified', got "
            f"{sampling_strategy!r}"
        )
    accelerator.print('sampling_audit=' + json.dumps(sampling_audit, sort_keys=True), flush=True)
    if fixed_batch_sampler is not None:
        train_dl = DataLoader(
            train_subset,
            batch_sampler=fixed_batch_sampler,
            num_workers=dataloader_workers,
            **dataloader_kwargs,
        )
    else:
        train_dl = DataLoader(
            train_subset,
            batch_size=config['batch_size'],
            shuffle=train_shuffle,
            sampler=train_sampler,
            generator=train_generator,
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
    optimizer_target = model
    pretrained_backbone = (config.get('model_config') or {}).get('pretrained_backbone')
    freeze_encoder = False
    if pretrained_backbone:
        freeze_encoder = bool(
            config.get('freeze_encoder', False)
            or (config.get('model_config') or {}).get('freeze_encoder', False)
        )
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
                'betas': config.get('optimizer_betas', (0.9, 0.999)),
                'momentum': config.get('optimizer_momentum', 0.99),
                'nesterov': config.get('optimizer_nesterov', True),
                }, OptimizerParamGroupTarget(optimizer_target) if isinstance(optimizer_target, list) else optimizer_target)

    scheduler_config = dict(config.get('scheduler') or {})
    scheduler_name = str(
        scheduler_config.get('name', 'diffusers_cosine_warmup')
    ).strip().lower()
    if scheduler_name == 'cosine_annealing':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(scheduler_config.get('t_max', max_steps)),
            eta_min=float(scheduler_config.get('eta_min', 0.0)),
        )
    elif scheduler_name == 'one_cycle':
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=float(scheduler_config['max_lr']),
            total_steps=int(scheduler_config.get('total_steps', max_steps)),
            pct_start=float(scheduler_config.get('pct_start', 0.3)),
            final_div_factor=float(
                scheduler_config.get('final_div_factor', 1e4)
            ),
        )
    elif scheduler_name == 'diffusers_cosine_warmup':
        lr_scheduler = get_scheduler(
            'diffusers_cosine_warmup',
            optimizer,
            initial_lr=learning_rate,
            max_steps=max_steps,
            warmup_steps=config.get('warmup_steps', 1000),
        )
    else:
        raise ValueError(
            f"Unsupported scheduler.name={scheduler_name!r}; expected "
            "'diffusers_cosine_warmup', 'cosine_annealing', or 'one_cycle'"
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

    model, optimizer, train_dl, val_dl = accelerator.prepare(
            model, optimizer, train_dl, val_dl,
        )
    # NOTE: we intentionally do NOT prepare lr_scheduler with Accelerate.
    # AcceleratedScheduler calls scheduler.step() num_processes times per
    # optimizer step (when split_batches=False), which makes the LR schedule
    # run num_processes-x faster than intended.  Instead we step the raw
    # scheduler ourselves exactly once per optimizer step inside the
    # sync_gradients guard.
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
            unwrapped_model,
            optimizer,
            lr_scheduler,
            checkpoint,
            checkpoint_path,
            load_weights_only=weights_only,
            ema_model=ema_model,
        )

        accelerator.print(f"Loaded checkpoint '{checkpoint_path}'"+ (f" and resuming from step {start_step}" if resume_full_state else " (weights only)"))
        
    train_iterator = iter(train_dl)
    sampling_audit_every = int(config.get('sampling_audit_every', save_every))
    if sampling_audit_every <= 0:
        raise ValueError("sampling_audit_every must be positive")
    verify_finite_gradients_steps = int(
        config.get('verify_finite_gradients_steps', 0)
    )
    if verify_finite_gradients_steps < 0:
        raise ValueError("verify_finite_gradients_steps cannot be negative")
    max_amp_overflow_events = int(config.get('max_amp_overflow_events', 0))
    if max_amp_overflow_events < 0:
        raise ValueError("max_amp_overflow_events cannot be negative")
    gradient_health = {
        'checked_steps': 0,
        'nonfinite_events': [],
        'max_amp_overflow_events': max_amp_overflow_events,
    }

    def write_gradient_health() -> None:
        if not accelerator.is_main_process or verify_finite_gradients_steps <= 0:
            return
        output_path = os.path.join(out_dir, 'gradient_health.json')
        temporary_path = output_path + '.partial'
        with open(temporary_path, 'w', encoding='utf-8') as stream:
            json.dump(gradient_health, stream, indent=2, sort_keys=True)
            stream.write('\n')
        os.replace(temporary_path, output_path)

    write_gradient_health()

    def write_fixed_sampling_audit(training_step: int) -> None:
        if fixed_batch_sampler is None or not accelerator.is_main_process:
            return
        observed = fixed_batch_sampler.observed_audit()
        observed['training_step_completed'] = int(training_step)
        observed['target'] = sampling_audit
        output_path = os.path.join(out_dir, 'sampling_observed.json')
        temporary_path = output_path + '.partial'
        with open(temporary_path, 'w', encoding='utf-8') as stream:
            json.dump(observed, stream, indent=2, sort_keys=True)
            stream.write('\n')
        os.replace(temporary_path, output_path)
        accelerator.print('sampling_observed=' + json.dumps(observed, sort_keys=True))

    write_fixed_sampling_audit(start_step)
    latest_val_loss = None
    latest_ema_val_loss = None
    validation_confusion_metric = Confusion()
    validation_balanced_accuracy_metric = BalancedAccuracy()
    
    progress_bar = tqdm(
        range(start_step, config['num_iterations']),
        total=config['num_iterations'],
        initial=start_step,
        disable=not accelerator.is_main_process,
        dynamic_ncols=True,
    )
    benchmark_started_at = None
    benchmark_data_wait_seconds = 0.0
    benchmark_measured_steps = 0
    if benchmark_enabled and benchmark_warmup_steps == 0:
        if accelerator.device.type == 'cuda':
            torch.cuda.synchronize(accelerator.device)
            torch.cuda.reset_peak_memory_stats(accelerator.device)
        benchmark_started_at = time.perf_counter()

    def get_model_input(batch):
        image = batch['image'].float()
        if surface_mask_channel:
            surface_mask = batch['surface_mask'].float()
            image = torch.cat([image, surface_mask], dim=1)
        return center_pad_input_depth(image, input_pad_depth_to)

    def prepare_loss_inputs(preds, batch):
        if isinstance(preds, (list, tuple)):
            loss_preds = []
            targets = None
            ignore_mask = None
            for idx, pred_level in enumerate(preds):
                current_loss_preds, current_targets, current_ignore_mask = prepare_loss_inputs(
                    pred_level,
                    batch,
                )
                loss_preds.append(current_loss_preds)
                if idx == 0:
                    targets = current_targets
                    ignore_mask = current_ignore_mask
            return type(preds)(loss_preds), targets, ignore_mask

        if normal_pooled_mode:
            crop_shape = tuple(int(v) for v in batch['image'].shape[-3:])
            if tuple(int(v) for v in preds.shape[-3:]) != crop_shape:
                preds = F.interpolate(
                    preds,
                    size=crop_shape,
                    mode='trilinear',
                    align_corners=True,
                )
            pooling_dtype = preds.dtype
            flat_points = batch['flat_points_local_zyx'].to(dtype=pooling_dtype)
            flat_normals = batch['flat_normals_local_zyx'].to(dtype=pooling_dtype)
            pooled_logits, pooled_valid = pool_logits_along_normals(
                preds,
                flat_points,
                flat_normals,
                batch['flat_valid'],
                neg_dist=float(pooling_config.get('neg_dist', 10.0)),
                pos_dist=float(pooling_config.get('pos_dist', 10.0)),
                sample_step=float(pooling_config.get('sample_step', 0.5)),
                align_corners=True,
            )
            targets = batch['flat_target']
            ignore_mask = ((batch['flat_supervision'] <= 0) | (batch['flat_valid'] <= 0) | (pooled_valid <= 0)).to(dtype=targets.dtype)
            return pooled_logits, targets, ignore_mask

        if mode in _FULL_3D_TRAINING_MODES:
            crop_shape = tuple(int(v) for v in batch['image'].shape[-3:])
            if tuple(int(v) for v in preds.shape[-3:]) != crop_shape:
                preds = F.interpolate(
                    preds,
                    size=crop_shape,
                    mode='trilinear',
                    align_corners=True,
                )
            targets = batch['inklabels']
            ignore_mask = (batch['supervision_mask'] <= 0).to(dtype=targets.dtype)
            return preds, targets, ignore_mask

        targets = (torch.amax(batch['inklabels'], dim=2) > 0).to(dtype=batch['inklabels'].dtype)
        supervision_mask = torch.amax(batch['supervision_mask'], dim=2)
        ignore_mask = (supervision_mask <= 0).to(dtype=targets.dtype)
        output_size = tuple(int(value) for value in preds.shape[-2:])
        if tuple(int(value) for value in targets.shape[-2:]) != output_size:
            targets = F.interpolate(
                targets.float(), size=output_size, mode='nearest'
            ).to(dtype=batch['inklabels'].dtype)
            ignore_mask = F.interpolate(
                ignore_mask.float(), size=output_size, mode='nearest'
            ).to(dtype=targets.dtype)
        return preds, targets, ignore_mask

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

    def save_checkpoint(step, *, force=False, filename=None, validation_metrics=None):
        if not accelerator.is_main_process or (
            not force and (step + 1) % save_every != 0
        ):
            return
        checkpoint = {
            'model': accelerator.get_state_dict(model),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'config': config,
            'step': step,
            'wandb_run_id': wandb.run.id if wandb.run is not None else config.get('wandb_run_id'),
        }
        if validation_metrics is not None:
            checkpoint['validation_metrics'] = dict(validation_metrics)
        if ema_model is not None and ema_save_in_checkpoint:
            checkpoint['ema_model'] = ema_model.state_dict()
            checkpoint['ema_optimizer_step'] = optimizer_step
        checkpoint_name = filename or f'ckpt_{step + 1:06}.pth'
        torch.save(checkpoint, os.path.join(out_dir, checkpoint_name))

    for step in progress_bar:
        model.train()
        if frozen_encoder is not None:
            frozen_encoder.eval()
        batch_fetch_started_at = time.perf_counter()
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_dl)
            batch = next(train_iterator)
        batch_fetch_seconds = time.perf_counter() - batch_fetch_started_at
        if full_3d_label_dilation > 0.0 or full_3d_supervision_dilation > 0.0:
            _apply_cucim_label_dilation(batch, full_3d_label_dilation, full_3d_supervision_dilation)

        with accelerator.accumulate(model):
            with accelerator.autocast():
                preds = _forward_model(
                    model,
                    get_model_input(batch),
                    model_crop_size,
                    stitched=use_stitched_forward,
                    use_gradient_checkpointing=stitched_gradient_checkpointing,
                )
            loss_preds, targets, ignore_mask = prepare_loss_inputs(
                preds,
                batch,
            )
            primary_loss_preds = loss_preds[0] if isinstance(loss_preds, (list, tuple)) else loss_preds
            targets_with_ignore = build_deep_supervision_targets(targets, loss_preds, mode='nearest')
            ds_ignore = build_deep_supervision_targets(ignore_mask, loss_preds, mode='nearest')
            targets_with_ignore = (
                type(targets_with_ignore)(
                    torch.cat([target_level, ignore_level], dim=1)
                    for target_level, ignore_level in zip(targets_with_ignore, ds_ignore)
                )
                if isinstance(targets_with_ignore, (list, tuple))
                else torch.cat([targets_with_ignore, ds_ignore], dim=1)
            )
            l = loss(
                type(loss_preds)(v.float() for v in loss_preds) if isinstance(loss_preds, (list, tuple)) else loss_preds.float(),
                (
                    type(targets_with_ignore)(v.float() for v in targets_with_ignore)
                    if isinstance(targets_with_ignore, (list, tuple))
                    else targets_with_ignore.float()
                ),
            )
            if not torch.isfinite(l):
                raise RuntimeError(f"Non-finite loss at step {step}")
            accelerator.backward(l)
            if grad_clip is not None and grad_clip > 0 and accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), grad_clip)
            gradient_event = None
            if step < verify_finite_gradients_steps and accelerator.sync_gradients:
                gradient_health['checked_steps'] += 1
                non_finite_gradients = [
                    name
                    for name, parameter in model.named_parameters()
                    if parameter.grad is not None
                    and not bool(torch.isfinite(parameter.grad).all().item())
                ]
                if non_finite_gradients:
                    gradient_event = {
                        'step': int(step),
                        'parameters': non_finite_gradients[:10],
                    }
                    gradient_health['nonfinite_events'].append(gradient_event)
                    accelerator.print(
                        f"AMP gradient overflow at step {step}: "
                        + ', '.join(non_finite_gradients[:10])
                    )
                    if len(gradient_health['nonfinite_events']) > max_amp_overflow_events:
                        write_gradient_health()
                        raise RuntimeError(
                            "Exceeded permitted AMP gradient overflow events: "
                            f"{len(gradient_health['nonfinite_events'])} > "
                            f"{max_amp_overflow_events}"
                        )
            optimizer.step()
            if gradient_event is not None:
                gradient_event['optimizer_step_was_skipped'] = bool(
                    accelerator.optimizer_step_was_skipped
                )
            if step < verify_finite_gradients_steps and accelerator.sync_gradients:
                write_gradient_health()
            optimizer.zero_grad(set_to_none=True)
            if accelerator.sync_gradients:
                lr_scheduler.step()
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

        if benchmark_enabled:
            if step >= benchmark_warmup_steps:
                benchmark_data_wait_seconds += batch_fetch_seconds
                benchmark_measured_steps += 1
            if step + 1 == benchmark_warmup_steps:
                if accelerator.device.type == 'cuda':
                    torch.cuda.synchronize(accelerator.device)
                    torch.cuda.reset_peak_memory_stats(accelerator.device)
                benchmark_started_at = time.perf_counter()

        # Distributed reductions are collective ops — all ranks must participate.
        should_log = step % log_every == 0
        if should_log:
            train_loss = float(_distributed_mean_scalar(accelerator, l).item())
        else:
            train_loss = float(l.item())
        if accelerator.is_main_process:
            refresh_progress_bar(train_loss)

        reduced_loss_metrics = None
        if should_log:
            latest_loss_metrics = getattr(loss, 'latest_metrics', None)
            if isinstance(latest_loss_metrics, dict):
                reduced_loss_metrics = _distributed_mean_metrics(accelerator, latest_loss_metrics)

        if accelerator.is_main_process and should_log:
            log_dict = {
                'train/loss': train_loss,
                'train/lr': optimizer.param_groups[0]['lr'],
                'step': step,
            }
            if reduced_loss_metrics is not None:
                log_dict.update(reduced_loss_metrics)
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
            train_preview_volume_logits = (preds[0] if isinstance(preds, (list, tuple)) else preds).detach() if normal_pooled_mode else None
            if mode in _FULL_3D_TRAINING_MODES:
                (train_preview_batch,train_preview_preds,train_preview_targets,train_preview_ignore_mask) = _build_full_3d_preview_batch(
                    batch,
                    train_preview_preds,
                    train_preview_targets,
                    train_preview_ignore_mask,
                )
            train_preview.add_batch(train_preview_batch,train_preview_preds,train_preview_targets,train_preview_ignore_mask,volume_logits=train_preview_volume_logits)
            model.eval()
            val_loss_total = torch.zeros((), device=accelerator.device, dtype=torch.float32)
            val_loss_batches = torch.zeros((), device=accelerator.device, dtype=torch.float32)
            val_bce_unsmoothed_total = torch.zeros((), device=accelerator.device, dtype=torch.float32)
            ema_val_loss_total = torch.zeros((), device=accelerator.device, dtype=torch.float32)
            ema_val_loss_batches = torch.zeros((), device=accelerator.device, dtype=torch.float32)
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
                save_checkpoint(step)
                continue
            val_iterator = iter(val_dl)
            preview_batch_indices = set(random.sample(range(num_val_batches), k=min(val_preview_batches, num_val_batches)))
            with torch.no_grad():
                for val_batch_idx in range(num_val_batches):
                    val_batch = next(val_iterator)
                    if full_3d_label_dilation > 0.0 or full_3d_supervision_dilation > 0.0:
                        _apply_cucim_label_dilation(val_batch, full_3d_label_dilation, full_3d_supervision_dilation)
                    with accelerator.autocast():
                        val_preds = _forward_model(
                            model,
                            get_model_input(val_batch),
                            model_crop_size,
                            stitched=use_stitched_forward,
                            use_gradient_checkpointing=stitched_gradient_checkpointing,
                        )
                        
                    val_loss_preds, val_targets, val_ignore_mask = prepare_loss_inputs(val_preds,val_batch)
                    primary_val_loss_preds = val_loss_preds[0] if isinstance(val_loss_preds, (list, tuple)) else val_loss_preds
                    preview_preds = primary_val_loss_preds
                    preview_volume_preds = val_preds[0] if isinstance(val_preds, (list, tuple)) else val_preds
                    val_targets_with_ignore = build_deep_supervision_targets(val_targets, val_loss_preds, mode='nearest')
                    ds_ignore = build_deep_supervision_targets(val_ignore_mask, val_loss_preds, mode='nearest')
                    val_targets_with_ignore = (
                        type(val_targets_with_ignore)(
                            torch.cat([target_level, ignore_level], dim=1)
                            for target_level, ignore_level in zip(val_targets_with_ignore, ds_ignore)
                        )
                        if isinstance(val_targets_with_ignore, (list, tuple))
                        else torch.cat([val_targets_with_ignore, ds_ignore], dim=1)
                    )
                    val_l = loss(
                        type(val_loss_preds)(v.float() for v in val_loss_preds) if isinstance(val_loss_preds, (list, tuple)) else val_loss_preds.float(),
                        (
                            type(val_targets_with_ignore)(v.float() for v in val_targets_with_ignore)
                            if isinstance(val_targets_with_ignore, (list, tuple))
                            else val_targets_with_ignore.float()
                        ),
                    )
                    val_loss_total = val_loss_total + _distributed_mean_scalar(accelerator, val_l)
                    val_loss_batches = val_loss_batches + 1.0
                    # The configured loss may smooth its BCE targets, which charges
                    # confident correct predictions too; track plain BCE separately
                    # so calibration drift is not read as a generalization failure.
                    val_bce_unsmoothed_total = val_bce_unsmoothed_total + _distributed_mean_scalar(
                        accelerator,
                        _masked_unsmoothed_bce_with_logits(
                            primary_val_loss_preds,
                            val_targets,
                            val_ignore_mask,
                        ),
                    )
                    batch_counts = validation_confusion_metric.compute_batch(
                        ValidationMetricBatch(
                            logits=primary_val_loss_preds.detach(),
                            targets=val_targets.detach(),
                            valid_mask=(val_ignore_mask <= 0).detach(),
                        )
                    )
                    gathered_batch_counts = accelerator.gather_for_metrics(torch.stack((batch_counts.tp, batch_counts.fp, batch_counts.fn, batch_counts.tn)).unsqueeze(0))
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
                            ema_val_preds = _forward_model(
                                ema_model,
                                get_model_input(val_batch),
                                model_crop_size,
                                stitched=use_stitched_forward,
                                use_gradient_checkpointing=stitched_gradient_checkpointing,
                            )
                            
                        ema_val_loss_preds, _, _ = prepare_loss_inputs(ema_val_preds,val_batch)
                        ema_val_targets_with_ignore = build_deep_supervision_targets(val_targets, ema_val_loss_preds, mode='nearest')
                        ds_ignore = build_deep_supervision_targets(val_ignore_mask, ema_val_loss_preds, mode='nearest')
                        ema_val_targets_with_ignore = (
                            type(ema_val_targets_with_ignore)(
                                torch.cat([target_level, ignore_level], dim=1)
                                for target_level, ignore_level in zip(ema_val_targets_with_ignore, ds_ignore)
                            )
                            if isinstance(ema_val_targets_with_ignore, (list, tuple))
                            else torch.cat([ema_val_targets_with_ignore, ds_ignore], dim=1)
                        )
                        ema_val_l = loss(
                            type(ema_val_loss_preds)(v.float() for v in ema_val_loss_preds) if isinstance(ema_val_loss_preds, (list, tuple)) else ema_val_loss_preds.float(),
                            (
                                type(ema_val_targets_with_ignore)(v.float() for v in ema_val_targets_with_ignore)
                                if isinstance(ema_val_targets_with_ignore, (list, tuple))
                                else ema_val_targets_with_ignore.float()
                            ),
                        )
                        ema_val_loss_total = ema_val_loss_total + _distributed_mean_scalar(
                            accelerator,
                            ema_val_l,
                        )
                        ema_val_loss_batches = ema_val_loss_batches + 1.0
                        preview_preds = ema_val_loss_preds[0] if isinstance(ema_val_loss_preds, (list, tuple)) else ema_val_loss_preds
                        preview_volume_preds = ema_val_preds[0] if isinstance(ema_val_preds, (list, tuple)) else ema_val_preds

                    if val_batch_idx in preview_batch_indices:
                        val_preview_batch = val_batch
                        val_preview_preds = preview_preds.detach()
                        val_preview_targets = val_targets.detach()
                        val_preview_ignore = val_ignore_mask.detach()
                        val_preview_volume_logits = preview_volume_preds.detach() if normal_pooled_mode else None
                        if mode in _FULL_3D_TRAINING_MODES:
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

            mean_val_loss = float((val_loss_total / val_loss_batches.clamp_min(1.0)).item())
            mean_val_bce_unsmoothed = float(
                (val_bce_unsmoothed_total / val_loss_batches.clamp_min(1.0)).item()
            )
            mean_ema_val_loss = None
            if float(ema_val_loss_batches.item()) > 0.0:
                mean_ema_val_loss = float(
                    (ema_val_loss_total / ema_val_loss_batches.clamp_min(1.0)).item()
                )
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
                        'val/bce_unsmoothed': mean_val_bce_unsmoothed,
                        'val/balanced_accuracy': balanced_accuracy,
                        'val/tp': float(validation_counts.tp.item()),
                        'val/fp': float(validation_counts.fp.item()),
                        'val/fn': float(validation_counts.fn.item()),
                        'val/tn': float(validation_counts.tn.item()),
                    }
                )
                validation_record = {
                    'step': int(step),
                    'val_loss': float(mean_val_loss),
                    'val_bce_unsmoothed': float(mean_val_bce_unsmoothed),
                    'val_balanced_accuracy': float(balanced_accuracy),
                    'val_tp': float(validation_counts.tp.item()),
                    'val_fp': float(validation_counts.fp.item()),
                    'val_fn': float(validation_counts.fn.item()),
                    'val_tn': float(validation_counts.tn.item()),
                    'val_batches': int(num_val_batches),
                    'learning_rate': float(optimizer.param_groups[0]['lr']),
                }
                _append_jsonl(
                    os.path.join(out_dir, 'validation_metrics.jsonl'),
                    validation_record,
                )
                if best_checkpoint_metric is not None:
                    candidate = validation_record[best_checkpoint_metric]
                    improved = (
                        best_checkpoint_value is None
                        or (
                            best_checkpoint_metric == 'val_loss'
                            and candidate < best_checkpoint_value
                        )
                        or (
                            best_checkpoint_metric == 'val_balanced_accuracy'
                            and candidate > best_checkpoint_value
                        )
                    )
                    if improved:
                        best_checkpoint_value = float(candidate)
                        best_name = f'best_{best_checkpoint_metric}.pth'
                        save_checkpoint(
                            step,
                            force=True,
                            filename=best_name,
                            validation_metrics=validation_record,
                        )
                        with open(
                            os.path.join(out_dir, 'best_checkpoint.json'),
                            'w',
                            encoding='utf-8',
                        ) as stream:
                            json.dump(
                                {
                                    'checkpoint': best_name,
                                    'metric': best_checkpoint_metric,
                                    'value': best_checkpoint_value,
                                    'step': int(step),
                                },
                                stream,
                                indent=2,
                                sort_keys=True,
                            )
                            stream.write('\n')
                if wandb.run is not None:
                    wandb.log(log_dict, step=step)

        save_checkpoint(step)
        if fixed_batch_sampler is not None and (
            (step + 1) % sampling_audit_every == 0
            or step + 1 == config['num_iterations']
        ):
            write_fixed_sampling_audit(step + 1)

    accelerator.wait_for_everyone()
    if benchmark_enabled:
        if accelerator.device.type == 'cuda':
            torch.cuda.synchronize(accelerator.device)
        if benchmark_started_at is None:
            raise RuntimeError("benchmark timer never started")
        summary = _benchmark_summary(
            elapsed_seconds=time.perf_counter() - benchmark_started_at,
            measured_steps=benchmark_measured_steps,
            batch_size=config['batch_size'],
            world_size=accelerator.num_processes,
            data_wait_seconds=benchmark_data_wait_seconds,
            peak_allocated_bytes=(
                torch.cuda.max_memory_allocated(accelerator.device)
                if accelerator.device.type == 'cuda' else 0
            ),
            peak_reserved_bytes=(
                torch.cuda.max_memory_reserved(accelerator.device)
                if accelerator.device.type == 'cuda' else 0
            ),
        )
        summary.update({
            'warmup_steps': benchmark_warmup_steps,
            'device': str(accelerator.device),
        })
        if accelerator.is_main_process:
            output_path = benchmark_config.get(
                'output_path', os.path.join(out_dir, 'benchmark_summary.json')
            )
            with open(output_path, 'w', encoding='utf-8') as stream:
                json.dump(summary, stream, indent=2, sort_keys=True)
                stream.write('\n')
            accelerator.print(f"benchmark_summary={output_path}")

if __name__ == '__main__':
    train()
