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
from tqdm import tqdm

from vesuvius.neural_tracing.datasets.dataset_rowcol_cond import EdtSegDataset
from vesuvius.neural_tracing.datasets.dataset_defaults import (
    setdefault_rowcol_cond_dataset_config,
    validate_rowcol_cond_dataset_config,
)
from vesuvius.neural_tracing.datasets.growth_direction import (
    growth_direction_channel_count,
)
from vesuvius.neural_tracing.datasets.targets import RowColTargets
from vesuvius.neural_tracing.loss.trace_losses import compute_trace_losses
from vesuvius.models.training.optimizers import create_optimizer
from vesuvius.models.training.lr_schedulers import get_scheduler
from vesuvius.neural_tracing.nets.models import make_model
from vesuvius.neural_tracing.trainers.loss_config import TraceLossConfig
from vesuvius.neural_tracing.trainers.rowcol_cond_visualization import (
    make_dense_visualization,
)

import multiprocessing


def seed_worker(worker_id):
    """Seed worker for reproducibility."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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
    # Source masks are used by on-device EDT target generation. Keeping EDT
    # out of dataloader workers avoids cupyx defaulting every worker to GPU 0.
    return {
        'vol': torch.stack([b['vol'] for b in batch]),
        'cond': torch.stack([b['cond'] for b in batch]),
        'cond_direction': [b['cond_direction'] for b in batch],
        'velocity_dir': torch.stack([b['velocity_dir'] for b in batch]),
        'velocity_loss_weight': torch.stack([b['velocity_loss_weight'] for b in batch]),
        'trace_loss_weight': torch.stack([b['trace_loss_weight'] for b in batch]),
        'cond_gt': torch.stack([b['cond_gt'] for b in batch]),
        'masked_seg': torch.stack([b['masked_seg'] for b in batch]),
        'neighbor_seg': torch.stack([b['neighbor_seg'] for b in batch]),
    }


def prepare_batch(batch, config):
    """Prepare batch tensors for training."""
    return RowColTargets.from_batch(batch, config)


@click.command()
@click.argument('config_path', type=click.Path(exists=True))
def train(config_path):
    """Train a row/col conditioned trace-ODE model."""

    with open(config_path, 'r') as f:
        config = json.load(f)

    setdefault_rowcol_cond_dataset_config(config)

    validate_rowcol_cond_dataset_config(config)

    config['in_channels'] = 2 + growth_direction_channel_count()
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
    config.setdefault('wandb_resume', False)
    config.setdefault('wandb_resume_mode', 'allow')
    config.setdefault('compile_model', True)
    config.setdefault('separate_eager_eval_for_logging', True)

    config['targets'] = {
        'velocity_dir': {'out_channels': 3, 'activation': 'none'},
        'surface_attract': {'out_channels': 3, 'activation': 'none'},
        'trace_validity': {'out_channels': 1, 'activation': 'none'},
    }

    out_dir = config['out_dir']
    os.makedirs(out_dir, exist_ok=True)

    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])

    dataloader_config = accelerate.DataLoaderConfiguration(
        non_blocking=bool(config.get('non_blocking', True))
    )

    accelerator = accelerate.Accelerator(
        mixed_precision=config.get('mixed_precision', 'no'),
        gradient_accumulation_steps=config.get('grad_acc_steps', 1),
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

    loss_config = TraceLossConfig.from_config(config)

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
        accelerator.print("Growth direction channels: True")
        accelerator.print("Output: velocity_dir (3ch) + surface_attract (3ch) + trace_validity (1ch)")
        accelerator.print(
            f"Velocity direction loss: lambda={loss_config.lambda_velocity_dir}, "
            f"dilation={config.get('trace_target_dilation_radius')}"
        )
        if loss_config.lambda_velocity_smooth > 0.0:
            accelerator.print(
                f"Velocity smoothness loss: lambda={loss_config.lambda_velocity_smooth}, "
                f"normalize={loss_config.velocity_smooth_normalize}"
            )
        if loss_config.lambda_trace_integration > 0.0:
            accelerator.print(
                f"Trace integration loss: lambda={loss_config.lambda_trace_integration}, "
                f"steps={loss_config.trace_integration_steps}, "
                f"step_size={loss_config.trace_integration_step_size}, "
                f"max_points={loss_config.trace_integration_max_points}, "
                f"detach_steps={loss_config.trace_integration_detach_steps}"
            )
        accelerator.print(
            f"Trace ODE losses: lambda_attract={loss_config.lambda_surface_attract}, "
            f"lambda_validity={loss_config.lambda_trace_validity}, "
            f"dilation={config.get('trace_target_dilation_radius')}, "
            f"attract_mode=trace_band, "
            f"attract_radius={config.get('trace_surface_attract_radius')}"
        )
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

    progress_bar = tqdm(
        total=config['num_iterations'],
        initial=start_iteration,
        disable=not accelerator.is_local_main_process
    )

    for iteration in range(start_iteration, config['num_iterations']):
        if config['verbose']:
            accelerator.print(f"starting iteration {iteration}")
        should_log_this_iteration = (
            (iteration > 0 or config.get('log_at_step_zero', False))
            and iteration % config['log_frequency'] == 0
        )
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_dataloader)
            batch = next(train_iterator)

        if config['verbose']:
            accelerator.print(f"got batch, keys: {batch.keys()}")

        prepared = prepare_batch(batch, config)
        inputs = prepared.inputs
        velocity_dir_target = prepared.velocity_dir
        velocity_loss_weight = prepared.velocity_loss_weight
        trace_loss_weight = prepared.trace_loss_weight
        trace_validity_target = prepared.trace_validity
        trace_validity_weight = prepared.trace_validity_weight
        surface_attract_target = prepared.surface_attract
        surface_attract_weight = prepared.surface_attract_weight

        wandb_log = {}

        with accelerator.accumulate(model):
            # Forward pass
            output = model(inputs)
            grad_norm = None
            total_loss, loss_metrics = compute_trace_losses(
                output,
                prepared,
                loss_config,
                random_trace_sample=True,
            )
            wandb_log.update({
                key: value.detach().item()
                for key, value in loss_metrics.items()
            })

            if torch.isnan(total_loss).any():
                raise ValueError('loss is NaN')

            do_optimizer_step = True
            accelerator.backward(total_loss)
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

        wandb_log['loss'] = total_loss.detach().item()
        wandb_log['current_lr'] = optimizer.param_groups[0]['lr']
        if grad_norm is not None:
            wandb_log['grad_norm'] = float(grad_norm.detach().item() if torch.is_tensor(grad_norm) else grad_norm)

        postfix = {
            'loss': f"{wandb_log['loss']:.4f}",
        }
        if loss_config.lambda_velocity_dir > 0.0:
            postfix['vel_dir'] = f"{wandb_log['velocity_dir_loss']:.4f}"
        if loss_config.lambda_velocity_smooth > 0.0:
            postfix['vel_smooth'] = f"{wandb_log['velocity_smooth_loss']:.4f}"
        if loss_config.lambda_trace_integration > 0.0:
            postfix['trace_int'] = f"{wandb_log['trace_integration_loss']:.4f}"
        if loss_config.lambda_surface_attract > 0.0:
            postfix['attract'] = f"{wandb_log['surface_attract_loss']:.4f}"
        if loss_config.lambda_trace_validity > 0.0:
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
                if loss_config.lambda_velocity_dir > 0.0:
                    val_metric_sums['val_velocity_dir_loss'] = 0.0
                if loss_config.lambda_velocity_smooth > 0.0:
                    val_metric_sums['val_velocity_smooth_loss'] = 0.0
                if loss_config.lambda_trace_integration > 0.0:
                    val_metric_sums['val_trace_integration_loss'] = 0.0
                if loss_config.lambda_surface_attract > 0.0:
                    val_metric_sums['val_surface_attract_loss'] = 0.0
                if loss_config.lambda_trace_validity > 0.0:
                    val_metric_sums['val_trace_validity_loss'] = 0.0

                first_val_vis = None
                for val_batch_idx in range(val_batches_per_log):
                    try:
                        val_batch = next(val_iterator)
                    except StopIteration:
                        val_iterator = iter(val_dataloader)
                        val_batch = next(val_iterator)

                    val_prepared = prepare_batch(val_batch, config)

                    with accelerator.autocast():
                        val_output = eval_forward_model(val_prepared.inputs)
                    val_total_loss, val_metrics = compute_trace_losses(
                        val_output,
                        val_prepared,
                        loss_config,
                        random_trace_sample=False,
                    )
                    for key, value in val_metrics.items():
                        val_metric_sums[f'val_{key}'] += value.item()

                    val_metric_sums['val_loss'] += val_total_loss.item()

                    if val_batch_idx == 0:
                        first_val_vis = {
                            'inputs': val_prepared.inputs,
                            'velocity_dir_pred': val_output.get('velocity_dir'),
                            'velocity_dir_target': val_prepared.velocity_dir,
                            'velocity_loss_weight': val_prepared.velocity_loss_weight,
                            'trace_loss_weight': val_prepared.trace_loss_weight,
                            'trace_validity_pred': val_output.get('trace_validity'),
                            'trace_validity_target': val_prepared.trace_validity,
                            'trace_validity_weight': val_prepared.trace_validity_weight,
                            'surface_attract_pred': val_output.get('surface_attract'),
                            'surface_attract_target': val_prepared.surface_attract,
                            'surface_attract_weight': val_prepared.surface_attract_weight,
                        }

                for key, value in val_metric_sums.items():
                    wandb_log[key] = value / val_batches_per_log

                # Create visualization
                train_img_path = f'{out_dir}/{iteration:06}_train.png'
                val_img_path = f'{out_dir}/{iteration:06}_val.png'

                train_velocity_dir_pred = output.get('velocity_dir')
                train_trace_validity_pred = output.get('trace_validity')
                train_surface_attract_pred = output.get('surface_attract')
                if first_val_vis is not None:
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
