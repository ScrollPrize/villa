#!/usr/bin/env python
"""
Trainer for row/col conditioned trace-ODE target prediction.

Trains velocity, trace validity, and trace-band surface attraction heads from
conditioned surfaces.
"""
import os
import json
import click
import torch
import wandb
import random
import accelerate
import numpy as np
from tqdm import tqdm

from vesuvius.neural_tracing.datasets.dataset_rowcol_cond import EdtSegDataset
from vesuvius.neural_tracing.datasets.rowcol_cond_config import (
    prepare_rowcol_cond_train_config,
    resolve_rowcol_cond_optimizer_config,
    resolve_rowcol_cond_scheduler_config,
)
from vesuvius.neural_tracing.datasets.targets import RowColTargets
from vesuvius.neural_tracing.loss.trace_losses import compute_trace_losses
from vesuvius.models.training.optimizers import create_optimizer
from vesuvius.models.training.lr_schedulers import get_scheduler
from vesuvius.neural_tracing.nets.models import (
    checkpoint_wandb_run_id,
    load_checkpoint,
    load_checkpoint_payload,
    make_model,
    strip_state,
)
from vesuvius.neural_tracing.trainers.loss_config import TraceLossConfig
from vesuvius.neural_tracing.trainers.rowcol_cond_visualization import (
    make_trace_visualization,
)


def seed_worker(worker_id):
    """Seed worker for reproducibility."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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


def checkpoint_model_state_dict(accelerator, model):
    """Return a wrapper-free model state dict for portable checkpoints."""
    unwrapped_model = accelerator.unwrap_model(model)
    return strip_state(unwrapped_model.state_dict())


@click.command()
@click.argument('config_path', type=click.Path(exists=True))
def train(config_path):
    """Train a row/col conditioned trace-ODE model."""

    with open(config_path, 'r') as f:
        config = json.load(f)

    prepare_rowcol_cond_train_config(config)

    out_dir = config['out_dir']
    os.makedirs(out_dir, exist_ok=True)

    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])

    dataloader_config = accelerate.DataLoaderConfiguration(
        non_blocking=bool(config['non_blocking'])
    )

    accelerator = accelerate.Accelerator(
        mixed_precision=config['mixed_precision'],
        gradient_accumulation_steps=config['grad_acc_steps'],
        dataloader_config=dataloader_config,
    )

    preloaded_ckpt = None
    wandb_resume = bool(config['wandb_resume'])
    wandb_run_id = config.get('wandb_run_id')
    if wandb_resume and wandb_run_id is None and 'load_ckpt' in config:
        preloaded_ckpt = load_checkpoint_payload(config['load_ckpt'], map_location='cpu', weights_only=False)
        wandb_run_id = checkpoint_wandb_run_id(preloaded_ckpt)
        if wandb_run_id is not None:
            config['wandb_run_id'] = wandb_run_id

    if 'wandb_project' in config and accelerator.is_main_process:
        wandb_kwargs = {
            'project': config['wandb_project'],
            'entity': config.get('wandb_entity', None),
            'config': config,
        }
        if wandb_resume:
            wandb_kwargs['resume'] = config['wandb_resume_mode']
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
    if config['force_recompute_patches']:
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
    num_val = max(1, int(num_patches * config['val_fraction']))
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

    train_num_workers = max(0, int(config['num_workers']))
    val_num_workers = max(0, int(config['val_num_workers']))
    pin_memory = bool(config['pin_memory'])
    train_prefetch_factor = max(1, int(config['prefetch_factor']))
    val_prefetch_factor = max(1, int(config['val_prefetch_factor']))
    train_persistent_workers = bool(config['persistent_workers']) and train_num_workers > 0
    val_persistent_workers = bool(config['persistent_workers']) and val_num_workers > 0

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
        train_dataloader_kwargs['multiprocessing_context'] = 'spawn'
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
        val_dataloader_kwargs['multiprocessing_context'] = 'spawn'
    val_dataloader = torch.utils.data.DataLoader(val_dataset, **val_dataloader_kwargs)

    model = make_model(config)

    if config['compile_model']:
        model = torch.compile(model)
        if accelerator.is_main_process:
            accelerator.print("Model compiled with torch.compile")

    scheduler_type, scheduler_kwargs = resolve_rowcol_cond_scheduler_config(config)
    optimizer_type, optimizer_kwargs = resolve_rowcol_cond_optimizer_config(config)
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
        model, _, ckpt, _ = load_checkpoint(
            config['load_ckpt'],
            model=model,
            checkpoint=preloaded_ckpt,
            allow_partial_weight_load=config['allow_partial_weight_load'],
            map_location='cpu',
            weights_only=False,
            print_fn=accelerator.print,
            return_checkpoint=True,
        )

        if not config['load_weights_only']:
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

    # Keep the scheduler out of accelerator.prepare. Accelerate wraps prepared
    # schedulers and, with sharded dataloaders, advances them once per process.
    # This trainer defines num_iterations as optimizer-update iterations, so the
    # LR schedule should advance once per real optimizer step.
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )

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
            (iteration > 0 or config['log_at_step_zero'])
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
                if accelerator.sync_gradients and not getattr(optimizer, "step_was_skipped", False):
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
                model.eval()

                val_batches_per_log = max(1, int(config['val_batches_per_log']))
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
                        val_output = model(val_prepared.inputs)
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

                train_vis = {
                    'inputs': inputs,
                    'velocity_dir_pred': output.get('velocity_dir'),
                    'velocity_dir_target': velocity_dir_target,
                    'velocity_loss_weight': velocity_loss_weight,
                    'trace_loss_weight': trace_loss_weight,
                    'trace_validity_pred': output.get('trace_validity'),
                    'trace_validity_target': trace_validity_target,
                    'trace_validity_weight': trace_validity_weight,
                    'surface_attract_pred': output.get('surface_attract'),
                    'surface_attract_target': surface_attract_target,
                    'surface_attract_weight': surface_attract_weight,
                }
                if first_val_vis is not None:
                    make_trace_visualization(train_vis, train_img_path)
                    make_trace_visualization(first_val_vis, val_img_path)

                    if wandb.run is not None:
                        wandb_log['train_image'] = wandb.Image(train_img_path)
                        wandb_log['val_image'] = wandb.Image(val_img_path)

                model.train()

        if (
            (iteration > 0 or config['ckpt_at_step_zero'])
            and iteration % config['ckpt_frequency'] == 0
            and accelerator.is_main_process
        ):
            torch.save({
                'model': checkpoint_model_state_dict(accelerator, model),
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
            'model': checkpoint_model_state_dict(accelerator, model),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'config': config,
            'step': config['num_iterations'],
            'wandb_run_id': wandb.run.id if wandb.run is not None else config.get('wandb_run_id'),
        }, f'{out_dir}/ckpt_final.pth')


if __name__ == '__main__':
    train()
