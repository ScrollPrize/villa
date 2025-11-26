"""
Training script for slot-based masked conditioning neural tracing.

This is a simplified training script specifically for the slotted conditioning
variant, which uses fixed slots + masking instead of u/v direction conditioning.
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
from einops import rearrange
import torch.nn.functional as F

from vesuvius.neural_tracing.dataset import load_datasets
from vesuvius.neural_tracing.datasets.dataset_slotted import HeatmapDatasetSlotted
from vesuvius.models.training.loss.nnunet_losses import MemoryEfficientSoftDiceLoss
from vesuvius.models.training.optimizers import create_optimizer
from vesuvius.models.training.lr_schedulers import get_scheduler
from vesuvius.neural_tracing.models import make_model
from vesuvius.neural_tracing.cropping import recrop
from vesuvius.neural_tracing.visualization import make_canvas


def prepare_batch(batch, config, recrop_center=None, recrop_size=None):
    """Prepare batch tensors for slotted conditioning training."""
    if recrop_size is None:
        recrop_size = config.get('crop_size')
    if recrop_center is None:
        recrop_center = torch.tensor(batch['volume'].shape[-3:]) // 2

    uv_heatmaps_out_mask = batch.get('uv_heatmaps_out_mask')
    uv_heatmaps_out_mask_cf = None
    if uv_heatmaps_out_mask is not None:
        uv_heatmaps_out_mask_cf = rearrange(uv_heatmaps_out_mask, 'b z y x c -> b c z y x')

    condition_mask = batch.get('condition_mask')
    condition_mask_channels = []
    if condition_mask is not None and config.get("include_condition_mask_channel", True):
        condition_mask_channels.append(rearrange(condition_mask, 'b z y x c -> b c z y x'))

    use_localiser = bool(config.get('use_localiser', False))
    input_parts = [
        batch['volume'].unsqueeze(1),
        rearrange(batch['uv_heatmaps_in'], 'b z y x c -> b c z y x'),
        *condition_mask_channels,
    ]
    if use_localiser:
        input_parts.insert(1, batch['localiser'].unsqueeze(1))

    inputs = torch.cat(input_parts, dim=1)
    targets = rearrange(batch['uv_heatmaps_out'], 'b z y x c -> b c z y x')

    if recrop_size is not None:
        inputs = recrop(inputs, recrop_center, recrop_size)
        targets = recrop(targets, recrop_center, recrop_size)
        if uv_heatmaps_out_mask_cf is not None:
            uv_heatmaps_out_mask_cf = recrop(uv_heatmaps_out_mask_cf, recrop_center, recrop_size)

    if uv_heatmaps_out_mask_cf is not None:
        batch['uv_heatmaps_out_mask'] = rearrange(uv_heatmaps_out_mask_cf, 'b c z y x -> b z y x c')

    return inputs, targets


def make_loss_fn(config):
    """Create loss function based on config. Returns per-example losses."""
    binary = config.get('binary', False)

    def loss_fn(target_pred, targets, mask):
        if binary:
            targets_binary = (targets > 0.5).long()
            bce = torch.nn.BCEWithLogitsLoss(reduction='none')(target_pred, targets_binary.float()).mean(dim=(1, 2, 3, 4))
            dice_loss_fn = MemoryEfficientSoftDiceLoss(apply_nonlin=torch.sigmoid, batch_dice=False, ddp=False)
            dice = torch.stack([
                dice_loss_fn(target_pred[i:i+1], targets_binary[i:i+1]) for i in range(target_pred.shape[0])
            ])
            return bce + dice
        else:
            if mask is None:
                mask = torch.ones_like(targets)
            per_batch = ((target_pred - targets) ** 2 * mask).sum(dim=(1, 2, 3, 4)) / mask.sum(dim=(1, 2, 3, 4))
            return per_batch

    return loss_fn


def compute_slot_multistep_loss(model, inputs, targets, mask, config, loss_fn):
    """
    Multistep training for masked-slot conditioning.

    At each step, supervise a subset of still-masked slots, then feed those predictions
    back into the conditioning channels for the next forward pass.

    Returns:
        tuple: (total_loss, predictions_for_visualization)
    """
    multistep_count = int(config.get('multistep_count', 1))
    if multistep_count <= 1:
        raise ValueError("compute_slot_multistep_loss called with multistep_count <= 1")

    slots_per_step = max(1, int(config.get('slots_per_step', 1)))
    use_localiser = bool(config.get('use_localiser', False))
    include_condition_mask = bool(config.get('include_condition_mask_channel', True))

    # Slice inputs back into components so we can update conditioning between steps
    channel_idx = 0
    volume = inputs[:, channel_idx : channel_idx + 1]
    channel_idx += 1

    localiser = None
    if use_localiser:
        localiser = inputs[:, channel_idx : channel_idx + 1]
        channel_idx += 1

    slot_channels = targets.shape[1]
    uv_cond = inputs[:, channel_idx : channel_idx + slot_channels]
    channel_idx += slot_channels

    cond_mask = None
    if include_condition_mask:
        cond_mask = inputs[:, channel_idx : channel_idx + slot_channels]

    # Channels with non-zero mask are the ones we should eventually supervise
    remaining_slots = (mask.flatten(2).sum(dim=2) > 0)
    if not remaining_slots.any():
        raise ValueError("slot multistep expects uv_heatmaps_out_mask to mark at least one slot")

    current_cond = uv_cond.clone()
    current_cond_mask = cond_mask.clone() if cond_mask is not None else None
    preds_for_vis = torch.zeros_like(targets)
    step_losses = []

    def make_step_inputs():
        parts = [volume]
        if use_localiser:
            parts.append(localiser)
        parts.append(current_cond)
        if include_condition_mask and current_cond_mask is not None:
            parts.append(current_cond_mask)
        return torch.cat(parts, dim=1)

    for step_idx in range(multistep_count):
        step_selector = torch.zeros_like(remaining_slots)
        for b in range(remaining_slots.shape[0]):
            available = torch.nonzero(remaining_slots[b], as_tuple=False).flatten()
            if available.numel() == 0:
                continue
            if step_idx == multistep_count - 1 or available.numel() <= slots_per_step:
                chosen = available
            else:
                chosen = available[torch.randperm(available.numel(), device=available.device)[:slots_per_step]]
            step_selector[b, chosen] = True
            remaining_slots[b, chosen] = False

        if not step_selector.any():
            break

        step_selector_cf = step_selector[:, :, None, None, None]
        step_mask = mask * step_selector_cf

        step_inputs = make_step_inputs()
        outputs = model(step_inputs)
        step_pred = outputs.get('uv_heatmaps', outputs) if isinstance(outputs, dict) else outputs
        if step_pred.shape[1] != slot_channels:
            raise ValueError(f"slot multistep expected {slot_channels} channels, got {step_pred.shape[1]}")

        step_loss = loss_fn(step_pred, targets, step_mask).mean()
        step_losses.append(step_loss)

        # Accumulate predictions for visualisation and update conditioning for next step
        preds_for_vis = torch.where(step_selector_cf, step_pred, preds_for_vis)
        pred_heatmaps = torch.sigmoid(step_pred.detach())
        current_cond = torch.where(step_selector_cf, pred_heatmaps, current_cond)
        if current_cond_mask is not None:
            current_cond_mask = torch.where(step_selector_cf, torch.ones_like(current_cond_mask), current_cond_mask)

    if not step_losses:
        raise ValueError("slot multistep did not select any slots to supervise")

    total_loss = torch.stack(step_losses).mean()
    return total_loss, preds_for_vis


@click.command()
@click.argument('config_path', type=click.Path(exists=True))
def train(config_path):
    """Train a slot-based masked conditioning model."""

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Force slotted variant settings
    config['dataset_variant'] = 'slotted'
    config['masked_conditioning'] = True
    config.setdefault('use_localiser', False)
    config.setdefault('masked_include_diag', True)
    config.setdefault('include_condition_mask_channel', True)
    config.setdefault('step_count', 1)

    # Calculate slot count based on step_count
    slot_count = 4 * config['step_count']
    if config['masked_include_diag']:
        slot_count += 1
    config.setdefault('conditioning_channels', slot_count * 2 if config['include_condition_mask_channel'] else slot_count)
    config.setdefault('out_channels', slot_count)

    # Multistep settings
    config.setdefault('multistep_count', 1)
    config.setdefault('slots_per_step', 1)
    multistep_enabled = int(config.get('multistep_count', 1)) > 1

    # Training settings
    config.setdefault('num_iterations', 250000)
    config.setdefault('log_frequency', 100)
    config.setdefault('ckpt_frequency', 5000)
    config.setdefault('grad_clip', 5)

    out_dir = config['out_dir']
    os.makedirs(out_dir, exist_ok=True)
    grad_clip = int(config['grad_clip'])

    # Set random seeds
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])

    # Create loss function
    loss_fn = make_loss_fn(config)

    # Setup accelerator
    accelerator = accelerate.Accelerator(
        mixed_precision=config.get('mixed_precision', 'no'),
        gradient_accumulation_steps=config.get('grad_acc_steps', 1),
    )

    # Initialize wandb
    if 'wandb_project' in config and accelerator.is_main_process:
        wandb.init(project=config['wandb_project'], entity=config.get('wandb_entity', None), config=config)

    # Create datasets
    train_patches, val_patches = load_datasets(config)
    train_dataset = HeatmapDatasetSlotted(config, train_patches)
    val_dataset = HeatmapDatasetSlotted(config, val_patches)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config['batch_size'], num_workers=config.get('num_workers', 4)
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config['batch_size'] * 2, num_workers=1
    )

    # Create model
    model = make_model(config)
    config.setdefault('compile_model', config.get('compile', True))
    if config['compile_model']:
        model = torch.compile(model)
        if accelerator.is_main_process:
            accelerator.print("Model compiled with torch.compile")

    # Setup optimizer
    optimizer_config = config.get('optimizer') or {}
    if isinstance(optimizer_config, str):
        optimizer_config = {'name': optimizer_config}
    config['optimizer'] = {
        'name': 'adamw',
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        **optimizer_config
    }
    optimizer = create_optimizer(optimizer_config, model)

    # Setup scheduler
    scheduler_type = config.setdefault('scheduler', 'diffusers_cosine_warmup')
    scheduler_kwargs = dict(config.get('scheduler_kwargs', {}) or {})
    scheduler_kwargs.setdefault('warmup_steps', config.get('lr_warmup_steps', 1000))
    config['scheduler_kwargs'] = scheduler_kwargs
    config.setdefault('lr_warmup_steps', scheduler_kwargs['warmup_steps'])
    total_scheduler_steps = config['num_iterations'] * accelerator.state.num_processes
    lr_scheduler = get_scheduler(
        scheduler_type=scheduler_type,
        optimizer=optimizer,
        initial_lr=optimizer_config.get('learning_rate', config.get('learning_rate', 1e-3)),
        max_steps=total_scheduler_steps,
        **scheduler_kwargs,
    )

    # Load checkpoint if specified
    if 'load_ckpt' in config:
        print(f'Loading checkpoint {config["load_ckpt"]}')
        ckpt = torch.load(config['load_ckpt'], map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])

    # Prepare with accelerator
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )

    # Print configuration
    if accelerator.is_main_process:
        accelerator.print("\n=== Slot Training Configuration ===")
        accelerator.print(f"Slot count: {slot_count}")
        accelerator.print(f"Multistep: {multistep_enabled} (count={config.get('multistep_count', 1)})")
        accelerator.print(f"Slots per step: {config.get('slots_per_step', 1)}")
        accelerator.print(f"Include diag: {config['masked_include_diag']}")
        accelerator.print(f"Include condition mask channel: {config['include_condition_mask_channel']}")
        accelerator.print(f"Optimizer: {optimizer_config.get('name', 'adamw')}")
        accelerator.print(f"Scheduler: {scheduler_type}")
        accelerator.print(f"Initial LR: {optimizer_config.get('learning_rate', config.get('learning_rate', 1e-3))}")
        accelerator.print(f"Grad Clip: {grad_clip}")
        accelerator.print(f"Binary: {config.get('binary', False)}")
        accelerator.print("====================================\n")

    val_iterator = iter(val_dataloader)

    # Training loop
    progress_bar = tqdm(total=config['num_iterations'], disable=not accelerator.is_local_main_process)
    for iteration, batch in enumerate(train_dataloader):

        inputs, targets = prepare_batch(batch, config)
        if 'uv_heatmaps_out_mask' in batch:
            mask = rearrange(batch['uv_heatmaps_out_mask'], 'b z y x c -> b c z y x')
        else:
            mask = torch.ones_like(targets)

        if iteration == 0 and accelerator.is_main_process:
            cond_mask = batch.get('condition_mask')
            cond_mask_channels = cond_mask.shape[-1] if (cond_mask is not None and config.get("include_condition_mask_channel", True)) else 0
            accelerator.print("First batch input summary:")
            accelerator.print(f"  inputs: {tuple(inputs.shape)}")
            accelerator.print(f"  targets: {tuple(targets.shape)} | mask_present={'uv_heatmaps_out_mask' in batch}")

        wandb_log = {}
        with accelerator.accumulate(model):
            if multistep_enabled:
                total_loss, target_pred_for_vis = compute_slot_multistep_loss(
                    model, inputs, targets, mask, config, loss_fn
                )
            else:
                outputs = model(inputs)
                target_pred = outputs['uv_heatmaps'] if isinstance(outputs, dict) else outputs
                total_loss = loss_fn(target_pred, targets, mask).mean()
                target_pred_for_vis = target_pred

            if torch.isnan(total_loss):
                raise ValueError('loss is NaN')
            accelerator.backward(total_loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        wandb_log['loss'] = total_loss.detach().item()
        progress_bar.set_postfix({'loss': wandb_log['loss']})
        progress_bar.update(1)

        # Validation and logging
        if iteration % config['log_frequency'] == 0:
            with torch.no_grad():
                model.eval()

                val_batch = next(val_iterator)
                val_inputs, val_targets = prepare_batch(val_batch, config)
                if 'uv_heatmaps_out_mask' in val_batch:
                    val_mask = rearrange(val_batch['uv_heatmaps_out_mask'], 'b z y x c -> b c z y x')
                else:
                    val_mask = torch.ones_like(val_targets)

                if multistep_enabled:
                    total_val_loss, val_target_pred_for_vis = compute_slot_multistep_loss(
                        model, val_inputs, val_targets, val_mask, config, loss_fn
                    )
                else:
                    val_outputs = model(val_inputs)
                    val_target_pred = val_outputs['uv_heatmaps'] if isinstance(val_outputs, dict) else val_outputs
                    total_val_loss = loss_fn(val_target_pred, val_targets, val_mask).mean()
                    val_target_pred_for_vis = val_target_pred

                wandb_log['val_loss'] = total_val_loss.item()

                # Create and save visualization
                cond_start = 2 if config.get('use_localiser', False) else 1
                log_image_ext = config.get('log_image_ext', 'jpg')
                make_canvas(inputs, targets, target_pred_for_vis, config,
                           cond_channel_start=cond_start,
                           save_path=f'{out_dir}/{iteration:06}_train.{log_image_ext}')
                make_canvas(val_inputs, val_targets, val_target_pred_for_vis, config,
                           cond_channel_start=cond_start,
                           save_path=f'{out_dir}/{iteration:06}_val.{log_image_ext}')

                model.train()

        # Save checkpoint
        if iteration % config['ckpt_frequency'] == 0:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
                'step': iteration,
            }, f'{out_dir}/ckpt_{iteration:06}.pth')

        if wandb.run is not None:
            wandb.log(wandb_log)

        if iteration == config['num_iterations']:
            break


if __name__ == '__main__':
    train()
