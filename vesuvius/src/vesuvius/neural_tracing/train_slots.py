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
import torch.utils.checkpoint
from einops import rearrange
import torch.nn.functional as F

from vesuvius.neural_tracing.dataset import load_datasets
from vesuvius.neural_tracing.datasets.dataset_slotted import HeatmapDatasetSlotted
from vesuvius.models.training.loss.nnunet_losses import DeepSupervisionWrapper, DC_and_BCE_loss
from vesuvius.models.training.optimizers import create_optimizer
from vesuvius.models.training.lr_schedulers import get_scheduler
from vesuvius.neural_tracing.deep_supervision import _resize_for_ds, _compute_ds_weights
from vesuvius.neural_tracing.models import make_model
from vesuvius.models.training.loss.losses import CosineSimilarityLoss
from vesuvius.neural_tracing.visualization import make_canvas, print_training_config


def prepare_batch(batch, config):
    """Prepare batch tensors for slotted conditioning training."""
    use_localiser = bool(config.get('use_localiser', False))
    input_parts = [
        batch['volume'].unsqueeze(1),
        rearrange(batch['uv_heatmaps_in'], 'b z y x c -> b c z y x'),
    ]
    if use_localiser:
        input_parts.insert(1, batch['localiser'].unsqueeze(1))

    inputs = torch.cat(input_parts, dim=1)
    targets = rearrange(batch['uv_heatmaps_out'], 'b z y x c -> b c z y x')

    return inputs, targets


def make_loss_fn(config):
    """Create loss function based on config. Returns per-example losses."""
    binary = config.get('binary', False)

    # Create loss function once (not per call)
    dc_bce_loss_fn = DC_and_BCE_loss(
        bce_kwargs={},
        soft_dice_kwargs={'batch_dice': False, 'ddp': False},
        weight_ce=1.0,
        weight_dice=1.0
    ) if binary else None

    def loss_fn(target_pred, targets, mask):
        if binary:
            if mask is None:
                mask = torch.ones_like(targets)
            mask = mask.to(dtype=target_pred.dtype)
            targets_binary = (targets > 0.5).to(target_pred.dtype)

            # Use DC_and_BCE_loss per sample to get per-batch losses
            losses = torch.stack([
                dc_bce_loss_fn(
                    target_pred[i:i+1],
                    targets_binary[i:i+1],
                    loss_mask=mask[i:i+1]
                ) for i in range(target_pred.shape[0])
            ])
            return losses
        else:
            if mask is None:
                mask = torch.ones_like(targets)
            per_batch = ((target_pred - targets) ** 2 * mask).sum(dim=(1, 2, 3, 4)) / mask.sum(dim=(1, 2, 3, 4)).clamp_min(1e-6)
            return per_batch

    return loss_fn


def compute_slot_multistep_loss(model, inputs, targets, mask, config, loss_fn, known_out_mask=None):
    """
    Multistep training for masked-slot conditioning.

    At each step, supervise a subset of still-masked slots, then feed those predictions
    back into the conditioning channels for the next forward pass.

    Args:
        known_out_mask: Optional [B, C] bool tensor indicating which output channels are
            known (conditioning inputs). Used for reconstruction loss on first step.

    Returns:
        tuple: (total_loss, predictions_for_visualization)
    """
    multistep_count = int(config.get('multistep_count', 1))
    if multistep_count <= 1:
        raise ValueError("compute_slot_multistep_loss called with multistep_count <= 1")

    slots_per_step = max(1, int(config.get('slots_per_step', 1)))
    use_localiser = bool(config.get('use_localiser', False))

    # Slice inputs back into components so we can update conditioning between steps
    channel_idx = 0
    volume = inputs[:, channel_idx : channel_idx + 1]
    channel_idx += 1

    localiser = None
    if use_localiser:
        localiser = inputs[:, channel_idx : channel_idx + 1]
        channel_idx += 1

    slot_channels = targets.shape[1]
    cond_channels = inputs.shape[1] - channel_idx
    if cond_channels <= 0:
        raise ValueError("slot multistep expected at least one conditioning channel")
    uv_cond = inputs[:, channel_idx : channel_idx + cond_channels]

    # Channels with non-zero mask are the ones we should eventually supervise
    remaining_slots = (mask.flatten(2).sum(dim=2) > 0)
    if not remaining_slots.any():
        raise ValueError("slot multistep expects uv_heatmaps_out_mask to mark at least one slot")

    current_cond = uv_cond.clone()
    preds_for_vis = torch.zeros_like(targets)
    step_losses = []

    def make_step_inputs():
        parts = [volume]
        if use_localiser:
            parts.append(localiser)
        parts.append(current_cond)
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

        # Add known reconstruction loss on FIRST step only
        use_known_recon = config.get("use_known_recon", False)
        if step_idx == 0 and use_known_recon and known_out_mask is not None:
            known_recon_weight = float(config.get("known_recon_weight", 0.25))
            known_mask_cf = known_out_mask[:, :, None, None, None].float()
            known_mask_full = known_mask_cf.expand_as(targets)
            known_loss = loss_fn(step_pred, targets, known_mask_full).mean()
            step_loss = step_loss + known_recon_weight * known_loss

        step_losses.append(step_loss)

        # Accumulate predictions for visualisation and update conditioning for next step
        preds_for_vis = torch.where(step_selector_cf, step_pred, preds_for_vis)
        pred_heatmaps = torch.sigmoid(step_pred.detach())

        # Only update cardinal slots (0 to num_cardinals-1), never update diag_in
        # Output layout: [cardinals..., diag_out], Conditioning layout: [cardinals..., diag_in]
        # diag_out and diag_in are different spatial locations - don't mix them
        include_diag = config.get('masked_include_diag', True)
        num_cardinals = slot_channels - 1 if include_diag else slot_channels

        # Select only cardinal predictions to feed back
        cond_selector = step_selector[:, :num_cardinals]
        pred_for_cond = pred_heatmaps[:, :num_cardinals]

        # Pad to match cond_channels if needed (leaving diag_in unchanged)
        if cond_channels > num_cardinals:
            pad_size = cond_channels - num_cardinals
            cond_selector = F.pad(cond_selector, (0, pad_size), value=False)
            pred_for_cond = torch.cat([pred_for_cond, current_cond[:, num_cardinals:]], dim=1)

        cond_selector_cf = cond_selector[:, :, None, None, None]
        current_cond = torch.where(cond_selector_cf, pred_for_cond, current_cond)

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
    config.setdefault('use_localiser', True)
    config.setdefault('masked_include_diag', True)
    config.setdefault('step_count', 1)
    config.setdefault('slotted_allow_spatial_transforms', False)
    config.setdefault('flip_uv_directions', False)

    # Calculate channel counts based on step_count
    # Input: 4 cardinal + diag_in (5 conditioning slots)
    # Output: 4 cardinal + diag_out (5 output slots) - diag_in is never predicted
    cardinal_slots = 4 * config['step_count']
    if config['masked_include_diag']:
        conditioning_slots = cardinal_slots + 1  # diag_in only
        out_slots = cardinal_slots + 1  # diag_out only (diag_in is conditioning, not output)
    else:
        conditioning_slots = cardinal_slots
        out_slots = cardinal_slots
    # Conditioning channels = number of slots (direction encoding applied in dataset)
    config.setdefault('conditioning_channels', conditioning_slots)
    config.setdefault('out_channels', out_slots)

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
    grad_clip = config.setdefault('grad_clip', 5)

    # Deep supervision and auxiliary task settings (aligned with train.py)
    ds_enabled = config.setdefault('enable_deep_supervision', False)
    use_seg = config.setdefault('aux_segmentation', False)
    use_normals = config.setdefault('aux_normals', False)
    use_srf_overlap = config.setdefault('aux_srf_overlap', False)
    seg_loss_weight = config.setdefault('seg_loss_weight', 1.0)
    normals_loss_weight = config.setdefault('normals_loss_weight', 1.0)
    srf_overlap_loss_weight = config.setdefault('srf_overlap_loss_weight', 1.0)
    step_size = config['step_size']
    srf_overlap_loss_start_iter = config.setdefault('srf_overlap_loss_start_iter', 0)


    # Set random seeds
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])

    # Create loss functions
    loss_fn = make_loss_fn(config)
    normals_loss_fn = CosineSimilarityLoss(dim=1, eps=1e-8)
    ds_cache = {
        'uv': {'weights': None, 'loss_fn': None},
        'seg': {'weights': None, 'loss_fn': None},
        'normals': {'weights': None, 'loss_fn': None},
        'srf_overlap': {'weights': None, 'loss_fn': None},
    }

    # Setup accelerator
    accelerator = accelerate.Accelerator(
        mixed_precision=config.get('mixed_precision', 'no'),
        gradient_accumulation_steps=config.get('grad_acc_steps', 1),
    )

    # Initialize wandb
    if 'wandb_project' in config and accelerator.is_main_process:
        wandb.init(project=config['wandb_project'], entity=config.get('wandb_entity', None), config=config)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def make_generator(offset=0):
        gen = torch.Generator()
        gen.manual_seed(config['seed'] + accelerator.process_index * 1000 + offset)
        return gen

    # Create datasets
    train_patches, val_patches = load_datasets(config)
    multistep_count = int(config.get('multistep_count', 1))
    bidirectional = False  # Slots don't use directional conditioning
    train_dataset = HeatmapDatasetSlotted(config, train_patches, multistep_count, bidirectional)
    val_dataset = HeatmapDatasetSlotted(config, val_patches, multistep_count, bidirectional)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        num_workers=config.get('num_workers', 4),
        worker_init_fn=seed_worker,
        generator=make_generator(0),
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'] * 2,
        num_workers=1,
        worker_init_fn=seed_worker,
        generator=make_generator(1),
    )

    # Create model
    model = make_model(config)
    config.setdefault('compile_model', config.get('compile', True))
    if config['compile_model']:
        model = torch.compile(model)
        if accelerator.is_main_process:
            accelerator.print("Model compiled with torch.compile")

    # Setup scheduler and optimizer (aligned with train.py)
    scheduler_type = config.setdefault('scheduler', 'diffusers_cosine_warmup')
    scheduler_kwargs = dict(config.setdefault('scheduler_kwargs', {}) or {})
    scheduler_kwargs.setdefault('warmup_steps', config.setdefault('lr_warmup_steps', 1000))
    config['scheduler_kwargs'] = scheduler_kwargs
    total_scheduler_steps = config['num_iterations'] * accelerator.state.num_processes

    optimizer_config = config.setdefault('optimizer', 'adamw')
    # Handle optimizer being either a string or a dict
    if isinstance(optimizer_config, dict):
        optimizer_type = optimizer_config.get('name', 'adamw')
        optimizer_kwargs = dict(optimizer_config)
    else:
        optimizer_type = optimizer_config
        optimizer_kwargs = dict(config.setdefault('optimizer_kwargs', {}) or {})
    optimizer_kwargs.setdefault('learning_rate', config.setdefault('learning_rate', 1e-3))
    optimizer_kwargs.setdefault('weight_decay', config.setdefault('weight_decay', 1e-4))
    config['optimizer_kwargs'] = optimizer_kwargs
    optimizer = create_optimizer({'name': optimizer_type, **optimizer_kwargs}, model)

    lr_scheduler = get_scheduler(
        scheduler_type=scheduler_type,
        optimizer=optimizer,
        initial_lr=optimizer_kwargs['learning_rate'],
        max_steps=total_scheduler_steps,
        **scheduler_kwargs,
    )

    # Load checkpoint if specified
    if 'load_ckpt' in config:
        print(f'Loading checkpoint {config["load_ckpt"]}')
        ckpt = torch.load(config['load_ckpt'], map_location='cpu', weights_only=False)
        state_dict = ckpt['model']
        # Handle compiled model state dict (strip _orig_mod. prefix if needed)
        if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            # Check if current model is compiled
            model_keys = set(model.state_dict().keys())
            if not any(k.startswith('_orig_mod.') for k in model_keys):
                # Strip prefix from checkpoint
                state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
                print('Stripped _orig_mod. prefix from checkpoint state dict')
        # Filter out duplicate shared_decoder.encoder keys (old architecture had encoder copy in decoder)
        filtered_keys = [k for k in state_dict.keys() if 'shared_decoder.encoder.' in k]
        if filtered_keys:
            state_dict = {k: v for k, v in state_dict.items() if 'shared_decoder.encoder.' not in k}
            print(f'Filtered out {len(filtered_keys)} duplicate shared_decoder.encoder keys')
        model.load_state_dict(state_dict)
        # Only load optimizer state if optimizer type matches (avoid SGD/Adam mismatch)
        ckpt_optim_type = type(ckpt['optimizer']['param_groups'][0].get('betas', None))
        curr_optim_type = type(optimizer.param_groups[0].get('betas', None))
        if ckpt_optim_type == curr_optim_type:
            optimizer.load_state_dict(ckpt['optimizer'])
        else:
            print(f'Skipping optimizer state load (optimizer type changed)')

    # Prepare with accelerator
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )

    # Print configuration (using shared print_training_config from train.py)
    if accelerator.is_main_process:
        print_training_config(config, accelerator)
        # Additional slot-specific info
        accelerator.print("\n=== Slot-Specific Configuration ===")
        accelerator.print(f"Conditioning slots: {conditioning_slots}, Output slots: {out_slots}")
        accelerator.print(f"Multistep: {multistep_enabled} (count={config.get('multistep_count', 1)})")
        accelerator.print(f"Slots per step: {config.get('slots_per_step', 1)}")
        accelerator.print(f"Include diag: {config['masked_include_diag']}")
        accelerator.print("====================================\n")

    val_iterator = iter(val_dataloader)

    def require_head(outputs, name):
        if isinstance(outputs, dict) and name in outputs:
            return outputs[name]
        raise ValueError(f"aux_{name} is enabled but model did not return '{name}'")

    def compute_loss_with_ds(pred, target, mask, base_loss_fn, cache_key):
        # Wrap base_loss_fn to return scalar (mean over batch) for DS wrapper compatibility
        def mean_loss_fn(p, t, m):
            loss = base_loss_fn(p, t, m)
            return loss.mean() if loss.dim() > 0 else loss

        if ds_enabled and isinstance(pred, (list, tuple)):
            cache = ds_cache[cache_key]
            if cache['weights'] is None or len(cache['weights']) != len(pred):
                cache['weights'] = _compute_ds_weights(len(pred))
                cache['loss_fn'] = DeepSupervisionWrapper(mean_loss_fn, cache['weights'])
            elif cache['loss_fn'] is None:
                cache['loss_fn'] = DeepSupervisionWrapper(mean_loss_fn, cache['weights'])
            targets_resized = [_resize_for_ds(target, t.shape[2:], mode='trilinear', align_corners=False) for t in pred]
            masks_resized = None
            if mask is not None:
                masks_resized = [_resize_for_ds(mask, t.shape[2:], mode='nearest') for t in pred]
            loss = cache['loss_fn'](pred, targets_resized, masks_resized)
            pred_for_vis = pred[0]
        else:
            if isinstance(pred, (list, tuple)):
                pred = pred[0]
            loss = mean_loss_fn(pred, target, mask)
            pred_for_vis = pred
        return loss, pred_for_vis

    # Training loop
    progress_bar = tqdm(total=config['num_iterations'], disable=not accelerator.is_local_main_process)
    for iteration, batch in enumerate(train_dataloader):

        inputs, targets = prepare_batch(batch, config)
        if 'uv_heatmaps_out_mask' in batch:
            mask = rearrange(batch['uv_heatmaps_out_mask'], 'b z y x c -> b c z y x')
        else:
            mask = torch.ones_like(targets)

        if iteration == 0 and accelerator.is_main_process:
            accelerator.print("First batch input summary:")
            accelerator.print(f"  inputs: {tuple(inputs.shape)}")
            accelerator.print(f"  targets: {tuple(targets.shape)} | mask_present={'uv_heatmaps_out_mask' in batch}")

        wandb_log = {}
        seg_loss = normals_loss = None
        seg_for_vis = seg_pred_for_vis = normals_for_vis = normals_pred_for_vis = None
        srf_overlap_for_vis = srf_overlap_pred_for_vis = None

        with accelerator.accumulate(model):
            if multistep_enabled:
                known_out_mask = batch.get('known_out_mask')
                heatmap_loss, target_pred_for_vis = compute_slot_multistep_loss(
                    model, inputs, targets, mask, config, loss_fn, known_out_mask=known_out_mask
                )
                outputs = None  # Multistep doesn't return full outputs for aux losses yet
            else:
                outputs = model(inputs)
                target_pred = outputs['uv_heatmaps'] if isinstance(outputs, dict) else outputs
                heatmap_loss, target_pred_for_vis = compute_loss_with_ds(target_pred, targets, mask, loss_fn, 'uv')

            total_loss = heatmap_loss

            # Auxiliary segmentation loss
            if use_seg and outputs is not None:
                seg = batch.get('seg')
                if seg is not None:
                    seg = rearrange(seg, 'b z y x -> b 1 z y x') if seg.dim() == 4 else seg.unsqueeze(1)
                    seg_mask = (seg > 0).float()
                    seg_pred = require_head(outputs, 'seg')
                    seg_loss, seg_pred_for_vis = compute_loss_with_ds(
                        seg_pred, seg, seg_mask, loss_fn, 'seg'
                    )
                    total_loss = total_loss + seg_loss_weight * seg_loss
                    seg_for_vis = seg

            # Auxiliary normals loss
            if use_normals and outputs is not None:
                normals = batch.get('normals')
                if normals is not None:
                    normals = rearrange(normals, 'b z y x c -> b c z y x')
                    normals_mask = (normals.abs().sum(dim=1, keepdim=True) > 0).float()
                    normals_pred = require_head(outputs, 'normals')
                    normals_loss, normals_pred_for_vis = compute_loss_with_ds(
                        normals_pred, normals, normals_mask, normals_loss_fn, 'normals'
                    )
                    total_loss = total_loss + normals_loss_weight * normals_loss
                    normals_for_vis = normals

            # Surface overlap loss - model directly predicts surface overlap segmentation
            srf_overlap_loss = None
            if use_srf_overlap and outputs is not None:
                srf_overlap_mask = batch.get('srf_overlap_mask')
                if srf_overlap_mask is not None:
                    from vesuvius.neural_tracing.surf_overlap_loss import compute_surf_overlap_loss
                    srf_overlap_mask = srf_overlap_mask.unsqueeze(1)  # [B, 1, Z, Y, X]
                    # Get model's srf_overlap prediction
                    pred_srf_overlap = require_head(outputs, 'srf_overlap')
                    if isinstance(pred_srf_overlap, (list, tuple)):
                        pred_srf_overlap = pred_srf_overlap[0]
                    srf_overlap_loss = compute_surf_overlap_loss(pred_srf_overlap, srf_overlap_mask)
                    # Apply loss only after warmup iteration
                    if iteration >= srf_overlap_loss_start_iter:
                        total_loss = total_loss + srf_overlap_loss_weight * srf_overlap_loss
                    # Store for visualization
                    srf_overlap_for_vis = srf_overlap_mask
                    srf_overlap_pred_for_vis = pred_srf_overlap

            if torch.isnan(total_loss).any():
                raise ValueError('loss is NaN')
            accelerator.backward(total_loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        wandb_log['loss'] = total_loss.detach().item()
        wandb_log['first_step_heatmap_loss'] = heatmap_loss.detach().item()
        if use_seg and seg_loss is not None:
            wandb_log['seg_loss'] = (seg_loss_weight * seg_loss).detach().item()
        if use_normals and normals_loss is not None:
            wandb_log['normals_loss'] = (normals_loss_weight * normals_loss).detach().item()
        if use_srf_overlap and srf_overlap_loss is not None:
            wandb_log['srf_overlap_loss'] = (srf_overlap_loss_weight * srf_overlap_loss).detach().item()
        # Build tqdm postfix with all active losses
        postfix = {'total': f"{wandb_log['loss']:.3f}", 'hm': f"{wandb_log['first_step_heatmap_loss']:.3f}"}
        if use_seg and seg_loss is not None:
            postfix['seg'] = f"{wandb_log['seg_loss']:.3f}"
        if use_normals and normals_loss is not None:
            postfix['norm'] = f"{wandb_log['normals_loss']:.3f}"
        if use_srf_overlap and srf_overlap_loss is not None:
            postfix['srf'] = f"{wandb_log['srf_overlap_loss']:.3f}"
        progress_bar.set_postfix(postfix)
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

                val_seg_loss = val_normals_loss = None
                val_seg_for_vis = val_seg_pred_for_vis = val_normals_for_vis = val_normals_pred_for_vis = None
                val_srf_overlap_for_vis = val_srf_overlap_pred_for_vis = None

                if multistep_enabled:
                    val_known_out_mask = val_batch.get('known_out_mask')
                    val_heatmap_loss, val_target_pred_for_vis = compute_slot_multistep_loss(
                        model, val_inputs, val_targets, val_mask, config, loss_fn, known_out_mask=val_known_out_mask
                    )
                    val_outputs = None
                else:
                    val_outputs = model(val_inputs)
                    val_target_pred = val_outputs['uv_heatmaps'] if isinstance(val_outputs, dict) else val_outputs
                    val_heatmap_loss, val_target_pred_for_vis = compute_loss_with_ds(val_target_pred, val_targets, val_mask, loss_fn, 'uv')

                total_val_loss = val_heatmap_loss

                # Auxiliary segmentation loss for validation
                if use_seg and val_outputs is not None:
                    val_seg = val_batch.get('seg')
                    if val_seg is not None:
                        val_seg = rearrange(val_seg, 'b z y x -> b 1 z y x') if val_seg.dim() == 4 else val_seg.unsqueeze(1)
                        val_seg_mask = (val_seg > 0).float()
                        val_seg_pred = require_head(val_outputs, 'seg')
                        val_seg_loss, val_seg_pred_for_vis = compute_loss_with_ds(
                            val_seg_pred, val_seg, val_seg_mask, loss_fn, 'seg'
                        )
                        total_val_loss = total_val_loss + seg_loss_weight * val_seg_loss
                        val_seg_for_vis = val_seg

                # Auxiliary normals loss for validation
                if use_normals and val_outputs is not None:
                    val_normals = val_batch.get('normals')
                    if val_normals is not None:
                        val_normals = rearrange(val_normals, 'b z y x c -> b c z y x')
                        val_normals_mask = (val_normals.abs().sum(dim=1, keepdim=True) > 0).float()
                        val_normals_pred = require_head(val_outputs, 'normals')
                        val_normals_loss, val_normals_pred_for_vis = compute_loss_with_ds(
                            val_normals_pred, val_normals, val_normals_mask, normals_loss_fn, 'normals'
                        )
                        total_val_loss = total_val_loss + normals_loss_weight * val_normals_loss
                        val_normals_for_vis = val_normals

                # Validation surface overlap loss - model directly predicts surface overlap segmentation
                val_srf_overlap_loss = None
                if use_srf_overlap and val_outputs is not None:
                    val_srf_overlap_mask = val_batch.get('srf_overlap_mask')
                    if val_srf_overlap_mask is not None:
                        from vesuvius.neural_tracing.surf_overlap_loss import compute_surf_overlap_loss
                        val_srf_overlap_mask = val_srf_overlap_mask.unsqueeze(1)
                        # Get model's srf_overlap prediction
                        val_pred_srf_overlap = require_head(val_outputs, 'srf_overlap')
                        if isinstance(val_pred_srf_overlap, (list, tuple)):
                            val_pred_srf_overlap = val_pred_srf_overlap[0]
                        val_srf_overlap_loss = compute_surf_overlap_loss(val_pred_srf_overlap, val_srf_overlap_mask)
                        total_val_loss = total_val_loss + srf_overlap_loss_weight * val_srf_overlap_loss
                        # Store for visualization
                        val_srf_overlap_for_vis = val_srf_overlap_mask
                        val_srf_overlap_pred_for_vis = val_pred_srf_overlap

                wandb_log['val_loss'] = total_val_loss.item()
                wandb_log['val_first_step_heatmap_loss'] = val_heatmap_loss.item()
                if use_seg and val_seg_loss is not None:
                    wandb_log['val_seg_loss'] = (seg_loss_weight * val_seg_loss).item()
                if use_normals and val_normals_loss is not None:
                    wandb_log['val_normals_loss'] = (normals_loss_weight * val_normals_loss).item()
                if use_srf_overlap and val_srf_overlap_loss is not None:
                    wandb_log['val_srf_overlap_loss'] = (srf_overlap_loss_weight * val_srf_overlap_loss).item()

                # Create and save visualization
                cond_start = 2 if config.get('use_localiser', False) else 1
                log_image_ext = config.get('log_image_ext', 'jpg')
                train_img_path = f'{out_dir}/{iteration:06}_train.{log_image_ext}'
                val_img_path = f'{out_dir}/{iteration:06}_val.{log_image_ext}'
                train_unknown_mask = batch.get('uv_heatmaps_unknown_mask', mask)
                val_unknown_mask = val_batch.get('uv_heatmaps_unknown_mask', val_mask)
                make_canvas(inputs, targets, target_pred_for_vis, config,
                           seg=seg_for_vis, seg_pred=seg_pred_for_vis,
                           normals=normals_for_vis, normals_pred=normals_pred_for_vis, normals_mask=None,
                           srf_overlap=srf_overlap_for_vis, srf_overlap_pred=srf_overlap_pred_for_vis,
                           cond_channel_start=cond_start,
                           save_path=train_img_path,
                           unknown_mask=train_unknown_mask)
                make_canvas(val_inputs, val_targets, val_target_pred_for_vis, config,
                           seg=val_seg_for_vis, seg_pred=val_seg_pred_for_vis,
                           normals=val_normals_for_vis, normals_pred=val_normals_pred_for_vis, normals_mask=None,
                           srf_overlap=val_srf_overlap_for_vis, srf_overlap_pred=val_srf_overlap_pred_for_vis,
                           cond_channel_start=cond_start,
                           save_path=val_img_path,
                           unknown_mask=val_unknown_mask)

                if wandb.run is not None:
                    wandb_log['train_image'] = wandb.Image(train_img_path)
                    wandb_log['val_image'] = wandb.Image(val_img_path)

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
