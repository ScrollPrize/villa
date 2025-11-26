
import os
import math
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
import matplotlib.pyplot as plt
import torch.nn.functional as F

from vesuvius.neural_tracing.dataset import HeatmapDatasetV2, HeatmapDatasetV2Masked, load_datasets, make_heatmaps
from vesuvius.neural_tracing.datasets.PatchInCubeDataset import PatchInCubeDataset
from vesuvius.models.training.loss.nnunet_losses import DeepSupervisionWrapper, MemoryEfficientSoftDiceLoss
from vesuvius.models.training.optimizers import create_optimizer
from vesuvius.models.training.lr_schedulers import get_scheduler
from vesuvius.neural_tracing.deep_supervision import _resize_for_ds, _compute_ds_weights
from vesuvius.neural_tracing.models import make_model
from vesuvius.neural_tracing.cropping import (safe_crop_with_padding,
                                              transform_to_first_crop_space,
                                              recrop)
from vesuvius.models.training.loss.losses import CosineSimilarityLoss



def prepare_batch(batch, config, recrop_center=None, recrop_size=None):
    if recrop_size is None:
        recrop_size = config.get('crop_size')
    if recrop_center is None:
        recrop_center = torch.tensor(batch['volume'].shape[-3:]) // 2

    uv_heatmaps_out_mask = batch.get('uv_heatmaps_out_mask')
    uv_heatmaps_out_mask_full = uv_heatmaps_out_mask  # keep original (channel-last) for later multi-step crops
    uv_heatmaps_out_mask_cf = None
    if uv_heatmaps_out_mask is not None:
        uv_heatmaps_out_mask_cf = rearrange(uv_heatmaps_out_mask, 'b z y x c -> b c z y x')

    condition_mask = batch.get('condition_mask')
    condition_mask_channels = []
    if condition_mask is not None and config.get("include_condition_mask_channel", False):
        condition_mask_channels.append(rearrange(condition_mask, 'b z y x c -> b c z y x'))

    use_localiser = bool(config.get('use_localiser', True))
    input_parts = [
        batch['volume'].unsqueeze(1),
        rearrange(batch['uv_heatmaps_in'], 'b z y x c -> b c z y x'),
        *condition_mask_channels,
    ]
    if use_localiser:
        input_parts.insert(1, batch['localiser'].unsqueeze(1))

    inputs = torch.cat(input_parts, dim=1)
    targets = rearrange(batch['uv_heatmaps_out'], 'b z y x c -> b c z y x')

    use_seg = config.get("aux_segmentation", False)
    use_normals = config.get("aux_normals", False)

    def require_from_batch(key):
        value = batch.get(key)
        if value is None:
            raise ValueError(f"Batch missing '{key}' while auxiliary head is enabled")
        return value

    seg = seg_mask = None
    normals = normals_mask = None
    if use_seg:
        seg = require_from_batch('seg').unsqueeze(1)  # [B,1,D,H,W]
        seg_mask = require_from_batch('seg_mask').unsqueeze(1)
    if use_normals:
        normals = rearrange(require_from_batch('normals'), 'b z y x c -> b c z y x')  # [B,3,D,H,W]
        normals_mask = require_from_batch('normals_mask').unsqueeze(1)

    if recrop_size is not None:
        inputs = recrop(inputs, recrop_center, recrop_size)
        targets = recrop(targets, recrop_center, recrop_size)
        if uv_heatmaps_out_mask_cf is not None:
            uv_heatmaps_out_mask_cf = recrop(uv_heatmaps_out_mask_cf, recrop_center, recrop_size)
        if use_seg:
            seg = recrop(seg, recrop_center, recrop_size)
            seg_mask = recrop(seg_mask, recrop_center, recrop_size)
        if use_normals:
            normals = recrop(normals, recrop_center, recrop_size)
            normals_mask = recrop(normals_mask, recrop_center, recrop_size)

    if uv_heatmaps_out_mask_cf is not None:
        # Retain the full mask alongside the recropped version for multi-step supervision.
        batch['uv_heatmaps_out_mask'] = rearrange(uv_heatmaps_out_mask_cf, 'b c z y x -> b z y x c')
        batch['uv_heatmaps_out_mask_full'] = uv_heatmaps_out_mask_full

    return inputs, targets, seg, seg_mask, normals, normals_mask

@click.command()
@click.argument('config_path', type=click.Path(exists=True))
def train(config_path):

    with open(config_path, 'r') as f:
        config = json.load(f)

    multistep_mode = str(config.get('multistep_mode', 'chain')).lower()
    if multistep_mode == 'chain':
        config.setdefault('multistep_count', 1)  # default to single-step; multistep is opt-in
        config.setdefault('multistep_prob', 1.0)
        config.setdefault('multistep_samples', 8)

    config.setdefault('num_iterations', 250000)
    log_image_max_samples = config.setdefault('log_image_max_samples', 4)
    log_image_grid_cols = config.setdefault('log_image_grid_cols', 2)
    log_image_ext = config.setdefault('log_image_ext', 'jpg')
    log_image_quality = config.setdefault('log_image_quality', 80)

    out_dir = config['out_dir']
    os.makedirs(out_dir, exist_ok=True)
    ds_enabled = bool(config.get('enable_deep_supervision', False))
    multistep_enabled = int(config.get('multistep_count', 1)) > 1
    multistep_mode = str(config.get('multistep_mode', 'chain')).lower()
    chain_multistep = multistep_enabled and multistep_mode == 'chain'
    slot_multistep = multistep_enabled and multistep_mode == 'slots'
    if multistep_mode not in ('chain', 'slots'):
        raise ValueError("multistep_mode must be 'chain' or 'slots'")
    seg_loss_weight = float(config.get("seg_loss_weight", 1.0))
    normals_loss_weight = float(config.get("normals_loss_weight", 1.0))
    use_seg = config.get("aux_segmentation", False)
    use_normals = config.get("aux_normals", False)
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])

    normals_loss_fn = CosineSimilarityLoss(dim=1, eps=1e-8)
    ds_cache = {
        'uv': {'weights': None, 'loss_fn': None},
        'seg': {'weights': None, 'loss_fn': None},
        'normals': {'weights': None, 'loss_fn': None},
    }

    config.setdefault('grad_clip', 5)
    grad_clip = int(config['grad_clip'])

    accelerator = accelerate.Accelerator(
        mixed_precision=config['mixed_precision'],
        gradient_accumulation_steps=config['grad_acc_steps'] if 'grad_acc_steps' in config else 1,
    )

    if 'wandb_project' in config and accelerator.is_main_process:
        wandb.init(project=config['wandb_project'], entity=config.get('wandb_entity', None), config=config)

    if config['representation'] == 'heatmap':
        train_patches, val_patches = load_datasets(config)
        dataset_variant = config.get('dataset_variant', 'standard')
        if dataset_variant == 'masked' or config.get('masked_conditioning', False):
            config.setdefault('use_localiser', False)
        else:
            config.setdefault('use_localiser', True)
        if dataset_variant == 'masked' or config.get('masked_conditioning', False):
            slot_count = 4 * int(config.get('step_count', 1))
            if config.get('masked_include_diag', False):
                slot_count += 1
            conditioning_channels = slot_count * 2 if config.get('include_condition_mask_channel', False) else slot_count
            config.setdefault('conditioning_channels', conditioning_channels)
            config.setdefault('out_channels', slot_count)
        dataset_cls = HeatmapDatasetV2Masked if dataset_variant == 'masked' or config.get('masked_conditioning', False) else HeatmapDatasetV2
        train_dataset = dataset_cls(config, train_patches)
        val_dataset = dataset_cls(config, val_patches)
    else:
        train_dataset = val_dataset = PatchInCubeDataset(config)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], num_workers=config['num_workers'])
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=config['batch_size'] * 2, num_workers=1)

    # FIXME: need separate data-loaders for multi-step and single-step training, since have different target shapes

    model = make_model(config)
    config.setdefault('compile_model', config.get('compile', True))
    compile_enabled = config['compile_model']
    if compile_enabled:
        try:
            model = torch.compile(model)
            if accelerator.is_main_process:
                accelerator.print("Model compiled with torch.compile")
        except Exception as e:
            if accelerator.is_main_process:
                accelerator.print(f"torch.compile failed ({e}); continuing without compilation")

    optimizer_config = config.get('optimizer') or {}
    if isinstance(optimizer_config, str):
        optimizer_config = {'name': optimizer_config}
    else:
        optimizer_config = dict(optimizer_config)
    optimizer_config.setdefault('name', config.get('optimizer_name', 'adamw'))
    optimizer_config.setdefault('learning_rate', config.get('learning_rate', 1e-3))
    optimizer_config.setdefault('weight_decay', config.get('weight_decay', 1e-4))
    config['optimizer'] = optimizer_config

    optimizer = create_optimizer(optimizer_config, model)

    scheduler_type = config.setdefault('scheduler', 'diffusers_cosine_warmup')
    scheduler_kwargs = dict(config.get('scheduler_kwargs', {}) or {})
    scheduler_kwargs.setdefault('warmup_steps', config.get('lr_warmup_steps', 1000))
    config['scheduler_kwargs'] = scheduler_kwargs
    config.setdefault('lr_warmup_steps', scheduler_kwargs['warmup_steps'])
    total_scheduler_steps = config['num_iterations'] * accelerator.state.num_processes  # See comment below on accelerator.prepare
    # FIXME: accelerator.prepare wraps schedulers so that they step once per process; multiply steps to compensate
    lr_scheduler = get_scheduler(
        scheduler_type=scheduler_type,
        optimizer=optimizer,
        initial_lr=optimizer_config.get('learning_rate', config.get('learning_rate', 1e-3)),
        max_steps=total_scheduler_steps,
        **scheduler_kwargs,
    )

    if 'load_ckpt' in config:
        print(f'loading checkpoint {config["load_ckpt"]}')
        ckpt = torch.load(config['load_ckpt'], map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        # Note we don't load the lr_scheduler state (i.e. training starts 'hot'), nor any config

    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )

    if accelerator.is_main_process:
        accelerator.print("\n=== Training Configuration ===")
        accelerator.print(f"Optimizer: {optimizer_config.get('name', 'adamw')}")
        accelerator.print(f"Scheduler: {scheduler_type}")
        accelerator.print(f"Initial LR: {optimizer_config.get('learning_rate', config.get('learning_rate', 1e-3))}")
        accelerator.print(f"Weight Decay: {optimizer_config.get('weight_decay', config.get('weight_decay', 1e-4))}")
        accelerator.print(f"Grad Clip: {grad_clip}")
        accelerator.print(f"Deep Supervision: {ds_enabled}")
        accelerator.print(f"Binary: {config.get('binary', False)}")
        accelerator.print("")
        accelerator.print("Point Perturbation:")
        pp = config.get('point_perturbation', {})
        if pp:
            accelerator.print(f"  perturb_probability: {pp.get('perturb_probability', 'not set')}")
            accelerator.print(f"  uv_max_perturbation: {pp.get('uv_max_perturbation', 'not set')}")
            accelerator.print(f"  w_max_perturbation: {pp.get('w_max_perturbation', 'not set')}")
            accelerator.print(f"  main_component_distance_factor: {pp.get('main_component_distance_factor', 'not set')}")
        else:
            accelerator.print("  (not configured)")
        accelerator.print("")
        accelerator.print("Step Settings:")
        accelerator.print(f"  step_size: {config.get('step_size', 'not set')}")
        accelerator.print(f"  step_count: {config.get('step_count', 1)}")
        accelerator.print("==============================\n")

    val_iterator = iter(val_dataloader)

    def loss_fn(target_pred, targets, mask):
        if config['binary']:
            targets_binary = (targets > 0.5).long()  # FIXME: should instead not do the gaussian conv in data-loader!
            from vesuvius.models.training.loss.nnunet_losses import DC_and_BCE_loss
            mask_for_loss = mask
            if mask_for_loss is None:
                mask_for_loss = torch.ones_like(targets_binary, dtype=targets_binary.dtype, device=targets_binary.device)
            # Use the loss' native masking so masked voxels are excluded from both BCE and Dice.
            return DC_and_BCE_loss(
                bce_kwargs={'reduction': 'none'},
                soft_dice_kwargs={'ddp': False}
            )(target_pred, targets_binary, mask_for_loss)
        else:
            if mask is None:
                mask = torch.ones_like(targets)
            bce = F.binary_cross_entropy_with_logits(
                target_pred, targets, reduction='none'
            )
            return (bce * mask).sum() / (mask.sum() + 1e-8)

    def loss_fn_per_example(target_pred, targets, mask):
        """Per-sample variant for multistep training (no deep supervision)."""
        if config['binary']:
            targets_binary = (targets > 0.5).long()
            bce = F.binary_cross_entropy_with_logits(
                target_pred, targets_binary.float(), reduction='none'
            ).mean(dim=(1, 2, 3, 4))
            dice_loss_fn = MemoryEfficientSoftDiceLoss(apply_nonlin=torch.sigmoid, batch_dice=False, ddp=False)
            dice = torch.stack([
                dice_loss_fn(target_pred[i:i+1], targets_binary[i:i+1]) for i in range(target_pred.shape[0])
            ])
            return bce + dice
        else:
            if mask is None:
                mask = torch.ones_like(targets)
            mask_sum = mask.flatten(1).sum(dim=1)
            bce = F.binary_cross_entropy_with_logits(
                target_pred, targets, reduction='none'
            ).flatten(1)
            per_batch = (bce * mask.flatten(1)).sum(dim=1) / (mask_sum + 1e-8)
            return per_batch

    def compute_multistep_loss_and_pred(model, inputs, targets, batch, config):
        """Multistep sampling + importance-weighted loss; returns scalar loss and stacked preds for viz."""
        multistep_count = int(config.get('multistep_count', 1))
        if multistep_count <= 1:
            raise ValueError("compute_multistep_loss_and_pred called with multistep_count <= 1")
        sample_count = int(config.get('multistep_samples', 1))
        use_localiser = bool(config.get('use_localiser', True))

        def unwrap_uv_pred(pred):
            if isinstance(pred, dict):
                pred = pred.get('uv_heatmaps', pred)
            if isinstance(pred, (list, tuple)):
                pred = pred[0]
            return pred

        first_step_mask = None
        if 'uv_heatmaps_out_mask' in batch:
            first_step_mask = rearrange(batch['uv_heatmaps_out_mask'], 'b z y x c -> b c z y x')
        step_mask_source = batch.get('uv_heatmaps_out_mask_full', batch.get('uv_heatmaps_out_mask'))

        outputs = model(inputs)
        target_pred = unwrap_uv_pred(outputs)

        if targets.shape[1] <= target_pred.shape[1]:
            raise ValueError("multistep training expects targets to have more channels than model outputs")

        first_step_targets = targets[:, ::multistep_count]
        if first_step_mask is not None:
            first_step_mask = first_step_mask[:, ::multistep_count]
        else:
            first_step_mask = torch.ones_like(first_step_targets)
        first_step_loss = loss_fn_per_example(target_pred, first_step_targets, first_step_mask)

        # Direction to extend (assumes exactly one of prev_u/prev_v is set).
        step_directions = batch['uv_heatmaps_in'].amax(dim=[1, 2, 3])[:, :2].argmax(dim=-1)

        first_step_cube_radius = 4
        later_step_cube_radius = 2  # smaller radius for later steps to reduce variance

        def sample_for_next_step(step_pred_for_dir, num_samples, cube_radius):
            cube_center = torch.argmax(step_pred_for_dir.view(step_pred_for_dir.shape[0], -1), dim=1)
            cube_center = torch.stack(torch.unravel_index(cube_center, step_pred_for_dir.shape[1:]), dim=-1)  # batch, zyx
            cube_center = torch.clamp(
                cube_center,
                torch.tensor(cube_radius, device=cube_center.device),
                torch.tensor(step_pred_for_dir.shape[1:], device=cube_center.device) - cube_radius - 1
            )

            sample_zyxs_in_subcrop = torch.randint(
                -cube_radius, cube_radius + 1, [1, num_samples, 3], device=cube_center.device
            ) + cube_center[:, None, :]
            cube_volume = (2 * cube_radius + 1) ** 3
            sample_logits = step_pred_for_dir[
                torch.arange(step_pred_for_dir.shape[0], device=step_pred_for_dir.device)[:, None].expand(-1, num_samples),
                sample_zyxs_in_subcrop[..., 0],
                sample_zyxs_in_subcrop[..., 1],
                sample_zyxs_in_subcrop[..., 2]
            ]

            temperature = 20.0
            sample_unnormalised_probs = torch.sigmoid(sample_logits / temperature)
            proposal_probs = torch.full_like(sample_unnormalised_probs, 1.0 / cube_volume)

            return sample_zyxs_in_subcrop, sample_unnormalised_probs, proposal_probs

        first_step_pred_for_dir = target_pred[torch.arange(target_pred.shape[0]), step_directions]
        first_sample_zyxs_in_first_subcrop, first_step_sample_unnormalised_probs, first_step_proposal_probs = sample_for_next_step(
            first_step_pred_for_dir, num_samples=sample_count, cube_radius=first_step_cube_radius)

        first_step_pred_vis = torch.full_like(target_pred, -100.0)
        first_step_pred_vis[torch.arange(target_pred.shape[0]), step_directions] = target_pred.detach()[torch.arange(target_pred.shape[0]), step_directions]

        outer_crop_shape = torch.tensor(batch['volume'].shape[-3:], device=first_sample_zyxs_in_first_subcrop.device)
        outer_crop_center = outer_crop_shape // 2
        first_sample_zyxs_in_outer_crop = first_sample_zyxs_in_first_subcrop + (outer_crop_shape - config['crop_size']) // 2

        losses_by_sample_by_later_step = []
        step_unnormalised_probs_by_sample_by_later_step = []
        step_proposal_probs_by_sample_by_later_step = []
        all_step_preds_vis = [first_step_pred_vis]

        for sample_idx in range(sample_count):
            current_center_in_outer_crop = first_sample_zyxs_in_outer_crop[:, sample_idx, :]
            prev_center_in_outer_crop = outer_crop_center.expand(current_center_in_outer_crop.shape[0], -1)

            sample_losses = []
            sample_step_unnormalised_probs = [first_step_sample_unnormalised_probs[:, sample_idx]]
            sample_step_proposal_probs = [first_step_proposal_probs[:, sample_idx]]

            for step_idx in range(1, multistep_count):
                min_corner_new_subcrop_in_outer = current_center_in_outer_crop - config['crop_size'] // 2

                prev_heatmap = torch.cat([
                    make_heatmaps([prev_center_in_outer_crop[iib:iib+1]], min_corner_new_subcrop_in_outer[iib], config['crop_size'])
                    for iib in range(min_corner_new_subcrop_in_outer.shape[0])
                ], dim=0).to(prev_center_in_outer_crop.device)
                prev_uv = torch.zeros([prev_heatmap.shape[0], 2, *prev_heatmap.shape[1:]], device=prev_heatmap.device, dtype=prev_heatmap.dtype)
                prev_uv[torch.arange(prev_heatmap.shape[0]), step_directions] = prev_heatmap

                # Prepare the volume (sub-)crop and localiser. For the localiser we take the original since
                # we still want the center of it (conceptually we create a new localiser at the new center)
                step_volume_crop = safe_crop_with_padding(
                    batch['volume'],
                    min_corner_new_subcrop_in_outer,
                    config['crop_size']
                )
                # FIXME: this makes assumptions about how prepare_batch arranges stuff; can we unify?
                step_input_parts = [step_volume_crop.unsqueeze(1)]
                if use_localiser:
                    step_input_parts.append(inputs[:, 1:2])  # borrow the original localiser subcrop
                step_input_parts.extend([
                    prev_uv,  # prev_u, prev_v
                    torch.zeros_like(prev_uv[:, :1]),  # prev_diag
                ])
                step_inputs = torch.cat(step_input_parts, dim=1)

                # Inputs are pure data (no requires_grad), so use the non-reentrant
                # variant to keep gradients flowing through model parameters.
                step_pred = torch.utils.checkpoint.checkpoint(model, step_inputs, use_reentrant=False)
                step_pred = unwrap_uv_pred(step_pred)

                step_targets = safe_crop_with_padding(
                    batch['uv_heatmaps_out'],
                    min_corner_new_subcrop_in_outer,
                    config['crop_size']
                )
                step_targets = rearrange(step_targets[..., step_idx::multistep_count], 'b z y x c -> b c z y x')

                if step_mask_source is not None:
                    step_mask = safe_crop_with_padding(
                        step_mask_source,
                        min_corner_new_subcrop_in_outer,
                        config['crop_size']
                    )
                    step_mask = rearrange(step_mask[..., step_idx::multistep_count], 'b z y x c -> b c z y x')
                else:
                    step_mask = torch.ones_like(step_targets)

                if sample_idx == 0:
                    step_pred_filtered = torch.full_like(step_pred, -100.0)
                    step_pred_filtered[torch.arange(step_pred.shape[0]), step_directions] = step_pred.detach()[torch.arange(step_pred.shape[0]), step_directions]
                    first_step_crop_min = outer_crop_center - config['crop_size'] // 2
                    offset = min_corner_new_subcrop_in_outer - first_step_crop_min
                    step_pred_in_first_crop = transform_to_first_crop_space(step_pred_filtered, offset, config['crop_size'])
                    all_step_preds_vis.append(step_pred_in_first_crop)

                step_pred = step_pred[torch.arange(step_pred.shape[0]), step_directions].unsqueeze(1)
                step_targets = step_targets[torch.arange(step_targets.shape[0]), step_directions].unsqueeze(1)
                step_mask = step_mask[torch.arange(step_mask.shape[0]), step_directions].unsqueeze(1)

                step_loss = loss_fn_per_example(step_pred, step_targets, step_mask)
                sample_losses.append(step_loss)

                if step_idx < multistep_count - 1:
                    step_pred_for_dir = step_pred.squeeze(1)
                    sample_zyxs_in_subcrop, sample_unnormalised_probs, proposal_probs = sample_for_next_step(
                        step_pred_for_dir, num_samples=1, cube_radius=later_step_cube_radius)
                    sample_step_unnormalised_probs.append(sample_unnormalised_probs.squeeze(1))
                    sample_step_proposal_probs.append(proposal_probs.squeeze(1))
                    prev_center_in_outer_crop = current_center_in_outer_crop
                    current_center_in_outer_crop = sample_zyxs_in_subcrop.squeeze(1) + min_corner_new_subcrop_in_outer

            losses_by_sample_by_later_step.append(sample_losses)
            step_unnormalised_probs_by_sample_by_later_step.append(sample_step_unnormalised_probs)
            step_proposal_probs_by_sample_by_later_step.append(sample_step_proposal_probs)

        # First step loss is added directly without importance weighting
        loss = first_step_loss

        # Later step losses are accumulated and weighted using self-normalized importance sampling
        # later_step_idx is the index into losses_by_sample_by_later_step (which excludes first step)
        cumulative_weights = [1] * sample_count
        for later_step_idx in range(multistep_count - 1):

            # Update cumulative importance weights
            for sample_idx in range(sample_count):
                target_prob = step_unnormalised_probs_by_sample_by_later_step[sample_idx][later_step_idx]
                proposal_prob = step_proposal_probs_by_sample_by_later_step[sample_idx][later_step_idx]
                cumulative_weights[sample_idx] = cumulative_weights[sample_idx] * target_prob / (proposal_prob + 1e-8)
            cumulative_weights_stacked = torch.stack(cumulative_weights, dim=0)  # sample_count, batch_size

            step_losses = torch.stack(
                [losses_by_sample_by_later_step[sample_idx][later_step_idx] for sample_idx in range(sample_count)],
                dim=0
            )
            weighted_loss_sum = (step_losses * cumulative_weights_stacked).sum(dim=0)
            total_weights_sum = cumulative_weights_stacked.sum(dim=0)
            loss = loss + (weighted_loss_sum / (total_weights_sum + 1e-8))

        loss = loss.mean()

        target_pred_all_steps = rearrange(torch.stack(all_step_preds_vis, dim=2), 'b uv s z y x -> b (uv s) z y x')

        return loss, target_pred_all_steps

    def compute_slot_multistep_loss_and_pred(model, inputs, targets, mask, batch, config):
        """
        Multistep training for masked-slot conditioning.
        At each step, supervise a subset of still-masked slots, then feed those predictions back
        into the conditioning channels (and condition mask, if present) for the next forward.
        """
        multistep_count = int(config.get('multistep_count', 1))
        if multistep_count <= 1:
            raise ValueError("compute_slot_multistep_loss_and_pred called with multistep_count <= 1")
        slots_per_step = max(1, int(config.get('slots_per_step', 1)))
        use_localiser = bool(config.get('use_localiser', True))
        include_condition_mask = bool(config.get('include_condition_mask_channel', False))

        # Slice inputs back into components so we can update conditioning between steps.
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

        # Channels with non-zero mask are the ones we should eventually supervise.
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

            step_loss = loss_fn_per_example(step_pred, targets, step_mask).mean()
            step_losses.append(step_loss)

            # Accumulate predictions for visualisation and update conditioning for next step.
            preds_for_vis = torch.where(step_selector_cf, step_pred, preds_for_vis)
            pred_heatmaps = torch.sigmoid(step_pred.detach())
            current_cond = torch.where(step_selector_cf, pred_heatmaps, current_cond)
            if current_cond_mask is not None:
                current_cond_mask = torch.where(step_selector_cf, torch.ones_like(current_cond_mask), current_cond_mask)

        if not step_losses:
            raise ValueError("slot multistep did not select any slots to supervise")

        total_loss = torch.stack(step_losses).mean()
        return total_loss, preds_for_vis

    def require_aux(value, name):
        if value is None:
            raise ValueError(f"aux_{name} is enabled but batch is missing '{name}'")
        return value

    def require_head(outputs, name):
        if isinstance(outputs, dict) and name in outputs:
            return outputs[name]
        raise ValueError(f"aux_{name} is enabled but model did not return '{name}'")

    def compute_loss_with_ds(pred, target, mask, base_loss_fn, cache_key):
        pred_for_vis = pred
        if ds_enabled and isinstance(pred, (list, tuple)):
            cache = ds_cache[cache_key]
            if cache['weights'] is None or len(cache['weights']) != len(pred):
                cache['weights'] = _compute_ds_weights(len(pred))
                cache['loss_fn'] = DeepSupervisionWrapper(base_loss_fn, cache['weights'])
            elif cache['loss_fn'] is None:
                cache['loss_fn'] = DeepSupervisionWrapper(base_loss_fn, cache['weights'])
            targets_resized = [_resize_for_ds(target, t.shape[2:], mode='trilinear', align_corners=False) for t in pred]
            masks_resized = None
            if mask is not None:
                masks_resized = [_resize_for_ds(mask, t.shape[2:], mode='nearest') for t in pred]
            loss = cache['loss_fn'](pred, targets_resized, masks_resized)
            pred_for_vis = pred[0]
        else:
            if isinstance(pred, (list, tuple)):
                pred = pred[0]
            loss = base_loss_fn(pred, target, mask)
            pred_for_vis = pred
        return loss, pred_for_vis

    progress_bar = tqdm(total=config['num_iterations'], disable=not accelerator.is_local_main_process)
    for iteration, batch in enumerate(train_dataloader):

        inputs, targets, seg, seg_mask, normals, normals_mask = prepare_batch(batch, config)
        if 'uv_heatmaps_out_mask' in batch:
            mask = rearrange(batch['uv_heatmaps_out_mask'], 'b z y x c -> b c z y x')
        else:
            mask = torch.ones_like(targets)

        if iteration == 0 and accelerator.is_main_process:
            cond_mask = batch.get('condition_mask')
            cond_mask_channels = cond_mask.shape[-1] if (cond_mask is not None and config.get("include_condition_mask_channel", False)) else 0
            accelerator.print("First batch input summary:")
            accelerator.print(f"  inputs: {tuple(inputs.shape)} | channels: volume=1, localiser={'1' if config.get('use_localiser', True) else '0'}, "
                              f"uv_heatmaps_in={batch['uv_heatmaps_in'].shape[-1]}, condition_mask={cond_mask_channels}")
            accelerator.print(f"  targets: {tuple(targets.shape)} | mask_present={'uv_heatmaps_out_mask' in batch}")
            if seg is not None or normals is not None:
                accelerator.print(f"  aux: seg={'yes' if seg is not None else 'no'}, normals={'yes' if normals is not None else 'no'}")

        wandb_log = {}
        target_pred_for_vis = None
        with accelerator.accumulate(model):
            if chain_multistep:
                if ds_enabled or use_seg or use_normals:
                    raise ValueError("chain multistep currently supports heatmaps only; disable deep supervision and auxiliary heads.")
                total_loss, target_pred_for_vis = compute_multistep_loss_and_pred(
                    model, inputs, targets, batch, config
                )
                seg_pred_for_vis = None
                normals_pred_for_vis = None
            elif slot_multistep:
                if ds_enabled or use_seg or use_normals:
                    raise ValueError("slot multistep currently supports heatmaps only; disable deep supervision and auxiliary heads.")
                total_loss, target_pred_for_vis = compute_slot_multistep_loss_and_pred(
                    model, inputs, targets, mask, batch, config
                )
                seg_pred_for_vis = None
                normals_pred_for_vis = None
            else:
                outputs = model(inputs)
                target_pred = outputs['uv_heatmaps'] if isinstance(outputs, dict) else outputs
                heatmap_loss, target_pred_for_vis = compute_loss_with_ds(
                    target_pred, targets, mask, loss_fn, 'uv'
                )
                total_loss = heatmap_loss
                seg_pred_for_vis = None
                normals_pred_for_vis = None

                if use_seg:
                    seg = require_aux(seg, 'seg')
                    seg_mask_for_loss = seg_mask if seg_mask is not None else torch.ones_like(seg)
                    seg_pred = require_head(outputs, 'seg')
                    seg_loss, seg_pred_for_vis = compute_loss_with_ds(
                        seg_pred, seg, seg_mask_for_loss, loss_fn, 'seg'
                    )
                    total_loss = total_loss + seg_loss_weight * seg_loss
                if use_normals:
                    normals = require_aux(normals, 'normals')
                    normals_pred = require_head(outputs, 'normals')
                    normals_loss, normals_pred_for_vis = compute_loss_with_ds(
                        normals_pred, normals, normals_mask, normals_loss_fn, 'normals'
                    )
                    total_loss = total_loss + normals_loss_weight * normals_loss

            if torch.isnan(total_loss):
                raise ValueError('loss is NaN')
            accelerator.backward(total_loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        wandb_log['loss'] = total_loss.detach().item()
        if use_seg:
            wandb_log['seg_loss'] = seg_loss.detach().item()
            wandb_log['heatmap_loss'] = heatmap_loss.detach().item()
        if use_normals:
            wandb_log['normals_loss'] = normals_loss.detach().item()
        progress_bar.set_postfix({'loss': wandb_log['loss']})
        progress_bar.update(1)

        if iteration % config['log_frequency'] == 0:
            with torch.no_grad():
                model.eval()

                val_batch = next(val_iterator)
                val_inputs, val_targets, val_seg, val_seg_mask, val_normals, val_normals_mask = prepare_batch(val_batch, config)
                if 'uv_heatmaps_out_mask' in val_batch:
                    val_mask = rearrange(val_batch['uv_heatmaps_out_mask'], 'b z y x c -> b c z y x')
                else:
                    val_mask = torch.ones_like(val_targets)
                val_seg_pred_for_vis = None
                val_normals_pred_for_vis = None
                if chain_multistep:
                    if ds_enabled or use_seg or use_normals:
                        raise ValueError("chain multistep validation currently supports heatmaps only; disable deep supervision and auxiliary heads.")
                    total_val_loss, val_target_pred_for_vis = compute_multistep_loss_and_pred(
                        model, val_inputs, val_targets, val_batch, config
                    )
                elif slot_multistep:
                    if ds_enabled or use_seg or use_normals:
                        raise ValueError("slot multistep validation currently supports heatmaps only; disable deep supervision and auxiliary heads.")
                    total_val_loss, val_target_pred_for_vis = compute_slot_multistep_loss_and_pred(
                        model, val_inputs, val_targets, val_mask, val_batch, config
                    )
                else:
                    val_outputs = model(val_inputs)
                    val_target_pred = val_outputs['uv_heatmaps'] if isinstance(val_outputs, dict) else val_outputs
                    val_heatmap_loss, val_target_pred_for_vis = compute_loss_with_ds(
                        val_target_pred, val_targets, val_mask, loss_fn, 'uv'
                    )
                    total_val_loss = val_heatmap_loss
                    if use_seg:
                        val_seg = require_aux(val_seg, 'seg')
                        val_seg_mask_for_loss = val_seg_mask if val_seg_mask is not None else torch.ones_like(val_seg)
                        val_seg_pred = require_head(val_outputs, 'seg')
                        val_seg_loss, val_seg_pred_for_vis = compute_loss_with_ds(
                            val_seg_pred, val_seg, val_seg_mask_for_loss, loss_fn, 'seg'
                        )
                        total_val_loss = total_val_loss + seg_loss_weight * val_seg_loss
                    if use_normals:
                        val_normals = require_aux(val_normals, 'normals')
                        val_normals_pred = require_head(val_outputs, 'normals')
                        val_normals_loss, val_normals_pred_for_vis = compute_loss_with_ds(
                            val_normals_pred, val_normals, val_normals_mask, normals_loss_fn, 'normals'
                        )
                        total_val_loss = total_val_loss + normals_loss_weight * val_normals_loss
                wandb_log['val_loss'] = total_val_loss.item()
                if not chain_multistep and use_seg:
                    wandb_log['val_seg_loss'] = val_seg_loss.item()
                    wandb_log['val_heatmap_loss'] = val_heatmap_loss.item()
                if not chain_multistep and use_normals:
                    wandb_log['val_normals_loss'] = val_normals_loss.item()

                if False:
                    def squish(x):
                        return torch.stack([x[:, :8].amax(dim=1), x[:, 8:16].amax(dim=1), torch.zeros_like(x[:, 0])], dim=1)
                    canvas = torch.stack([
                        inputs[:, :1].expand(-1, 3, -1, -1, -1),
                        squish(inputs[:, 2:]),
                        squish(target_pred),
                        squish(targets),
                    ], dim=-1)
                    canvas_mask = torch.stack([torch.ones_like(mask), mask, torch.ones_like(mask), mask], dim=-1)
                    canvas = (canvas * 0.5 + 0.5).clip(0, 1) * canvas_mask
                    canvas = rearrange(canvas[:, :, canvas.shape[2] // 2], 'b uvw y x v -> (b y) (v x) uvw')
                else:
                    def make_canvas(inputs, targets, target_pred, seg=None, seg_pred=None, normals=None, normals_pred=None, normals_mask=None):
                        sample_count = min(inputs.shape[0], log_image_max_samples)
                        inputs = inputs[:sample_count]
                        targets = targets[:sample_count]
                        target_pred = target_pred[:sample_count]
                        if seg is not None:
                            seg = seg[:sample_count]
                        if seg_pred is not None:
                            seg_pred = seg_pred[:sample_count]
                        if normals is not None:
                            normals = normals[:sample_count]
                        if normals_pred is not None:
                            normals_pred = normals_pred[:sample_count]
                        if normals_mask is not None:
                            normals_mask = normals_mask[:sample_count]

                        if multistep_enabled:
                            multistep_count = int(config.get('multistep_count', 1))
                            if multistep_count > 1 and targets.shape[1] % multistep_count == 0:
                                uv_channels = targets.shape[1] // multistep_count
                                targets = rearrange(targets, 'b (uv s) z y x -> b uv s z y x', uv=uv_channels).amax(dim=2)
                                target_pred = rearrange(target_pred, 'b (uv s) z y x -> b uv s z y x', uv=uv_channels).amax(dim=2)

                        colours_by_step = torch.rand([targets.shape[1], 3], device=inputs.device) * 0.7 + 0.2
                        colours_by_step = torch.cat([torch.ones([3, 3], device=inputs.device), colours_by_step], dim=0)  # white for conditioning points
                        def overlay_crosshair(x):
                            x = x.clone()
                            red = torch.tensor([0.8, 0, 0], device=x.device)
                            x[:, x.shape[1] // 2 - 7 : x.shape[1] // 2 - 1, x.shape[2] // 2, :] = red
                            x[:, x.shape[1] // 2 + 2 : x.shape[1] // 2 + 8, x.shape[2] // 2, :] = red
                            x[:, x.shape[1] // 2, x.shape[2] // 2 - 7 : x.shape[2] // 2 - 1, :] = red
                            x[:, x.shape[1] // 2, x.shape[2] // 2 + 2 : x.shape[2] // 2 + 8, :] = red
                            return x
                        def inputs_slice(dim):
                            return overlay_crosshair(inputs[:, 0].select(dim=dim + 1, index=inputs.shape[(dim + 2)] // 2)[..., None].expand(-1, -1, -1, 3) * 0.5 + 0.5)
                        def projections(x):
                            x = torch.cat([inputs[:, 2:5], x], dim=1)
                            coloured = x[..., None] * colours_by_step[None, :, None, None, None, :]
                            return torch.cat([overlay_crosshair(coloured.amax(dim=(1, dim + 2))) for dim in range(3)], dim=1)
                        def seg_overlay(mask, colour, alpha=0.6):
                            views = []
                            volume = inputs[:, 0]
                            for dim in range(3):
                                vol_slice = volume.select(dim=dim + 1, index=volume.shape[dim + 1] // 2)[..., None].expand(-1, -1, -1, 3) * 0.5 + 0.5
                                mask_slice = mask[:, 0].select(dim=dim + 1, index=mask.shape[dim + 1] // 2)[..., None].clamp(0, 1)
                                coloured = vol_slice * (1 - mask_slice * alpha) + colour * (mask_slice * alpha)
                                views.append(overlay_crosshair(coloured))
                            return torch.cat(views, dim=1)

                        views = [
                            torch.cat([inputs_slice(dim) for dim in range(3)], dim=1),
                            projections(F.sigmoid(target_pred)),
                            projections(targets),
                        ]
                        if seg is not None:
                            views.append(seg_overlay((seg != 0).float(), torch.tensor([0.0, 1.0, 0.0], device=inputs.device)))
                            if seg_pred is not None:
                                seg_pred_vis = seg_pred
                                if isinstance(seg_pred_vis, (list, tuple)):
                                    seg_pred_vis = seg_pred_vis[0]
                                seg_pred_mask = torch.sigmoid(seg_pred_vis)  # keep probabilities to show confidence instead of a binary fill
                                views.append(seg_overlay(seg_pred_mask, torch.tensor([0.0, 0.0, 1.0], device=inputs.device), alpha=0.45))
                        def normals_vis(n, alpha=0.6):
                            n = torch.tanh(n)  # keep in [-1,1]
                            n = (n + 1) / 2    # to [0,1] RGB
                            slices = []
                            for dim in range(3):
                                mid_idx = n.shape[dim + 2] // 2
                                n_slice = n.select(dim=dim + 2, index=mid_idx)
                                n_slice = rearrange(n_slice, 'b c h w -> b h w c')

                                vol_slice = inputs[:, 0].select(dim=dim + 1, index=inputs.shape[dim + 2] // 2)[..., None].expand(-1, -1, -1, 3) * 0.5 + 0.5
                                blended = vol_slice * (1 - alpha) + n_slice * alpha
                                slices.append(overlay_crosshair(blended))
                            return torch.cat(slices, dim=1)
                        if normals is not None:
                            views.append(normals_vis(normals))
                            if normals_pred is not None:
                                n_pred = normals_pred
                                if isinstance(n_pred, (list, tuple)):
                                    n_pred = n_pred[0]
                                views.append(normals_vis(n_pred))

                        canvas = torch.stack(views, dim=-1)
                        sample_canvases = rearrange(canvas.clip(0, 1), 'b y x rgb v -> b y (v x) rgb').cpu()
                        b, h, w, c = sample_canvases.shape
                        cols = min(log_image_grid_cols, b)
                        rows = math.ceil(b / cols)
                        grid = torch.zeros((rows * h, cols * w, c), dtype=sample_canvases.dtype)
                        for idx in range(b):
                            row, col = divmod(idx, cols)
                            grid[row * h : (row + 1) * h, col * w : (col + 1) * w] = sample_canvases[idx]
                        return grid

                train_canvas = make_canvas(inputs, targets, target_pred_for_vis, seg, seg_pred_for_vis, normals, normals_pred_for_vis, normals_mask)
                val_canvas = make_canvas(val_inputs, val_targets, val_target_pred_for_vis, val_seg, val_seg_pred_for_vis, val_normals, val_normals_pred_for_vis, val_normals_mask)
                save_kwargs = {'format': log_image_ext}
                if log_image_ext in ('jpg', 'jpeg'):
                    save_kwargs['pil_kwargs'] = {'quality': log_image_quality}
                plt.imsave(f'{out_dir}/{iteration:06}_train.{log_image_ext}', train_canvas, **save_kwargs)
                plt.imsave(f'{out_dir}/{iteration:06}_val.{log_image_ext}', val_canvas, **save_kwargs)

                model.train()

        if iteration % config['ckpt_frequency'] == 0:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
                'step': iteration,
            }, f'{out_dir}/ckpt_{iteration:06}.pth' )

        if wandb.run is not None:
            wandb.log(wandb_log)

        if iteration == config['num_iterations']:
            break


if __name__ == '__main__':
    train()
