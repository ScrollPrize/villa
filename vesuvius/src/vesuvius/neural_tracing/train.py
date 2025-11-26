
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

from vesuvius.neural_tracing.dataset import HeatmapDatasetV2, load_datasets, make_heatmaps
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
from vesuvius.neural_tracing.visualization import make_canvas, print_training_config



def prepare_batch(batch, recrop_center, recrop_size):
    if recrop_center is None:
        recrop_center = torch.tensor(batch['volume'].shape[-3:]) // 2
    inputs = torch.cat([
        batch['volume'].unsqueeze(1),
        batch['localiser'].unsqueeze(1),
        rearrange(batch['uv_heatmaps_in'], 'b z y x c -> b c z y x'),
    ], dim=1)
    targets = rearrange(batch['uv_heatmaps_out'], 'b z y x c -> b c z y x')
    return recrop(inputs, recrop_center, recrop_size), recrop(targets, recrop_center, recrop_size)

@click.command()
@click.argument('config_path', type=click.Path(exists=True))
def train(config_path):

    with open(config_path, 'r') as f:
        config = json.load(f)

    out_dir = config['out_dir']
    os.makedirs(out_dir, exist_ok=True)

    ds_enabled = config.setdefault('enable_deep_supervision', False)
    multistep_enabled = config.setdefault('multistep_count', 1) > 1
    use_seg = config.setdefault('aux_segmentation', False)
    use_normals = config.setdefault('aux_normals', False)
    seg_loss_weight = config.setdefault('seg_loss_weight', 1.0)
    normals_loss_weight = config.setdefault('normals_loss_weight', 1.0)
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

    accelerator = accelerate.Accelerator(
        mixed_precision=config['mixed_precision'],
        gradient_accumulation_steps=config.setdefault('grad_acc_steps', 1),
    )

    if 'wandb_project' in config and accelerator.is_main_process:
        wandb.init(project=config['wandb_project'], entity=config.get('wandb_entity', None), config=config)

    if config['representation'] == 'heatmap':
        train_patches, val_patches = load_datasets(config)
        train_dataset = HeatmapDatasetV2(config, train_patches)
        val_dataset = HeatmapDatasetV2(config, val_patches)
    else:
        train_dataset = val_dataset = PatchInCubeDataset(config)
        
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], num_workers=config['num_workers'])
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=config['batch_size'] * 2, num_workers=1)

    # FIXME: need separate data-loaders for multi-step and single-step training, since have different target shapes

    model = make_model(config)
    config.setdefault('compile_model', config.get('compile', True))
    compile_enabled = config['compile_model']
    if compile_enabled:
        model = torch.compile(model)

    scheduler_type = config.setdefault('scheduler', 'diffusers_cosine_warmup')
    scheduler_kwargs = dict(config.setdefault('scheduler_kwargs', {}) or {})
    scheduler_kwargs.setdefault('warmup_steps', config.setdefault('lr_warmup_steps', 1000))
    config['scheduler_kwargs'] = scheduler_kwargs
    total_scheduler_steps = config['num_iterations'] * accelerator.state.num_processes  # See comment below on accelerator.prepare
    # FIXME: accelerator.prepare wraps schedulers so that they step once per process; multiply steps to compensate

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

    grad_clip = config.setdefault('grad_clip', 5)

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
        print_training_config(config, accelerator)

    val_iterator = iter(val_dataloader)

    def loss_fn(target_pred, targets, mask):
        """Compute per-batch losses (returns shape [batch_size], caller must apply .mean())."""
        if config['binary']:
            targets_binary = (targets > 0.5).long()  # FIXME: should instead not do the gaussian conv in data-loader!
            # FIXME: nasty; fix DC_and_BCE_loss themselves to support not reducing over batch dim
            bce = torch.nn.BCEWithLogitsLoss(reduction='none')(target_pred, targets_binary.float()).mean(dim=(1, 2, 3, 4))
            from vesuvius.models.training.loss.nnunet_losses import MemoryEfficientSoftDiceLoss
            dice_loss_fn = MemoryEfficientSoftDiceLoss(apply_nonlin=torch.sigmoid, batch_dice=False, ddp=False)
            dice = torch.stack([
                dice_loss_fn(target_pred[i:i+1], targets_binary[i:i+1]) for i in range(target_pred.shape[0])
            ])
            return bce + dice
        else:
            # TODO: should this instead weight each element in batch equally regardless of valid area?
            per_batch = ((target_pred - targets) ** 2 * mask).sum(dim=(1, 2, 3, 4)) / mask.sum(dim=(1, 2, 3, 4))
            return per_batch

    def compute_multistep_loss_and_pred(model, inputs, targets, mask, batch, config):
        """Multistep sampling + importance-weighted loss; returns scalar loss and stacked preds for viz."""
        multistep_count = int(config.get('multistep_count', 1))
        if multistep_count <= 1:
            raise ValueError("compute_multistep_loss_and_pred called with multistep_count <= 1")
        sample_count = int(config.get('multistep_samples', 1))

        target_pred = model(inputs)

        if targets.shape[1] <= target_pred.shape[1]:
            raise ValueError("multistep training expects targets to have more channels than model outputs")

        first_step_targets = targets[:, ::multistep_count]
        first_step_loss = loss_fn(target_pred, first_step_targets, mask)

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
                step_inputs = torch.cat([
                    step_volume_crop.unsqueeze(1),
                    inputs[:, 1:2],  # borrow the original localiser subcrop
                    prev_uv,  # prev_u, prev_v
                    torch.zeros_like(prev_uv[:, :1]),  # prev_diag
                ], dim=1)

                # TODO: make checkpointing optional
                step_pred = torch.utils.checkpoint.checkpoint(model, step_inputs)

                step_targets = safe_crop_with_padding(
                    batch['uv_heatmaps_out'],
                    min_corner_new_subcrop_in_outer,
                    config['crop_size']
                )
                step_targets = rearrange(step_targets[..., step_idx::multistep_count], 'b z y x c -> b c z y x')

                if sample_idx == 0:
                    step_pred_filtered = torch.full_like(step_pred, -100.0)
                    step_pred_filtered[torch.arange(step_pred.shape[0]), step_directions] = step_pred.detach()[torch.arange(step_pred.shape[0]), step_directions]
                    first_step_crop_min = outer_crop_center - config['crop_size'] // 2
                    offset = min_corner_new_subcrop_in_outer - first_step_crop_min
                    step_pred_in_first_crop = transform_to_first_crop_space(step_pred_filtered, offset, config['crop_size'])
                    all_step_preds_vis.append(step_pred_in_first_crop)

                # Since the model runs in single-cond mode for this step, it predicts a point along
                # the cond direction, but also one/two along the other direction; those others are
                # not included in gt targets (because they're not part of the chain). We therefore
                # only take targets and preds in the direction of the along-chain conditioning
                step_pred = step_pred[torch.arange(step_pred.shape[0]), step_directions].unsqueeze(1)
                step_targets = step_targets[torch.arange(step_targets.shape[0]), step_directions].unsqueeze(1)

                step_loss = loss_fn(step_pred, step_targets, mask)
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

    def require_head(outputs, name):
        if isinstance(outputs, dict) and name in outputs:
            return outputs[name]
        raise ValueError(f"aux_{name} is enabled but model did not return '{name}'")

    def compute_loss_with_ds(pred, target, mask, base_loss_fn, cache_key):
        # Wrap base_loss_fn to return scalar (mean over batch) for DS wrapper compatibility
        # Some loss functions (e.g., loss_fn) return per-batch [B], others (e.g., CosineSimilarityLoss) return scalar
        def mean_loss_fn(p, t, m):
            loss = base_loss_fn(p, t, m)
            return loss.mean() if loss.dim() > 0 else loss

        pred_for_vis = pred
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

    progress_bar = tqdm(total=config['num_iterations'], disable=not accelerator.is_local_main_process)
    for iteration, batch in enumerate(train_dataloader):

        inputs, targets = prepare_batch(batch, None, config['crop_size'])
        mask = torch.ones_like(targets[:, :1, ...])  # TODO!

        wandb_log = {}
        target_pred_for_vis = None
        with accelerator.accumulate(model):
            if multistep_enabled:
                if ds_enabled or use_seg or use_normals:
                    raise ValueError("multistep currently supports heatmaps only; disable deep supervision and auxiliary heads.")
                total_loss, target_pred_for_vis = compute_multistep_loss_and_pred(
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
                seg = seg_mask = seg_pred_for_vis = None
                normals = normals_mask = normals_pred_for_vis = None

                if use_seg:
                    seg = batch['seg'].unsqueeze(1)
                    seg_mask = (seg > 0).float()  # mask where seg is labeled
                    seg_pred = require_head(outputs, 'seg')
                    seg_loss, seg_pred_for_vis = compute_loss_with_ds(
                        seg_pred, seg, seg_mask, loss_fn, 'seg'
                    )
                    total_loss = total_loss + seg_loss_weight * seg_loss
                if use_normals:
                    normals = rearrange(batch['normals'], 'b z y x c -> b c z y x')
                    normals_mask = (normals.abs().sum(dim=1, keepdim=True) > 0).float()  # mask where normals exist
                    normals_pred = require_head(outputs, 'normals')
                    normals_loss, normals_pred_for_vis = compute_loss_with_ds(
                        normals_pred, normals, normals_mask, normals_loss_fn, 'normals'
                    )
                    total_loss = total_loss + normals_loss_weight * normals_loss

            if torch.isnan(total_loss).any():
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
                val_inputs, val_targets = prepare_batch(val_batch, None, config['crop_size'])
                val_mask = torch.ones_like(val_targets[:, :1, ...])  # TODO!
                val_seg = val_seg_mask = val_seg_pred_for_vis = None
                val_normals = val_normals_mask = val_normals_pred_for_vis = None
                if multistep_enabled:
                    if ds_enabled or use_seg or use_normals:
                        raise ValueError("multistep validation currently supports heatmaps only; disable deep supervision and auxiliary heads.")
                    total_val_loss, val_target_pred_for_vis = compute_multistep_loss_and_pred(
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
                        val_seg = val_batch['seg'].unsqueeze(1)
                        val_seg_mask = (val_seg > 0).float()  # mask where seg is labeled
                        val_seg_pred = require_head(val_outputs, 'seg')
                        val_seg_loss, val_seg_pred_for_vis = compute_loss_with_ds(
                            val_seg_pred, val_seg, val_seg_mask, loss_fn, 'seg'
                        )
                        total_val_loss = total_val_loss + seg_loss_weight * val_seg_loss
                    if use_normals:
                        val_normals = rearrange(val_batch['normals'], 'b z y x c -> b c z y x')
                        val_normals_mask = (val_normals.abs().sum(dim=1, keepdim=True) > 0).float()  # mask where normals exist
                        val_normals_pred = require_head(val_outputs, 'normals')
                        val_normals_loss, val_normals_pred_for_vis = compute_loss_with_ds(
                            val_normals_pred, val_normals, val_normals_mask, normals_loss_fn, 'normals'
                        )
                        total_val_loss = total_val_loss + normals_loss_weight * val_normals_loss
                wandb_log['val_loss'] = total_val_loss.item()
                if not multistep_enabled and use_seg:
                    wandb_log['val_seg_loss'] = val_seg_loss.item()
                    wandb_log['val_heatmap_loss'] = val_heatmap_loss.item()
                if not multistep_enabled and use_normals:
                    wandb_log['val_normals_loss'] = val_normals_loss.item()

                log_image_ext = config.get('log_image_ext', 'jpg')
                make_canvas(inputs, targets, target_pred_for_vis, config,
                           seg=seg, seg_pred=seg_pred_for_vis, normals=normals,
                           normals_pred=normals_pred_for_vis, normals_mask=normals_mask,
                           save_path=f'{out_dir}/{iteration:06}_train.{log_image_ext}')
                make_canvas(val_inputs, val_targets, val_target_pred_for_vis, config,
                           seg=val_seg, seg_pred=val_seg_pred_for_vis, normals=val_normals,
                           normals_pred=val_normals_pred_for_vis, normals_mask=val_normals_mask,
                           save_path=f'{out_dir}/{iteration:06}_val.{log_image_ext}')

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
