
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
from vesuvius.models.training.loss.nnunet_losses import DeepSupervisionWrapper, MemoryEfficientSoftDiceLoss
from vesuvius.models.training.optimizers import create_optimizer
from vesuvius.models.training.lr_schedulers import get_scheduler
from vesuvius.neural_tracing.deep_supervision import _resize_for_ds, _compute_ds_weights
from vesuvius.neural_tracing.models import make_model, strip_state, resolve_checkpoint_path
from vesuvius.neural_tracing.cropping import safe_crop_with_padding, transform_to_first_crop_space
from vesuvius.models.training.loss.losses import CosineSimilarityLoss
from vesuvius.neural_tracing.visualization import make_canvas, print_training_config


@click.command()
@click.argument('config_path', type=click.Path(exists=True))
def train(config_path):

    with open(config_path, 'r') as f:
        config = json.load(f)

    out_dir = config['out_dir']
    os.makedirs(out_dir, exist_ok=True)

    ds_enabled = config.setdefault('enable_deep_supervision', False)
    use_seg = config.setdefault('aux_segmentation', False)
    use_normals = config.setdefault('aux_normals', False)
    use_srf_overlap = config.setdefault('aux_srf_overlap', False)
    seg_loss_weight = config.setdefault('seg_loss_weight', 1.0)
    normals_loss_weight = config.setdefault('normals_loss_weight', 1.0)
    srf_overlap_loss_weight = config.setdefault('srf_overlap_loss_weight', 1.0)
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])

    normals_loss_fn = CosineSimilarityLoss(dim=1, eps=1e-8)
    ds_cache = {
        'uv': {'weights': None, 'loss_fn': None},
        'seg': {'weights': None, 'loss_fn': None},
        'normals': {'weights': None, 'loss_fn': None},
        'srf_overlap': {'weights': None, 'loss_fn': None},
    }

    accelerator = accelerate.Accelerator(
        mixed_precision=config['mixed_precision'],
        gradient_accumulation_steps=config.setdefault('grad_acc_steps', 1),
    )

    if 'wandb_project' in config and accelerator.is_main_process:
        wandb.init(project=config['wandb_project'], entity=config.get('wandb_entity', None), config=config)
    
    if 'multistep_max_count' in config:
        multistep_increase_iters = np.asarray(config['multistep_count_increase_iters'], dtype=np.int64)
        assert len(multistep_increase_iters) == config['multistep_max_count'] - 1
        assert np.all(multistep_increase_iters >= 0) and np.all(multistep_increase_iters < config['num_iterations'])
        assert np.all(np.diff(multistep_increase_iters) > 0)
    else:
        assert 'multistep_count_increase_iters' not in config
        multistep_increase_iters = []
    multistep_count = 1
    bidirectional = False  # ...since at first we're single-step for which bidirectional is not supported

    train_patches, val_patches = load_datasets(config, shard_idx=accelerator.process_index, total_shards=accelerator.num_processes)

    def make_dataloaders():
        train_dataset = HeatmapDatasetV2(config, train_patches, multistep_count=multistep_count, bidirectional=bidirectional)
        val_dataset = HeatmapDatasetV2(config, val_patches, multistep_count=multistep_count, bidirectional=bidirectional)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], num_workers=config['num_workers'], pin_memory=True)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=config['batch_size'] * 2, num_workers=1, pin_memory=True)
        def to_gpu(dataloader):  # we don't use accelerator.prepare since we handle sharding ourselves
            for batch in dataloader:
                yield {k: v.to(accelerator.device, non_blocking=True) for k, v in batch.items()}
        return to_gpu(train_dataloader), to_gpu(val_dataloader)

    train_dataloader, val_dataloader = make_dataloaders()

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
        ckpt_path = resolve_checkpoint_path(config['load_ckpt'])
        print(f'loading checkpoint {ckpt_path}')
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

        model.load_state_dict(strip_state(ckpt['model']))
        # Only load optimizer state if optimizer type matches (avoid SGD/Adam mismatch)
        ckpt_optim_type = type(ckpt['optimizer']['param_groups'][0].get('betas', None))
        curr_optim_type = type(optimizer.param_groups[0].get('betas', None))
        if ckpt_optim_type == curr_optim_type:
            optimizer.load_state_dict(ckpt['optimizer'])
        else:
            print(f'Skipping optimizer state load (optimizer type changed)')
        lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
        first_iteration = ckpt['step']
        # Note we don't load the config saved with the ckpt!

        if len(multistep_increase_iters) > 0:
            multistep_count = 1 + np.searchsorted(multistep_increase_iters, first_iteration)
            print(f'resuming with multistep_count = {multistep_count}')
            if multistep_count > 1:
                bidirectional = config.get('bidirectional', False)
                train_dataloader, val_dataloader = make_dataloaders()

    else:
        first_iteration = 0

    model, optimizer, lr_scheduler = accelerator.prepare(
        model, optimizer, lr_scheduler  # note we do the dataloaders in make_dataloaders
    )

    if accelerator.is_main_process:
        print_training_config(config, accelerator)

    train_iterator = iter(train_dataloader)
    val_iterator = iter(val_dataloader)

    def prepare_inputs(batch, min_corner_in_outer, prev_uvd_cropped):
        # Prepare the volume (sub-)crop and localiser. For the localiser we take a crop at the
        # original center (conceptually we create a new localiser at the new subcrop center)
        crop_size = config['crop_size']
        volume_crop = safe_crop_with_padding(
            batch['volume'],
            min_corner_in_outer,
            crop_size
        )
        outer_shape = batch['localiser'].shape
        localiser = batch['localiser'][
            :,
            outer_shape[1] // 2 - crop_size // 2 : outer_shape[1] // 2 + crop_size // 2,
            outer_shape[2] // 2 - crop_size // 2 : outer_shape[2] // 2 + crop_size // 2,
            outer_shape[3] // 2 - crop_size // 2 : outer_shape[3] // 2 + crop_size // 2,
        ]
        inputs = torch.cat([
            volume_crop.unsqueeze(1),
            localiser.unsqueeze(1),
            prev_uvd_cropped,  # prev_u, prev_v, prev_diag
        ], dim=1)
        return inputs

    def loss_fn(target_pred, targets, mask):
        """Compute per-batch losses (returns shape [batch_size], caller must apply .mean())."""
        if config['binary']:
            targets_binary = (targets > 0.5).long()  # FIXME: should instead not do the gaussian conv in data-loader!
            # FIXME: nasty; fix DC_and_BCE_loss themselves to support not reducing over batch dim
            # FIXME: nasty treatment of mask
            bce = torch.nn.BCEWithLogitsLoss(reduction='none')(target_pred * mask, targets_binary.float()).mean(dim=(1, 2, 3, 4))
            from vesuvius.models.training.loss.nnunet_losses import MemoryEfficientSoftDiceLoss
            dice_loss_fn = MemoryEfficientSoftDiceLoss(apply_nonlin=torch.sigmoid, batch_dice=False, ddp=False)
            dice = torch.stack([
                dice_loss_fn(target_pred[i:i+1] * mask[i:i+1], targets_binary[i:i+1] * mask[i:i+1]) for i in range(target_pred.shape[0])
            ])
            return bce + dice
        else:
            # TODO: should this instead weight each element in batch equally regardless of valid area?
            per_batch = ((target_pred - targets) ** 2 * mask).sum(dim=(1, 2, 3, 4)) / mask.sum(dim=(1, 2, 3, 4))
            return per_batch

    def compute_loss_and_pred(batch):

        sample_count = config.get('multistep_samples', 1)
        sampling_mode = config.get('multistep_sampling_mode', 'categorical')
        assert sampling_mode in ['categorical', 'expectation']

        outer_crop_shape = torch.tensor(batch['volume'].shape[-3:], device=accelerator.device)
        outer_crop_center = outer_crop_shape // 2

        first_min_corner_in_outer = outer_crop_center - config['crop_size'] // 2
        first_prev_uvd_subcrop = rearrange(safe_crop_with_padding(batch['uv_heatmaps_in'], first_min_corner_in_outer, config['crop_size']), 'b z y x c -> b c z y x')
        first_step_inputs = prepare_inputs(batch, first_min_corner_in_outer, first_prev_uvd_subcrop)
        targets = rearrange(safe_crop_with_padding(batch['uv_heatmaps_out'], first_min_corner_in_outer, config['crop_size']), 'b z y x c -> b c z y x')
        mask = torch.ones_like(targets[:, :1, ...])  # TODO

        if multistep_count > 1:
            outputs = torch.utils.checkpoint.checkpoint(model, first_step_inputs, use_reentrant=False)
        else:
            outputs = model(first_step_inputs)
        
        target_pred = outputs['uv_heatmaps'] if isinstance(outputs, dict) else outputs

        assert targets.shape[1] == target_pred.shape[1] * multistep_count
        first_step_targets = targets[:, ::multistep_count]

        # First step: compute loss with deep supervision if enabled
        if ds_enabled and isinstance(target_pred, (list, tuple)):
            # Deep supervision: compute loss at each scale and sum (weighted), but preserve per-batch structure
            cache = ds_cache['uv']
            if cache['weights'] is None or len(cache['weights']) != len(target_pred):
                cache['weights'] = _compute_ds_weights(len(target_pred))
            targets_resized = [_resize_for_ds(first_step_targets, t.shape[2:], mode='trilinear', align_corners=False) for t in target_pred]
            masks_resized = None
            if mask is not None:
                masks_resized = [_resize_for_ds(mask, t.shape[2:], mode='nearest') for t in target_pred]

            # Compute per-batch losses at each scale
            first_step_heatmap_loss = torch.zeros([], device=target_pred[0].device)
            for i, weight in enumerate(cache['weights']):
                if weight != 0:
                    scale_loss = loss_fn(target_pred[i], targets_resized[i], masks_resized[i] if masks_resized else mask)
                    first_step_heatmap_loss = first_step_heatmap_loss + weight * scale_loss
        else:
            if isinstance(target_pred, (list, tuple)):
                target_pred = target_pred[0]
            first_step_heatmap_loss = loss_fn(target_pred, first_step_targets, mask)

        first_step_heatmap_loss = first_step_heatmap_loss.mean()  # over batch
        first_step_loss = first_step_heatmap_loss

        # Add auxiliary segmentation & normal losses for first step
        if use_seg:
            seg = safe_crop_with_padding(batch['seg'], first_min_corner_in_outer, config['crop_size']).unsqueeze(1)
            seg_mask = torch.ones_like(seg)  # supervise full crop
            seg_pred = require_head(outputs, 'seg')
            seg_loss, seg_pred_for_vis = compute_loss_with_ds(
                seg_pred, seg, seg_mask, loss_fn, 'seg'
            )
            first_step_loss = first_step_loss + seg_loss_weight * seg_loss
        else:
            seg_loss = seg = seg_pred_for_vis = None
        if use_normals:
            normals = rearrange(safe_crop_with_padding(batch['normals'], first_min_corner_in_outer, config['crop_size']), 'b z y x c -> b c z y x')
            normals_mask = (normals.abs().sum(dim=1, keepdim=True) > 0).float()
            normals_pred = require_head(outputs, 'normals')
            normals_loss, normals_pred_for_vis = compute_loss_with_ds(
                normals_pred, normals, normals_mask, normals_loss_fn, 'normals'
            )
            first_step_loss = first_step_loss + normals_loss_weight * normals_loss
        else:
            normals_loss = normals = normals_pred_for_vis = None
        if use_srf_overlap and batch.get('srf_overlap_mask') is not None:
            from vesuvius.neural_tracing.surf_overlap_loss import compute_surf_overlap_loss
            srf_overlap_mask_raw = batch['srf_overlap_mask']
            # Crop srf_overlap mask to match the subcrop
            srf_overlap_mask_cropped = safe_crop_with_padding(srf_overlap_mask_raw.unsqueeze(-1), first_min_corner_in_outer, config['crop_size']).squeeze(-1)
            srf_overlap_mask_cropped = srf_overlap_mask_cropped.unsqueeze(1)  # [B, 1, Z, Y, X]
            # Use the heatmap predictions to compute srf_overlap loss
            target_pred_for_srf_overlap = target_pred[0] if isinstance(target_pred, (list, tuple)) else target_pred
            srf_overlap_loss = compute_surf_overlap_loss(target_pred_for_srf_overlap, srf_overlap_mask_cropped, mask).mean()
            first_step_loss = first_step_loss + srf_overlap_loss_weight * srf_overlap_loss
        else:
            srf_overlap_loss = None

        # Direction to extend (assumes exactly one of prev_u/prev_v is set)
        step_directions = batch['uv_heatmaps_in'].amax(dim=[1, 2, 3])[:, :2].argmax(dim=-1)

        first_step_cube_radius = 4
        later_step_cube_radius = 2  # smaller radius for later steps to reduce variance

        def sample_for_next_step(step_pred_for_dir, num_samples, cube_radius):
            if sampling_mode == 'categorical':

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

            else:

                # TODO: we could instead split into blobs first (non-differentiably) then use expectations
                #  over near-blob (voronoi?) regions

                temperature = 1.0  # setting this too high will collapse centroid to the center of the crop
                prob_map = torch.sigmoid(step_pred_for_dir / temperature)
                prob_normalized = prob_map / (prob_map.sum(dim=(1, 2, 3), keepdim=True) + 1.e-8)

                batch_size = step_pred_for_dir.shape[0]
                shape = step_pred_for_dir.shape[1:]
                device = step_pred_for_dir.device

                z_coords = torch.arange(shape[0], device=device, dtype=torch.float32)
                y_coords = torch.arange(shape[1], device=device, dtype=torch.float32)
                x_coords = torch.arange(shape[2], device=device, dtype=torch.float32)
                zyx_grid = torch.stack(torch.meshgrid(z_coords, y_coords, x_coords, indexing='ij'), dim=-1)

                centroid = (prob_normalized.unsqueeze(-1) * zyx_grid).sum(dim=(1, 2, 3))

                if num_samples > 1:
                    noise_sigma = 1.5
                    noise = torch.randn(batch_size, num_samples, 3, device=device) * noise_sigma
                    sample_zyxs_in_subcrop = centroid[:, None, :] + noise
                else:
                    sample_zyxs_in_subcrop = centroid[:, None, :]

                sample_zyxs_in_subcrop = sample_zyxs_in_subcrop.clamp(
                    torch.tensor([0, 0, 0], device=device),
                    torch.tensor(shape, device=device) - 1.
                )

                # TODO: not correct in num_samples>1 case, and unclear what to do in num_samples=1 case
                sample_unnormalised_probs = torch.full([batch_size, num_samples], 1. / num_samples, device=device)
                proposal_probs = sample_unnormalised_probs

            return sample_zyxs_in_subcrop, sample_unnormalised_probs, proposal_probs

        first_step_pred_for_dir = target_pred[torch.arange(target_pred.shape[0]), step_directions]
        first_sample_zyxs_in_first_subcrop, first_step_sample_unnormalised_probs, first_step_proposal_probs = sample_for_next_step(
            first_step_pred_for_dir, num_samples=sample_count, cube_radius=first_step_cube_radius)

        first_sample_zyxs_in_outer_crop = first_sample_zyxs_in_first_subcrop + (outer_crop_shape - config['crop_size']) // 2

        losses_by_sample_by_later_step = []
        step_unnormalised_probs_by_sample_by_later_step = []
        step_proposal_probs_by_sample_by_later_step = []
        all_step_targets_vis = [first_step_targets]
        all_step_preds_vis = [target_pred.detach()]

        # FIXME: if we supported multi-step for non-chain cases, then we could always enable it hence wouldn't need this
        # Note for batch elements where multi-step targets are not available, we still compute multi-step predictions,
        # but do not calculate the loss nor visualise them
        multistep_available_device = mask.device
        if multistep_count > 1:
            multistep_targets_available = batch['uv_heatmaps_out'][:, ..., 1::multistep_count].amax(dim=(1, 2, 3, 4)) > 0
        else:
            multistep_targets_available = torch.zeros(batch['uv_heatmaps_out'].shape[0], dtype=bool, device=multistep_available_device)
        multistep_targets_available = multistep_targets_available.to(multistep_available_device, non_blocking=True)
        mask = mask * multistep_targets_available[:, None, None, None, None]

        for sample_idx in range(sample_count):
            current_center_in_outer_crop = first_sample_zyxs_in_outer_crop[:, sample_idx, :]
            prev_center_in_outer_crop = outer_crop_center.expand(current_center_in_outer_crop.shape[0], -1)

            sample_losses = []
            sample_step_unnormalised_probs = [first_step_sample_unnormalised_probs[:, sample_idx]]
            sample_step_proposal_probs = [first_step_proposal_probs[:, sample_idx]]

            def do_step(direction):
                nonlocal current_center_in_outer_crop, prev_center_in_outer_crop
                assert direction in ['forward', 'backward']

                min_corner_new_subcrop_in_outer = current_center_in_outer_crop - config['crop_size'] // 2

                prev_heatmap = torch.cat([
                    make_heatmaps([prev_center_in_outer_crop[iib:iib+1]], min_corner_new_subcrop_in_outer[iib], config['crop_size'])
                    for iib in range(min_corner_new_subcrop_in_outer.shape[0])
                ], dim=0).to(prev_center_in_outer_crop.device, non_blocking=True)
                prev_uv = torch.zeros([prev_heatmap.shape[0], 2, *prev_heatmap.shape[1:]], device=prev_heatmap.device, dtype=prev_heatmap.dtype)
                prev_uv[torch.arange(prev_heatmap.shape[0]), step_directions] = prev_heatmap

                prev_uvd = torch.cat([prev_uv, torch.zeros_like(prev_uv[:, :1])], dim=1)
                step_inputs = prepare_inputs(batch, min_corner_new_subcrop_in_outer, prev_uvd)

                # TODO: make checkpointing optional
                step_pred = torch.utils.checkpoint.checkpoint(model, step_inputs, use_reentrant=False)
                step_pred = step_pred['uv_heatmaps'] if isinstance(outputs, dict) else outputs

                if direction == 'forward':
                    step_targets = batch['uv_heatmaps_out'][..., step_idx::multistep_count]
                elif step_idx > 1:
                    # When reversing 'later' steps, the target is the target of the previous forward step
                    step_targets = batch['uv_heatmaps_out'][..., step_idx - 2::multistep_count]
                else:
                    # When reversing the first step, the target is the initial center, i.e. center of the outer
                    # crop, which isn't a standard target
                    step_targets = batch['center_heatmap'].unsqueeze(-1).expand(-1, -1, -1, -1, 2)
                step_targets = rearrange(
                    safe_crop_with_padding(
                        step_targets,
                        min_corner_new_subcrop_in_outer,
                        config['crop_size']
                    ),
                    'b z y x c -> b c z y x'
                )

                if sample_idx == 0:
                    step_pred_filtered = torch.full_like(step_pred, -100.0)
                    where_multistep_targets_available = torch.where(multistep_targets_available)[0]
                    step_pred_filtered[where_multistep_targets_available, step_directions[multistep_targets_available]] = step_pred.detach()[where_multistep_targets_available, step_directions[multistep_targets_available]]
                    step_targets_filtered = torch.zeros_like(step_targets)
                    step_targets_filtered[where_multistep_targets_available, step_directions[multistep_targets_available]] = step_targets.detach()[where_multistep_targets_available, step_directions[multistep_targets_available]]
                    first_step_crop_min = outer_crop_center - config['crop_size'] // 2
                    offset = (min_corner_new_subcrop_in_outer - first_step_crop_min).round().int()
                    step_target_in_first_crop = transform_to_first_crop_space(step_targets_filtered, offset, config['crop_size'])
                    step_pred_in_first_crop = transform_to_first_crop_space(step_pred_filtered, offset, config['crop_size'])
                    all_step_targets_vis.append(step_target_in_first_crop)
                    all_step_preds_vis.append(step_pred_in_first_crop)

                # Since the model runs in single-cond mode for this step, it predicts a point along
                # the cond direction, but also one/two along the other direction; those others are
                # not included in gt targets (because they're not part of the chain). We therefore
                # only take targets and preds in the direction of the along-chain conditioning
                step_pred = step_pred[torch.arange(step_pred.shape[0]), step_directions].unsqueeze(1)
                step_targets = step_targets[torch.arange(step_targets.shape[0]), step_directions].unsqueeze(1)

                step_loss = loss_fn(step_pred, step_targets, mask)
                sample_losses.append(step_loss)

                step_pred_for_dir = step_pred.squeeze(1)
                sample_zyxs_in_subcrop, sample_unnormalised_probs, proposal_probs = sample_for_next_step(
                    step_pred_for_dir, num_samples=1, cube_radius=later_step_cube_radius)
                sample_zyxs_in_outer = sample_zyxs_in_subcrop.squeeze(1) + min_corner_new_subcrop_in_outer
                sample_step_unnormalised_probs.append(sample_unnormalised_probs.squeeze(1))
                sample_step_proposal_probs.append(proposal_probs.squeeze(1))
                prev_center_in_outer_crop = torch.where(multistep_targets_available[:, None], current_center_in_outer_crop, prev_center_in_outer_crop)
                current_center_in_outer_crop = torch.where(multistep_targets_available[:, None], sample_zyxs_in_outer, current_center_in_outer_crop)

            for step_idx in range(1, multistep_count):
                do_step(direction='forward')

            if bidirectional:

                # Take steps back along the chain. current_center_in_outer_crop is the output point from the last forward
                # step (i.e. what would be the next center if we took more steps forward); it becomes the first conditioning for
                # the reverse direction. prev_center_in_outer_crop is the previous point of the forward chain, i.e. the last
                # center point at which the model was evaluated (and also the first center for the reverse direction)

                assert multistep_count > 1  # FIXME: in principle we could support the 1-step case, but need to get the gt (unperturbed) target

                # Swap current and previous points, ready to go backwards
                current_center_in_outer_crop, prev_center_in_outer_crop = prev_center_in_outer_crop, current_center_in_outer_crop

                for step_idx in range(multistep_count - 1, 0, -1):
                    do_step(direction='backward')

            losses_by_sample_by_later_step.append(sample_losses)
            step_unnormalised_probs_by_sample_by_later_step.append(sample_step_unnormalised_probs)
            step_proposal_probs_by_sample_by_later_step.append(sample_step_proposal_probs)

        # Later step losses are accumulated and weighted using self-normalized importance sampling
        # later_step_idx is the index into losses_by_sample_by_later_step (which excludes first step)
        later_step_loss = torch.zeros([], device=mask.device)
        cumulative_weights = [1] * sample_count
        for later_step_idx in range((multistep_count - 1) * (2 if bidirectional else 1)):

            # Update cumulative importance weights
            for sample_idx in range(sample_count):
                target_prob = step_unnormalised_probs_by_sample_by_later_step[sample_idx][later_step_idx]
                proposal_prob = step_proposal_probs_by_sample_by_later_step[sample_idx][later_step_idx]
                cumulative_weights[sample_idx] = cumulative_weights[sample_idx] * target_prob / (proposal_prob + 1e-8)
            cumulative_weights_stacked = torch.stack(cumulative_weights, dim=0)  # sample_count, batch_size

            step_losses = torch.stack([
                losses_by_sample_by_later_step[sample_idx][later_step_idx]
                for sample_idx in range(sample_count)
            ], dim=0)
            weighted_loss_sum = (step_losses * cumulative_weights_stacked).sum(dim=0)
            total_weights_sum = cumulative_weights_stacked.sum(dim=0)
            later_step_loss = later_step_loss + (weighted_loss_sum / (total_weights_sum + 1e-8))

        later_step_loss = later_step_loss.mean()
        total_loss = first_step_loss + later_step_loss

        target_all_steps = rearrange(torch.stack(all_step_targets_vis, dim=2), 'b uv s z y x -> b (uv s) z y x')
        target_pred_all_steps = rearrange(torch.stack(all_step_preds_vis, dim=2), 'b uv s z y x -> b (uv s) z y x')

        return (
            total_loss, first_step_heatmap_loss, later_step_loss, seg_loss, normals_loss, srf_overlap_loss,
            first_step_inputs, target_all_steps, target_pred_all_steps,
            seg, seg_pred_for_vis, normals, normals_pred_for_vis
        )

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

    progress_bar = tqdm(total=config['num_iterations'], initial=first_iteration, disable=not accelerator.is_local_main_process)
    for iteration in range(first_iteration, config['num_iterations'] + 1):

        if iteration in multistep_increase_iters:
            multistep_count += 1
            bidirectional = config.get('bidirectional', False)
            print(f'increasing multistep_count to {multistep_count}')
            train_dataloader, val_dataloader = make_dataloaders()
            train_iterator = iter(train_dataloader)
            val_iterator = iter(val_dataloader)

        batch = next(train_iterator)

        wandb_log = {}
        with accelerator.accumulate(model):

            (
                total_loss, first_step_heatmap_loss, later_step_loss, seg_loss, normals_loss, srf_overlap_loss,
                inputs_for_vis, targets_for_vis, target_pred_for_vis,
                seg_for_vis, seg_pred_for_vis, normals_for_vis, normals_pred_for_vis
            ) = compute_loss_and_pred(batch)

            if torch.isnan(total_loss).any():
                raise ValueError('loss is NaN')
            accelerator.backward(total_loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        wandb_log['loss'] = total_loss.detach().item()
        wandb_log['first_step_heatmap_loss'] = first_step_heatmap_loss.detach().item()
        if multistep_count > 1:
            wandb_log['later_step_loss'] = later_step_loss.detach().item()
        if use_seg:
            wandb_log['seg_loss'] = seg_loss.detach().item()
        if use_normals:
            wandb_log['normals_loss'] = normals_loss.detach().item()
        if use_srf_overlap and srf_overlap_loss is not None:
            wandb_log['srf_overlap_loss'] = srf_overlap_loss.detach().item()
        progress_bar.set_postfix({'loss': wandb_log['loss']})
        progress_bar.update(1)

        if iteration % config['log_frequency'] == 0:
            with torch.no_grad():
                model.eval()

                val_batch = next(val_iterator)

                (
                    total_val_loss, val_first_step_heatmap_loss, val_later_step_loss, val_seg_loss, val_normals_loss, val_srf_overlap_loss,
                    val_inputs_for_vis, val_targets_for_vis, val_target_pred_for_vis,
                    val_seg_for_vis, val_seg_pred_for_vis, val_normals_for_vis, val_normals_pred_for_vis
                ) = compute_loss_and_pred(val_batch)

                wandb_log['val_loss'] = total_val_loss.item()
                wandb_log['val_first_step_heatmap_loss'] = val_first_step_heatmap_loss.item()
                if multistep_count > 1:
                    wandb_log['val_later_step_loss'] = val_later_step_loss.item()
                if use_seg:
                    wandb_log['val_seg_loss'] = val_seg_loss.item()
                if use_normals:
                    wandb_log['val_normals_loss'] = val_normals_loss.item()
                if use_srf_overlap and val_srf_overlap_loss is not None:
                    wandb_log['val_srf_overlap_loss'] = val_srf_overlap_loss.item()

                log_image_ext = config.get('log_image_ext', 'jpg')
                make_canvas(inputs_for_vis, targets_for_vis, target_pred_for_vis, config,
                           seg=seg_for_vis, seg_pred=seg_pred_for_vis, normals=normals_for_vis,
                           normals_pred=normals_pred_for_vis, normals_mask=None,
                           save_path=f'{out_dir}/{iteration:06}_train.{log_image_ext}')
                make_canvas(val_inputs_for_vis, val_targets_for_vis, val_target_pred_for_vis, config,
                           seg=val_seg_for_vis, seg_pred=val_seg_pred_for_vis, normals=val_normals_for_vis,
                           normals_pred=val_normals_pred_for_vis, normals_mask=None,
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


if __name__ == '__main__':
    train()
