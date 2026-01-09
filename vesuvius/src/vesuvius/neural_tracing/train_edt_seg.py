import os
import json
import click
import torch
import wandb
import random
import accelerate
import numpy as np
from tqdm import tqdm

from vesuvius.neural_tracing.dataset_edt_seg import EdtSegDataset
from vesuvius.neural_tracing.deep_supervision import _resize_for_ds, _compute_ds_weights
from vesuvius.models.training.loss.losses import SignedDistanceLoss
from vesuvius.models.training.loss.nnunet_losses import DeepSupervisionWrapper
from vesuvius.models.training.optimizers import create_optimizer
from vesuvius.models.training.lr_schedulers import get_scheduler
from vesuvius.neural_tracing.models import make_model

import multiprocessing
multiprocessing.set_start_method('spawn', force=True) # thread contention in edt package in the dataset makes fork use almost 100% of cpu 
                                                      # which slows training dramatically


def seed_worker(worker_id):
    """Seed worker for reproducibility."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def prepare_batch(batch):
    """Prepare batch tensors for training."""
    vol = batch['vol'].unsqueeze(1)    # [B, 1, D, H, W]
    cond = batch['cond'].unsqueeze(1)  # [B, 1, D, H, W]
    inputs = torch.cat([vol, cond], dim=1)  # [B, 2, D, H, W]

    seg_target = batch['seg']  # [B, D, H, W] binary
    dt_target = batch['dt'].unsqueeze(1)  # [B, 1, D, H, W]
    skel_target = batch['skel']  # [B, D, H, W] skeleton

    # Loss mask: compute seg loss only on unknown spatial half (plane_mask=0)
    # plane_mask=1 where conditioning is provided, =0 where model must predict
    loss_mask = 1.0 - batch['plane_mask']  # [B, D, H, W], 1=compute loss, 0=skip

    return inputs, seg_target, dt_target, skel_target, loss_mask


def make_visualization(inputs, dt_target, seg_pred, dt_pred, skel_target, plane_mask, save_path):
    """Create and save GIF visualization cycling through z-slices."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    b = 0
    D = inputs.shape[2]  # depth

    # Precompute 3D arrays
    vol_3d = inputs[b, 0].cpu().numpy()
    cond_3d = inputs[b, 1].cpu().numpy()
    skel_3d = skel_target[b].cpu().numpy()
    dt_gt_3d = dt_target[b, 0].cpu().numpy()
    mask_3d = plane_mask[b].cpu().numpy()
    seg_pred_3d = torch.argmax(seg_pred[b], dim=0).cpu().numpy()
    dt_pred_3d = dt_pred[b, 0].cpu().numpy()

    # Global min/max for consistent DT colormap
    dt_vmin = min(dt_gt_3d.min(), dt_pred_3d.min())
    dt_vmax = max(dt_gt_3d.max(), dt_pred_3d.max())

    # Setup figure
    fig = plt.figure(figsize=(16, 8))
    axes = [[None for _ in range(4)] for _ in range(2)]
    axes[0][0] = fig.add_subplot(2, 4, 1)
    axes[0][1] = fig.add_subplot(2, 4, 2, projection='3d')
    axes[0][2] = fig.add_subplot(2, 4, 3)
    axes[0][3] = fig.add_subplot(2, 4, 4)
    axes[1][0] = fig.add_subplot(2, 4, 5)
    axes[1][1] = fig.add_subplot(2, 4, 6)
    axes[1][2] = fig.add_subplot(2, 4, 7)
    axes[1][3] = fig.add_subplot(2, 4, 8)

    # 3D scatter plot (static) - GT skeleton: green=known, red=unknown
    skel_coords = np.argwhere(skel_3d > 0.5)
    if len(skel_coords) > 0:
        max_pts = 2000
        if len(skel_coords) > max_pts:
            idx = np.random.choice(len(skel_coords), max_pts, replace=False)
            skel_coords = skel_coords[idx]
        known_mask = mask_3d[skel_coords[:, 0], skel_coords[:, 1], skel_coords[:, 2]] > 0.5
        known_pts = skel_coords[known_mask]
        unknown_pts = skel_coords[~known_mask]
        ax3d = axes[0][1]
        if len(known_pts) > 0:
            ax3d.scatter(known_pts[:, 2], known_pts[:, 1], known_pts[:, 0],
                        c='green', s=1, alpha=0.6, label='known')
        if len(unknown_pts) > 0:
            ax3d.scatter(unknown_pts[:, 2], unknown_pts[:, 1], unknown_pts[:, 0],
                        c='red', s=1, alpha=0.6, label='unknown')
        ax3d.set_xlabel('X')
        ax3d.set_ylabel('Y')
        ax3d.set_zlabel('Z')
        ax3d.legend(loc='upper right', markerscale=4)
    axes[0][1].set_title('Skel GT: known(green) / unknown(red)')

    # Create image objects for animation
    z0 = D // 2
    vol_slice = vol_3d[z0]
    vol_norm = (vol_slice - vol_slice.min()) / (vol_slice.max() - vol_slice.min() + 1e-8)

    im_vol = axes[0][0].imshow(vol_slice, cmap='gray')
    axes[0][0].set_title(f'Volume (z={z0})')
    axes[0][0].axis('off')

    im_skel = axes[0][2].imshow(skel_3d[z0], cmap='gray')
    axes[0][2].set_title('Skel GT')
    axes[0][2].axis('off')

    mask_overlay = np.stack([vol_norm, vol_norm, vol_norm], axis=-1)
    im_mask = axes[0][3].imshow(mask_overlay)
    axes[0][3].set_title('Mask (green) + Cond (yellow)')
    axes[0][3].axis('off')

    overlay = np.stack([vol_norm, vol_norm, vol_norm], axis=-1)
    im_pred = axes[1][0].imshow(overlay)
    axes[1][0].set_title('Pred (red=unknown, blue=known)')
    axes[1][0].axis('off')

    im_dt_gt = axes[1][1].imshow(dt_gt_3d[z0], cmap='RdBu', vmin=dt_vmin, vmax=dt_vmax)
    axes[1][1].set_title('DT GT')
    axes[1][1].axis('off')

    im_dt_p = axes[1][2].imshow(dt_pred_3d[z0], cmap='RdBu', vmin=dt_vmin, vmax=dt_vmax)
    axes[1][2].set_title('DT Pred')
    axes[1][2].axis('off')

    im_seg_p = axes[1][3].imshow(seg_pred_3d[z0], cmap='gray')
    axes[1][3].set_title('Seg Pred')
    axes[1][3].axis('off')

    plt.tight_layout()

    def update(z):
        vol_slice = vol_3d[z]
        vol_norm = (vol_slice - vol_slice.min()) / (vol_slice.max() - vol_slice.min() + 1e-8)
        mask_slice = mask_3d[z]
        cond_slice = cond_3d[z]
        seg_p_slice = seg_pred_3d[z]

        # Volume
        im_vol.set_data(vol_slice)
        axes[0][0].set_title(f'Volume (z={z})')

        # Skel GT
        im_skel.set_data(skel_3d[z])

        # Mask + Cond overlay
        mask_overlay = np.stack([vol_norm, vol_norm, vol_norm], axis=-1)
        mask_overlay[mask_slice > 0.5, 1] = mask_overlay[mask_slice > 0.5, 1] * 0.5 + 0.5
        cond_pts = cond_slice > 0.5
        mask_overlay[cond_pts, 0] = 1.0
        mask_overlay[cond_pts, 1] = 1.0
        mask_overlay[cond_pts, 2] = 0.0
        im_mask.set_data(mask_overlay)

        # Pred overlay
        overlay = np.stack([vol_norm, vol_norm, vol_norm], axis=-1)
        pred_mask = seg_p_slice > 0.5
        unknown_pred = pred_mask & (mask_slice < 0.5)
        known_pred = pred_mask & (mask_slice > 0.5)
        overlay[unknown_pred, 0] = overlay[unknown_pred, 0] * 0.5 + 0.5
        overlay[unknown_pred, 1] = overlay[unknown_pred, 1] * 0.5
        overlay[unknown_pred, 2] = overlay[unknown_pred, 2] * 0.5
        overlay[known_pred, 2] = overlay[known_pred, 2] * 0.5 + 0.5
        im_pred.set_data(overlay)

        # DT
        im_dt_gt.set_data(dt_gt_3d[z])
        im_dt_p.set_data(dt_pred_3d[z])

        # Seg pred
        im_seg_p.set_data(seg_p_slice)

        return [im_vol, im_skel, im_mask, im_pred, im_dt_gt, im_dt_p, im_seg_p]

    # Sample every few slices for speed
    z_step = max(1, D // 32)
    z_frames = list(range(0, D, z_step))

    anim = FuncAnimation(fig, update, frames=z_frames, interval=150, blit=True)
    anim.save(save_path, writer='pillow', fps=6)
    plt.close(fig)


@click.command()
@click.argument('config_path', type=click.Path(exists=True))
def train(config_path):
    """Train an EDT segmentation model."""

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Defaults
    config.setdefault('in_channels', 2)
    config.setdefault('out_channels', 1)  # dt only
    config.setdefault('step_count', 1)  # Required by make_model
    config.setdefault('num_iterations', 250000)
    config.setdefault('log_frequency', 100)
    config.setdefault('ckpt_frequency', 5000)
    config.setdefault('grad_clip', 5)
    config.setdefault('learning_rate', 0.01)
    config.setdefault('weight_decay', 3e-5)
    config.setdefault('batch_size', 4)
    config.setdefault('num_workers', 4)
    config.setdefault('seed', 0)
    ds_enabled = config.setdefault('enable_deep_supervision', False)

    out_dir = config['out_dir']
    os.makedirs(out_dir, exist_ok=True)

    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])

    accelerator = accelerate.Accelerator(
        mixed_precision=config.get('mixed_precision', 'no'),
        gradient_accumulation_steps=config.get('grad_acc_steps', 1),
    )

    if 'wandb_project' in config and accelerator.is_main_process:
        wandb.init(
            project=config['wandb_project'],
            entity=config.get('wandb_entity', None),
            config=config
        )

    skel_threshold = config.get('skel_threshold', 0.5)  # Only used for visualization

    dt_loss_fn = SignedDistanceLoss(
        beta=config.get('sdt_beta', 1.0),
        eikonal=config.get('sdt_eikonal', True),
        eikonal_weight=config.get('sdt_eikonal_weight', 0.01),
        laplacian=config.get('sdt_laplacian', True),
        laplacian_weight=config.get('sdt_laplacian_weight', 0.01),
        surface_sigma=config.get('sdt_surface_sigma', None),  # Gaussian weight centered at d=0
        reduction='mean',
    )

    ds_cache = {
        'dt': {'weights': None, 'loss_fn': None},
    }

    def compute_loss_with_ds(pred, target, base_loss_fn, cache_key, resize_mode='trilinear', skel=None, loss_mask=None):
        """Compute loss with optional deep supervision, skeleton support, and loss masking."""
        def mean_loss_fn(p, t, s=None, mask=None):
            if s is not None:
                loss = base_loss_fn(p, t, s)
            else:
                loss = base_loss_fn(p, t)

            # Apply loss mask if provided (for spatial masking of loss)
            if mask is not None and loss.dim() > 0:
                # Resize mask to match loss shape if needed
                if mask.shape[2:] != loss.shape[2:]:
                    mask = torch.nn.functional.interpolate(
                        mask, size=loss.shape[2:], mode='nearest'
                    )
                # Masked mean: only count voxels where mask > 0
                masked_loss = loss * mask
                loss = masked_loss.sum() / mask.sum().clamp(min=1e-6)
            elif loss.dim() > 0:
                loss = loss.mean()
            return loss

        if ds_enabled and isinstance(pred, (list, tuple)):
            cache = ds_cache[cache_key]
            if cache['weights'] is None or len(cache['weights']) != len(pred):
                cache['weights'] = _compute_ds_weights(len(pred))
                cache['loss_fn'] = DeepSupervisionWrapper(lambda p, t: mean_loss_fn(p, t), cache['weights'])
            elif cache['loss_fn'] is None:
                cache['loss_fn'] = DeepSupervisionWrapper(lambda p, t: mean_loss_fn(p, t), cache['weights'])
            align = False if resize_mode == 'nearest' else False
            targets_resized = [_resize_for_ds(target, p.shape[2:], mode=resize_mode, align_corners=align if resize_mode != 'nearest' else None) for p in pred]
            # Deep supervision with skeleton not yet supported - use first scale only for skeleton loss
            if skel is not None:
                # For skeleton loss, compute on full resolution only with mask
                loss = mean_loss_fn(pred[0], targets_resized[0], skel, loss_mask)
            else:
                loss = cache['loss_fn'](pred, targets_resized)
            pred_for_vis = pred[0]
        else:
            if isinstance(pred, (list, tuple)):
                pred = pred[0]
            loss = mean_loss_fn(pred, target, skel, loss_mask)
            pred_for_vis = pred
        return loss, pred_for_vis

    def make_generator(offset=0):
        gen = torch.Generator()
        gen.manual_seed(config['seed'] + accelerator.process_index * 1000 + offset)
        return gen

    # train with augmentation, val without
    train_dataset = EdtSegDataset(config, apply_augmentation=True)
    val_dataset = EdtSegDataset(config, apply_augmentation=False)

    # Train/val split by indices
    num_patches = len(train_dataset)
    num_val = max(1, int(num_patches * config.get('val_fraction', 0.1)))
    num_train = num_patches - num_val

    indices = torch.randperm(num_patches, generator=torch.Generator().manual_seed(config['seed'])).tolist()
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]

    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(val_dataset, val_indices)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        worker_init_fn=seed_worker,
        generator=make_generator(0),
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=1,
        worker_init_fn=seed_worker,
        generator=make_generator(1),
    )

    model = make_model(config)

    if config.get('compile_model', True):
        model = torch.compile(model)
        if accelerator.is_main_process:
            accelerator.print("Model compiled with torch.compile")

    optimizer = create_optimizer({
        'name': 'sgd',
        'learning_rate': config['learning_rate'],
        'momentum': config.get('momentum', 0.99),
        'nesterov': config.get('nesterov', True),
        'weight_decay': config['weight_decay'],
    }, model)

    lr_scheduler = get_scheduler(
        scheduler_type='poly',
        optimizer=optimizer,
        initial_lr=config['learning_rate'],
        max_steps=config['num_iterations'],
        exponent=config.get('poly_exponent', 0.9),
    )

    start_iteration = 0
    if 'load_ckpt' in config:
        print(f'Loading checkpoint {config["load_ckpt"]}')
        ckpt = torch.load(config['load_ckpt'], map_location='cpu', weights_only=True)
        state_dict = ckpt['model']
        # Handle compiled model state dict
        if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            model_keys = set(model.state_dict().keys())
            if not any(k.startswith('_orig_mod.') for k in model_keys):
                state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
                print('Stripped _orig_mod. prefix from checkpoint state dict')
        model.load_state_dict(state_dict)
        start_iteration = ckpt.get('step', 0)

    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )

    if accelerator.is_main_process:
        accelerator.print("\n=== DT-Only Training Configuration ===")
        accelerator.print(f"Input channels: {config['in_channels']}")
        accelerator.print(f"Output channels: {config['out_channels']} (dt only)")
        accelerator.print(f"Optimizer: SGD (lr={config['learning_rate']}, momentum={config.get('momentum', 0.99)})")
        accelerator.print(f"Scheduler: PolyLR (exponent={config.get('poly_exponent', 0.9)})")
        accelerator.print(f"Train samples: {num_train}, Val samples: {num_val}")
        accelerator.print("=======================================\n")

    val_iterator = iter(val_dataloader)
    train_iterator = iter(train_dataloader)
    grad_clip = config['grad_clip']

    progress_bar = tqdm(
        total=config['num_iterations'],
        initial=start_iteration,
        disable=not accelerator.is_local_main_process
    )

    for iteration in range(start_iteration, config['num_iterations']):

        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_dataloader)
            batch = next(train_iterator)

        inputs, seg_target, dt_target, skel_target, loss_mask = prepare_batch(batch)

        wandb_log = {}

        with accelerator.accumulate(model):
            # Forward pass
            output = model(inputs)
            dt_pred = output['dt']  # [B, 1, D, H, W] or list for DS

            # Distance transform loss (NO masking - compute everywhere)
            dt_loss, dt_pred_for_vis = compute_loss_with_ds(
                dt_pred, dt_target, dt_loss_fn, 'dt', resize_mode='trilinear'
            )

            total_loss = dt_loss

            if torch.isnan(total_loss).any():
                raise ValueError('loss is NaN')

            accelerator.backward(total_loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        wandb_log['loss'] = total_loss.detach().item()
        wandb_log['dt_loss'] = dt_loss.detach().item()
        wandb_log['lr'] = optimizer.param_groups[0]['lr']

        progress_bar.set_postfix({
            'loss': f"{wandb_log['loss']:.4f}",
            'dt': f"{wandb_log['dt_loss']:.4f}",
        })
        progress_bar.update(1)

        if iteration % config['log_frequency'] == 0 and accelerator.is_main_process:
            with torch.no_grad():
                model.eval()

                try:
                    val_batch = next(val_iterator)
                except StopIteration:
                    val_iterator = iter(val_dataloader)
                    val_batch = next(val_iterator)

                val_inputs, val_seg_target, val_dt_target, val_skel_target, val_loss_mask = prepare_batch(val_batch)

                val_output = model(val_inputs)
                val_dt_pred = val_output['dt']

                val_dt_loss, val_dt_pred_for_vis = compute_loss_with_ds(
                    val_dt_pred, val_dt_target, dt_loss_fn, 'dt', resize_mode='trilinear'
                )

                # Derive seg from thresholded DT for visualization
                val_seg_from_dt = (val_dt_pred_for_vis.abs() < skel_threshold).float()
                val_seg_for_skel = torch.cat([1 - val_seg_from_dt, val_seg_from_dt], dim=1)

                wandb_log['val_loss'] = val_dt_loss.item()
                wandb_log['val_dt_loss'] = val_dt_loss.item()

                # Create visualization (saved as GIF)
                train_img_path = f'{out_dir}/{iteration:06}_train.gif'
                val_img_path = f'{out_dir}/{iteration:06}_val.gif'

                # plane_mask: 1=known/conditioned, 0=unknown (inverse of loss_mask)
                train_plane_mask = 1.0 - loss_mask
                val_plane_mask = 1.0 - val_loss_mask

                # Derive seg from thresholded DT for visualization
                train_seg_from_dt = (dt_pred_for_vis.abs() < skel_threshold).float()
                train_seg_for_vis = torch.cat([1 - train_seg_from_dt, train_seg_from_dt], dim=1)
                val_seg_for_vis = val_seg_for_skel  # Already computed above

                make_visualization(
                    inputs, dt_target, train_seg_for_vis, dt_pred_for_vis, skel_target,
                    train_plane_mask, train_img_path
                )
                make_visualization(
                    val_inputs, val_dt_target, val_seg_for_vis, val_dt_pred_for_vis, val_skel_target,
                    val_plane_mask, val_img_path
                )

                if wandb.run is not None:
                    wandb_log['train_image'] = wandb.Video(train_img_path, fps=6, format='gif')
                    wandb_log['val_image'] = wandb.Video(val_img_path, fps=6, format='gif')

                model.train()

        if iteration % config['ckpt_frequency'] == 0 and accelerator.is_main_process:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
                'step': iteration,
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
        }, f'{out_dir}/ckpt_final.pth')


if __name__ == '__main__':
    train()
