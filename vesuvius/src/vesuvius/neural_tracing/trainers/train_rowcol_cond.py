"""
Trainer for row/col conditioned displacement field prediction.

Trains a model to predict dense 3D displacement fields from extrapolated surfaces,
with optional SDT (Signed Distance Transform) prediction.
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
from vesuvius.neural_tracing.loss.displacement_losses import surface_sampled_loss
from vesuvius.models.training.loss.nnunet_losses import DC_and_BCE_loss
from vesuvius.models.training.optimizers import create_optimizer
from vesuvius.models.training.lr_schedulers import get_scheduler
from vesuvius.neural_tracing.models import make_model
from accelerate.utils import TorchDynamoPlugin

import multiprocessing
multiprocessing.set_start_method('spawn', force=True)


def seed_worker(worker_id):
    """Seed worker for reproducibility."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def collate_with_padding(batch):
    """Collate batch with padding for variable-length point data."""
    # Stack fixed-size tensors normally
    vol = torch.stack([b['vol'] for b in batch])
    cond = torch.stack([b['cond'] for b in batch])
    extrap_surface = torch.stack([b['extrap_surface'] for b in batch])

    # Pad variable-length data
    coords_list = [b['extrap_coords'] for b in batch]
    disp_list = [b['gt_displacement'] for b in batch]
    max_points = max(len(c) for c in coords_list)

    B = len(batch)
    padded_coords = torch.zeros(B, max_points, 3)
    padded_disp = torch.zeros(B, max_points, 3)
    valid_mask = torch.zeros(B, max_points)

    for i, (c, d) in enumerate(zip(coords_list, disp_list)):
        n = len(c)
        padded_coords[i, :n] = c
        padded_disp[i, :n] = d
        valid_mask[i, :n] = 1.0

    result = {
        'vol': vol, 'cond': cond, 'extrap_surface': extrap_surface,
        'extrap_coords': padded_coords, 'gt_displacement': padded_disp,
        'valid_mask': valid_mask
    }

    # Optional SDT
    if 'sdt' in batch[0]:
        result['sdt'] = torch.stack([b['sdt'] for b in batch])

    # Optional heatmap target
    if 'heatmap_target' in batch[0]:
        result['heatmap_target'] = torch.stack([b['heatmap_target'] for b in batch])

    # Optional other_wraps
    if 'other_wraps' in batch[0]:
        result['other_wraps'] = torch.stack([b['other_wraps'] for b in batch])

    return result


def prepare_batch(batch, use_sdt=False, use_heatmap=False):
    """Prepare batch tensors for training."""
    vol = batch['vol'].unsqueeze(1)                    # [B, 1, D, H, W]
    cond = batch['cond'].unsqueeze(1)                  # [B, 1, D, H, W]
    extrap_surf = batch['extrap_surface'].unsqueeze(1) # [B, 1, D, H, W]

    input_list = [vol, cond, extrap_surf]
    if 'other_wraps' in batch:
        other_wraps = batch['other_wraps'].unsqueeze(1)  # [B, 1, D, H, W]
        input_list.append(other_wraps)

    inputs = torch.cat(input_list, dim=1)  # [B, 3 or 4, D, H, W]

    extrap_coords = batch['extrap_coords']       # [B, N, 3]
    gt_displacement = batch['gt_displacement']   # [B, N, 3]
    valid_mask = batch['valid_mask']             # [B, N]

    sdt_target = batch['sdt'].unsqueeze(1) if use_sdt and 'sdt' in batch else None  # [B, 1, D, H, W]
    heatmap_target = batch['heatmap_target'].unsqueeze(1) if use_heatmap and 'heatmap_target' in batch else None  # [B, 1, D, H, W]

    return inputs, extrap_coords, gt_displacement, valid_mask, sdt_target, heatmap_target


def make_visualization(inputs, disp_pred, extrap_coords, gt_displacement, valid_mask,
                       sdt_pred=None, sdt_target=None,
                       heatmap_pred=None, heatmap_target=None,
                       save_path=None):
    """Create and save PNG visualization of middle z-slice."""
    import matplotlib.pyplot as plt

    b = 0
    D = inputs.shape[2]  # depth

    # Precompute 3D arrays
    vol_3d = inputs[b, 0].cpu().numpy()
    cond_3d = inputs[b, 1].cpu().numpy()
    extrap_surf_3d = inputs[b, 2].cpu().numpy()
    other_wraps_3d = inputs[b, 3].cpu().numpy() if inputs.shape[1] > 3 else None

    # Displacement magnitude
    disp_3d = disp_pred[b].cpu().numpy()  # [3, D, H, W]
    disp_mag_3d = np.linalg.norm(disp_3d, axis=0)  # [D, H, W]

    # Setup figure - add columns for heatmap if present
    n_cols = 4
    if sdt_pred is not None:
        n_cols += 1
    if heatmap_pred is not None:
        n_cols += 1
    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))

    z0 = D // 2
    vol_slice = vol_3d[z0]
    vol_norm = (vol_slice - vol_slice.min()) / (vol_slice.max() - vol_slice.min() + 1e-8)

    # Row 0: Volume, Cond, Extrap surface, Displacement magnitude
    im_vol = axes[0, 0].imshow(vol_slice, cmap='gray')
    axes[0, 0].set_title(f'Volume (z={z0})')
    axes[0, 0].axis('off')

    im_cond = axes[0, 1].imshow(cond_3d[z0], cmap='gray')
    axes[0, 1].set_title('Conditioning')
    axes[0, 1].axis('off')

    im_extrap = axes[0, 2].imshow(extrap_surf_3d[z0], cmap='gray')
    axes[0, 2].set_title('Extrap Surface')
    axes[0, 2].axis('off')

    disp_vmax = np.percentile(disp_mag_3d, 99)
    im_disp_mag = axes[0, 3].imshow(disp_mag_3d[z0], cmap='hot', vmin=0, vmax=disp_vmax)
    axes[0, 3].set_title('Disp Magnitude')
    axes[0, 3].axis('off')

    # Row 1: Displacement components (dz, dy, dx) + optional SDT
    disp_vmax_comp = np.percentile(np.abs(disp_3d), 99)
    im_dz = axes[1, 0].imshow(disp_3d[0, z0], cmap='RdBu', vmin=-disp_vmax_comp, vmax=disp_vmax_comp)
    axes[1, 0].set_title('dz (pred)')
    axes[1, 0].axis('off')

    im_dy = axes[1, 1].imshow(disp_3d[1, z0], cmap='RdBu', vmin=-disp_vmax_comp, vmax=disp_vmax_comp)
    axes[1, 1].set_title('dy (pred)')
    axes[1, 1].axis('off')

    im_dx = axes[1, 2].imshow(disp_3d[2, z0], cmap='RdBu', vmin=-disp_vmax_comp, vmax=disp_vmax_comp)
    axes[1, 2].set_title('dx (pred)')
    axes[1, 2].axis('off')

    # Overlay: cond + extrap + other_wraps
    overlay = np.stack([vol_norm, vol_norm, vol_norm], axis=-1)
    cond_pts = cond_3d[z0] > 0.5
    extrap_pts = extrap_surf_3d[z0] > 0.5
    overlay[cond_pts, 1] = 1.0  # green for conditioning
    overlay[extrap_pts, 0] = 1.0  # red for extrapolated
    if other_wraps_3d is not None:
        other_pts = other_wraps_3d[z0] > 0.5
        overlay[other_pts, 2] = 1.0  # blue for other wraps
    im_overlay = axes[1, 3].imshow(overlay)
    title = 'Cond(G) + Extrap(R)' + (' + Other(B)' if other_wraps_3d is not None else '')
    axes[1, 3].set_title(title)
    axes[1, 3].axis('off')

    # Track current column for optional visualizations
    col_idx = 4

    # Optional SDT
    im_sdt_pred = im_sdt_gt = None
    sdt_pred_3d = sdt_gt_3d = None
    sdt_col = None
    if sdt_pred is not None:
        sdt_col = col_idx
        sdt_pred_3d = sdt_pred[b, 0].cpu().numpy()
        sdt_gt_3d = sdt_target[b, 0].cpu().numpy() if sdt_target is not None else np.zeros_like(sdt_pred_3d)
        sdt_vmax = max(np.abs(sdt_pred_3d).max(), np.abs(sdt_gt_3d).max())

        im_sdt_pred = axes[0, sdt_col].imshow(sdt_pred_3d[z0], cmap='RdBu', vmin=-sdt_vmax, vmax=sdt_vmax)
        axes[0, sdt_col].set_title('SDT Pred')
        axes[0, sdt_col].axis('off')

        im_sdt_gt = axes[1, sdt_col].imshow(sdt_gt_3d[z0], cmap='RdBu', vmin=-sdt_vmax, vmax=sdt_vmax)
        axes[1, sdt_col].set_title('SDT GT')
        axes[1, sdt_col].axis('off')
        col_idx += 1

    # Optional Heatmap
    im_hm_pred = im_hm_gt = None
    hm_pred_3d = hm_gt_3d = None
    hm_col = None
    if heatmap_pred is not None:
        hm_col = col_idx
        hm_pred_3d = torch.sigmoid(heatmap_pred[b, 0]).cpu().numpy()
        hm_gt_3d = heatmap_target[b, 0].cpu().numpy() if heatmap_target is not None else np.zeros_like(hm_pred_3d)

        im_hm_pred = axes[0, hm_col].imshow(hm_pred_3d[z0], cmap='hot', vmin=0, vmax=1)
        axes[0, hm_col].set_title('Heatmap Pred')
        axes[0, hm_col].axis('off')

        im_hm_gt = axes[1, hm_col].imshow(hm_gt_3d[z0], cmap='hot', vmin=0, vmax=1)
        axes[1, hm_col].set_title('Heatmap GT')
        axes[1, hm_col].axis('off')
        col_idx += 1

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


@click.command()
@click.argument('config_path', type=click.Path(exists=True))
def train(config_path):
    """Train a displacement field prediction model with optional SDT."""

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Defaults
    config.setdefault('in_channels', 3)  # vol + cond + extrap_surface
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
    config.setdefault('use_sdt', False)
    config.setdefault('lambda_sdt', 1.0)
    config.setdefault('use_heatmap_targets', False)
    config.setdefault('lambda_heatmap', 1.0)
    config.setdefault('displacement_loss_type', 'vector_l2')

    # Build targets dict based on config
    targets = {
        'displacement': {'out_channels': 3, 'activation': 'none'}
    }
    use_sdt = config.get('use_sdt', False)
    if use_sdt:
        targets['sdt'] = {'out_channels': 1, 'activation': 'none'}
    use_heatmap = config.get('use_heatmap_targets', False)
    if use_heatmap:
        targets['heatmap'] = {'out_channels': 1, 'activation': 'none'}
    config['targets'] = targets

    out_dir = config['out_dir']
    os.makedirs(out_dir, exist_ok=True)

    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])

    dynamo_plugin = TorchDynamoPlugin(
            backend="inductor",  # Options: "inductor", "aot_eager", "aot_nvfuser", etc.
            mode="default",      # Options: "default", "reduce-overhead", "max-autotune"
            fullgraph=False,
            dynamic=False,
            use_regional_compilation=False
        )
    

    accelerator = accelerate.Accelerator(
        mixed_precision=config.get('mixed_precision', 'no'),
        gradient_accumulation_steps=config.get('grad_acc_steps', 1),
        dynamo_plugin=dynamo_plugin
    )

    if 'wandb_project' in config and accelerator.is_main_process:
        wandb.init(
            project=config['wandb_project'],
            entity=config.get('wandb_entity', None),
            config=config
        )

    # Setup SDT loss if enabled
    sdt_loss_fn = None
    if use_sdt:
        from vesuvius.models.training.loss.losses import SignedDistanceLoss
        sdt_loss_fn = SignedDistanceLoss(
            beta=config.get('sdt_beta', 1.0),
            eikonal=config.get('sdt_eikonal', True),
            eikonal_weight=config.get('sdt_eikonal_weight', 0.01),
            laplacian=config.get('sdt_laplacian', True),
            laplacian_weight=config.get('sdt_laplacian_weight', 0.01),
            surface_sigma=config.get('sdt_surface_sigma', 3.0),
            reduction='mean',
        )

    lambda_sdt = config.get('lambda_sdt', 1.0)
    lambda_heatmap = config.get('lambda_heatmap', 1.0)
    disp_loss_type = config.get('displacement_loss_type', 'vector_l2')

    # Setup heatmap loss if enabled (BCE + Dice)
    heatmap_loss_fn = None
    if use_heatmap:
        heatmap_loss_fn = DC_and_BCE_loss(
            bce_kwargs={},
            soft_dice_kwargs={'batch_dice': False, 'ddp': False},
            weight_ce=1.0,
            weight_dice=1.0
        )

    def make_generator(offset=0):
        gen = torch.Generator()
        gen.manual_seed(config['seed'] + accelerator.process_index * 1000 + offset)
        return gen

    # Train with augmentation, val without
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
        collate_fn=collate_with_padding,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=1,
        worker_init_fn=seed_worker,
        generator=make_generator(1),
        collate_fn=collate_with_padding,
    )

    model = make_model(config)

    if config.get('compile_model', True):
        model = torch.compile(model)
        if accelerator.is_main_process:
            accelerator.print("Model compiled with torch.compile")

    optimizer = create_optimizer({
        'name': 'adamw',
        'learning_rate': config.get('learning_rate', 1e-3),
        'weight_decay': config.get('weight_decay', 1e-4),
    }, model)

    lr_scheduler = get_scheduler(
        scheduler_type='diffusers_cosine_warmup',
        optimizer=optimizer,
        initial_lr=config.get('learning_rate', 1e-3),
        max_steps=config['num_iterations'],
        warmup_steps=config.get('warmup_steps', 5000),
    )

    start_iteration = 0
    if 'load_ckpt' in config:
        print(f'Loading checkpoint {config["load_ckpt"]}')
        ckpt = torch.load(config['load_ckpt'], map_location='cpu', weights_only=False)
        state_dict = ckpt['model']
        # Handle compiled model state dict
        if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            model_keys = set(model.state_dict().keys())
            if not any(k.startswith('_orig_mod.') for k in model_keys):
                state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
                print('Stripped _orig_mod. prefix from checkpoint state dict')
        model.load_state_dict(state_dict)

        if not config.get('load_weights_only', False):
            start_iteration = ckpt.get('step', 0)
            # Load optimizer state if optimizer type matches (SGD vs Adam check via betas)
            ckpt_optim_type = type(ckpt['optimizer']['param_groups'][0].get('betas', None))
            curr_optim_type = type(optimizer.param_groups[0].get('betas', None))
            if ckpt_optim_type == curr_optim_type:
                optimizer.load_state_dict(ckpt['optimizer'])
                print('Loaded optimizer state (momentum preserved)')
            else:
                print('Skipping optimizer state load (optimizer type changed)')

    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )

    if accelerator.is_main_process:
        accelerator.print("\n=== Displacement Field Training Configuration ===")
        accelerator.print(f"Input channels: {config['in_channels']}")
        accelerator.print(f"Displacement loss: {disp_loss_type}")
        output_str = "Output: displacement (3ch)"
        if use_sdt:
            output_str += " + SDT (1ch)"
        if use_heatmap:
            output_str += " + heatmap (1ch)"
        accelerator.print(output_str)
        if use_sdt:
            accelerator.print(f"Lambda SDT: {lambda_sdt}")
        if use_heatmap:
            accelerator.print(f"Lambda heatmap: {lambda_heatmap}")
        accelerator.print(f"Optimizer: AdamW (lr={config.get('learning_rate', 1e-3)})")
        accelerator.print(f"Scheduler: diffusers_cosine_warmup (warmup={config.get('warmup_steps', 5000)})")
        accelerator.print(f"Train samples: {num_train}, Val samples: {num_val}")
        accelerator.print("=================================================\n")

    if config['verbose']:
            print("creating iterators...")
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
            print(f"starting iteration {iteration}")
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_dataloader)
            batch = next(train_iterator)

        if config['verbose']:
            print(f"got batch, keys: {batch.keys()}")

        inputs, extrap_coords, gt_displacement, valid_mask, sdt_target, heatmap_target = prepare_batch(batch, use_sdt, use_heatmap)

        wandb_log = {}

        with accelerator.accumulate(model):
            # Forward pass
            output = model(inputs)
            disp_pred = output['displacement']  # [B, 3, D, H, W]

            # Displacement loss
            surf_loss = surface_sampled_loss(disp_pred, extrap_coords, gt_displacement, valid_mask,
                                             loss_type=disp_loss_type)
            total_loss = surf_loss

            wandb_log['surf_loss'] = surf_loss.detach().item()

            # Optional SDT loss
            if use_sdt:
                sdt_pred = output['sdt']  # [B, 1, D, H, W]
                sdt_loss = sdt_loss_fn(sdt_pred, sdt_target)
                total_loss = total_loss + lambda_sdt * sdt_loss
                wandb_log['sdt_loss'] = sdt_loss.detach().item()

            # Optional heatmap loss (BCE + Dice)
            heatmap_pred = None
            if use_heatmap:
                heatmap_pred = output['heatmap']  # [B, 1, D, H, W]
                heatmap_target_binary = (heatmap_target > 0.5).float()
                heatmap_loss = heatmap_loss_fn(heatmap_pred, heatmap_target_binary)
                total_loss = total_loss + lambda_heatmap * heatmap_loss
                wandb_log['heatmap_loss'] = heatmap_loss.detach().item()

            if torch.isnan(total_loss).any():
                raise ValueError('loss is NaN')

            accelerator.backward(total_loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        wandb_log['loss'] = total_loss.detach().item()
        wandb_log['lr'] = optimizer.param_groups[0]['lr']

        postfix = {
            'loss': f"{wandb_log['loss']:.4f}",
            'surf': f"{wandb_log['surf_loss']:.4f}",
        }
        if use_sdt:
            postfix['sdt'] = f"{wandb_log['sdt_loss']:.4f}"
        if use_heatmap:
            postfix['hm'] = f"{wandb_log['heatmap_loss']:.4f}"
        progress_bar.set_postfix(postfix)
        progress_bar.update(1)

        if iteration % config['log_frequency'] == 0 and accelerator.is_main_process:
            with torch.no_grad():
                model.eval()

                try:
                    val_batch = next(val_iterator)
                except StopIteration:
                    val_iterator = iter(val_dataloader)
                    val_batch = next(val_iterator)

                val_inputs, val_extrap_coords, val_gt_displacement, val_valid_mask, val_sdt_target, val_heatmap_target = prepare_batch(val_batch, use_sdt, use_heatmap)

                val_output = model(val_inputs)
                val_disp_pred = val_output['displacement']

                val_surf_loss = surface_sampled_loss(val_disp_pred, val_extrap_coords, val_gt_displacement, val_valid_mask,
                                                     loss_type=disp_loss_type)
                val_total_loss = val_surf_loss

                wandb_log['val_surf_loss'] = val_surf_loss.item()

                val_sdt_pred = None
                if use_sdt:
                    val_sdt_pred = val_output['sdt']
                    val_sdt_loss = sdt_loss_fn(val_sdt_pred, val_sdt_target)
                    val_total_loss = val_total_loss + lambda_sdt * val_sdt_loss
                    wandb_log['val_sdt_loss'] = val_sdt_loss.item()

                val_heatmap_pred = None
                if use_heatmap:
                    val_heatmap_pred = val_output['heatmap']
                    val_heatmap_target_binary = (val_heatmap_target > 0.5).float()
                    val_heatmap_loss = heatmap_loss_fn(val_heatmap_pred, val_heatmap_target_binary)
                    val_total_loss = val_total_loss + lambda_heatmap * val_heatmap_loss
                    wandb_log['val_heatmap_loss'] = val_heatmap_loss.item()

                wandb_log['val_loss'] = val_total_loss.item()

                # Create visualization
                train_img_path = f'{out_dir}/{iteration:06}_train.png'
                val_img_path = f'{out_dir}/{iteration:06}_val.png'

                train_sdt_pred = output.get('sdt') if use_sdt else None
                train_heatmap_pred = heatmap_pred if use_heatmap else None
                make_visualization(
                    inputs, disp_pred, extrap_coords, gt_displacement, valid_mask,
                    sdt_pred=train_sdt_pred, sdt_target=sdt_target,
                    heatmap_pred=train_heatmap_pred, heatmap_target=heatmap_target,
                    save_path=train_img_path
                )
                make_visualization(
                    val_inputs, val_disp_pred, val_extrap_coords, val_gt_displacement, val_valid_mask,
                    sdt_pred=val_sdt_pred, sdt_target=val_sdt_target,
                    heatmap_pred=val_heatmap_pred, heatmap_target=val_heatmap_target,
                    save_path=val_img_path
                )

                if wandb.run is not None:
                    wandb_log['train_image'] = wandb.Image(train_img_path)
                    wandb_log['val_image'] = wandb.Image(val_img_path)

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
