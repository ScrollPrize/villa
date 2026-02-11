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
import copy
import random
import accelerate
import numpy as np
from tqdm import tqdm

from vesuvius.neural_tracing.datasets.dataset_rowcol_cond import EdtSegDataset
from vesuvius.neural_tracing.loss.displacement_losses import (
    surface_sampled_loss,
    surface_sampled_normal_loss,
    smoothness_loss,
)
from vesuvius.models.training.loss.nnunet_losses import DC_and_BCE_loss
from vesuvius.models.training.loss.skeleton_recall import DC_SkelREC_and_CE_loss
from vesuvius.models.training.optimizers import create_optimizer
from vesuvius.models.training.lr_schedulers import get_scheduler
from vesuvius.neural_tracing.models import make_model
from vesuvius.neural_tracing.trainers.rowcol_cond_visualization import make_visualization
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
    weight_list = [
        b['point_weights'] if 'point_weights' in b else torch.ones(len(b['extrap_coords']), dtype=torch.float32)
        for b in batch
    ]
    max_points = max(len(c) for c in coords_list)

    B = len(batch)
    padded_coords = torch.zeros(B, max_points, 3)
    padded_disp = torch.zeros(B, max_points, 3)
    valid_mask = torch.zeros(B, max_points)
    padded_point_weights = torch.zeros(B, max_points)
    padded_point_normals = torch.zeros(B, max_points, 3)
    has_point_normals = 'point_normals' in batch[0]

    for i, (c, d, w) in enumerate(zip(coords_list, disp_list, weight_list)):
        n = len(c)
        padded_coords[i, :n] = c
        padded_disp[i, :n] = d
        valid_mask[i, :n] = 1.0
        padded_point_weights[i, :n] = w
        if has_point_normals:
            padded_point_normals[i, :n] = batch[i]['point_normals']

    result = {
        'vol': vol, 'cond': cond, 'extrap_surface': extrap_surface,
        'extrap_coords': padded_coords, 'gt_displacement': padded_disp,
        'valid_mask': valid_mask, 'point_weights': padded_point_weights
    }
    if has_point_normals:
        result['point_normals'] = padded_point_normals

    # Optional SDT
    if 'sdt' in batch[0]:
        result['sdt'] = torch.stack([b['sdt'] for b in batch])

    # Optional heatmap target
    if 'heatmap_target' in batch[0]:
        result['heatmap_target'] = torch.stack([b['heatmap_target'] for b in batch])

    # Optional segmentation target (full segmentation + skeleton)
    if 'segmentation' in batch[0]:
        result['segmentation'] = torch.stack([b['segmentation'] for b in batch])
        result['segmentation_skel'] = torch.stack([b['segmentation_skel'] for b in batch])

    # Optional other_wraps
    if 'other_wraps' in batch[0]:
        result['other_wraps'] = torch.stack([b['other_wraps'] for b in batch])

    return result


def prepare_batch(batch, use_sdt=False, use_heatmap=False, use_segmentation=False):
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
    point_weights = batch['point_weights'] if 'point_weights' in batch else torch.ones_like(valid_mask)  # [B, N]
    point_normals = batch.get('point_normals', None)  # [B, N, 3] or None

    sdt_target = batch['sdt'].unsqueeze(1) if use_sdt and 'sdt' in batch else None  # [B, 1, D, H, W]
    heatmap_target = batch['heatmap_target'].unsqueeze(1) if use_heatmap and 'heatmap_target' in batch else None  # [B, 1, D, H, W]

    seg_target = None
    seg_skel = None
    if use_segmentation and 'segmentation' in batch:
        seg_target = batch['segmentation'].unsqueeze(1)  # [B, 1, D, H, W]
        seg_skel = batch['segmentation_skel'].unsqueeze(1)  # [B, 1, D, H, W]

    return inputs, extrap_coords, gt_displacement, valid_mask, point_weights, point_normals, sdt_target, heatmap_target, seg_target, seg_skel


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
    config.setdefault('use_segmentation', False)
    config.setdefault('lambda_segmentation', 1.0)
    config.setdefault('segmentation_loss', {})
    config.setdefault('supervise_conditioning', False)
    config.setdefault('cond_supervision_weight', 0.1)
    config.setdefault('lambda_cond_disp', 0.0)
    config.setdefault('displacement_supervision', 'vector')  # 'vector' or 'normal_scalar'
    config.setdefault('displacement_loss_type', 'vector_l2')
    config.setdefault('displacement_huber_beta', 5.0)
    config.setdefault('normal_loss_type', 'normal_huber')
    config.setdefault('normal_loss_beta', config.get('displacement_huber_beta', 5.0))
    config.setdefault('lambda_smooth', 0.0)
    config.setdefault('eval_perturbed_val', False)
    config.setdefault('log_perturbed_val_images', False)
    config.setdefault('val_batches_per_log', 25)

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
    use_segmentation = config.get('use_segmentation', False)
    if use_segmentation:
        targets['segmentation'] = {'out_channels': 2, 'activation': 'none'}
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
    lambda_segmentation = config.get('lambda_segmentation', 1.0)
    lambda_cond_disp = config.get('lambda_cond_disp', 0.0)
    lambda_smooth = config.get('lambda_smooth', 0.0)
    if config.get('supervise_conditioning', False) and lambda_cond_disp > 0.0:
        raise ValueError(
            "supervise_conditioning=True adds nonzero conditioning displacement targets, "
            "which conflicts with lambda_cond_disp > 0 (zero-displacement penalty on conditioning voxels). "
            "Set lambda_cond_disp to 0 when supervise_conditioning is enabled."
        )
    mask_cond_from_seg_loss = config.get('mask_cond_from_seg_loss', False)
    disp_supervision = str(config.get('displacement_supervision', 'vector')).lower()
    if disp_supervision not in {'vector', 'normal_scalar'}:
        raise ValueError(
            "displacement_supervision must be 'vector' or 'normal_scalar', "
            f"got {disp_supervision!r}"
        )
    if disp_supervision == 'normal_scalar' and not config.get('use_extrapolation', True):
        raise ValueError("displacement_supervision='normal_scalar' requires use_extrapolation=True")
    disp_loss_type = config.get('displacement_loss_type', 'vector_l2')
    disp_huber_beta = config.get('displacement_huber_beta', 5.0)
    normal_loss_type = str(config.get('normal_loss_type', 'normal_huber')).lower()
    normal_loss_beta = float(config.get('normal_loss_beta', disp_huber_beta))
    if normal_loss_type in {'huber', 'l2', 'l1'}:
        normal_loss_type = f'normal_{normal_loss_type}'

    # Setup heatmap loss if enabled (BCE + Dice)
    heatmap_loss_fn = None
    if use_heatmap:
        heatmap_loss_fn = DC_and_BCE_loss(
            bce_kwargs={},
            soft_dice_kwargs={'batch_dice': False, 'ddp': False},
            weight_ce=1.0,
            weight_dice=1.0
        )

    # Setup segmentation loss if enabled (MedialSurfaceRecall)
    seg_loss_fn = None
    if use_segmentation:
        seg_loss_cfg = config.get('segmentation_loss', {})
        soft_dice_kwargs = {
            'batch_dice': seg_loss_cfg.get('batch_dice', False),
            'smooth': seg_loss_cfg.get('smooth', 1e-5),
            'do_bg': seg_loss_cfg.get('do_bg', False),
            'ddp': seg_loss_cfg.get('ddp', False),
        }
        if 'soft_dice_kwargs' in seg_loss_cfg:
            soft_dice_kwargs.update(seg_loss_cfg['soft_dice_kwargs'])
        seg_loss_fn = DC_SkelREC_and_CE_loss(
            soft_dice_kwargs=soft_dice_kwargs,
            soft_skelrec_kwargs={
                'batch_dice': soft_dice_kwargs.get('batch_dice'),
                'smooth': soft_dice_kwargs.get('smooth'),
                'do_bg': soft_dice_kwargs.get('do_bg'),
                'ddp': soft_dice_kwargs.get('ddp'),
            },
            ce_kwargs=seg_loss_cfg.get('ce_kwargs', {}),
            weight_ce=seg_loss_cfg.get('weight_ce', 1),
            weight_dice=seg_loss_cfg.get('weight_dice', 1),
            weight_srec=seg_loss_cfg.get('weight_srec', 1),
            ignore_label=seg_loss_cfg.get('ignore_label', None),
        )

    def make_generator(offset=0):
        gen = torch.Generator()
        gen.manual_seed(config['seed'] + accelerator.process_index * 1000 + offset)
        return gen

    # If requested, recompute patch caches exactly once on the main process.
    # Then disable force_recompute so train/val dataset construction just reads cache.
    if config.get('force_recompute_patches', False):
        if accelerator.is_main_process:
            accelerator.print("force_recompute_patches=True: recomputing patch cache once on main process...")
            _recompute_ds = EdtSegDataset(config, apply_augmentation=False)
            del _recompute_ds
            accelerator.print("Patch cache recompute complete.")
        accelerator.wait_for_everyone()
        config = dict(config)
        config['force_recompute_patches'] = False

    # Train with augmentation, val without
    train_dataset = EdtSegDataset(config, apply_augmentation=True)
    val_dataset = EdtSegDataset(config, apply_augmentation=False)
    val_pert_dataset = None
    if config.get('eval_perturbed_val', False):
        val_pert_config = copy.deepcopy(config)
        val_pert_cfg = dict(val_pert_config.get('cond_local_perturb') or {})
        val_pert_cfg['enabled'] = True
        val_pert_cfg['apply_without_augmentation'] = True
        val_pert_config['cond_local_perturb'] = val_pert_cfg
        val_pert_dataset = EdtSegDataset(val_pert_config, apply_augmentation=False)

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
    if val_pert_dataset is not None:
        val_pert_dataset = _restrict_dataset_samples(val_pert_dataset, val_indices)

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
    val_pert_dataloader = None
    if val_pert_dataset is not None:
        val_pert_dataloader = torch.utils.data.DataLoader(
            val_pert_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=1,
            worker_init_fn=seed_worker,
            generator=make_generator(2),
            collate_fn=collate_with_padding,
        )

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

    if val_pert_dataloader is not None:
        model, optimizer, train_dataloader, val_dataloader, val_pert_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, val_dataloader, val_pert_dataloader, lr_scheduler
        )
    else:
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, val_dataloader, lr_scheduler
        )

    if accelerator.is_main_process:
        accelerator.print("\n=== Displacement Field Training Configuration ===")
        accelerator.print(f"Input channels: {config['in_channels']}")
        if disp_supervision == 'normal_scalar':
            accelerator.print(f"Displacement supervision: {disp_supervision} ({normal_loss_type}, beta={normal_loss_beta})")
        else:
            accelerator.print(f"Displacement supervision: {disp_supervision} ({disp_loss_type}, beta={disp_huber_beta})")
        output_str = "Output: displacement (3ch)"
        if use_sdt:
            output_str += " + SDT (1ch)"
        if use_heatmap:
            output_str += " + heatmap (1ch)"
        if use_segmentation:
            output_str += " + segmentation (2ch)"
        accelerator.print(output_str)
        if use_sdt:
            accelerator.print(f"Lambda SDT: {lambda_sdt}")
        if use_heatmap:
            accelerator.print(f"Lambda heatmap: {lambda_heatmap}")
        if use_segmentation:
            accelerator.print(f"Lambda segmentation: {lambda_segmentation}")
        if lambda_cond_disp > 0.0:
            accelerator.print(f"Lambda cond disp: {lambda_cond_disp}")
        accelerator.print(f"Supervise conditioning: {config.get('supervise_conditioning', False)}")
        if config.get('supervise_conditioning', False):
            accelerator.print(f"Cond supervision weight: {config.get('cond_supervision_weight', 0.1)}")
        optimizer_summary = f"Optimizer: {optimizer_type} (lr={optimizer_kwargs['learning_rate']}, weight_decay={optimizer_kwargs.get('weight_decay', 0)})"
        scheduler_details = ", ".join(f"{k}={v}" for k, v in scheduler_kwargs.items())
        scheduler_summary = f"Scheduler: {scheduler_type}" + (f" ({scheduler_details})" if scheduler_details else "")
        accelerator.print(optimizer_summary)
        accelerator.print(scheduler_summary)
        accelerator.print(f"Train samples: {num_train}, Val samples: {num_val}")
        accelerator.print(f"Eval perturbed val: {config.get('eval_perturbed_val', False)}")
        accelerator.print("=================================================\n")

    if config['verbose']:
            print("creating iterators...")
    val_iterator = iter(val_dataloader)
    val_pert_iterator = iter(val_pert_dataloader) if val_pert_dataloader is not None else None
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

        inputs, extrap_coords, gt_displacement, valid_mask, point_weights, point_normals, sdt_target, heatmap_target, seg_target, seg_skel = prepare_batch(
            batch, use_sdt, use_heatmap, use_segmentation
        )

        wandb_log = {}

        with accelerator.accumulate(model):
            # Forward pass
            output = model(inputs)
            disp_pred = output['displacement']  # [B, 3, D, H, W]

            # Displacement loss
            if disp_supervision == 'normal_scalar':
                if point_normals is None:
                    raise ValueError("point_normals missing from batch while displacement_supervision='normal_scalar'")
                surf_loss = surface_sampled_normal_loss(
                    disp_pred, extrap_coords, gt_displacement, point_normals, valid_mask,
                    loss_type=normal_loss_type, beta=normal_loss_beta, sample_weights=point_weights
                )
            else:
                surf_loss = surface_sampled_loss(
                    disp_pred, extrap_coords, gt_displacement, valid_mask,
                    loss_type=disp_loss_type, beta=disp_huber_beta, sample_weights=point_weights
                )
            total_loss = surf_loss

            wandb_log['surf_loss'] = surf_loss.detach().item()

            # Smoothness loss on displacement field
            if lambda_smooth > 0:
                smooth_loss = smoothness_loss(disp_pred)
                weighted_smooth_loss = lambda_smooth * smooth_loss
                total_loss = total_loss + weighted_smooth_loss
                wandb_log['smooth_loss'] = weighted_smooth_loss.detach().item()

            # Optional SDT loss
            if use_sdt:
                sdt_pred = output['sdt']  # [B, 1, D, H, W]
                sdt_loss = sdt_loss_fn(sdt_pred, sdt_target)
                weighted_sdt_loss = lambda_sdt * sdt_loss
                total_loss = total_loss + weighted_sdt_loss
                wandb_log['sdt_loss'] = weighted_sdt_loss.detach().item()

            # Optional heatmap loss (BCE + Dice)
            heatmap_pred = None
            if use_heatmap:
                heatmap_pred = output['heatmap']  # [B, 1, D, H, W]
                heatmap_target_binary = (heatmap_target > 0.5).float()
                heatmap_loss = heatmap_loss_fn(heatmap_pred, heatmap_target_binary)
                weighted_heatmap_loss = lambda_heatmap * heatmap_loss
                total_loss = total_loss + weighted_heatmap_loss
                wandb_log['heatmap_loss'] = weighted_heatmap_loss.detach().item()

            # Optional segmentation loss (MedialSurfaceRecall)
            if use_segmentation:
                seg_pred = output['segmentation']  # [B, 2, D, H, W]

                # Optionally mask out conditioning region from seg loss
                seg_loss_mask = None
                if mask_cond_from_seg_loss:
                    cond_mask_seg = (inputs[:, 1:2] > 0.5).float()  # [B, 1, D, H, W]
                    seg_loss_mask = (cond_mask_seg < 0.5).float()   # 1 everywhere except cond

                seg_loss = seg_loss_fn(seg_pred, seg_target.long(), seg_skel.long(), loss_mask=seg_loss_mask)
                weighted_seg_loss = lambda_segmentation * seg_loss
                total_loss = total_loss + weighted_seg_loss
                wandb_log['seg_loss'] = weighted_seg_loss.detach().item()

            if lambda_cond_disp > 0.0:
                cond_mask = (inputs[:, 1:2] > 0.5).float()
                disp_mag_sq = (disp_pred ** 2).sum(dim=1, keepdim=True)
                cond_loss = (disp_mag_sq * cond_mask).sum() / cond_mask.sum().clamp(min=1.0)
                weighted_cond_loss = lambda_cond_disp * cond_loss
                total_loss = total_loss + weighted_cond_loss
                wandb_log['cond_disp_loss'] = weighted_cond_loss.detach().item()

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
        if lambda_smooth > 0:
            postfix['smooth'] = f"{wandb_log['smooth_loss']:.4f}"
        if use_sdt:
            postfix['sdt'] = f"{wandb_log['sdt_loss']:.4f}"
        if use_heatmap:
            postfix['hm'] = f"{wandb_log['heatmap_loss']:.4f}"
        if use_segmentation:
            postfix['seg'] = f"{wandb_log['seg_loss']:.4f}"
        if lambda_cond_disp > 0.0:
            postfix['cond'] = f"{wandb_log['cond_disp_loss']:.4f}"
        progress_bar.set_postfix(postfix)
        progress_bar.update(1)

        if iteration % config['log_frequency'] == 0 and accelerator.is_main_process:
            with torch.no_grad():
                model.eval()

                val_batches_per_log = max(1, int(config.get('val_batches_per_log', 25)))
                val_metric_sums = {
                    'val_surf_loss': 0.0,
                    'val_loss': 0.0,
                }
                if lambda_smooth > 0:
                    val_metric_sums['val_smooth_loss'] = 0.0
                if use_sdt:
                    val_metric_sums['val_sdt_loss'] = 0.0
                if use_heatmap:
                    val_metric_sums['val_heatmap_loss'] = 0.0
                if use_segmentation:
                    val_metric_sums['val_seg_loss'] = 0.0
                if lambda_cond_disp > 0.0:
                    val_metric_sums['val_cond_disp_loss'] = 0.0

                first_val_vis = None
                for val_batch_idx in range(val_batches_per_log):
                    try:
                        val_batch = next(val_iterator)
                    except StopIteration:
                        val_iterator = iter(val_dataloader)
                        val_batch = next(val_iterator)

                    val_inputs, val_extrap_coords, val_gt_displacement, val_valid_mask, val_point_weights, val_point_normals, val_sdt_target, val_heatmap_target, val_seg_target, val_seg_skel = prepare_batch(
                        val_batch, use_sdt, use_heatmap, use_segmentation
                    )

                    val_output = model(val_inputs)
                    val_disp_pred = val_output['displacement']

                    if disp_supervision == 'normal_scalar':
                        if val_point_normals is None:
                            raise ValueError("val point_normals missing while displacement_supervision='normal_scalar'")
                        val_surf_loss = surface_sampled_normal_loss(
                            val_disp_pred, val_extrap_coords, val_gt_displacement, val_point_normals, val_valid_mask,
                            loss_type=normal_loss_type, beta=normal_loss_beta, sample_weights=val_point_weights
                        )
                    else:
                        val_surf_loss = surface_sampled_loss(
                            val_disp_pred, val_extrap_coords, val_gt_displacement, val_valid_mask,
                            loss_type=disp_loss_type, beta=disp_huber_beta, sample_weights=val_point_weights
                        )
                    val_total_loss = val_surf_loss
                    val_metric_sums['val_surf_loss'] += val_surf_loss.item()

                    val_sdt_pred = None
                    if lambda_smooth > 0:
                        val_smooth_loss = smoothness_loss(val_disp_pred)
                        val_weighted_smooth_loss = lambda_smooth * val_smooth_loss
                        val_total_loss = val_total_loss + val_weighted_smooth_loss
                        val_metric_sums['val_smooth_loss'] += val_weighted_smooth_loss.item()

                    if use_sdt:
                        val_sdt_pred = val_output['sdt']
                        val_sdt_loss = sdt_loss_fn(val_sdt_pred, val_sdt_target)
                        val_weighted_sdt_loss = lambda_sdt * val_sdt_loss
                        val_total_loss = val_total_loss + val_weighted_sdt_loss
                        val_metric_sums['val_sdt_loss'] += val_weighted_sdt_loss.item()

                    val_heatmap_pred = None
                    if use_heatmap:
                        val_heatmap_pred = val_output['heatmap']
                        val_heatmap_target_binary = (val_heatmap_target > 0.5).float()
                        val_heatmap_loss = heatmap_loss_fn(val_heatmap_pred, val_heatmap_target_binary)
                        val_weighted_heatmap_loss = lambda_heatmap * val_heatmap_loss
                        val_total_loss = val_total_loss + val_weighted_heatmap_loss
                        val_metric_sums['val_heatmap_loss'] += val_weighted_heatmap_loss.item()

                    if use_segmentation:
                        val_seg_pred = val_output['segmentation']
                        val_seg_loss_mask = None
                        if mask_cond_from_seg_loss:
                            val_cond_mask_seg = (val_inputs[:, 1:2] > 0.5).float()
                            val_seg_loss_mask = (val_cond_mask_seg < 0.5).float()
                        val_seg_loss = seg_loss_fn(
                            val_seg_pred, val_seg_target.long(), val_seg_skel.long(), loss_mask=val_seg_loss_mask
                        )
                        val_weighted_seg_loss = lambda_segmentation * val_seg_loss
                        val_total_loss = val_total_loss + val_weighted_seg_loss
                        val_metric_sums['val_seg_loss'] += val_weighted_seg_loss.item()

                    if lambda_cond_disp > 0.0:
                        val_cond_mask = (val_inputs[:, 1:2] > 0.5).float()
                        val_disp_mag_sq = (val_disp_pred ** 2).sum(dim=1, keepdim=True)
                        val_cond_loss = (val_disp_mag_sq * val_cond_mask).sum() / val_cond_mask.sum().clamp(min=1.0)
                        val_weighted_cond_loss = lambda_cond_disp * val_cond_loss
                        val_total_loss = val_total_loss + val_weighted_cond_loss
                        val_metric_sums['val_cond_disp_loss'] += val_weighted_cond_loss.item()

                    val_metric_sums['val_loss'] += val_total_loss.item()

                    if val_batch_idx == 0:
                        first_val_vis = {
                            'inputs': val_inputs,
                            'disp_pred': val_disp_pred,
                            'extrap_coords': val_extrap_coords,
                            'gt_displacement': val_gt_displacement,
                            'valid_mask': val_valid_mask,
                            'sdt_pred': val_sdt_pred,
                            'sdt_target': val_sdt_target,
                            'heatmap_pred': val_heatmap_pred,
                            'heatmap_target': val_heatmap_target,
                            'seg_pred': val_output.get('segmentation') if use_segmentation else None,
                            'seg_target': val_seg_target if use_segmentation else None,
                        }

                for key, value in val_metric_sums.items():
                    wandb_log[key] = value / val_batches_per_log

                # Create visualization
                train_img_path = f'{out_dir}/{iteration:06}_train.png'
                val_img_path = f'{out_dir}/{iteration:06}_val.png'

                train_sdt_pred = output.get('sdt') if use_sdt else None
                train_heatmap_pred = heatmap_pred if use_heatmap else None
                train_seg_pred = output.get('segmentation') if use_segmentation else None
                make_visualization(
                    inputs, disp_pred, extrap_coords, gt_displacement, valid_mask,
                    sdt_pred=train_sdt_pred, sdt_target=sdt_target,
                    heatmap_pred=train_heatmap_pred, heatmap_target=heatmap_target,
                    seg_pred=train_seg_pred, seg_target=seg_target if use_segmentation else None,
                    save_path=train_img_path
                )
                make_visualization(
                    first_val_vis['inputs'], first_val_vis['disp_pred'],
                    first_val_vis['extrap_coords'], first_val_vis['gt_displacement'],
                    first_val_vis['valid_mask'],
                    sdt_pred=first_val_vis['sdt_pred'], sdt_target=first_val_vis['sdt_target'],
                    heatmap_pred=first_val_vis['heatmap_pred'], heatmap_target=first_val_vis['heatmap_target'],
                    seg_pred=first_val_vis['seg_pred'],
                    seg_target=first_val_vis['seg_target'],
                    save_path=val_img_path
                )

                if wandb.run is not None:
                    wandb_log['train_image'] = wandb.Image(train_img_path)
                    wandb_log['val_image'] = wandb.Image(val_img_path)

                if val_pert_dataloader is not None:
                    try:
                        val_pert_batch = next(val_pert_iterator)
                    except StopIteration:
                        val_pert_iterator = iter(val_pert_dataloader)
                        val_pert_batch = next(val_pert_iterator)

                    val_pert_inputs, val_pert_extrap_coords, val_pert_gt_displacement, val_pert_valid_mask, val_pert_point_weights, val_pert_point_normals, val_pert_sdt_target, val_pert_heatmap_target, val_pert_seg_target, val_pert_seg_skel = prepare_batch(
                        val_pert_batch, use_sdt, use_heatmap, use_segmentation
                    )

                    val_pert_output = model(val_pert_inputs)
                    val_pert_disp_pred = val_pert_output['displacement']

                    if disp_supervision == 'normal_scalar':
                        if val_pert_point_normals is None:
                            raise ValueError("val_pert point_normals missing while displacement_supervision='normal_scalar'")
                        val_pert_surf_loss = surface_sampled_normal_loss(
                            val_pert_disp_pred, val_pert_extrap_coords, val_pert_gt_displacement, val_pert_point_normals, val_pert_valid_mask,
                            loss_type=normal_loss_type, beta=normal_loss_beta, sample_weights=val_pert_point_weights
                        )
                    else:
                        val_pert_surf_loss = surface_sampled_loss(
                            val_pert_disp_pred, val_pert_extrap_coords, val_pert_gt_displacement, val_pert_valid_mask,
                            loss_type=disp_loss_type, beta=disp_huber_beta, sample_weights=val_pert_point_weights
                        )
                    val_pert_total_loss = val_pert_surf_loss
                    wandb_log['val_pert_surf_loss'] = val_pert_surf_loss.item()

                    val_pert_sdt_pred = None
                    if lambda_smooth > 0:
                        val_pert_smooth_loss = smoothness_loss(val_pert_disp_pred)
                        val_pert_weighted_smooth_loss = lambda_smooth * val_pert_smooth_loss
                        val_pert_total_loss = val_pert_total_loss + val_pert_weighted_smooth_loss
                        wandb_log['val_pert_smooth_loss'] = val_pert_weighted_smooth_loss.item()

                    if use_sdt:
                        val_pert_sdt_pred = val_pert_output['sdt']
                        val_pert_sdt_loss = sdt_loss_fn(val_pert_sdt_pred, val_pert_sdt_target)
                        val_pert_weighted_sdt_loss = lambda_sdt * val_pert_sdt_loss
                        val_pert_total_loss = val_pert_total_loss + val_pert_weighted_sdt_loss
                        wandb_log['val_pert_sdt_loss'] = val_pert_weighted_sdt_loss.item()

                    val_pert_heatmap_pred = None
                    if use_heatmap:
                        val_pert_heatmap_pred = val_pert_output['heatmap']
                        val_pert_heatmap_target_binary = (val_pert_heatmap_target > 0.5).float()
                        val_pert_heatmap_loss = heatmap_loss_fn(val_pert_heatmap_pred, val_pert_heatmap_target_binary)
                        val_pert_weighted_heatmap_loss = lambda_heatmap * val_pert_heatmap_loss
                        val_pert_total_loss = val_pert_total_loss + val_pert_weighted_heatmap_loss
                        wandb_log['val_pert_heatmap_loss'] = val_pert_weighted_heatmap_loss.item()

                    if use_segmentation:
                        val_pert_seg_pred = val_pert_output['segmentation']
                        val_pert_seg_loss_mask = None
                        if mask_cond_from_seg_loss:
                            val_pert_cond_mask_seg = (val_pert_inputs[:, 1:2] > 0.5).float()
                            val_pert_seg_loss_mask = (val_pert_cond_mask_seg < 0.5).float()
                        val_pert_seg_loss = seg_loss_fn(
                            val_pert_seg_pred, val_pert_seg_target.long(), val_pert_seg_skel.long(), loss_mask=val_pert_seg_loss_mask
                        )
                        val_pert_weighted_seg_loss = lambda_segmentation * val_pert_seg_loss
                        val_pert_total_loss = val_pert_total_loss + val_pert_weighted_seg_loss
                        wandb_log['val_pert_seg_loss'] = val_pert_weighted_seg_loss.item()

                    if lambda_cond_disp > 0.0:
                        val_pert_cond_mask = (val_pert_inputs[:, 1:2] > 0.5).float()
                        val_pert_disp_mag_sq = (val_pert_disp_pred ** 2).sum(dim=1, keepdim=True)
                        val_pert_cond_loss = (val_pert_disp_mag_sq * val_pert_cond_mask).sum() / val_pert_cond_mask.sum().clamp(min=1.0)
                        val_pert_weighted_cond_loss = lambda_cond_disp * val_pert_cond_loss
                        val_pert_total_loss = val_pert_total_loss + val_pert_weighted_cond_loss
                        wandb_log['val_pert_cond_disp_loss'] = val_pert_weighted_cond_loss.item()

                    wandb_log['val_pert_loss'] = val_pert_total_loss.item()

                    if config.get('log_perturbed_val_images', False):
                        val_pert_img_path = f'{out_dir}/{iteration:06}_val_pert.png'
                        make_visualization(
                            val_pert_inputs, val_pert_disp_pred, val_pert_extrap_coords, val_pert_gt_displacement, val_pert_valid_mask,
                            sdt_pred=val_pert_sdt_pred, sdt_target=val_pert_sdt_target,
                            heatmap_pred=val_pert_heatmap_pred, heatmap_target=val_pert_heatmap_target,
                            seg_pred=val_pert_output.get('segmentation') if use_segmentation else None,
                            seg_target=val_pert_seg_target if use_segmentation else None,
                            save_path=val_pert_img_path
                        )
                        if wandb.run is not None:
                            wandb_log['val_pert_image'] = wandb.Image(val_pert_img_path)

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
