import json
import math
import os
from copy import deepcopy
import wandb
import numpy as np 
import random
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import accelerate
from accelerate.utils import GradientAccumulationPlugin, set_seed
import click
from vesuvius.models.utils import InitWeights_He
from vesuvius.models.training.optimizers import create_optimizer
from vesuvius.models.training.lr_schedulers import get_scheduler
from vesuvius.neural_tracing.nets.models import make_model
from common import (
    build_preview_montage,
    save_val_preview_tif,
    to_uint8_image,
    to_uint8_label,
    to_uint8_probability,
)
from flat_ink_dataset import FlatInkDataset
from losses import create_loss_from_config
from stitching import resolve_model_and_loader_patch_sizes, run_stitched_model_forward


@click.command()
@click.argument('config_path', type=click.Path(exists=True))
def train(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)

    checkpoint_path = config.get('checkpoint')
    checkpoint = None
    weights_only = bool(config.get('weights_only', False))
    resume_full_state = False
    start_step = 0
    if checkpoint_path:
        if not os.path.isabs(checkpoint_path):
            checkpoint_path = os.path.join(os.path.dirname(config_path), checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        resume_full_state = not weights_only
        if resume_full_state:
            checkpoint_step = checkpoint.get('step')
            if checkpoint_step is None:
                raise ValueError(
                    f"Checkpoint '{checkpoint_path}' is missing 'step', cannot resume training state"
                )
            start_step = int(checkpoint_step) + 1

    if str(config.get('model_type', '')).strip().lower() == 'dinov2':
        config.setdefault('model_config', {})
        if 'pretrained_backbone' in config:
            config['model_config'].setdefault('pretrained_backbone', config['pretrained_backbone'])
        if 'pretrained_decoder_type' in config:
            config['model_config'].setdefault('pretrained_decoder_type', config['pretrained_decoder_type'])
        if not config['model_config'].get('pretrained_backbone'):
            raise ValueError(
                "model_type='dinov2' requires model_config.pretrained_backbone "
                "or a top-level pretrained_backbone entry"
            )
        config['model_type'] = 'unet'

    ema_config = config.get('ema') or {}
    ema_enabled = bool(ema_config.get('enabled', False))
    ema_decay = float(ema_config.get('decay', 0.999))
    ema_start_step = int(ema_config.get('start_step', 0))
    ema_update_every_steps = int(ema_config.get('update_every_steps', 1))
    ema_validate = bool(ema_config.get('validate', ema_enabled))
    ema_save_in_checkpoint = bool(ema_config.get('save_in_checkpoint', ema_enabled))
    config['ema'] = {
        'enabled': ema_enabled,
        'decay': ema_decay,
        'start_step': ema_start_step,
        'update_every_steps': ema_update_every_steps,
        'validate': ema_validate,
        'save_in_checkpoint': ema_save_in_checkpoint,
    }

    config.setdefault('volume_auth_json', None)
    requested_stitch_factor = int(config.get('stitch_factor', 1))
    use_stitched_forward = bool(
        config.get('use_stitched_forward', requested_stitch_factor > 1)
    )
    model_crop_size = tuple(config['patch_size'])
    loader_patch_size = model_crop_size
    stitch_factor = 1
    if use_stitched_forward:
        model_crop_size, loader_patch_size, stitch_factor = resolve_model_and_loader_patch_sizes(config)
    config['crop_size'] = list(model_crop_size)
    config['patch_size'] = list(model_crop_size)
    config['stitch_factor'] = stitch_factor
    config['use_stitched_forward'] = use_stitched_forward
    config['targets']['ink']['out_channels'] = 1
    config['targets']['ink']['activation'] = 'none'
    learning_rate = config.get('learning_rate', 0.01)
    grad_acc_steps = int(config.get('grad_acc_steps', 1))
    grad_clip = config.get('grad_clip')
    max_steps = config.get('max_steps', math.ceil(config['num_iterations'] / grad_acc_steps))

    dataloader_config = accelerate.DataLoaderConfiguration(non_blocking = True)

    # The training loop reuses the dataloader indefinitely, so keep accumulation
    # boundaries independent of dataloader exhaustion.
    gradient_accumulation_plugin = GradientAccumulationPlugin(
        num_steps=grad_acc_steps,
        sync_with_dataloader=False,
    )

    accelerator = accelerate.Accelerator(
        mixed_precision              = config.get('mixed_precision', "fp16"),
        gradient_accumulation_plugin = gradient_accumulation_plugin,
        dataloader_config            = dataloader_config
    )

    if 'wandb_project' in config and accelerator.is_main_process:
        wandb_kwargs = {
            'project' : config['wandb_project'],
            'entity'  : config['wandb_entity'],
            'config'  : config
        }
        if config.get('wandb_resume', False):
            wandb_run_id = config.get('wandb_run_id')
            if not wandb_run_id and checkpoint is not None:
                wandb_run_id = checkpoint.get('wandb_run_id')
            if not wandb_run_id:
                raise ValueError(
                    "wandb_resume=true requires wandb_run_id in config or checkpoint"
                )
            wandb_kwargs['id'] = wandb_run_id
            wandb_kwargs['resume'] = 'must'

        wandb.init(**wandb_kwargs)

    out_dir = config['out_dir']
    os.makedirs(out_dir, exist_ok=True)
    train_preview_dir = os.path.join(out_dir, 'train_previews')
    os.makedirs(train_preview_dir, exist_ok=True)
    val_preview_dir = os.path.join(out_dir, 'val_previews')
    os.makedirs(val_preview_dir, exist_ok=True)

    set_seed(config['seed'])

    dataset_config = deepcopy(config)
    dataset_config['patch_size'] = list(loader_patch_size)

    shared_ds = FlatInkDataset(dataset_config, do_augmentations=False)
    train_ds = FlatInkDataset(dataset_config, do_augmentations=True, patches=shared_ds.patches)
    val_ds = shared_ds

    num_patches = len(train_ds)
    num_val     = int(max(1, num_patches * config.get('val_fraction', 0.1)))
    num_train   = num_patches - num_val
    
    indices = torch.randperm(num_patches, generator=torch.Generator().manual_seed(config['seed'])).tolist()
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]
    train_subset = Subset(train_ds, train_indices)
    val_subset = Subset(val_ds, val_indices)

    dataloader_workers = int(config.get('dataloader_workers', 0))
    dataloader_kwargs = {}
    if dataloader_workers > 0:
        # CUDA is initialized before dataloader iteration via accelerate/model setup.
        # Spawn workers instead of forking from a CUDA-initialized parent process.
        dataloader_kwargs['multiprocessing_context'] = 'spawn'
        dataloader_kwargs['persistent_workers'] = True

    train_dl = DataLoader(
        train_subset,
        batch_size=config['batch_size'],
        shuffle=True,
        generator=torch.Generator().manual_seed(config['seed']),
        num_workers=dataloader_workers,
        **dataloader_kwargs,
    )
    # Validation only consumes a capped number of batches (`val_steps`), so
    # shuffle to sample a different deterministic subset on each pass.
    val_dl = DataLoader(
        val_subset,
        batch_size=config['batch_size'],
        shuffle=True,
        generator=torch.Generator().manual_seed(config['seed'] + 1),
        num_workers=dataloader_workers,
        **dataloader_kwargs,
    )

    model = make_model(config)
    optimizer_target = model
    pretrained_backbone = (config.get('model_config') or {}).get('pretrained_backbone')
    if pretrained_backbone:
        freeze_encoder = bool(config.get('freeze_encoder', False))
        encoder_lr_mult = float(config.get('encoder_lr_mult', 1.0))
        if not 0.0 <= encoder_lr_mult <= 1.0:
            raise ValueError(f"encoder_lr_mult must be between 0 and 1 inclusive, got {encoder_lr_mult}")

        encoder_params = list(model.shared_encoder.parameters())
        if freeze_encoder:
            for param in encoder_params:
                param.requires_grad = False

        if freeze_encoder or encoder_lr_mult != 1.0:
            encoder_param_ids = {id(param) for param in encoder_params}
            other_params = [
                param for param in model.parameters()
                if param.requires_grad and id(param) not in encoder_param_ids
            ]
            optimizer_target = []
            if other_params:
                optimizer_target.append({'params': other_params})
            if not freeze_encoder and encoder_params:
                optimizer_target.append({
                    'params': encoder_params,
                    'lr': learning_rate * encoder_lr_mult,
                })
            if not optimizer_target:
                raise ValueError("No trainable parameters remain after applying freeze_encoder")

    optimizer = create_optimizer({
                'name': config.get('optimizer', 'sgd'),
                'learning_rate': learning_rate,
                'weight_decay': config.get('weight_decay', 3e-5),
                }, optimizer_target)

    lr_scheduler = get_scheduler(
        'diffusers_cosine_warmup',
        optimizer,
        initial_lr=learning_rate,
        max_steps=max_steps,
        warmup_steps=config.get('warmup_steps', 1000),
    )

    if not (config.get('model_config') or {}).get('pretrained_backbone'):
        model.apply(InitWeights_He(neg_slope=0.2))

    loss = create_loss_from_config(config)

    model, optimizer, train_dl, val_dl, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dl, val_dl, lr_scheduler
        )
    unwrapped_model = accelerator.unwrap_model(model)
    ema_model = deepcopy(unwrapped_model) if ema_enabled else None
    if ema_model is not None:
        ema_model.eval()
        for parameter in ema_model.parameters():
            parameter.requires_grad_(False)

    optimizer_step = 0
    if checkpoint is not None:
        model_state = checkpoint.get('model')
        if model_state is None:
            raise ValueError(f"Checkpoint '{checkpoint_path}' is missing 'model'")
        model.load_state_dict(model_state)

        if resume_full_state:
            optimizer_state = checkpoint.get('optimizer')
            lr_scheduler_state = checkpoint.get('lr_scheduler')
            if optimizer_state is None or lr_scheduler_state is None:
                raise ValueError(
                    f"Checkpoint '{checkpoint_path}' is missing optimizer or lr_scheduler state"
                )
            optimizer.load_state_dict(optimizer_state)
            lr_scheduler.load_state_dict(lr_scheduler_state)

            if ema_model is not None:
                ema_model_state = checkpoint.get('ema_model')
                if ema_model_state is not None:
                    ema_model.load_state_dict(ema_model_state)
                    optimizer_step = int(checkpoint.get('ema_optimizer_step', 0))

        accelerator.print(
            f"Loaded checkpoint '{checkpoint_path}'"
            + (f" and resuming from step {start_step}" if resume_full_state else " (weights only)")
        )

    train_iterator = iter(train_dl)
    val_every = int(config.get('val_every', 500))
    save_every = int(config.get('save_every', val_every))
    log_every = config.get('log_every', 1)
    val_preview_batches = config.get('val_preview_batches', 3)

    progress_bar = tqdm(
        range(start_step, config['num_iterations']),
        disable=not accelerator.is_main_process,
        dynamic_ncols=True,
    )
    latest_val_loss = None
    latest_ema_val_loss = None

    def forward_ink(image, active_model=None):
        active_model = model if active_model is None else active_model
        if use_stitched_forward:
            return run_stitched_model_forward(active_model, image, model_crop_size)['ink']
        return active_model(image)['ink']

    def append_preview_tiles(preview_inputs, preview_labels, preview_probabilities, batch, preds, targets, ignore_mask):
        input_mid_slice = batch['image'].float()[:, :, batch['image'].shape[2] // 2]
        if input_mid_slice.shape[-2:] != preds.shape[-2:]:
            input_mid_slice = F.interpolate(
                input_mid_slice,
                size=preds.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )

        gathered_inputs = accelerator.gather_for_metrics(input_mid_slice)
        gathered_targets = accelerator.gather_for_metrics(targets)
        gathered_ignore_masks = accelerator.gather_for_metrics(ignore_mask)
        gathered_probabilities = accelerator.gather_for_metrics(torch.sigmoid(preds.float()))

        if not accelerator.is_main_process:
            return

        input_tiles = gathered_inputs[:, 0].detach().cpu().numpy()
        label_tiles = gathered_targets[:, 0].detach().cpu().numpy()
        ignore_mask_tiles = gathered_ignore_masks[:, 0].detach().cpu().numpy()
        probability_tiles = gathered_probabilities[:, 0].detach().cpu().numpy()

        for input_tile, label_tile, ignore_mask_tile, probability_tile in zip(
            input_tiles,
            label_tiles,
            ignore_mask_tiles,
            probability_tiles,
        ):
            preview_inputs.append(to_uint8_image(input_tile))
            preview_labels.append(to_uint8_label(label_tile, ignore_mask_tile))
            preview_probabilities.append(to_uint8_probability(probability_tile))

    def refresh_progress_bar(current_train_loss):
        if not accelerator.is_main_process:
            return

        postfix = {
            'loss': f'{current_train_loss:.4f}',
        }
        if latest_val_loss is not None:
            postfix['val_loss'] = f'{latest_val_loss:.4f}'
        if latest_ema_val_loss is not None:
            postfix['ema_val_loss'] = f'{latest_ema_val_loss:.4f}'
        postfix['lr'] = f"{optimizer.param_groups[0]['lr']:.2e}"
        progress_bar.set_postfix(postfix, refresh=False)
        progress_bar.update(0)

    for step in progress_bar:
        model.train()

        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_dl)
            batch = next(train_iterator)

        with accelerator.accumulate(model):
            with accelerator.autocast():
                preds = forward_ink(batch['image'])
            targets = (torch.amax(batch['inklabels'].float(), dim=2) > 0).float()
            supervision_mask = torch.amax(batch['supervision_mask'].float(), dim=2)
            ignore_mask = (supervision_mask <= 0).float()
            targets_with_ignore = torch.cat([targets, ignore_mask], dim=1)
            l = loss(preds.float(), targets_with_ignore)
            if not torch.isfinite(l):
                raise RuntimeError(f"Non-finite loss at step {step}")
            accelerator.backward(l)
            if grad_clip is not None and grad_clip > 0 and accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            overflow_step_skipped = bool(
                accelerator.sync_gradients and getattr(optimizer, 'step_was_skipped', False)
            )
            lr_scheduler.step()
            optimizer.zero_grad()
            if accelerator.sync_gradients and not overflow_step_skipped:
                optimizer_step += 1
                if ema_model is not None and optimizer_step >= ema_start_step:
                    if (optimizer_step - ema_start_step) % ema_update_every_steps == 0:
                        ema_state = ema_model.state_dict()
                        for name, model_value in unwrapped_model.state_dict().items():
                            ema_value = ema_state[name]
                            model_value = model_value.detach()
                            if torch.is_floating_point(ema_value):
                                ema_value.lerp_(model_value.to(dtype=ema_value.dtype), 1.0 - ema_decay)
                            else:
                                ema_value.copy_(model_value)

        train_loss = l.item()
        if accelerator.is_main_process:
            refresh_progress_bar(train_loss)
            if overflow_step_skipped:
                tqdm.write(f'step {step} | optimizer step skipped due to fp16 overflow')

        if accelerator.is_main_process and step % log_every == 0:
            log_dict = {
                'train/loss': train_loss,
                'train/lr': optimizer.param_groups[0]['lr'],
                'train/overflow_step_skipped': int(overflow_step_skipped),
                'step': step,
            }
            latest_loss_metrics = getattr(loss, 'latest_metrics', None)
            if isinstance(latest_loss_metrics, dict):
                log_dict.update(latest_loss_metrics)
            if wandb.run is not None:
                wandb.log(log_dict, step=step)

        if step % val_every == 0 and step > 0:
            train_preview_inputs = []
            train_preview_labels = []
            train_preview_probabilities = []
            append_preview_tiles(
                train_preview_inputs,
                train_preview_labels,
                train_preview_probabilities,
                batch,
                preds.detach(),
                targets.detach(),
                ignore_mask.detach(),
            )
            model.eval()
            val_losses = []
            ema_val_losses = []
            val_preview_inputs = []
            val_preview_labels = []
            val_preview_probabilities = []
            val_iterator = iter(val_dl)
            num_val_batches = min(len(val_dl), config.get('val_steps', 10))
            preview_batch_indices = set(
                random.sample(range(num_val_batches), k=min(val_preview_batches, num_val_batches))
            )
            with torch.no_grad():
                for val_batch_idx in range(num_val_batches):
                    val_batch = next(val_iterator)
                    with accelerator.autocast():
                        val_preds = forward_ink(val_batch['image'])
                    val_targets = torch.amax(val_batch['inklabels'].float(), dim=2)
                    val_supervision_mask = torch.amax(val_batch['supervision_mask'].float(), dim=2)
                    val_targets = (val_targets > 0).float()
                    val_ignore_mask = (val_supervision_mask <= 0).float()
                    val_targets_with_ignore = torch.cat([val_targets, val_ignore_mask], dim=1)
                    val_l = loss(val_preds.float(), val_targets_with_ignore)
                    val_losses.append(val_l.item())
                    if ema_model is not None and ema_validate:
                        with accelerator.autocast():
                            ema_val_preds = forward_ink(val_batch['image'], active_model=ema_model)
                        ema_val_l = loss(ema_val_preds.float(), val_targets_with_ignore)
                        ema_val_losses.append(ema_val_l.item())

                    if val_batch_idx in preview_batch_indices:
                        append_preview_tiles(
                            val_preview_inputs,
                            val_preview_labels,
                            val_preview_probabilities,
                            val_batch,
                            val_preds.detach(),
                            val_targets.detach(),
                            val_ignore_mask.detach(),
                        )

            mean_val_loss = np.mean(val_losses)
            mean_ema_val_loss = np.mean(ema_val_losses) if ema_val_losses else None
            if accelerator.is_main_process:
                latest_val_loss = float(mean_val_loss)
                latest_ema_val_loss = (
                    float(mean_ema_val_loss) if mean_ema_val_loss is not None else None
                )
                refresh_progress_bar(train_loss)
                train_preview_montage = build_preview_montage(
                    train_preview_inputs,
                    train_preview_labels,
                    train_preview_probabilities,
                )
                val_preview_montage = build_preview_montage(
                    val_preview_inputs,
                    val_preview_labels,
                    val_preview_probabilities,
                )
                save_val_preview_tif(
                    os.path.join(train_preview_dir, f'train_preview_{step:06}.tif'),
                    train_preview_inputs,
                    train_preview_labels,
                    train_preview_probabilities,
                )
                save_val_preview_tif(
                    os.path.join(val_preview_dir, f'val_preview_{step:06}.tif'),
                    val_preview_inputs,
                    val_preview_labels,
                    val_preview_probabilities,
                )
                if wandb.run is not None:
                    log_dict = {'val/loss': mean_val_loss}
                    if mean_ema_val_loss is not None:
                        log_dict['val/loss_ema'] = mean_ema_val_loss
                    if train_preview_montage is not None:
                        log_dict['train/preview'] = wandb.Image(
                            train_preview_montage,
                            caption=f"step {step} train preview",
                        )
                    if val_preview_montage is not None:
                        log_dict['val/preview'] = wandb.Image(
                            val_preview_montage,
                            caption=f"step {step} val preview",
                        )
                    wandb.log(log_dict, step=step)

        if accelerator.is_main_process and step % save_every == 0 and step > 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
                'step': step,
                'wandb_run_id': wandb.run.id if wandb.run is not None else config.get('wandb_run_id'),
            }
            if ema_model is not None and ema_save_in_checkpoint:
                checkpoint['ema_model'] = ema_model.state_dict()
                checkpoint['ema_optimizer_step'] = optimizer_step
            torch.save(checkpoint, f'{out_dir}/ckpt_{step:06}.pth')

if __name__ == '__main__':
    train()
