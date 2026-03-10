import json
import math
import os
import wandb
import numpy as np 
import random
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import accelerate
from accelerate.utils import TorchDynamoPlugin, GradientAccumulationPlugin, set_seed
import click
from vesuvius.models.training.optimizers import create_optimizer
from vesuvius.models.training.lr_schedulers import get_scheduler
from vesuvius.models.training.loss.nnunet_losses import DC_and_CE_loss
from vesuvius.neural_tracing.nets.models import make_model
from common import save_val_preview_tif, to_uint8_image, to_uint8_label, to_uint8_probability
from flat_ink_dataset import FlatInkDataset

@click.command()
@click.argument('config_path', type=click.Path(exists=True))
def train(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)

    config['crop_size'] = config['patch_size']
    learning_rate = config.get('learning_rate', 0.01)
    grad_acc_steps = int(config.get('grad_acc_steps', 1))
    max_steps = config.get('max_steps', math.ceil(config['num_iterations'] / grad_acc_steps))

    out_dir = config['out_dir']
    os.makedirs(out_dir, exist_ok=True)
    val_preview_dir = os.path.join(out_dir, 'val_previews')
    os.makedirs(val_preview_dir, exist_ok=True)

    set_seed(config['seed'])

    dynamo_plugin = TorchDynamoPlugin(
        backend   = "inductor",
        mode      = "default"
    )

    dataloader_config = accelerate.DataLoaderConfiguration(
        non_blocking = True
    )

    # The training loop reuses the dataloader indefinitely, so keep accumulation
    # boundaries independent of dataloader exhaustion.
    gradient_accumulation_plugin = GradientAccumulationPlugin(
        num_steps=grad_acc_steps,
        sync_with_dataloader=False,
    )

    accelerator = accelerate.Accelerator(
        mixed_precision              = config.get('mixed_precision', "fp16"),
        gradient_accumulation_plugin = gradient_accumulation_plugin,
        dynamo_plugin                = dynamo_plugin,
        dataloader_config            = dataloader_config
    )

    if 'wandb_project' in config and accelerator.is_main_process:
        wandb_kwargs = {
            'project' : config['wandb_project'],
            'entity'  : config['wandb_entity'],
            'config'  : config
        }

        wandb.init(**wandb_kwargs)

    shared_ds = FlatInkDataset(
        config,
        do_augmentations=False
    )

    train_ds = FlatInkDataset(
        config,
        do_augmentations=True,
        patches=shared_ds.patches,
    )

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

    train_dl = DataLoader(
        train_subset,
        batch_size=config['batch_size'],
        shuffle=True,
        generator=torch.Generator().manual_seed(config['seed']),
        num_workers=dataloader_workers,
    )
    val_dl = DataLoader(
        val_subset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=dataloader_workers,
    )

    model = make_model(config)

    optimizer = create_optimizer({
                'name': config.get('optimizer', 'sgd'),
                'learning_rate': learning_rate,
                'weight_decay': config.get('weight_decay', 3e-5),
                }, model)

    lr_scheduler = get_scheduler(
        'diffusers_cosine_warmup',
        optimizer,
        initial_lr=learning_rate,
        max_steps=max_steps,
        warmup_steps=config.get('warmup_steps', 1000),
    )

    loss = DC_and_CE_loss(
        soft_dice_kwargs={},
        ce_kwargs={'label_smoothing': 0.1},
        weight_dice=0.25,
        weight_ce=1.0,
        ignore_label=-1,
    )

    model, optimizer, train_dl, val_dl, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dl, val_dl, lr_scheduler
        )

    train_iterator = iter(train_dl)
    val_every = config.get('val_every', 500)
    log_every = config.get('log_every', 1)
    val_preview_batches = config.get('val_preview_batches', 2)

    start_step = 0
    for step in tqdm(range(start_step, config['num_iterations']), disable=not accelerator.is_main_process):
        model.train()

        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_dl)
            batch = next(train_iterator)

        with accelerator.accumulate(model), accelerator.autocast():
            preds = model(batch['image'])['ink']
            targets = (torch.amax(batch['inklabels'].float(), dim=2) > 0).long()
            supervision_mask = torch.amax(batch['supervision_mask'].float(), dim=2)
            targets[supervision_mask <= 0] = -1
            l = loss(preds, targets)
            accelerator.backward(l)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        if accelerator.is_main_process and step % log_every == 0:
            log_dict = {
                'train/loss': l.item(),
                'train/lr': optimizer.param_groups[0]['lr'],
                'step': step,
            }
            if wandb.run is not None:
                wandb.log(log_dict, step=step)

        if step % val_every == 0 and step > 0:
            model.eval()
            val_losses = []
            val_preview_inputs = []
            val_preview_labels = []
            val_preview_probabilities = []
            val_iterator = iter(val_dl)
            num_val_batches = min(len(val_dl), config.get('val_steps', 10))
            with torch.no_grad(), accelerator.autocast():
                for val_batch_idx in range(num_val_batches):
                    val_batch = next(val_iterator)
                    val_preds = model(val_batch['image'])['ink']
                    if val_preds.shape[1] == 1:
                        val_probabilities = torch.sigmoid(val_preds.float())
                    else:
                        val_probabilities = torch.softmax(val_preds.float(), dim=1)[:, 1:2]
                    val_targets = torch.amax(val_batch['inklabels'].float(), dim=2)
                    val_supervision_mask = torch.amax(val_batch['supervision_mask'].float(), dim=2)
                    if val_targets.shape[-2:] != val_preds.shape[-2:]:
                        val_targets = F.interpolate(val_targets, size=val_preds.shape[-2:], mode='nearest')
                        val_supervision_mask = F.interpolate(val_supervision_mask, size=val_preds.shape[-2:], mode='nearest')
                    val_targets = (val_targets > 0).long()
                    val_targets[val_supervision_mask <= 0] = -1
                    val_l = loss(val_preds, val_targets)
                    val_losses.append(val_l.item())

                    if val_batch_idx < val_preview_batches:
                        input_mid_slice = val_batch['image'].float()[:, :, val_batch['image'].shape[2] // 2]
                        if input_mid_slice.shape[-2:] != val_preds.shape[-2:]:
                            input_mid_slice = F.interpolate(
                                input_mid_slice,
                                size=val_preds.shape[-2:],
                                mode='bilinear',
                                align_corners=False,
                            )

                        gathered_inputs = accelerator.gather_for_metrics(input_mid_slice)
                        gathered_targets = accelerator.gather_for_metrics(val_targets.float())
                        gathered_probabilities = accelerator.gather_for_metrics(val_probabilities)

                        if accelerator.is_main_process:
                            input_tiles = gathered_inputs[:, 0].detach().cpu().numpy()
                            label_tiles = gathered_targets[:, 0].detach().cpu().numpy()
                            probability_tiles = gathered_probabilities[:, 0].detach().cpu().numpy()

                            for input_tile, label_tile, probability_tile in zip(input_tiles, label_tiles, probability_tiles):
                                val_preview_inputs.append(to_uint8_image(input_tile))
                                val_preview_labels.append(to_uint8_label(label_tile))
                                val_preview_probabilities.append(to_uint8_probability(probability_tile))

            mean_val_loss = np.mean(val_losses)
            if accelerator.is_main_process:
                tqdm.write(f'step {step} | val_loss {mean_val_loss:.4f}')
                save_val_preview_tif(
                    os.path.join(val_preview_dir, f'val_preview_{step:06}.tif'),
                    val_preview_inputs,
                    val_preview_labels,
                    val_preview_probabilities,
                )
                if wandb.run is not None:
                    wandb.log({'val/loss': mean_val_loss}, step=step)
                    
                    torch.save({
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'config': config,
                            'step': step,
                            'wandb_run_id': wandb.run.id if wandb.run is not None else config.get('wandb_run_id'),
                        }, f'{out_dir}/ckpt_{step:06}.pth')

if __name__ == '__main__':
    train()
