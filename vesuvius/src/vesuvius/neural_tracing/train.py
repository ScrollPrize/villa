
import os
import math
import json
import click
import torch
import wandb
import random
import diffusers
import accelerate
import numpy as np
from tqdm import tqdm
from einops import rearrange
import matplotlib.pyplot as plt
import torch.nn.functional as F

from dataset import HeatmapDatasetV2, load_datasets
from vesuvius.neural_tracing.datasets.PatchInCubeDataset import PatchInCubeDataset
from vesuvius.models.training.loss.nnunet_losses import DeepSupervisionWrapper
from vesuvius.neural_tracing.deep_supervision import _resize_for_ds, _compute_ds_weights
from models import make_model


def prepare_batch(batch):
    inputs = torch.cat([
        batch['volume'].unsqueeze(1),
        batch['localiser'].unsqueeze(1),
        rearrange(batch['uv_heatmaps_in'], 'b z y x c -> b c z y x'),
    ], dim=1)
    targets = rearrange(batch['uv_heatmaps_out'], 'b z y x c -> b c z y x')
    return inputs, targets

@click.command()
@click.argument('config_path', type=click.Path(exists=True))
def train(config_path):

    with open(config_path, 'r') as f:
        config = json.load(f)


    config.setdefault('num_iterations', 250000)
    log_image_max_samples = config.get('log_image_max_samples', 16)
    log_image_grid_cols = config.get('log_image_grid_cols', 4)
    log_image_ext = config.get('log_image_ext', 'jpg')
    log_image_quality = config.get('log_image_quality', 80)

    out_dir = config['out_dir']
    os.makedirs(out_dir, exist_ok=True)

    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])

    accelerator = accelerate.Accelerator(
        mixed_precision=config['mixed_precision'],
        gradient_accumulation_steps=config['grad_acc_steps'] if 'grad_acc_steps' in config else 1,
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

    model = make_model(config)
    compile_enabled = config.get('compile_model', config.get('compile', True))
    if compile_enabled:
        try:
            model = torch.compile(model)
            if accelerator.is_main_process:
                accelerator.print("Model compiled with torch.compile")
        except Exception as e:
            if accelerator.is_main_process:
                accelerator.print(f"torch.compile failed ({e}); continuing without compilation")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    lr_scheduler = diffusers.optimization.get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config['lr_warmup_steps'],
        num_training_steps=config['num_iterations'] * accelerate.PartialState().num_processes,  # FIXME: nasty adjustment by num_processes here accounts for the fact that accelerator's prepare_scheduler weirdly causes the scheduler to take num_processes scheduler-steps per optimiser-step
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
    val_iterator = iter(val_dataloader)

    def loss_fn(target_pred, targets, mask):
        if config['binary']:
            targets_binary = (targets > 0.5).long()  # FIXME: should instead not do the gaussian conv in data-loader!
            from vesuvius.models.training.loss.nnunet_losses import DC_and_BCE_loss
            # Apply mask by zeroing invalid regions before computing the combined loss.
            return DC_and_BCE_loss(bce_kwargs={}, soft_dice_kwargs={'ddp': False})(
                target_pred * mask, targets_binary * mask
            )
        else:
            # TODO: should this instead weight each element in batch equally regardless of valid area?
            return ((target_pred - targets) ** 2 * mask).sum() / mask.sum()
            
    progress_bar = tqdm(total=config['num_iterations'], disable=not accelerator.is_local_main_process)
    for iteration, batch in enumerate(train_dataloader):

        inputs, targets = prepare_batch(batch)
        if 'uv_heatmaps_out_mask' in batch:
            mask = rearrange(batch['uv_heatmaps_out_mask'], 'b z y x c -> b c z y x')
        else:
            mask = torch.ones_like(targets)

        wandb_log = {}
        with accelerator.accumulate(model):
            target_pred = model(inputs)
            loss = loss_fn(target_pred, targets, mask)
            if torch.isnan(loss):
                raise ValueError('loss is NaN')
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        wandb_log['loss'] = loss.detach().item()
        progress_bar.set_postfix({'loss': loss.detach().item()})
        progress_bar.update(1)

        if iteration % config['log_frequency'] == 0:
            with torch.no_grad():
                model.eval()

                val_batch = next(val_iterator)
                val_inputs, val_targets = prepare_batch(val_batch)
                if 'uv_heatmaps_out_mask' in val_batch:
                    val_mask = rearrange(val_batch['uv_heatmaps_out_mask'], 'b z y x c -> b c z y x')
                else:
                    val_mask = torch.ones_like(val_targets)
                val_target_pred = model(val_inputs)
                val_loss = loss_fn(val_target_pred, val_targets, val_mask)
                wandb_log['val_loss'] = val_loss.item()

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
                    def make_canvas(inputs, targets, target_pred):
                        sample_count = min(inputs.shape[0], log_image_max_samples)
                        inputs = inputs[:sample_count]
                        targets = targets[:sample_count]
                        target_pred = target_pred[:sample_count]

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
                        canvas = torch.stack([
                            torch.cat([inputs_slice(dim) for dim in range(3)], dim=1),
                            projections(F.sigmoid(target_pred)),
                            projections(targets),
                        ], dim=-1)
                        sample_canvases = rearrange(canvas.clip(0, 1), 'b y x rgb v -> b y (v x) rgb').cpu()
                        b, h, w, c = sample_canvases.shape
                        cols = min(log_image_grid_cols, b)
                        rows = math.ceil(b / cols)
                        grid = torch.zeros((rows * h, cols * w, c), dtype=sample_canvases.dtype)
                        for idx in range(b):
                            row, col = divmod(idx, cols)
                            grid[row * h : (row + 1) * h, col * w : (col + 1) * w] = sample_canvases[idx]
                        return grid

                target_pred_for_vis = target_pred[0] if isinstance(target_pred, (list, tuple)) else target_pred
                val_target_pred_for_vis = val_target_pred[0] if isinstance(val_target_pred, (list, tuple)) else val_target_pred

                train_canvas = make_canvas(inputs, targets, target_pred_for_vis)
                val_canvas = make_canvas(val_inputs, val_targets, val_target_pred_for_vis)
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
