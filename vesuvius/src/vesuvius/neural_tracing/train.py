
import os
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

from dataset import PatchInCubeDataset, HeatmapDatasetV2, load_datasets, make_heatmaps
from models import make_model


def recrop(x, center, size):
    return x[
        ...,
        center[0] - size // 2 : center[0] + size // 2,
        center[1] - size // 2 : center[1] + size // 2,
        center[2] - size // 2 : center[2] + size // 2
    ]


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

    # FIXME: need separate data-loaders for multi-step and single-step training, since have different target shapes

    model = make_model(config)

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

    def loss_fn(target_pred, targets, mask, reduce_batch=True):
        if config['binary']:
            targets_binary = (targets > 0.5).long()  # FIXME: should instead not do the gaussian conv in data-loader!
            if reduce_batch:
                from vesuvius.models.training.loss.nnunet_losses import DC_and_BCE_loss
                loss = DC_and_BCE_loss(bce_kwargs={}, soft_dice_kwargs={'ddp': False})(target_pred, targets_binary)
                return loss
            else:
                # FIXME: nasty; fix DC_and_BCE_loss to support this directly
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
            return per_batch.mean() if reduce_batch else per_batch
            
    progress_bar = tqdm(total=config['num_iterations'], disable=not accelerator.is_local_main_process)
    for iteration, batch in enumerate(train_dataloader):

        inputs, targets = prepare_batch(batch, None, config['crop_size'])
        mask = torch.ones_like(targets[:, :1, ...])  # TODO!

        wandb_log = {}
        with accelerator.accumulate(model):
            target_pred = model(inputs)
            if targets.shape[1] > target_pred.shape[1]:
                # Multi-step training -- step_count points predicted per network output; multistep_count such predictions chained together
                # TODO: support patch topology other than chain!
                assert targets.shape[1] // target_pred.shape[1] == config['multistep_count']

                first_step_targets = targets[:, ::config['multistep_count']]
                first_step_pred = target_pred
                first_step_loss = loss_fn(first_step_pred, first_step_targets, mask)

                sample_count = config['multistep_samples']

                # FIXME: this is a nasty hack to determine which direction (U/V) we should move along, under
                #  the assumption that exactly one of prev_u and prev_v is given
                step_directions = batch['uv_heatmaps_in'].amax(dim=[1, 2, 3])[:, :2].argmax(dim=-1)
                first_step_pred_for_dir = first_step_pred[torch.arange(target_pred.shape[0]), step_directions]

                # Sample `sample_count` points from a small cube around the argmax
                # TODO: try other strategies, e.g. sample directly from categorical distribution parametrised by (high-temperature) heatmap
                cube_radius = 4
                cube_center = torch.argmax(first_step_pred_for_dir.view(first_step_pred_for_dir.shape[0], -1), dim=1)
                cube_center = torch.stack(torch.unravel_index(cube_center, first_step_pred_for_dir.shape[1:]), dim=-1)  # batch, zyx
                cube_center = torch.clamp(cube_center, torch.tensor(cube_radius).to(cube_center.device), torch.tensor(first_step_pred_for_dir.shape[1:], device=cube_center.device) - cube_radius - 1)
                first_sample_zyxs_in_first_subcrop = torch.randint(-cube_radius, cube_radius + 1, [1, sample_count, 3], device=cube_center.device) + cube_center[:, None, :]  # batch, sample, zyx

                # Use the corresponding probabilities (at somewhat raised temperature), renormalised, as weights for each of the samples
                sample_logits = first_step_pred_for_dir[
                    torch.arange(first_step_pred.shape[0], device=first_step_pred.device)[:, None].expand(-1, sample_count),
                    first_sample_zyxs_in_first_subcrop[..., 0],
                    first_sample_zyxs_in_first_subcrop[..., 1],
                    first_sample_zyxs_in_first_subcrop[..., 2]
                ]
                weight_temperature = 1.e-1  # smooth out the probabilities a bit
                sample_weights = F.sigmoid(sample_logits * weight_temperature)
                sample_weights = sample_weights / sample_weights.sum(dim=1, keepdim=True)  # batch, sample

                # TODO: support transpose augmentation (or become sure that it works already...)
                assert not config['augmentation']['allow_transposes']

                first_sample_zyxs_in_outer_crop = first_sample_zyxs_in_first_subcrop + (torch.tensor(batch['volume'].shape[-3:], device=first_sample_zyxs_in_first_subcrop.device) - config['crop_size']) // 2

                loss = first_step_loss
                step_idx = 1  # TODO...
                for sample_idx in range(sample_count):
                    first_sample_zyx = first_sample_zyxs_in_outer_crop[:, sample_idx, :]
                    min_corner_new_subcrop_in_outer = first_sample_zyx - config['crop_size'] // 2

                    # Prepare the conditioning prev_u/_v; this is in the space of the new subcrop that's centered at
                    # first_sample_zyx, and marks the position that was the center of the first subcrop
                    outer_crop_shape = torch.tensor(batch['volume'].shape[-3:], device=first_sample_zyx.device)
                    first_center_in_outer_crop = outer_crop_shape // 2
                    prev_heatmap = torch.cat([
                        make_heatmaps([first_center_in_outer_crop[None]], min_corner_new_subcrop_in_outer[iib], config['crop_size'])
                        for iib in range(min_corner_new_subcrop_in_outer.shape[0])
                    ], dim=0).to(first_center_in_outer_crop.device)
                    prev_uv = torch.zeros([prev_heatmap.shape[0], 2, *prev_heatmap.shape[1:]], device=prev_heatmap.device, dtype=prev_heatmap.dtype)
                    prev_uv[torch.arange(prev_heatmap.shape[0]), step_directions] = prev_heatmap

                    # Prepare the volume (sub-)crop and localiser. For the localiser we take the original since
                    # we still want the center of it (conceptually we create a new localiser at the new center)
                    max_corner_new_subcrop_in_outer = min_corner_new_subcrop_in_outer + config['crop_size']
                    step_volume_crop = torch.stack([
                        batch['volume'][
                            iib,
                            min_corner_new_subcrop_in_outer[iib, 0] : max_corner_new_subcrop_in_outer[iib, 0],
                            min_corner_new_subcrop_in_outer[iib, 1] : max_corner_new_subcrop_in_outer[iib, 1],
                            min_corner_new_subcrop_in_outer[iib, 2] : max_corner_new_subcrop_in_outer[iib, 2]
                        ]
                        for iib in range(min_corner_new_subcrop_in_outer.shape[0])
                    ], dim=0)
                    # FIXME: this makes assumptions about how prepare_batch arranges stuff; can we unify?
                    step_inputs = torch.cat([
                        step_volume_crop.unsqueeze(1),
                        inputs[:, 1:2],  # borrow the original localiser subcrop
                        prev_uv,  # prev_u, prev_v
                        torch.zeros_like(prev_uv[:, :1]),  # prev_diag
                    ], dim=1)

                    step_pred = model(step_inputs)

                    step_targets = torch.stack([
                        batch['uv_heatmaps_out'][
                            iib,
                            min_corner_new_subcrop_in_outer[iib, 0] : max_corner_new_subcrop_in_outer[iib, 0],
                            min_corner_new_subcrop_in_outer[iib, 1] : max_corner_new_subcrop_in_outer[iib, 1],
                            min_corner_new_subcrop_in_outer[iib, 2] : max_corner_new_subcrop_in_outer[iib, 2]
                        ]
                        for iib in range(min_corner_new_subcrop_in_outer.shape[0])
                    ], dim=0)
                    step_targets = rearrange(step_targets[..., step_idx::config['multistep_count']], 'b z y x c -> b c z y x')

                    # Since the model runs in single-cond mode for this step, it predicts a point along
                    # the cond direction, but also one/two along the other direction; those others are
                    # not included in gt targets (because they're not part of the chain). We therefore
                    # only take targets and preds in the direction of the along-chain conditioning
                    step_pred = step_pred[torch.arange(step_pred.shape[0]), step_directions].unsqueeze(1)
                    step_targets = step_targets[torch.arange(step_targets.shape[0]), step_directions].unsqueeze(1)

                    step_loss = loss_fn(step_pred, step_targets, mask, reduce_batch=False)
                    loss += (step_loss * sample_weights[..., sample_idx]).sum()

            else:
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
                val_inputs, val_targets = prepare_batch(val_batch, None, config['crop_size'])
                val_mask = torch.ones_like(val_targets[:, :1, ...])  # TODO!
                val_target_pred = model(val_inputs)
                if val_targets.shape[1] > val_target_pred.shape[1]:
                    # TODO: calculate multistep loss if enabled
                    val_targets = val_targets[:, ::config['multistep_count']]
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
                        colours_by_step = torch.rand([target_pred.shape[1], 3], device=inputs.device) * 0.7 + 0.2
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
                        if targets.shape[1] > target_pred.shape[1]:
                            assert targets.shape[1] // target_pred.shape[1] == config['multistep_count']
                            targets_vis = rearrange(targets, 'b (uv s) z y x -> b uv s z y x', uv=2).amax(dim=2)
                            # TODO: also visualise multi-step predictions
                        else:
                            targets_vis = targets
                        canvas = torch.stack([
                            torch.cat([inputs_slice(dim) for dim in range(3)], dim=1),
                            projections(F.sigmoid(target_pred)),
                            projections(targets_vis),
                        ], dim=-1)
                        return rearrange(canvas.clip(0, 1), 'b y x rgb v -> (b y) (v x) rgb').cpu()

                train_canvas = make_canvas(inputs, targets, target_pred)
                val_canvas = make_canvas(val_inputs, val_targets, val_target_pred)
                plt.imsave(f'{out_dir}/{iteration:06}_train.png', train_canvas)
                plt.imsave(f'{out_dir}/{iteration:06}_val.png', val_canvas)

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
