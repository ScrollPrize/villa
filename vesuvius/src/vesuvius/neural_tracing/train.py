
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

from dataset import PatchInCubeDataset, HeatmapDataset
from vesuvius_unet3d import Vesuvius3dUnetModel
from song_unet3d import SongUnet3dModel
from sampling import sample_ddim


def prepare_batch(batch, noise_scheduler, generative, cfg_uncond_prob):
    clean_heatmaps = batch['uv_heatmaps']
    if generative:
        # FIXME: should use non-uniform timestep distribution
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, size=[clean_heatmaps.shape[0]], device=clean_heatmaps.device).long()
        noise = torch.randn_like(clean_heatmaps)
        noisy_heatmaps = noise_scheduler.add_noise(clean_heatmaps, noise, timesteps)
        conditioning_mask = 0 if torch.rand(clean_heatmaps.shape[0]) < cfg_uncond_prob else 1
    else:
        timesteps = torch.zeros(clean_heatmaps.shape[0], device=clean_heatmaps.device, dtype=torch.long)
        noisy_heatmaps = torch.zeros_like(clean_heatmaps)
        conditioning_mask = 1
    if (not generative) or noise_scheduler.config.prediction_type == 'sample':
        targets = clean_heatmaps
    elif noise_scheduler.config.prediction_type == 'v_prediction':
        targets = noise_scheduler.get_velocity(clean_heatmaps, noise, timesteps)
    elif noise_scheduler.config.prediction_type == 'epsilon':
        targets = noise
    else:
        assert False
    inputs = torch.cat([
        batch['volume'].unsqueeze(1) * conditioning_mask,
        batch['localiser'].unsqueeze(1) * conditioning_mask,  # TODO: should this one be removed for cfg?
        rearrange(noisy_heatmaps, 'b z y x c -> b c z y x')
    ], dim=1)
    targets = rearrange(targets, 'b z y x c -> b c z y x')
    return timesteps, inputs, targets


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
    )

    if 'wandb_project' in config and accelerator.is_main_process:
        wandb.init(project=config['wandb_project'], entity=config.get('wandb_entity', None), config=config)

    if config['representation'] == 'heatmap':
        dataset = HeatmapDataset(config)
    else:
        dataset = PatchInCubeDataset(config)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'], num_workers=config['num_workers'])

    # model = Vesuvius3dUnetModel(in_channels=18, out_channels=16, config=config)
    model = SongUnet3dModel(
        img_resolution=config['crop_size'],
        in_channels=18,
        out_channels=16,
        model_channels=32,
        channel_mult=[1, 2, 4, 8],
        num_blocks=2,  # 4
        encoder_type='residual',
        dropout=0.1,
        attn_resolutions=[16, 8],
    )

    noise_scheduler = diffusers.DDPMScheduler(
        num_train_timesteps=1000,
        prediction_type='sample',  # TODO: try v-prediction?
        beta_schedule='linear',  # squaredcos_cap_v2 should give even-smaller SNR at T=1000
        timestep_spacing='trailing',  # ...so we 'see' step 1000 during inference
        clip_sample=True,  # FIXME: does this mean we should tell the sampler use_clipped_model_output also?
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    lr_scheduler = diffusers.optimization.get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config['lr_warmup_steps'],
        num_training_steps=config['num_iterations'] * accelerate.PartialState().num_processes,  # FIXME: nasty adjustment by num_processes here accounts for the fact that accelerator's prepare_scheduler weirdly causes the scheduler to take num_processes scheduler-steps per optimiser-step
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    progress_bar = tqdm(total=config['num_iterations'], disable=not accelerator.is_local_main_process)
    for iteration, batch in enumerate(train_dataloader):

        timesteps, inputs, targets = prepare_batch(batch, noise_scheduler, config['generative'], config['cfg_uncond_prob'])
        mask = torch.ones_like(targets[:, :1, ...])  # TODO!

        wandb_log = {}
        with accelerator.accumulate(model):
            target_pred = model(inputs, timesteps)
            # TODO: should this instead weight each element in batch equally regardless of valid area?
            loss = ((target_pred - targets)**2 * mask).sum() / mask.sum()
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
                plt.imsave(f'{out_dir}/{iteration:06}_denoised.png', canvas.cpu())

                eval_num_samples = 4
                assert targets.shape[0] == 1  # since the argmin needs to be per-iib otherwise, and we need either a loop or a separate dimensions for sample vs batch
                gt_heatmaps = rearrange(batch['uv_heatmaps'], 'b z y x c -> b c z y x')
                if config['generative']:
                    sample_heatmaps = sample_ddim(
                        model,
                        torch.tile(inputs[:, :2], (eval_num_samples, 1, 1, 1, 1)),
                        noise_scheduler,
                        (eval_num_samples, *targets.shape[1:]),
                        torch.Generator().manual_seed(config['seed']),
                        cfg_scale=config['cfg_scale']
                    )
                    sample_heatmaps = sample_heatmaps[torch.argmin((sample_heatmaps - gt_heatmaps).abs().mean(dim=(1, 2, 3, 4)))]
                else:
                    sample_heatmaps = target_pred.squeeze(0)
                canvas = torch.stack([inputs[:, :1].expand(-1, 3, -1, -1, -1), squish(sample_heatmaps[None]), squish(gt_heatmaps)], dim=-1)
                canvas_mask = torch.stack([torch.ones_like(mask), torch.ones_like(mask), mask], dim=-1)
                canvas = (canvas * 0.5 + 0.5).clip(0, 1) * canvas_mask
                canvas = rearrange(canvas[:, :, canvas.shape[2] // 2], 'b uvw y x v -> (b y) (v x) uvw')
                plt.imsave(f'{out_dir}/{iteration:06}_sample.png', canvas.cpu())

                model.train()

        if iteration % config['ckpt_frequency'] == 0:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'noise_scheduler': noise_scheduler.config,
                'step': iteration
            }, f'{out_dir}/ckpt_{iteration:06}.pth' )

        if wandb.run is not None:
            wandb.log(wandb_log)


if __name__ == '__main__':
    train()
