
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

from dataset import PatchWithCubeDataset
from resnet3d import ResNet3DEncoder


class Model(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.denoiser = diffusers.models.UNet2DConditionModel(
            sample_size=config['patch_size'],
            in_channels=3,
            out_channels=3,
            encoder_hid_dim=128,
        )
        self.encoder = ResNet3DEncoder(in_channels=1, channels=[32, 64, 96, 128], blocks=[2, 2, 2, 2])

    def forward(self, inputs, timesteps, volume):
        conditioning = self.encoder(volume.unsqueeze(1))
        conditioning = rearrange(conditioning, 'b c z y x -> b (z y x) c')
        # TODO: positional encoding! either before the resnet or after...
        return self.denoiser(inputs, timesteps, encoder_hidden_states=conditioning)


def prepare_batch(batch, noise_scheduler):
    clean_zyxs = batch['patch_zyx']
    # FIXME: should use non-uniform timestep distribution
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, size=[clean_zyxs.shape[0]], device=clean_zyxs.device).long()
    noise = torch.randn_like(clean_zyxs)
    noisy_zyxs = noise_scheduler.add_noise(clean_zyxs, noise, timesteps)
    if noise_scheduler.config.prediction_type == 'v_prediction':
        targets = noise_scheduler.get_velocity(clean_zyxs, noise, timesteps)
    elif noise_scheduler.config.prediction_type == 'epsilon':
        targets = noise
    elif noise_scheduler.config.prediction_type == 'sample':
        targets = clean_zyxs
    else:
        assert False
    inputs = rearrange(noisy_zyxs, 'b y x c -> b c y x')
    targets = rearrange(targets, 'b y x c -> b c y x')
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

    dataset = PatchWithCubeDataset(config)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'], num_workers=config['num_workers'])

    model = Model(config)

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
        timesteps, inputs, targets = prepare_batch(batch, noise_scheduler)
        inputs *= 0.  # TODO: remove this! means we just learn deterministic mapping from conditioning volume to target
        mask = batch['patch_valid'].unsqueeze(1)
        wandb_log = {}
        with accelerator.accumulate(model):
            model_output = model(inputs, timesteps, batch['volume'])
            target_pred = model_output['sample']
            # TODO: should this instead weight each element in batch equally regardless of valid area?
            loss = ((target_pred - targets)**2 * mask).sum() / mask.sum()
            if torch.isnan(loss):
                raise ValueError('loss is NaN')
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        wandb_log['loss'] = loss.detach().item()
        progress_bar.set_postfix({'loss': loss.detach().item()})
        progress_bar.update(1)
        if iteration % config['log_frequency'] == 0:
            with torch.no_grad():
                canvas = torch.stack([inputs, target_pred, targets], dim=-1)
                canvas_mask = torch.stack([mask, torch.ones_like(mask), mask], dim=-1)
                canvas = (canvas * 0.5 + 0.5).clip(0, 1) * canvas_mask
                canvas = rearrange(canvas, 'b zyx y x v -> (b y) (v x) zyx')
            plt.imsave(f'{out_dir}/{iteration:06}.png', canvas.cpu())
        if wandb.run is not None:
            wandb.log(wandb_log)


@click.command()
@click.argument('config_path', type=click.Path(exists=True))
def infer(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    print(f"Inference with config: {config_path}")
    # TODO: Implement inference logic using config


@click.group()
def main():
    pass

main.add_command(train)
main.add_command(infer)

if __name__ == '__main__':
    main()
