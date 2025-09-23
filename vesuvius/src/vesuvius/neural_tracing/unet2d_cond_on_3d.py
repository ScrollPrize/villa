
import torch
import diffusers
import numpy as np
from einops import rearrange

from resnet3d import ResNet3DEncoder


class PatchConditionedOn3dModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.denoiser = diffusers.models.UNet2DConditionModel(
            sample_size=config['patch_size'],
            in_channels=3,
            out_channels=3,
            encoder_hid_dim=128,
            down_block_types=(
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "DownBlock2D",
            ),
            mid_block_type="UNetMidBlock2DCrossAttn",
            up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
            block_out_channels=(320, 640, 1280),
        )
        self.encoder = ResNet3DEncoder(in_channels=4, channels=[32, 64, 96, 128], blocks=[2, 2, 2, 2])
        self.register_buffer('position_embeddings', torch.from_numpy(np.indices((config['crop_size'], config['crop_size'], config['crop_size']), dtype=np.float32)))

    def forward(self, inputs, timesteps, volume):
        # TODO: consider adding position-embedding only after resnet
        conditioning = self.encoder(torch.cat([torch.tile(self.position_embeddings, (volume.shape[0], 1, 1, 1, 1)), volume.unsqueeze(1)], dim=1))
        conditioning = rearrange(conditioning, 'b c z y x -> b (z y x) c')
        return self.denoiser(inputs, timesteps, encoder_hidden_states=conditioning)

