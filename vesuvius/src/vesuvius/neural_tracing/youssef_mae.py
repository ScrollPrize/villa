
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange


torch.backends.cuda.enable_flash_sdp(True)


@dataclass
class ModelConfig:
    """Configuration for 3D ViT model parameters"""
    image_size: int = 64
    image_patch_size: int = 16
    frames: int = 16
    frame_patch_size: int = 4
    num_classes: int = 2
    dim: int = 384
    depth: int = 6
    heads: int = 6
    mlp_dim: int = 1024
    channels: int = 1
    dim_head: int = 64
    dropout: float = 0.1
    emb_dropout: float = 0.1
    flash_attn_type: str = 'pytorch'

    @property
    def num_patches_h(self):
        return self.image_size // self.image_patch_size

    @property
    def num_patches_w(self):
        return self.image_size // self.image_patch_size

    @property
    def num_patches_f(self):
        return self.frames // self.frame_patch_size

    @property
    def total_patches(self):
        return self.num_patches_h * self.num_patches_w * self.num_patches_f

    @property
    def expected_output_shape(self):
        """Expected output shape for given batch size"""
        return lambda batch_size: (batch_size, self.num_classes, self.num_patches_f, self.num_patches_h, self.num_patches_w)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class FlashAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., flash_attn_type='pytorch'):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.flash_attn_type = flash_attn_type
        self.dropout = dropout

        self.norm = nn.LayerNorm(dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        # Try to import flash_attn if using that backend
        if flash_attn_type == 'flash_attn':
            try:
                from flash_attn import flash_attn_func
                self.flash_attn_func = flash_attn_func
                # print("Using flash_attn package for attention")
            except ImportError:
                print("flash_attn package not found, falling back to PyTorch SDPA")
                self.flash_attn_type = 'pytorch'

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        # Determine dropout probability based on training mode
        dropout_p = self.dropout if self.training else 0.0

        if self.flash_attn_type == 'flash_attn' and hasattr(self, 'flash_attn_func'):
            # Use dedicated flash_attn package
            # Rearrange for flash_attn: (batch, seqlen, nheads, headdim)
            q = rearrange(q, 'b h n d -> b n h d')
            k = rearrange(k, 'b h n d -> b n h d')
            v = rearrange(v, 'b h n d -> b n h d')

            out = self.flash_attn_func(
                q, k, v,
                dropout_p=dropout_p,
                softmax_scale=self.scale,
                causal=False
            )

            # Rearrange back: (batch, seqlen, nheads, headdim) -> (batch, seqlen, nheads * headdim)
            out = rearrange(out, 'b n h d -> b n (h d)')

        else:
            # Use PyTorch's scaled_dot_product_attention (includes Flash Attention optimizations)
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=dropout_p,
                scale=self.scale,
                is_causal=False
            )
            out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(config.depth):
            self.layers.append(nn.ModuleList([
                FlashAttention(
                    dim=config.dim,
                    heads=config.heads,
                    dim_head=config.dim_head,
                    dropout=config.dropout,
                    flash_attn_type=config.flash_attn_type
                ),
                FeedForward(config.dim, config.mlp_dim, config.dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT3DSegmentation(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Validate configuration
        assert config.image_size % config.image_patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        assert config.frames % config.frame_patch_size == 0, 'Frames must be divisible by frame patch size'

        # Calculate patch dimensions
        patch_dim = config.channels * config.image_patch_size * config.image_patch_size * config.frame_patch_size

        # if config.image_size == config.frames == 256 and config.image_patch_size == config.frame_patch_size == 16 and config.channels == 1:  # i.e. the configuration it was trained for
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (f pf) (h p1) (w p2) -> b (f h w) (p1 p2 pf c)',
                     p1 = config.image_patch_size, p2 = config.image_patch_size, pf = config.frame_patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, config.dim),
            nn.LayerNorm(config.dim),
        )
        # elif config.image_size == config.frames == 128 and config.image_patch_size == config.frame_patch_size == 8 and config.channels == 5:
        #     self.to_patch_embedding = ...

            # No cls token for segmentation - only positional embeddings for patches
        self.pos_embedding = nn.Parameter(torch.randn(1, config.total_patches, config.dim))
        self.dropout = nn.Dropout(config.emb_dropout)

        self.transformer = Transformer(config)

        # Segmentation head - applies to each patch token
        self.segmentation_head = nn.Sequential(
            nn.LayerNorm(config.dim),
            nn.Linear(config.dim, config.num_classes)
        )

    def forward(self, video):
        # video shape: (batch, channels, frames, height, width)
        x = self.to_patch_embedding(video)
        b, n, _ = x.shape

        # Add positional embeddings (no cls token)
        x += self.pos_embedding[:, :n]
        x = self.dropout(x)

        # Process through transformer
        x = self.transformer(x)

        # Apply segmentation head to each patch token
        x = self.segmentation_head(x)  # Shape: (batch, num_patches, num_classes)

        # Reshape back to spatial dimensions
        x = rearrange(x, 'b (f h w) c -> b c f h w',
                     f=self.config.num_patches_f, h=self.config.num_patches_h, w=self.config.num_patches_w)

        return x

    def get_patch_embeddings(self, video):
        """
        Return the patch embeddings before the segmentation head
        Useful for analysis or feature extraction
        """
        x = self.to_patch_embedding(video)
        b, n, _ = x.shape
        x += self.pos_embedding[:, :n]
        x = self.dropout(x)
        x = self.transformer(x)
        return x


def load_mae_encoder_to_vit3d(mae_checkpoint_path, vit3d_model, device='cuda'):
    checkpoint = torch.load(mae_checkpoint_path, map_location=device)
    mae_state_dict = checkpoint['state_dict']

    vit3d_state_dict = vit3d_model.state_dict()

    loaded_keys = []
    shape_mismatches = []

    for mae_key in mae_state_dict:
        if mae_key.startswith('model.') and not any(x in mae_key for x in ['decoder', 'mask_token']):
            vit3d_key = mae_key.replace('model.', '')
            if vit3d_key in vit3d_state_dict:
                if mae_state_dict[mae_key].shape == vit3d_state_dict[vit3d_key].shape:
                    vit3d_state_dict[vit3d_key] = mae_state_dict[mae_key]
                    loaded_keys.append(vit3d_key)
                else:
                    shape_mismatches.append((vit3d_key, mae_state_dict[mae_key].shape, vit3d_state_dict[vit3d_key].shape))

    if shape_mismatches:
        print(f"\nShape mismatches found:")
        for key, mae_shape, vit3d_shape in shape_mismatches:
            print(f"{key}: MAE {mae_shape} != ViT3D {vit3d_shape}")

    vit3d_model.load_state_dict(vit3d_state_dict, strict=False)
    return loaded_keys


class Vesuvius3dViTModel(nn.Module):

    def __init__(self, mae_ckpt_path, in_channels, out_channels, input_size=256, patch_size=16):
        super(Vesuvius3dViTModel, self).__init__()

        # Calculate patch dimensions
        self.patch_size = patch_size
        self.input_size = input_size
        self.output_size = input_size // patch_size
        self.voxel_size = 4
        self.intermediate_size = self.output_size * self.voxel_size
        dim = 512

        self.model = ViT3DSegmentation(
            ModelConfig(
                image_size=input_size,
                image_patch_size=patch_size,
                frames=input_size,
                frame_patch_size=patch_size,
                num_classes=dim,
                channels=in_channels,
                dim=dim,
                depth=16,
                heads=16,
                mlp_dim=1024,
                flash_attn_type='flash_attn'
            )
        )

        if mae_ckpt_path is not None:
            loaded = load_mae_encoder_to_vit3d(mae_ckpt_path, self.model)
            print(f"Loaded {len(loaded)} encoder weights from MAE")

        if False:
            for param in self.model.parameters():
                param.requires_grad = False

        self.voxel_channels = 48
        self.token_decoder = nn.Linear(dim, self.voxel_size**3 * self.voxel_channels)

        self.conv_decoder = nn.Sequential(
            nn.Conv3d(self.voxel_channels, self.voxel_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(self.voxel_channels, out_channels, 3, padding=1)
        )

        print(f"Model initialized:")
        print(f"  Input size: {input_size}x{input_size}x{input_size}")
        print(f"  Patch size: {patch_size}x{patch_size}x{patch_size}")
        print(f"  Token grid: {self.output_size}x{self.output_size}x{self.output_size}")
        print(f"  Intermediate size: {self.intermediate_size}x{self.intermediate_size}x{self.intermediate_size}")
        print(f"  Voxel size per token: {self.voxel_size}x{self.voxel_size}x{self.voxel_size}")

    def forward(self, x, unused_cond):
        batch_size = x.shape[0]
        embeddings = self.model(x)

        if embeddings.dim() == 5:
            embeddings = embeddings.view(batch_size, embeddings.shape[1], -1).transpose(1, 2)
        elif embeddings.dim() == 3:
            pass
        else:
            raise ValueError(f"Unexpected ViT output shape: {embeddings.shape}")

        num_tokens = embeddings.shape[1]
        expected_tokens = self.output_size ** 3

        if num_tokens != expected_tokens:
            print(f"Warning: Expected {expected_tokens} tokens, got {num_tokens}")

        decoded_tokens = self.token_decoder(embeddings)

        decoded_tokens = decoded_tokens.view(
            batch_size, self.voxel_channels,
            self.output_size, self.output_size, self.output_size,
            self.voxel_size, self.voxel_size, self.voxel_size
        )

        intermediate_vol = decoded_tokens.permute(0, 1, 2, 5, 3, 6, 4, 7).contiguous()
        intermediate_vol = intermediate_vol.view(
            batch_size, self.voxel_channels,
            self.intermediate_size,
            self.intermediate_size,
            self.intermediate_size
        )

        decoded_vol = self.conv_decoder(intermediate_vol)

        output = F.interpolate(
            decoded_vol,
            size=(self.input_size, self.input_size, self.input_size),
            mode='trilinear',
            align_corners=False
        )

        return output
