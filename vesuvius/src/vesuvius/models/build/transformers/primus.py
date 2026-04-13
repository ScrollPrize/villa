from typing import Tuple
import math
import torch
from torch import nn

from timm.layers import RotaryEmbeddingCat
from .abstract_arch import (
    AbstractDynamicNetworkArchitectures,
    test_submodules_loadable,
)
from .eva import Eva
from .patch_encode_decode import LayerNormNd, PatchDecode, PatchEmbed_deeper
from vesuvius.models.utils import InitWeights_He
from einops import rearrange


_PRIMUS_CONFIGS = {
    "S": {
        "eva_depth": 12,
        "eva_numheads": 6,
        "embed_dim": 396,
    },
    "B": {
        "eva_depth": 12,
        "eva_numheads": 12,
        "embed_dim": 792,
    },
    "M": {
        "eva_depth": 16,
        "eva_numheads": 12,
        "embed_dim": 864,
    },
    "L": {
        "eva_depth": 24,
        "eva_numheads": 16,
        "embed_dim": 1056,
    },
}


class Primus(AbstractDynamicNetworkArchitectures):

    def __init__(
        self,
        input_channels: int,
        embed_dim: int,
        patch_embed_size: Tuple[int, ...],
        num_classes: int,
        eva_depth: int = 24,
        eva_numheads: int = 16,
        input_shape: Tuple[int, ...] = None,
        decoder_norm=LayerNormNd,
        decoder_act=nn.GELU,
        num_register_tokens: int = 0,
        use_rot_pos_emb: bool = True,
        use_abs_pos_embed: bool = True,
        pos_emb_type: str = "rope",
        mlp_ratio=4 * 2 / 3,
        drop_path_rate=0,  # drops computations (multihead attention, mlp), Implementation of scaling might be useless here because this is not batch normed
        patch_drop_rate: float = 0.0,  # drops input patches, may be used for MAE style pretraining
        proj_drop_rate: float = 0.0,  # drops out things related to the projection. That is in the MLP and at the end of EVA attention
        attn_drop_rate: float = 0.0,  # drops attention, meaning connections between patches may bebroken up at random
        rope_impl=RotaryEmbeddingCat,
        rope_kwargs=None,
        init_values=None,
        scale_attn_inner=False,
    ):
        """
        Architecture as proposed in the Primus paper (https://arxiv.org/pdf/2503.01835)
        `Primus: Enforcing Attention Usage for 3D Medical Image Segmentation`

        consists of simple patch_embedding, a EVA ViT encoder with a few adatptations and a simple patch decoder.
        """
        assert input_shape is not None
        assert len(input_shape) in (2, 3), "Only 2D and 3D inputs are supported"
        assert len(patch_embed_size) == len(input_shape), "patch_embed_size dimensionality must match input_shape"
        assert all([j % i == 0 for i, j in zip(patch_embed_size, input_shape)])
        assert tuple(patch_embed_size) == tuple([8] * len(input_shape)), (
            f"deeper patch embedding downsamples by 8x. Use patch_embed_size={tuple([8] * len(input_shape))}."
        )
        self.ndim = len(input_shape)
        self._input_shape = tuple(input_shape)

        super().__init__()
        self.key_to_encoder = "eva"
        self.key_to_stem = "down_projection"
        self.keys_to_in_proj = (
            "down_projection.stem.blocks.0.conv1.conv",
            "down_projection.stem.blocks.0.conv1.all_modules.0",
        )
        self.key_to_lpe = "eva.pos_embed"

        self.down_projection = PatchEmbed_deeper(
            input_channels=input_channels,
            embed_dim=embed_dim,
            base_features=32,
            ndim=self.ndim,
            depth_per_level=(1, 1, 1),
            embed_proj_3x3x3=False,
            embed_block_style="residual",
            embed_block_type="basic",
        )
        self.up_projection = PatchDecode(
            patch_embed_size, embed_dim, num_classes, norm=decoder_norm, activation=decoder_act
        )

        # we need to compute the ref_feat_shape for eva
        self.eva = Eva(
            embed_dim=embed_dim,
            depth=eva_depth,
            num_heads=eva_numheads,
            ref_feat_shape=tuple([i // ds for i, ds in zip(input_shape, patch_embed_size)]),
            num_reg_tokens=num_register_tokens,
            use_rot_pos_emb=use_rot_pos_emb,
            use_abs_pos_emb=use_abs_pos_embed,
            pos_emb_type=pos_emb_type,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
            patch_drop_rate=patch_drop_rate,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            rope_impl=rope_impl,
            rope_kwargs=rope_kwargs,
            init_values=init_values,
            scale_attn_inner=scale_attn_inner,
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=1e-6)

        if num_register_tokens > 0:
            self.register_tokens = (
                nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim)) if num_register_tokens else None
            )
            nn.init.normal_(self.register_tokens, std=1e-6)
        else:
            self.register_tokens = None

        self.down_projection.apply(InitWeights_He(1e-2))
        self.up_projection.apply(InitWeights_He(1e-2))
        # eva has its own initialization

    def restore_full_sequence(self, x, keep_indices, num_patches):
        """
        Restore the full sequence by filling blanks with mask tokens and reordering.
        """
        if keep_indices is None:
            return x, None

        B, num_kept, C = x.shape
        device = x.device

        # Initialize with mask tokens expanded to full sequence
        restored = self.mask_token.expand(B, num_patches, -1).clone()
        restored_mask = torch.zeros(B, num_patches, dtype=torch.bool, device=device)


        # keep_indices shape: (B, num_kept)
        batch_indices = torch.arange(B, device=device)[:, None].expand(-1, num_kept)

        # Assign kept patches to their positions
        restored[batch_indices, keep_indices] = x
        restored_mask[batch_indices, keep_indices] = True

        return (restored, restored_mask)

    def forward(self, x, ret_mask=False):
        expected_ndim = self.ndim + 2
        if x.ndim != expected_ndim:
            raise ValueError(
                f"Primus expected input with {expected_ndim} dims "
                f"(B, C, {', '.join(['spatial'] * self.ndim)}), got shape {tuple(x.shape)}."
            )

        spatial = tuple(x.shape[2:])
        if spatial != tuple(self._input_shape):
            raise ValueError(
                "Primus received a runtime spatial shape that does not match the configured input_shape. "
                f"Expected {tuple(self._input_shape)}, got {spatial}. "
                "Primus uses fixed positional embeddings and requires a consistent patch size."
            )

        full_spatial = x.shape[2:]  # Full spatial dimensions (H, W) or (D, H, W)
        x = self.down_projection(x)

        B, C = x.shape[:2]
        spatial_shape = x.shape[2:]  # (H, W) or (D, H, W)
        num_patches = math.prod(spatial_shape)

        if self.ndim == 2:
            x = rearrange(x, "b c h w -> b (h w) c")
        else:
            x = rearrange(x, "b c d h w -> b (d h w) c")

        if self.register_tokens is not None:
            x = torch.cat(
                (
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x,
                ),
                dim=1,
            )
        x, keep_indices = self.eva(x)

        if self.register_tokens is not None:
            x = x[:, self.register_tokens.shape[1] :]  # Removes the register tokens

        # In-fill in-active patches with empty tokens
        restored_x, restoration_mask = self.restore_full_sequence(x, keep_indices, num_patches)

        # Rearrange back to spatial format
        if self.ndim == 2:
            H, W = spatial_shape
            x = rearrange(restored_x, "b (h w) c -> b c h w", h=H, w=W)
            if restoration_mask is not None:
                mask = rearrange(restoration_mask, "b (h w) -> b h w", h=H, w=W)
                full_mask = (
                    mask.repeat_interleave(full_spatial[0] // H, dim=1)
                    .repeat_interleave(full_spatial[1] // W, dim=2)
                )
                full_mask = full_mask[:, None, ...]  # Add channel dimension [B, 1, H, W]
            else:
                full_mask = None
        else:
            D, H, W = spatial_shape
            x = rearrange(restored_x, "b (d h w) c -> b c d h w", d=D, h=H, w=W)
            if restoration_mask is not None:
                mask = rearrange(restoration_mask, "b (d h w) -> b d h w", d=D, h=H, w=W)
                full_mask = (
                    mask.repeat_interleave(full_spatial[0] // D, dim=1)
                    .repeat_interleave(full_spatial[1] // H, dim=2)
                    .repeat_interleave(full_spatial[2] // W, dim=3)
                )
                full_mask = full_mask[:, None, ...]  # Add channel dimension [B, 1, D, H, W]
            else:
                full_mask = None

        dec_out = self.up_projection(x)
        if ret_mask:
            return dec_out, full_mask
        else:
            return dec_out

    def compute_conv_feature_map_size(self, input_size):
        raise NotImplementedError("yuck")


class PrimusX(Primus):

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        config_name: str,
        patch_embed_size: Tuple[int, ...],
        input_shape: Tuple[int, ...] = None,
        drop_path_rate=0,  # drops computations (multihead attention, mlp), Implementation of scaling might be useless here because this is not batch normed
        patch_drop_rate: float = 0.0,  # drops input patches, may be used for MAE style pretraining
        rope_impl=RotaryEmbeddingCat,
        rope_kwargs=None,
        init_values=None,
        scale_attn_inner=False,
    ):
        conf = _PRIMUS_CONFIGS[config_name]
        super().__init__(
            input_channels=input_channels,
            embed_dim=conf["embed_dim"],
            patch_embed_size=patch_embed_size,
            num_classes=output_channels,
            eva_depth=conf["eva_depth"],
            eva_numheads=conf["eva_numheads"],
            input_shape=input_shape,
            drop_path_rate=drop_path_rate,
            patch_drop_rate=patch_drop_rate,
            rope_impl=rope_impl,
            rope_kwargs=rope_kwargs,
            init_values=init_values,
            scale_attn_inner=scale_attn_inner,
        )


class PrimusS(PrimusX):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        patch_embed_size: Tuple[int, ...],
        input_shape: Tuple[int, ...] = None,
        drop_path_rate=0,  # drops computations (multihead attention, mlp), Implementation of scaling might be useless here because this is not batch normed
        patch_drop_rate: float = 0.0,  # drops input patches, may be used for MAE style pretraining
        rope_impl=RotaryEmbeddingCat,
        rope_kwargs=None,
        init_values=0.1,
        scale_attn_inner=True,
    ):
        """
        Official Primus-S Architecture as proposed in the Primus paper (https://arxiv.org/pdf/2503.01835)
        `Primus: Enforcing Attention Usage for 3D Medical Image Segmentation`

        consists of simple patch_embedding, a EVA ViT encoder with a few adatptations and a simple patch decoder.
        """
        super().__init__(
            input_channels=input_channels,
            output_channels=output_channels,
            config_name="S",
            patch_embed_size=patch_embed_size,
            input_shape=input_shape,
            drop_path_rate=drop_path_rate,
            patch_drop_rate=patch_drop_rate,
            rope_impl=rope_impl,
            rope_kwargs=rope_kwargs,
            init_values=init_values,
            scale_attn_inner=scale_attn_inner,
        )


class PrimusB(PrimusX):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        patch_embed_size: Tuple[int, ...],
        input_shape: Tuple[int, ...] = None,
        drop_path_rate=0,  # drops computations (multihead attention, mlp), Implementation of scaling might be useless here because this is not batch normed
        patch_drop_rate: float = 0.0,  # drops input patches, may be used for MAE style pretraining
        rope_impl=RotaryEmbeddingCat,
        rope_kwargs=None,
        init_values=0.1,
        scale_attn_inner=True,
    ):
        """
        Official Primus-B Architecture as proposed in the Primus paper (https://arxiv.org/pdf/2503.01835)
        `Primus: Enforcing Attention Usage for 3D Medical Image Segmentation`

        consists of simple patch_embedding, a EVA ViT encoder with a few adatptations and a simple patch decoder.
        """
        super().__init__(
            input_channels=input_channels,
            output_channels=output_channels,
            config_name="B",
            patch_embed_size=patch_embed_size,
            input_shape=input_shape,
            drop_path_rate=drop_path_rate,
            patch_drop_rate=patch_drop_rate,
            rope_impl=rope_impl,
            rope_kwargs=rope_kwargs,
            init_values=init_values,
            scale_attn_inner=scale_attn_inner,
        )


class PrimusM(PrimusX):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        patch_embed_size: Tuple[int, ...],
        input_shape: Tuple[int, ...] = None,
        drop_path_rate=0,  # drops computations (multihead attention, mlp), Implementation of scaling might be useless here because this is not batch normed
        patch_drop_rate: float = 0.0,  # drops input patches, may be used for MAE style pretraining
        rope_impl=RotaryEmbeddingCat,
        rope_kwargs=None,
        init_values=0.1,
        scale_attn_inner=True,
    ):
        """
        Official Primus-M Architecture as proposed in the Primus paper (https://arxiv.org/pdf/2503.01835)
        `Primus: Enforcing Attention Usage for 3D Medical Image Segmentation`

        consists of simple patch_embedding, a EVA ViT encoder with a few adatptations and a simple patch decoder.
        """
        super().__init__(
            input_channels=input_channels,
            output_channels=output_channels,
            config_name="M",
            patch_embed_size=patch_embed_size,
            input_shape=input_shape,
            drop_path_rate=drop_path_rate,
            patch_drop_rate=patch_drop_rate,
            rope_impl=rope_impl,
            rope_kwargs=rope_kwargs,
            init_values=init_values,
            scale_attn_inner=scale_attn_inner,
        )


class PrimusL(PrimusX):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        patch_embed_size: Tuple[int, ...],
        input_shape: Tuple[int, ...] = None,
        drop_path_rate=0,  # drops computations (multihead attention, mlp), Implementation of scaling might be useless here because this is not batch normed
        patch_drop_rate: float = 0.0,  # drops input patches, may be used for MAE style pretraining
        rope_impl=RotaryEmbeddingCat,
        rope_kwargs=None,
        init_values=0.1,
        scale_attn_inner=True,
    ):
        """
        Official Primus-L Architecture as proposed in the Primus paper (https://arxiv.org/pdf/2503.01835)
        `Primus: Enforcing Attention Usage for 3D Medical Image Segmentation`

        consists of simple patch_embedding, a EVA ViT encoder with a few adatptations and a simple patch decoder.
        """
        super().__init__(
            input_channels=input_channels,
            output_channels=output_channels,
            config_name="L",
            patch_embed_size=patch_embed_size,
            input_shape=input_shape,
            drop_path_rate=drop_path_rate,
            patch_drop_rate=patch_drop_rate,
            rope_impl=rope_impl,
            rope_kwargs=rope_kwargs,
            init_values=init_values,
            scale_attn_inner=scale_attn_inner,
        )


if __name__ == "__main__":
    from fvcore.nn import parameter_count, FlopCountAnalysis, parameter_count_table
    import time

    print("Primus S")
    x = torch.rand([1, 1, 96, 96, 96], device="cuda", dtype=torch.float32)
    model = PrimusS(1, 2, (8, 8, 8), (96, 96, 96)).cuda()
    _ = model(x)
    print(f"Parameter count: {parameter_count(model)[''] / 1e6:.2f}M")
    print(FlopCountAnalysis(model, x))
    print(parameter_count_table(model, max_depth=2))

    test_submodules_loadable(model)

    time.sleep(5)
    print("Primus B")
    model = PrimusB(1, 2, (8, 8, 8), (96, 96, 96)).cuda()
    _ = model(x)
    print(f"Parameter count: {parameter_count(model)[''] / 1e6:.2f}M")
    print(FlopCountAnalysis(model, x))
    print(parameter_count_table(model, max_depth=2))

    test_submodules_loadable(model)
    time.sleep(5)

    print("Primus M")
    model = PrimusM(1, 2, (8, 8, 8), (96, 96, 96)).cuda()
    _ = model(x)
    print(f"Parameter count: {parameter_count(model)[''] / 1e6:.2f}M")
    print(FlopCountAnalysis(model, x))
    print(parameter_count_table(model, max_depth=2))

    test_submodules_loadable(model)
    time.sleep(5)

    print("Primus L")
    model = PrimusL(1, 2, (8, 8, 8), (96, 96, 96)).cuda()
    _ = model(x)
    print(f"Parameter count: {parameter_count(model)[''] / 1e6:.2f}M")
    print(FlopCountAnalysis(model, x))
    print(parameter_count_table(model, max_depth=2))

    test_submodules_loadable(model)
    time.sleep(5)
