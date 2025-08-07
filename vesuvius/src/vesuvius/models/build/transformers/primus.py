from typing import Tuple
import torch
from torch import nn

from timm.layers import RotaryEmbeddingCat
from .abstract_arch import (
    AbstractDynamicNetworkArchitectures,
    test_submodules_loadable,
)
from .eva import Eva
from .patch_encode_decode import LayerNormNd, PatchDecode, PatchEmbed
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
        assert len(input_shape) == 3, "Currently on ly 3d is supported"
        assert all([j % i == 0 for i, j in zip(patch_embed_size, input_shape)])

        super().__init__()
        self.key_to_encoder = "eva"
        self.key_to_stem = "down_projection"
        self.keys_to_in_proj = ("down_projection.proj",)
        self.key_to_lpe = "eva.pos_embed"

        self.down_projection = PatchEmbed(patch_embed_size, input_channels, embed_dim)
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
        # self.mask_token =
        self.mask_token: torch.Tensor
        self.register_buffer("mask_token", torch.zeros(1, 1, embed_dim))

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

        # Create mask tokens for missing patches
        num_masked = num_patches - num_kept
        mask_tokens = self.mask_token.repeat(B, num_masked, 1)

        # Prepare an empty tensor for the restored sequence
        restored = torch.zeros(B, num_patches, C, device=device)
        restored_mask = torch.zeros(B, num_patches, dtype=torch.bool, device=device)

        # Assign the kept patches and mask tokens in the correct positions
        for i in range(B):
            kept_pos = keep_indices[i]
            # masked_pos_prior = torch.tensor([j for j in range(num_patches) if j not in kept_pos], device=device)
            # replacement of list comprehension
            # kept_pos_tensor = torch.tensor(kept_pos, device=device)  # Ensure kept_pos is a tensor
            all_indices = torch.arange(num_patches, device=device)  # Create tensor of all indices
            mask = torch.ones(num_patches, device=device, dtype=torch.bool)  # Start with all True
            mask[kept_pos] = False  # Set kept positions to False
            masked_pos = all_indices[mask]  # Extract indices not in kept_pos

            restored[i, kept_pos] = x[i]
            restored[i, masked_pos] = mask_tokens[i, : len(masked_pos)]
            restored_mask[i, kept_pos] = True

        return (restored, restored_mask)

    def forward(self, x, ret_mask=False):
        FW, FH, FD = x.shape[2:]  # Full W , ...
        x = self.down_projection(x)
        # last output of the encoder is the input to EVA
        B, C, W, H, D = x.shape
        num_patches = W * H * D

        x = rearrange(x, "b c w h d -> b (w h d) c")
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
        x = rearrange(restored_x, "b (w h d) c -> b c w h d", h=H, w=W, d=D)
        if restoration_mask is not None:
            mask = rearrange(restoration_mask, "b (w h d) -> b w h d", h=H, w=W, d=D)
            full_mask = (
                mask.repeat_interleave(FW // W, dim=1)
                .repeat_interleave(FH // H, dim=2)
                .repeat_interleave(FD // D, dim=3)
            )
            full_mask = full_mask[:, None, ...]  # Add channel dimension  # [B, 1, W, H, D]
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
