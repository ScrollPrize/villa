# Adapted from dynamic-network-architectures for Dinovol.
# See THIRD_PARTY_NOTICES.md and LICENSES/Apache-2.0.txt.

from collections import OrderedDict
from functools import partial
import math
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from timm.layers import trunc_normal_, use_fused_attn, DropPath, SwiGLU, GluMlp, Mlp
from torch import nn
from torch.nn import LayerNorm
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import CheckpointPolicy, checkpoint, create_selective_checkpoint_contexts
from einops import rearrange

from dinovol_2.model.patch_encode_decode import PatchEmbed, PatchEmbedDeeper
from dinovol_2.model.rope import (
    CompassRopePositionEmbedding,
    MixedRopePositionEmbedding,
    RopeCoords,
    RopeEmbedding,
    RopePositionEmbedding,
    apply_rotary_embedding,
)
from dinovol_2.ops.point_embeddings import sample_grid_points


def _save_convolutions_policy(ctx, op, *args, **kwargs):
    # Convolution outputs are kept; the norms and activations between them are cheap
    # pointwise work, recomputed in backward instead of saving each one's output as well.
    if op is torch.ops.aten.convolution.default:
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE


_save_convolutions_context = partial(create_selective_checkpoint_contexts, _save_convolutions_policy)


def _save_attention_policy(ctx, op, *args, **kwargs):
    # Keep fused attention (including its backward auxiliaries), but recompute
    # projections, normalization and the MLP. Avoid saving quadratic math-SDPA
    # intermediates when a fused backend is unavailable.
    if op in (
        torch.ops.aten._scaled_dot_product_cudnn_attention.default,
        torch.ops.aten._scaled_dot_product_efficient_attention.default,
    ):
        dropout_index = 5
    elif op in (
        torch.ops.aten._scaled_dot_product_flash_attention.default,
        torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default,
    ):
        dropout_index = 3
    else:
        return CheckpointPolicy.PREFER_RECOMPUTE
    dropout = kwargs.get("dropout_p", args[dropout_index] if len(args) > dropout_index else 0.)
    # Skipping a random SDPA call during eager replay would shift the RNG stream
    # for subsequent dropout/drop-path operations in the checkpointed block.
    return CheckpointPolicy.MUST_SAVE if dropout == 0. else CheckpointPolicy.PREFER_RECOMPUTE


_save_attention_context = partial(create_selective_checkpoint_contexts, _save_attention_policy)


class InitWeights_He(object):
    def __init__(self, neg_slope: float = 1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)


_ATTENTION_BACKENDS = {
    "cudnn": SDPBackend.CUDNN_ATTENTION,
    "flash": SDPBackend.FLASH_ATTENTION,
    "efficient": SDPBackend.EFFICIENT_ATTENTION,
    "math": SDPBackend.MATH,
}


def attention_priority(backend: Optional[str]) -> Optional[list]:
    """SDPA kernel priority with `backend` first and the others as fallbacks."""
    if backend is None:
        return None
    if backend not in _ATTENTION_BACKENDS:
        raise ValueError(f"attention_backend must be one of {sorted(_ATTENTION_BACKENDS)}, got {backend!r}")
    preferred = _ATTENTION_BACKENDS[backend]
    return [preferred] + [other for other in _ATTENTION_BACKENDS.values() if other != preferred]


class EvaAttention(nn.Module):
    fused_attn: torch.jit.Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = True,
            qkv_fused: bool = True,
            num_prefix_tokens: int = 1,
            qkv_bias_separate: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            attn_head_dim: Optional[int] = None,
            norm_layer: Optional[Callable] = None,
            attention_backend: Optional[str] = "flash",
            qk_norm: bool = False,
    ):
        """

        Args:
            dim:
            num_heads:
            qkv_bias:
            qkv_fused:
            attn_drop:
            proj_drop:
            attn_head_dim:
            norm_layer:
            attention_backend: preferred fused SDPA kernel (see attention_priority);
                None keeps PyTorch's default choice.
            qk_norm: apply RMSNorm over each query/key head before RoPE.
        """
        super().__init__()
        self.num_heads = num_heads
        if attn_head_dim is None and dim % num_heads != 0:
            raise ValueError(f"dim must be divisible by num_heads, got dim={dim}, num_heads={num_heads}")
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = head_dim ** -0.5
        self.num_prefix_tokens = num_prefix_tokens
        self.fused_attn = use_fused_attn()
        # Applied around each call, so compiled graphs record the choice when traced.
        self.attention_priority = attention_priority(attention_backend)
        self.qkv_bias_separate = qkv_bias_separate

        if qkv_fused:
            self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
            self.q_proj = self.k_proj = self.v_proj = None
            if qkv_bias:
                self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
                self.register_buffer('k_bias', torch.zeros(all_head_dim), persistent=False)
                self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
            else:
                self.q_bias = self.k_bias = self.v_bias = None
        else:
            self.q_proj = nn.Linear(dim, all_head_dim, bias=qkv_bias)
            self.k_proj = nn.Linear(dim, all_head_dim, bias=False)
            self.v_proj = nn.Linear(dim, all_head_dim, bias=qkv_bias)
            self.qkv = None
            self.q_bias = self.k_bias = self.v_bias = None

        self.q_norm = nn.RMSNorm(head_dim, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-6) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.norm = norm_layer(all_head_dim) if norm_layer is not None else nn.Identity()
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
            self,
            x,
            rope: Optional[RopeEmbedding] = None,
            attn_mask: Optional[torch.Tensor] = None,
            rope_shape: Optional[Tuple[int, ...]] = None,
    ):
        B, N, C = x.shape

        if self.qkv is not None:
            if self.q_bias is None:
                qkv = self.qkv(x)
            else:
                qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
                if self.qkv_bias_separate:
                    qkv = self.qkv(x)
                    qkv += qkv_bias
                else:
                    qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)
            qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
        else:
            q = self.q_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)  # B, num_heads, N, C
            k = self.k_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)
            v = self.v_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)

        # Normalize each head independently; keep the projection dtype under AMP.
        q = self.q_norm(q).type_as(v)
        k = self.k_norm(k).type_as(v)

        if rope is not None:
            q = apply_rotary_embedding(q, rope, prefix_tokens=self.num_prefix_tokens).type_as(v)
            k = apply_rotary_embedding(k, rope, prefix_tokens=self.num_prefix_tokens).type_as(v)

        x = self._attention(q, k, v, attn_mask=attn_mask, spatial_shape=rope_shape)
        x = x.transpose(1, 2).reshape(B, N, -1)
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def _attention(self, q, k, v, *, attn_mask=None, spatial_shape=None):
        if not self.fused_attn:
            raise RuntimeError("Fused attention should be used.")
        dropout_p = self.attn_drop.p if self.training else 0.
        if self.attention_priority is None:
            return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p)
        with sdpa_kernel(self.attention_priority, set_priority=True):
            return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p)


class EvaAxialAttention(EvaAttention):
    """One axis pass, with global prefix queries and prefix keys in each line.

    Project and rotate tokens once before grouping spatial lines. Prefix attention
    uses P x N interactions; patch attention uses N x (axis_length + P), never N x N.
    """

    def __init__(self, *args, axis: int, ndim: int, **kwargs):
        super().__init__(*args, **kwargs)
        if ndim not in (2, 3) or not 0 <= axis < ndim:
            raise ValueError("axial attention requires a valid axis in a 2D or 3D grid")
        self.axis = axis
        self.ndim = ndim

    def _attention(self, q, k, v, *, attn_mask=None, spatial_shape=None):
        if attn_mask is not None:
            raise ValueError("axial attention does not support an explicit attention mask; use voxel masking")
        if (spatial_shape is None or len(spatial_shape) != self.ndim
                or any(size <= 0 for size in spatial_shape)
                or math.prod(spatial_shape) != q.shape[-2] - self.num_prefix_tokens):
            raise ValueError("axial attention requires spatial_shape matching the patch token grid")
        B, H, _, C = q.shape
        P = self.num_prefix_tokens
        line_length = spatial_shape[self.axis]
        other_axes = [axis for axis in range(self.ndim) if axis != self.axis]
        other_shape = [spatial_shape[axis] for axis in other_axes]
        line_count = math.prod(other_shape)
        # [B, H, spatial..., C] -> [B, other spatial..., H, axis, C].
        order = [0, *[axis + 2 for axis in other_axes], 1, self.axis + 2, self.ndim + 2]
        inverse = [order.index(index) for index in range(len(order))]

        def lines(tensor):
            return tensor[:, :, P:].reshape(B, H, *spatial_shape, C).permute(order).reshape(
                B * line_count, H, line_length, C
            )

        def line_keys(tensor):
            patches = lines(tensor)
            if not P:
                return patches
            prefix = tensor[:, :, :P].unsqueeze(1).expand(B, line_count, H, P, C)
            return torch.cat((prefix.reshape(B * line_count, H, P, C), patches), dim=-2)

        patches = super()._attention(lines(q), line_keys(k), line_keys(v))
        patches = patches.reshape(B, *other_shape, H, line_length, C).permute(inverse)
        patches = patches.reshape(B, H, math.prod(spatial_shape), C)
        if not P:
            return patches
        prefix = super()._attention(q[:, :, :P], k, v)
        return torch.cat((prefix, patches), dim=-2)


class EvaBlock(nn.Module):

    def __init__(
            self,
            dim: int,
            num_heads: int,
            qkv_bias: bool = True,
            qkv_fused: bool = True,
            mlp_ratio: float = 4.,
            swiglu_mlp: bool = False,
            scale_mlp: bool = False,
            scale_attn_inner: bool = False,
            num_prefix_tokens: int = 1,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            init_values: Optional[float] = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            attn_head_dim: Optional[int] = None,
            drop_path_scale: bool = True,
            rope_impl=None,
            rope_kwargs=None,
            ndim: Optional[int] = None,
            attention_type: str = "full",
            attention_backend: Optional[str] = "flash",
            qk_norm: bool = False,
    ):
        """

        Args:
            dim:
            num_heads:
            qkv_bias:
            qkv_fused:
            mlp_ratio:
            swiglu_mlp:
            scale_mlp:
            scale_attn_inner:
            proj_drop:
            attn_drop:
            drop_path:
            init_values:
            act_layer:
            norm_layer:
            attn_head_dim:
        """
        super().__init__()
        if rope_kwargs is None:
            rope_kwargs = {}
        if attention_type not in {"full", "axial"}:
            raise ValueError(f"Unknown attention_type={attention_type!r}; expected 'full' or 'axial'")
        if attention_type == "axial" and ndim not in (2, 3):
            raise ValueError("axial attention requires ndim=2 or ndim=3")
        self.attention_type = attention_type
        self.norm1 = norm_layer(dim)
        attention_kwargs = dict(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qkv_fused=qkv_fused,
            num_prefix_tokens=num_prefix_tokens,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            attn_head_dim=attn_head_dim,
            norm_layer=norm_layer if scale_attn_inner else None,
            attention_backend=attention_backend,
            qk_norm=qk_norm,
        )
        self.attn = (EvaAxialAttention(**attention_kwargs, axis=0, ndim=ndim)
                     if attention_type == "axial" else EvaAttention(**attention_kwargs))
        self.axial_norms = nn.ModuleList()
        self.axial_attns = nn.ModuleList()
        self.axial_gammas = nn.ParameterList()
        if attention_type == "axial":
            for axis in range(1, ndim):
                self.axial_norms.append(norm_layer(dim))
                self.axial_attns.append(EvaAxialAttention(**attention_kwargs, axis=axis, ndim=ndim))
                if init_values is not None:
                    self.axial_gammas.append(nn.Parameter(init_values * torch.ones(dim)))
        self.rope_embed = None
        if rope_impl is not None:
            if attn_head_dim is None and dim % num_heads != 0:
                raise ValueError(f"dim must be divisible by num_heads, got dim={dim}, num_heads={num_heads}")
            head_dim = attn_head_dim if attn_head_dim is not None else dim // num_heads
            rope_kwargs_local = dict(rope_kwargs)
            self.rope_embed = rope_impl(
                head_dim,
                ndim=ndim,
                num_heads=num_heads,
                **rope_kwargs_local,
            )
        self.gamma_1 = nn.Parameter(init_values * torch.ones(dim)) if init_values is not None else None
        self.drop_path1 = DropPath(drop_path, drop_path_scale) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        hidden_features = int(dim * mlp_ratio)
        if swiglu_mlp:
            if scale_mlp:
                # when norm in SwiGLU used, an impl with separate fc for gate & x is used
                self.mlp = SwiGLU(
                    in_features=dim,
                    hidden_features=hidden_features,
                    norm_layer=norm_layer if scale_mlp else None,
                    drop=proj_drop,
                )
            else:
                # w/o any extra norm, an impl with packed weights is used, matches existing GluMLP
                self.mlp = GluMlp(
                    in_features=dim,
                    hidden_features=hidden_features * 2,
                    norm_layer=norm_layer if scale_mlp else None,
                    act_layer=nn.SiLU,
                    gate_last=False,
                    drop=proj_drop,
                )
        else:
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=hidden_features,
                act_layer=act_layer,
                norm_layer=norm_layer if scale_mlp else None,
                drop=proj_drop,
            )
        self.gamma_2 = nn.Parameter(init_values * torch.ones(dim)) if init_values is not None else None
        self.drop_path2 = DropPath(drop_path, drop_path_scale) if drop_path > 0. else nn.Identity()

    def forward(
            self,
            x,
            rope: Optional[RopeEmbedding] = None,
            attn_mask: Optional[torch.Tensor] = None,
            rope_shape: Optional[Tuple[int, ...]] = None,
            rope_coords: Optional[RopeCoords] = None,
    ):
        if rope is None and self.rope_embed is not None:
            if rope_coords is not None:
                rope = self.rope_embed.get_embed_from_coords(rope_coords)
            else:
                if rope_shape is None:
                    raise ValueError("rope_shape must be provided when using per-block RoPE")
                rope = self.rope_embed.get_embed(rope_shape)
        update = self.attn(self.norm1(x), rope=rope, attn_mask=attn_mask, rope_shape=rope_shape)
        x = x + self.drop_path1(update if self.gamma_1 is None else self.gamma_1 * update)
        for index, (norm, attention) in enumerate(zip(self.axial_norms, self.axial_attns)):
            gamma = self.axial_gammas[index] if self.gamma_1 is not None else None
            update = attention(norm(x), rope=rope, attn_mask=attn_mask, rope_shape=rope_shape)
            x = x + self.drop_path1(update if gamma is None else gamma * update)
        update = self.mlp(self.norm2(x))
        x = x + self.drop_path2(update if self.gamma_2 is None else self.gamma_2 * update)
        return x


class Eva(nn.Module):
    """ Eva Vision Transformer w/ Abs & Rotary Pos Embed

    This class implements the EVA and EVA02 models that were based on the BEiT ViT variant
      * EVA - abs pos embed, global avg pool
      * EVA02 - abs + rope pos embed, global avg pool, SwiGLU, scale Norm in MLP (ala normformer)


    """

    @staticmethod
    def _assert_patch_aligned(
            spatial_shape: Tuple[int, ...],
            patch_size: Tuple[int, ...],
            *,
            context: str,
    ) -> None:
        remainders = [int(size) % int(patch) for size, patch in zip(spatial_shape, patch_size)]
        if any(remainders):
            raise AssertionError(
                f"{context} must be divisible by patch_size for PatchEmbedDeeper, "
                f"got spatial_shape={tuple(spatial_shape)} and patch_size={tuple(patch_size)}"
            )

    @staticmethod
    def _normalize_optional_positive_int(
            value: Optional[int],
            *,
            name: str,
    ) -> Optional[int]:
        if value is None:
            return None
        value = int(value)
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
        return value

    def __init__(
            self,
            input_channels: int = 1,
            global_crops_size: Tuple[int, ...] = None,
            local_crops_size: Tuple[int, ...] = None,
            embed_dim: int = 864,
            patch_size: Tuple[int, ...] = (8, 8, 8),
            embedding_type: str = "default",
            depth: int = 24,
            num_heads: int = 12,
            qkv_bias: bool = True,
            qkv_fused: bool = False,
            mlp_ratio: float = 4 * 2 / 3,
            swiglu_mlp: bool = True,
            scale_mlp: bool = True,
            scale_attn_inner: bool = False,
            pos_drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            # drops out things related to the projection. That is in the MLP and at the end of EVA attention
            attn_drop_rate: float = 0.,
            # drops attention, meaning connections between patches may bebroken up at random
            drop_path_rate: float = 0.,
            # drops computations (multihead attention, mlp), Implementation of scaling might be useless here because this is not batch normed
            drop_path_uniform: bool = False,
            norm_layer: Callable = LayerNorm,
            init_values: Optional[float] = None,
            class_token: bool = True,
            use_abs_pos_emb: bool = False,
            use_rot_pos_emb: bool = True,
            dynamic_img_size: bool = False,
            num_reg_tokens: int = 0,
            untie_cls_and_patch_norms: bool = False,
            untie_global_and_local_cls_norm: bool = False,
            drop_path_scale: bool = True,
            rope_impl=RopePositionEmbedding,
            rope_kwargs=None,
            grad_checkpointing=False,
            checkpoint_save_attention: bool = True,
            deeper_embed_batch_chunk_size: Optional[int] = 1,
            deeper_embed_stage_channels: Optional[Tuple[int, ...]] = None,
            deeper_embed_depth_per_level: Optional[Tuple[int, ...]] = None,
            deeper_embed_use_stem: bool = True,
            deeper_embed_final_projection: bool = True,
            deeper_embed_channels_last: bool = False,
            deeper_embed_norm_backend: str = "auto",
            deeper_embed_stage_strides=None,
            deeper_embed_stem_channels: Optional[int] = None,
            deeper_embed_stem_type: Optional[str] = None,
            attention_type: str = "full",
            attention_backend: Optional[str] = "flash",
            qk_norm: bool = False,
    ):
        """
        Diff to timm implementation

        - removed patch embedding, we expect embeded patches
        - removed classification token, we use features at the end
        - removed head
        - dynamic image size is not supported, but left in for future stuff
        - self.cls_token removed
        - removed postnorm block support
        """
        super().__init__()

        self.input_channels = input_channels
        self.patch_size = [patch_size] * 3 if isinstance(patch_size, int) else patch_size
        self.ndim = len(self.patch_size)
        self.embedding_type = str(embedding_type).lower()
        self.global_crops_size = [global_crops_size] * 3 if isinstance(global_crops_size, int) else global_crops_size
        self.local_crops_size = [local_crops_size] * 3 if isinstance(local_crops_size, int) else local_crops_size
        self.deeper_embed_batch_chunk_size = self._normalize_optional_positive_int(
            deeper_embed_batch_chunk_size,
            name="deeper_embed_batch_chunk_size",
        )
        # Return the CNN embedding's intermediate stage outputs of global views as
        # features["embed_skips"] (used by the dense point head).
        self.return_embed_skips = False
        # When point coordinates are given, embed_skips keeps full grids only from this
        # level on (None below it); every level is returned sampled at the points.
        self.embed_skip_grid_start = 0

        if self.embedding_type == "deeper":
            self._assert_patch_aligned(
                tuple(self.global_crops_size),
                tuple(self.patch_size),
                context="global_crops_size",
            )
            self._assert_patch_aligned(
                tuple(self.local_crops_size),
                tuple(self.patch_size),
                context="local_crops_size",
            )

        self.global_ref_feat_shape = tuple([i // ds for i, ds in zip(self.global_crops_size, self.patch_size)])
        self.local_ref_feat_shape = tuple([i // ds for i, ds in zip(self.local_crops_size, self.patch_size)])

        if self.embedding_type == "default":
            self.down_projection = PatchEmbed(
                patch_size=tuple(self.patch_size),
                input_channels=input_channels,
                embed_dim=embed_dim,
            )
        elif self.embedding_type == "deeper":
            self.down_projection = PatchEmbedDeeper(
                patch_size=tuple(self.patch_size),
                input_channels=input_channels,
                embed_dim=embed_dim,
                stage_channels=deeper_embed_stage_channels,
                depth_per_level=deeper_embed_depth_per_level,
                use_stem=deeper_embed_use_stem,
                final_projection=deeper_embed_final_projection,
                stage_strides=deeper_embed_stage_strides,
                stem_channels=deeper_embed_stem_channels,
                stem_type=deeper_embed_stem_type,
                channels_last=deeper_embed_channels_last,
                norm_backend=deeper_embed_norm_backend,
            )
        else:
            raise ValueError(
                f"unsupported embedding_type={embedding_type!r}; expected 'default' or 'deeper'"
            )

        if rope_kwargs is None:
            rope_kwargs = {}

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.dynamic_img_size = dynamic_img_size
        self.grad_checkpointing = grad_checkpointing
        self.checkpoint_save_attention = checkpoint_save_attention

        self.num_reg_tokens = num_reg_tokens
        self.num_class_tokens = (1 if class_token else 0)
        self.num_prefix_tokens = self.num_class_tokens + self.num_reg_tokens

        num_patches = np.prod(self.global_ref_feat_shape)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        self.reg_token = nn.Parameter(torch.zeros(1, num_reg_tokens, embed_dim)) if num_reg_tokens else None
        self.cls_embed = class_token and self.reg_token is None

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + self.num_class_tokens, embed_dim)) if use_abs_pos_emb else None
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        self.use_per_block_rope = bool(
            use_rot_pos_emb and rope_impl in (MixedRopePositionEmbedding, CompassRopePositionEmbedding)
        )
        if use_rot_pos_emb and embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim must be divisible by num_heads, got embed_dim={embed_dim}, num_heads={num_heads}"
            )
        head_dim = embed_dim // num_heads
        rope_dim_multiple = 2 if rope_impl is CompassRopePositionEmbedding else 2 * self.ndim
        if use_rot_pos_emb and head_dim % rope_dim_multiple != 0:
            raise ValueError(
                f"RoPE requires head_dim divisible by {rope_dim_multiple}, got head_dim={head_dim}, ndim={self.ndim}"
            )
        if use_rot_pos_emb and not self.use_per_block_rope:
            self.rope_embed = rope_impl(
                head_dim,
                ndim=self.ndim,
                num_heads=num_heads,
                **rope_kwargs,
            )
        else:
            self.rope_embed = None

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        block_fn = EvaBlock
        self.blocks = nn.ModuleList([
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qkv_fused=qkv_fused,
                qk_norm=qk_norm,
                mlp_ratio=mlp_ratio,
                swiglu_mlp=swiglu_mlp,
                scale_mlp=scale_mlp,
                scale_attn_inner=scale_attn_inner,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                init_values=init_values,
                num_prefix_tokens=self.num_prefix_tokens,
                drop_path_scale=drop_path_scale,
                rope_impl=rope_impl if self.use_per_block_rope else None,
                rope_kwargs=rope_kwargs if self.use_per_block_rope else None,
                ndim=self.ndim,
                attention_type=attention_type,
                attention_backend=attention_backend,
            )
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)
        self.cls_norm = norm_layer(embed_dim) if untie_cls_and_patch_norms else None
        self.local_cls_norm = norm_layer(embed_dim) if untie_global_and_local_cls_norm else None
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self._init_weights()
        if self.embedding_type == "deeper" and deeper_embed_channels_last:
            # Convert after initialization so initial values do not depend on the layout.
            self.down_projection.to(memory_format=self.down_projection.memory_format)

    def _init_weights(self):
        def init_fn(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(init_fn)
        self.down_projection.apply(InitWeights_He(1e-2))

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            trunc_normal_(self.cls_token, std=.02)
        if self.reg_token is not None:
            trunc_normal_(self.reg_token, std=.02)
        if self.mask_token is not None:
            nn.init.zeros_(self.mask_token)

        # Inline fix_init_weight
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            if hasattr(layer.attn.proj, 'weight'):
                rescale(layer.attn.proj.weight.data, layer_id + 1)
            if hasattr(layer.mlp.fc2, 'weight'):
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    @torch.jit.ignore
    def no_weight_decay(self):
        nwd = {'pos_embed', 'cls_token'}
        return nwd

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(
            stem=r'^cls_token|pos_embed|patch_embed',  # stem and embed
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))],
        )
        return matcher

    def _pos_embed(self, x, *spatial) -> Tuple[torch.Tensor, Optional[RopeEmbedding]]:
        """
        Computes positional embeddings with interpolation if needed.

        Args:
            x (torch.Tensor): Input tensor after patch embedding, shape (B, N, C).
            spatial: Spatial dimensions of the original image before patch embedding.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Positionally encoded input.
        """
        pos_embed = self.pos_embed
        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        source_size = tuple(self.global_ref_feat_shape)
        target_size = tuple(dim // patch for dim, patch in zip(spatial, self.patch_size))

        # If needed, interpolate only patch embeddings
        if source_size != target_size:
            if pos_embed is not None:
                pos_embed = self.interpolate_pos_encoding_nd(
                    pos_embed,
                    source_size=source_size,
                    target_size=target_size,
                    num_prefix_tokens=self.num_class_tokens
                )
        rot_pos_embed = self._get_rot_pos_embed(target_size)

        # Add interpolated positional embeddings
        if pos_embed is not None:
            x = x + pos_embed

        if self.reg_token is not None:
            reg_tokens = self.reg_token.expand(x.shape[0], -1, -1)
            if self.cls_token is not None:
                x = torch.cat((x[:, :1], reg_tokens, x[:, 1:]), dim=1)
            else:
                x = torch.cat((reg_tokens, x), dim=1)

        x = self.pos_drop(x)

        return x, rot_pos_embed

    def _get_rot_pos_embed(self, target_size: Tuple[int, ...]) -> Optional[RopeEmbedding]:
        if self.rope_embed is None:
            return None
        return self.rope_embed.get_embed(target_size)

    def _get_shared_per_block_rope_coords(self, target_size: Tuple[int, ...]) -> Optional[RopeCoords]:
        if not self.use_per_block_rope:
            return None
        for block in self.blocks:
            if hasattr(block, "rope_embed") and block.rope_embed is not None:
                return block.rope_embed.get_coords(target_size)
            if isinstance(block, nn.ModuleList):
                for inner_block in block:
                    if hasattr(inner_block, "rope_embed") and inner_block.rope_embed is not None:
                        return inner_block.rope_embed.get_coords(target_size)
        return None

    def interpolate_pos_encoding_nd(
            self,
            pos_embed: torch.Tensor,
            source_size: tuple,
            target_size: tuple,
            num_prefix_tokens: int = 1,
    ) -> torch.Tensor:
        """
        Interpolates positional embeddings to match a new spatial size.

        Args:
            pos_embed (torch.Tensor): Positional embeddings (1, N, D).
            source_size: Original source grid size.
            target_size: New target grid size.
            num_prefix_tokens (int): Number of special tokens (e.g., CLS, registers).

        Returns:
            torch.Tensor: Rescaled positional embeddings (1, N_new, D).
        """
        _, N, C = pos_embed.shape
        N = N - num_prefix_tokens  # Remove prefix tokens

        previous_dtype = pos_embed.dtype
        pos_embed = pos_embed.float()

        if num_prefix_tokens > 0:
            pos_prefix, pos_embed = pos_embed[:, :num_prefix_tokens], pos_embed[:, num_prefix_tokens:]
        else:
            pos_prefix = None

        ndim = len(source_size)
        if ndim not in (2, 3):
            raise ValueError(f"Only 2D and 3D positional interpolation are supported, got ndim={ndim}")

        interpolation_mode = "bilinear" if ndim == 2 else "trilinear"

        # Reshape from (1, N, C) -> (1, C, *source_size)
        pos_embed = pos_embed.reshape(1, *source_size, C)
        permute_order = (0, ndim + 1, *range(1, ndim + 1))
        pos_embed = pos_embed.permute(*permute_order)

        # Interpolate to the new spatial grid.
        pos_embed = F.interpolate(pos_embed, size=target_size, mode=interpolation_mode, align_corners=False)

        # Reshape back to (1, N, C)
        inverse_permute = (0, *range(2, ndim + 2), 1)
        pos_embed = pos_embed.permute(*inverse_permute).reshape(1, -1, C)

        # Reattach prefix tokens
        if pos_prefix is not None:
            pos_embed = torch.cat([pos_prefix, pos_embed], dim=1)

        pos_embed = pos_embed.to(previous_dtype)

        return pos_embed

    def _mask_input_voxels(self, x, masks):
        """Hide masked voxel regions before any overlapping convolution."""
        patch_shape = tuple(size // patch for size, patch in zip(x.shape[2:], self.patch_size))
        expected_shape = (x.shape[0], math.prod(patch_shape))
        if masks.dtype != torch.bool or tuple(masks.shape) != expected_shape:
            raise ValueError(f"masks must be boolean with shape {expected_shape}, got {masks.dtype} {tuple(masks.shape)}")
        voxel_mask = masks.reshape(x.shape[0], 1, *patch_shape)
        for axis, patch in enumerate(self.patch_size, start=2):
            voxel_mask = voxel_mask.repeat_interleave(patch, dim=axis)

        # Never modify the shared crop in place: the teacher needs its original values.
        return x.masked_fill(voxel_mask, 0)

    def _run_down_projection(self, x, **kwargs):
        """With grad checkpointing, the CNN embedding is selectively checkpointed: backward
        keeps the convolution outputs and recomputes the norms and activations between them."""
        if (self.embedding_type == "deeper" and self.grad_checkpointing and torch.is_grad_enabled()
                and not torch.jit.is_scripting()):
            return checkpoint(self.down_projection, x, use_reentrant=False, context_fn=_save_convolutions_context,
                              **kwargs)
        return self.down_projection(x, **kwargs)

    def _down_project_with_skips(self, x, skips):
        batch_chunk_size = self.deeper_embed_batch_chunk_size
        if batch_chunk_size is None or x.shape[0] <= batch_chunk_size:
            tokens, levels = self._run_down_projection(x, return_skips=True)
            skips.extend(levels)
            return tokens
        chunks = x.split(batch_chunk_size)
        outputs = [self._run_down_projection(chunk, return_skips=True) for chunk in chunks]
        skips.extend(torch.cat(level, dim=0) for level in zip(*(levels for _, levels in outputs)))
        return torch.cat([tokens for tokens, _ in outputs], dim=0)

    def prepare_tokens_with_masks(self, x, masks=None, *, view_kind: str = "global", skips=None):
        """skips: optional list that receives the CNN embedding's intermediate stage outputs."""
        spatial = tuple(x.shape[2:])
        self._assert_patch_aligned(spatial, tuple(self.patch_size), context="input shape")
        if masks is not None:
            x = self._mask_input_voxels(x, masks)
        batch_chunk_size = self.deeper_embed_batch_chunk_size
        if skips is not None:
            if self.embedding_type != "deeper":
                raise ValueError("embedding skips require the deeper CNN embedding")
            x = self._down_project_with_skips(x, skips)
        elif self.embedding_type == "deeper" and batch_chunk_size is not None and x.shape[0] > batch_chunk_size:
            x = torch.cat([self._run_down_projection(chunk) for chunk in x.split(batch_chunk_size)], dim=0)
        else:
            x = self._run_down_projection(x)
        if self.ndim == 2:
            x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        else:
            x = rearrange(x, 'b c d h w -> b (d h w) c').contiguous()

        if masks is not None:
            # Keep an explicit learned marker after hiding the underlying voxels.
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype), x)

        target_patch_shape = tuple(dim // patch for dim, patch in zip(spatial, self.patch_size))
        x, rot_pos_embed = self._pos_embed(x, *spatial)

        return x, rot_pos_embed, target_patch_shape

    def forward_features_list(self, x_list, masks_list, *, view_kind: str = "global", point_coordinates=None):
        if not isinstance(x_list, list):
            return self.forward_features(x_list, masks_list, view_kind=view_kind, point_coordinates=point_coordinates)
        if point_coordinates is not None:
            raise ValueError("point_coordinates require a single input tensor, not a list")
        output = []
        for x, masks in zip(x_list, masks_list):
            x_out = self.forward_features(x, masks, view_kind=view_kind)
            output.append(x_out)
        return output

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Warm-start newly separated norms from the corresponding trained norm.
        # With both options disabled, checkpoint keys are unchanged.
        for name in ("cls_norm", "local_cls_norm"):
            norm = getattr(self, name)
            if norm is None:
                continue
            source = "cls_norm" if name == "local_cls_norm" and self.cls_norm is not None else "norm"
            for key in norm.state_dict():
                target_key = f"{prefix}{name}.{key}"
                source_key = f"{prefix}{source}.{key}"
                if target_key not in state_dict and source_key in state_dict:
                    state_dict[target_key] = state_dict[source_key].clone()
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def _format_features(self, x, masks, *, view_kind: str):
        if view_kind not in {"global", "local"}:
            raise ValueError(f"Unknown view_kind={view_kind!r}")
        if self.cls_norm is None and self.local_cls_norm is None:
            normalized = self.norm(x)
            prefix = normalized[:, :self.num_prefix_tokens]
            patches = normalized[:, self.num_prefix_tokens:]
        else:
            prefix_norm = self.cls_norm if self.cls_norm is not None else self.norm
            if self.training and view_kind == "local" and self.local_cls_norm is not None:
                prefix_norm = self.local_cls_norm
            prefix = prefix_norm(x[:, :self.num_prefix_tokens])
            patches = self.norm(x[:, self.num_prefix_tokens:])
        return {
            "x_norm_clstoken": prefix[:, 0] if self.num_class_tokens > 0 else None,
            "x_norm_regtokens": prefix[:, self.num_class_tokens:],
            "x_norm_patchtokens": patches,
            "x_prenorm": x,
            "masks": masks,
        }

    def forward_features(self, x, masks=None, *, view_kind: str = "global", point_coordinates=None):
        skips = [] if self.return_embed_skips and view_kind == "global" else None
        input_shape = tuple(x.shape[2:])
        x, rot_pos_embed, rope_shape = self.prepare_tokens_with_masks(x, masks, view_kind=view_kind, skips=skips)
        skips, skip_points = self._sample_embed_skips(skips, point_coordinates, input_shape)
        rope_coords = self._get_shared_per_block_rope_coords(rope_shape)
        for blk in self.blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x, rope=rot_pos_embed, rope_shape=rope_shape, rope_coords=rope_coords,
                               use_reentrant=False, **self._attention_checkpoint_kwargs())
            else:
                x = blk(x, rope=rot_pos_embed, rope_shape=rope_shape, rope_coords=rope_coords)
        return self._with_skips(self._format_features(x, masks, view_kind=view_kind), skips, skip_points)

    def _attention_checkpoint_kwargs(self):
        return {"context_fn": _save_attention_context} if self.checkpoint_save_attention else {}

    def _sample_embed_skips(self, skips, point_coordinates, input_shape):
        """Sample the CNN skips at [B, P, 3] input-voxel point coordinates before the blocks run.

        The sampling backward scatters into skip-sized gradient grids. Sampled here, autograd
        runs it after the block backward, just before the CNN backward that consumes those
        grids, instead of holding them through the block backward. Grids below
        embed_skip_grid_start are dropped (None) since only their samples are used.
        """
        if skips is None or point_coordinates is None:
            return skips, None
        points = [sample_grid_points(level.permute(0, 2, 3, 4, 1), point_coordinates, input_shape) for level in skips]
        grids = [level if index >= self.embed_skip_grid_start else None for index, level in enumerate(skips)]
        return grids, points

    @staticmethod
    def _with_skips(features, skips, skip_points=None):
        if skips is not None:
            features["embed_skips"] = skips
        if skip_points is not None:
            features["embed_skip_points"] = skip_points
        return features

    def forward(self, x, masks=None, is_training=True, *, view_kind: str = "global", point_coordinates=None):
        return self.forward_features_list(x, masks, view_kind=view_kind, point_coordinates=point_coordinates)

    def load_pretrained_weights(self, state_dict, backbone_only=False, unchunk=False):
        if isinstance(state_dict, str):
            from dinovol_2.training.checkpoint import backbone_state
            saved = torch.load(state_dict, map_location="cpu", weights_only=False)
            state_dict, _ = backbone_state(saved)

        if unchunk:
            state_dict = self.unchunk_state_dict(state_dict)
        state_dict = dict(state_dict)
        if self.rope_embed is None:
            for key in [key for key in state_dict if key.startswith("rope_embed.")]:
                state_dict.pop(key)
        load_result = self.load_state_dict(state_dict, strict=False)

        missing_keys = list(load_result.missing_keys)
        unexpected_keys = list(load_result.unexpected_keys)
        allowed_missing_keys = self.warm_start_missing_rope(missing_keys)

        disallowed_missing = [key for key in missing_keys if key not in allowed_missing_keys]
        if disallowed_missing or unexpected_keys:
            details = []
            if disallowed_missing:
                details.append(f"missing keys: {disallowed_missing}")
            if unexpected_keys:
                details.append(f"unexpected keys: {unexpected_keys}")
            raise RuntimeError("failed to load pretrained backbone weights cleanly: " + "; ".join(details))

        return load_result

    def warm_start_missing_rope(self, missing_keys) -> set[str]:
        """RoPE keys a checkpoint may lack (e.g. one written with axial RoPE).

        Missing mixed-RoPE frequencies are reset to the axial layout, which reproduces
        axial RoPE exactly; `periods` buffers are rebuilt deterministically at init.
        Returns the keys (relative to this backbone) that may therefore be missing.
        """
        missing_keys = set(missing_keys)
        allowed_missing_keys = set()
        if self.rope_embed is not None and hasattr(self.rope_embed, "reset_mixed_frequencies_to_axial"):
            allowed_missing_keys.add("rope_embed.mix_frequencies")
            if "rope_embed.mix_frequencies" in missing_keys:
                self.rope_embed.reset_mixed_frequencies_to_axial()

        for module_name, module in self.named_modules():
            if not module_name or not hasattr(module, "rope_embed") or module.rope_embed is None:
                continue
            prefix = f"{module_name}.rope_embed"
            if hasattr(module.rope_embed, "reset_mixed_frequencies_to_axial"):
                allowed_missing_keys.add(f"{prefix}.mix_frequencies")
                if f"{prefix}.mix_frequencies" in missing_keys:
                    module.rope_embed.reset_mixed_frequencies_to_axial()
            allowed_missing_keys.add(f"{prefix}.periods")
        return allowed_missing_keys

    def unchunk_state_dict(self, state_dict):
        """
        Convert a state_dict from EvaWithChunking (nested blocks)
        to Eva (flat blocks).
        """
        if not any([key.startswith("blocks.0.0") for key in state_dict.keys()]):
            return state_dict

        new_state_dict = OrderedDict()
        for key, val in state_dict.items():
            if key.startswith("blocks."):
                parts = key.split(".")
                # e.g. "blocks.0.1.attn.qkv.weight"
                # parts[1] = chunk idx, parts[2] = inner idx
                if parts[2].isdigit():
                    chunk_idx = int(parts[1])
                    inner_idx = int(parts[2])
                    # compute new flat index
                    flat_idx = chunk_idx * 9999 + inner_idx  # temporary large stride
                    # rewrite key
                    new_key = ".".join(["blocks", str(flat_idx)] + parts[3:])
                    new_state_dict[new_key] = val
                else:
                    # already a normal block key (no extra index)
                    new_state_dict[key] = val
            else:
                new_state_dict[key] = val

        # Fix flat indices back to consecutive 0..N
        # because above we used a stride
        mapping = {old: new for new, old in enumerate(sorted(set(
            int(k.split(".")[1]) for k in new_state_dict if k.startswith("blocks.")
        )))}
        final_state_dict = OrderedDict()
        for key, val in new_state_dict.items():
            if key.startswith("blocks."):
                parts = key.split(".")
                parts[1] = str(mapping[int(parts[1])])
                final_state_dict[".".join(parts)] = val
            else:
                final_state_dict[key] = val

        return final_state_dict


class BlockChunk(nn.ModuleList):
    def forward(self, x, rope=None, attn_mask=None, rope_shape=None, rope_coords=None):
        for blk in self:
            x = blk(x, rope=rope, attn_mask=attn_mask, rope_shape=rope_shape, rope_coords=rope_coords)
        return x


class EvaWithChunking(Eva):
    def __init__(self, *args, block_chunks: int = 1, **kwargs):
        super().__init__(*args, **kwargs)

        self.block_chunks = block_chunks
        self.chunked_blocks = block_chunks > 0 and block_chunks < len(self.blocks)

        if self.chunked_blocks:
            self._apply_block_chunking()

    def _apply_block_chunking(self):
        depth = len(self.blocks)
        chunksize = depth // self.block_chunks
        chunks = []
        for i in range(0, depth, chunksize):
            block_chunk = BlockChunk(self.blocks[i: i + chunksize])
            chunks.append(block_chunk)
        self.blocks = nn.ModuleList(chunks)

    def forward_features(self, x, masks=None, *, view_kind: str = "global", point_coordinates=None):
        skips = [] if self.return_embed_skips and view_kind == "global" else None
        input_shape = tuple(x.shape[2:])
        x, rot_pos_embed, rope_shape = self.prepare_tokens_with_masks(x, masks, view_kind=view_kind, skips=skips)
        skips, skip_points = self._sample_embed_skips(skips, point_coordinates, input_shape)
        rope_coords = self._get_shared_per_block_rope_coords(rope_shape)

        if self.chunked_blocks:
            for chunk in self.blocks:
                if self.grad_checkpointing and not torch.jit.is_scripting():
                    x = checkpoint(chunk, x, rope=rot_pos_embed, rope_shape=rope_shape, rope_coords=rope_coords,
                                   use_reentrant=False, **self._attention_checkpoint_kwargs())
                else:
                    x = chunk(x, rope=rot_pos_embed, rope_shape=rope_shape, rope_coords=rope_coords)
        else:
            for blk in self.blocks:
                if self.grad_checkpointing and not torch.jit.is_scripting():
                    x = checkpoint(blk, x, rope=rot_pos_embed, rope_shape=rope_shape, rope_coords=rope_coords,
                                   use_reentrant=False, **self._attention_checkpoint_kwargs())
                else:
                    x = blk(x, rope=rot_pos_embed, rope_shape=rope_shape, rope_coords=rope_coords)

        return self._with_skips(self._format_features(x, masks, view_kind=view_kind), skips, skip_points)

    def forward(self, x, masks=None, is_training=True, *, view_kind: str = "global", point_coordinates=None):
        return self.forward_features_list(x, masks, view_kind=view_kind, point_coordinates=point_coordinates)


class Dinov2PrimusEncL(Eva):
    def __init__(self,
                 input_channels,
                 input_shape):
        super().__init__(
            input_channels=input_channels,
            global_crops_size=96,
            local_crops_size=input_shape,
            embed_dim=864,
            patch_size=(8, 8, 8),
            depth=24,
            num_heads=16,
            mlp_ratio=2.66666666,
            attn_drop_rate=0.2,
            drop_path_rate=0.2
        )
