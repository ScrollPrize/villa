from collections import OrderedDict
import math
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from timm.layers import trunc_normal_, use_fused_attn, DropPath, SwiGLU, GluMlp, Mlp
from torch import nn
from torch.nn import LayerNorm
from torch.utils.checkpoint import checkpoint
from einops import rearrange

from dinovol_2.model.patch_encode_decode import PatchEmbed, PatchEmbedDeeper
from dinovol_2.model.rope import RopeEmbedding, RopePositionEmbedding, apply_rotary_embedding


class InitWeights_He(object):
    def __init__(self, neg_slope: float = 1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)


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
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.norm = norm_layer(all_head_dim) if norm_layer is not None else nn.Identity()
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(
            self,
            x,
            rope: Optional[RopeEmbedding] = None,
            attn_mask: Optional[torch.Tensor] = None,
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
        
        if rope is not None:
            q = apply_rotary_embedding(q, rope, prefix_tokens=self.num_prefix_tokens).type_as(v)
            k = apply_rotary_embedding(k, rope, prefix_tokens=self.num_prefix_tokens).type_as(v)
        
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            raise RuntimeError("Fused attention should be used.")
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))
            
            if attn_mask is not None:
                attn_mask = attn_mask.to(torch.bool)
                attn = attn.masked_fill(~attn_mask[:, None, None, :], float("-inf"))
            attn = attn.softmax(dim=-1)
            
            attn = self.attn_drop(attn)
            x = attn @ v
        
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


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
            drop_path_scale: bool = True
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
        self.norm1 = norm_layer(dim)
        self.attn = EvaAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qkv_fused=qkv_fused,
            num_prefix_tokens=num_prefix_tokens,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            attn_head_dim=attn_head_dim,
            norm_layer=norm_layer if scale_attn_inner else None,
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
    
    def forward(self, x, rope: Optional[RopeEmbedding] = None, attn_mask: Optional[torch.Tensor] = None):
        if self.gamma_1 is None:
            x = x + self.drop_path1(self.attn(self.norm1(x), rope=rope, attn_mask=attn_mask))
            x = x + self.drop_path2(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), rope=rope, attn_mask=attn_mask))
            x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
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
    def _normalize_optional_chunk_shape(
            value: Optional[int | Tuple[int, ...]],
            *,
            ndim: int,
            name: str,
    ) -> Optional[Tuple[int, ...]]:
        if value is None:
            return None
        if isinstance(value, int):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
            return tuple([int(value)] * ndim)
        result = tuple(int(v) for v in value)
        if len(result) != ndim:
            raise ValueError(f"{name} must provide {ndim} values, got {result}")
        if any(v <= 0 for v in result):
            raise ValueError(f"{name} must be positive, got {result}")
        return result

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
            drop_path_scale: bool = True,
            rope_impl=RopePositionEmbedding,
            rope_kwargs=None,
            grad_checkpointing=False,
            deeper_embed_patch_chunk_size: Optional[int | Tuple[int, ...]] = None,
            deeper_embed_batch_chunk_size: Optional[int] = None,
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
        self.global_input_size = tuple(int(size) for size in self.global_crops_size)
        self.local_input_size = tuple(int(size) for size in self.local_crops_size)
        self.deeper_embed_patch_halo = tuple(0 for _ in range(self.ndim))
        self.deeper_embed_patch_chunk_size = self._normalize_optional_chunk_shape(
            deeper_embed_patch_chunk_size,
            ndim=self.ndim,
            name="deeper_embed_patch_chunk_size",
        )
        self.deeper_embed_batch_chunk_size = self._normalize_optional_positive_int(
            deeper_embed_batch_chunk_size,
            name="deeper_embed_batch_chunk_size",
        )

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
        
        if use_rot_pos_emb:
            if embed_dim % num_heads != 0:
                raise ValueError(
                    f"embed_dim must be divisible by num_heads, got embed_dim={embed_dim}, num_heads={num_heads}"
                )
            head_dim = embed_dim // num_heads
            if head_dim % (2 * self.ndim) != 0:
                raise ValueError(
                    f"RoPE requires head_dim divisible by 2 * ndim, got head_dim={head_dim}, ndim={self.ndim}"
                )
            self.rope_embed = rope_impl(
                head_dim,
                ndim=self.ndim,
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
                drop_path_scale=drop_path_scale
            )
            for i in range(depth)])
        
        self.norm = norm_layer(embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        self._init_weights()
    
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
            trunc_normal_(self.mask_token, std=.02)
        
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

    def _resolve_target_spatial_shape(self, spatial: tuple[int, ...], *, view_kind: str) -> tuple[int, ...]:
        spatial = tuple(int(dim) for dim in spatial)
        if view_kind == "global":
            target = tuple(int(dim) for dim in self.global_crops_size)
            input_size = tuple(int(dim) for dim in self.global_input_size)
        elif view_kind == "local":
            target = tuple(int(dim) for dim in self.local_crops_size)
            input_size = tuple(int(dim) for dim in self.local_input_size)
        else:
            raise ValueError(f"unknown view_kind={view_kind!r}")

        if spatial == input_size or spatial == target:
            return target
        raise ValueError(
            f"unexpected input shape {spatial} for embedding_type={self.embedding_type!r} and view_kind={view_kind!r}; "
            f"expected {input_size} or {target}"
        )

    def _crop_embedded_grid(self, x: torch.Tensor, target_spatial: tuple[int, ...]) -> torch.Tensor:
        target_patch_shape = tuple(int(size) // int(patch) for size, patch in zip(target_spatial, self.patch_size))
        current_patch_shape = tuple(int(dim) for dim in x.shape[2:])
        if current_patch_shape == target_patch_shape:
            return x

        starts = []
        for current, target in zip(current_patch_shape, target_patch_shape):
            delta = current - target
            if delta < 0 or delta % 2 != 0:
                raise ValueError(
                    f"cannot center-crop embedded grid from {current_patch_shape} to {target_patch_shape}"
                )
            starts.append(delta // 2)
        slices = tuple(slice(start, start + size) for start, size in zip(starts, target_patch_shape))
        return x[(slice(None), slice(None), *slices)]
    
    def prepare_tokens_with_masks(self, x, masks=None, *, view_kind: str = "global"):
        spatial = tuple(x.shape[2:])
        self._assert_patch_aligned(spatial, tuple(self.patch_size), context="input shape")
        if self.embedding_type == "deeper":
            target_spatial = self._resolve_target_spatial_shape(spatial, view_kind=view_kind)
            target_patch_shape = tuple(int(size) // int(patch) for size, patch in zip(target_spatial, self.patch_size))
            if (
                    self.deeper_embed_patch_chunk_size is not None
                    or self.deeper_embed_batch_chunk_size is not None
            ):
                x = self.down_projection.forward_tiled(
                    x,
                    target_patch_shape=target_patch_shape,
                    patch_chunk_size=self.deeper_embed_patch_chunk_size,
                    batch_chunk_size=self.deeper_embed_batch_chunk_size,
                )
            else:
                x = self.down_projection(x)
                x = self._crop_embedded_grid(x, target_spatial)
        else:
            x = self.down_projection(x)
            target_spatial = spatial
        if self.ndim == 2:
            x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        else:
            x = rearrange(x, 'b c d h w -> b (d h w) c').contiguous()
        
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype), x)
        
        x, rot_pos_embed = self._pos_embed(x, *target_spatial)
        
        return x, rot_pos_embed
    
    def forward_features_list(self, x_list, masks_list, *, view_kind: str = "global"):
        if not isinstance(x_list, list):
            return self.forward_features(x_list, masks_list, view_kind=view_kind)
        output = []
        for x, masks in zip(x_list, masks_list):
            x_out = self.forward_features(x, masks, view_kind=view_kind)
            output.append(x_out)
        return output
    
    def forward_features(self, x, masks=None, *, view_kind: str = "global"):
        x, rot_pos_embed = self.prepare_tokens_with_masks(x, masks, view_kind=view_kind)
        for blk in self.blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x, rope=rot_pos_embed)
            else:
                x = blk(x, rope=rot_pos_embed)
        x = self.norm(x)
        outputs = {
            "x_norm_clstoken": x[:, 0] if self.num_class_tokens > 0 else None,
            "x_norm_regtokens": x[:, self.num_class_tokens:self.num_prefix_tokens],
            "x_norm_patchtokens": x[:, self.num_prefix_tokens:],
            "x_prenorm": x,
            "masks": masks,
        }
        return outputs
    
    def forward(self, x, masks=None, is_training=True, *, view_kind: str = "global"):
        return self.forward_features_list(x, masks, view_kind=view_kind)
    
    def load_pretrained_weights(self, state_dict, backbone_only=False, unchunk=False):
        if isinstance(state_dict, str):
            state_dict = torch.load(state_dict, map_location="cpu", weights_only=False)['teacher']
            new_state_dict = {}
            for k, v in state_dict.items():
                if not k.startswith("backbone."):
                    continue
                new_key = k.replace("backbone.", "", 1)
                new_state_dict[new_key] = v
            state_dict = new_state_dict
        
        if unchunk:
            state_dict = self.unchunk_state_dict(state_dict)
        return self.load_state_dict(state_dict)
    
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
    def forward(self, x, rope=None, attn_mask=None):
        for blk in self:
            x = blk(x, rope=rope, attn_mask=attn_mask)
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
    
    def forward_features(self, x, masks=None, *, view_kind: str = "global"):
        x, rot_pos_embed = self.prepare_tokens_with_masks(x, masks, view_kind=view_kind)
        
        if self.chunked_blocks:
            for chunk in self.blocks:
                if self.grad_checkpointing and not torch.jit.is_scripting():
                    x = checkpoint(chunk, x, rope=rot_pos_embed)
                else:
                    x = chunk(x, rope=rot_pos_embed)
        else:
            for blk in self.blocks:
                if self.grad_checkpointing and not torch.jit.is_scripting():
                    x = checkpoint(blk, x, rope=rot_pos_embed)
                else:
                    x = blk(x, rope=rot_pos_embed)
        
        x = self.norm(x)
        outputs = {
            "x_norm_clstoken": x[:, 0] if self.num_class_tokens > 0 else None,
            "x_norm_regtokens": x[:, self.num_class_tokens:self.num_prefix_tokens],
            "x_norm_patchtokens": x[:, self.num_prefix_tokens:],
            "x_prenorm": x,
            "masks": masks,
        }
        return outputs
    
    def forward(self, x, masks=None, is_training=True, *, view_kind: str = "global"):
        return self.forward_features_list(x, masks, view_kind=view_kind)


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
