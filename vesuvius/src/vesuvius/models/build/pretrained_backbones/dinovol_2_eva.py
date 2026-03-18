from collections import OrderedDict
import math
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from timm.layers import DropPath, GluMlp, Mlp, SwiGLU, trunc_normal_, use_fused_attn
from torch import nn
from torch.nn import LayerNorm
from torch.utils.checkpoint import checkpoint

from vesuvius.models.build.transformers.patch_encode_decode import PatchEmbed

from .rope import RopeEmbedding, RopePositionEmbedding, apply_rotary_embedding


class InitWeights_He:
    def __init__(self, neg_slope: float = 1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
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
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        attn_head_dim: Optional[int] = None,
        norm_layer: Optional[Callable] = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        if attn_head_dim is None and dim % num_heads != 0:
            raise ValueError(f"dim must be divisible by num_heads, got dim={dim}, num_heads={num_heads}")
        head_dim = dim // num_heads if attn_head_dim is None else attn_head_dim
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
                self.register_buffer("k_bias", torch.zeros(all_head_dim), persistent=False)
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

    def forward(self, x, rope: Optional[RopeEmbedding] = None, attn_mask: Optional[torch.Tensor] = None):
        bsz, n_tokens, channels = x.shape

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
            qkv = qkv.reshape(bsz, n_tokens, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
        else:
            q = self.q_proj(x).reshape(bsz, n_tokens, self.num_heads, -1).transpose(1, 2)
            k = self.k_proj(x).reshape(bsz, n_tokens, self.num_heads, -1).transpose(1, 2)
            v = self.v_proj(x).reshape(bsz, n_tokens, self.num_heads, -1).transpose(1, 2)

        if rope is not None:
            q = apply_rotary_embedding(q, rope, prefix_tokens=self.num_prefix_tokens).type_as(v)
            k = apply_rotary_embedding(k, rope, prefix_tokens=self.num_prefix_tokens).type_as(v)

        if not self.fused_attn:
            raise RuntimeError("Fused attention should be used.")

        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        x = x.transpose(1, 2).reshape(bsz, n_tokens, channels)
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
        mlp_ratio: float = 4.0,
        swiglu_mlp: bool = False,
        scale_mlp: bool = False,
        scale_attn_inner: bool = False,
        num_prefix_tokens: int = 1,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        init_values: Optional[float] = None,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = LayerNorm,
        attn_head_dim: Optional[int] = None,
        drop_path_scale: bool = True,
    ):
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
        self.drop_path1 = DropPath(drop_path, drop_path_scale) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        hidden_features = int(dim * mlp_ratio)
        if swiglu_mlp:
            if scale_mlp:
                self.mlp = SwiGLU(
                    in_features=dim,
                    hidden_features=hidden_features,
                    norm_layer=norm_layer if scale_mlp else None,
                    drop=proj_drop,
                )
            else:
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
        self.drop_path2 = DropPath(drop_path, drop_path_scale) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, rope: Optional[RopeEmbedding] = None, attn_mask: Optional[torch.Tensor] = None):
        if self.gamma_1 is None:
            x = x + self.drop_path1(self.attn(self.norm1(x), rope=rope, attn_mask=attn_mask))
            x = x + self.drop_path2(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), rope=rope, attn_mask=attn_mask))
            x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class Eva(nn.Module):
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
        pos_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
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
        deeper_embed_patch_chunk_size=None,
        deeper_embed_batch_chunk_size=None,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.patch_size = [patch_size] * 3 if isinstance(patch_size, int) else patch_size
        self.ndim = len(self.patch_size)
        self.embedding_type = str(embedding_type).lower()
        if self.embedding_type != "default":
            raise ValueError(
                f"Unsupported embedding_type={embedding_type!r} for pretrained DINOv2 backbone loading. "
                "Only 'default' is supported."
            )
        self.global_crops_size = [global_crops_size] * 3 if isinstance(global_crops_size, int) else global_crops_size
        self.local_crops_size = [local_crops_size] * 3 if isinstance(local_crops_size, int) else local_crops_size
        self.global_input_size = tuple(int(size) for size in self.global_crops_size)
        self.local_input_size = tuple(int(size) for size in self.local_crops_size)
        self.global_ref_feat_shape = tuple(i // ds for i, ds in zip(self.global_crops_size, self.patch_size))
        self.local_ref_feat_shape = tuple(i // ds for i, ds in zip(self.local_crops_size, self.patch_size))
        self.down_projection = PatchEmbed(
            patch_size=tuple(self.patch_size),
            input_channels=input_channels,
            embed_dim=embed_dim,
        )

        del deeper_embed_patch_chunk_size
        del deeper_embed_batch_chunk_size

        if rope_kwargs is None:
            rope_kwargs = {}

        self.num_features = self.embed_dim = embed_dim
        self.dynamic_img_size = dynamic_img_size
        self.grad_checkpointing = grad_checkpointing
        self.num_reg_tokens = num_reg_tokens
        self.num_class_tokens = 1 if class_token else 0
        self.num_prefix_tokens = self.num_class_tokens + self.num_reg_tokens

        num_patches = np.prod(self.global_ref_feat_shape)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        self.reg_token = nn.Parameter(torch.zeros(1, num_reg_tokens, embed_dim)) if num_reg_tokens else None
        self.cls_embed = class_token and self.reg_token is None
        self.pos_embed = (
            nn.Parameter(torch.zeros(1, num_patches + self.num_class_tokens, embed_dim))
            if use_abs_pos_emb
            else None
        )
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
            self.rope_embed = rope_impl(head_dim, ndim=self.ndim, **rope_kwargs)
        else:
            self.rope_embed = None

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList([
            EvaBlock(
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
                drop_path_scale=drop_path_scale,
            )
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self._init_weights()

    def _init_weights(self):
        def init_fn(module):
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(init_fn)
        self.down_projection.apply(InitWeights_He(1e-2))

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            trunc_normal_(self.cls_token, std=0.02)
        if self.reg_token is not None:
            trunc_normal_(self.reg_token, std=0.02)
        if self.mask_token is not None:
            trunc_normal_(self.mask_token, std=0.02)

        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            if hasattr(layer.attn.proj, "weight"):
                rescale(layer.attn.proj.weight.data, layer_id + 1)
            if hasattr(layer.mlp, "fc2") and hasattr(layer.mlp.fc2, "weight"):
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    def _pos_embed(self, x, *spatial) -> Tuple[torch.Tensor, Optional[RopeEmbedding]]:
        pos_embed = self.pos_embed
        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        source_size = tuple(self.global_ref_feat_shape)
        target_size = tuple(dim // patch for dim, patch in zip(spatial, self.patch_size))
        if source_size != target_size and pos_embed is not None:
            pos_embed = self.interpolate_pos_encoding_nd(
                pos_embed,
                source_size=source_size,
                target_size=target_size,
                num_prefix_tokens=self.num_class_tokens,
            )
        rot_pos_embed = self._get_rot_pos_embed(target_size)

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

    def interpolate_pos_encoding_nd(self, pos_embed, source_size, target_size, num_prefix_tokens=1):
        _, _, channels = pos_embed.shape
        previous_dtype = pos_embed.dtype
        pos_embed = pos_embed.float()

        if num_prefix_tokens > 0:
            pos_prefix, pos_embed = pos_embed[:, :num_prefix_tokens], pos_embed[:, num_prefix_tokens:]
        else:
            pos_prefix = None

        ndim = len(source_size)
        if ndim not in (2, 3):
            raise ValueError(f"Only 2D and 3D positional interpolation are supported, got ndim={ndim}")

        pos_embed = pos_embed.reshape(1, *source_size, channels)
        pos_embed = pos_embed.permute(0, ndim + 1, *range(1, ndim + 1))
        pos_embed = F.interpolate(
            pos_embed,
            size=target_size,
            mode="bilinear" if ndim == 2 else "trilinear",
            align_corners=False,
        )
        pos_embed = pos_embed.permute(0, *range(2, ndim + 2), 1).reshape(1, -1, channels)

        if pos_prefix is not None:
            pos_embed = torch.cat([pos_prefix, pos_embed], dim=1)
        return pos_embed.to(previous_dtype)

    def prepare_tokens_with_masks(self, x, masks=None):
        spatial = tuple(int(dim) for dim in x.shape[2:])
        if any(int(size) % int(patch) != 0 for size, patch in zip(spatial, self.patch_size)):
            raise ValueError(
                f"input shape must be divisible by patch_size, got spatial_shape={spatial} "
                f"and patch_size={tuple(self.patch_size)}"
            )
        x = self.down_projection(x)
        if self.ndim == 2:
            x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        else:
            x = rearrange(x, "b c d h w -> b (d h w) c").contiguous()

        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype), x)

        x, rot_pos_embed = self._pos_embed(x, *spatial)
        return x, rot_pos_embed

    def forward_features_list(self, x_list, masks_list):
        if not isinstance(x_list, list):
            return self.forward_features(x_list, masks_list)
        output = []
        for x, masks in zip(x_list, masks_list):
            output.append(self.forward_features(x, masks))
        return output

    def forward_features(self, x, masks=None):
        x, rot_pos_embed = self.prepare_tokens_with_masks(x, masks)
        for blk in self.blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x, rope=rot_pos_embed)
            else:
                x = blk(x, rope=rot_pos_embed)
        x = self.norm(x)
        return {
            "x_norm_clstoken": x[:, 0] if self.num_class_tokens > 0 else None,
            "x_norm_regtokens": x[:, self.num_class_tokens:self.num_prefix_tokens],
            "x_norm_patchtokens": x[:, self.num_prefix_tokens:],
            "x_prenorm": x,
            "masks": masks,
        }

    def forward(self, x, masks=None, is_training=True):
        del is_training
        return self.forward_features_list(x, masks)

    def load_pretrained_weights(self, state_dict, backbone_only=False, unchunk=False):
        del backbone_only
        if isinstance(state_dict, str):
            state_dict = torch.load(state_dict)["teacher"]
            state_dict = {
                key.replace("backbone.", "", 1): value
                for key, value in state_dict.items()
                if key.startswith("backbone.")
            }
        if unchunk:
            state_dict = self.unchunk_state_dict(state_dict)
        return self.load_state_dict(state_dict)

    def unchunk_state_dict(self, state_dict):
        if not any(key.startswith("blocks.0.0") for key in state_dict.keys()):
            return state_dict

        new_state_dict = OrderedDict()
        for key, val in state_dict.items():
            if key.startswith("blocks."):
                parts = key.split(".")
                if parts[2].isdigit():
                    chunk_idx = int(parts[1])
                    inner_idx = int(parts[2])
                    flat_idx = chunk_idx * 9999 + inner_idx
                    new_key = ".".join(["blocks", str(flat_idx)] + parts[3:])
                    new_state_dict[new_key] = val
                else:
                    new_state_dict[key] = val
            else:
                new_state_dict[key] = val

        mapping = {
            old: new
            for new, old in enumerate(sorted(set(int(k.split(".")[1]) for k in new_state_dict if k.startswith("blocks."))))
        }
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
            chunks.append(BlockChunk(self.blocks[i:i + chunksize]))
        self.blocks = nn.ModuleList(chunks)

    def forward_features(self, x, masks=None):
        x, rot_pos_embed = self.prepare_tokens_with_masks(x, masks)
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
        return {
            "x_norm_clstoken": x[:, 0] if self.num_class_tokens > 0 else None,
            "x_norm_regtokens": x[:, self.num_class_tokens:self.num_prefix_tokens],
            "x_norm_patchtokens": x[:, self.num_prefix_tokens:],
            "x_prenorm": x,
            "masks": masks,
        }
