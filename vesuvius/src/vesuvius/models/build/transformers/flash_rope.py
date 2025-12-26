from typing import Callable, Optional

import torch
from torch import nn
from torch.nn import functional as F

from timm.layers import DropPath, GluMlp, Mlp, SwiGLU
from timm.layers.pos_embed_sincos import apply_rot_embed_cat
from torch.nn import LayerNorm


class FlashRoPEAttention(nn.Module):
    """EVA-style attention with RoPE, backed by flash_attn."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qkv_fused: bool = True,
        qkv_bias_separate: bool = False,
        num_prefix_tokens: int = 1,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        attn_head_dim: Optional[int] = None,
        norm_layer: Optional[Callable] = None,
        qk_norm: bool = False,
        scale_norm: bool = True,
        rotate_half: bool = False,
    ):
        super().__init__()
        if scale_norm or qk_norm:
            assert norm_layer is not None, "norm_layer must be provided if qk_norm or scale_norm is True"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        self.head_dim = head_dim
        attn_dim = head_dim * self.num_heads
        self.scale = head_dim ** -0.5
        self.num_prefix_tokens = num_prefix_tokens
        self.qkv_bias_separate = qkv_bias_separate
        self.rotate_half = rotate_half

        if qkv_fused:
            self.qkv = nn.Linear(dim, attn_dim * 3, bias=False)
            self.q_proj = self.k_proj = self.v_proj = None
            if qkv_bias:
                self.q_bias = nn.Parameter(torch.zeros(attn_dim))
                self.register_buffer("k_bias", torch.zeros(attn_dim), persistent=False)
                self.v_bias = nn.Parameter(torch.zeros(attn_dim))
            else:
                self.q_bias = self.k_bias = self.v_bias = None
        else:
            self.q_proj = nn.Linear(dim, attn_dim, bias=qkv_bias)
            self.k_proj = nn.Linear(dim, attn_dim, bias=False)
            self.v_proj = nn.Linear(dim, attn_dim, bias=qkv_bias)
            self.qkv = None
            self.q_bias = self.k_bias = self.v_bias = None

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.norm = norm_layer(attn_dim) if scale_norm else nn.Identity()
        self.proj = nn.Linear(attn_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        try:
            from flash_attn import flash_attn_func
        except ImportError as exc:
            raise ImportError(
                "flash_attn is required for RoPE attention. Install it via the models extra."
            ) from exc
        self.flash_attn_func = flash_attn_func

    def forward(
        self,
        x: torch.Tensor,
        rope: Optional[torch.Tensor] = None,
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
            q, k, v = qkv.unbind(0)
        else:
            q = self.q_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)
            k = self.k_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)
            v = self.v_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)

        q, k = self.q_norm(q), self.k_norm(k)

        if rope is not None:
            npt = self.num_prefix_tokens
            if npt > 0:
                q = torch.cat(
                    [q[:, :, :npt, :], apply_rot_embed_cat(q[:, :, npt:, :], rope, half=self.rotate_half)],
                    dim=2,
                ).type_as(v)
                k = torch.cat(
                    [k[:, :, :npt, :], apply_rot_embed_cat(k[:, :, npt:, :], rope, half=self.rotate_half)],
                    dim=2,
                ).type_as(v)
            else:
                q = apply_rot_embed_cat(q, rope, half=self.rotate_half).type_as(v)
                k = apply_rot_embed_cat(k, rope, half=self.rotate_half).type_as(v)

        if attn_mask is not None:
            raise ValueError("RoPE flash_attn does not support attn_mask.")
        if not q.is_cuda:
            raise RuntimeError("RoPE flash_attn requires CUDA tensors.")
        if q.dtype not in (torch.float16, torch.bfloat16):
            raise TypeError("RoPE flash_attn requires fp16 or bf16 tensors.")

        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        x = self.flash_attn_func(
            q,
            k,
            v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            softmax_scale=self.scale,
            causal=False,
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FlashRoPEBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        qkv_fused: bool = True,
        mlp_ratio: float = 4.0,
        swiglu_mlp: bool = False,
        swiglu_align_to: int = 0,
        scale_mlp: bool = False,
        scale_attn_inner: bool = False,
        num_prefix_tokens: int = 1,
        attn_type: str = "rope",
        rotate_half: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        init_values: Optional[float] = None,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = LayerNorm,
        attn_head_dim: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = FlashRoPEAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qkv_fused=qkv_fused,
            num_prefix_tokens=num_prefix_tokens,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            attn_head_dim=attn_head_dim,
            norm_layer=norm_layer,
            scale_norm=scale_attn_inner,
            rotate_half=rotate_half,
        )
        self.gamma_1 = nn.Parameter(init_values * torch.ones(dim)) if init_values is not None else None
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        hidden_features = int(dim * mlp_ratio)
        if swiglu_mlp:
            if scale_mlp or swiglu_align_to:
                self.mlp = SwiGLU(
                    in_features=dim,
                    hidden_features=hidden_features,
                    norm_layer=norm_layer if scale_mlp else None,
                    drop=proj_drop,
                    align_to=swiglu_align_to,
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
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        rope: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.gamma_1 is None:
            x = x + self.drop_path1(self.attn(self.norm1(x), rope=rope, attn_mask=attn_mask))
            x = x + self.drop_path2(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), rope=rope, attn_mask=attn_mask))
            x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
        return x
