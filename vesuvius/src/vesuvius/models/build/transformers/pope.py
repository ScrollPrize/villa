import math
from typing import Callable, List, Optional

import torch
from torch import nn
from torch.nn import functional as F

from timm.layers import DropPath, GluMlp, Mlp, SwiGLU
from timm.layers.pos_embed_sincos import build_fourier_pos_embed, freq_bands, pixel_freq_bands
from torch.nn import LayerNorm


def build_pope_pos_embed(
    feat_shape: List[int],
    bands: Optional[torch.Tensor] = None,
    dim: int = 64,
    max_res: int = 224,
    temperature: float = 10000.0,
    linear_bands: bool = False,
    in_pixels: bool = True,
    ref_feat_shape: Optional[List[int]] = None,
    grid_offset: float = 0.0,
    grid_indexing: str = "ij",
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
):
    num_spatial = len(feat_shape)
    if bands is None:
        if dim < num_spatial:
            raise ValueError(f"PoPE dim ({dim}) must be >= number of spatial dims ({num_spatial}).")
        num_bands = max(1, dim // num_spatial)
        if in_pixels:
            bands = pixel_freq_bands(
                num_bands,
                float(max_res),
                linear_bands=linear_bands,
                device=device,
            )
        else:
            bands = freq_bands(
                num_bands,
                temperature=temperature,
                step=1,
                device=device,
            )
    else:
        if device is None:
            device = bands.device
        if dtype is None:
            dtype = bands.dtype

    sin_emb, cos_emb = build_fourier_pos_embed(
        feat_shape,
        bands=bands,
        num_bands=bands.shape[0],
        max_res=max_res,
        temperature=temperature,
        linear_bands=linear_bands,
        in_pixels=in_pixels,
        ref_feat_shape=ref_feat_shape,
        grid_offset=grid_offset,
        grid_indexing=grid_indexing,
        device=device,
        dtype=dtype,
    )
    num_spatial_dim = 1
    for x in feat_shape:
        num_spatial_dim *= x
    sin_emb = sin_emb.reshape(num_spatial_dim, -1)
    cos_emb = cos_emb.reshape(num_spatial_dim, -1)
    return sin_emb, cos_emb


class PoPEEmbedding(nn.Module):
    """Polar Coordinate Positional Embedding (PoPE) sin/cos builder."""

    def __init__(
        self,
        dim: int,
        max_res: int = 224,
        temperature: float = 10000,
        in_pixels: bool = True,
        linear_bands: bool = False,
        feat_shape: Optional[List[int]] = None,
        ref_feat_shape: Optional[List[int]] = None,
        grid_offset: float = 0.0,
        grid_indexing: str = "ij",
    ):
        super().__init__()
        self.dim = dim
        self.max_res = max_res
        self.temperature = temperature
        self.in_pixels = in_pixels
        self.linear_bands = linear_bands
        self.feat_shape = feat_shape
        self.ref_feat_shape = ref_feat_shape
        self.grid_offset = grid_offset
        self.grid_indexing = grid_indexing

        num_spatial = 1
        if feat_shape is not None:
            num_spatial = len(feat_shape)
        elif ref_feat_shape is not None:
            num_spatial = len(ref_feat_shape)
        if dim < num_spatial:
            raise ValueError(f"PoPE dim ({dim}) must be >= number of spatial dims ({num_spatial}).")
        num_bands = max(1, dim // num_spatial)
        self.effective_dim = num_spatial * num_bands

        if feat_shape is None:
            if in_pixels:
                bands = pixel_freq_bands(
                    num_bands,
                    float(max_res),
                    linear_bands=linear_bands,
                )
            else:
                bands = freq_bands(
                    num_bands,
                    temperature=temperature,
                    step=1,
                )
            self.register_buffer("bands", bands, persistent=False)
            self.pos_embed = None
        else:
            self.bands = None
            self.register_buffer(
                "pos_embed",
                self._get_pos_embed_values(feat_shape=feat_shape),
                persistent=False,
            )

    def _get_pos_embed_values(self, feat_shape: List[int]):
        sin_emb, cos_emb = build_pope_pos_embed(
            feat_shape=feat_shape,
            dim=self.dim,
            max_res=self.max_res,
            temperature=self.temperature,
            linear_bands=self.linear_bands,
            in_pixels=self.in_pixels,
            ref_feat_shape=self.ref_feat_shape,
            grid_offset=self.grid_offset,
            grid_indexing=self.grid_indexing,
        )
        return torch.cat([sin_emb, cos_emb], -1)

    def update_feat_shape(self, feat_shape: List[int]):
        if self.feat_shape is not None and feat_shape != self.feat_shape:
            assert self.pos_embed is not None
            self.pos_embed = self._get_pos_embed_values(feat_shape).to(
                device=self.pos_embed.device,
                dtype=self.pos_embed.dtype,
            )
            self.feat_shape = feat_shape

    def get_embed(self, shape: Optional[List[int]] = None):
        if shape is not None and self.bands is not None:
            sin_emb, cos_emb = build_pope_pos_embed(
                shape,
                bands=self.bands,
                dim=self.dim,
                max_res=self.max_res,
                temperature=self.temperature,
                linear_bands=self.linear_bands,
                in_pixels=self.in_pixels,
                ref_feat_shape=self.ref_feat_shape,
                grid_offset=self.grid_offset,
                grid_indexing=self.grid_indexing,
            )
            return torch.cat([sin_emb, cos_emb], -1)
        elif self.pos_embed is not None:
            return self.pos_embed
        else:
            raise RuntimeError("get_embed() requires pre-computed pos embed or valid shape with cached bands")


class PoPEAttention(nn.Module):
    """EVA attention with PoPE positional encoding."""

    fused_attn: torch.jit.Final[bool]

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

        self.phase_bias = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))
        try:
            from flash_attn import flash_attn_func
        except ImportError as exc:
            raise ImportError(
                "flash_attn is required for PoPEAttention. Install it via the models extra."
            ) from exc
        self.flash_attn_func = flash_attn_func

    def _apply_pope(self, q: torch.Tensor, k: torch.Tensor, rope: torch.Tensor):
        npt = self.num_prefix_tokens
        sin_emb, cos_emb = rope.tensor_split(2, dim=-1)
        if sin_emb.ndim == 2:
            sin_emb = sin_emb.unsqueeze(0).unsqueeze(0)  # [1, 1, N, D]
            cos_emb = cos_emb.unsqueeze(0).unsqueeze(0)
        elif sin_emb.ndim == 3:
            sin_emb = sin_emb.unsqueeze(1)  # [B, 1, N, D]
            cos_emb = cos_emb.unsqueeze(1)
        else:
            raise ValueError(f"Unexpected PoPE embedding shape: {rope.shape}")

        q_prefix, k_prefix = q[:, :, :npt, :], k[:, :, :npt, :]
        q_body, k_body = q[:, :, npt:, :], k[:, :, npt:, :]

        mu_q_body = F.softplus(q_body)
        mu_k_body = F.softplus(k_body)

        delta = self.phase_bias.to(dtype=cos_emb.dtype).clamp(min=-2 * math.pi, max=0.0)
        cos_delta = torch.cos(delta).unsqueeze(0).unsqueeze(2)
        sin_delta = torch.sin(delta).unsqueeze(0).unsqueeze(2)
        cos_k = cos_emb * cos_delta - sin_emb * sin_delta
        sin_k = sin_emb * cos_delta + cos_emb * sin_delta

        q_cos = mu_q_body * cos_emb
        q_sin = mu_q_body * sin_emb
        k_cos = mu_k_body * cos_k
        k_sin = mu_k_body * sin_k

        if npt > 0:
            mu_q_prefix = F.softplus(q_prefix)
            mu_k_prefix = F.softplus(k_prefix)
            zeros = torch.zeros_like(mu_q_prefix)
            q_cos = torch.cat([mu_q_prefix, q_cos], dim=2)
            q_sin = torch.cat([zeros, q_sin], dim=2)
            k_cos = torch.cat([mu_k_prefix, k_cos], dim=2)
            k_sin = torch.cat([zeros, k_sin], dim=2)

        q = torch.cat([q_cos, q_sin], dim=-1)
        k = torch.cat([k_cos, k_sin], dim=-1)
        return q, k

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
            q, k = self._apply_pope(q, k, rope)
            q = q.type_as(v)
            k = k.type_as(v)

        if attn_mask is not None:
            raise ValueError("PoPEAttention with flash_attn does not support attn_mask.")
        if not q.is_cuda:
            raise RuntimeError("PoPEAttention with flash_attn requires CUDA tensors.")
        if q.dtype not in (torch.float16, torch.bfloat16):
            raise TypeError("PoPEAttention with flash_attn requires fp16 or bf16 tensors.")

        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        x = self.flash_attn_func(
            q,
            k,
            v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            softmax_scale=self.scale * math.sqrt(2.0),
            causal=False,
        )
        x = x.transpose(1, 2)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PoPEBlock(nn.Module):
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
        attn_type: str = "pope",
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
        self.attn = PoPEAttention(
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
