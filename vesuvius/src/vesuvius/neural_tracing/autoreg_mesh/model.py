from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from vesuvius.models.build.pretrained_backbones.dinov2 import build_dinov2_backbone
from vesuvius.models.build.pretrained_backbones.rope import (
    MixedRopePositionEmbedding,
    apply_rotary_embedding,
)
from vesuvius.neural_tracing.autoreg_mesh.config import validate_autoreg_mesh_config
from vesuvius.neural_tracing.autoreg_mesh.serialization import IGNORE_INDEX


PROMPT_TOKEN_TYPE = 0
GENERATED_TOKEN_TYPE = 1
START_TOKEN_TYPE = 2

_ROPE_DTYPE_ALIASES = {
    "fp32": torch.float32,
    "float32": torch.float32,
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}


def _resolve_rope_dtype(value):
    if value is None or isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _ROPE_DTYPE_ALIASES:
            return _ROPE_DTYPE_ALIASES[normalized]
    raise ValueError(f"unsupported rope_dtype value: {value!r}")
def _batched_rope_from_coords(rope: MixedRopePositionEmbedding, coords: Tensor) -> tuple[Tensor, Tensor]:
    if coords.ndim != 3 or coords.shape[-1] != 3:
        raise ValueError(f"coords must have shape [B, T, 3], got {tuple(coords.shape)}")
    rope_dtype = rope.periods.dtype
    rope_values = [rope.get_embed_from_coords(sample_coords.to(dtype=rope_dtype)) for sample_coords in coords]
    sin = torch.stack([item[0] for item in rope_values], dim=0)
    cos = torch.stack([item[1] for item in rope_values], dim=0)
    return sin, cos


def _batched_shared_rope_from_coords(
    rope: MixedRopePositionEmbedding,
    query_coords: Tensor,
    key_coords: Tensor,
) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
    if query_coords.ndim != 3 or query_coords.shape[-1] != 3:
        raise ValueError(f"query_coords must have shape [B, Tq, 3], got {tuple(query_coords.shape)}")
    if key_coords.ndim != 3 or key_coords.shape[-1] != 3:
        raise ValueError(f"key_coords must have shape [B, Tk, 3], got {tuple(key_coords.shape)}")
    if query_coords.shape[0] != key_coords.shape[0]:
        raise ValueError("query_coords and key_coords must have matching batch size")

    query_sin, query_cos = [], []
    key_sin, key_cos = [], []
    rope_dtype = rope.periods.dtype
    for q_coords, k_coords in zip(query_coords, key_coords, strict=True):
        q_coords = q_coords.to(dtype=rope_dtype)
        k_coords = k_coords.to(dtype=rope_dtype)
        shift, jitter, rescale = rope._sample_coord_augmentation_params(device=q_coords.device, dtype=q_coords.dtype)
        q_coords = rope.apply_coord_augmentation_params(q_coords, shift=shift, jitter=jitter, rescale=rescale)
        k_coords = rope.apply_coord_augmentation_params(k_coords, shift=shift, jitter=jitter, rescale=rescale)
        q_sin, q_cos = rope.get_embed_from_coords(q_coords)
        k_sin, k_cos = rope.get_embed_from_coords(k_coords)
        query_sin.append(q_sin)
        query_cos.append(q_cos)
        key_sin.append(k_sin)
        key_cos.append(k_cos)
    return (
        torch.stack(query_sin, dim=0),
        torch.stack(query_cos, dim=0),
    ), (
        torch.stack(key_sin, dim=0),
        torch.stack(key_cos, dim=0),
    )


class RotarySelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, rope: MixedRopePositionEmbedding, dropout: float = 0.0) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = int(num_heads)
        self.head_dim = int(dim // num_heads)
        self.scale = self.head_dim ** -0.5
        self.rope = rope
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.dropout = float(dropout)

    def forward(self, x: Tensor, coords: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        batch_size, seq_len, dim = x.shape
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        sin, cos = _batched_rope_from_coords(self.rope, coords)
        q = apply_rotary_embedding(q, (sin, cos)).type_as(v)
        k = apply_rotary_embedding(k, (sin, cos)).type_as(v)
        q = q * self.scale
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).reshape(batch_size, seq_len, dim)
        return self.proj(out)


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        rope: MixedRopePositionEmbedding | None = None,
        use_rope: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = int(num_heads)
        self.head_dim = int(dim // num_heads)
        self.scale = self.head_dim ** -0.5
        self.rope = rope
        self.use_rope = bool(use_rope and rope is not None)
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)
        self.dropout = float(dropout)

    def forward(self, x: Tensor, memory: Tensor, *, query_coords: Tensor | None = None, memory_coords: Tensor | None = None) -> Tensor:
        batch_size, seq_len, dim = x.shape
        memory_len = int(memory.shape[1])
        q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(memory).reshape(batch_size, memory_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(memory).reshape(batch_size, memory_len, self.num_heads, self.head_dim).transpose(1, 2)
        if self.use_rope:
            if query_coords is None or memory_coords is None:
                raise ValueError("query_coords and memory_coords are required when cross-attention RoPE is enabled")
            (q_sin, q_cos), (k_sin, k_cos) = _batched_shared_rope_from_coords(self.rope, query_coords, memory_coords)
            q = apply_rotary_embedding(q, (q_sin, q_cos)).type_as(v)
            k = apply_rotary_embedding(k, (k_sin, k_cos)).type_as(v)
        q = q * self.scale
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).reshape(batch_size, seq_len, dim)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=True),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, dim, bias=True),
            nn.Dropout(float(dropout)),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        rope: MixedRopePositionEmbedding,
        *,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        enable_cross_attention: bool = True,
        cross_attention_use_rope: bool = True,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = RotarySelfAttention(dim, num_heads, rope=rope, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = (
            CrossAttention(dim, num_heads, rope=rope, use_rope=cross_attention_use_rope, dropout=dropout)
            if enable_cross_attention else None
        )
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x: Tensor, coords: Tensor, attn_mask: Tensor, memory: Tensor, memory_coords: Tensor) -> Tensor:
        x = x + self.self_attn(self.norm1(x), coords=coords, attn_mask=attn_mask)
        if self.cross_attn is not None:
            x = x + self.cross_attn(self.norm2(x), memory=memory, query_coords=coords, memory_coords=memory_coords)
        x = x + self.ffn(self.norm3(x))
        return x


class AutoregMeshModel(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = validate_autoreg_mesh_config(config)
        self.input_shape = tuple(int(v) for v in self.config["input_shape"])
        self.patch_size = tuple(int(v) for v in self.config["patch_size"])
        self.coarse_grid_shape = (
            int(self.input_shape[0] // self.patch_size[0]),
            int(self.input_shape[1] // self.patch_size[1]),
            int(self.input_shape[2] // self.patch_size[2]),
        )
        self.offset_num_bins = tuple(int(v) for v in self.config["offset_num_bins"])
        self.decoder_dim = int(self.config["decoder_dim"])
        self.decoder_depth = int(self.config["decoder_depth"])
        self.decoder_num_heads = int(self.config["decoder_num_heads"])
        self.head_dim = self.decoder_dim // self.decoder_num_heads
        self.cross_attention_every_n_blocks = int(self.config["cross_attention_every_n_blocks"])
        self.pointer_temperature = float(self.config["pointer_temperature"])
        self.coarse_prediction_mode = str(self.config["coarse_prediction_mode"])
        self.conditioning_feature_debias_mode = str(self.config["conditioning_feature_debias_mode"])
        self.conditioning_feature_debias_basis_source = str(self.config["conditioning_feature_debias_basis_source"])
        self.conditioning_feature_debias_components = int(self.config["conditioning_feature_debias_components"])

        self.backbone = None
        memory_in_dim = self.decoder_dim
        backbone_name = self.config.get("dinov2_backbone")
        if backbone_name:
            self.backbone = build_dinov2_backbone(
                backbone_name,
                int(self.config.get("input_channels", 1)),
                self.input_shape,
                config_path=self.config.get("dinov2_config_path"),
            )
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
            self.backbone.eval()
            memory_in_dim = int(self.backbone.embed_dim)
            if self.conditioning_feature_debias_mode != "none":
                if self.conditioning_feature_debias_components >= memory_in_dim:
                    raise ValueError(
                        "conditioning_feature_debias_components must be smaller than Dinovol embed_dim, "
                        f"got components={self.conditioning_feature_debias_components} embed_dim={memory_in_dim}"
                    )

        if self.conditioning_feature_debias_mode != "none" and self.backbone is None:
            raise ValueError("conditioning_feature_debias_mode requires a configured dinov2_backbone")

        self.memory_proj = nn.Linear(memory_in_dim, self.decoder_dim, bias=True)
        self.memory_coord_mlp = nn.Sequential(
            nn.Linear(3, self.decoder_dim, bias=True),
            nn.GELU(),
            nn.Linear(self.decoder_dim, self.decoder_dim, bias=True),
        )
        self.xyz_mlp = nn.Sequential(
            nn.Linear(3, self.decoder_dim, bias=True),
            nn.GELU(),
            nn.Linear(self.decoder_dim, self.decoder_dim, bias=True),
        )
        self.lattice_mlp = nn.Sequential(
            nn.Linear(2, self.decoder_dim, bias=True),
            nn.GELU(),
            nn.Linear(self.decoder_dim, self.decoder_dim, bias=True),
        )
        self.direction_embedding = nn.Embedding(4, self.decoder_dim)
        self.token_type_embedding = nn.Embedding(3, self.decoder_dim)
        self.offset_embeddings = nn.ModuleList(
            [nn.Embedding(num_bins, self.decoder_dim) for num_bins in self.offset_num_bins]
        )
        self.start_token = nn.Parameter(torch.zeros(self.decoder_dim))
        self.register_buffer("coarse_cell_starts", self._build_coarse_cell_starts(), persistent=False)

        self.rope = MixedRopePositionEmbedding(
            self.head_dim,
            ndim=3,
            num_heads=self.decoder_num_heads,
            base=self.config.get("rope_base"),
            min_period=self.config.get("rope_min_period"),
            max_period=self.config.get("rope_max_period"),
            normalize_coords=self.config.get("rope_normalize_coords", "separate"),
            shift_coords=self.config.get("rope_shift_coords"),
            jitter_coords=self.config.get("rope_jitter_coords"),
            rescale_coords=self.config.get("rope_rescale_coords"),
            dtype=_resolve_rope_dtype(self.config.get("rope_dtype")),
        )
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(
                    self.decoder_dim,
                    self.decoder_num_heads,
                    rope=self.rope,
                    mlp_ratio=float(self.config["decoder_mlp_ratio"]),
                    dropout=float(self.config["decoder_dropout"]),
                    enable_cross_attention=(block_idx % self.cross_attention_every_n_blocks == 0),
                    cross_attention_use_rope=bool(self.config.get("cross_attention_use_rope", True)),
                )
                for block_idx in range(self.decoder_depth)
            ]
        )
        self.final_norm = nn.LayerNorm(self.decoder_dim)
        if self.coarse_prediction_mode == "joint_pointer":
            self.pointer_query = nn.Linear(self.decoder_dim, self.decoder_dim, bias=True)
            self.pointer_key = nn.Linear(self.decoder_dim, self.decoder_dim, bias=False)
            self.pointer_query_z = None
            self.pointer_query_y = None
            self.pointer_query_x = None
            self.pointer_key_z = None
            self.pointer_key_y = None
            self.pointer_key_x = None
        else:
            self.pointer_query = None
            self.pointer_key = None
            self.pointer_query_z = nn.Linear(self.decoder_dim, self.decoder_dim, bias=True)
            self.pointer_query_y = nn.Linear(self.decoder_dim, self.decoder_dim, bias=True)
            self.pointer_query_x = nn.Linear(self.decoder_dim, self.decoder_dim, bias=True)
            self.pointer_key_z = nn.Linear(self.decoder_dim, self.decoder_dim, bias=False)
            self.pointer_key_y = nn.Linear(self.decoder_dim, self.decoder_dim, bias=False)
            self.pointer_key_x = nn.Linear(self.decoder_dim, self.decoder_dim, bias=False)
        self.offset_head = nn.Linear(self.decoder_dim, 3 * max(self.offset_num_bins), bias=True)
        self.stop_head = nn.Linear(self.decoder_dim, 1, bias=True)
        self.position_refine_head = nn.Linear(self.decoder_dim, 3, bias=True)
        nn.init.zeros_(self.position_refine_head.weight)
        nn.init.zeros_(self.position_refine_head.bias)
        debias_basis = torch.empty(0, 0, dtype=torch.float32)
        if self.conditioning_feature_debias_mode != "none":
            debias_basis = self._build_conditioning_feature_debias_basis(
                device=next(self.backbone.parameters()).device,
            )
        self.register_buffer("conditioning_feature_debias_basis", debias_basis, persistent=False)
        self.last_conditioning_feature_debias_norm_ratio: float = 1.0

    def train(self, mode: bool = True):
        super().train(mode)
        if self.backbone is not None:
            self.backbone.eval()
        return self

    def _normalize_xyz(self, xyz: Tensor) -> Tensor:
        shape = torch.tensor(self.input_shape, device=xyz.device, dtype=xyz.dtype)
        return (2.0 * (xyz / torch.clamp(shape - 1.0, min=1.0))) - 1.0

    def _build_coarse_cell_starts(self) -> Tensor:
        grid_shape = (
            int(self.input_shape[0] // self.patch_size[0]),
            int(self.input_shape[1] // self.patch_size[1]),
            int(self.input_shape[2] // self.patch_size[2]),
        )
        patch = torch.tensor(self.patch_size, dtype=torch.float32)
        z = torch.arange(grid_shape[0], dtype=torch.float32) * patch[0]
        y = torch.arange(grid_shape[1], dtype=torch.float32) * patch[1]
        x = torch.arange(grid_shape[2], dtype=torch.float32) * patch[2]
        return torch.stack(torch.meshgrid(z, y, x, indexing="ij"), dim=-1).reshape(-1, 3)

    def _make_memory_patch_centers(self, grid_shape: tuple[int, int, int], *, device: torch.device, dtype: torch.dtype) -> Tensor:
        gz, gy, gx = grid_shape
        patch = torch.tensor(self.patch_size, device=device, dtype=dtype)
        z_axis = (torch.arange(gz, device=device, dtype=dtype) + 0.5) * patch[0]
        y_axis = (torch.arange(gy, device=device, dtype=dtype) + 0.5) * patch[1]
        x_axis = (torch.arange(gx, device=device, dtype=dtype) + 0.5) * patch[2]
        coords = torch.stack(torch.meshgrid(z_axis, y_axis, x_axis, indexing="ij"), dim=-1).reshape(-1, 3)
        return self._normalize_xyz(coords)

    @torch.no_grad()
    def _build_conditioning_feature_debias_basis(self, *, device: torch.device) -> Tensor:
        if self.conditioning_feature_debias_basis_source != "zero_volume_svd":
            raise ValueError(
                "conditioning_feature_debias_basis_source must currently be 'zero_volume_svd', "
                f"got {self.conditioning_feature_debias_basis_source!r}"
            )
        zero_volume = torch.zeros(
            (1, int(self.config.get("input_channels", 1)), *self.input_shape),
            device=device,
            dtype=torch.float32,
        )
        features = self.backbone(zero_volume)[0]
        features = F.normalize(features, p=2, dim=1)
        channels = int(features.shape[1])
        flat = features.reshape(channels, -1)
        flat = flat - flat.mean(dim=1, keepdim=True)
        u, _, _ = torch.linalg.svd(flat, full_matrices=False)
        return u[:, :self.conditioning_feature_debias_components].contiguous().to(dtype=torch.float32)

    def _debias_conditioning_features(self, raw_tokens: Tensor) -> Tensor:
        if self.conditioning_feature_debias_mode == "none":
            self.last_conditioning_feature_debias_norm_ratio = 1.0
            return raw_tokens
        basis = self.conditioning_feature_debias_basis.to(device=raw_tokens.device, dtype=torch.float32)
        x = raw_tokens.to(torch.float32)
        original_norm = torch.linalg.norm(x, dim=-1, keepdim=True)
        x_t = x.transpose(1, 2)
        coeff = torch.matmul(basis.transpose(0, 1).unsqueeze(0), x_t)
        projection = torch.matmul(basis.unsqueeze(0), coeff)
        debiased = x_t - projection
        debiased = debiased.transpose(1, 2)
        debiased_norm = torch.linalg.norm(debiased, dim=-1, keepdim=True)
        scale = original_norm / debiased_norm.clamp(min=1e-6)
        debiased = debiased * scale
        ratio = (debiased_norm / original_norm.clamp(min=1e-6)).mean().item()
        self.last_conditioning_feature_debias_norm_ratio = float(ratio)
        return debiased.to(dtype=raw_tokens.dtype)

    def encode_conditioning(self, volume: Tensor | None, vol_tokens: Tensor | None = None) -> dict[str, Tensor]:
        if vol_tokens is None:
            if self.backbone is None:
                raise ValueError("AutoregMeshModel requires vol_tokens or a configured dinov2_backbone")
            if volume is None:
                raise ValueError("volume must be provided when vol_tokens is None")
            with torch.no_grad():
                features = self.backbone(volume)[0]
            batch_size, channels, gz, gy, gx = features.shape
            raw_tokens = features.flatten(2).transpose(1, 2).contiguous()
            grid_shape = (int(gz), int(gy), int(gx))
        else:
            raw_tokens = vol_tokens
            batch_size = int(raw_tokens.shape[0])
            grid_shape = tuple(int(v) for v in (self.input_shape[0] // self.patch_size[0], self.input_shape[1] // self.patch_size[1], self.input_shape[2] // self.patch_size[2]))

        raw_tokens = self._debias_conditioning_features(raw_tokens)
        memory_tokens = self.memory_proj(raw_tokens)
        patch_centers = self._make_memory_patch_centers(
            grid_shape,
            device=memory_tokens.device,
            dtype=memory_tokens.dtype,
        )
        patch_centers = patch_centers.unsqueeze(0).expand(batch_size, -1, -1)
        memory_tokens = memory_tokens + self.memory_coord_mlp(patch_centers)
        return {
            "memory_tokens": memory_tokens,
            "memory_patch_centers": patch_centers,
            "coarse_grid_shape": grid_shape,
        }

    def _unflatten_coarse_ids(self, coarse_ids: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        gz, gy, gx = self.coarse_grid_shape
        coarse = coarse_ids.clamp(min=0)
        z = coarse // (gy * gx)
        rem = coarse % (gy * gx)
        y = rem // gx
        x = rem % gx
        return z, y, x

    def _flatten_coarse_axis_ids(self, z: Tensor, y: Tensor, x: Tensor) -> Tensor:
        _, gy, gx = self.coarse_grid_shape
        return (z * (gy * gx)) + (y * gx) + x

    def _factorized_axis_memory(self, memory_tokens: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        gz, gy, gx = self.coarse_grid_shape
        memory_5d = memory_tokens.reshape(memory_tokens.shape[0], gz, gy, gx, memory_tokens.shape[-1])
        z_memory = memory_5d.mean(dim=(2, 3))
        y_memory = memory_5d.mean(dim=(1, 3))
        x_memory = memory_5d.mean(dim=(1, 2))
        return z_memory, y_memory, x_memory

    def _compute_coarse_outputs(self, hidden: Tensor, memory_tokens: Tensor) -> dict[str, Any]:
        if self.coarse_prediction_mode == "joint_pointer":
            pointer_q = F.normalize(self.pointer_query(hidden), dim=-1)
            pointer_k = F.normalize(self.pointer_key(memory_tokens), dim=-1)
            coarse_logits = torch.einsum("btd,bnd->btn", pointer_q, pointer_k) / self.pointer_temperature
            pred_coarse_ids = coarse_logits.argmax(dim=-1)
            z_ids, y_ids, x_ids = self._unflatten_coarse_ids(pred_coarse_ids)
            return {
                "coarse_logits": coarse_logits,
                "coarse_axis_logits": None,
                "pred_coarse_ids": pred_coarse_ids,
                "pred_coarse_axis_ids": {"z": z_ids, "y": y_ids, "x": x_ids},
            }

        z_memory, y_memory, x_memory = self._factorized_axis_memory(memory_tokens)
        z_query = F.normalize(self.pointer_query_z(hidden), dim=-1)
        y_query = F.normalize(self.pointer_query_y(hidden), dim=-1)
        x_query = F.normalize(self.pointer_query_x(hidden), dim=-1)
        z_key = F.normalize(self.pointer_key_z(z_memory), dim=-1)
        y_key = F.normalize(self.pointer_key_y(y_memory), dim=-1)
        x_key = F.normalize(self.pointer_key_x(x_memory), dim=-1)
        z_logits = torch.einsum("btd,bzd->btz", z_query, z_key) / self.pointer_temperature
        y_logits = torch.einsum("btd,byd->bty", y_query, y_key) / self.pointer_temperature
        x_logits = torch.einsum("btd,bxd->btx", x_query, x_key) / self.pointer_temperature
        z_ids = z_logits.argmax(dim=-1)
        y_ids = y_logits.argmax(dim=-1)
        x_ids = x_logits.argmax(dim=-1)
        pred_coarse_ids = self._flatten_coarse_axis_ids(z_ids, y_ids, x_ids)
        return {
            "coarse_logits": None,
            "coarse_axis_logits": {"z": z_logits, "y": y_logits, "x": x_logits},
            "pred_coarse_ids": pred_coarse_ids,
            "pred_coarse_axis_ids": {"z": z_ids, "y": y_ids, "x": x_ids},
        }

    def _soft_decode_local_xyz(
        self,
        coarse_logits: Tensor | None,
        coarse_axis_logits: dict[str, Tensor] | None,
        offset_logits: Tensor,
        pred_refine_residual: Tensor,
    ) -> Tensor:
        if self.coarse_prediction_mode == "joint_pointer":
            if coarse_logits is None:
                raise ValueError("coarse_logits are required for joint_pointer mode")
            coarse_probs = torch.softmax(coarse_logits, dim=-1)
            coarse_starts = self.coarse_cell_starts.to(device=coarse_logits.device, dtype=coarse_logits.dtype)
            expected_coarse_start = torch.einsum("btn,nc->btc", coarse_probs, coarse_starts)
        else:
            if coarse_axis_logits is None:
                raise ValueError("coarse_axis_logits are required for axis_factorized mode")
            expected_starts = []
            for axis_name, size, patch in zip(
                ("z", "y", "x"),
                self.coarse_grid_shape,
                self.patch_size,
                strict=True,
            ):
                axis_logits = coarse_axis_logits[axis_name]
                axis_probs = torch.softmax(axis_logits, dim=-1)
                axis_positions = torch.arange(size, device=axis_logits.device, dtype=axis_logits.dtype)
                expected_starts.append(torch.einsum("btn,n->bt", axis_probs, axis_positions) * float(patch))
            expected_coarse_start = torch.stack(expected_starts, dim=-1)

        expected_offsets = []
        for axis, bins in enumerate(self.offset_num_bins):
            axis_logits = offset_logits[:, :, axis, :bins]
            axis_probs = torch.softmax(axis_logits, dim=-1)
            bin_width = float(self.patch_size[axis]) / float(bins)
            bin_centers = (torch.arange(bins, device=axis_logits.device, dtype=axis_logits.dtype) + 0.5) * bin_width
            expected_offsets.append(torch.einsum("btn,n->bt", axis_probs, bin_centers))
        expected_offset = torch.stack(expected_offsets, dim=-1)
        return expected_coarse_start + expected_offset + pred_refine_residual.to(dtype=expected_coarse_start.dtype)

    def _gather_memory_tokens(self, memory_tokens: Tensor, coarse_ids: Tensor, valid_mask: Tensor) -> Tensor:
        safe_ids = torch.where(valid_mask, coarse_ids.clamp(min=0), torch.zeros_like(coarse_ids))
        gathered = torch.gather(
            memory_tokens,
            dim=1,
            index=safe_ids.unsqueeze(-1).expand(-1, -1, memory_tokens.shape[-1]),
        )
        return gathered * valid_mask.unsqueeze(-1).to(dtype=gathered.dtype)

    def _embed_offset_bins(self, offset_bins: Tensor, valid_mask: Tensor) -> Tensor:
        embedded = torch.zeros(
            offset_bins.shape[0],
            offset_bins.shape[1],
            self.decoder_dim,
            device=offset_bins.device,
            dtype=self.start_token.dtype,
        )
        for axis, embedding in enumerate(self.offset_embeddings):
            axis_bins = offset_bins[..., axis]
            axis_valid = valid_mask & (axis_bins >= 0)
            safe_bins = torch.where(axis_valid, axis_bins, torch.zeros_like(axis_bins))
            axis_embed = embedding(safe_bins)
            embedded = embedded + axis_embed * axis_valid.unsqueeze(-1).to(dtype=axis_embed.dtype)
        return embedded

    def _build_input_embeddings(
        self,
        *,
        coarse_ids: Tensor,
        offset_bins: Tensor,
        xyz: Tensor,
        strip_coords: Tensor,
        direction_id: Tensor,
        token_type: Tensor,
        sequence_mask: Tensor,
        geometry_valid_mask: Tensor,
        memory_tokens: Tensor,
    ) -> tuple[Tensor, Tensor]:
        gathered_memory = self._gather_memory_tokens(
            memory_tokens,
            coarse_ids,
            geometry_valid_mask & (coarse_ids >= 0),
        )
        offset_embed = self._embed_offset_bins(offset_bins, geometry_valid_mask)
        safe_xyz = torch.where(geometry_valid_mask.unsqueeze(-1), xyz, torch.zeros_like(xyz))
        xyz_norm = self._normalize_xyz(safe_xyz)
        x = gathered_memory
        x = x + offset_embed
        x = x + self.xyz_mlp(xyz_norm) * geometry_valid_mask.unsqueeze(-1).to(dtype=x.dtype)
        x = x + self.lattice_mlp(strip_coords)
        x = x + self.direction_embedding(direction_id).unsqueeze(1)
        x = x + self.token_type_embedding(token_type)
        start_mask = (token_type == START_TOKEN_TYPE) & sequence_mask
        if start_mask.any():
            x = x + start_mask.unsqueeze(-1).to(dtype=x.dtype) * self.start_token.view(1, 1, -1)
        x = x * sequence_mask.unsqueeze(-1).to(dtype=x.dtype)
        return x, xyz_norm

    def _build_teacher_forced_generation_inputs(self, batch: dict) -> dict[str, Tensor]:
        target_mask = batch["target_mask"]
        target_valid_mask = batch.get("target_valid_mask", batch["target_coarse_ids"] >= 0)
        target_coarse_ids = batch["target_coarse_ids"]
        target_offset_bins = batch["target_offset_bins"]
        target_xyz = batch["target_xyz"]
        target_strip_coords = batch["target_strip_coords"]

        shifted_coarse = torch.full_like(target_coarse_ids, IGNORE_INDEX)
        shifted_offset = torch.full_like(target_offset_bins, IGNORE_INDEX)
        shifted_xyz = torch.zeros_like(target_xyz)
        shifted_valid = torch.zeros_like(target_mask, dtype=torch.bool)
        if target_coarse_ids.shape[1] > 1:
            shifted_coarse[:, 1:] = target_coarse_ids[:, :-1]
            shifted_offset[:, 1:, :] = target_offset_bins[:, :-1, :]
            shifted_xyz[:, 1:, :] = target_xyz[:, :-1, :]
            shifted_valid[:, 1:] = target_valid_mask[:, :-1]
        shifted_xyz[:, 0, :] = batch["prompt_anchor_xyz"]
        shifted_valid[:, 0] = batch.get("prompt_anchor_valid", torch.ones_like(target_mask[:, 0], dtype=torch.bool))

        token_type = torch.full_like(target_coarse_ids, GENERATED_TOKEN_TYPE)
        token_type[:, 0] = START_TOKEN_TYPE
        return {
            "coarse_ids": shifted_coarse,
            "offset_bins": shifted_offset,
            "xyz": shifted_xyz,
            "strip_coords": target_strip_coords,
            "token_type": token_type,
            "sequence_mask": target_mask,
            "geometry_valid_mask": shifted_valid,
        }

    def _build_scheduled_generation_inputs(
        self,
        batch: dict,
        *,
        teacher_outputs: dict[str, Tensor],
        scheduled_sampling_prob: float,
        scheduled_sampling_pattern: str,
        offset_feedback_enabled: bool,
        refine_feedback_enabled: bool,
    ) -> dict[str, Tensor]:
        generation_inputs = self._build_teacher_forced_generation_inputs(batch)
        if float(scheduled_sampling_prob) <= 0.0:
            return generation_inputs

        if str(scheduled_sampling_pattern) != "stripwise_full_strip_greedy":
            raise ValueError(
                "scheduled_sampling_pattern must currently be 'stripwise_full_strip_greedy', "
                f"got {scheduled_sampling_pattern!r}"
            )

        target_mask = generation_inputs["sequence_mask"]
        batch_size, target_len = target_mask.shape
        if target_len <= 1:
            return generation_inputs

        replace_mask = torch.zeros_like(target_mask, dtype=torch.bool)
        shifted_strip_ids = torch.zeros_like(target_mask, dtype=torch.long)
        shifted_strip_ids[:, 1:] = batch["target_strip_positions"][:, :-1, 0]
        for batch_idx in range(batch_size):
            num_strips = int(batch["num_strips"][batch_idx].item())
            if num_strips <= 0:
                continue
            sampled_strips = torch.rand((num_strips,), device=target_mask.device) < float(scheduled_sampling_prob)
            replace_mask[batch_idx, 1:] = sampled_strips[shifted_strip_ids[batch_idx, 1:].clamp(min=0, max=max(num_strips - 1, 0))]
        replace_mask &= target_mask
        if not bool(replace_mask.any()):
            return generation_inputs

        shifted_pred_coarse = torch.full_like(generation_inputs["coarse_ids"], IGNORE_INDEX)
        shifted_pred_offset = torch.full_like(generation_inputs["offset_bins"], IGNORE_INDEX)
        shifted_pred_xyz = torch.zeros_like(generation_inputs["xyz"])
        shifted_pred_valid = torch.zeros_like(generation_inputs["geometry_valid_mask"], dtype=torch.bool)
        shifted_pred_coarse[:, 1:] = teacher_outputs["pred_coarse_ids"][:, :-1]
        if bool(offset_feedback_enabled):
            shifted_pred_offset[:, 1:, :] = teacher_outputs["pred_offset_bins"][:, :-1, :]
            shifted_pred_xyz[:, 1:, :] = (
                teacher_outputs["pred_xyz_refined"][:, :-1, :]
                if bool(refine_feedback_enabled)
                else teacher_outputs["pred_xyz"][:, :-1, :]
            )
        shifted_pred_valid[:, 1:] = True

        generation_inputs["coarse_ids"] = torch.where(replace_mask, shifted_pred_coarse, generation_inputs["coarse_ids"])
        if bool(offset_feedback_enabled):
            generation_inputs["offset_bins"] = torch.where(
                replace_mask.unsqueeze(-1),
                shifted_pred_offset,
                generation_inputs["offset_bins"],
            )
            generation_inputs["xyz"] = torch.where(
                replace_mask.unsqueeze(-1),
                shifted_pred_xyz,
                generation_inputs["xyz"],
            )
            generation_inputs["geometry_valid_mask"] = torch.where(
                replace_mask,
                shifted_pred_valid,
                generation_inputs["geometry_valid_mask"],
            )
        return generation_inputs

    def _build_attention_mask(self, seq_mask: Tensor, *, dtype: torch.dtype) -> Tensor:
        batch_size, seq_len = seq_mask.shape
        causal = torch.tril(torch.ones((seq_len, seq_len), device=seq_mask.device, dtype=torch.bool))
        valid = seq_mask[:, None, None, :]
        allowed = causal[None, None, :, :] & valid
        attn_mask = torch.zeros((batch_size, 1, seq_len, seq_len), device=seq_mask.device, dtype=dtype)
        attn_mask = attn_mask.masked_fill(~allowed, float("-inf"))
        return attn_mask

    def forward_from_encoded(
        self,
        batch: dict,
        *,
        memory_tokens: Tensor,
        memory_patch_centers: Tensor,
        generation_inputs: dict[str, Tensor] | None = None,
    ) -> dict[str, Tensor]:
        prompt = batch["prompt_tokens"]
        prompt_token_type = torch.full_like(prompt["coarse_ids"], PROMPT_TOKEN_TYPE)
        prompt_embeddings, prompt_coords = self._build_input_embeddings(
            coarse_ids=prompt["coarse_ids"],
            offset_bins=prompt["offset_bins"],
            xyz=prompt["xyz"],
            strip_coords=prompt["strip_coords"],
            direction_id=batch["direction_id"],
            token_type=prompt_token_type,
            sequence_mask=prompt["mask"],
            geometry_valid_mask=prompt["valid_mask"],
            memory_tokens=memory_tokens,
        )

        if generation_inputs is None:
            generation_inputs = self._build_teacher_forced_generation_inputs(batch)
        generation_embeddings, generation_coords = self._build_input_embeddings(
            coarse_ids=generation_inputs["coarse_ids"],
            offset_bins=generation_inputs["offset_bins"],
            xyz=generation_inputs["xyz"],
            strip_coords=generation_inputs["strip_coords"],
            direction_id=batch["direction_id"],
            token_type=generation_inputs["token_type"],
            sequence_mask=generation_inputs["sequence_mask"],
            geometry_valid_mask=generation_inputs["geometry_valid_mask"],
            memory_tokens=memory_tokens,
        )

        seq_embeddings = torch.cat([prompt_embeddings, generation_embeddings], dim=1)
        seq_coords = torch.cat([prompt_coords, generation_coords], dim=1)
        seq_mask = torch.cat([prompt["mask"], generation_inputs["sequence_mask"]], dim=1)
        attn_mask = self._build_attention_mask(seq_mask, dtype=seq_embeddings.dtype)

        x = seq_embeddings
        for block in self.blocks:
            x = block(x, coords=seq_coords, attn_mask=attn_mask, memory=memory_tokens, memory_coords=memory_patch_centers)
            x = x * seq_mask.unsqueeze(-1).to(dtype=x.dtype)
        x = self.final_norm(x)

        prompt_len = int(prompt["mask"].shape[1])
        hidden = x[:, prompt_len:, :]
        coarse_outputs = self._compute_coarse_outputs(hidden, memory_tokens)

        max_bins = max(self.offset_num_bins)
        offset_logits = self.offset_head(hidden).reshape(hidden.shape[0], hidden.shape[1], 3, max_bins)
        stop_logits = self.stop_head(hidden).squeeze(-1)
        pred_offset_bins = []
        for axis, bins in enumerate(self.offset_num_bins):
            pred_offset_bins.append(offset_logits[:, :, axis, :bins].argmax(dim=-1))
        pred_offset_bins_tensor = torch.stack(pred_offset_bins, dim=-1)
        pred_xyz_bin_center = self.decode_local_xyz(coarse_outputs["pred_coarse_ids"], pred_offset_bins_tensor)
        pred_refine_residual = self.position_refine_head(hidden)
        pred_xyz_soft = self._soft_decode_local_xyz(
            coarse_outputs["coarse_logits"],
            coarse_outputs["coarse_axis_logits"],
            offset_logits,
            pred_refine_residual,
        )
        pred_xyz_refined = pred_xyz_bin_center + pred_refine_residual

        return {
            "coarse_logits": coarse_outputs["coarse_logits"],
            "coarse_axis_logits": coarse_outputs["coarse_axis_logits"],
            "offset_logits": offset_logits,
            "stop_logits": stop_logits,
            "pred_coarse_ids": coarse_outputs["pred_coarse_ids"],
            "pred_coarse_axis_ids": coarse_outputs["pred_coarse_axis_ids"],
            "pred_offset_bins": pred_offset_bins_tensor,
            "pred_refine_residual": pred_refine_residual,
            "pred_xyz": pred_xyz_bin_center,
            "pred_xyz_soft": pred_xyz_soft,
            "pred_xyz_refined": pred_xyz_refined,
            "memory_tokens": memory_tokens,
            "coarse_grid_shape": self.coarse_grid_shape,
            "coarse_prediction_mode": self.coarse_prediction_mode,
        }

    def forward(
        self,
        batch: dict,
        *,
        scheduled_sampling_prob: float = 0.0,
        scheduled_sampling_pattern: str = "stripwise_full_strip_greedy",
        scheduled_sampling_offset_feedback_enabled: bool = True,
        scheduled_sampling_refine_feedback_enabled: bool = True,
    ) -> dict[str, Tensor]:
        encoded = self.encode_conditioning(batch.get("volume"), batch.get("vol_tokens"))
        generation_inputs = None
        if self.training and float(scheduled_sampling_prob) > 0.0:
            with torch.no_grad():
                teacher_outputs = self.forward_from_encoded(
                    batch,
                    memory_tokens=encoded["memory_tokens"],
                    memory_patch_centers=encoded["memory_patch_centers"],
                )
            generation_inputs = self._build_scheduled_generation_inputs(
                batch,
                teacher_outputs=teacher_outputs,
                scheduled_sampling_prob=float(scheduled_sampling_prob),
                scheduled_sampling_pattern=str(scheduled_sampling_pattern),
                offset_feedback_enabled=bool(scheduled_sampling_offset_feedback_enabled),
                refine_feedback_enabled=bool(scheduled_sampling_refine_feedback_enabled),
            )
        outputs = self.forward_from_encoded(
            batch,
            memory_tokens=encoded["memory_tokens"],
            memory_patch_centers=encoded["memory_patch_centers"],
            generation_inputs=generation_inputs,
        )
        outputs["coarse_grid_shape"] = encoded["coarse_grid_shape"]
        return outputs

    def decode_local_xyz(self, coarse_ids: Tensor, offset_bins: Tensor) -> Tensor:
        grid_shape = (
            int(self.input_shape[0] // self.patch_size[0]),
            int(self.input_shape[1] // self.patch_size[1]),
            int(self.input_shape[2] // self.patch_size[2]),
        )
        gyx = grid_shape[1] * grid_shape[2]
        coarse = coarse_ids.clamp(min=0)
        z = coarse // gyx
        rem = coarse % gyx
        y = rem // grid_shape[2]
        x = rem % grid_shape[2]
        cell = torch.stack([z, y, x], dim=-1).to(dtype=torch.float32)
        patch = torch.tensor(self.patch_size, device=coarse_ids.device, dtype=torch.float32)
        starts = cell * patch.view(1, 1, 3)

        coords = torch.zeros_like(offset_bins, dtype=torch.float32)
        for axis, bins in enumerate(self.offset_num_bins):
            width = float(self.patch_size[axis]) / float(bins)
            coords[..., axis] = starts[..., axis] + (offset_bins[..., axis].to(dtype=torch.float32) + 0.5) * width
        max_coord = torch.tensor(self.input_shape, device=coarse_ids.device, dtype=torch.float32) - 1e-4
        coords = torch.maximum(coords, torch.zeros_like(coords))
        return torch.minimum(coords, max_coord.view(1, 1, 3))


def build_pseudo_inference_batch(
    *,
    prompt_tokens: dict[str, Tensor],
    prompt_anchor_xyz: Tensor,
    direction_id: Tensor,
    target_coarse_ids: Tensor,
    target_offset_bins: Tensor,
    target_xyz: Tensor,
    target_strip_coords: Tensor,
) -> dict[str, Tensor]:
    batch_size, target_len = target_coarse_ids.shape
    target_mask = torch.ones((batch_size, target_len), device=target_coarse_ids.device, dtype=torch.bool)
    return {
        "prompt_tokens": prompt_tokens,
        "prompt_anchor_xyz": prompt_anchor_xyz,
        "prompt_anchor_valid": torch.ones((batch_size,), device=prompt_anchor_xyz.device, dtype=torch.bool),
        "direction_id": direction_id,
        "target_coarse_ids": target_coarse_ids,
        "target_offset_bins": target_offset_bins,
        "target_valid_mask": target_coarse_ids >= 0,
        "target_xyz": target_xyz,
        "target_strip_coords": target_strip_coords,
        "target_mask": target_mask,
    }
