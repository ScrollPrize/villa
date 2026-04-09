from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from vesuvius.models.build.pretrained_backbones.dinov2 import build_dinov2_backbone
from vesuvius.neural_tracing.autoreg_mesh.config import validate_autoreg_mesh_config
from vesuvius.neural_tracing.autoreg_mesh.serialization import IGNORE_INDEX


PROMPT_TOKEN_TYPE = 0
GENERATED_TOKEN_TYPE = 1
START_TOKEN_TYPE = 2


def _rotate_half(x: Tensor) -> Tensor:
    x_first, x_second = x.chunk(2, dim=-1)
    return torch.cat((-x_second, x_first), dim=-1)


def _apply_rotary_embedding(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    return (x * cos) + (_rotate_half(x) * sin)


class ExplicitCoordMixedRope(nn.Module):
    """Minimal mixed 3D RoPE with explicit-coordinate support for decoder tokens."""

    def __init__(
        self,
        head_dim: int,
        *,
        num_heads: int,
        ndim: int = 3,
        base: float = 100.0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if ndim != 3:
            raise ValueError(f"ExplicitCoordMixedRope expects ndim=3, got {ndim}")
        if head_dim % (2 * ndim) != 0:
            raise ValueError(
                f"head_dim must be divisible by {2 * ndim} for mixed 3D RoPE, got head_dim={head_dim}"
            )
        self.head_dim = int(head_dim)
        self.num_heads = int(num_heads)
        self.ndim = int(ndim)
        self.dtype = dtype
        self.freqs_per_axis = self.head_dim // (2 * self.ndim)
        self.axis_dim = self.head_dim // self.ndim
        self.num_pairs = self.head_dim // 2
        self.base = float(base)

        self.register_buffer("periods", torch.empty(self.freqs_per_axis, dtype=dtype), persistent=True)
        self.mix_frequencies = nn.Parameter(
            torch.empty(self.num_heads, self.num_pairs, self.ndim, dtype=dtype)
        )
        self._init_weights()

    def _build_periods(self, count: int, denominator_dim: int, *, device: torch.device | None = None) -> Tensor:
        device = device or torch.device("cpu")
        return self.base ** (2 * torch.arange(count, device=device, dtype=self.dtype) / denominator_dim)

    def _init_weights(self) -> None:
        periods = self._build_periods(self.freqs_per_axis, self.axis_dim, device=self.periods.device)
        self.periods.data.copy_(periods)
        self.reset_mixed_frequencies_to_random_oriented()

    @torch.no_grad()
    def reset_mixed_frequencies_to_random_oriented(self) -> None:
        inv_periods = self.periods.reciprocal()
        basis = torch.randn(
            self.num_heads,
            self.ndim,
            self.ndim,
            device=self.periods.device,
            dtype=self.periods.dtype,
        )
        basis, r = torch.linalg.qr(basis)
        diag = torch.diagonal(r, dim1=-2, dim2=-1)
        signs = torch.where(diag < 0, -torch.ones_like(diag), torch.ones_like(diag))
        basis = basis * signs.unsqueeze(-2)
        negative_det = torch.linalg.det(basis) < 0
        if negative_det.any():
            basis[negative_det, :, 0] *= -1

        mixed = torch.empty_like(self.mix_frequencies)
        for axis in range(self.ndim):
            start = axis * self.freqs_per_axis
            end = start + self.freqs_per_axis
            mixed[:, start:end, :] = inv_periods[None, :, None] * basis[:, None, :, axis]
        self.mix_frequencies.copy_(mixed)

    def get_embed_from_coords(self, coords: Tensor) -> tuple[Tensor, Tensor]:
        coords = coords.to(dtype=self.periods.dtype)
        if coords.ndim != 3 or coords.shape[-1] != self.ndim:
            raise ValueError(f"coords must have shape [B, T, {self.ndim}], got {tuple(coords.shape)}")
        angles = 2.0 * math.pi * torch.einsum("btd,hpd->bhtp", coords, self.mix_frequencies)
        angles = torch.cat([angles, angles], dim=-1)
        return torch.sin(angles), torch.cos(angles)


class RotarySelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, rope: ExplicitCoordMixedRope, dropout: float = 0.0) -> None:
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

        sin, cos = self.rope.get_embed_from_coords(coords)
        q = _apply_rotary_embedding(q, sin, cos).to(dtype=v.dtype)
        k = _apply_rotary_embedding(k, sin, cos).to(dtype=v.dtype)
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
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = int(num_heads)
        self.head_dim = int(dim // num_heads)
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)
        self.dropout = float(dropout)

    def forward(self, x: Tensor, memory: Tensor) -> Tensor:
        batch_size, seq_len, dim = x.shape
        memory_len = int(memory.shape[1])
        q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(memory).reshape(batch_size, memory_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(memory).reshape(batch_size, memory_len, self.num_heads, self.head_dim).transpose(1, 2)
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
        rope: ExplicitCoordMixedRope,
        *,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        enable_cross_attention: bool = True,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = RotarySelfAttention(dim, num_heads, rope=rope, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(dim, num_heads, dropout=dropout) if enable_cross_attention else None
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x: Tensor, coords: Tensor, attn_mask: Tensor, memory: Tensor) -> Tensor:
        x = x + self.self_attn(self.norm1(x), coords=coords, attn_mask=attn_mask)
        if self.cross_attn is not None:
            x = x + self.cross_attn(self.norm2(x), memory=memory)
        x = x + self.ffn(self.norm3(x))
        return x


class AutoregMeshModel(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = validate_autoreg_mesh_config(config)
        self.input_shape = tuple(int(v) for v in self.config["input_shape"])
        self.patch_size = tuple(int(v) for v in self.config["patch_size"])
        self.offset_num_bins = tuple(int(v) for v in self.config["offset_num_bins"])
        self.decoder_dim = int(self.config["decoder_dim"])
        self.decoder_depth = int(self.config["decoder_depth"])
        self.decoder_num_heads = int(self.config["decoder_num_heads"])
        self.head_dim = self.decoder_dim // self.decoder_num_heads
        self.cross_attention_every_n_blocks = int(self.config["cross_attention_every_n_blocks"])
        self.pointer_temperature = float(self.config["pointer_temperature"])

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

        self.rope = ExplicitCoordMixedRope(
            self.head_dim,
            num_heads=self.decoder_num_heads,
            ndim=3,
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
                )
                for block_idx in range(self.decoder_depth)
            ]
        )
        self.final_norm = nn.LayerNorm(self.decoder_dim)
        self.pointer_query = nn.Linear(self.decoder_dim, self.decoder_dim, bias=True)
        self.pointer_key = nn.Linear(self.decoder_dim, self.decoder_dim, bias=False)
        self.offset_head = nn.Linear(self.decoder_dim, 3 * max(self.offset_num_bins), bias=True)
        self.stop_head = nn.Linear(self.decoder_dim, 1, bias=True)

    def _normalize_xyz(self, xyz: Tensor) -> Tensor:
        shape = torch.tensor(self.input_shape, device=xyz.device, dtype=xyz.dtype)
        return (2.0 * (xyz / torch.clamp(shape - 1.0, min=1.0))) - 1.0

    def _make_memory_patch_centers(self, grid_shape: tuple[int, int, int], *, device: torch.device, dtype: torch.dtype) -> Tensor:
        gz, gy, gx = grid_shape
        patch = torch.tensor(self.patch_size, device=device, dtype=dtype)
        z_axis = (torch.arange(gz, device=device, dtype=dtype) + 0.5) * patch[0]
        y_axis = (torch.arange(gy, device=device, dtype=dtype) + 0.5) * patch[1]
        x_axis = (torch.arange(gx, device=device, dtype=dtype) + 0.5) * patch[2]
        coords = torch.stack(torch.meshgrid(z_axis, y_axis, x_axis, indexing="ij"), dim=-1).reshape(-1, 3)
        return self._normalize_xyz(coords)

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
        }

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
    ) -> dict[str, Tensor]:
        generation_inputs = self._build_teacher_forced_generation_inputs(batch)
        if float(scheduled_sampling_prob) <= 0.0:
            return generation_inputs

        target_mask = generation_inputs["sequence_mask"]
        batch_size, target_len = target_mask.shape
        if target_len <= 1:
            return generation_inputs

        replace_mask = torch.zeros_like(target_mask, dtype=torch.bool)
        sampled = torch.rand((batch_size, target_len - 1), device=target_mask.device) < float(scheduled_sampling_prob)
        replace_mask[:, 1:] = sampled & target_mask[:, 1:]
        if not bool(replace_mask.any()):
            return generation_inputs

        shifted_pred_coarse = torch.full_like(generation_inputs["coarse_ids"], IGNORE_INDEX)
        shifted_pred_offset = torch.full_like(generation_inputs["offset_bins"], IGNORE_INDEX)
        shifted_pred_xyz = torch.zeros_like(generation_inputs["xyz"])
        shifted_pred_valid = torch.zeros_like(generation_inputs["geometry_valid_mask"], dtype=torch.bool)
        shifted_pred_coarse[:, 1:] = teacher_outputs["pred_coarse_ids"][:, :-1]
        shifted_pred_offset[:, 1:, :] = teacher_outputs["pred_offset_bins"][:, :-1, :]
        shifted_pred_xyz[:, 1:, :] = teacher_outputs["pred_xyz"][:, :-1, :]
        shifted_pred_valid[:, 1:] = True

        generation_inputs["coarse_ids"] = torch.where(replace_mask, shifted_pred_coarse, generation_inputs["coarse_ids"])
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
            x = block(x, coords=seq_coords, attn_mask=attn_mask, memory=memory_tokens)
            x = x * seq_mask.unsqueeze(-1).to(dtype=x.dtype)
        x = self.final_norm(x)

        prompt_len = int(prompt["mask"].shape[1])
        hidden = x[:, prompt_len:, :]
        pointer_q = F.normalize(self.pointer_query(hidden), dim=-1)
        pointer_k = F.normalize(self.pointer_key(memory_tokens), dim=-1)
        coarse_logits = torch.einsum("btd,bnd->btn", pointer_q, pointer_k) / self.pointer_temperature

        max_bins = max(self.offset_num_bins)
        offset_logits = self.offset_head(hidden).reshape(hidden.shape[0], hidden.shape[1], 3, max_bins)
        stop_logits = self.stop_head(hidden).squeeze(-1)
        pred_coarse_ids = coarse_logits.argmax(dim=-1)
        pred_offset_bins = []
        for axis, bins in enumerate(self.offset_num_bins):
            pred_offset_bins.append(offset_logits[:, :, axis, :bins].argmax(dim=-1))
        pred_offset_bins_tensor = torch.stack(pred_offset_bins, dim=-1)

        return {
            "coarse_logits": coarse_logits,
            "offset_logits": offset_logits,
            "stop_logits": stop_logits,
            "pred_coarse_ids": pred_coarse_ids,
            "pred_offset_bins": pred_offset_bins_tensor,
            "pred_xyz": self.decode_local_xyz(pred_coarse_ids, pred_offset_bins_tensor),
            "memory_tokens": memory_tokens,
        }

    def forward(self, batch: dict, *, scheduled_sampling_prob: float = 0.0) -> dict[str, Tensor]:
        encoded = self.encode_conditioning(batch.get("volume"), batch.get("vol_tokens"))
        generation_inputs = None
        if self.training and float(scheduled_sampling_prob) > 0.0:
            with torch.no_grad():
                teacher_outputs = self.forward_from_encoded(batch, memory_tokens=encoded["memory_tokens"])
            generation_inputs = self._build_scheduled_generation_inputs(
                batch,
                teacher_outputs=teacher_outputs,
                scheduled_sampling_prob=float(scheduled_sampling_prob),
            )
        return self.forward_from_encoded(
            batch,
            memory_tokens=encoded["memory_tokens"],
            generation_inputs=generation_inputs,
        )

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
