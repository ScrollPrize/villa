from __future__ import annotations

import dataclasses
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
from vesuvius.neural_tracing.autoreg_fiber.config import validate_autoreg_fiber_config
from vesuvius.neural_tracing.autoreg_fiber.serialization import IGNORE_INDEX
from vesuvius.neural_tracing.autoreg_mesh.model import (
    ATTENTION_SCALING_STANDARD,
    DecoderBlock,
    _batched_rope_from_coords,
    _normalize_attention_scaling_mode,
    _resolve_rope_dtype,
)


PROMPT_TOKEN_TYPE = 0
GENERATED_TOKEN_TYPE = 1
START_TOKEN_TYPE = 2


@dataclasses.dataclass
class FiberKVCache:
    """Per-layer KV cache for streaming autoregressive inference on the fiber decoder.

    `self_attn_kv[i]` is the cumulative `(k, v)` for block `i`'s self-attention,
    shape `(B, num_heads, seq_len, head_dim)` after the embedding chain of
    `prompt + START + N stepped tokens`. `cross_attn_kv[i]` holds the K/V over
    the (static, per-window) memory tokens for blocks that have cross-attention;
    it is `None` for blocks without.

    The cache also stores the last input token's `xyz` and `valid` so that the
    tangent-conditioning input on the next step is identical to what
    `forward_from_encoded` would produce on a length-(t+1) generation buffer.
    Optionally caches the pointer-head key projections (constant per window).
    """

    self_attn_kv: list[tuple[Tensor, Tensor]]
    cross_attn_kv: list[tuple[Tensor, Tensor] | None]
    seq_len: int
    last_input_xyz: Tensor
    last_input_valid: Tensor
    pointer_key_norm: Tensor | None = None
    axis_pointer_keys: dict[str, Tensor] | None = None


class AutoregFiberModel(nn.Module):
    """Autoregressive decoder for one ordered fiber path inside one volume crop."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = validate_autoreg_fiber_config(config)
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
        self.attention_scaling_mode = _normalize_attention_scaling_mode(self.config.get("attention_scaling_mode"))
        self.pointer_temperature = float(self.config["pointer_temperature"])
        self.conditioning_feature_debias_mode = str(self.config["conditioning_feature_debias_mode"])
        self.conditioning_feature_debias_basis_source = str(self.config["conditioning_feature_debias_basis_source"])
        self.conditioning_feature_debias_components = int(self.config["conditioning_feature_debias_components"])
        self.tangent_conditioning_enabled = bool(self.config.get("tangent_conditioning_enabled", True))
        self.coarse_prediction_mode = str(self.config["coarse_prediction_mode"])

        self.backbone = None
        memory_in_dim = int(self.config.get("input_channels", 1))
        backbone_name = self.config.get("dinov2_backbone")
        if backbone_name:
            self.backbone = build_dinov2_backbone(
                backbone_name,
                int(self.config.get("input_channels", 1)),
                self.input_shape,
                config_path=self.config.get("dinov2_config_path"),
            )
            backbone_patch_size = tuple(int(v) for v in self.backbone.patch_embed_size)
            if backbone_patch_size != self.patch_size:
                raise ValueError(
                    "autoreg_fiber patch_size must match the selected Dinov2 backbone patch stride; "
                    f"config patch_size={self.patch_size!r} backbone patch_embed_size={backbone_patch_size!r}"
                )
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
            self.backbone.eval()
            memory_in_dim = int(self.backbone.embed_dim)
            if (
                self.conditioning_feature_debias_mode != "none"
                and self.conditioning_feature_debias_components >= memory_in_dim
            ):
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
        self.tangent_mlp = nn.Sequential(
            nn.Linear(3, self.decoder_dim, bias=True),
            nn.GELU(),
            nn.Linear(self.decoder_dim, self.decoder_dim, bias=True),
        )
        self.position_embedding = nn.Embedding(int(self.config["max_fiber_position_embeddings"]), self.decoder_dim)
        self.token_type_embedding = nn.Embedding(3, self.decoder_dim)
        self.offset_embeddings = nn.ModuleList(
            [nn.Embedding(num_bins, self.decoder_dim) for num_bins in self.offset_num_bins]
        )
        self.start_token = nn.Parameter(torch.zeros(self.decoder_dim))

        self.register_buffer("coarse_cell_starts", self._build_coarse_cell_starts(), persistent=False)
        patch = torch.tensor(self.patch_size, dtype=torch.float32)
        self.register_buffer("coarse_cell_centers", self.coarse_cell_starts + 0.5 * patch.view(1, 3), persistent=False)

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
                    attention_scaling_mode=self.attention_scaling_mode,
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
        patch = torch.tensor(self.patch_size, dtype=torch.float32)
        z = torch.arange(self.coarse_grid_shape[0], dtype=torch.float32) * patch[0]
        y = torch.arange(self.coarse_grid_shape[1], dtype=torch.float32) * patch[1]
        x = torch.arange(self.coarse_grid_shape[2], dtype=torch.float32) * patch[2]
        return torch.stack(torch.meshgrid(z, y, x, indexing="ij"), dim=-1).reshape(-1, 3)

    def _make_memory_patch_centers(self, grid_shape: tuple[int, int, int], *, device: torch.device, dtype: torch.dtype) -> Tensor:
        gz, gy, gx = grid_shape
        patch = torch.tensor(self.patch_size, device=device, dtype=dtype)
        z_axis = (torch.arange(gz, device=device, dtype=dtype) + 0.5) * patch[0]
        y_axis = (torch.arange(gy, device=device, dtype=dtype) + 0.5) * patch[1]
        x_axis = (torch.arange(gx, device=device, dtype=dtype) + 0.5) * patch[2]
        coords = torch.stack(torch.meshgrid(z_axis, y_axis, x_axis, indexing="ij"), dim=-1).reshape(-1, 3)
        return self._normalize_xyz(coords)

    @torch.inference_mode()
    def _build_conditioning_feature_debias_basis(self, *, device: torch.device) -> Tensor:
        if self.conditioning_feature_debias_basis_source != "zero_volume_svd":
            raise ValueError("conditioning_feature_debias_basis_source must currently be 'zero_volume_svd'")
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
        debiased = debiased * (original_norm / debiased_norm.clamp(min=1e-6))
        ratio = (debiased_norm / original_norm.clamp(min=1e-6)).mean().item()
        self.last_conditioning_feature_debias_norm_ratio = float(ratio)
        return debiased.to(dtype=raw_tokens.dtype)

    def encode_conditioning(self, volume: Tensor | None, vol_tokens: Tensor | None = None) -> dict[str, Tensor]:
        if vol_tokens is None:
            if volume is None:
                raise ValueError("volume must be provided when vol_tokens is None")
            if self.backbone is not None:
                with torch.no_grad():
                    features = self.backbone(volume)[0]
            else:
                features = F.avg_pool3d(volume, kernel_size=self.patch_size, stride=self.patch_size)
            batch_size, channels, gz, gy, gx = features.shape
            raw_tokens = features.flatten(2).transpose(1, 2).contiguous()
            grid_shape = (int(gz), int(gy), int(gx))
        else:
            raw_tokens = vol_tokens
            batch_size = int(raw_tokens.shape[0])
            grid_shape = self.coarse_grid_shape
        raw_tokens = self._debias_conditioning_features(raw_tokens)
        memory_tokens = self.memory_proj(raw_tokens)
        patch_centers = self._make_memory_patch_centers(grid_shape, device=memory_tokens.device, dtype=memory_tokens.dtype)
        patch_centers = patch_centers.unsqueeze(0).expand(batch_size, -1, -1)
        memory_tokens = memory_tokens + self.memory_coord_mlp(patch_centers)
        return {
            "memory_tokens": memory_tokens,
            "memory_patch_centers": patch_centers,
            "coarse_grid_shape": grid_shape,
        }

    def _unflatten_coarse_ids(self, coarse_ids: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        _, gy, gx = self.coarse_grid_shape
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
        mem_f = memory_5d.float()
        z_memory = mem_f.mean(dim=(2, 3))
        y_memory = mem_f.mean(dim=(1, 3))
        x_memory = mem_f.mean(dim=(1, 2))
        return z_memory, y_memory, x_memory

    def _compute_coarse_outputs(self, hidden: Tensor, memory_tokens: Tensor) -> dict[str, Any]:
        if self.coarse_prediction_mode == "joint_pointer":
            with torch.autocast(device_type=hidden.device.type, enabled=False):
                hidden_f = hidden.float()
                mem_f = memory_tokens.float()
                pointer_q = F.normalize(self.pointer_query(hidden_f), dim=-1)
                pointer_k = F.normalize(self.pointer_key(mem_f), dim=-1)
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
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            hidden_f = hidden.float()
            z_query = F.normalize(self.pointer_query_z(hidden_f), dim=-1)
            y_query = F.normalize(self.pointer_query_y(hidden_f), dim=-1)
            x_query = F.normalize(self.pointer_query_x(hidden_f), dim=-1)
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
            for axis_name, size, patch in zip(("z", "y", "x"), self.coarse_grid_shape, self.patch_size, strict=True):
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

    def _tangent_vectors(self, xyz: Tensor, valid_mask: Tensor) -> Tensor:
        tangent = torch.zeros_like(xyz)
        if xyz.shape[1] <= 1:
            return tangent
        prev = torch.zeros_like(xyz)
        prev[:, 1:, :] = xyz[:, :-1, :]
        prev_valid = torch.zeros_like(valid_mask)
        prev_valid[:, 1:] = valid_mask[:, :-1]
        raw = xyz - prev
        raw = torch.where((valid_mask & prev_valid).unsqueeze(-1), raw, torch.zeros_like(raw))
        norm = raw.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return raw / norm

    def _build_input_embeddings(
        self,
        *,
        coarse_ids: Tensor,
        offset_bins: Tensor,
        xyz: Tensor,
        positions: Tensor,
        token_type: Tensor,
        sequence_mask: Tensor,
        geometry_valid_mask: Tensor,
        memory_tokens: Tensor,
    ) -> tuple[Tensor, Tensor]:
        geometry_valid = geometry_valid_mask & torch.isfinite(xyz).all(dim=-1)
        gathered_memory = self._gather_memory_tokens(memory_tokens, coarse_ids, geometry_valid & (coarse_ids >= 0))
        offset_embed = self._embed_offset_bins(offset_bins, geometry_valid)
        safe_xyz = torch.where(geometry_valid.unsqueeze(-1), xyz, torch.zeros_like(xyz))
        xyz_norm = self._normalize_xyz(safe_xyz)
        x = gathered_memory
        x = x + offset_embed
        x = x + self.xyz_mlp(xyz_norm) * geometry_valid.unsqueeze(-1).to(dtype=x.dtype)
        safe_positions = positions.clamp(min=0, max=int(self.config["max_fiber_position_embeddings"]) - 1)
        x = x + self.position_embedding(safe_positions)
        if self.tangent_conditioning_enabled:
            tangent = self._tangent_vectors(safe_xyz, geometry_valid)
            x = x + self.tangent_mlp(tangent) * geometry_valid.unsqueeze(-1).to(dtype=x.dtype)
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
        target_positions = batch["target_positions"]

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
            "positions": target_positions,
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
        offset_feedback_enabled: bool,
        refine_feedback_enabled: bool,
    ) -> dict[str, Tensor]:
        generation_inputs = self._build_teacher_forced_generation_inputs(batch)
        if float(scheduled_sampling_prob) <= 0.0:
            return generation_inputs
        target_mask = generation_inputs["sequence_mask"]
        if target_mask.shape[1] <= 1:
            return generation_inputs
        replace_mask = torch.zeros_like(target_mask, dtype=torch.bool)
        replace_mask[:, 1:] = torch.rand_like(target_mask[:, 1:].to(dtype=torch.float32)) < float(scheduled_sampling_prob)
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
        _, seq_len = seq_mask.shape
        causal = torch.tril(torch.ones((seq_len, seq_len), device=seq_mask.device, dtype=torch.bool))
        valid = seq_mask[:, None, None, :]
        allowed = causal[None, None, :, :] & valid
        attn_mask = torch.zeros((seq_mask.shape[0], 1, seq_len, seq_len), device=seq_mask.device, dtype=dtype)
        return attn_mask.masked_fill(~allowed, float("-inf"))

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
            positions=prompt["positions"],
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
            positions=generation_inputs["positions"],
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
        scheduled_sampling_pattern: str = "linear_token_greedy",
        scheduled_sampling_offset_feedback_enabled: bool = True,
        scheduled_sampling_refine_feedback_enabled: bool = True,
    ) -> dict[str, Tensor]:
        if str(scheduled_sampling_pattern) != "linear_token_greedy":
            raise ValueError("autoreg_fiber scheduled_sampling_pattern must be 'linear_token_greedy'")
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
        gyx = self.coarse_grid_shape[1] * self.coarse_grid_shape[2]
        coarse = coarse_ids.clamp(min=0)
        z = coarse // gyx
        rem = coarse % gyx
        y = rem // self.coarse_grid_shape[2]
        x = rem % self.coarse_grid_shape[2]
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

    # --- KV-cache fast inference -------------------------------------------- #

    def _precompute_pointer_key_cache(self, memory_tokens: Tensor) -> tuple[Tensor | None, dict[str, Tensor] | None]:
        """Precompute the per-window pointer-key tensors used by the coarse head.

        Returns ``(pointer_key_norm, axis_pointer_keys)`` where exactly one is non-None
        depending on ``coarse_prediction_mode``. The values are constant across all
        generation steps that share the same memory tokens, so callers can reuse them.
        """

        if self.coarse_prediction_mode == "joint_pointer":
            with torch.autocast(device_type=memory_tokens.device.type, enabled=False):
                mem_f = memory_tokens.float()
                return F.normalize(self.pointer_key(mem_f), dim=-1), None
        z_memory, y_memory, x_memory = self._factorized_axis_memory(memory_tokens)
        with torch.autocast(device_type=memory_tokens.device.type, enabled=False):
            axis_keys = {
                "z": F.normalize(self.pointer_key_z(z_memory), dim=-1),
                "y": F.normalize(self.pointer_key_y(y_memory), dim=-1),
                "x": F.normalize(self.pointer_key_x(x_memory), dim=-1),
            }
        return None, axis_keys

    def _compute_outputs_from_hidden(
        self,
        hidden: Tensor,
        memory_tokens: Tensor,
        *,
        pointer_key_norm: Tensor | None = None,
        axis_pointer_keys: dict[str, Tensor] | None = None,
    ) -> dict[str, Any]:
        """Run the prediction heads on a hidden slice. Mirrors the head section of
        ``forward_from_encoded`` but supports optional cached pointer keys so that the
        constant per-window projections are not redone on every step."""

        if self.coarse_prediction_mode == "joint_pointer":
            with torch.autocast(device_type=hidden.device.type, enabled=False):
                hidden_f = hidden.float()
                if pointer_key_norm is None:
                    mem_f = memory_tokens.float()
                    pointer_k = F.normalize(self.pointer_key(mem_f), dim=-1)
                else:
                    pointer_k = pointer_key_norm
                pointer_q = F.normalize(self.pointer_query(hidden_f), dim=-1)
                coarse_logits = torch.einsum("btd,bnd->btn", pointer_q, pointer_k) / self.pointer_temperature
            pred_coarse_ids = coarse_logits.argmax(dim=-1)
            z_ids, y_ids, x_ids = self._unflatten_coarse_ids(pred_coarse_ids)
            coarse_outputs: dict[str, Any] = {
                "coarse_logits": coarse_logits,
                "coarse_axis_logits": None,
                "pred_coarse_ids": pred_coarse_ids,
                "pred_coarse_axis_ids": {"z": z_ids, "y": y_ids, "x": x_ids},
            }
        else:
            if axis_pointer_keys is None:
                _, axis_pointer_keys = self._precompute_pointer_key_cache(memory_tokens)
            assert axis_pointer_keys is not None
            with torch.autocast(device_type=hidden.device.type, enabled=False):
                hidden_f = hidden.float()
                z_query = F.normalize(self.pointer_query_z(hidden_f), dim=-1)
                y_query = F.normalize(self.pointer_query_y(hidden_f), dim=-1)
                x_query = F.normalize(self.pointer_query_x(hidden_f), dim=-1)
                z_logits = torch.einsum("btd,bzd->btz", z_query, axis_pointer_keys["z"]) / self.pointer_temperature
                y_logits = torch.einsum("btd,byd->bty", y_query, axis_pointer_keys["y"]) / self.pointer_temperature
                x_logits = torch.einsum("btd,bxd->btx", x_query, axis_pointer_keys["x"]) / self.pointer_temperature
            z_ids = z_logits.argmax(dim=-1)
            y_ids = y_logits.argmax(dim=-1)
            x_ids = x_logits.argmax(dim=-1)
            pred_coarse_ids = self._flatten_coarse_axis_ids(z_ids, y_ids, x_ids)
            coarse_outputs = {
                "coarse_logits": None,
                "coarse_axis_logits": {"z": z_logits, "y": y_logits, "x": x_logits},
                "pred_coarse_ids": pred_coarse_ids,
                "pred_coarse_axis_ids": {"z": z_ids, "y": y_ids, "x": x_ids},
            }

        max_bins = max(self.offset_num_bins)
        offset_logits = self.offset_head(hidden).reshape(hidden.shape[0], hidden.shape[1], 3, max_bins)
        stop_logits = self.stop_head(hidden).squeeze(-1)
        pred_offset_bins = []
        for axis, bins in enumerate(self.offset_num_bins):
            pred_offset_bins.append(offset_logits[:, :, axis, :bins].argmax(dim=-1))
        pred_offset_bins_tensor = torch.stack(pred_offset_bins, dim=-1)
        pred_refine_residual = self.position_refine_head(hidden)
        return {
            "coarse_logits": coarse_outputs["coarse_logits"],
            "coarse_axis_logits": coarse_outputs["coarse_axis_logits"],
            "offset_logits": offset_logits,
            "stop_logits": stop_logits,
            "pred_coarse_ids": coarse_outputs["pred_coarse_ids"],
            "pred_coarse_axis_ids": coarse_outputs["pred_coarse_axis_ids"],
            "pred_offset_bins": pred_offset_bins_tensor,
            "pred_refine_residual": pred_refine_residual,
            "coarse_prediction_mode": self.coarse_prediction_mode,
        }

    @torch.inference_mode()
    def init_kv_cache(
        self,
        *,
        prompt_tokens: dict[str, Tensor],
        prompt_anchor_xyz: Tensor,
        prompt_anchor_valid: Tensor,
        target_start_position: int | Tensor,
        memory_tokens: Tensor,
        memory_patch_centers: Tensor,
    ) -> tuple[dict[str, Any], FiberKVCache]:
        """Prime the KV cache with ``prompt + START`` and return the head outputs
        for the START position (i.e. the model's prediction for the first generated
        point).

        After this call, ``cache.seq_len == prompt_len + 1``, and a single call to
        :meth:`step_from_encoded_cached` advances by one step.

        Equivalent (up to floating-point ordering) to a call to
        :meth:`forward_from_encoded` with a teacher-forced generation buffer of
        length 1.
        """

        device = memory_tokens.device
        batch_size = int(memory_tokens.shape[0])

        # Embed the prompt by itself so prompt-internal tangents match training.
        prompt_token_type = torch.full_like(prompt_tokens["coarse_ids"], PROMPT_TOKEN_TYPE)
        prompt_embeddings, prompt_coords = self._build_input_embeddings(
            coarse_ids=prompt_tokens["coarse_ids"],
            offset_bins=prompt_tokens["offset_bins"],
            xyz=prompt_tokens["xyz"],
            positions=prompt_tokens["positions"],
            token_type=prompt_token_type,
            sequence_mask=prompt_tokens["mask"],
            geometry_valid_mask=prompt_tokens["valid_mask"],
            memory_tokens=memory_tokens,
        )

        # Build a single START token that consumes the prompt anchor xyz.
        start_coarse_ids = torch.full((batch_size, 1), IGNORE_INDEX, dtype=torch.long, device=device)
        start_offset_bins = torch.full((batch_size, 1, 3), IGNORE_INDEX, dtype=torch.long, device=device)
        anchor = prompt_anchor_xyz
        if anchor.ndim == 2:
            start_xyz = anchor.unsqueeze(1)
        elif anchor.ndim == 3:
            start_xyz = anchor
        else:
            raise ValueError(f"prompt_anchor_xyz must be (B,3) or (B,1,3); got {tuple(anchor.shape)}")
        if isinstance(target_start_position, int):
            start_positions = torch.full((batch_size, 1), int(target_start_position), dtype=torch.long, device=device)
        else:
            start_positions = target_start_position.to(device=device, dtype=torch.long).view(batch_size, 1)
        start_token_type = torch.full((batch_size, 1), START_TOKEN_TYPE, dtype=torch.long, device=device)
        start_seq_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=device)
        if prompt_anchor_valid.ndim == 1:
            start_geom_valid = prompt_anchor_valid.view(batch_size, 1)
        else:
            start_geom_valid = prompt_anchor_valid.view(batch_size, 1)
        start_embeddings, start_coords = self._build_input_embeddings(
            coarse_ids=start_coarse_ids,
            offset_bins=start_offset_bins,
            xyz=start_xyz,
            positions=start_positions,
            token_type=start_token_type,
            sequence_mask=start_seq_mask,
            geometry_valid_mask=start_geom_valid,
            memory_tokens=memory_tokens,
        )

        seq_embeddings = torch.cat([prompt_embeddings, start_embeddings], dim=1)
        seq_coords = torch.cat([prompt_coords, start_coords], dim=1)
        seq_mask = torch.cat([prompt_tokens["mask"], start_seq_mask], dim=1)
        attn_mask = self._build_attention_mask(seq_mask, dtype=seq_embeddings.dtype)

        self_attn_kv: list[tuple[Tensor, Tensor]] = []
        cross_attn_kv: list[tuple[Tensor, Tensor] | None] = []
        x = seq_embeddings
        for block in self.blocks:
            batch_size_b, seq_len_b, dim_b = x.shape
            qkv = block.self_attn.qkv(block.norm1(x)).reshape(
                batch_size_b, seq_len_b, 3, block.self_attn.num_heads, block.self_attn.head_dim
            )
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            sin, cos = _batched_rope_from_coords(block.self_attn.rope, seq_coords)
            q = apply_rotary_embedding(q, (sin, cos)).type_as(v)
            k = apply_rotary_embedding(k, (sin, cos)).type_as(v)
            self_attn_kv.append((k, v))
            q = block.self_attn._scale_queries(q)
            sa_out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
            )
            sa_out = sa_out.transpose(1, 2).reshape(batch_size_b, seq_len_b, dim_b)
            x = x + block.self_attn.proj(sa_out)
            if block.cross_attn is not None:
                ca_k, ca_v = block.cross_attn.precompute_kv(memory_tokens, memory_coords=memory_patch_centers)
                cross_attn_kv.append((ca_k, ca_v))
                x = x + block.cross_attn.forward_cached(
                    block.norm2(x), ca_k, ca_v, query_coords=seq_coords
                )
            else:
                cross_attn_kv.append(None)
            x = x + block.ffn(block.norm3(x))
            x = x * seq_mask.unsqueeze(-1).to(dtype=x.dtype)
        x = self.final_norm(x)

        hidden_last = x[:, -1:, :]
        pointer_key_norm, axis_pointer_keys = self._precompute_pointer_key_cache(memory_tokens)
        outputs = self._compute_outputs_from_hidden(
            hidden_last,
            memory_tokens,
            pointer_key_norm=pointer_key_norm,
            axis_pointer_keys=axis_pointer_keys,
        )

        cache = FiberKVCache(
            self_attn_kv=self_attn_kv,
            cross_attn_kv=cross_attn_kv,
            seq_len=int(seq_mask.shape[1]),
            last_input_xyz=start_xyz.squeeze(1),
            last_input_valid=start_geom_valid.squeeze(1),
            pointer_key_norm=pointer_key_norm,
            axis_pointer_keys=axis_pointer_keys,
        )
        return outputs, cache

    @torch.inference_mode()
    def step_from_encoded_cached(
        self,
        *,
        next_coarse_ids: Tensor,
        next_offset_bins: Tensor,
        next_xyz: Tensor,
        next_position: Tensor,
        cache: FiberKVCache,
        memory_tokens: Tensor,
    ) -> tuple[dict[str, Any], FiberKVCache]:
        """Advance the cache by exactly one generated token.

        ``next_coarse_ids``, ``next_offset_bins``, ``next_xyz`` describe the *input*
        for the new generation step (typically the previous step's prediction).
        Shapes: ``(B, 1)``, ``(B, 1, 3)``, ``(B, 1, 3)``. ``next_position`` is
        ``(B, 1)`` (long).

        The new token's tangent-conditioning input is computed from
        ``cache.last_input_xyz`` and the new ``next_xyz`` so the embedding is
        bit-equivalent to what :meth:`forward_from_encoded` would produce on the
        full generation buffer.
        """

        device = memory_tokens.device
        batch_size = int(memory_tokens.shape[0])
        if next_coarse_ids.shape != (batch_size, 1):
            raise ValueError(f"next_coarse_ids must be (B,1); got {tuple(next_coarse_ids.shape)}")
        if next_offset_bins.shape != (batch_size, 1, 3):
            raise ValueError(f"next_offset_bins must be (B,1,3); got {tuple(next_offset_bins.shape)}")
        if next_xyz.shape != (batch_size, 1, 3):
            raise ValueError(f"next_xyz must be (B,1,3); got {tuple(next_xyz.shape)}")

        # Build a (B, 2, *) input where index 0 is the *previous* input and index 1
        # is the new one. We only keep the new token's embedding; the previous-token
        # slot exists solely so that ``_build_input_embeddings`` computes the tangent
        # exactly as the non-cached forward path would.
        prev_coarse = torch.full((batch_size, 1), IGNORE_INDEX, dtype=torch.long, device=device)
        prev_offset = torch.full((batch_size, 1, 3), IGNORE_INDEX, dtype=torch.long, device=device)
        prev_xyz = cache.last_input_xyz.view(batch_size, 1, 3).to(device=device, dtype=next_xyz.dtype)
        prev_valid = cache.last_input_valid.view(batch_size, 1).to(device=device, dtype=torch.bool)
        # The previous position number is only used by the position embedding for the
        # previous slot, whose final embedding we discard, so any in-range value works.
        prev_position = (next_position.to(device=device, dtype=torch.long) - 1).clamp(min=0)
        prev_token_type = torch.full((batch_size, 1), GENERATED_TOKEN_TYPE, dtype=torch.long, device=device)
        new_token_type = torch.full((batch_size, 1), GENERATED_TOKEN_TYPE, dtype=torch.long, device=device)

        combined_coarse = torch.cat([prev_coarse, next_coarse_ids.to(device=device, dtype=torch.long)], dim=1)
        combined_offset = torch.cat([prev_offset, next_offset_bins.to(device=device, dtype=torch.long)], dim=1)
        combined_xyz = torch.cat([prev_xyz, next_xyz.to(device=device, dtype=prev_xyz.dtype)], dim=1)
        combined_positions = torch.cat([prev_position, next_position.to(device=device, dtype=torch.long).view(batch_size, 1)], dim=1)
        combined_token_type = torch.cat([prev_token_type, new_token_type], dim=1)
        combined_seq_mask = torch.ones((batch_size, 2), dtype=torch.bool, device=device)
        new_valid = torch.ones((batch_size, 1), dtype=torch.bool, device=device)
        combined_geom_valid = torch.cat([prev_valid, new_valid], dim=1)

        combined_embeddings, combined_coords = self._build_input_embeddings(
            coarse_ids=combined_coarse,
            offset_bins=combined_offset,
            xyz=combined_xyz,
            positions=combined_positions,
            token_type=combined_token_type,
            sequence_mask=combined_seq_mask,
            geometry_valid_mask=combined_geom_valid,
            memory_tokens=memory_tokens,
        )
        new_embedding = combined_embeddings[:, 1:2, :]
        new_coords = combined_coords[:, 1:2, :]

        new_self_attn_kv: list[tuple[Tensor, Tensor]] = []
        x = new_embedding
        for layer_idx, block in enumerate(self.blocks):
            x, (full_k, full_v) = block.forward_cached(
                x,
                new_coords,
                cache.self_attn_kv[layer_idx],
                cache.cross_attn_kv[layer_idx],
            )
            new_self_attn_kv.append((full_k, full_v))
        x = self.final_norm(x)

        outputs = self._compute_outputs_from_hidden(
            x,
            memory_tokens,
            pointer_key_norm=cache.pointer_key_norm,
            axis_pointer_keys=cache.axis_pointer_keys,
        )
        new_cache = FiberKVCache(
            self_attn_kv=new_self_attn_kv,
            cross_attn_kv=cache.cross_attn_kv,
            seq_len=cache.seq_len + 1,
            last_input_xyz=combined_xyz[:, 1, :].to(dtype=cache.last_input_xyz.dtype),
            last_input_valid=new_valid.squeeze(1),
            pointer_key_norm=cache.pointer_key_norm,
            axis_pointer_keys=cache.axis_pointer_keys,
        )
        return outputs, new_cache


def build_pseudo_inference_batch(
    *,
    prompt_tokens: dict[str, Tensor],
    prompt_anchor_xyz: Tensor,
    target_coarse_ids: Tensor,
    target_offset_bins: Tensor,
    target_xyz: Tensor,
    target_positions: Tensor,
    target_valid_mask: Tensor,
) -> dict[str, Tensor]:
    batch_size, target_len = target_coarse_ids.shape
    target_mask = torch.ones((batch_size, target_len), device=target_coarse_ids.device, dtype=torch.bool)
    return {
        "prompt_tokens": prompt_tokens,
        "prompt_anchor_xyz": prompt_anchor_xyz,
        "prompt_anchor_valid": torch.ones((batch_size,), device=prompt_anchor_xyz.device, dtype=torch.bool),
        "target_coarse_ids": target_coarse_ids,
        "target_offset_bins": target_offset_bins,
        "target_xyz": target_xyz,
        "target_positions": target_positions,
        "target_mask": target_mask,
        "target_valid_mask": target_valid_mask,
        "target_supervision_mask": target_mask & target_valid_mask,
        "target_stop": torch.zeros((batch_size, target_len), device=target_coarse_ids.device, dtype=torch.float32),
        "target_lengths": torch.full((batch_size,), target_len, device=target_coarse_ids.device, dtype=torch.long),
    }


def count_trainable_parameters(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


__all__ = [
    "ATTENTION_SCALING_STANDARD",
    "GENERATED_TOKEN_TYPE",
    "PROMPT_TOKEN_TYPE",
    "START_TOKEN_TYPE",
    "AutoregFiberModel",
    "FiberKVCache",
    "build_pseudo_inference_batch",
    "count_trainable_parameters",
]
