"""DINO-guided pseudo-label generator for 3D ink detection.

For each 256x256x256 chunk in a training batch, produce a label tensor as

    label = binarize( sigmoid(UNet(chunk)) * cos_sim_to_ref(chunk) )

where:
- UNet is a frozen snapshot of an in-progress ink-detection model (EMA weights).
- cos_sim_to_ref is computed from the dinovol-2 backbone over a sliding window
  of 128x128x128 patches with configurable stride+blending. Each window emits
  16x16x16 patch tokens (patch_size=8). Cosine similarity to a precomputed
  reference embedding is accumulated into a 32x32x32 grid spanning the chunk,
  then trilinearly upsampled to 256x256x256 and renormalized to [0, 1].

The class is designed to be instantiated once per DDP rank and called from the
trainer step (NOT from dataset workers). It runs in inference_mode and uses
bf16 by default to match the trainer's mixed_precision setting.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from koine_machines.inference.infer import (
    extract_state_dict_entry_from_payload,
    load_checkpoint_payload,
)
from koine_machines.models.make_model import make_model
from vesuvius.models.build.pretrained_backbones.dinovol_2_builder import (
    build_dinovol_2_backbone,
)


_CHUNK_SIZE = 256
_DINO_WINDOW = 128
_DINO_PATCH_SIZE = 8
_TOKENS_PER_WINDOW_AXIS = _DINO_WINDOW // _DINO_PATCH_SIZE  # 16
_TOKENS_PER_CHUNK_AXIS = _CHUNK_SIZE // _DINO_PATCH_SIZE    # 32
_DINO_EMBED_DIM = 864


def _build_unet(unet_ckpt_path: Path | str, device: torch.device, dtype: torch.dtype) -> torch.nn.Module:
    payload = load_checkpoint_payload(unet_ckpt_path)
    cfg = payload["config"]
    cfg.setdefault("model_config", {})
    cfg["in_channels"] = 1  # full_3d mode (override is normally done in train.py:217)
    model = make_model(cfg)
    _, sd = extract_state_dict_entry_from_payload(payload, unet_ckpt_path, prefer_ema=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected keys in UNet checkpoint: {unexpected[:5]}")
    if missing:
        # Allow only running-mean type keys to be missing; bail on anything else.
        bad = [k for k in missing if "running" not in k and "num_batches" not in k]
        if bad:
            raise RuntimeError(f"missing UNet weights: {bad[:5]}")
    model.eval().to(device=device, dtype=dtype)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _build_dino(dino_ckpt_path: Path | str, device: torch.device, dtype: torch.dtype) -> torch.nn.Module:
    ckpt = torch.load(str(dino_ckpt_path), map_location="cpu", weights_only=False)
    model_cfg = ckpt["config"]["model"]
    backbone = build_dinovol_2_backbone(model_cfg)
    sd = {
        k.replace("backbone.", "", 1): v
        for k, v in ckpt["teacher"].items()
        if k.startswith("backbone.")
    }
    backbone.load_pretrained_weights(sd)
    backbone.eval().to(device=device, dtype=dtype)
    for p in backbone.parameters():
        p.requires_grad_(False)
    return backbone


def _load_ref_embedding(path: Path | str, device: torch.device) -> torch.Tensor:
    arr = np.load(str(path)).astype(np.float32)
    if arr.ndim != 1 or arr.shape[0] != _DINO_EMBED_DIM:
        raise ValueError(
            f"reference embedding must be ({_DINO_EMBED_DIM},); got {arr.shape}"
        )
    t = torch.from_numpy(arr).to(device=device)
    t = t / t.norm().clamp_min(1e-12)
    return t


def _gaussian_window_3d(size: int, sigma: float) -> torch.Tensor:
    """Separable Gaussian centered on the cube; returns size^3 weights in [0,1]."""
    centers = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
    w1d = torch.exp(-(centers ** 2) / (2.0 * sigma ** 2))
    w1d = w1d / w1d.max()
    w3d = w1d[:, None, None] * w1d[None, :, None] * w1d[None, None, :]
    return w3d


def _window_starts(stride: int) -> list[int]:
    """Window start positions on one axis, snapped to include first and last fits."""
    last = _CHUNK_SIZE - _DINO_WINDOW
    starts: list[int] = list(range(0, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)
    return starts


class DinoGuidedLabelGenerator:
    """Frozen UNet + frozen DINO backbone -> binary ink pseudo-labels."""

    def __init__(
        self,
        *,
        unet_ckpt: str | Path,
        dino_ckpt: str | Path,
        ref_embedding: str | Path,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        dino_stride: int = 64,
        dino_minibatch: int = 8,
        dino_blend_sigma: float = 4.0,
        threshold: float = 0.5,
        # Accept and ignore unknown keys so callers can splat a JSON config.
        **_unused,
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.dino_stride = int(dino_stride)
        self.dino_minibatch = int(dino_minibatch)
        self.threshold = float(threshold)

        self.unet = _build_unet(Path(unet_ckpt), self.device, self.dtype)
        self.dino = _build_dino(Path(dino_ckpt), self.device, self.dtype)
        self.ref_emb = _load_ref_embedding(Path(ref_embedding), self.device).to(self.dtype)

        self._starts = _window_starts(self.dino_stride)
        # Precomputed Gaussian blend kernel over the 16-token axis of one window.
        self._weight = _gaussian_window_3d(_TOKENS_PER_WINDOW_AXIS, dino_blend_sigma).to(self.device)

    @torch.inference_mode()
    def generate(self, image_b1zyx: torch.Tensor) -> torch.Tensor:
        """image_b1zyx: [B, 1, 256, 256, 256] (any float dtype, on self.device).

        Returns float tensor [B, 1, 256, 256, 256] with values in {0, 1}.
        """
        if image_b1zyx.shape[-3:] != (_CHUNK_SIZE,) * 3:
            raise ValueError(
                f"expected chunk size {_CHUNK_SIZE}^3, got {tuple(image_b1zyx.shape)}"
            )
        img = image_b1zyx.to(device=self.device, dtype=self.dtype, non_blocking=True)

        unet_out = self.unet(img)
        logits = unet_out["ink"] if isinstance(unet_out, dict) else unet_out
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        unet_prob = torch.sigmoid(logits.float())          # [B, 1, 256, 256, 256]

        sim_full = self._dino_sim_map(img)                  # [B, 1, 256, 256, 256] in [0, 1]
        product = unet_prob * sim_full
        return (product > self.threshold).float()

    @torch.inference_mode()
    def _dino_sim_map(self, image_b1zyx: torch.Tensor) -> torch.Tensor:
        B = image_b1zyx.shape[0]
        coords: list[tuple[int, int, int]] = [
            (z, y, x) for z in self._starts for y in self._starts for x in self._starts
        ]
        Nw = len(coords)

        # Stack windows in (window-major, batch-minor) order.
        windows = torch.empty(
            (Nw * B, 1, _DINO_WINDOW, _DINO_WINDOW, _DINO_WINDOW),
            device=self.device, dtype=self.dtype,
        )
        for wi, (z0, y0, x0) in enumerate(coords):
            windows[wi * B:(wi + 1) * B] = image_b1zyx[
                :, :, z0:z0 + _DINO_WINDOW, y0:y0 + _DINO_WINDOW, x0:x0 + _DINO_WINDOW
            ]

        # DINO forward in mini-batches; collect cosine similarities reshaped to 16^3.
        sim_blocks: list[torch.Tensor] = []
        for i in range(0, windows.shape[0], self.dino_minibatch):
            sub = windows[i:i + self.dino_minibatch]
            tokens = self.dino.forward_features(sub)["x_norm_patchtokens"]  # [n, 4096, 864]
            tokens = F.normalize(tokens.float(), dim=-1)
            sim = tokens @ self.ref_emb.float()                              # [n, 4096]
            sim_blocks.append(sim.view(sim.shape[0],
                                       _TOKENS_PER_WINDOW_AXIS,
                                       _TOKENS_PER_WINDOW_AXIS,
                                       _TOKENS_PER_WINDOW_AXIS))
        sims = torch.cat(sim_blocks, dim=0)                                  # [Nw*B, 16, 16, 16]

        acc = torch.zeros(B, _TOKENS_PER_CHUNK_AXIS, _TOKENS_PER_CHUNK_AXIS, _TOKENS_PER_CHUNK_AXIS,
                          device=self.device, dtype=torch.float32)
        wacc = torch.zeros_like(acc)
        weight = self._weight                                                 # [16, 16, 16]
        for wi, (z0, y0, x0) in enumerate(coords):
            sub = sims[wi * B:(wi + 1) * B]                                   # [B, 16, 16, 16]
            oz, oy, ox = z0 // _DINO_PATCH_SIZE, y0 // _DINO_PATCH_SIZE, x0 // _DINO_PATCH_SIZE
            acc[:, oz:oz + _TOKENS_PER_WINDOW_AXIS,
                   oy:oy + _TOKENS_PER_WINDOW_AXIS,
                   ox:ox + _TOKENS_PER_WINDOW_AXIS] += sub * weight
            wacc[:, oz:oz + _TOKENS_PER_WINDOW_AXIS,
                    oy:oy + _TOKENS_PER_WINDOW_AXIS,
                    ox:ox + _TOKENS_PER_WINDOW_AXIS] += weight

        sim_grid = acc / wacc.clamp_min(1e-6)                                 # [B, 32, 32, 32] in [-1, 1]
        sim_full = F.interpolate(
            sim_grid.unsqueeze(1),
            size=(_CHUNK_SIZE, _CHUNK_SIZE, _CHUNK_SIZE),
            mode="trilinear",
            align_corners=False,
        )                                                                     # [B, 1, 256, 256, 256]
        return ((sim_full + 1.0) / 2.0).clamp(0.0, 1.0)
