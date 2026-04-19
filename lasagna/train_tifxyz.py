"""Train 3D UNet to predict lasagna channels from CT, using tifxyz surfaces as ground truth.

Derives the 8 training channels (cos, grad_mag, 6x direction encoding) on-the-fly
on GPU from tifxyz surface voxelizations. No pre-computed label zarrs needed.

Usage:
  python lasagna/train_tifxyz.py \
      --config lasagna/configs/tifxyz_train.json \
      --patch-size 128 --batch-size 2 --epochs 100

Config JSON format:
  {
    "datasets": [
      {
        "volume_path": "/path/to/volume.zarr",
        "volume_scale": 0,
        "segments_path": "/path/to/tifxyz/",
        "z_range": [2000, 7000]
      },
      ...
    ]
  }
"""
from __future__ import annotations

# Must be set before `import torch` (before any CUDA alloc). Expandable
# segments let a reserved block grow contiguously instead of fragmenting
# into per-size pools — without this the vis inference forward OOMs
# reliably after training forwards on the same GPU even with several
# GB free in the reserved pool.
import os as _os
_os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True",
)
# Limit per-process thread pools to avoid 2000%+ CPU per dataloader worker.
# Must be set before importing numpy, blosc, torch, etc.
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "BLOSC_NTHREADS", "NUMEXPR_MAX_THREADS", "NUMBA_NUM_THREADS"):
    _os.environ.setdefault(_k, "4")
del _k, _os

# Pre-warm numba LLVM runtime before any other imports. torch pulls in
# opt_einsum, whose backends/__init__.py eagerly imports
# opt_einsum.backends.tensorflow, which loads tensorflow's LLVM and poisons
# llvmlite's runtime. A later @njit compile then aborts with
# "LLVM ERROR: Symbol not found: NRT_MemInfo_call_dtor". Forcing numba to
# initialize first keeps its runtime symbols registered.
import numba as _numba  # noqa: E402
import numpy as _np  # noqa: E402

@_numba.njit
def _numba_warmup():
    return _np.zeros(1, dtype=_np.int32)

_numba_warmup()
del _numba, _np, _numba_warmup

import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from types import SimpleNamespace


def _init_distributed():
    """Read torchrun env vars and bring up the process group.

    Returns (rank, local_rank, world_size, is_dist).
    Single-GPU runs (no torchrun) → (0, 0, 1, False) and no init.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_dist = world_size > 1
    if is_dist and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size, is_dist


def _allreduce_max(value: int, world_size: int, device) -> int:
    if world_size <= 1:
        return value
    t = torch.tensor([int(value)], device=device, dtype=torch.long)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return int(t.item())


def _allreduce_mean(value: float, world_size: int, device) -> float:
    if world_size <= 1:
        return float(value)
    t = torch.tensor([float(value)], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item() / world_size)

# Ensure lasagna/ dir is on sys.path for sibling imports
_LASAGNA_DIR = os.path.dirname(os.path.abspath(__file__))
if _LASAGNA_DIR not in sys.path:
    sys.path.insert(0, _LASAGNA_DIR)

from vesuvius.models.build.build_network_from_config import NetworkFromConfig

from tifxyz_labels import (
    compute_patch_labels,
    encode_direction_channels,
    scale_space_pool_validity,
    decode_to_tensor,
    tensor_unsigned_angle_deg,
)
from gpu_pause import GpuPauseServer
from tifxyz_lasagna_dataset import (
    TifxyzLasagnaDataset,
    augment_batch_inplace,
    collate_variable_surfaces,
)
from lasagna3d.dataset_vis import (
    build_inference_context,
    render_batch_figure,
    default_vis_title,
    _sample_from_batch,
)
from PIL import Image


TAG = "[train_tifxyz]"


def _filter_batch(batch: dict, keep_list: list) -> dict:
    """Return a shallow copy of collated batch dict with only the given sample indices."""
    B_orig = len(batch["patch_info"])
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor) and v.ndim >= 1 and v.shape[0] == B_orig:
            keep_t = torch.tensor(keep_list, device=v.device, dtype=torch.long)
            out[k] = v.index_select(0, keep_t)
        elif isinstance(v, list) and len(v) == B_orig:
            out[k] = [v[i] for i in keep_list]
        else:
            out[k] = v
    return out

_CHANNEL_NAMES = [
    "cos", "grad_mag",
    "dir0_z", "dir1_z", "dir0_y", "dir1_y", "dir0_x", "dir1_x",
]


# ---------------------------------------------------------------------------
# GPU target computation from dataset outputs
# ---------------------------------------------------------------------------

def compute_batch_targets(
    batch: dict,
    device: torch.device,
    same_surface_threshold: float | None = None,
    same_surface_groups_batch: list[list[list[int]] | None] | None = None,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[list[int]],
    list[list[torch.Tensor]],
    list[list[dict]],
]:
    """Compute training targets on GPU from dataset batch.

    Runs EDT + chain ordering + cos/grad_mag derivation + raw-normal
    slerp densification + last-second double-angle encoding on each
    sample. This function is the single place where training-side
    same-surface detection happens: EDTs are built here, handed to
    :func:`tifxyz_labels.detect_same_surface_groups` when a threshold
    is provided, and the resulting groups + the already-computed EDTs
    are passed down to :func:`compute_patch_labels` so nothing is
    recomputed. ``compute_patch_labels`` itself never detects.

    Args:
        batch: dict from collate_variable_surfaces
        device: CUDA device
        same_surface_threshold: optional voxel-median distance threshold
            for the same-surface detector. ``None`` disables detection
            for the whole batch (unless overridden by an explicit
            ``same_surface_groups_batch`` entry).
        same_surface_groups_batch: optional per-sample explicit groups
            (``list[list[int]]`` per sample, or ``None`` to fall back
            to threshold-based detection for that sample). Used by
            `dataset overlap --vis-dir` to render the exact pairs its
            analysis flagged.

    Returns:
        targets: (B, 8, Z, Y, X) float32
        validity: (B, 1, Z, Y, X) float32 — where cos/grad_mag are valid
        dir_sparse_mask: (B, 1, Z, Y, X) float32 — voxels with original
            splatted raw normals (hard supervision).
        dir_dense_mask:  (B, 1, Z, Y, X) float32 — voxels filled by the
            slerp blend inside the validity bracket.
        dir_axis_weight: (B, 6, Z, Y, X) float32 — per-plane relevance
            weight derived from the densified raw normal's in-plane
            magnitude.
        merge_groups_batch: per-sample ``merge_groups`` lists mapping
            each original surface slot → merged slot index (identity
            when the merge is disabled).
        merged_masks_batch: per-sample list of merged surface masks
            (CUDA bool tensors) — exactly the tensors the loss saw.
            Training doesn't consume this; it's exposed so the vis
            can render the post-merge state without duplicating the
            merge logic.
        merged_chain_info_batch: per-sample merged chain_info lists,
            one entry per merged group.
    """
    B = batch["image"].shape[0]
    surface_masks_list = batch["surface_masks"]  # list of (Ni, Z, Y, X)
    surface_chain_info_batch = batch["surface_chain_info"]  # list[list[dict]]
    tensor_moments_batch = batch["tensor_moments"].to(device)  # (B, 6, Z, Y, X)
    normals_valid_batch = batch["normals_valid"].to(device)  # (B, 1, Z, Y, X)

    spatial_shape = batch["image"].shape[2:]  # (Z, Y, X)
    all_targets = []
    all_validity = []
    all_sparse_masks = []
    all_dense_masks = []
    all_axis_weight = []
    merge_groups_batch: list[list[int]] = []
    merged_masks_batch: list[list[torch.Tensor]] = []
    merged_chain_info_batch: list[list[dict]] = []

    from tifxyz_labels import (
        edt_torch, edt_torch_with_indices, detect_same_surface_groups,
    )

    for b in range(B):
        # Convert surface masks to CUDA bool tensors
        masks_b = surface_masks_list[b].to(device)  # (N, Z, Y, X)
        N = masks_b.shape[0]
        cuda_masks = [masks_b[i] > 0.5 for i in range(N)]

        # Tensor moments + splat validity for this sample
        tm = tensor_moments_batch[b]  # (6, Z, Y, X)
        nv = normals_valid_batch[b, 0] > 0.5  # (Z, Y, X)

        chain_info_b = surface_chain_info_batch[b]

        # --- Same-surface merge resolution ---
        explicit_groups = None
        if same_surface_groups_batch is not None and b < len(same_surface_groups_batch):
            explicit_groups = same_surface_groups_batch[b]

        dts: list[torch.Tensor] | None = None
        fts: list[torch.Tensor] | None = None
        groups: list[list[int]] | None = None
        if explicit_groups is not None:
            groups = explicit_groups
        elif same_surface_threshold is not None and N >= 2:
            # Detect here so compute_patch_labels doesn't duplicate
            # EDT work.
            # - ``dts`` are from ``~surface_mask`` — used for cos
            #   routing and same-surface detection (distance fields
            #   unchanged from before).
            # - ``fts`` are from ``~(surface_mask & normals_valid)`` —
            #   used only for the tensor gather, so the feature
            #   transform always lands on a voxel that actually
            #   carries splatted tensor moments.
            dts = []
            fts = []
            for m in cuda_masks:
                dts.append(edt_torch((~m).to(torch.uint8)))
                ft_src = m & nv
                _, ft = edt_torch_with_indices((~ft_src).to(torch.uint8))
                fts.append(ft)
            groups = detect_same_surface_groups(
                dts, cuda_masks, chain_info_b,
                threshold=float(same_surface_threshold),
            )

        # Compute labels on GPU using the patch's externally-built chains.
        result = compute_patch_labels(
            surface_masks=cuda_masks,
            tensor_moments=tm,
            normals_valid=nv,
            surface_chain_info=chain_info_b,
            device=device,
            same_surface_groups=groups,
            precomputed_dts=dts,
            precomputed_fts=fts,
        )

        all_targets.append(result["targets"])          # (8, Z, Y, X)
        all_validity.append(result["validity"])        # (Z, Y, X)
        all_sparse_masks.append(result["dir_sparse_mask"])  # (Z, Y, X) bool
        all_dense_masks.append(result["dir_dense_mask"])    # (Z, Y, X) bool
        all_axis_weight.append(result["dir_axis_weight"])   # (6, Z, Y, X)
        merge_groups_batch.append(
            list(result.get("merge_groups", list(range(N))))
        )
        merged_masks_batch.append(
            list(result.get("merged_surface_masks", cuda_masks))
        )
        merged_chain_info_batch.append(
            list(result.get("merged_chain_info", surface_chain_info_batch[b]))
        )

    targets = torch.stack(all_targets, dim=0)       # (B, 8, Z, Y, X)
    validity = torch.stack(all_validity, dim=0).unsqueeze(1).float()  # (B, 1, Z, Y, X)
    dir_sparse_mask = torch.stack(all_sparse_masks, dim=0).unsqueeze(1).float()
    dir_dense_mask = torch.stack(all_dense_masks, dim=0).unsqueeze(1).float()
    dir_axis_weight = torch.stack(all_axis_weight, dim=0)  # (B, 6, Z, Y, X)

    return (
        targets, validity,
        dir_sparse_mask, dir_dense_mask, dir_axis_weight,
        merge_groups_batch, merged_masks_batch, merged_chain_info_batch,
    )


# ---------------------------------------------------------------------------
# CT intensity augmentation (spatial augmentation lives in the dataset
# via `augment_batch_inplace`, which re-splats raw normals from the
# flipped/rotated surface_geometry; the 6-channel double-angle encoding
# is derived last-second inside `compute_patch_labels`).
# ---------------------------------------------------------------------------

def augment_intensity(image: torch.Tensor) -> torch.Tensor:
    """Random CT intensity augmentation (applied to image only).

    Mirrors train_unet_3d.augment_intensity: brightness delta,
    contrast factor, gamma, noise, and a final clamp to [0, 1].
    """
    if torch.rand(1).item() < 0.5:
        delta = (torch.rand(1, device=image.device).item() - 0.5) * 0.3
        image = image + delta
    if torch.rand(1).item() < 0.5:
        factor = 0.7 + torch.rand(1, device=image.device).item() * 0.6
        mean = image.mean()
        image = (image - mean) * factor + mean
    if torch.rand(1).item() < 0.5:
        gamma = 0.7 + torch.rand(1, device=image.device).item() * 1.3
        image = image.clamp(0, 1).pow(gamma)
    if torch.rand(1).item() < 0.3:
        noise = torch.randn_like(image) * 0.03
        image = image + noise
    return image.clamp(0, 1)


# ---------------------------------------------------------------------------
# Loss functions (reused from train_unet_3d.py)
# ---------------------------------------------------------------------------

def direction_smoothness_loss(
    pred_dir: torch.Tensor, mask: torch.Tensor,
) -> torch.Tensor:
    """L1 smoothness regularizer on the 6 direction channels.

    Penalizes absolute finite-difference gradients of ``pred_dir`` along
    each spatial axis, averaged over all positions where BOTH endpoints
    of the finite difference are supervised (``mask`` is 1 at both).
    This lets the model produce a smooth direction field inside the
    validity bracket without fighting the out-of-bracket garbage.

    Args:
        pred_dir: ``(B, 6, Z, Y, X)`` — predicted direction channels.
        mask:     ``(B, 1, Z, Y, X)`` or ``(B, Z, Y, X)`` — direction
                  validity (1 in bracket, 0 outside).
    """
    if mask.ndim == pred_dir.ndim - 1:
        mask = mask.unsqueeze(1)
    mask = (mask > 0.5).float()

    gz = (pred_dir[:, :, 1:, :, :] - pred_dir[:, :, :-1, :, :]).abs()
    gy = (pred_dir[:, :, :, 1:, :] - pred_dir[:, :, :, :-1, :]).abs()
    gx = (pred_dir[:, :, :, :, 1:] - pred_dir[:, :, :, :, :-1]).abs()

    mz = mask[:, :, 1:, :, :] * mask[:, :, :-1, :, :]
    my = mask[:, :, :, 1:, :] * mask[:, :, :, :-1, :]
    mx = mask[:, :, :, :, 1:] * mask[:, :, :, :, :-1]

    def _avg(grad: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        # Broadcasting m (B, 1, ...) over the 6 direction channels.
        num = (grad * m).sum()
        denom = (m.sum() * grad.shape[1]).clamp(min=1.0)
        return num / denom

    return (_avg(gz, mz) + _avg(gy, my) + _avg(gx, mx)) / 3.0


def sorted_distribution_loss(pred_cos, gt_cos, validity, grad_mag,
                             window=40, stride=30, grad_mag_min=0.05):
    """Phase-invariant distribution matching via sort-based 1D Wasserstein.

    Extracts overlapping 3D windows, keeps only windows with sufficient
    GT grad_mag density (surfaces present), sorts pred and GT cos values
    within each window, computes L1 between sorted vectors.

    Each window's contribution is weighted by its mean grad_mag (closer
    surfaces → higher weight), matching the per-voxel weighting used
    by the main cos loss.
    """
    B, C, Z, Y, X = pred_cos.shape
    losses = []
    weights = []

    for z0 in range(0, Z - window + 1, stride):
        for y0 in range(0, Y - window + 1, stride):
            for x0 in range(0, X - window + 1, stride):
                z1, y1, x1 = z0 + window, y0 + window, x0 + window

                v = validity[:, :, z0:z1, y0:y1, x0:x1]
                gm = grad_mag[:, :, z0:z1, y0:y1, x0:x1]

                v_sum = v.sum().clamp(min=1)
                gm_mean = (gm * v).sum() / v_sum
                if gm_mean < grad_mag_min:
                    continue

                mask = v[:, 0] > 0.5
                p = pred_cos[:, 0, z0:z1, y0:y1, x0:x1]
                g = gt_cos[:, 0, z0:z1, y0:y1, x0:x1]

                # Weight = mean grad_mag × 20 (same normalization as cos loss)
                w = gm_mean * 20.0

                for b in range(B):
                    m = mask[b]
                    if m.sum() < 10:
                        continue
                    p_sorted = torch.sort(p[b][m])[0]
                    g_sorted = torch.sort(g[b][m])[0]
                    losses.append(F.l1_loss(p_sorted, g_sorted))
                    weights.append(w)

    if not losses:
        return pred_cos.new_zeros(1).squeeze()
    losses_t = torch.stack(losses)
    weights_t = torch.stack(weights)
    return (losses_t * weights_t).sum() / weights_t.sum().clamp(min=1e-8)


def fft_magnitude_loss(pred_cos, gt_cos, validity, grad_mag,
                       block=16, stride=12, grad_mag_min=0.05):
    """Phase-invariant frequency loss via FFT magnitude matching.

    Computes 3D rFFT on overlapping blocks of pred and GT cos,
    compares magnitude spectra (L1). Phase is discarded entirely.
    DC component excluded (mean already supervised by cos MSE).
    Weighted by mean grad_mag per block (same as cos loss weighting).
    """
    B, C, Z, Y, X = pred_cos.shape
    losses = []
    weights = []

    for z0 in range(0, Z - block + 1, stride):
        for y0 in range(0, Y - block + 1, stride):
            for x0 in range(0, X - block + 1, stride):
                z1, y1, x1 = z0 + block, y0 + block, x0 + block

                v = validity[:, :, z0:z1, y0:y1, x0:x1]
                # Only fully-valid blocks
                if v.min() < 0.5:
                    continue

                gm = grad_mag[:, :, z0:z1, y0:y1, x0:x1]
                gm_mean = gm.mean()
                if gm_mean < grad_mag_min:
                    continue

                p = pred_cos[:, 0, z0:z1, y0:y1, x0:x1]
                g = gt_cos[:, 0, z0:z1, y0:y1, x0:x1]

                fft_p = torch.fft.rfftn(p.float(), dim=(-3, -2, -1))
                fft_g = torch.fft.rfftn(g.float(), dim=(-3, -2, -1))
                mag_p = fft_p.abs()
                mag_g = fft_g.abs()

                # Exclude DC (mean already supervised by cos MSE)
                mag_p[:, 0, 0, 0] = 0
                mag_g[:, 0, 0, 0] = 0

                loss = F.l1_loss(mag_p, mag_g)
                w = gm_mean * 20.0
                losses.append(loss)
                weights.append(w)

    if not losses:
        return pred_cos.new_zeros(1).squeeze()
    losses_t = torch.stack(losses)
    weights_t = torch.stack(weights)
    return (losses_t * weights_t).sum() / weights_t.sum().clamp(min=1e-8)


class MaskedMSE(nn.Module):
    def forward(self, pred, target, mask=None, weight=None):
        diff = (pred - target) ** 2
        if mask is None:
            return diff.mean()
        if mask.ndim == pred.ndim - 1:
            mask = mask.unsqueeze(1)
        effective = mask * weight if weight is not None else mask
        diff = diff * effective
        denom = effective.sum()
        return diff.sum() / denom.clamp(min=1.0)


class MaskedSmoothL1(nn.Module):
    def forward(self, pred, target, mask=None, weight=None):
        diff = F.smooth_l1_loss(pred, target, reduction="none")
        if mask is None:
            return diff.mean()
        if mask.ndim == pred.ndim - 1:
            mask = mask.unsqueeze(1)
        effective = mask * weight if weight is not None else mask
        diff = diff * effective
        denom = effective.sum()
        return diff.sum() / denom.clamp(min=1.0)


class ScaleSpaceLoss3D(nn.Module):
    """Multi-scale masked loss.

    Per-scale weights are a decaying geometric series with decay
    ``scale_decay`` (default 0.8) — scale 0 gets 1.0, scale 1 gets
    0.8, scale 2 gets 0.64, etc. — then re-normalized so the weights
    sum to exactly 1. This keeps the loss magnitude independent of
    ``num_scales`` and biases supervision toward the fine scales
    without dropping coarse structure entirely.

    For ``num_scales=5, scale_decay=0.8`` the normalized weights are
    approximately ``[0.355, 0.284, 0.227, 0.182, 0.145]``.
    """

    def __init__(
        self,
        base_loss: nn.Module,
        num_scales: int,
        scale_decay: float = 0.8,
    ) -> None:
        super().__init__()
        self.base_loss = base_loss
        self.num_scales = num_scales
        self.scale_decay = float(scale_decay)
        self.pool = nn.AvgPool3d(kernel_size=2, stride=2)

        raw = [self.scale_decay ** s for s in range(num_scales)]
        s = sum(raw)
        # Register as a plain buffer so .to(device) migrates it with
        # the module. Keeps autograd out of the weights entirely.
        self.register_buffer(
            "scale_weights",
            torch.tensor([w / s for w in raw], dtype=torch.float32),
        )

    def forward(self, pred, target, mask=None, weight=None):
        x, y = pred, target
        m = None
        if mask is not None:
            m = mask
            if m.ndim == 4:
                m = m.unsqueeze(1)
            m = (m > 0.5).float()
        w = weight

        total = torch.zeros((), device=pred.device, dtype=pred.dtype)
        for scale in range(self.num_scales):
            sw = float(self.scale_weights[scale].item())
            total = total + sw * self.base_loss(x, y, mask=m, weight=w)
            if scale < self.num_scales - 1:
                if x.size(2) < 2 or x.size(3) < 2 or x.size(4) < 2:
                    break
                if m is not None:
                    # Masked-average pooling: at each coarser scale,
                    # x and y are averaged only over the *valid* voxels
                    # of every 2x2x2 block. The validity mask itself
                    # follows the any-valid rule (max-pool), shared with
                    # the vis via scale_space_pool_validity. This way a
                    # single valid voxel in a block still produces a
                    # meaningful coarse target instead of being silently
                    # diluted by the seven invalid neighbours.
                    eps = 1e-6
                    m_count = F.avg_pool3d(m, kernel_size=2, stride=2)
                    denom = m_count.clamp_min(eps)
                    x = F.avg_pool3d(x * m, kernel_size=2, stride=2) / denom
                    y = F.avg_pool3d(y * m, kernel_size=2, stride=2) / denom
                    if w is not None:
                        w = F.avg_pool3d(w * m, kernel_size=2, stride=2) / denom
                    m = scale_space_pool_validity(m)
                else:
                    x = self.pool(x)
                    y = self.pool(y)
                    if w is not None:
                        w = self.pool(w)
        return total


# ---------------------------------------------------------------------------
# Model construction (same as train_unet_3d.py)
# ---------------------------------------------------------------------------

def _reinit_decoder_scales(model: nn.Module, n_scales: int) -> list[str]:
    """Re-initialize the last *n_scales* decoder stages (highest resolution).

    Resets: decoder conv blocks, upsample modules, seg heads / task heads,
    and the encoder stem (which feeds the highest-res skip).

    Supports both architectures:
      - shared_decoder + task_heads  (default when deep_supervision=False)
      - task_decoders["output"]      (separate decoder per task)
    """
    from vesuvius.models.utils import InitWeights_He
    init_fn = InitWeights_He(neg_slope=1e-2)

    # Find the decoder — shared or per-task
    if hasattr(model, "shared_decoder") and model.shared_decoder is not None:
        decoder = model.shared_decoder
        dec_prefix = "shared_decoder"
    elif hasattr(model, "task_decoders") and "output" in model.task_decoders:
        decoder = model.task_decoders["output"]
        dec_prefix = "task_decoders.output"
    else:
        raise RuntimeError("Cannot find decoder on model (no shared_decoder or task_decoders.output)")

    n_dec = len(decoder.stages)
    if n_scales > n_dec:
        n_scales = n_dec

    reinit_names: list[str] = []

    for i in range(n_dec - n_scales, n_dec):
        decoder.stages[i].apply(init_fn)
        reinit_names.append(f"{dec_prefix}.stages.{i}")
        decoder.transpconvs[i].apply(init_fn)
        reinit_names.append(f"{dec_prefix}.transpconvs.{i}")

    if hasattr(decoder, "seg_layers"):
        for i in range(len(decoder.seg_layers)):
            decoder.seg_layers[i].apply(init_fn)
            reinit_names.append(f"{dec_prefix}.seg_layers.{i}")

    if hasattr(model, "task_heads"):
        for name, head in model.task_heads.items():
            head.apply(init_fn)
            reinit_names.append(f"task_heads.{name}")

    if hasattr(model, "shared_encoder") and hasattr(model.shared_encoder, "stem"):
        model.shared_encoder.stem.apply(init_fn)
        reinit_names.append("shared_encoder.stem")

    return reinit_names


def build_model(
    patch_size: int,
    device: str,
    weights: Optional[str] = None,
    norm_type: Optional[str] = None,
    upsample_mode: Optional[str] = None,
    batch_size: int = 2,
    strict: bool = False,
    model_patch_size: Optional[int] = None,
    reinit_decoder_scales: int = 0,
    refine: bool = False,
) -> Tuple[nn.Module, str, str]:
    ckpt = None
    if weights is not None:
        ckpt = torch.load(weights, map_location=device, weights_only=False)
        if isinstance(ckpt, dict):
            if norm_type is None:
                norm_type = ckpt.get("norm_type", "instance")
            if upsample_mode is None:
                upsample_mode = ckpt.get("upsample_mode", "transpconv")
            if strict:
                ckpt_patch = ckpt.get("patch_size")
                if ckpt_patch is not None and int(ckpt_patch) != int(patch_size):
                    raise ValueError(
                        f"Strict load: patch_size mismatch — checkpoint "
                        f"was trained at {ckpt_patch}, caller asked for "
                        f"{patch_size}. The architecture autoconfigures "
                        f"on patch size; they must match."
                    )
    if norm_type is None:
        norm_type = "instance"
    if upsample_mode is None:
        upsample_mode = "transpconv"

    mgr = SimpleNamespace()
    model_config = {"autoconfigure": True, "architecture_type": "unet"}
    if norm_type == "group":
        model_config["norm_op"] = "nn.GroupNorm"
        model_config["norm_op_kwargs"] = {"num_groups": 32, "affine": True, "eps": 1e-5}
    elif norm_type == "none":
        model_config["norm_op"] = None
    model_config["upsample_mode"] = upsample_mode
    mgr.model_config = model_config
    mgr.targets = {"output": {"out_channels": 8, "activation": "none"}}
    # Model-architecture patch size: defaults to training patch size,
    # but can be overridden (e.g. to match a checkpoint trained at a
    # different size while training at a new patch size).
    arch_patch = int(model_patch_size) if model_patch_size else int(patch_size)
    mgr.train_patch_size = (arch_patch, arch_patch, arch_patch)
    if arch_patch != int(patch_size):
        print(
            f"{TAG} model_patch_size={arch_patch} overrides "
            f"training patch_size={patch_size} for architecture autoconfigure",
            flush=True,
        )
    mgr.train_batch_size = batch_size
    mgr.in_channels = 11 if refine else 1
    mgr.autoconfigure = True
    mgr.spacing = [1, 1, 1]
    mgr.model_name = "lasagna_tifxyz_3d"

    model = NetworkFromConfig(mgr).to(device)

    if ckpt is None:
        # NetworkFromConfig's conv Encoder/Decoder don't run an
        # explicit init pass, so without this the weights fall back
        # to PyTorch's default Conv3d.reset_parameters —
        # kaiming_uniform_(a=sqrt(5)) — which undershoots the target
        # variance for ReLU/LeakyReLU activations and slows early
        # convergence. Match nnUNet's canonical init
        # (kaiming_normal_ + neg_slope=1e-2, zero biases) so the
        # LR-warmup phase isn't spent correcting scale.
        from vesuvius.models.utils import InitWeights_He
        model.apply(InitWeights_He(neg_slope=1e-2))

    if ckpt is not None:
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
        if strict:
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                raise RuntimeError(
                    f"Strict checkpoint load failed for {weights}:\n"
                    f"  missing ({len(missing)}): "
                    f"{list(missing)[:5]}{'...' if len(missing) > 5 else ''}\n"
                    f"  unexpected ({len(unexpected)}): "
                    f"{list(unexpected)[:5]}{'...' if len(unexpected) > 5 else ''}\n"
                    f"This usually means the checkpoint was trained with a "
                    f"different patch_size / norm_type / upsample_mode than "
                    f"the caller asked for."
                )
            print(f"{TAG} loaded {len(state_dict)} params (strict) from {weights}")
        else:
            model_state = model.state_dict()
            filtered = {k: v for k, v in state_dict.items()
                        if k in model_state and model_state[k].shape == v.shape}
            missing = [k for k in model_state if k not in filtered]
            skipped = [k for k in state_dict if k not in filtered]
            model_state.update(filtered)
            model.load_state_dict(model_state)
            print(f"{TAG} loaded {len(filtered)}/{len(model_state)} params from {weights}")
            if skipped:
                print(f"{TAG} skipped from checkpoint: {skipped[:5]}{'...' if len(skipped) > 5 else ''}")
            if missing:
                print(f"{TAG} randomly initialized: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    # Expand stem conv from 1→11 input channels when loading a
    # non-refine checkpoint into a refine model.  Channel 0 keeps the
    # pretrained weight; channels 1–10 are zero-initialized so the
    # model starts by ignoring the refinement inputs.
    if refine and ckpt is not None:
        stem = getattr(getattr(model, "shared_encoder", None), "stem", None)
        if stem is not None:
            for m in stem.modules():
                if isinstance(m, (nn.Conv3d, nn.Conv2d)):
                    if m.weight.shape[1] == 11:
                        # Find the matching key in the loaded state dict
                        for k, v in (ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt).items():
                            if v.shape[0] == m.weight.shape[0] and len(v.shape) == len(m.weight.shape) and v.shape[1] == 1:
                                # Already loaded the 1ch weight into an 11ch param
                                # via strict=False — now the param has random init.
                                # Re-set: channel 0 = checkpoint, rest = 0
                                with torch.no_grad():
                                    new_w = torch.zeros_like(m.weight)
                                    new_w[:, 0:1] = v.to(new_w.device)
                                    m.weight.copy_(new_w)
                                print(f"{TAG} expanded stem conv {k}: "
                                      f"1→11 input channels (zeros for ch 1–10)")
                                break
                    break  # only the first conv in stem

    if reinit_decoder_scales > 0 and ckpt is not None:
        names = _reinit_decoder_scales(model, reinit_decoder_scales)
        print(f"{TAG} re-initialized {len(names)} modules "
              f"(last {reinit_decoder_scales} decoder scales): "
              f"{names}", flush=True)

    return model, norm_type, upsample_mode


# ---------------------------------------------------------------------------
# Training-adaptive GT deformation
# ---------------------------------------------------------------------------


class DeformationStore:
    """Per-sample low-res deformation fields, disk-backed via np.memmap.

    Each dataset config entry gets its own memmap file keyed by
    ``segments_path`` and ``dataset_idx`` for portability.  Deformations
    are stored as float16 to halve disk/memory use and converted to
    float32 only when loaded onto GPU.
    """

    def __init__(
        self,
        patches: list,   # list of ChunkPatch from dataset.patches
        config: dict,    # the parsed JSON config (has "datasets" key)
        grid_size: int,
        run_dir: Path,
    ):
        self.grid_size = grid_size
        self.deform_dir = run_dir / "deformations"
        self.deform_dir.mkdir(parents=True, exist_ok=True)

        # Group patches by (segments_path, dataset_idx) and count.
        from collections import OrderedDict
        groups: OrderedDict[str, int] = OrderedDict()
        # Map global patch idx -> (group_key, local_idx)
        self._index: list[tuple[str, int]] = []

        # We need the segments_path per patch.  It's on the config but
        # not on ChunkPatch directly.  Build from config datasets list.
        ds_key_map: dict[int, str] = {}   # dataset_idx -> segments_path
        for di, ds in enumerate(config["datasets"]):
            sp = ds.get("segments_path", "")
            if sp:
                ds_key_map[di] = sp

        for patch in patches:
            sp = ds_key_map.get(patch.dataset_idx, f"unknown_{patch.dataset_idx}")
            key = self._make_key(sp, patch.dataset_idx)
            if key not in groups:
                groups[key] = 0
            local_i = patch.dataset_local_idx
            # Ensure count covers this index
            groups[key] = max(groups[key], local_i + 1)

        # Second pass: build index
        for patch in patches:
            sp = ds_key_map.get(patch.dataset_idx, f"unknown_{patch.dataset_idx}")
            key = self._make_key(sp, patch.dataset_idx)
            self._index.append((key, patch.dataset_local_idx))

        # Open/create memmaps — scalar offset (1, G, G, G) per sample
        shape_tail = (1, grid_size, grid_size, grid_size)
        self._memmaps: dict[str, np.memmap] = {}
        for key, count in groups.items():
            path = self.deform_dir / f"{key}.bin"
            shape = (count, *shape_tail)
            expected_bytes = int(np.prod(shape)) * 2  # float16
            if path.exists() and path.stat().st_size == expected_bytes:
                mm = np.memmap(path, dtype=np.float16, mode="r+", shape=shape)
            else:
                mm = np.memmap(path, dtype=np.float16, mode="w+", shape=shape)
            self._memmaps[key] = mm

    @staticmethod
    def _make_key(segments_path: str, dataset_idx: int) -> str:
        sanitized = segments_path.replace("/", "_").strip("_")
        return f"{sanitized}__ds{dataset_idx}"

    def get(self, global_indices: list[int]) -> torch.Tensor:
        """Load deformations for batch, returns (B, 1, G, G, G) float32 scalar offsets."""
        out = []
        for gi in global_indices:
            key, li = self._index[gi]
            arr = np.array(self._memmaps[key][li], dtype=np.float32)
            out.append(torch.from_numpy(arr))
        return torch.stack(out)

    def put(self, global_indices: list[int], deforms: torch.Tensor):
        """Write optimized deformations back as float16."""
        arr = deforms.detach().cpu().to(torch.float16).numpy()
        for i, gi in enumerate(global_indices):
            key, li = self._index[gi]
            self._memmaps[key][li] = arr[i]

    def flush(self):
        for mm in self._memmaps.values():
            mm.flush()


def _augment_deform(deform, flip_z, flip_y, flip_x, k):
    """Transform scalar deformation field to match spatial augmentation.

    deform: (B, 1, G, G, G) scalar offset (sign-invariant under flips).
    Only spatial permutations needed — no component sign changes.
    """
    d = deform
    if flip_z:
        d = torch.flip(d, [2])
    if flip_y:
        d = torch.flip(d, [3])
    if flip_x:
        d = torch.flip(d, [4])
    if k % 4:
        d = torch.rot90(d, k % 4, dims=(3, 4))
    return d


def _unaugment_deform(deform, flip_z, flip_y, flip_x, k):
    """Inverse of _augment_deform: augmented space -> pre-aug storage space."""
    d = deform
    if (4 - k) % 4:
        d = torch.rot90(d, (4 - k) % 4, dims=(3, 4))
    if flip_x:
        d = torch.flip(d, [4])
    if flip_y:
        d = torch.flip(d, [3])
    if flip_z:
        d = torch.flip(d, [2])
    return d


def _compute_normal_field(cos_targets, grid_size):
    """Compute unit normal direction at deformation grid resolution.

    The spatial gradient of the cos channel points along the surface
    normal (perpendicular to iso-surfaces of the cos field).

    Args:
        cos_targets: (B, 1, Z, Y, X) cos channel from targets
        grid_size: G — deformation grid resolution per axis

    Returns: (B, 3, G, G, G) unit normals in (dz, dy, dx) order
    """
    cos_ds = F.adaptive_avg_pool3d(cos_targets, (grid_size, grid_size, grid_size))
    cos_pad = F.pad(cos_ds, (1, 1, 1, 1, 1, 1), mode="replicate")
    gz = (cos_pad[:, :, 2:, 1:-1, 1:-1] - cos_pad[:, :, :-2, 1:-1, 1:-1]) / 2
    gy = (cos_pad[:, :, 1:-1, 2:, 1:-1] - cos_pad[:, :, 1:-1, :-2, 1:-1]) / 2
    gx = (cos_pad[:, :, 1:-1, 1:-1, 2:] - cos_pad[:, :, 1:-1, 1:-1, :-2]) / 2
    normals = torch.cat([gz, gy, gx], dim=1)  # (B, 3, G, G, G)
    norm = normals.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return normals / norm


def _enforce_positive_jacobian(deform_scalar, normal_dir):
    """Scale back scalar deformation where the warp Jacobian folds.

    Computes the 3x3 Jacobian of the displacement field (scalar * normal)
    at the coarse grid via central finite differences, checks
    det(I + J) > 0.  Where non-positive, halves the scalar iteratively.

    Args:
        deform_scalar: (B, 1, G, G, G) scalar offsets (modified in-place)
        normal_dir: (B, 3, G, G, G) unit normals
    """
    disp = deform_scalar * normal_dir  # (B, 3, G, G, G)
    disp_pad = F.pad(disp, (1, 1, 1, 1, 1, 1), mode="replicate")
    # Jacobian J[i,j] = d(disp_i)/d(x_j), central finite diffs
    # i = component (0=z, 1=y, 2=x), j = spatial axis
    dz = (disp_pad[:, :, 2:, 1:-1, 1:-1] - disp_pad[:, :, :-2, 1:-1, 1:-1]) / 2
    dy = (disp_pad[:, :, 1:-1, 2:, 1:-1] - disp_pad[:, :, 1:-1, :-2, 1:-1]) / 2
    dx = (disp_pad[:, :, 1:-1, 1:-1, 2:] - disp_pad[:, :, 1:-1, 1:-1, :-2]) / 2
    # det(I + J) where J columns are (dz, dy, dx)
    # I + J has rows for each component, columns for each spatial axis
    # Row i, col j = delta_ij + J[i,j]
    a00 = 1 + dz[:, 0]; a01 = dy[:, 0]; a02 = dx[:, 0]
    a10 = dz[:, 1]; a11 = 1 + dy[:, 1]; a12 = dx[:, 1]
    a20 = dz[:, 2]; a21 = dy[:, 2]; a22 = 1 + dx[:, 2]
    det = (a00 * (a11 * a22 - a12 * a21)
           - a01 * (a10 * a22 - a12 * a20)
           + a02 * (a10 * a21 - a11 * a20))
    # Where det <= 0, scale back scalar
    bad = det <= 0  # (B, G, G, G)
    if bad.any():
        # Halve scalar at bad locations (repeat until positive or 8 tries)
        for _ in range(8):
            deform_scalar.data[bad.unsqueeze(1).expand_as(deform_scalar)] *= 0.5
            disp = deform_scalar * normal_dir
            disp_pad = F.pad(disp, (1, 1, 1, 1, 1, 1), mode="replicate")
            dz = (disp_pad[:, :, 2:, 1:-1, 1:-1] - disp_pad[:, :, :-2, 1:-1, 1:-1]) / 2
            dy = (disp_pad[:, :, 1:-1, 2:, 1:-1] - disp_pad[:, :, 1:-1, :-2, 1:-1]) / 2
            dx = (disp_pad[:, :, 1:-1, 1:-1, 2:] - disp_pad[:, :, 1:-1, 1:-1, :-2]) / 2
            a00 = 1 + dz[:, 0]; a01 = dy[:, 0]; a02 = dx[:, 0]
            a10 = dz[:, 1]; a11 = 1 + dy[:, 1]; a12 = dx[:, 1]
            a20 = dz[:, 2]; a21 = dy[:, 2]; a22 = 1 + dx[:, 2]
            det = (a00 * (a11 * a22 - a12 * a21)
                   - a01 * (a10 * a22 - a12 * a20)
                   + a02 * (a10 * a21 - a11 * a20))
            bad = det <= 0
            if not bad.any():
                break


def _build_warp_grid(deform_full, device):
    """Build a normalized sampling grid from a full-res displacement field.

    deform_full: (B, 3, Z, Y, X) displacement in voxel units (dz, dy, dx).
    Returns grid suitable for F.grid_sample: (B, Z, Y, X, 3) in [-1, 1].
    """
    B, _, Z, Y, X = deform_full.shape
    gz = torch.arange(Z, device=device, dtype=torch.float32)
    gy = torch.arange(Y, device=device, dtype=torch.float32)
    gx = torch.arange(X, device=device, dtype=torch.float32)
    grid_z, grid_y, grid_x = torch.meshgrid(gz, gy, gx, indexing="ij")
    # grid_sample expects (x, y, z) order in last dim
    base = torch.stack([grid_x, grid_y, grid_z], dim=-1)  # (Z, Y, X, 3)
    base = base.unsqueeze(0).expand(B, -1, -1, -1, -1)
    # Convert displacement (dz, dy, dx) -> (dx, dy, dz)
    disp = deform_full.permute(0, 2, 3, 4, 1).flip(-1)
    grid = base + disp
    # Normalize each axis to [-1, 1]
    grid[..., 0] = 2.0 * grid[..., 0] / max(X - 1, 1) - 1.0
    grid[..., 1] = 2.0 * grid[..., 1] / max(Y - 1, 1) - 1.0
    grid[..., 2] = 2.0 * grid[..., 2] / max(Z - 1, 1) - 1.0
    return grid


def _apply_warp(cos_gm, validity, deform_full):
    """Warp cos/grad_mag targets and validity mask using displacement field.

    cos_gm:      (B, 2, Z, Y, X) — channels [cos, grad_mag]
    validity:    (B, 1, Z, Y, X)
    deform_full: (B, 3, Z, Y, X) — displacement in voxel units

    Returns (warped_cos_gm, warped_validity).
    """
    grid = _build_warp_grid(deform_full, cos_gm.device)
    warped_cg = F.grid_sample(
        cos_gm, grid, mode="bilinear", padding_mode="zeros", align_corners=True,
    )
    warped_v = F.grid_sample(
        validity, grid, mode="bilinear", padding_mode="zeros", align_corners=True,
    )
    return warped_cg, warped_v


def _deform_inner_loop(
    pred_cos_gm,   # (B, 2, Z, Y, X) detached model prediction
    targets,       # (B, 8, Z, Y, X) original targets
    validity,      # (B, 1, Z, Y, X) original validity
    deform,        # (B, 1, G, G, G) scalar offset (requires_grad)
    n_iters: int,
    inner_lr: float,
    max_frac: float,
    scale_loss_mse_fn,
    scale_loss_l1_fn,
):
    """Optimize a normal-aligned scalar deformation field.

    The deformation is parameterized as a scalar offset per grid point,
    multiplied by the local surface normal direction (computed from the
    cos channel gradient).  This constrains motion along normals and
    greatly reduces self-intersection risk.

    Uses a log-space LR ramp from ``inner_lr`` to ``inner_lr * 10000``
    and tracks the best (lowest loss) deformation seen at any iteration.

    Returns the best deform tensor (B, 1, G, G, G) on GPU, grad detached.
    """
    G = deform.shape[-1]
    device = pred_cos_gm.device

    # Cast everything to float32 for stable inner loop
    pred_f = pred_cos_gm.float()
    cos_gm_orig = targets[:, 0:2].float()
    val_f = validity.float()

    # Compute normal direction from cos gradient at grid resolution
    with torch.no_grad():
        normal_dir = _compute_normal_field(cos_gm_orig[:, 0:1], G)

    # Pre-compute max scalar displacement from original grad_mag
    # and a validity mask at grid resolution (only deform where GT exists)
    with torch.no_grad():
        gm_ds = F.adaptive_avg_pool3d(cos_gm_orig[:, 1:2], (G, G, G))
        max_disp = max_frac / gm_ds.clamp(min=0.02)  # (B, 1, G, G, G)
        val_ds = F.adaptive_avg_pool3d(val_f, (G, G, G))
        valid_mask = (val_ds > 0.01).float()  # (B, 1, G, G, G)

    inner_opt = torch.optim.SGD([deform], lr=inner_lr)
    Z, Y, X = targets.shape[2:]

    # Log-space LR ramp: inner_lr → inner_lr * 10000
    lr_log_start = math.log10(inner_lr)
    lr_log_end = math.log10(inner_lr * 10000.0)

    best_loss = float("inf")
    best_deform = deform.detach().clone()

    with torch.amp.autocast("cuda", enabled=False):
        for it in range(n_iters):
            frac = it / max(n_iters - 1, 1)
            lr_now = 10 ** (lr_log_start + frac * (lr_log_end - lr_log_start))
            for pg in inner_opt.param_groups:
                pg["lr"] = lr_now

            inner_opt.zero_grad()
            # Scalar × normal → 3D displacement
            disp_3d = deform * normal_dir  # (B, 3, G, G, G)
            deform_full = F.interpolate(
                disp_3d, size=(Z, Y, X), mode="trilinear", align_corners=False,
            )
            warped_cg, warped_v = _apply_warp(cos_gm_orig, val_f, deform_full)
            cos_w = warped_cg[:, 1:2] * 20.0
            loss = (
                scale_loss_mse_fn(pred_f[:, 0:1], warped_cg[:, 0:1], mask=warped_v, weight=cos_w)
                + scale_loss_l1_fn(pred_f[:, 1:2], warped_cg[:, 1:2], mask=warped_v, weight=cos_w)
            )
            loss.backward()
            inner_opt.step()

            # Clamp scalar magnitude, zero outside valid region,
            # and enforce positive Jacobian
            with torch.no_grad():
                deform.data.clamp_(-max_disp, max_disp)
                deform.data.mul_(valid_mask)
                _enforce_positive_jacobian(deform, normal_dir)

                cur_loss = loss.item()
                if cur_loss < best_loss:
                    best_loss = cur_loss
                    best_deform = deform.detach().clone()

    return best_deform


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _log_vis(log_images, tag, image, pred, targets, mask, step,
             targets_original=None):
    with torch.no_grad():
        b = min(4, image.size(0))
        mid_z = pred.size(2) // 2
        mid_z_img = image.size(2) // 2

        log_images(f"{tag}/input_z", image[:b, :, mid_z_img], step)
        m = mask[:b, :, mid_z]

        for i, name in enumerate(_CHANNEL_NAMES):
            p = pred[:b, i:i+1, mid_z]
            t = targets[:b, i:i+1, mid_z]
            log_images(f"{tag}/{name}_pred", p.clamp(0, 1), step)
            log_images(f"{tag}/{name}_gt", (t * m).clamp(0, 1), step)
            # Show original (pre-deformation) GT for cos and grad_mag
            if targets_original is not None and i < 2:
                to = targets_original[:b, i:i+1, mid_z]
                log_images(f"{tag}/{name}_gt_original", (to * m).clamp(0, 1), step)

        log_images(f"{tag}/validity", m, step)


# ---------------------------------------------------------------------------
# Multi-scale refinement helpers
# ---------------------------------------------------------------------------

# Scale offset → scale channel constant (tells model what resolution CT is at)
_SCALE_CHANNEL_VALUES = {-1: 0.5, 0: 1.0, 1: 2.0}

# Training mode definitions: list of (offset, has_prior) tuples per mode.
# Modes 0–2: single-scale (full batch, 1 pass)
# Modes 3–5: cross-scale chains (half batch, 2 passes)
# Mode 6: self-refinement N→N (half batch, 2 passes, N chosen randomly)
REFINE_MODES_FIXED = [
    [(-1, False)],               # 0: scale -1 only
    [(0, False)],                # 1: scale 0 only
    [(1, False)],                # 2: scale +1 only
    [(-1, False), (0, True)],    # 3: -1 → 0
    [(0, False), (1, True)],     # 4: 0 → +1
    [(-1, False), (1, True)],    # 5: -1 → +1
]


def _pick_refine_mode(
    available_offsets: set,
    rank: int,
    world_size: int,
    device: torch.device,
    is_dist: bool,
) -> Tuple[int, list]:
    """Pick a random training mode, synced across DDP ranks.

    Returns (mode_index, scale_sequence) where scale_sequence is a
    list of (offset, has_prior) tuples.
    """
    if is_dist:
        buf = torch.zeros(2, dtype=torch.long, device=device)
        if rank == 0:
            mode_idx = int(np.random.randint(0, 7))
            # For mode 6 (self-refine), pick N from available offsets
            refine_n = int(np.random.choice(
                sorted(available_offsets),
            )) if available_offsets else 0
            buf[0] = mode_idx
            buf[1] = refine_n
        dist.broadcast(buf, src=0)
        mode_idx = int(buf[0].item())
        refine_n = int(buf[1].item())
    else:
        mode_idx = int(np.random.randint(0, 7))
        refine_n = int(np.random.choice(
            sorted(available_offsets),
        )) if available_offsets else 0

    if mode_idx < 6:
        seq = REFINE_MODES_FIXED[mode_idx]
    else:
        # Mode 6: self-refinement N → N
        seq = [(refine_n, False), (refine_n, True)]

    # Check availability: all offsets in the sequence must be available.
    # If not, fall back to mode 1 (scale 0 only).
    needed = {off for off, _ in seq}
    if not needed.issubset(available_offsets):
        seq = [(0, False)]

    return mode_idx, seq


def _read_ct_at_offset(
    patch_info_list: list,
    vol_groups: dict,
    scale_offset: int,
    patch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Read CT at an offset zarr level for a batch of samples.

    Returns (B, 1, P, P, P) float32 tensor on device.
    """
    from vesuvius.neural_tracing.datasets.common import _read_volume_crop

    crops = []
    P = patch_size
    for pi in patch_info_list:
        ds_idx = pi["dataset_idx"]
        base_level = pi["scale"]
        target_level = base_level - scale_offset

        group = vol_groups.get(ds_idx)
        if group is None:
            # No group available — return zeros
            crops.append(np.zeros((P, P, P), dtype=np.float32))
            continue

        try:
            arr = group[str(target_level)]
        except (KeyError, IndexError):
            crops.append(np.zeros((P, P, P), dtype=np.float32))
            continue

        crop_size = (P, P, P)
        if scale_offset == 1:
            # Finer level: use fine_offset from dataset
            fine_off = pi.get("fine_offset")
            base_min = pi["world_min"]
            if fine_off is not None:
                fine_min = np.array([
                    int((base_min[d] + fine_off[d]) * 2)
                    for d in range(3)
                ], dtype=np.int64)
            else:
                # Fallback: center
                center = pi["world_center"]
                fine_min = np.array([
                    int(round(center[d] * 2 - P / 2))
                    for d in range(3)
                ], dtype=np.int64)
            target_min = fine_min
        else:
            coord_factor = 2.0 ** scale_offset
            center = pi["world_center"]
            target_min = np.array([
                int(round(center[d] * coord_factor - P / 2))
                for d in range(3)
            ], dtype=np.int64)

        target_max = target_min + P
        vol_crop = _read_volume_crop(
            arr, crop_size=crop_size,
            min_corner=target_min, max_corner=target_max,
            image_normalization="unit",
        )
        crops.append(np.asarray(vol_crop, dtype=np.float32))

    batch = np.stack(crops, axis=0)[:, np.newaxis]  # (B, 1, P, P, P)
    return torch.as_tensor(batch, dtype=torch.float32, device=device)


def _resample_prior(
    pred: torch.Tensor,
    src_offset: int,
    tgt_offset: int,
    patch_size: int,
    fine_offsets: list = None,
) -> torch.Tensor:
    """Resample a prediction from source scale to target scale coordinates.

    Args:
        pred: (B, 8, P, P, P) prediction at source scale
        src_offset: source scale offset (-1, 0, +1)
        tgt_offset: target scale offset (-1, 0, +1)
        patch_size: P
        fine_offsets: list of (3,) arrays, one per batch element — the
            fine_offset from patch_info. Used to compute the crop position
            when going to a finer scale.

    Returns: (B, 8, P, P, P) resampled to target coordinates
    """
    if src_offset == tgt_offset:
        return pred  # same scale, no resampling

    P = patch_size
    # Span of target world in source voxels
    span = int(round(P * (2.0 ** (src_offset - tgt_offset))))

    if span >= P:
        # Target covers less world → just return pred (shouldn't happen
        # in our chain directions, but handle gracefully)
        return pred

    # Crop region in source: target is somewhere within source.
    # For now, use center crop. If fine_offsets are provided and going
    # to a finer scale, compute the exact position.
    # The target's world region starts at an offset within the source's
    # wider world. In source voxels, the offset depends on the scale
    # relationship and the fine_offset.
    B = pred.shape[0]

    # Default: center crop
    start = (P - span) // 2
    cropped = pred[:, :, start:start+span, start:start+span, start:start+span]

    # Upsample to P
    resampled = F.interpolate(
        cropped, size=(P, P, P), mode="trilinear", align_corners=False,
    )
    return resampled


def _build_refine_input(
    image: torch.Tensor,
    prior: Optional[torch.Tensor],
    scale_offset: int,
) -> torch.Tensor:
    """Build 11-channel model input: CT + prior + validity + scale.

    Args:
        image: (B, 1, Z, Y, X) CT
        prior: (B, 8, Z, Y, X) prior prediction or None
        scale_offset: -1, 0, or +1

    Returns: (B, 11, Z, Y, X)
    """
    B, _, Z, Y, X = image.shape
    device = image.device

    if prior is not None:
        validity = torch.ones(B, 1, Z, Y, X, device=device, dtype=image.dtype)
        scale_ch = torch.full(
            (B, 1, Z, Y, X), _SCALE_CHANNEL_VALUES[scale_offset],
            device=device, dtype=image.dtype,
        )
        return torch.cat([image, prior, validity, scale_ch], dim=1)
    else:
        # No prior: zeros for prior+validity, scale channel still active
        zeros = torch.zeros(B, 9, Z, Y, X, device=device, dtype=image.dtype)
        scale_ch = torch.full(
            (B, 1, Z, Y, X), _SCALE_CHANNEL_VALUES[scale_offset],
            device=device, dtype=image.dtype,
        )
        return torch.cat([image, zeros, scale_ch], dim=1)


def _get_gt_at_offset(
    batch: dict,
    targets_0: torch.Tensor,
    validity_0: torch.Tensor,
    dir_sparse_mask_0: torch.Tensor,
    dir_dense_mask_0: torch.Tensor,
    dir_axis_weight_0: torch.Tensor,
    scale_offset: int,
    device: torch.device,
    compute_batch_targets_fn=None,
    same_surface_threshold: float = 0.0,
):
    """Get GT at a given scale offset.

    offset 0: pass through scale-0 GT
    offset -1: avg_pool scale-0 GT by 2
    offset +1: compute from fine-res batch data (native)
    """
    if scale_offset == 0:
        return targets_0, validity_0, dir_sparse_mask_0, dir_dense_mask_0, dir_axis_weight_0

    if scale_offset == -1:
        # Coarser: pool from scale 0
        v = validity_0
        v_count = F.avg_pool3d(v, kernel_size=2, stride=2)
        tgt = F.avg_pool3d(targets_0 * v, 2, 2) / v_count.clamp(min=1e-8)
        # grad_mag scales with resolution (coarser = larger spacing per voxel)
        tgt[:, 1:2] *= 2.0
        val = F.max_pool3d(v, 2, 2)
        dsm = F.max_pool3d(dir_sparse_mask_0, 2, 2)
        ddm = F.max_pool3d(dir_dense_mask_0, 2, 2)
        daw = F.avg_pool3d(dir_axis_weight_0 * v, 2, 2) / v_count.clamp(min=1e-8)
        return tgt, val, dsm, ddm, daw

    if scale_offset == 1:
        # Finer: compute from native fine-res data
        if compute_batch_targets_fn is None or "surface_masks_fine" not in batch:
            # Fallback: upsample from scale 0
            P = targets_0.shape[2]
            tgt_up = F.interpolate(
                targets_0, size=(P, P, P), mode="trilinear",
                align_corners=False,
            )
            # No — targets_0 is already P. For +1 we need P output from
            # a P/2 center crop upsampled.  This is the fallback path;
            # the primary path uses native fine GT below.
            center = P // 4
            half = P // 2
            tgt_crop = targets_0[:, :, center:center+half, center:center+half, center:center+half]
            val_crop = validity_0[:, :, center:center+half, center:center+half, center:center+half]
            tgt_up = F.interpolate(tgt_crop, size=(P, P, P), mode="trilinear", align_corners=False)
            val_up = F.interpolate(val_crop, size=(P, P, P), mode="nearest")
            dsm_crop = dir_sparse_mask_0[:, :, center:center+half, center:center+half, center:center+half]
            ddm_crop = dir_dense_mask_0[:, :, center:center+half, center:center+half, center:center+half]
            daw_crop = dir_axis_weight_0[:, :, center:center+half, center:center+half, center:center+half]
            dsm_up = F.interpolate(dsm_crop, size=(P, P, P), mode="nearest")
            ddm_up = F.interpolate(ddm_crop, size=(P, P, P), mode="nearest")
            daw_up = F.interpolate(daw_crop, size=(P, P, P), mode="trilinear", align_corners=False)
            return tgt_up, val_up, dsm_up, ddm_up, daw_up

        # Native fine GT: use fine-res surface_masks from dataset
        # Build a synthetic batch dict with the fine-res data
        fine_batch = {
            "surface_masks": batch["surface_masks_fine"],
            "tensor_moments": batch["tensor_moments_fine"].to(device),
            "normals_valid": batch["normals_valid_fine"].to(device),
            "surface_chain_info": batch["surface_chain_info"],
            "num_surfaces": batch["num_surfaces"],
        }
        (
            tgt_fine, val_fine,
            dsm_fine, ddm_fine, daw_fine,
            _, _, _,
        ) = compute_batch_targets_fn(
            fine_batch, device,
            same_surface_threshold=same_surface_threshold,
        )
        return tgt_fine, val_fine, dsm_fine, ddm_fine, daw_fine

    raise ValueError(f"Unknown scale_offset: {scale_offset}")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    config_path: str,
    log_dir: str = "runs/tifxyz3d",
    run_name: str = "tifxyz",
    epochs: int = 100,
    batch_size: int = 2,
    lr: float = 1e-2,
    patch_size: int = 128,
    label_patch_size: Optional[int] = None,
    model_patch_size: Optional[int] = None,
    w_cos: float = 1.0,
    w_mag: float = 1.0,
    w_dir: float = 1.0,
    w_dir_dense: float = 0.1,
    w_smooth: float = 0.1,
    num_workers: int = 4,
    val_fraction: float = 0.15,
    device: str = "cuda",
    weights: Optional[str] = None,
    norm_type: str = "none",
    upsample_mode: str = "trilinear",
    output_sigmoid: bool = False,
    precision: str = "bf16",
    verbose: bool = False,
    wandb_enabled: bool = False,
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    wandb_tags: Optional[List[str]] = None,
    reinit_decoder_scales: int = 0,
    himag_filter: bool = True,
    w_dist: float = 0.0,
    w_fft: float = 0.0,
    num_loss_scales: int = 5,
    deform_enabled: bool = True,
    deform_stride: int = 8,
    deform_inner_iters: int = 100,
    deform_inner_lr: float = 1000.0,
    deform_max_frac: float = 0.3,
    refine: bool = False,
) -> None:
    rank, local_rank, world_size, is_dist = _init_distributed()
    is_main = rank == 0

    # Pin each rank to its visible device. With CUDA_VISIBLE_DEVICES,
    # local_rank indexes into the (already-masked) visible set, so
    # cuda:{local_rank} always refers to a device the process owns.
    if is_dist:
        device = f"cuda:{local_rank}"
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"

    # Rank 0 picks the run_dir name, then broadcasts it so every rank
    # resolves the same path (timestamps would otherwise drift across
    # process startup).
    if is_main:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir_str = str(Path(log_dir) / f"{timestamp}_{run_name}")
    else:
        run_dir_str = ""
    if is_dist:
        bcast = [run_dir_str]
        dist.broadcast_object_list(bcast, src=0)
        run_dir_str = bcast[0]
    run_dir = Path(run_dir_str)
    if is_main:
        run_dir.mkdir(parents=True, exist_ok=True)
    if is_dist:
        dist.barrier()

    # Load config
    with open(config_path, "r") as f:
        config = json.load(f)

    config["patch_size"] = patch_size
    effective_label_patch = (
        int(label_patch_size) if label_patch_size is not None
        else int(patch_size)
    )
    config["label_patch_size"] = effective_label_patch
    if effective_label_patch != int(patch_size):
        config["random_paste_offset"] = True
        print(
            f"{TAG} label_patch_size={effective_label_patch} < "
            f"patch_size={patch_size}: GT region placed at random "
            f"offset inside larger CT crop (up to L/2 cropping per "
            f"edge during training)",
            flush=True,
        )

    same_surface_threshold = config.get("same_surface_threshold")
    if same_surface_threshold is not None:
        same_surface_threshold = float(same_surface_threshold)
        print(
            f"{TAG} same_surface_threshold={same_surface_threshold} — duplicate "
            "wraps will be merged inside compute_patch_labels",
            flush=True,
        )

    # Save run config
    run_config = {
        "config_path": config_path,
        "epochs": epochs, "batch_size": batch_size, "lr": lr,
        "patch_size": patch_size,
        "label_patch_size": effective_label_patch,
        "w_cos": w_cos, "w_mag": w_mag, "w_dir": w_dir,
        "w_dir_dense": w_dir_dense, "w_smooth": w_smooth,
        "num_workers": num_workers, "val_fraction": val_fraction,
        "device": device, "weights": weights,
        "norm_type": norm_type, "upsample_mode": upsample_mode,
        "output_sigmoid": output_sigmoid,
        "precision": precision,
        "same_surface_threshold": same_surface_threshold,
        "deform_enabled": deform_enabled,
        "deform_stride": deform_stride,
        "deform_inner_iters": deform_inner_iters,
        "deform_inner_lr": deform_inner_lr,
        "deform_max_frac": deform_max_frac,
        "refine": refine,
        "cmd": " ".join(sys.argv),
    }
    if is_main:
        (run_dir / "config.json").write_text(json.dumps(run_config, indent=2) + "\n")

    # Optional W&B init. Soft dep — failure never breaks training.
    # Only rank 0 talks to W&B.
    wandb_run = None
    if wandb_enabled and is_main:
        try:
            import wandb
            wandb_run = wandb.init(
                project=wandb_project or "lasagna-tifxyz",
                entity=wandb_entity,
                name=wandb_run_name or f"{timestamp}_{run_name}",
                dir=str(run_dir),
                config=run_config,
                tags=wandb_tags or None,
                resume="allow",
            )
            print(f"{TAG} W&B logging enabled: {wandb_run.url}", flush=True)
        except Exception as e:
            print(
                f"{TAG} W&B init failed ({e}); continuing without W&B",
                flush=True,
            )
            wandb_run = None

    # Dataset
    print(f"{TAG} building dataset...", flush=True)
    # When --refine is set, inject refine_mode into config so the
    # dataset opens zarr groups and computes fine-res GT, and force
    # scale_aug off (mutually exclusive).
    if refine:
        config["refine_mode"] = True
        config["scale_aug_prob"] = 0.0
    # include_geometry + include_patch_ref are required for the full
    # render_batch_figure path (arrow drawers + fresh inference crop).
    full_dataset = TifxyzLasagnaDataset(
        config, apply_augmentation=False,
        include_geometry=True,
        include_patch_ref=True,
        refine_mode=refine,
    )

    # Volume groups for multi-scale CT reads (used by refinement mode)
    refine_vol_groups = full_dataset.volume_groups if refine else {}

    # Probe which zarr levels are available per dataset for refinement
    refine_available_offsets: set = set()
    if refine and refine_vol_groups:
        # Check all datasets, collect the intersection of available offsets
        all_avail = []
        for ds_idx, group in refine_vol_groups.items():
            avail = set()
            avail.add(0)  # base level always available
            # Find the base scale from the first patch of this dataset
            base_level = None
            for p in full_dataset.patches:
                if p.dataset_idx == ds_idx:
                    base_level = p.scale
                    break
            if base_level is not None:
                if str(base_level + 1) in group:
                    avail.add(-1)  # coarser
                if base_level > 0 and str(base_level - 1) in group:
                    avail.add(1)  # finer
            all_avail.append(avail)
        # Use intersection so all datasets support the chosen mode
        refine_available_offsets = set.intersection(*all_avail) if all_avail else {0}
        if is_main:
            print(f"{TAG} refine mode: available offsets = "
                  f"{sorted(refine_available_offsets)}", flush=True)

    n = len(full_dataset)
    # Fixed 10-sample val set, deterministically spread across the
    # dataset via evenly-spaced indices. `val_fraction` is ignored.
    n_val = min(10, n)
    val_indices = sorted({
        int(round(i)) for i in np.linspace(0, n - 1, n_val)
    })
    val_set = set(val_indices)
    train_indices = [i for i in range(n) if i not in val_set]
    n_train = len(train_indices)
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    # Scale augmentation: second dataset instance with scale_aug_active.
    # GT is emitted at 2× resolution; pooling happens in the train loop
    # AFTER compute_batch_targets so EDT/cos run at full res.
    # Disabled when --refine is set (mutually exclusive).
    scale_aug_enabled = full_dataset.scale_aug_prob > 0 and not refine
    scale_aug_factor = full_dataset.scale_aug_factor
    train_dataset_aug = None
    if scale_aug_enabled:
        if is_main:
            print(f"{TAG} building scale-aug dataset (f={scale_aug_factor})...",
                  flush=True)
        full_dataset_aug = TifxyzLasagnaDataset(
            config, apply_augmentation=False,
            include_geometry=True,
            include_patch_ref=True,
        )
        full_dataset_aug.scale_aug_active = True
        train_dataset_aug = Subset(full_dataset_aug, train_indices)

    if is_dist:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank,
            shuffle=True, drop_last=True,
        )
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank,
            shuffle=False, drop_last=False,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler,
            num_workers=num_workers, pin_memory=True, timeout=120,
            collate_fn=collate_variable_surfaces,
        )
        if train_dataset_aug is not None:
            train_sampler_aug = DistributedSampler(
                train_dataset_aug, num_replicas=world_size, rank=rank,
                shuffle=True, drop_last=True,
            )
            train_loader_aug = DataLoader(
                train_dataset_aug, batch_size=batch_size,
                sampler=train_sampler_aug,
                num_workers=num_workers, pin_memory=True, timeout=120,
                collate_fn=collate_variable_surfaces,
            )
        val_loader = DataLoader(
            val_dataset, batch_size=1, sampler=val_sampler,
            num_workers=num_workers, timeout=120,
            collate_fn=collate_variable_surfaces,
        )
    else:
        train_sampler = None
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, timeout=120,
            collate_fn=collate_variable_surfaces,
        )
        if train_dataset_aug is not None:
            train_loader_aug = DataLoader(
                train_dataset_aug, batch_size=batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=True, timeout=120,
                collate_fn=collate_variable_surfaces,
            )
        val_loader = DataLoader(
            val_dataset, batch_size=1, shuffle=False,
            num_workers=num_workers, timeout=120,
            collate_fn=collate_variable_surfaces,
        )
    if is_main:
        aug_str = f" (scale_aug f={scale_aug_factor})" if scale_aug_enabled else ""
        print(f"{TAG} {n_train} train / {n_val} val patches "
              f"(world_size={world_size}){aug_str}", flush=True)

    # GT deformation store (disk-backed, per-sample low-res warp fields)
    deform_store: DeformationStore | None = None
    deform_grid_size = 0
    if deform_enabled:
        deform_grid_size = effective_label_patch // deform_stride
        if deform_grid_size < 2:
            raise ValueError(
                f"deform_stride={deform_stride} too large for "
                f"label_patch_size={effective_label_patch}"
            )
        deform_store = DeformationStore(
            full_dataset.patches, config, deform_grid_size, run_dir,
        )
        if is_main:
            print(
                f"{TAG} deformation enabled: grid={deform_grid_size}³, "
                f"stride={deform_stride}, inner_iters={deform_inner_iters}, "
                f"inner_lr={deform_inner_lr}, max_frac={deform_max_frac}",
                flush=True,
            )

    # Model
    model, norm_type, upsample_mode = build_model(
        patch_size, device, weights, norm_type=norm_type,
        upsample_mode=upsample_mode, batch_size=batch_size,
        model_patch_size=model_patch_size,
        reinit_decoder_scales=reinit_decoder_scales,
        refine=refine,
    )
    if is_dist:
        # DDP broadcasts module state from rank 0 on init, so
        # checkpoint loads on rank 0 (or random init from rank 0)
        # propagate to all ranks automatically.
        model = DDP(
            model, device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=True,
            find_unused_parameters=False,
        )
    # Convenience handle for state_dict / inference forward that
    # bypasses DDP's collective hooks.
    model_core = model.module if is_dist else model
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    # LR warmup + per-step cosine decay
    warmup_steps = 200
    total_steps = epochs * len(train_loader)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        return 1.0
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_steps, 1),
    )

    # Precision
    if precision == "bf16":
        amp_dtype = torch.bfloat16
        use_autocast = True
        scaler = torch.amp.GradScaler(enabled=False)
    elif precision == "fp16":
        amp_dtype = torch.float16
        use_autocast = True
        scaler = torch.amp.GradScaler(enabled=(device != "cpu"))
    else:
        amp_dtype = torch.float32
        use_autocast = False
        scaler = torch.amp.GradScaler(enabled=False)
    if is_main:
        print(f"{TAG} precision: {precision}, upsample: {upsample_mode}", flush=True)

    # Losses
    mse_loss = MaskedMSE()
    smooth_l1_loss = MaskedSmoothL1()
    scale_loss_mse = ScaleSpaceLoss3D(mse_loss, num_scales=num_loss_scales)
    scale_loss_l1 = ScaleSpaceLoss3D(smooth_l1_loss, num_scales=num_loss_scales)

    # GPU pause/resume server — allows other apps to reclaim the GPU.
    pause_server = GpuPauseServer() if is_main else None

    def _gpu_offload():
        import gc
        model.cpu()
        for s in optimizer.state.values():
            for k, v in s.items():
                if isinstance(v, torch.Tensor):
                    s[k] = v.cpu()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    def _gpu_reload():
        model.to(device)
        for s in optimizer.state.values():
            for k, v in s.items():
                if isinstance(v, torch.Tensor):
                    s[k] = v.to(device)

    writer = SummaryWriter(log_dir=str(run_dir)) if is_main else None
    best_val_loss = float("inf")
    global_step = 0

    # Dual-sink logging helpers. TB gets everything unchanged; W&B
    # gets scalars at full cadence and images gated to every
    # WANDB_IMG_EVERY steps and JPEG-compressed for bandwidth.
    import io as _io
    WANDB_IMG_EVERY = 1000

    def _wandb_image_allowed(step: int) -> bool:
        return wandb_run is not None and (step % WANDB_IMG_EVERY == 0)

    def _to_jpg_wandb(arr_hwc_uint8):
        import wandb
        buf = _io.BytesIO()
        Image.fromarray(arr_hwc_uint8).save(buf, format="JPEG", quality=85)
        buf.seek(0)
        return wandb.Image(Image.open(buf).copy())

    def _chw_to_hwc_uint8(t):
        a = t.detach().float().clamp(0, 1).cpu().numpy()
        if a.shape[0] == 1:
            a = np.repeat(a, 3, axis=0)
        elif a.shape[0] > 3:
            a = a[:3]
        return (a.transpose(1, 2, 0) * 255.0 + 0.5).astype(np.uint8)

    def log_scalar(tag, value, step):
        if not is_main:
            return
        writer.add_scalar(tag, value, step)
        if wandb_run is not None:
            wandb_run.log({tag: value}, step=step)

    def log_image(tag, img_hwc_uint8, step):
        if not is_main:
            return
        writer.add_image(tag, img_hwc_uint8, step, dataformats="HWC")
        if _wandb_image_allowed(step):
            wandb_run.log({tag: _to_jpg_wandb(img_hwc_uint8)}, step=step)

    def log_images(tag, imgs_bchw, step):
        if not is_main:
            return
        writer.add_images(tag, imgs_bchw, step)
        if _wandb_image_allowed(step):
            gallery = [
                _to_jpg_wandb(_chw_to_hwc_uint8(imgs_bchw[i]))
                for i in range(imgs_bchw.shape[0])
            ]
            wandb_run.log({tag: gallery}, step=step)

    # Lasagna3d-style full-vis logging: one JPEG per ~1000 samples,
    # written to run_dir/vis and mirrored into TB as an image.
    # The live model is handed to render_batch_figure so the figure
    # includes inference + residual rows, not just supervision.
    vis_ctx = build_inference_context(
        model_path=None, config=config,
        same_surface_threshold=same_surface_threshold,
    )
    arch_patch = int(model_patch_size) if model_patch_size else int(patch_size)
    vis_ctx["model_build_patch_size"] = arch_patch
    # Run vis inference at the model's native build patch size, so
    # --model-patch-size controls the CT crop fed to the vis model.
    vis_ctx["inference_size_eff"] = arch_patch
    vis_ctx["compare_size"] = min(int(patch_size), arch_patch)
    vis_ctx["output_sigmoid"] = bool(output_sigmoid)
    vis_ctx["loss_weights"] = (float(w_cos), float(w_mag), float(w_dir))
    vis_out_dir = run_dir / "vis"
    if is_main:
        vis_out_dir.mkdir(exist_ok=True)
    vis_interval_steps = max(1, 1000 // max(batch_size, 1))

    def _log_full_vis(batch_for_vis, tag: str, step: int) -> None:
        if not is_main:
            return
        try:
            sample0 = _sample_from_batch(batch_for_vis)
            patch_info0 = sample0["patch_info"]
            ds_name = str(patch_info0.get("dataset_name",
                                          patch_info0.get("dataset", tag)))
            idx0 = int(patch_info0.get("patch_idx", step))
            title = (f"step={step}  "
                     + default_vis_title(ds_name, idx0, sample0))
            out_path = vis_out_dir / f"{tag}_step_{step:07d}.jpg"
            was_training = model.training
            model.eval()
            # Release fragmented free blocks so the vis inference
            # forward can find a contiguous activation buffer.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            try:
                render_batch_figure(
                    batch_for_vis, out_path, title,
                    arrow_seed=step,
                    inference_ctx=vis_ctx,
                    model=model_core,
                    inference_image_override=batch_for_vis["image"],
                )
            finally:
                if was_training:
                    model.train()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            img_rgb = np.asarray(Image.open(out_path).convert("RGB"))
            log_image(f"{tag}/full_vis", img_rgb, step)
        except Exception as e:
            print(f"{TAG} full_vis ({tag}) render failed at step "
                  f"{step}: {e}", flush=True)

    VIS_SAMPLES = 8

    # Initial eval before training so we see init/load performance.
    if is_main:
        print(f"{TAG} running pre-training eval at step 0...", flush=True)
    _init_vis_batch: list = []
    init_val_loss = _evaluate(
        model, val_loader, scale_loss_mse, scale_loss_l1,
        device, log_scalar, log_images, global_step,
        w_cos, w_mag, w_dir,
        w_dir_dense=w_dir_dense, w_smooth=w_smooth,
        amp_dtype=amp_dtype, use_autocast=use_autocast,
        verbose=verbose and is_main,
        same_surface_threshold=same_surface_threshold,
        output_sigmoid=output_sigmoid,
        vis_batch_out=_init_vis_batch,
        device_type=device_type,
        world_size=world_size,
        is_main=is_main,
        deform_store=deform_store,
        deform_inner_iters=deform_inner_iters,
        deform_inner_lr=deform_inner_lr,
        deform_max_frac=deform_max_frac,
        vis_samples=VIS_SAMPLES,
        w_dist=w_dist,
        w_fft=w_fft,
        refine=refine,
    )
    if _init_vis_batch:
        _log_full_vis(_init_vis_batch[0], "val", global_step)
    _init_vis_batch.clear()
    if is_main:
        print(f"{TAG} init val={init_val_loss:.4f}", flush=True)

    last_val_loss = init_val_loss
    val_every_steps = 200

    # Hi-mag skip hysteresis. The per-sample screen rejects samples
    # with scale-space L1 >= 1 to keep pathological grad_mag outliers
    # out of the gradient. Early in training, almost everything is
    # above the threshold and the screen would starve the loss; this
    # state machine disables the filter when 100 consecutive samples
    # exceed threshold and re-enables it once >50% of a 100-sample
    # window is back under threshold. In DDP we all-reduce per-step
    # (above, total) counts so every rank updates state identically.
    from collections import deque
    HIMAG_WINDOW = 100
    HIMAG_DISABLE_ABOVE = 75  # disable when >= this many of window are above
    HIMAG_ENABLE_ABOVE = 50   # re-enable when < this many of window are above
    himag_flags: "deque[int]" = deque(maxlen=HIMAG_WINDOW)
    himag_enabled = himag_filter

    for epoch in range(epochs):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        epoch_losses: List[float] = []

        # Two-iterator setup for per-batch scale augmentation.
        iter_normal = iter(train_loader)
        iter_aug = None
        if scale_aug_enabled:
            if is_dist and hasattr(train_loader_aug, 'sampler'):
                train_loader_aug.sampler.set_epoch(epoch)
            iter_aug = iter(train_loader_aug)
        steps_per_epoch = len(train_loader)

        if is_main:
            train_iter = tqdm(
                range(steps_per_epoch),
                desc=f"epoch {epoch + 1}/{epochs} train",
                dynamic_ncols=True,
                mininterval=0.0, miniters=1,
            )
        else:
            train_iter = range(steps_per_epoch)

        n_seen = 0
        n_skipped = 0
        n_read_err = 0
        n_hi_mag = 0
        n_scale_aug = 0
        n_surfaces_pre = 0
        n_surfaces_post = 0

        for _step_i in train_iter:
            # Per-batch scale-aug decision.  In DDP all ranks must
            # agree, so rank 0 draws and broadcasts.
            if scale_aug_enabled:
                if is_dist:
                    flag = torch.zeros(1, dtype=torch.long, device=device)
                    if rank == 0:
                        flag[0] = int(np.random.random() < full_dataset.scale_aug_prob)
                    dist.broadcast(flag, src=0)
                    use_aug = bool(flag.item())
                else:
                    use_aug = np.random.random() < full_dataset.scale_aug_prob
            else:
                use_aug = False

            try:
                if use_aug:
                    batch = next(iter_aug, None)
                    if batch is None:
                        iter_aug = iter(train_loader_aug)
                        batch = next(iter_aug, None)
                else:
                    batch = next(iter_normal, None)
                    if batch is None:
                        iter_normal = iter(train_loader)
                        batch = next(iter_normal, None)
            except Exception as _dl_err:
                # DataLoader timeout or worker crash — treat as read error.
                print(f"{TAG} dataloader error: {type(_dl_err).__name__}: {_dl_err}",
                      flush=True)
                batch = None
            # All samples this rank raised in __getitem__ (transient
            # zarr/S3 read errors). Coordinate with other ranks so
            # that if ANY rank is empty, all skip this step.
            local_read_err = int(batch is None)
            global_any_read_err = _allreduce_max(
                local_read_err, world_size, device,
            )
            if global_any_read_err:
                if local_read_err:
                    n_read_err += 1
                n_skipped += 1
                if is_main:
                    train_iter.set_postfix(
                        loss="---",
                        skip=f"{100.0 * n_skipped / max(n_seen + 1, 1):.1f}%",
                        rderr=n_read_err,
                        merge=f"{100.0 * (1.0 - n_surfaces_post / max(n_surfaces_pre, 1)):.1f}%",
                    )
                continue

            # Geometry-level spatial augmentation: flips the image,
            # surface_masks, padding_mask and surface_geometry, then
            # re-splats raw_normals + normals_valid from the
            # augmented geometry so downstream target derivation sees
            # a fully consistent frame. Skipped entirely when the
            # sampled transform is identity.
            batch = augment_batch_inplace(batch)
            image = batch["image"].to(device, non_blocking=True)

            # Compute targets on GPU from (post-aug) surface masks.
            (
                targets, validity,
                dir_sparse_mask, dir_dense_mask, dir_axis_weight,
                _merge_groups, _merged_masks, _merged_chain_info,
            ) = compute_batch_targets(
                batch, device,
                same_surface_threshold=same_surface_threshold,
            )

            # Scale aug: pool GT from crop_size → crop_size/f, then paste
            # into a crop_size-shaped tensor at a random offset.  The
            # surrounding region has validity=0 → unsupervised context.
            if use_aug:
                sf = scale_aug_factor
                v = validity  # (B, 1, Z, Y, X)
                # If any spatial dim is smaller than the pool kernel,
                # pad to at least sf so avg_pool3d doesn't crash.
                pad_needed = [max(0, sf - v.shape[d]) for d in (4, 3, 2)]  # x, y, z
                if any(p > 0 for p in pad_needed):
                    _p = (0, pad_needed[0], 0, pad_needed[1], 0, pad_needed[2])
                    v = F.pad(v, _p, value=0)
                    targets = F.pad(targets, _p, value=0)
                    dir_sparse_mask = F.pad(dir_sparse_mask, _p, value=0)
                    dir_dense_mask = F.pad(dir_dense_mask, _p, value=0)
                    dir_axis_weight = F.pad(dir_axis_weight, _p, value=0)
                    validity = v
                v_count = F.avg_pool3d(v, kernel_size=sf, stride=sf)

                # Valid-pool value tensors (avg only over valid voxels).
                tgt_sm = (
                    F.avg_pool3d(targets * v, sf, sf)
                    / v_count.clamp(min=1e-8)
                )
                daw_sm = (
                    F.avg_pool3d(dir_axis_weight * v, sf, sf)
                    / v_count.clamp(min=1e-8)
                )
                # Max-pool binary masks (any-valid / any-present).
                val_sm = F.max_pool3d(v, sf, sf)
                dsp_sm = F.max_pool3d(dir_sparse_mask, sf, sf)
                ddn_sm = F.max_pool3d(dir_dense_mask, sf, sf)

                # Paste pooled block at the offset chosen by the dataset
                # (aligned with the CT read so GT and CT match).
                # Transform offset by the spatial augmentation (flips +
                # rot90) that was applied to the image and GT tensors.
                sm_shape = tgt_sm.shape[2:]  # (Z/f, Y/f, X/f)
                full_shape = image.shape[2:]  # (Z, Y, X) = crop_size
                off = list(batch["patch_info"][0]["scale_aug_offset"])
                aug = batch.get("_aug")
                if aug:
                    fz, fy, fx, k = aug
                    if fz:
                        off[0] = full_shape[0] - sm_shape[0] - off[0]
                    if fy:
                        off[1] = full_shape[1] - sm_shape[1] - off[1]
                    if fx:
                        off[2] = full_shape[2] - sm_shape[2] - off[2]
                    # rot90 k times in (Y, X) plane:
                    # k=1: (oy, ox) → (S_x - sm_x - ox, oy)
                    # k=2: (oy, ox) → (S_y - sm_y - oy, S_x - sm_x - ox)
                    # k=3: (oy, ox) → (ox, S_y - sm_y - oy)
                    for _ in range(k % 4):
                        oy, ox = off[1], off[2]
                        sy_full, sx_full = full_shape[1], full_shape[2]
                        sm_y, sm_x = sm_shape[1], sm_shape[2]
                        off[1] = sx_full - sm_x - ox
                        off[2] = oy
                        # After rot90, sm_shape Y and X swap for
                        # the next rotation step.  But since
                        # crop_size is cubic and sm_shape is cubic
                        # (crop_size/f), they stay the same.
                sz, sy, sx = (
                    slice(off[0], off[0] + sm_shape[0]),
                    slice(off[1], off[1] + sm_shape[1]),
                    slice(off[2], off[2] + sm_shape[2]),
                )

                B_ = image.shape[0]
                targets      = torch.zeros(B_, 8, *full_shape, device=device)
                validity     = torch.zeros(B_, 1, *full_shape, device=device)
                dir_sparse_mask = torch.zeros(B_, 1, *full_shape, device=device)
                dir_dense_mask  = torch.zeros(B_, 1, *full_shape, device=device)
                dir_axis_weight = torch.zeros(B_, 6, *full_shape, device=device)

                # grad_mag = 1/spacing was computed at full res. At the
                # pooled (coarser) scale, distances are f× smaller in
                # voxel units, so grad_mag should be f× larger.
                tgt_sm[:, 1:2] *= sf

                targets[:, :, sz, sy, sx]         = tgt_sm
                validity[:, :, sz, sy, sx]        = val_sm
                dir_sparse_mask[:, :, sz, sy, sx] = dsp_sm
                dir_dense_mask[:, :, sz, sy, sx]  = ddn_sm
                dir_axis_weight[:, :, sz, sy, sx] = daw_sm

                del tgt_sm, val_sm, dsp_sm, ddn_sm, daw_sm
                n_scale_aug += 1

            n_seen += 1
            B = batch["image"].shape[0]
            for mg in _merge_groups:
                n_surfaces_pre += len(mg)
                n_surfaces_post += len(set(int(x) for x in mg))

            local_validity_empty = int(validity.sum().item() == 0)
            global_any_empty = _allreduce_max(
                local_validity_empty, world_size, device,
            )
            if global_any_empty:
                # In single-GPU mode, just skip. In DDP, an empty rank
                # would desync allreduce with active ranks — so we
                # collectively skip when ANY rank has no valid voxels.
                n_skipped += 1
                if is_main:
                    train_iter.set_postfix(
                        loss="---",
                        skip=f"{100.0 * n_skipped / max(n_seen, 1):.1f}%",
                        merge=f"{100.0 * (1.0 - n_surfaces_post / max(n_surfaces_pre, 1)):.1f}%",
                    )
                continue

            # Intensity augmentation on CT only (spatial is already done).
            image = augment_intensity(image)

            # --- Multi-scale refinement path ---
            if refine:
                mode_idx, scale_seq = _pick_refine_mode(
                    refine_available_offsets, rank, world_size,
                    device, is_dist,
                )
                n_passes = len(scale_seq)
                B_eff = image.shape[0] if n_passes == 1 else max(1, image.shape[0] // 2)
                # Slice batch when multi-pass
                if B_eff < image.shape[0]:
                    image = image[:B_eff]
                    targets = targets[:B_eff]
                    validity = validity[:B_eff]
                    dir_sparse_mask = dir_sparse_mask[:B_eff]
                    dir_dense_mask = dir_dense_mask[:B_eff]
                    dir_axis_weight = dir_axis_weight[:B_eff]
                    batch = _filter_batch(batch, list(range(B_eff)))

                # Save scale-0 GT for resampling to other scales
                targets_0 = targets
                validity_0 = validity
                dsm_0 = dir_sparse_mask
                ddm_0 = dir_dense_mask
                daw_0 = dir_axis_weight

                total_loss = image.new_zeros(1).squeeze()
                prev_pred = None
                prev_offset = None
                P = patch_size

                with torch.amp.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_autocast):
                    for pass_i, (offset, has_prior) in enumerate(scale_seq):
                        # Read CT at this offset
                        if offset == 0:
                            ct = image
                        else:
                            ct = _read_ct_at_offset(
                                batch["patch_info"][:B_eff],
                                refine_vol_groups, offset, P, device,
                            )
                            ct = augment_intensity(ct)

                        # Build prior
                        if has_prior and prev_pred is not None:
                            prior = _resample_prior(
                                prev_pred.detach(), prev_offset, offset, P,
                            )
                        else:
                            prior = None
                        model_in = _build_refine_input(ct, prior, offset)

                        # Forward
                        results = model(model_in)
                        raw_pred = results["output"]
                        pred_pass = torch.sigmoid(raw_pred) if output_sigmoid else raw_pred

                        # GT at this scale
                        tgt, val, dsm, ddm, daw = _get_gt_at_offset(
                            batch, targets_0, validity_0, dsm_0, ddm_0, daw_0,
                            offset, device,
                            compute_batch_targets_fn=compute_batch_targets,
                            same_surface_threshold=same_surface_threshold,
                        )

                        # Loss (same structure as non-refine)
                        cos_mask_p = val
                        cos_mag_w = tgt[:, 1:2] * 20.0
                        l_cos = scale_loss_mse(pred_pass[:, 0:1], tgt[:, 0:1], mask=cos_mask_p, weight=cos_mag_w)
                        l_mag = scale_loss_l1(pred_pass[:, 1:2], tgt[:, 1:2], mask=cos_mask_p, weight=cos_mag_w)
                        dir_mask_p = ((dsm + ddm) > 0.5).float()
                        l_dir_s = scale_loss_mse(pred_pass[:, 2:8], tgt[:, 2:8], mask=dsm, weight=daw)
                        l_dir_d = scale_loss_mse(pred_pass[:, 2:8], tgt[:, 2:8], mask=ddm, weight=daw)
                        l_smooth = direction_smoothness_loss(pred_pass[:, 2:8], dir_mask_p)

                        pass_loss = (
                            w_cos * l_cos
                            + w_mag * l_mag
                            + w_dir * l_dir_s
                            + w_dir_dense * l_dir_d
                            + w_smooth * l_smooth
                        )
                        total_loss = total_loss + pass_loss

                        prev_pred = pred_pass
                        prev_offset = offset

                    # Set variables for logging (use last pass values)
                    pred = prev_pred
                    loss = total_loss
                    loss_cos = l_cos
                    loss_mag = l_mag
                    loss_dir_sparse = l_dir_s
                    loss_dir_dense = l_dir_d
                    loss_smooth = l_smooth
                    loss_dist = pred.new_zeros(1).squeeze()
                    loss_fft = pred.new_zeros(1).squeeze()
                    cos_mask = val
                    dir_sparse_mask = dsm
                    targets = tgt
                    targets_original = targets_0
                    targets_deformed = None
                    cos_mask_original = validity_0
                    with torch.no_grad():
                        t_pred = decode_to_tensor(
                            pred[:, 2:8].detach().permute(1, 0, 2, 3, 4)
                        )
                        t_gt_batch = decode_to_tensor(
                            tgt[:, 2:8].detach().permute(1, 0, 2, 3, 4)
                        )
                        dir_angle_deg_sparse = tensor_unsigned_angle_deg(
                            t_pred, t_gt_batch, mask=dsm[:, 0],
                        ).item()

            # Forward pass (skipped when refine block already ran above)
            if not refine:
              with torch.amp.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_autocast):
                results = model(image)
                raw_pred = results["output"]
                pred = torch.sigmoid(raw_pred) if output_sigmoid else raw_pred  # (B, 8, Z, Y, X)

                cos_mask = validity
                dir_mask = ((dir_sparse_mask + dir_dense_mask) > 0.5).float()

                # Per-sample grad-mag screen: flag any sample with
                # per-sample scale-space L1 >= 1 as an outlier, log it,
                # and drop it from the batch for this step. Runs under
                # no_grad with each scalar eagerly moved to a Python
                # float — nothing is retained past the with block, so
                # the screen adds no autograd graph to the forward.
                B = pred.shape[0]
                with torch.no_grad():
                    per_mag_vals: list[float] = []
                    for b in range(B):
                        v = scale_loss_l1(
                            pred[b:b+1, 1:2].detach(),
                            targets[b:b+1, 1:2],
                            mask=cos_mask[b:b+1],
                        )
                        per_mag_vals.append(float(v.item()))
                        del v
                local_above = [b for b, v in enumerate(per_mag_vals) if v >= 1.0]
                local_above_cnt = len(local_above)

                # Sync (above, total) across ranks so every rank
                # updates the hysteresis state identically.
                if is_dist:
                    cnt_t = torch.tensor(
                        [local_above_cnt, B],
                        device=device, dtype=torch.long,
                    )
                    dist.all_reduce(cnt_t, op=dist.ReduceOp.SUM)
                    global_above = int(cnt_t[0].item())
                    global_total = int(cnt_t[1].item())
                else:
                    global_above = local_above_cnt
                    global_total = B

                # Extend the 100-sample ring buffer: append 1 per
                # above-threshold sample and 0 per below. Order within
                # a step doesn't matter — only window counts drive
                # the state transitions.
                global_below = global_total - global_above
                himag_flags.extend([1] * global_above + [0] * global_below)
                window_above = sum(himag_flags)
                prev_enabled = himag_enabled
                if not himag_filter:
                    pass  # permanently disabled via --no-himag-filter
                elif himag_enabled:
                    # Disable when ≥75% of the 100-sample window is
                    # above threshold — the filter is starving the
                    # loss and needs to let the model recover.
                    if (len(himag_flags) == HIMAG_WINDOW
                            and window_above >= HIMAG_DISABLE_ABOVE):
                        himag_enabled = False
                else:
                    # Re-enable once <50% of the window is above.
                    if (len(himag_flags) == HIMAG_WINDOW
                            and window_above < HIMAG_ENABLE_ABOVE):
                        himag_enabled = True
                if is_main and prev_enabled != himag_enabled:
                    state = "ENABLED" if himag_enabled else "DISABLED"
                    print(
                        f"{TAG} hi-mag filter {state} at step "
                        f"{global_step} (window above "
                        f"{window_above}/{len(himag_flags)})",
                        flush=True,
                    )

                bad = local_above if himag_enabled else []
                if himag_enabled:
                    for b in bad:
                        pi = batch["patch_info"][b]
                        seg = pi.get("segment_uuid", "?")
                        idx_b = pi.get("idx", "?")
                        ds_name = pi.get("dataset_name", "?")
                        ds_idx = pi.get("dataset_idx", "?")
                        bbox = pi.get("world_bbox", ())
                        bbox_str = ",".join(str(int(v)) for v in bbox) if bbox else "?"
                        print(
                            f"{TAG} hi-mag skip step={global_step} "
                            f"ds=[{ds_idx}]{ds_name} idx={idx_b} "
                            f"seg={seg} bbox={bbox_str} "
                            f"mag_loss={per_mag_vals[b]:.4f}",
                            flush=True,
                        )
                n_hi_mag += len(bad)
                local_all_bad = int(len(bad) == B)
                global_any_all_bad = _allreduce_max(
                    local_all_bad, world_size, device,
                )
                if global_any_all_bad:
                    if is_main:
                        train_iter.set_postfix(
                            loss="---",
                            skip=f"{100.0 * n_skipped / max(n_seen, 1):.1f}%",
                            himag=n_hi_mag,
                            hifilt="on" if himag_enabled else "OFF",
                            merge=f"{100.0 * (1.0 - n_surfaces_post / max(n_surfaces_pre, 1)):.1f}%",
                        )
                    continue
                if bad:
                    keep_list = [b for b in range(B) if b not in bad]
                    keep = torch.tensor(
                        keep_list,
                        device=pred.device, dtype=torch.long,
                    )
                    pred = pred.index_select(0, keep)
                    targets = targets.index_select(0, keep)
                    cos_mask = cos_mask.index_select(0, keep)
                    dir_mask = dir_mask.index_select(0, keep)
                    dir_sparse_mask = dir_sparse_mask.index_select(0, keep)
                    dir_dense_mask = dir_dense_mask.index_select(0, keep)
                    dir_axis_weight = dir_axis_weight.index_select(0, keep)
                    image = image.index_select(0, keep)
                    batch = _filter_batch(batch, keep_list)

                # --- GT deformation inner loop ---
                targets_original = targets
                cos_mask_original = cos_mask
                targets_deformed = None
                if deform_store is not None and not use_aug:
                    batch_global_idxs = [
                        pi["global_idx"] for pi in batch["patch_info"]
                    ]
                    deform_batch = deform_store.get(batch_global_idxs).to(device)
                    aug = batch.get("_aug", (False, False, False, 0))
                    fz, fy, fx, kk = aug
                    deform_aug = _augment_deform(deform_batch, fz, fy, fx, kk)
                    deform_aug.requires_grad_(True)

                    deform_opt = _deform_inner_loop(
                        pred_cos_gm=pred[:, 0:2].detach(),
                        targets=targets,
                        validity=cos_mask,
                        deform=deform_aug,
                        n_iters=deform_inner_iters,
                        inner_lr=deform_inner_lr,
                        max_frac=deform_max_frac,
                        scale_loss_mse_fn=scale_loss_mse,
                        scale_loss_l1_fn=scale_loss_l1,
                    )

                    # Apply final warp to cos/grad_mag targets
                    with torch.no_grad():
                        G = deform_opt.shape[-1]
                        normal_dir = _compute_normal_field(
                            targets[:, 0:1].float(), G,
                        )
                        disp_3d = deform_opt * normal_dir
                        Z, Y, X = targets.shape[2:]
                        df = F.interpolate(
                            disp_3d, size=(Z, Y, X),
                            mode="trilinear", align_corners=False,
                        )
                        warped_cg, warped_v = _apply_warp(
                            targets[:, 0:2], cos_mask, df,
                        )
                        targets_deformed = targets.clone()
                        targets_deformed[:, 0:2] = warped_cg
                        targets = targets_deformed
                        cos_mask = warped_v

                    # Store back (un-augment first)
                    deform_preaugm = _unaugment_deform(deform_opt, fz, fy, fx, kk)
                    deform_store.put(batch_global_idxs, deform_preaugm)

                # Per-channel losses — weighted by grad_mag GT (closer surfaces → higher weight)
                # Normalize so 20 vx spacing (grad_mag=0.05) → weight 1.0
                # For scale-aug: grad_mag was already scaled by f, so
                # divide weight by f to keep per-voxel contribution
                # comparable to non-aug samples.
                _mag_weight_scale = (1.0 / scale_aug_factor) if use_aug else 1.0
                cos_mag_weight = targets[:, 1:2] * 20.0 * _mag_weight_scale
                loss_cos = scale_loss_mse(pred[:, 0:1], targets[:, 0:1], mask=cos_mask, weight=cos_mag_weight)
                loss_mag = scale_loss_l1(pred[:, 1:2], targets[:, 1:2], mask=cos_mask, weight=cos_mag_weight)
                loss_dir_sparse = scale_loss_mse(
                    pred[:, 2:8], targets[:, 2:8],
                    mask=dir_sparse_mask, weight=dir_axis_weight,
                )
                loss_dir_dense = scale_loss_mse(
                    pred[:, 2:8], targets[:, 2:8],
                    mask=dir_dense_mask, weight=dir_axis_weight,
                )
                loss_smooth = direction_smoothness_loss(pred[:, 2:8], dir_mask)
                loss_dist = (
                    sorted_distribution_loss(
                        pred[:, 0:1], targets_original[:, 0:1],
                        cos_mask, targets_original[:, 1:2],
                    )
                    if w_dist > 0 else pred.new_zeros(1).squeeze()
                )
                loss_fft = (
                    fft_magnitude_loss(
                        pred[:, 0:1], targets_original[:, 0:1],
                        cos_mask_original, targets_original[:, 1:2],
                    )
                    if w_fft > 0 else pred.new_zeros(1).squeeze()
                )
                loss = (
                    w_cos * loss_cos
                    + w_mag * loss_mag
                    + w_dir * loss_dir_sparse
                    + w_dir_dense * loss_dir_dense
                    + w_smooth * loss_smooth
                    + w_dist * loss_dist
                    + w_fft * loss_fft
                )

                # Sign-invariant unsigned angular error on sparse
                # direction voxels — reported in degrees for an
                # interpretable "how far off is the model's normal"
                # number, computed via Frobenius inner product on
                # tensor moments (|cos θ|² = ⟨N_p, N_g⟩_F) so that
                # n and -n are treated as the same direction.
                # decode_to_tensor expects channel-first (C, ...);
                # pred is (B, 6, Z, Y, X), so permute first.
                with torch.no_grad():
                    t_pred = decode_to_tensor(
                        pred[:, 2:8].detach().permute(1, 0, 2, 3, 4)
                    )
                    # Use the (pooled+pasted for scale-aug) direction
                    # targets decoded back to tensor form — always
                    # spatially aligned with pred and dir_sparse_mask.
                    t_gt_batch = decode_to_tensor(
                        targets[:, 2:8].detach().permute(1, 0, 2, 3, 4)
                    )
                    dir_angle_deg_sparse = tensor_unsigned_angle_deg(
                        t_pred,
                        t_gt_batch,
                        mask=dir_sparse_mask[:, 0],
                    ).item()

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            if global_step < warmup_steps:
                warmup_scheduler.step()
            else:
                cosine_scheduler.step()

            if pause_server:
                pause_server.check(_gpu_offload, _gpu_reload)

            epoch_losses.append(loss.item())
            if not math.isfinite(loss.item()):
                print(f"{TAG} NaN/Inf at step {global_step}: loss={loss.item()}", flush=True)

            running = sum(epoch_losses[-20:]) / min(len(epoch_losses), 20)
            if is_main:
                postfix = dict(
                    loss=f"{loss.item():.4f}",
                    avg20=f"{running:.4f}",
                    skip=f"{100.0 * n_skipped / max(n_seen, 1):.1f}%",
                    rderr=n_read_err,
                    himag=n_hi_mag,
                    hifilt="on" if himag_enabled else "OFF",
                    merge=f"{100.0 * (1.0 - n_surfaces_post / max(n_surfaces_pre, 1)):.1f}%",
                )
                if n_scale_aug > 0:
                    postfix["saug"] = n_scale_aug
                train_iter.set_postfix(**postfix)

            if global_step % 10 == 0:
                log_scalar("train/loss", loss.item(), global_step)
                log_scalar("train/loss_cos", loss_cos.item(), global_step)
                log_scalar("train/loss_mag", loss_mag.item(), global_step)
                log_scalar(
                    "train/loss_dir_sparse", loss_dir_sparse.item(), global_step,
                )
                log_scalar(
                    "train/loss_dir_dense", loss_dir_dense.item(), global_step,
                )
                log_scalar(
                    "train/loss_smooth", loss_smooth.item(), global_step,
                )
                if w_dist > 0:
                    log_scalar(
                        "train/loss_dist", loss_dist.item(), global_step,
                    )
                if w_fft > 0:
                    log_scalar(
                        "train/loss_fft", loss_fft.item(), global_step,
                    )
                log_scalar(
                    "train/dir_angle_deg_sparse",
                    dir_angle_deg_sparse,
                    global_step,
                )
                log_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
                if deform_store is not None:
                    # Log deformation magnitude stats (from stored values,
                    # regardless of whether this step applied deformation)
                    batch_global_idxs_log = [
                        pi["global_idx"] for pi in batch["patch_info"]
                    ]
                    _d = deform_store.get(batch_global_idxs_log)
                    _d_abs = _d.abs()  # (B, 1, G, G, G) scalar offset
                    log_scalar("train/deform_mean", _d_abs.mean().item(), global_step)
                    log_scalar("train/deform_max", _d_abs.max().item(), global_step)
                    del _d, _d_abs
                if n_scale_aug > 0:
                    total_samples = n_seen * B
                    log_scalar(
                        "train/scale_aug_frac",
                        n_scale_aug / max(total_samples, 1),
                        global_step,
                    )
                if refine:
                    log_scalar("train/refine_mode", mode_idx, global_step)

            if global_step % 100 == 0:
                _log_vis(log_images, "train", image, pred, targets, cos_mask, global_step,
                         targets_original=targets_original if targets_deformed is not None else None)
                aug = batch.get("_aug")
                if aug and is_main:
                    fz, fy, fx, kk = aug
                    print(
                        f"{TAG} vis step={global_step} "
                        f"aug=(flip_z={fz}, flip_y={fy}, "
                        f"flip_x={fx}, k={kk})",
                        flush=True,
                    )
                    log_scalar("train/aug_k", kk, global_step)

            if global_step % vis_interval_steps == 0:
                _log_full_vis(batch, "train", global_step)

            global_step += 1

            # Periodic validation mid-epoch so we see val trajectory
            # at a much higher resolution than once-per-epoch.
            if global_step > 0 and global_step % val_every_steps == 0:
                _vis_batch: list = []
                last_val_loss = _evaluate(
                    model, val_loader, scale_loss_mse, scale_loss_l1,
                    device, log_scalar, log_images, global_step,
                    w_cos, w_mag, w_dir,
                    w_dir_dense=w_dir_dense, w_smooth=w_smooth,
                    amp_dtype=amp_dtype, use_autocast=use_autocast,
                    verbose=verbose and is_main,
                    same_surface_threshold=same_surface_threshold,
                    output_sigmoid=output_sigmoid,
                    vis_batch_out=_vis_batch,
                    device_type=device_type,
                    world_size=world_size,
                    is_main=is_main,
                    deform_store=deform_store,
                    vis_samples=VIS_SAMPLES,
                    refine=refine,
                )
                if _vis_batch:
                    _log_full_vis(_vis_batch[0], "val", global_step)
                _vis_batch.clear()
                model.train()
                if is_main:
                    ckpt_data = {
                        "state_dict": model_core.state_dict(),
                        "norm_type": norm_type,
                        "upsample_mode": upsample_mode,
                        "output_sigmoid": output_sigmoid,
                        "precision": precision,
                        "patch_size": patch_size,
                        "in_channels": 11 if refine else 1,
                        "out_channels": 8,
                        "refine": refine,
                    }
                    torch.save(ckpt_data, run_dir / "model_current.pt")
                    if deform_store is not None:
                        deform_store.flush()
                    is_best = last_val_loss < best_val_loss
                    if is_best:
                        best_val_loss = last_val_loss
                        torch.save(ckpt_data, run_dir / "model_best.pt")
                    train_iter.write(
                        f"{TAG} step {global_step}  val={last_val_loss:.4f}"
                        + ("  [best]" if is_best else "")
                    )
                if is_dist:
                    dist.barrier()

        mean_train = sum(epoch_losses) / max(len(epoch_losses), 1)
        if is_main:
            print(
                f"epoch {epoch + 1}/{epochs}  "
                f"train={mean_train:.4f}  val={last_val_loss:.4f}  "
                f"lr={optimizer.param_groups[0]['lr']:.2e}",
                flush=True,
            )

    # Final checkpoint at end of training
    if is_main:
        ckpt_data = {
            "state_dict": model_core.state_dict(),
            "norm_type": norm_type,
            "upsample_mode": upsample_mode,
            "output_sigmoid": output_sigmoid,
            "precision": precision,
            "patch_size": patch_size,
            "in_channels": 11 if refine else 1,
            "out_channels": 8,
            "refine": refine,
        }
        torch.save(ckpt_data, run_dir / "model_current.pt")
        if last_val_loss < best_val_loss:
            best_val_loss = last_val_loss
            torch.save(ckpt_data, run_dir / "model_best.pt")
        if deform_store is not None:
            deform_store.flush()
        print(f"{TAG} saved final checkpoint (val={last_val_loss:.4f})", flush=True)
    if is_dist:
        dist.barrier()

    if pause_server is not None:
        pause_server.close()
    if writer is not None:
        writer.close()
    if wandb_run is not None:
        wandb_run.finish()
    if is_main:
        print(f"{TAG} done. Logs & checkpoints in {run_dir}", flush=True)
    if is_dist:
        dist.barrier()
        dist.destroy_process_group()


def _evaluate(
    model, loader, scale_loss_mse, scale_loss_l1,
    device, log_scalar, log_images, global_step,
    w_cos, w_mag, w_dir,
    w_dir_dense: float = 0.1,
    w_smooth: float = 0.1,
    amp_dtype=torch.bfloat16, use_autocast=True,
    verbose: bool = False,
    same_surface_threshold: float | None = None,
    output_sigmoid: bool = False,
    vis_batch_out: list | None = None,
    device_type: str = "cuda",
    world_size: int = 1,
    is_main: bool = True,
    deform_store: "DeformationStore | None" = None,
    deform_inner_iters: int = 100,
    deform_inner_lr: float = 1000.0,
    deform_max_frac: float = 0.3,
    vis_samples: int = 8,
    w_dist: float = 0.0,
    w_fft: float = 0.0,
    refine: bool = False,
):
    model.eval()
    losses = []
    losses_cos: list[float] = []
    losses_mag: list[float] = []
    losses_dir_sparse: list[float] = []
    losses_dir_dense: list[float] = []
    losses_smooth: list[float] = []
    angles_sparse_deg: list[float] = []
    vis_done = False
    _vis_acc: list[tuple] = []  # accumulated on every rank, gathered to rank 0

    val_iter = loader
    if verbose:
        val_iter = tqdm(loader, desc="val", dynamic_ncols=True, leave=False)

    with torch.no_grad(), torch.amp.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_autocast):
        for batch in val_iter:
            if batch is None:
                continue
            image = batch["image"].to(device)
            (
                targets, validity,
                dir_sparse_mask, dir_dense_mask, dir_axis_weight,
                _merge_groups, _merged_masks, _merged_chain_info,
            ) = compute_batch_targets(
                batch, device,
                same_surface_threshold=same_surface_threshold,
            )

            # Always run forward pass even if validity is empty — DDP
            # broadcasts buffers on every forward, so all ranks must
            # call model() the same number of times.
            if refine:
                model_in = _build_refine_input(image, None, 0)  # scale 0, no prior
            else:
                model_in = image
            results = model(model_in)
            raw_pred = results["output"]
            pred = torch.sigmoid(raw_pred) if output_sigmoid else raw_pred

            if validity.sum() == 0:
                continue

            cos_mask = validity
            dir_mask = ((dir_sparse_mask + dir_dense_mask) > 0.5).float()

            # Deformation inner loop (same as training)
            targets_original = targets
            cos_mask_original = cos_mask
            if deform_store is not None:
                batch_gidxs = [pi["global_idx"] for pi in batch["patch_info"]]
                deform_batch = deform_store.get(batch_gidxs).to(device)
                deform_batch.requires_grad_(True)

                with torch.enable_grad():
                    deform_opt = _deform_inner_loop(
                        pred[:, 0:2].detach(),
                        targets, cos_mask,
                        deform_batch,
                        n_iters=deform_inner_iters,
                        inner_lr=deform_inner_lr,
                        max_frac=deform_max_frac,
                        scale_loss_mse_fn=scale_loss_mse,
                        scale_loss_l1_fn=scale_loss_l1,
                    )

                G = deform_opt.shape[-1]
                normal_dir = _compute_normal_field(
                    targets[:, 0:1].float(), G,
                )
                disp_3d = deform_opt * normal_dir
                Z, Y, X = targets.shape[2:]
                df = F.interpolate(
                    disp_3d, size=(Z, Y, X),
                    mode="trilinear", align_corners=False,
                )
                warped_cg, warped_v = _apply_warp(
                    targets[:, 0:2], cos_mask, df,
                )
                targets = targets.clone()
                targets[:, 0:2] = warped_cg
                cos_mask = warped_v

                deform_store.put(batch_gidxs, deform_opt)

            loss_cos = scale_loss_mse(pred[:, 0:1], targets[:, 0:1], mask=cos_mask)
            loss_mag = scale_loss_l1(pred[:, 1:2], targets[:, 1:2], mask=cos_mask)
            loss_dir_sparse = scale_loss_mse(
                pred[:, 2:8], targets[:, 2:8],
                mask=dir_sparse_mask, weight=dir_axis_weight,
            )
            loss_dir_dense = scale_loss_mse(
                pred[:, 2:8], targets[:, 2:8],
                mask=dir_dense_mask, weight=dir_axis_weight,
            )
            loss_smooth = direction_smoothness_loss(pred[:, 2:8], dir_mask)
            loss_dist = (
                sorted_distribution_loss(
                    pred[:, 0:1], targets_original[:, 0:1],
                    cos_mask, targets_original[:, 1:2],
                )
                if w_dist > 0 else pred.new_zeros(1).squeeze()
            )
            loss_fft = (
                fft_magnitude_loss(
                    pred[:, 0:1], targets_original[:, 0:1],
                    cos_mask_original, targets_original[:, 1:2],
                )
                if w_fft > 0 else pred.new_zeros(1).squeeze()
            )
            loss = (
                w_cos * loss_cos
                + w_mag * loss_mag
                + w_dir * loss_dir_sparse
                + w_dir_dense * loss_dir_dense
                + w_smooth * loss_smooth
                + w_dist * loss_dist
                + w_fft * loss_fft
            )
            losses.append(loss.item())
            losses_cos.append(loss_cos.item())
            losses_mag.append(loss_mag.item())
            losses_dir_sparse.append(loss_dir_sparse.item())
            losses_dir_dense.append(loss_dir_dense.item())
            losses_smooth.append(loss_smooth.item())

            # Sign-invariant unsigned angular error on sparse voxels.
            # decode_to_tensor expects channel-first (C, ...); pred is
            # (B, 6, Z, Y, X), so permute to (6, B, Z, Y, X) first.
            t_pred_val = decode_to_tensor(
                pred[:, 2:8].permute(1, 0, 2, 3, 4)
            )
            t_gt_val = batch["tensor_moments"].to(
                device, non_blocking=True,
            ).permute(1, 0, 2, 3, 4)
            ang_deg = tensor_unsigned_angle_deg(
                t_pred_val,
                t_gt_val,
                mask=dir_sparse_mask[:, 0],
            ).item()
            angles_sparse_deg.append(ang_deg)

            if verbose:
                val_iter.set_postfix(loss=f"{loss.item():.4f}")

            # Accumulate vis samples on every rank (gathered to rank 0 below).
            if not vis_done and len(_vis_acc) < vis_samples:
                _vis_acc.append((
                    image.detach().cpu(),
                    pred.detach().cpu(),
                    targets.detach().cpu(),
                    cos_mask.detach().cpu(),
                    targets_original.detach().cpu(),
                ))
                if vis_batch_out is not None:
                    vis_batch_out.append(batch)
                if len(_vis_acc) >= vis_samples:
                    vis_done = True

    # Gather vis samples from all ranks to rank 0.
    if world_size > 1:
        # Each rank sends its list of tuples; rank 0 collects all.
        all_vis: list[list[tuple]] | None = [None] * world_size if is_main else None
        dist.gather_object(_vis_acc, all_vis, dst=0)
        if is_main and all_vis is not None:
            _vis_acc = []
            for rank_vis in all_vis:
                if rank_vis:
                    _vis_acc.extend(rank_vis)
            _vis_acc = _vis_acc[:vis_samples]

    if _vis_acc and is_main:
        _log_vis(
            log_images, "val",
            torch.cat([a[0] for a in _vis_acc], dim=0).to(device),
            torch.cat([a[1] for a in _vis_acc], dim=0).to(device),
            torch.cat([a[2] for a in _vis_acc], dim=0).to(device),
            torch.cat([a[3] for a in _vis_acc], dim=0).to(device),
            global_step,
            targets_original=(
                torch.cat([a[4] for a in _vis_acc], dim=0).to(device)
                if deform_store is not None else None
            ),
        )

    # Per-rank means → cross-rank average. Each rank's val shard is
    # roughly the same size (DistributedSampler pads to equal length).
    n = max(len(losses), 1)
    local_means = {
        "val/loss": sum(losses) / n,
        "val/loss_cos": sum(losses_cos) / n,
        "val/loss_mag": sum(losses_mag) / n,
        "val/loss_dir_sparse": sum(losses_dir_sparse) / n,
        "val/loss_dir_dense": sum(losses_dir_dense) / n,
        "val/loss_smooth": sum(losses_smooth) / n,
    }
    global_means = {
        k: _allreduce_mean(v, world_size, device)
        for k, v in local_means.items()
    }
    for k, v in global_means.items():
        log_scalar(k, v, global_step)
    # Always call allreduce on every rank to stay in sync, even when
    # the local shard happened to skip the angle metric.
    local_ang = (
        sum(angles_sparse_deg) / len(angles_sparse_deg)
        if angles_sparse_deg else 0.0
    )
    global_ang = _allreduce_mean(local_ang, world_size, device)
    if angles_sparse_deg or world_size > 1:
        log_scalar("val/dir_angle_deg_sparse", global_ang, global_step)
    return global_means["val/loss"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train 3D UNet on tifxyz-derived lasagna labels.",
    )
    parser.add_argument("--config", type=str, required=True,
                        help="JSON config with datasets list.")
    parser.add_argument("--log-dir", type=str, default="runs/tifxyz3d")
    parser.add_argument("--run-name", type=str, default="tifxyz")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--patch-size", type=int, default=128,
                        help="CT read patch size (model input).")
    parser.add_argument("--label-patch-size", type=int, default=None,
                        help="GT region size inside the CT crop. "
                             "Defaults to --patch-size. When smaller, "
                             "the label region is placed at a random "
                             "offset inside the larger CT crop at train "
                             "time (up to L/2 cropping per edge); val "
                             "uses the centered position.")
    parser.add_argument("--model-patch-size", type=int, default=None,
                        help="Patch size used only for model architecture "
                             "autoconfigure (stage count). Defaults to "
                             "--patch-size. Set this to the checkpoint's "
                             "patch size when continuing training at a "
                             "different training patch size.")
    parser.add_argument("--w-cos", type=float, default=1.0)
    parser.add_argument("--w-mag", type=float, default=1.0)
    parser.add_argument("--w-dir", type=float, default=1.0,
                        help="Weight on the sparse (original splat) "
                             "direction loss.")
    parser.add_argument("--w-dir-dense", type=float, default=0.1,
                        help="Weight on the DT-blended dense "
                             "direction loss. Default 0.1 so training "
                             "leans on the hard samples.")
    parser.add_argument("--w-smooth", type=float, default=0.1,
                        help="Weight on L1 spatial-gradient smoothness "
                             "regularization of predicted direction "
                             "channels inside the validity bracket.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--weights", type=str, default=None,
                        help="Checkpoint to resume from.")
    parser.add_argument("--reinit-decoder-scales", type=int, default=0,
                        help="After loading checkpoint, re-initialize the "
                             "last N decoder stages (highest-resolution "
                             "conv blocks, upsample modules, seg heads, "
                             "and encoder stem). Use with --weights.")
    parser.add_argument("--norm-type", type=str, default="none",
                        choices=["instance", "group", "none"])
    parser.add_argument("--upsample-mode", type=str, default="trilinear",
                        choices=["transpconv", "trilinear", "pixelshuffle"])
    parser.add_argument("--output-sigmoid", action="store_true", default=False,
                        help="Apply torch.sigmoid to model output. Off by "
                             "default, matching train_unet_3d.")
    parser.add_argument("--precision", type=str, default="bf16",
                        choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show per-sample progress bar with running loss.")
    parser.add_argument("--wandb", action="store_true", default=False,
                        help="Enable Weights & Biases logging (in "
                             "addition to TensorBoard).")
    parser.add_argument("--wandb-project", type=str, default=None,
                        help="W&B project name (default: lasagna-tifxyz).")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="W&B entity/user (default: user's default).")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                        help="W&B run name (default: <timestamp>_<run_name>).")
    parser.add_argument("--no-himag-filter", action="store_true",
                        help="Disable hi-mag sample filtering entirely. "
                             "All samples are kept regardless of grad_mag "
                             "loss magnitude.")
    parser.add_argument("--wandb-tags", type=str, default=None,
                        help="Comma-separated list of W&B run tags.")
    parser.add_argument("--w-dist", type=float, default=0.0,
                        help="Weight on sort-based distribution loss for cos "
                             "(phase-invariant). 0 = off.")
    parser.add_argument("--w-fft", type=float, default=0.0,
                        help="Weight on FFT magnitude loss for cos "
                             "(phase-invariant frequency matching). 0 = off.")
    parser.add_argument("--num-loss-scales", type=int, default=5,
                        help="Number of scales in scale-space loss (1 = no multi-scale).")
    parser.add_argument("--no-deform", action="store_true",
                        help="Disable per-sample GT deformation refinement.")
    parser.add_argument("--deform-stride", type=int, default=8,
                        help="Deformation grid stride. Grid size = "
                             "label_patch_size / stride (default 8 → 24³).")
    parser.add_argument("--deform-inner-iters", type=int, default=100,
                        help="Inner optimization iterations for deformation "
                             "per training step.")
    parser.add_argument("--deform-inner-lr", type=float, default=1000.0,
                        help="Start LR for inner deformation loop "
                             "(ramps to 100x on log scale).")
    parser.add_argument("--deform-max-frac", type=float, default=0.3,
                        help="Max displacement as fraction of inter-surface "
                             "distance.")
    parser.add_argument("--refine", action="store_true", default=False,
                        help="Enable multi-scale refinement mode. "
                             "Sets in_channels=11 (CT + 8ch prior + "
                             "validity + scale). Disables scale_aug. "
                             "Per-batch random mode selection from 7 "
                             "scale/chain configurations.")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    train(
        config_path=args.config,
        log_dir=args.log_dir,
        run_name=args.run_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patch_size=args.patch_size,
        label_patch_size=args.label_patch_size,
        model_patch_size=args.model_patch_size,
        w_cos=args.w_cos,
        w_mag=args.w_mag,
        w_dir=args.w_dir,
        w_dir_dense=args.w_dir_dense,
        w_smooth=args.w_smooth,
        num_workers=args.num_workers,
        val_fraction=args.val_fraction,
        device=device,
        weights=args.weights,
        norm_type=args.norm_type,
        upsample_mode=args.upsample_mode,
        output_sigmoid=args.output_sigmoid,
        precision=args.precision,
        verbose=args.verbose,
        wandb_enabled=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_tags=(
            [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
            if args.wandb_tags else None
        ),
        reinit_decoder_scales=args.reinit_decoder_scales,
        himag_filter=not args.no_himag_filter,
        w_dist=args.w_dist,
        w_fft=args.w_fft,
        num_loss_scales=args.num_loss_scales,
        deform_enabled=not args.no_deform,
        deform_stride=args.deform_stride,
        deform_inner_iters=args.deform_inner_iters,
        deform_inner_lr=args.deform_inner_lr,
        deform_max_frac=args.deform_max_frac,
        refine=args.refine,
    )


if __name__ == "__main__":
    main()
