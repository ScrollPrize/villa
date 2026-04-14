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
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from types import SimpleNamespace

# Ensure lasagna/ dir is on sys.path for sibling imports
_LASAGNA_DIR = os.path.dirname(os.path.abspath(__file__))
if _LASAGNA_DIR not in sys.path:
    sys.path.insert(0, _LASAGNA_DIR)

from vesuvius.models.build.build_network_from_config import NetworkFromConfig

from tifxyz_labels import (
    compute_patch_labels,
    encode_direction_channels,
    scale_space_pool_validity,
)
from tifxyz_lasagna_dataset import TifxyzLasagnaDataset, collate_variable_surfaces
from lasagna3d.dataset_vis import (
    build_inference_context,
    render_batch_figure,
    default_vis_title,
    _sample_from_batch,
)
from PIL import Image


TAG = "[train_tifxyz]"

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
    list[list[int]],
    list[list[torch.Tensor]],
    list[list[dict]],
]:
    """Compute training targets on GPU from dataset batch.

    Runs EDT + chain ordering + cos/grad_mag derivation on each sample.
    This function is the single place where training-side same-surface
    detection happens: EDTs are built here, handed to
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
        normals_valid: (B, 1, Z, Y, X) float32 — where directions are valid
        dir_weight: (B, 6, Z, Y, X) float32 — per-direction relevance weight
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
    direction_channels = batch["direction_channels"].to(device)  # (B, 6, Z, Y, X)
    normals_valid_batch = batch["normals_valid"].to(device)  # (B, 1, Z, Y, X)

    spatial_shape = batch["image"].shape[2:]  # (Z, Y, X)
    all_targets = []
    all_validity = []
    merge_groups_batch: list[list[int]] = []
    merged_masks_batch: list[list[torch.Tensor]] = []
    merged_chain_info_batch: list[list[dict]] = []

    from tifxyz_labels import edt_torch, detect_same_surface_groups

    for b in range(B):
        # Convert surface masks to CUDA bool tensors
        masks_b = surface_masks_list[b].to(device)  # (N, Z, Y, X)
        N = masks_b.shape[0]
        cuda_masks = [masks_b[i] > 0.5 for i in range(N)]

        # Direction channels for this sample
        dir_ch = direction_channels[b]  # (6, Z, Y, X)
        nv = normals_valid_batch[b, 0] > 0.5  # (Z, Y, X)

        chain_info_b = surface_chain_info_batch[b]

        # --- Same-surface merge resolution ---
        explicit_groups = None
        if same_surface_groups_batch is not None and b < len(same_surface_groups_batch):
            explicit_groups = same_surface_groups_batch[b]

        dts: list[torch.Tensor] | None = None
        groups: list[list[int]] | None = None
        if explicit_groups is not None:
            groups = explicit_groups
        elif same_surface_threshold is not None and N >= 2:
            # Detect here so compute_patch_labels doesn't duplicate EDT work.
            dts = [edt_torch((~m).to(torch.uint8)) for m in cuda_masks]
            groups = detect_same_surface_groups(
                dts, cuda_masks, chain_info_b,
                threshold=float(same_surface_threshold),
            )

        # Compute labels on GPU using the patch's externally-built chains.
        result = compute_patch_labels(
            surface_masks=cuda_masks,
            direction_channels=dir_ch,
            normals_valid=nv,
            surface_chain_info=chain_info_b,
            device=device,
            same_surface_groups=groups,
            precomputed_dts=dts,
        )

        all_targets.append(result["targets"])      # (8, Z, Y, X)
        all_validity.append(result["validity"])     # (Z, Y, X)
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
    normals_valid = normals_valid_batch  # (B, 1, Z, Y, X)

    # Direction weight: could use normal projections, but for now uniform
    dir_weight = normals_valid.expand(-1, 6, -1, -1, -1).clone()

    return (
        targets, validity, normals_valid, dir_weight,
        merge_groups_batch, merged_masks_batch, merged_chain_info_batch,
    )


# ---------------------------------------------------------------------------
# 3D augmentation (spatial + intensity)
# ---------------------------------------------------------------------------

def augment_spatial_3d(
    image: torch.Tensor,
    surface_masks: list[torch.Tensor],
    direction_channels: torch.Tensor,
    normals_valid: torch.Tensor,
) -> Tuple[torch.Tensor, list[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Random spatial augmentation: flips and 90-degree rotations.

    Applied BEFORE GPU target computation. Only flips and rotations
    (no interpolation) to preserve binary mask integrity.
    """
    # Note: augmentation done per-sample before collation would be cleaner
    # but for simplicity we skip spatial augmentation on the raw surfaces
    # and only apply it after target computation (see augment_targets).
    return image, surface_masks, direction_channels, normals_valid


def augment_targets_3d(
    image: torch.Tensor,
    targets: torch.Tensor,
    validity: torch.Tensor,
    normals_valid: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random 3D spatial augmentation on image + computed targets.

    Flips along each axis independently, plus random 90-degree rotation
    around Z axis. Direction channels need sign correction on flip.
    """
    # Flip Z (dim 2 in BCDHW)
    if torch.rand(1).item() < 0.5:
        image = torch.flip(image, [2])
        targets = torch.flip(targets, [2])
        validity = torch.flip(validity, [2])
        normals_valid = torch.flip(normals_valid, [2])

    # Flip Y (dim 3)
    if torch.rand(1).item() < 0.5:
        image = torch.flip(image, [3])
        targets = torch.flip(targets, [3])
        validity = torch.flip(validity, [3])
        normals_valid = torch.flip(normals_valid, [3])

    # Flip X (dim 4)
    if torch.rand(1).item() < 0.5:
        image = torch.flip(image, [4])
        targets = torch.flip(targets, [4])
        validity = torch.flip(validity, [4])
        normals_valid = torch.flip(normals_valid, [4])

    # Random 90-degree rotation around Z axis (dims 3, 4)
    k = int(torch.randint(0, 4, (1,)).item())
    if k:
        image = torch.rot90(image, k, dims=(3, 4))
        targets = torch.rot90(targets, k, dims=(3, 4))
        validity = torch.rot90(validity, k, dims=(3, 4))
        normals_valid = torch.rot90(normals_valid, k, dims=(3, 4))

    return image, targets, validity, normals_valid


def augment_intensity(image: torch.Tensor) -> torch.Tensor:
    """Random CT intensity augmentation (applied to image only)."""
    if torch.rand(1).item() < 0.5:
        delta = (torch.rand(1, device=image.device).item() - 0.5) * 0.3
        image = image + delta
    if torch.rand(1).item() < 0.5:
        factor = 0.7 + torch.rand(1, device=image.device).item() * 0.6
        mean = image.mean()
        image = (image - mean) * factor + mean
    if torch.rand(1).item() < 0.3:
        noise = torch.randn_like(image) * 0.03
        image = image + noise
    return image


# ---------------------------------------------------------------------------
# Loss functions (reused from train_unet_3d.py)
# ---------------------------------------------------------------------------

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
    def __init__(self, base_loss: nn.Module, num_scales: int) -> None:
        super().__init__()
        self.base_loss = base_loss
        self.num_scales = num_scales
        self.pool = nn.AvgPool3d(kernel_size=2, stride=2)

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
            total = total + self.base_loss(x, y, mask=m, weight=w)
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

def build_model(
    patch_size: int,
    device: str,
    weights: Optional[str] = None,
    norm_type: Optional[str] = None,
    upsample_mode: Optional[str] = None,
    batch_size: int = 2,
    strict: bool = False,
    model_patch_size: Optional[int] = None,
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
    mgr.in_channels = 1
    mgr.autoconfigure = True
    mgr.spacing = [1, 1, 1]
    mgr.model_name = "lasagna_tifxyz_3d"

    model = NetworkFromConfig(mgr).to(device)

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

    return model, norm_type, upsample_mode


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _log_vis(writer, tag, image, pred, targets, mask, step):
    with torch.no_grad():
        b = min(4, image.size(0))
        mid_z = pred.size(2) // 2
        mid_z_img = image.size(2) // 2

        writer.add_images(f"{tag}/input_z", image[:b, :, mid_z_img], step)
        m = mask[:b, :, mid_z]

        for i, name in enumerate(_CHANNEL_NAMES):
            p = pred[:b, i:i+1, mid_z]
            t = targets[:b, i:i+1, mid_z]
            writer.add_images(f"{tag}/{name}_pred", p.clamp(0, 1), step)
            writer.add_images(f"{tag}/{name}_gt", (t * m).clamp(0, 1), step)

        writer.add_images(f"{tag}/validity", m, step)


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
    model_patch_size: Optional[int] = None,
    w_cos: float = 1.0,
    w_mag: float = 1.0,
    w_dir: float = 1.0,
    num_workers: int = 4,
    val_fraction: float = 0.15,
    device: str = "cuda",
    weights: Optional[str] = None,
    norm_type: str = "none",
    upsample_mode: str = "trilinear",
    output_sigmoid: bool = False,
    precision: str = "bf16",
    verbose: bool = False,
) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(log_dir) / f"{timestamp}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    with open(config_path, "r") as f:
        config = json.load(f)

    config["patch_size"] = patch_size

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
        "patch_size": patch_size, "w_cos": w_cos, "w_mag": w_mag, "w_dir": w_dir,
        "num_workers": num_workers, "val_fraction": val_fraction,
        "device": device, "weights": weights,
        "norm_type": norm_type, "upsample_mode": upsample_mode,
        "output_sigmoid": output_sigmoid,
        "precision": precision,
        "same_surface_threshold": same_surface_threshold,
        "cmd": " ".join(sys.argv),
    }
    (run_dir / "config.json").write_text(json.dumps(run_config, indent=2) + "\n")

    # Dataset
    print(f"{TAG} building dataset...", flush=True)
    full_dataset = TifxyzLasagnaDataset(config, apply_augmentation=False)

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

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        collate_fn=collate_variable_surfaces,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_variable_surfaces,
    )
    print(f"{TAG} {n_train} train / {n_val} val patches", flush=True)

    # Model
    model, norm_type, upsample_mode = build_model(
        patch_size, device, weights, norm_type=norm_type,
        upsample_mode=upsample_mode, batch_size=batch_size,
        model_patch_size=model_patch_size,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    # LR warmup + cosine decay
    warmup_steps = 200
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        return 1.0
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

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
    print(f"{TAG} precision: {precision}, upsample: {upsample_mode}", flush=True)

    # Losses
    mse_loss = MaskedMSE()
    smooth_l1_loss = MaskedSmoothL1()
    scale_loss_mse = ScaleSpaceLoss3D(mse_loss, num_scales=3)
    scale_loss_l1 = ScaleSpaceLoss3D(smooth_l1_loss, num_scales=3)

    writer = SummaryWriter(log_dir=str(run_dir))
    best_val_loss = float("inf")
    global_step = 0

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
    vis_ctx["output_sigmoid"] = bool(output_sigmoid)
    vis_ctx["loss_weights"] = (float(w_cos), float(w_mag), float(w_dir))
    vis_out_dir = run_dir / "vis"
    vis_out_dir.mkdir(exist_ok=True)
    vis_interval_steps = max(1, 1000 // max(batch_size, 1))

    def _log_full_vis(batch_for_vis, tag: str, step: int) -> None:
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
            try:
                render_batch_figure(
                    batch_for_vis, out_path, title,
                    arrow_seed=step,
                    inference_ctx=vis_ctx,
                    model=model,
                )
            finally:
                if was_training:
                    model.train()
            img_rgb = np.asarray(Image.open(out_path).convert("RGB"))
            writer.add_image(
                f"{tag}/full_vis", img_rgb, step, dataformats="HWC",
            )
        except Exception as e:
            print(f"{TAG} full_vis ({tag}) render failed at step "
                  f"{step}: {e}", flush=True)

    # Initial eval before training so we see init/load performance.
    print(f"{TAG} running pre-training eval at step 0...", flush=True)
    init_val_loss = _evaluate(
        model, val_loader, scale_loss_mse, scale_loss_l1,
        device, writer, global_step,
        w_cos, w_mag, w_dir,
        amp_dtype=amp_dtype, use_autocast=use_autocast,
        verbose=verbose,
        same_surface_threshold=same_surface_threshold,
        output_sigmoid=output_sigmoid,
    )
    for _probe_batch in val_loader:
        _log_full_vis(_probe_batch, "val", 0)
        break
    print(f"{TAG} init val={init_val_loss:.4f}", flush=True)

    last_val_loss = init_val_loss
    val_every_steps = 200

    for epoch in range(epochs):
        model.train()
        epoch_losses: List[float] = []

        train_iter = tqdm(
            train_loader,
            desc=f"epoch {epoch + 1}/{epochs} train",
            dynamic_ncols=True,
            mininterval=0.0, miniters=1,
        )

        n_seen = 0
        n_skipped = 0
        n_hi_mag = 0
        n_surfaces_pre = 0
        n_surfaces_post = 0

        for batch in train_iter:
            image = batch["image"].to(device, non_blocking=True)

            # Compute targets on GPU from surface masks
            (
                targets, validity, normals_valid, dir_weight,
                _merge_groups, _merged_masks, _merged_chain_info,
            ) = compute_batch_targets(
                batch, device,
                same_surface_threshold=same_surface_threshold,
            )

            n_seen += 1
            for mg in _merge_groups:
                n_surfaces_pre += len(mg)
                n_surfaces_post += len(set(int(x) for x in mg))

            if validity.sum() == 0:
                n_skipped += 1
                train_iter.set_postfix(
                    loss="---",
                    skip=f"{100.0 * n_skipped / max(n_seen, 1):.1f}%",
                    merge=f"{100.0 * (1.0 - n_surfaces_post / max(n_surfaces_pre, 1)):.1f}%",
                )
                continue

            # Spatial augmentation on image + targets
            image, targets, validity, normals_valid = augment_targets_3d(
                image, targets, validity, normals_valid,
            )
            # Intensity augmentation on CT only
            image = augment_intensity(image)

            # Forward pass
            with torch.amp.autocast(device_type=device, dtype=amp_dtype, enabled=use_autocast):
                results = model(image)
                raw_pred = results["output"]
                pred = torch.sigmoid(raw_pred) if output_sigmoid else raw_pred  # (B, 8, Z, Y, X)

                # Combined validity: cos/grad_mag need surface validity, directions need normals validity
                cos_mask = validity
                dir_mask = (normals_valid > 0.5).float()

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
                bad = [b for b, v in enumerate(per_mag_vals) if v >= 1.0]
                for b in bad:
                    pi = batch["patch_info"][b]
                    seg = pi.get("segment_uuid", "?")
                    idx_b = pi.get("idx", "?")
                    print(
                        f"{TAG} hi-mag skip step={global_step} "
                        f"seg={seg} idx={idx_b} "
                        f"mag_loss={per_mag_vals[b]:.4f}",
                        flush=True,
                    )
                n_hi_mag += len(bad)
                if len(bad) == B:
                    train_iter.set_postfix(
                        loss="---",
                        skip=f"{100.0 * n_skipped / max(n_seen, 1):.1f}%",
                        himag=n_hi_mag,
                        merge=f"{100.0 * (1.0 - n_surfaces_post / max(n_surfaces_pre, 1)):.1f}%",
                    )
                    continue
                if bad:
                    keep = torch.tensor(
                        [b for b in range(B) if b not in bad],
                        device=pred.device, dtype=torch.long,
                    )
                    pred = pred.index_select(0, keep)
                    targets = targets.index_select(0, keep)
                    cos_mask = cos_mask.index_select(0, keep)
                    dir_mask = dir_mask.index_select(0, keep)
                    dir_weight = dir_weight.index_select(0, keep)

                # Per-channel losses
                loss_cos = scale_loss_mse(pred[:, 0:1], targets[:, 0:1], mask=cos_mask)
                loss_mag = scale_loss_l1(pred[:, 1:2], targets[:, 1:2], mask=cos_mask)
                loss_dir = scale_loss_mse(
                    pred[:, 2:8], targets[:, 2:8],
                    mask=dir_mask, weight=dir_weight,
                )
                loss = w_cos * loss_cos + w_mag * loss_mag + w_dir * loss_dir

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            if global_step < warmup_steps:
                warmup_scheduler.step()

            epoch_losses.append(loss.item())
            if not math.isfinite(loss.item()):
                print(f"{TAG} NaN/Inf at step {global_step}: loss={loss.item()}", flush=True)

            running = sum(epoch_losses[-20:]) / min(len(epoch_losses), 20)
            train_iter.set_postfix(
                loss=f"{loss.item():.4f}",
                avg20=f"{running:.4f}",
                skip=f"{100.0 * n_skipped / max(n_seen, 1):.1f}%",
                himag=n_hi_mag,
                merge=f"{100.0 * (1.0 - n_surfaces_post / max(n_surfaces_pre, 1)):.1f}%",
            )

            if global_step % 10 == 0:
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/loss_cos", loss_cos.item(), global_step)
                writer.add_scalar("train/loss_mag", loss_mag.item(), global_step)
                writer.add_scalar("train/loss_dir", loss_dir.item(), global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

            if global_step % 100 == 0:
                _log_vis(writer, "train", image, pred, targets, cos_mask, global_step)

            if global_step % vis_interval_steps == 0:
                _log_full_vis(batch, "train", global_step)

            global_step += 1

            # Periodic validation mid-epoch so we see val trajectory
            # at a much higher resolution than once-per-epoch.
            if global_step > 0 and global_step % val_every_steps == 0:
                last_val_loss = _evaluate(
                    model, val_loader, scale_loss_mse, scale_loss_l1,
                    device, writer, global_step,
                    w_cos, w_mag, w_dir,
                    amp_dtype=amp_dtype, use_autocast=use_autocast,
                    verbose=verbose,
                    same_surface_threshold=same_surface_threshold,
                    output_sigmoid=output_sigmoid,
                )
                for _probe_batch in val_loader:
                    _log_full_vis(_probe_batch, "val", global_step)
                    break
                model.train()
                train_iter.write(
                    f"{TAG} step {global_step}  val={last_val_loss:.4f}"
                )

        cosine_scheduler.step()

        val_loss = last_val_loss
        mean_train = sum(epoch_losses) / max(len(epoch_losses), 1)
        print(
            f"epoch {epoch + 1}/{epochs}  "
            f"train={mean_train:.4f}  val={val_loss:.4f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}",
            flush=True,
        )

        # Checkpoints
        ckpt_data = {
            "state_dict": model.state_dict(),
            "norm_type": norm_type,
            "upsample_mode": upsample_mode,
            "output_sigmoid": output_sigmoid,
            "precision": precision,
            "patch_size": patch_size,
            "in_channels": 1,
            "out_channels": 8,
        }
        torch.save(ckpt_data, run_dir / "model_current.pt")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt_data, run_dir / "model_best.pt")
            print(f"  -> new best val loss: {val_loss:.4f}", flush=True)

    writer.close()
    print(f"{TAG} done. Logs & checkpoints in {run_dir}", flush=True)


def _evaluate(
    model, loader, scale_loss_mse, scale_loss_l1,
    device, writer, global_step,
    w_cos, w_mag, w_dir,
    amp_dtype=torch.bfloat16, use_autocast=True,
    verbose: bool = False,
    same_surface_threshold: float | None = None,
    output_sigmoid: bool = False,
):
    model.eval()
    losses = []
    vis_done = False

    val_iter = loader
    if verbose:
        val_iter = tqdm(loader, desc="val", dynamic_ncols=True, leave=False)

    with torch.no_grad(), torch.amp.autocast(device_type=device, dtype=amp_dtype, enabled=use_autocast):
        for batch in val_iter:
            image = batch["image"].to(device)
            (
                targets, validity, normals_valid, dir_weight,
                _merge_groups, _merged_masks, _merged_chain_info,
            ) = compute_batch_targets(
                batch, device,
                same_surface_threshold=same_surface_threshold,
            )

            if validity.sum() == 0:
                continue

            results = model(image)
            raw_pred = results["output"]
            pred = torch.sigmoid(raw_pred) if output_sigmoid else raw_pred

            cos_mask = validity
            dir_mask = (normals_valid > 0.5).float()

            loss_cos = scale_loss_mse(pred[:, 0:1], targets[:, 0:1], mask=cos_mask)
            loss_mag = scale_loss_l1(pred[:, 1:2], targets[:, 1:2], mask=cos_mask)
            loss_dir = scale_loss_mse(pred[:, 2:8], targets[:, 2:8], mask=dir_mask, weight=dir_weight)
            loss = w_cos * loss_cos + w_mag * loss_mag + w_dir * loss_dir
            losses.append(loss.item())

            if verbose:
                val_iter.set_postfix(loss=f"{loss.item():.4f}")

            if not vis_done:
                _log_vis(writer, "val", image, pred, targets, cos_mask, global_step)
                vis_done = True

    mean_loss = sum(losses) / max(len(losses), 1)
    writer.add_scalar("val/loss", mean_loss, global_step)
    return mean_loss


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
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--model-patch-size", type=int, default=None,
                        help="Patch size used only for model architecture "
                             "autoconfigure (stage count). Defaults to "
                             "--patch-size. Set this to the checkpoint's "
                             "patch size when continuing training at a "
                             "different training patch size.")
    parser.add_argument("--w-cos", type=float, default=1.0)
    parser.add_argument("--w-mag", type=float, default=1.0)
    parser.add_argument("--w-dir", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--weights", type=str, default=None,
                        help="Checkpoint to resume from.")
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
        model_patch_size=args.model_patch_size,
        w_cos=args.w_cos,
        w_mag=args.w_mag,
        w_dir=args.w_dir,
        num_workers=args.num_workers,
        val_fraction=args.val_fraction,
        device=device,
        weights=args.weights,
        norm_type=args.norm_type,
        upsample_mode=args.upsample_mode,
        output_sigmoid=args.output_sigmoid,
        precision=args.precision,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
