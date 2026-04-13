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
from types import SimpleNamespace

# Ensure lasagna/ dir is on sys.path for sibling imports
_LASAGNA_DIR = os.path.dirname(os.path.abspath(__file__))
if _LASAGNA_DIR not in sys.path:
    sys.path.insert(0, _LASAGNA_DIR)

from vesuvius.models.build.build_network_from_config import NetworkFromConfig

from tifxyz_labels import compute_patch_labels, encode_direction_channels
from tifxyz_dataset import TifxyzLasagnaDataset, collate_variable_surfaces


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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute training targets on GPU from dataset batch.

    Runs EDT + chain ordering + cos/grad_mag derivation on each sample.

    Args:
        batch: dict from collate_variable_surfaces
        device: CUDA device

    Returns:
        targets: (B, 8, Z, Y, X) float32
        validity: (B, 1, Z, Y, X) float32 — where cos/grad_mag are valid
        normals_valid: (B, 1, Z, Y, X) float32 — where directions are valid
        dir_weight: (B, 6, Z, Y, X) float32 — per-direction relevance weight
    """
    B = batch["image"].shape[0]
    surface_masks_list = batch["surface_masks"]  # list of (Ni, Z, Y, X)
    direction_channels = batch["direction_channels"].to(device)  # (B, 6, Z, Y, X)
    normals_valid_batch = batch["normals_valid"].to(device)  # (B, 1, Z, Y, X)

    spatial_shape = batch["image"].shape[2:]  # (Z, Y, X)
    all_targets = []
    all_validity = []

    for b in range(B):
        # Convert surface masks to CUDA bool tensors
        masks_b = surface_masks_list[b].to(device)  # (N, Z, Y, X)
        N = masks_b.shape[0]
        cuda_masks = [masks_b[i] > 0.5 for i in range(N)]

        # Direction channels for this sample
        dir_ch = direction_channels[b]  # (6, Z, Y, X)
        nv = normals_valid_batch[b, 0] > 0.5  # (Z, Y, X)

        # Compute labels on GPU
        result = compute_patch_labels(
            surface_masks=cuda_masks,
            direction_channels=dir_ch,
            normals_valid=nv,
            device=device,
        )

        all_targets.append(result["targets"])      # (8, Z, Y, X)
        all_validity.append(result["validity"])     # (Z, Y, X)

    targets = torch.stack(all_targets, dim=0)       # (B, 8, Z, Y, X)
    validity = torch.stack(all_validity, dim=0).unsqueeze(1).float()  # (B, 1, Z, Y, X)
    normals_valid = normals_valid_batch  # (B, 1, Z, Y, X)

    # Direction weight: could use normal projections, but for now uniform
    dir_weight = normals_valid.expand(-1, 6, -1, -1, -1).clone()

    return targets, validity, normals_valid, dir_weight


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
                x = self.pool(x)
                y = self.pool(y)
                if m is not None:
                    invalid = 1.0 - m
                    invalid_pooled = F.max_pool3d(invalid, kernel_size=2, stride=2)
                    m = 1.0 - invalid_pooled
                if w is not None:
                    w = -F.max_pool3d(-w, kernel_size=2, stride=2)
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
) -> Tuple[nn.Module, str, str]:
    ckpt = None
    if weights is not None:
        ckpt = torch.load(weights, map_location=device, weights_only=False)
        if isinstance(ckpt, dict):
            if norm_type is None:
                norm_type = ckpt.get("norm_type", "instance")
            if upsample_mode is None:
                upsample_mode = ckpt.get("upsample_mode", "transpconv")
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
    mgr.train_patch_size = (patch_size, patch_size, patch_size)
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
    lr: float = 1e-4,
    patch_size: int = 128,
    w_cos: float = 1.0,
    w_mag: float = 1.0,
    w_dir: float = 1.0,
    num_workers: int = 4,
    val_fraction: float = 0.15,
    device: str = "cuda",
    weights: Optional[str] = None,
    norm_type: str = "instance",
    upsample_mode: str = "trilinear",
    precision: str = "bf16",
) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(log_dir) / f"{timestamp}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    with open(config_path, "r") as f:
        config = json.load(f)

    config["patch_size"] = patch_size

    # Save run config
    run_config = {
        "config_path": config_path,
        "epochs": epochs, "batch_size": batch_size, "lr": lr,
        "patch_size": patch_size, "w_cos": w_cos, "w_mag": w_mag, "w_dir": w_dir,
        "num_workers": num_workers, "val_fraction": val_fraction,
        "device": device, "weights": weights,
        "norm_type": norm_type, "upsample_mode": upsample_mode,
        "precision": precision, "cmd": " ".join(sys.argv),
    }
    (run_dir / "config.json").write_text(json.dumps(run_config, indent=2) + "\n")

    # Dataset
    print(f"{TAG} building dataset...", flush=True)
    full_dataset = TifxyzLasagnaDataset(config, apply_augmentation=False)

    n = len(full_dataset)
    n_val = max(1, int(n * val_fraction))
    n_train = n - n_val
    train_indices = list(range(n_train))
    val_indices = list(range(n_train, n))
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

    for epoch in range(epochs):
        model.train()
        epoch_losses: List[float] = []

        for batch in train_loader:
            image = batch["image"].to(device, non_blocking=True)

            # Compute targets on GPU from surface masks
            targets, validity, normals_valid, dir_weight = compute_batch_targets(
                batch, device,
            )

            if validity.sum() == 0:
                print(f"{TAG} WARNING: zero-validity batch, skipping", flush=True)
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
                pred = torch.sigmoid(results["output"])  # (B, 8, Z, Y, X)

                # Combined validity: cos/grad_mag need surface validity, directions need normals validity
                cos_mask = validity
                dir_mask = (normals_valid > 0.5).float()

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

            if global_step % 10 == 0:
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/loss_cos", loss_cos.item(), global_step)
                writer.add_scalar("train/loss_mag", loss_mag.item(), global_step)
                writer.add_scalar("train/loss_dir", loss_dir.item(), global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

            if global_step % 100 == 0:
                _log_vis(writer, "train", image, pred, targets, cos_mask, global_step)

            global_step += 1

        cosine_scheduler.step()

        # Validation
        val_loss = _evaluate(
            model, val_loader, scale_loss_mse, scale_loss_l1,
            device, writer, global_step,
            w_cos, w_mag, w_dir,
            amp_dtype=amp_dtype, use_autocast=use_autocast,
        )

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
            "precision": precision,
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
):
    model.eval()
    losses = []
    vis_done = False

    with torch.no_grad(), torch.amp.autocast(device_type=device, dtype=amp_dtype, enabled=use_autocast):
        for batch in loader:
            image = batch["image"].to(device)
            targets, validity, normals_valid, dir_weight = compute_batch_targets(
                batch, device,
            )

            if validity.sum() == 0:
                continue

            results = model(image)
            pred = torch.sigmoid(results["output"])

            cos_mask = validity
            dir_mask = (normals_valid > 0.5).float()

            loss_cos = scale_loss_mse(pred[:, 0:1], targets[:, 0:1], mask=cos_mask)
            loss_mag = scale_loss_l1(pred[:, 1:2], targets[:, 1:2], mask=cos_mask)
            loss_dir = scale_loss_mse(pred[:, 2:8], targets[:, 2:8], mask=dir_mask, weight=dir_weight)
            loss = w_cos * loss_cos + w_mag * loss_mag + w_dir * loss_dir
            losses.append(loss.item())

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
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--w-cos", type=float, default=1.0)
    parser.add_argument("--w-mag", type=float, default=1.0)
    parser.add_argument("--w-dir", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--weights", type=str, default=None,
                        help="Checkpoint to resume from.")
    parser.add_argument("--norm-type", type=str, default="instance",
                        choices=["instance", "group", "none"])
    parser.add_argument("--upsample-mode", type=str, default="trilinear",
                        choices=["transpconv", "trilinear", "pixelshuffle"])
    parser.add_argument("--precision", type=str, default="bf16",
                        choices=["bf16", "fp16", "fp32"])
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
        w_cos=args.w_cos,
        w_mag=args.w_mag,
        w_dir=args.w_dir,
        num_workers=args.num_workers,
        val_fraction=args.val_fraction,
        device=device,
        weights=args.weights,
        norm_type=args.norm_type,
        upsample_mode=args.upsample_mode,
        precision=args.precision,
    )


if __name__ == "__main__":
    main()
