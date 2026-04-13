"""Train 3D UNet to predict lasagna model channels from raw CT data.

Targets (cos, grad_mag, 3x2 direction encoding = 8 channels) are computed
on-the-fly on GPU from fitted.zarr files produced by run_batch.sh.

Usage:
  python lasagna/train_unet_3d.py \
      --images-dir /path/to/images \
      --label-dir /path/to/label_dir \
      --patch-size 128 --batch-size 2 --epochs 100
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import zarr
import tifffile
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter
from types import SimpleNamespace

from vesuvius.models.build.build_network_from_config import NetworkFromConfig


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _json_lookup(path: Path, dotted_key: str) -> Optional[float]:
    """Look up a dotted path (e.g. 'losses.pred_dt.max') in a JSON file."""
    try:
        with open(path) as f:
            obj = json.load(f)
        for part in dotted_key.split("."):
            obj = obj[part]
        return float(obj)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


class FittedZarrDataset(Dataset):
    """Paired CT volume + fitted.zarr labels for 3D UNet training.

    Loads multi-layer TIF images and corresponding fitted.zarr directories.
    Returns patches at full resolution (CT) and step resolution (labels).
    """

    def __init__(
        self,
        images_dir: str | Path,
        label_dir: str | Path,
        patch_size: Optional[int] = None,
        random_crop: bool = True,
        stats_filter: Optional[str] = "losses.pred_dt.max",
        stats_threshold: float = 5.0,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.label_dir = Path(label_dir)
        self.patch_size = patch_size
        self.random_crop = random_crop

        if not self.images_dir.is_dir():
            raise ValueError(f"images_dir does not exist: {self.images_dir}")
        if not self.label_dir.is_dir():
            raise ValueError(f"label_dir does not exist: {self.label_dir}")

        # Match CT TIFs to fitted.zarr directories
        all_samples: List[Tuple[Path, Path, str]] = []
        for img_path in sorted(self.images_dir.glob("*.tif")):
            stem = img_path.stem  # e.g. "sample_00033"
            zarr_path = self.label_dir / stem / "fitted.zarr"
            if zarr_path.is_dir():
                all_samples.append((img_path, zarr_path, stem))

        if not all_samples:
            raise ValueError(
                f"No matching CT+fitted.zarr pairs found.\n"
                f"  images_dir: {self.images_dir}\n"
                f"  label_dir: {self.label_dir}"
            )

        # Filter samples by stats.json value
        self.samples: List[Tuple[Path, Path, str]] = []
        n_skipped = 0
        for img_path, zarr_path, stem in all_samples:
            if stats_filter is not None:
                stats_path = self.label_dir / stem / "stats.json"
                if stats_path.is_file():
                    val = _json_lookup(stats_path, stats_filter)
                    if val is not None and val >= stats_threshold:
                        n_skipped += 1
                        print(f"[dataset] skip {stem}: {stats_filter}={val:.4f} >= {stats_threshold}")
                        continue
            self.samples.append((img_path, zarr_path, stem))

        if n_skipped:
            print(f"[dataset] skipped {n_skipped}/{len(all_samples)} samples "
                  f"({stats_filter} >= {stats_threshold})")

        if not self.samples:
            raise ValueError("All samples were filtered out")

        # Read step from first sample, verify all are consistent & isotropic
        root = zarr.open(str(self.samples[0][1]), mode="r")
        sp = root.attrs["spacing"]
        self.step = int(round(sp[0]))
        assert int(round(sp[1])) == self.step and int(round(sp[2])) == self.step, \
            f"Non-isotropic spacing not supported: {sp}"

        for _, zp, name in self.samples[1:]:
            r = zarr.open(str(zp), mode="r")
            s = int(round(r.attrs["spacing"][0]))
            if s != self.step:
                raise ValueError(f"Inconsistent step: {name} has step={s}, expected {self.step}")

        if self.patch_size is not None and self.patch_size % self.step != 0:
            raise ValueError(f"patch_size ({self.patch_size}) must be divisible by step ({self.step})")

        print(f"[dataset] {len(self.samples)} samples, step={self.step}, "
              f"patch_size={self.patch_size}, random_crop={self.random_crop}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        img_path, zarr_path, name = self.samples[idx]

        # Load zarr arrays
        root = zarr.open(str(zarr_path), mode="r")
        origin = root.attrs["origin_fullres"]  # [x0, y0, z0] in fullres voxels
        normal = np.array(root["normal"])       # (3, Z, Y, X)
        winding = np.array(root["winding"])     # (Z, Y, X)
        validity = np.array(root["validity"])   # (Z, Y, X)
        density = np.array(root["density"])     # (Z, Y, X)

        Z, Y, X = winding.shape
        step = self.step
        # origin is (x, y, z); image axes are (z, y, x)
        x0 = int(round(origin[0]))
        y0 = int(round(origin[1]))
        z0 = int(round(origin[2]))

        # Load CT and crop to zarr extent
        image = tifffile.imread(str(img_path))  # (D_full, H_full, W_full)
        if image.dtype == np.uint16:
            image = (image // 257).astype(np.uint8)
        image = image[z0:z0 + Z * step, y0:y0 + Y * step, x0:x0 + X * step]

        # Clamp labels to match available image extent (numpy silently
        # truncates slices that exceed array bounds)
        Z_eff = image.shape[0] // step
        Y_eff = image.shape[1] // step
        X_eff = image.shape[2] // step
        if Z_eff < Z or Y_eff < Y or X_eff < X:
            normal = normal[:, :Z_eff, :Y_eff, :X_eff]
            winding = winding[:Z_eff, :Y_eff, :X_eff]
            validity = validity[:Z_eff, :Y_eff, :X_eff]
            density = density[:Z_eff, :Y_eff, :X_eff]
            Z, Y, X = Z_eff, Y_eff, X_eff

        # Random or center crop to patch_size
        if self.patch_size is not None:
            ps = self.patch_size
            ps_l = ps // step  # patch size in label resolution
            if self.random_crop:
                zl = torch.randint(0, max(1, Z - ps_l + 1), (1,)).item()
                yl = torch.randint(0, max(1, Y - ps_l + 1), (1,)).item()
                xl = torch.randint(0, max(1, X - ps_l + 1), (1,)).item()
            else:
                zl = max(0, (Z - ps_l) // 2)
                yl = max(0, (Y - ps_l) // 2)
                xl = max(0, (X - ps_l) // 2)
            normal = normal[:, zl:zl + ps_l, yl:yl + ps_l, xl:xl + ps_l]
            winding = winding[zl:zl + ps_l, yl:yl + ps_l, xl:xl + ps_l]
            validity = validity[zl:zl + ps_l, yl:yl + ps_l, xl:xl + ps_l]
            density = density[zl:zl + ps_l, yl:yl + ps_l, xl:xl + ps_l]
            zi, yi, xi = zl * step, yl * step, xl * step
            image = image[zi:zi + ps, yi:yi + ps, xi:xi + ps]

            # Pad to exact patch_size if any dim ended up smaller
            # (validity=0 in padded region -> loss ignores it)
            if image.shape != (ps, ps, ps):
                img_pad = np.zeros((ps, ps, ps), dtype=image.dtype)
                img_pad[:image.shape[0], :image.shape[1], :image.shape[2]] = image
                image = img_pad
            lz, ly, lx = winding.shape
            if (lz, ly, lx) != (ps_l, ps_l, ps_l):
                def _pad3(a, t, nd4=False):
                    if nd4:
                        o = np.zeros((a.shape[0], t, t, t), dtype=a.dtype)
                        o[:, :a.shape[1], :a.shape[2], :a.shape[3]] = a
                    else:
                        o = np.zeros((t, t, t), dtype=a.dtype)
                        o[:a.shape[0], :a.shape[1], :a.shape[2]] = a
                    return o
                normal = _pad3(normal, ps_l, nd4=True)
                winding = _pad3(winding, ps_l)
                validity = _pad3(validity, ps_l)
                density = _pad3(density, ps_l)

        # To tensors
        image_t = torch.from_numpy(image.astype(np.float32)).unsqueeze(0) / 255.0
        normal_t = torch.from_numpy(normal.copy()).float()
        winding_t = torch.from_numpy(winding.copy()).float().unsqueeze(0)
        validity_t = torch.from_numpy(validity.copy()).float().unsqueeze(0)
        density_t = torch.from_numpy(density.copy()).float().unsqueeze(0)

        return {
            "image": image_t,       # (1, D, H, W)
            "normal": normal_t,     # (3, d, h, w)
            "winding": winding_t,   # (1, d, h, w)
            "validity": validity_t, # (1, d, h, w)
            "density": density_t,   # (1, d, h, w)
            "name": name,
        }


# ---------------------------------------------------------------------------
# GPU target computation
# ---------------------------------------------------------------------------

def _encode_dir(gx: torch.Tensor, gy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Double-angle direction encoding from two normal components.

    Returns (dir0, dir1) each in [0, 1].
    Same formula as fitted_to_unet_labels._encode_dir, ported to torch.
    """
    eps = 1e-8
    r2 = gx * gx + gy * gy + eps
    cos2t = (gx * gx - gy * gy) / r2
    sin2t = 2.0 * gx * gy / r2
    dir0 = 0.5 + 0.5 * cos2t
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    dir1 = 0.5 + 0.5 * (cos2t - sin2t) * inv_sqrt2
    return dir0, dir1


def compute_targets_3d(
    normal: torch.Tensor,
    winding: torch.Tensor,
    validity: torch.Tensor,
    density: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute 8-channel targets on GPU from fitted.zarr fields.

    Args:
        normal:   (B, 3, d, h, w)
        winding:  (B, 1, d, h, w)
        validity: (B, 1, d, h, w)
        density:  (B, 1, d, h, w)

    Returns:
        targets: (B, 8, d, h, w) — [cos, grad_mag, dir0_z, dir1_z, dir0_y, dir1_y, dir0_x, dir1_x]
        mask:    (B, 1, d, h, w) — validity mask
    """
    nx, ny, nz = normal[:, 0:1], normal[:, 1:2], normal[:, 2:3]
    cos = 0.5 + 0.5 * torch.cos(2.0 * math.pi * winding)
    grad_mag = density
    dir0_z, dir1_z = _encode_dir(nx, ny)   # XY plane (Z-slices)
    dir0_y, dir1_y = _encode_dir(nx, nz)   # XZ plane (Y-slices)
    dir0_x, dir1_x = _encode_dir(ny, nz)   # YZ plane (X-slices)
    targets = torch.cat([
        cos, grad_mag,
        dir0_z, dir1_z, dir0_y, dir1_y, dir0_x, dir1_x,
    ], dim=1)

    # Per-direction-pair relevance: normal's projection magnitude into slice plane
    w_dir_z = torch.sqrt(nx ** 2 + ny ** 2 + 1e-8)  # XY plane
    w_dir_y = torch.sqrt(nx ** 2 + nz ** 2 + 1e-8)  # XZ plane
    w_dir_x = torch.sqrt(ny ** 2 + nz ** 2 + 1e-8)  # YZ plane
    dir_weight = torch.cat([w_dir_z, w_dir_z, w_dir_y, w_dir_y, w_dir_x, w_dir_x], dim=1)

    return targets, validity, dir_weight


# ---------------------------------------------------------------------------
# 3D augmentation
# ---------------------------------------------------------------------------

def augment_3d(
    image: torch.Tensor,
    normal: torch.Tensor,
    winding: torch.Tensor,
    validity: torch.Tensor,
    density: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random 3D spatial augmentation applied BEFORE target computation.

    Flips along each axis independently (with normal sign correction),
    plus random 90-degree rotation around Z axis (with normal XY rotation).

    All tensors have batch dim: image (B,1,D,H,W), others (B,C,d,h,w).
    """
    # Flip Z (dim 2 in BCDHW)
    if torch.rand(1).item() < 0.5:
        image = torch.flip(image, [2])
        normal = torch.flip(normal, [2])
        normal = normal.clone()
        normal[:, 2:3] = -normal[:, 2:3]
        winding = torch.flip(winding, [2])
        validity = torch.flip(validity, [2])
        density = torch.flip(density, [2])

    # Flip Y (dim 3)
    if torch.rand(1).item() < 0.5:
        image = torch.flip(image, [3])
        normal = torch.flip(normal, [3])
        normal = normal.clone()
        normal[:, 1:2] = -normal[:, 1:2]
        winding = torch.flip(winding, [3])
        validity = torch.flip(validity, [3])
        density = torch.flip(density, [3])

    # Flip X (dim 4)
    if torch.rand(1).item() < 0.5:
        image = torch.flip(image, [4])
        normal = torch.flip(normal, [4])
        normal = normal.clone()
        normal[:, 0:1] = -normal[:, 0:1]
        winding = torch.flip(winding, [4])
        validity = torch.flip(validity, [4])
        density = torch.flip(density, [4])

    # Random 90-degree rotation around Z axis (rotate in XY plane, dims 3 & 4)
    k = int(torch.randint(0, 4, (1,)).item())
    if k:
        image = torch.rot90(image, k, dims=(3, 4))
        normal = torch.rot90(normal, k, dims=(3, 4))
        winding = torch.rot90(winding, k, dims=(3, 4))
        validity = torch.rot90(validity, k, dims=(3, 4))
        density = torch.rot90(density, k, dims=(3, 4))
        # Rotate normal XY components: (nx, ny) -> R(k*90°) * (nx, ny)
        nx = normal[:, 0:1].clone()
        ny = normal[:, 1:2].clone()
        cos_a = math.cos(k * math.pi / 2)
        sin_a = math.sin(k * math.pi / 2)
        normal = normal.clone()
        normal[:, 0:1] = cos_a * nx - sin_a * ny
        normal[:, 1:2] = sin_a * nx + cos_a * ny

    return image, normal, winding, validity, density


def augment_intensity(image: torch.Tensor) -> torch.Tensor:
    """Random CT intensity augmentation (applied to image only).

    image: (B, 1, D, H, W) float32 in [0, 1].
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
# Loss functions
# ---------------------------------------------------------------------------

class MaskedMSE(nn.Module):
    """MSE loss with optional per-pixel mask (dimension-agnostic).

    Reused from train_unet.py — works for both 2D and 3D.
    """

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        diff = (pred - target) ** 2
        if mask is None:
            return diff.mean()
        if mask.ndim == pred.ndim - 1:
            mask = mask.unsqueeze(1)
        effective = mask * weight if weight is not None else mask
        diff = diff * effective
        denom = effective.sum()
        return diff.sum() / denom.clamp(min=1.0)


class ScaleSpaceLoss3D(nn.Module):
    """Multi-scale loss for 3D volumes with validity mask.

    Adapted from ScaleSpaceLoss in train_unet.py:
    - AvgPool2d -> AvgPool3d
    - max_pool2d -> max_pool3d
    - 5D tensor checks
    """

    def __init__(self, base_loss: nn.Module, num_scales: int) -> None:
        super().__init__()
        if num_scales < 1:
            raise ValueError(f"num_scales must be >= 1, got {num_scales}")
        self.base_loss = base_loss
        self.num_scales = num_scales
        self.pool = nn.AvgPool3d(kernel_size=2, stride=2)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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
                    w = -F.max_pool3d(-w, kernel_size=2, stride=2)  # min_pool

        return total


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_model(
    patch_size: int,
    device: str,
    weights: Optional[str] = None,
    norm_type: Optional[str] = None,
    upsample_mode: Optional[str] = None,
    output_sigmoid: Optional[bool] = None,
    batch_size: int = 2,
) -> nn.Module:
    """Build 3D UNet via vesuvius NetworkFromConfig.

    Args:
        norm_type: "instance", "group", or "none".
            If None and weights is provided, auto-detects from checkpoint metadata.
        upsample_mode: "transpconv", "trilinear", or "pixelshuffle".
            If None and weights is provided, auto-detects from checkpoint metadata.
        output_sigmoid: Whether to apply sigmoid to model output.
            If None and weights is provided, auto-detects from checkpoint metadata.
    """
    # Auto-detect from checkpoint if not specified
    ckpt = None
    if weights is not None:
        ckpt = torch.load(weights, map_location=device, weights_only=False)
        if isinstance(ckpt, dict):
            if norm_type is None:
                norm_type = ckpt.get("norm_type", "instance")
            if upsample_mode is None:
                upsample_mode = ckpt.get("upsample_mode", "transpconv")
            if output_sigmoid is None:
                output_sigmoid = ckpt.get("output_sigmoid", True)
    if norm_type is None:
        norm_type = "instance"
    if upsample_mode is None:
        upsample_mode = "transpconv"
    if output_sigmoid is None:
        output_sigmoid = True

    mgr = SimpleNamespace()
    model_config = {"autoconfigure": True, "architecture_type": "unet"}
    if norm_type == "group":
        model_config["norm_op"] = "nn.GroupNorm"
        model_config["norm_op_kwargs"] = {"num_groups": 32, "affine": True, "eps": 1e-5}
    elif norm_type == "none":
        model_config["norm_op"] = None
    # else "instance" — keep defaults (InstanceNorm3d)
    model_config["upsample_mode"] = upsample_mode
    mgr.model_config = model_config
    # activation='none' so we handle output activation ourselves
    # (sigmoid or clamp, configured via output_sigmoid).
    mgr.targets = {"output": {"out_channels": 8, "activation": "none"}}
    mgr.train_patch_size = (patch_size, patch_size, patch_size)
    mgr.train_batch_size = batch_size
    mgr.in_channels = 1
    mgr.autoconfigure = True
    mgr.spacing = [1, 1, 1]
    mgr.model_name = "lasagna_3d"

    model = NetworkFromConfig(mgr).to(device)

    if ckpt is not None:
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
        # Filter keys to matching shapes for flexible loading
        model_state = model.state_dict()
        filtered = {k: v for k, v in state_dict.items()
                    if k in model_state and model_state[k].shape == v.shape}
        missing = [k for k in model_state if k not in filtered]
        skipped = [k for k in state_dict if k not in filtered]
        model_state.update(filtered)
        model.load_state_dict(model_state)
        print(f"[model] loaded {len(filtered)}/{len(model_state)} params from {weights}")
        if skipped:
            print(f"[model] skipped from checkpoint: {skipped[:5]}{'...' if len(skipped) > 5 else ''}")
        if missing:
            print(f"[model] randomly initialized: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    return model, norm_type, upsample_mode, output_sigmoid


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

_CHANNEL_NAMES = [
    "cos", "grad_mag",
    "dir0_z", "dir1_z", "dir0_y", "dir1_y", "dir0_x", "dir1_x",
]


def _log_vis(
    writer: SummaryWriter,
    tag: str,
    image: torch.Tensor,
    pred: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    step: int,
) -> None:
    """Log mid-Z-slice visualizations to TensorBoard."""
    with torch.no_grad():
        b = min(4, image.size(0))

        # Mid-slice CT input in all 3 planes
        mid_z_img = image.size(2) // 2
        mid_y_img = image.size(3) // 2
        mid_x_img = image.size(4) // 2
        writer.add_images(f"{tag}/input_z", image[:b, :, mid_z_img], step)
        writer.add_images(f"{tag}/input_y", image[:b, :, :, mid_y_img], step)
        writer.add_images(f"{tag}/input_x", image[:b, :, :, :, mid_x_img], step)

        # Mid Z-slice of predictions and targets
        mid_z = pred.size(2) // 2
        m = mask[:b, :, mid_z]  # (b, 1, h, w)

        for i, name in enumerate(_CHANNEL_NAMES):
            p = pred[:b, i:i + 1, mid_z]     # (b, 1, h, w)
            t = targets[:b, i:i + 1, mid_z]
            writer.add_images(f"{tag}/{name}_pred", p.clamp(0, 1), step)
            writer.add_images(f"{tag}/{name}_gt", (t * m).clamp(0, 1), step)

        writer.add_images(f"{tag}/validity", m, step)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    images_dir: str,
    label_dir: str,
    log_dir: str = "runs/unet3d",
    run_name: str = "3d",
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
    stats_filter: Optional[str] = "losses.pred_dt.max",
    stats_threshold: float = 5.0,
    norm_type: str = "instance",
    upsample_mode: str = "trilinear",
    precision: str = "bf16",
    output_sigmoid: bool = False,
) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(log_dir) / f"{timestamp}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save run config
    config = {
        "images_dir": images_dir, "label_dir": label_dir,
        "epochs": epochs, "batch_size": batch_size, "lr": lr,
        "patch_size": patch_size, "w_cos": w_cos, "w_mag": w_mag, "w_dir": w_dir,
        "num_workers": num_workers, "val_fraction": val_fraction,
        "device": device, "weights": weights,
        "stats_filter": stats_filter, "stats_threshold": stats_threshold,
        "norm_type": norm_type,
        "upsample_mode": upsample_mode,
        "precision": precision,
        "output_sigmoid": output_sigmoid,
        "cmd": " ".join(sys.argv),
    }
    # Git state for reproducibility
    git_diff = ""
    try:
        git_hash = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        git_diff = subprocess.run(
            ["git", "diff", "HEAD"], capture_output=True, text=True, timeout=10,
        ).stdout
        config["git_hash"] = git_hash
        if git_diff:
            (run_dir / "git_diff.patch").write_text(git_diff)
    except Exception:
        pass
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    # Datasets
    full_dataset = FittedZarrDataset(
        images_dir, label_dir, patch_size=patch_size, random_crop=True,
        stats_filter=stats_filter, stats_threshold=stats_threshold,
    )
    step = full_dataset.step

    # Train/val split by sample (last N go to val)
    n = len(full_dataset)
    n_val = max(1, int(n * val_fraction))
    n_train = n - n_val
    train_indices = list(range(n_train))
    val_indices = list(range(n_train, n))
    train_dataset = Subset(full_dataset, train_indices)

    eval_dataset = FittedZarrDataset(
        images_dir, label_dir, patch_size=patch_size, random_crop=False,
        stats_filter=stats_filter, stats_threshold=stats_threshold,
    )
    eval_subset = Subset(eval_dataset, val_indices)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    eval_loader = DataLoader(
        eval_subset, batch_size=1, shuffle=False, num_workers=num_workers,
    )
    print(f"[train] {n_train} train / {n_val} val samples, step={step}")

    # Model
    model, norm_type, upsample_mode, output_sigmoid = build_model(
        patch_size, device, weights, norm_type=norm_type,
        upsample_mode=upsample_mode, output_sigmoid=output_sigmoid,
        batch_size=batch_size,
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

    # Precision config
    if precision == "bf16":
        assert device == "cpu" or torch.cuda.is_bf16_supported(), \
            "bf16 requested but not supported on this GPU"
        amp_dtype = torch.bfloat16
        use_autocast = True
        scaler = torch.amp.GradScaler(enabled=False)
    elif precision == "fp16":
        amp_dtype = torch.float16
        use_autocast = True
        scaler = torch.amp.GradScaler(enabled=(device != "cpu"))
    else:  # fp32
        amp_dtype = torch.float32
        use_autocast = False
        scaler = torch.amp.GradScaler(enabled=False)
    print(f"[train] precision: {precision}, upsample: {upsample_mode}, "
          f"output_sigmoid: {output_sigmoid}, warmup: {warmup_steps} steps")

    # Losses
    masked_mse = MaskedMSE()
    scale_loss = ScaleSpaceLoss3D(masked_mse, num_scales=3)

    writer = SummaryWriter(log_dir=str(run_dir))
    writer.add_text("config", json.dumps(config, indent=2), 0)
    if git_diff:
        writer.add_text("git_diff", git_diff, 0)
    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(epochs):
        model.train()
        epoch_losses: List[float] = []

        for batch in train_loader:
            image = batch["image"].to(device, non_blocking=True)
            normal = batch["normal"].to(device, non_blocking=True)
            winding = batch["winding"].to(device, non_blocking=True)
            validity = batch["validity"].to(device, non_blocking=True)
            density = batch["density"].to(device, non_blocking=True)

            # Spatial augmentation (before target computation)
            image, normal, winding, validity, density = augment_3d(
                image, normal, winding, validity, density,
            )
            # CT-only intensity augmentation
            image = augment_intensity(image)

            # Compute 8-channel targets on GPU
            targets, mask, dir_weight = compute_targets_3d(normal, winding, validity, density)

            if mask.sum() == 0:
                print(f"WARNING: zero-validity batch, skipping: {batch['name']}")
                continue

            # Forward pass (mixed precision)
            with torch.amp.autocast(device_type=device, dtype=amp_dtype, enabled=use_autocast):
                results = model(image)
                pred_full = results["output"]  # (B, 8, D, H, W) at full res

                if output_sigmoid:
                    pred = torch.sigmoid(pred_full)
                else:
                    pred = pred_full

                # Upsample targets and mask to full resolution
                if step > 1:
                    targets = F.interpolate(targets, scale_factor=step, mode='trilinear', align_corners=False)
                    mask_up = F.interpolate(mask.float(), scale_factor=step, mode='trilinear', align_corners=False)
                    mask = (mask_up >= 1.0 - 1e-6).float()
                    dir_weight = F.interpolate(dir_weight, scale_factor=step, mode='trilinear', align_corners=False)

                # Per-channel-group losses with multi-scale
                loss_cos = scale_loss(pred[:, 0:1], targets[:, 0:1], mask=mask)
                loss_mag = scale_loss(pred[:, 1:2], targets[:, 1:2], mask=mask)
                loss_dir = scale_loss(pred[:, 2:8], targets[:, 2:8], mask=mask, weight=dir_weight)
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
                print(f"  NaN/Inf at step {global_step}: loss={loss.item()}  cos={loss_cos.item()}  mag={loss_mag.item()}  dir={loss_dir.item()}")

            if global_step % 10 == 0:
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/loss_cos", loss_cos.item(), global_step)
                writer.add_scalar("train/loss_mag", loss_mag.item(), global_step)
                writer.add_scalar("train/loss_dir", loss_dir.item(), global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

            if global_step % 1000 == 0:
                vis_pred = pred.clamp(0, 1) if not output_sigmoid else pred
                _log_vis(writer, "train", image, vis_pred, targets, mask, global_step)

            global_step += 1

        cosine_scheduler.step()

        # Validation
        val_loss = _evaluate(
            model, eval_loader, scale_loss, step, device, writer, global_step,
            w_cos, w_mag, w_dir, amp_dtype=amp_dtype, use_autocast=use_autocast,
            output_sigmoid=output_sigmoid,
        )

        mean_train = sum(epoch_losses) / max(len(epoch_losses), 1)
        print(
            f"epoch {epoch + 1}/{epochs}  "
            f"train={mean_train:.4f}  val={val_loss:.4f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        # Checkpoints
        ckpt_data = {
            "state_dict": model.state_dict(),
            "norm_type": norm_type,
            "upsample_mode": upsample_mode,
            "precision": precision,
            "output_sigmoid": output_sigmoid,
        }
        torch.save(ckpt_data, run_dir / "model_current.pt")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt_data, run_dir / "model_best.pt")
            print(f"  -> new best val loss: {val_loss:.4f}")

    writer.close()
    print(f"[train] done. Logs & checkpoints in {run_dir}")


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    scale_loss: ScaleSpaceLoss3D,
    step: int,
    device: str,
    writer: SummaryWriter,
    global_step: int,
    w_cos: float,
    w_mag: float,
    w_dir: float,
    amp_dtype: torch.dtype = torch.bfloat16,
    use_autocast: bool = True,
    output_sigmoid: bool = True,
) -> float:
    model.eval()
    losses: List[float] = []
    losses_cos: List[float] = []
    losses_mag: List[float] = []
    losses_dir: List[float] = []
    vis_done = False

    with torch.no_grad(), torch.amp.autocast(device_type=device, dtype=amp_dtype, enabled=use_autocast):
        for batch in loader:
            image = batch["image"].to(device)
            normal = batch["normal"].to(device)
            winding = batch["winding"].to(device)
            validity = batch["validity"].to(device)
            density = batch["density"].to(device)

            targets, mask, dir_weight = compute_targets_3d(normal, winding, validity, density)
            results = model(image)
            pred_full = results["output"]
            if output_sigmoid:
                pred = torch.sigmoid(pred_full)
            else:
                pred = pred_full.clamp(0, 1)

            # Upsample targets and mask to full resolution
            if step > 1:
                targets = F.interpolate(targets, scale_factor=step, mode='trilinear', align_corners=False)
                mask_up = F.interpolate(mask.float(), scale_factor=step, mode='trilinear', align_corners=False)
                mask = (mask_up >= 1.0 - 1e-6).float()
                dir_weight = F.interpolate(dir_weight, scale_factor=step, mode='trilinear', align_corners=False)

            loss_cos = scale_loss(pred[:, 0:1], targets[:, 0:1], mask=mask)
            loss_mag = scale_loss(pred[:, 1:2], targets[:, 1:2], mask=mask)
            loss_dir = scale_loss(pred[:, 2:8], targets[:, 2:8], mask=mask, weight=dir_weight)
            loss = w_cos * loss_cos + w_mag * loss_mag + w_dir * loss_dir
            losses.append(loss.item())
            losses_cos.append(loss_cos.item())
            losses_mag.append(loss_mag.item())
            losses_dir.append(loss_dir.item())

            if not vis_done:
                _log_vis(writer, "val", image, pred, targets, mask, global_step)
                vis_done = True

    n = max(len(losses), 1)
    mean_loss = sum(losses) / n
    writer.add_scalar("val/loss", mean_loss, global_step)
    writer.add_scalar("val/loss_cos", sum(losses_cos) / n, global_step)
    writer.add_scalar("val/loss_mag", sum(losses_mag) / n, global_step)
    writer.add_scalar("val/loss_dir", sum(losses_dir) / n, global_step)
    return mean_loss


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train 3D UNet on fitted.zarr labels from raw CT data.",
    )
    parser.add_argument("--images-dir", type=str, required=True,
                        help="Directory with CT image TIFFs (e.g. sample_00033.tif).")
    parser.add_argument("--label-dir", type=str, required=True,
                        help="Directory with per-sample fitted.zarr labels.")
    parser.add_argument("--log-dir", type=str, default="runs/unet3d",
                        help="TensorBoard log & checkpoint directory.")
    parser.add_argument("--run-name", type=str, default="3d",
                        help="Short name for the run subdirectory.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--patch-size", type=int, default=128,
                        help="Full-res patch size (must be divisible by step).")
    parser.add_argument("--w-cos", type=float, default=1.0)
    parser.add_argument("--w-mag", type=float, default=1.0)
    parser.add_argument("--w-dir", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-fraction", type=float, default=0.15,
                        help="Fraction of samples for validation.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--weights", type=str, default=None,
                        help="Checkpoint to resume from.")
    parser.add_argument("--stats-filter", type=str, default="losses.pred_dt.max",
                        help="Dotted JSON path in stats.json to filter samples "
                             "(set to empty string to disable).")
    parser.add_argument("--stats-threshold", type=float, default=5.0,
                        help="Skip samples where stats-filter value >= this.")
    parser.add_argument("--norm-type", type=str, default="instance",
                        choices=["instance", "group", "none"],
                        help="Normalization type (default: instance). "
                             "Saved in checkpoint for auto-detection at inference.")
    parser.add_argument("--upsample-mode", type=str, default="trilinear",
                        choices=["transpconv", "trilinear", "pixelshuffle"],
                        help="Decoder upsample mode (default: trilinear). "
                             "Saved in checkpoint for auto-detection at inference.")
    parser.add_argument("--precision", type=str, default="bf16",
                        choices=["bf16", "fp16", "fp32"],
                        help="Training precision (default: bf16).")
    parser.add_argument("--output-sigmoid", action="store_true", default=False,
                        help="Apply sigmoid to model output (default: off, use raw output + clamp).")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    train(
        images_dir=args.images_dir,
        label_dir=args.label_dir,
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
        stats_filter=args.stats_filter or None,
        stats_threshold=args.stats_threshold,
        norm_type=args.norm_type,
        upsample_mode=args.upsample_mode,
        precision=args.precision,
        output_sigmoid=args.output_sigmoid,
    )


if __name__ == "__main__":
    main()
