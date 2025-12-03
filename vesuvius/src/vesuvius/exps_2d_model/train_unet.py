from pathlib import Path
from typing import List, Tuple, Union, Optional, Set
from datetime import datetime
import time
import math
  
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import tifffile
import numpy as np
import kornia.augmentation as K
from gen_post_data import compute_label_supervision
from common import UNet, load_unet


class TiffLayerDataset(Dataset):
    """
    Dataset over individual layers of multi-layer TIFFs.

    For each TIFF in images_dir & its corresponding label TIFF in labels_dir,
    every layer (page) is treated as an independent sample.
    """

    def __init__(self, images_dir: Union[str, Path], labels_dir: Union[str, Path], patch_size: int = 256, random_crop: bool = True) -> None:
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.patch_size = patch_size
        self.random_crop = random_crop

        if not self.images_dir.is_dir():
            raise ValueError(f"images_dir does not exist: {self.images_dir}")
        if not self.labels_dir.is_dir():
            raise ValueError(f"labels_dir does not exist: {self.labels_dir}")

        self.samples: List[Tuple[Path, Path, int]] = []
        image_paths = sorted(self.images_dir.glob("*.tif"))
        if not image_paths:
            raise ValueError(f"No .tif files found in {self.images_dir}")

        for img_path in image_paths:
            # image:  sample_XYZ.tif
            # label:  sample_XYZ_surface.tif
            stem = img_path.stem
            label_name = f"{stem}_surface.tif"
            label_path = self.labels_dir / label_name
            if not label_path.is_file():
                raise ValueError(f"Missing label TIFF for {img_path.name} (expected {label_name}) in {self.labels_dir}")

            with tifffile.TiffFile(img_path) as tif:
                series = tif.series[0]
                shape = series.shape
            if len(shape) == 2:
                num_layers = 1
            elif len(shape) == 3:
                num_layers = shape[0]
            else:
                raise ValueError(f"Unsupported TIFF shape {shape} for {img_path}")

            for layer_idx in range(num_layers):
                self.samples.append((img_path, label_path, layer_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str, torch.Tensor, torch.Tensor, torch.Tensor]:
        img_path, label_path, layer_idx = self.samples[index]
 
        # Load only the required layer from each TIFF instead of the full stack.
        with tifffile.TiffFile(img_path) as tif:
            series = tif.series[0]
            if len(series.shape) == 2:
                image = series.asarray()
            else:
                image = series.asarray(key=layer_idx)
 
        with tifffile.TiffFile(label_path) as tif:
            series = tif.series[0]
            if len(series.shape) == 2:
                label = series.asarray()
            else:
                label = series.asarray(key=layer_idx)
 
        # Convert image to 8-bit if it is 16-bit (some stacks are uint8, some uint16).
        # 16-bit [0,65535] -> 8-bit [0,255] via integer division by 257.
        if image.dtype == np.uint16:
            image = (image // 257).astype(np.uint8)
 
        image_tensor = torch.from_numpy(image).unsqueeze(0).float()  # (1,H,W), CPU
        label_tensor = torch.from_numpy(label).long()  # (H,W) in {0,1,2}, CPU
 
        # Crop to a fixed patch size to normalize input sizes.
        ph = self.patch_size
        h, w = label_tensor.shape
        if h < ph or w < ph:
            raise ValueError(
                f"Sample from {img_path} layer {layer_idx} is smaller ({h}x{w}) than patch_size {ph}"
            )
 
        if self.random_crop:
            # Choose a random top-left corner and apply the same crop to image and label.
            top = torch.randint(0, h - ph + 1, (1,)).item()
            left = torch.randint(0, w - ph + 1, (1,)).item()
        else:
            # Deterministic center crop for evaluation / visualization.
            top = (h - ph) // 2
            left = (w - ph) // 2
 
        image_tensor = image_tensor[:, top : top + ph, left : left + ph]
        label_tensor = label_tensor[top : top + ph, left : left + ph]
 
        # Heavy geometry-based supervision is computed here on CPU only.
        sup = compute_label_supervision(label_tensor.numpy(), dbg=False)
        frac_pos = torch.from_numpy(sup["frac_pos"].astype(np.float32, copy=False))
        outer_cc_idx = torch.from_numpy(sup["outer_cc_idx"].astype(np.uint8, copy=False))
        mono = torch.from_numpy(sup["mono"].astype(np.float32, copy=False))
 
        sample_id = f"{img_path.name}[{layer_idx}]"
        return image_tensor, label_tensor, sample_id, frac_pos, outer_cc_idx, mono




def masked_mse_loss(pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Compute MSE using labels with semantics:
    0 -> target 0, contributes to loss
    1 -> target 1, contributes to loss (up-weighted so positives & negatives
         have equal total contribution)
    2 -> ignore (no contribution)

    The loss is class-balanced over non-ignored pixels so that the sum of
    positive weights equals the sum of negative weights.
    """
    if pred.ndim != 4 or pred.size(1) != 1:
        raise ValueError(f"Expected pred shape (N,1,H,W), got {tuple(pred.shape)}")

    pred = pred.squeeze(1)  # (N,H,W)

    if labels.shape != pred.shape:
        raise ValueError(f"pred & labels must have same spatial shape, got {tuple(pred.shape)} & {tuple(labels.shape)}")

    # Build masks
    with torch.no_grad():
        valid_mask = labels != 2
        pos_mask = (labels == 1) & valid_mask
        neg_mask = (labels == 0) & valid_mask

        n_pos = int(pos_mask.sum().item())
        n_neg = int(neg_mask.sum().item())

    if not valid_mask.any():
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    target = (labels == 1).float()

    diff = (pred - target) ** 2  # (N,H,W)

    # Per-pixel weights so that total positive weight ~= total negative weight.
    weights = torch.zeros_like(diff)

    if n_neg > 0:
        weights[neg_mask] = 1.0

    if n_pos > 0:
        if n_neg > 0:
            pos_weight = n_neg / max(n_pos, 1)
        else:
            pos_weight = 1.0
        weights[pos_mask] = float(pos_weight)

    # Restrict to supervised pixels
    diff = diff[valid_mask]
    weights = weights[valid_mask]

    if weights.sum() == 0:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    return (diff * weights).sum() / weights.sum()


class ImageGradientLoss(nn.Module):
    """
    Image-gradient loss wrapper.
 
    Given a base loss (e.g. nn.L1Loss or nn.MSELoss), this computes
    horizontal and vertical finite-difference gradients for both
    prediction and target. Optionally a per-pixel mask can be applied
    so that only masked regions contribute to the loss.
    """
 
    def __init__(self, base_loss: nn.Module) -> None:
        super().__init__()
        self.base_loss = base_loss
 
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(f"pred & target must have same shape, got {tuple(pred.shape)} & {tuple(target.shape)}")
        if pred.ndim != 4:
            raise ValueError(f"Expected pred/target shape (N,C,H,W), got {tuple(pred.shape)}")
 
        # Horizontal gradients: (..., H, W-1)
        dx_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        dx_target = target[:, :, :, 1:] - target[:, :, :, :-1]
 
        # Vertical gradients: (..., H-1, W)
        dy_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        dy_target = target[:, :, 1:, :] - target[:, :, :-1, :]
 
        if mask is not None:
            # Broadcast mask to (N,1,H,W) if given as (N,H,W).
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            # Derive masks for gradient locations.
            mask_x = mask[:, :, :, 1:] * mask[:, :, :, :-1]
            mask_y = mask[:, :, 1:, :] * mask[:, :, :-1, :]
            dx_pred = dx_pred * mask_x
            dx_target = dx_target * mask_x
            dy_pred = dy_pred * mask_y
            dy_target = dy_target * mask_y
 
        loss_x = self.base_loss(dx_pred, dx_target)
        loss_y = self.base_loss(dy_pred, dy_target)
        return 0.5 * (loss_x + loss_y)


class ScaleSpaceLoss(nn.Module):
    """
    Multi-scale (scale-space) loss wrapper.
 
    Given a base loss and a number of scales N, this applies the base loss
    at the original resolution and then repeatedly applies 2x2 pooling
    (stride 2) to prediction & target, computing the loss at each scale
    and returning the sum over scales.
 
    Optionally a per-pixel mask can be supplied; it is downscaled together
    with the inputs and applied multiplicatively at each scale. Mask
    downscaling uses a logical "all-valid" semantics: a pooled pixel is
    marked valid only if *all* contributing pixels were valid.
    """
 
    def __init__(self, base_loss: nn.Module, num_scales: int) -> None:
        super().__init__()
        if num_scales < 1:
            raise ValueError(f"num_scales must be >= 1, got {num_scales}")
        self.base_loss = base_loss
        self.num_scales = num_scales
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
 
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(f"pred & target must have same shape, got {tuple(pred.shape)} & {tuple(target.shape)}")
        if pred.ndim != 4:
            raise ValueError(f"Expected pred/target shape (N,C,H,W), got {tuple(pred.shape)}")
 
        x = pred
        y = target
        m = None
        if mask is not None:
            m = mask
            if m.ndim == 3:
                m = m.unsqueeze(1)
            # Ensure mask is in {0,1} for logical operations.
            m = (m > 0.5).float()
 
        total_loss = torch.zeros((), device=pred.device, dtype=pred.dtype)
 
        for scale in range(self.num_scales):
            if m is not None:
                total_loss = total_loss + self.base_loss(x * m, y * m)
            else:
                total_loss = total_loss + self.base_loss(x, y)
 
            if scale < self.num_scales - 1:
                # If spatial resolution is already below 2x2, stop downscaling.
                if x.size(2) < 2 or x.size(3) < 2:
                    break
                x = self.pool(x)
                y = self.pool(y)
                if m is not None:
                    # Downscale mask with "all-valid" semantics:
                    # pooled pixel is valid (1) only if all contributing pixels are valid.
                    invalid = 1.0 - m
                    invalid_pooled = torch.nn.functional.max_pool2d(invalid, kernel_size=2, stride=2)
                    m = 1.0 - invalid_pooled
 
        return total_loss


class MaskedMSE(nn.Module):
    """
    MSE loss with optional per-pixel mask.

    If mask is provided, it must be broadcastable to pred/target; the loss
    is computed as sum(mask * (pred - target)^2) / sum(mask). If mask is
    None, this reduces to standard mean-squared error.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(f"pred & target must have same shape, got {tuple(pred.shape)} & {tuple(target.shape)}")
        diff = (pred - target) ** 2
        if mask is None:
            return diff.mean()
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        diff = diff * mask
        denom = mask.sum()
        if denom <= 0:
            return torch.zeros((), device=pred.device, dtype=pred.dtype)
        return diff.sum() / denom


def compute_geom_targets(
    frac_pos_batch: torch.Tensor,
    outer_cc_idx_batch: torch.Tensor,
    mono_batch: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute geometry-based supervision from precomputed geometry maps.
 
    Inputs (already on the correct device):
    - frac_pos_batch: (N,H,W) float32 (negative sentinel outside valid region)
    - outer_cc_idx_batch: (N,H,W) integer, 0 outside outer CCs, 1..K inside
    - mono_batch: (N,H,W) float32 in [0,1]
 
    Returns:
    - cos_gt:   (N,1,H,W) float32 in [0,1]
    - frac_norm: (N,1,H,W) float32, raw frac_pos values (not normalized; zero outside valid region)
    - frac_mask: (N,1,H,W) float32 mask (1=valid, 0=ignored)
    """
    frac_pos = frac_pos_batch  # (N,H,W)
    outer_idx = outer_cc_idx_batch
    mono = mono_batch
 
    mono_clamped = mono.clamp(0.0, 1.0)
    cos_gt = 0.5 - 0.5 * torch.cos(math.pi * mono_clamped)
    cos_gt = cos_gt.unsqueeze(1)  # (N,1,H,W)
 
    valid_frac = (outer_idx > 0) & (frac_pos > -0.5)
    frac_raw = torch.zeros_like(frac_pos)
    frac_raw[valid_frac] = frac_pos[valid_frac]
    frac_norm = frac_raw.unsqueeze(1)                    # (N,1,H,W), raw values
    frac_mask = valid_frac.unsqueeze(1).float()          # (N,1,H,W)
 
    return cos_gt, frac_norm, frac_mask
 
 
def _gaussian_blur_2d(
    x: torch.Tensor,
    kernel_size: int = 5,
    sigma: float = 1.0,
) -> torch.Tensor:
    """
    Minimal 2D Gaussian blur implemented via depthwise conv, to avoid relying
    on torch.nn.functional.gaussian_blur (which may be missing in older PyTorch).
 
    x: (N,1,H,W)
    """
    if kernel_size <= 1 or sigma <= 0.0:
        return x
 
    # 1D Gaussian kernel.
    device = x.device
    dtype = x.dtype
    ks = int(kernel_size)
    half = ks // 2
    coords = torch.arange(-half, half + 1, device=device, dtype=dtype)
    gauss_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
    gauss_1d = gauss_1d / gauss_1d.sum()
 
    # Separable 2D kernel via two 1D convolutions.
    gauss_x = gauss_1d.view(1, 1, 1, ks)  # (outC,inC,H,W)
    gauss_y = gauss_1d.view(1, 1, ks, 1)
 
    padding = (half, half)
    x = F.conv2d(x, gauss_x, padding=(0, padding[0]))
    x = F.conv2d(x, gauss_y, padding=(padding[1], 0))
    return x
 
 
def erode_mask(
    mask: torch.Tensor,
    kernel_size: int = 7,
) -> torch.Tensor:
    """
    Binary mask erosion using max-pooling on the inverted mask.
 
    A pixel remains valid (1) only if all pixels in its local window were valid
    in the original mask. With the default kernel_size=5 this removes roughly
    a 2-pixel band along mask boundaries.
 
    mask: (N,1,H,W) or (N,H,W) with values in {0,1}.
    """
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    m = (mask > 0.5).float()
    if kernel_size <= 1:
        return m
    invalid = 1.0 - m
    pad = kernel_size // 2
    invalid_pooled = F.max_pool2d(invalid, kernel_size=kernel_size, stride=1, padding=pad)
    eroded = (invalid_pooled == 0).float()
    return eroded
 
 
def compute_frac_mag_dir(
    frac_norm: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Derive gradient magnitude & direction from a fractional-position tensor
    frac_norm of shape (N,1,H,W).
 
    Direction is encoded as 0.5 + 0.5 * cos(2*theta) where theta is the gradient
    direction, matching the supervision used for the direction branch.
 
    A small blur is applied before finite differencing to reduce noise and
    aliasing in the gradient estimates.
    """
    f = frac_norm
    # Apply isotropic Gaussian blur before computing spatial gradients.
    f_blur = _gaussian_blur_2d(f, kernel_size=7, sigma=2.0)
 
    gx = torch.zeros_like(f_blur)
    gy = torch.zeros_like(f_blur)
    gx[:, :, :, :-1] = f_blur[:, :, :, 1:] - f_blur[:, :, :, :-1]
    gy[:, :, :-1, :] = f_blur[:, :, 1:, :] - f_blur[:, :, :-1, :]
    eps = 1e-8
    mag = torch.sqrt(gx * gx + gy * gy + eps)
 
    # Gradient direction ground truth encoded as cos(2*theta).
    # Use identity cos(2*theta) = (gx^2 - gy^2) / (gx^2 + gy^2).
    r2 = gx * gx + gy * gy + eps
    cos2theta = (gx * gx - gy * gy) / r2
    dir_enc = 0.5 + 0.5 * cos2theta
    return mag, dir_enc
 
 
def compute_frac_grad_mag(frac_norm: torch.Tensor) -> torch.Tensor:
    """
    Backwards-compatible helper returning only gradient magnitude, implemented
    via compute_frac_mag_dir to avoid duplicating finite-difference logic.
    """
    mag, _ = compute_frac_mag_dir(frac_norm)
    return mag


def compute_geom_losses(
    pred: torch.Tensor,
    cos_gt: torch.Tensor,
    frac_norm: torch.Tensor,
    frac_mask: torch.Tensor,
    outer_cc_idx: torch.Tensor,
    cos_scale_loss: ScaleSpaceLoss,
    geom_scale_loss: ScaleSpaceLoss,
    w_cos: float,
    w_mag: float,
    w_dir: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute total geometry loss given predictions and targets.
 
    Total loss is a weighted sum of three terms:
    - w_cos * cos_loss  : multi-scale (scale-space) MSE on the cosine branch (channel 0).
    - w_mag * loss_mag  : multi-scale MSE on gradient magnitude branch (channel 1).
    - w_dir * loss_dir  : multi-scale MSE on gradient direction branch (channel 2,
                          encoded via cos(2*theta)).
    """
    if pred.ndim != 4 or pred.size(1) < 3:
        raise ValueError(f"Expected pred shape (N,3,H,W) or more channels, got {tuple(pred.shape)}")
 
    # Split prediction into branches.
    cos_pred = pred[:, 0:1]
    mag_pred = pred[:, 1:2]
    dir_pred = pred[:, 2:3]
 
    # Cosine loss: use multi-scale MSE with the same validity mask used for frac.
    loss_cos = cos_scale_loss(cos_pred, cos_gt, mask=frac_mask)
 
    # Geometry supervision is derived from frac_norm; gradients are computed
    # on the raw field, and masking is applied only afterwards for losses/vis.
    # frac_norm: (N,1,H,W), frac_mask: (N,1,H,W)
    m_full = (frac_mask > 0.5).float()
    m_eroded = erode_mask(m_full)
 
    # Magnitude & direction ground truth from shared helper (no masking on input).
    mag_gt, dir_gt = compute_frac_mag_dir(frac_norm)
 
    # Validity for gradients: both neighbors must be valid inside the eroded mask.
    m = m_eroded > 0.5
    m_x = torch.zeros_like(m)
    m_y = torch.zeros_like(m)
    m_x[:, :, :, :-1] = m[:, :, :, 1:] & m[:, :, :, :-1]
    m_y[:, :, :-1, :] = m[:, :, 1:, :] & m[:, :, :-1, :]
    m_grad = m_x & m_y  # (N,1,H,W)
 
    # outer_cc_idx is (N,H,W) or (N,1,H,W); bring to (N,H,W).
    if outer_cc_idx.ndim == 4:
        outer_idx_hw = outer_cc_idx[:, 0]
    else:
        outer_idx_hw = outer_cc_idx
 
    # Valid mask over H,W.
    valid_mask_hw = m_grad.squeeze(1)
 
    device = pred.device
    dtype = pred.dtype
    loss_mag = torch.zeros((), device=device, dtype=dtype)
    loss_dir = torch.zeros((), device=device, dtype=dtype)
 
    max_cc = int(outer_idx_hw.max().item())
 
    if max_cc > 0:
        for k in range(1, max_cc + 1):
            cc_mask_hw = (outer_idx_hw == k) & valid_mask_hw
            if not cc_mask_hw.any():
                continue
            cc_mask = cc_mask_hw.unsqueeze(1).float()  # (N,1,H,W)
 
            # Multi-scale MSE on magnitude and direction branches.
            loss_mag_k = geom_scale_loss(mag_pred, mag_gt, mask=cc_mask)
            loss_dir_k = geom_scale_loss(dir_pred, dir_gt, mask=cc_mask)
            loss_mag = loss_mag + loss_mag_k
            loss_dir = loss_dir + loss_dir_k
    else:
        # Fallback: no outer CCs; use global mask if there are any valid gradient pixels.
        valid_mask = valid_mask_hw
        if valid_mask.any():
            cc_mask = valid_mask.unsqueeze(1).float()
            loss_mag = geom_scale_loss(mag_pred, mag_gt, mask=cc_mask)
            loss_dir = geom_scale_loss(dir_pred, dir_gt, mask=cc_mask)
        else:
            loss_mag = torch.zeros((), device=device, dtype=dtype)
            loss_dir = torch.zeros((), device=device, dtype=dtype)
 
    total = w_cos * loss_cos + w_mag * loss_mag + w_dir * loss_dir
    return total, w_cos * loss_cos, w_mag * loss_mag, w_dir * loss_dir


def train(
    images_dir: Union[str, Path],
    labels_dir: Union[str, Path],
    log_dir: Union[str, Path],
    run_name: str,
    epochs: int = 10,
    batch_size: int = 2,
    lr: float = 1e-3,
    w_cos: float = 1.0,
    w_mag: float = 1.0,
    w_dir: float = 1.0,
    num_workers: int = 16,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    bench: bool = False,
    weights: Optional[str] = None,
) -> None:
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    base_log_dir = Path(log_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_log_dir / f"{timestamp}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    base_dataset = TiffLayerDataset(images_dir, labels_dir, patch_size=256, random_crop=True)

    # Build fixed test set: specific (numeric sample_id, layer) pairs.
    # Using ints for sample ids makes matching robust to leading zeros in filenames.
    test_specs = {
        (806, 0),
        (806, 128),
        (706, 0),
        (706, 128),
    }
    train_indices: List[int] = []
    test_indices: List[int] = []
    found_specs = set()

    for idx, (img_path, _label_path, layer_idx) in enumerate(base_dataset.samples):
        stem = img_path.stem
        parts = stem.split("_")
        sample_id = None
        if parts:
            try:
                sample_id = int(parts[-1])
            except ValueError:
                sample_id = None

        if sample_id is None:
            continue

        key = (sample_id, layer_idx)
        if key in test_specs:
            test_indices.append(idx)
            found_specs.add(key)
        else:
            train_indices.append(idx)

    missing_specs = test_specs - found_specs
    if missing_specs:
        print(f"Warning: some requested test samples not found in dataset: {sorted(missing_specs)}")

    train_dataset = Subset(base_dataset, train_indices)

    # For evaluation / visualization we use deterministic center crops.
    eval_base_dataset = TiffLayerDataset(
        images_dir,
        labels_dir,
        patch_size=base_dataset.patch_size,
        random_crop=False,
    )
    eval_dataset = Subset(eval_base_dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
 
 
    model = load_unet(
        device=device,
        weights=weights,
        in_channels=1,
        out_channels=3,
        base_channels=32,
        num_levels=6,
        max_channels=1024,
    )

    aug = nn.Sequential(
        K.RandomBrightness(0.3, p=0.5),
        K.RandomContrast(0.3, p=0.5),
        K.RandomGamma((0.7, 2.0), p=0.5),
        # K.RandomGaussianNoise(mean=0.0, std=0.05, p=0.5),
        K.RandomGaussianBlur((3, 3), (0.1, 2.0), p=0.3),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Base loss & wrappers for scale-space losses.
    base_mse = nn.MSELoss(reduction="mean")
    cos_scale_loss = ScaleSpaceLoss(base_mse, num_scales=3)
    geom_scale_loss = ScaleSpaceLoss(base_mse, num_scales=6)
 
    writer = SummaryWriter(log_dir=str(run_dir))
    train_vis_images_raw: Optional[torch.Tensor] = None
    train_vis_images: Optional[torch.Tensor] = None
    train_vis_preds: Optional[torch.Tensor] = None
    train_vis_targets: Optional[torch.Tensor] = None
    train_vis_ids: Optional[List[str]] = None
    train_vis_cos_gt: Optional[torch.Tensor] = None
    train_vis_cos_pred: Optional[torch.Tensor] = None
    train_vis_frac_gt: Optional[torch.Tensor] = None       # underlying frac field for vis
    train_vis_frac_mag_gt: Optional[torch.Tensor] = None   # gradient magnitude GT for vis
    train_vis_frac_mag_pred: Optional[torch.Tensor] = None # gradient magnitude prediction for vis
    train_vis_dir_gt: Optional[torch.Tensor] = None        # gradient direction GT for vis
    train_vis_dir_pred: Optional[torch.Tensor] = None      # gradient direction prediction for vis
    train_vis_outer_idx: Optional[torch.Tensor] = None
    test_gt_logged = False

    def _normalize_for_display(x: torch.Tensor) -> torch.Tensor:
        """
        Per-sample min-max normalization to [0,1] for a single tensor (N,C,H,W),
        used e.g. for visualizing index masks.
        """
        if x.ndim != 4:
            return x
        x_flat = x.reshape(x.size(0), -1)
        mins = x_flat.min(dim=1)[0].view(-1, 1, 1, 1)
        maxs = x_flat.max(dim=1)[0].view(-1, 1, 1, 1)
        denom = (maxs - mins).clamp_min(1e-6)
        x_norm = (x - mins) / denom
        return torch.clamp(x_norm, 0.0, 1.0)

    def _normalize_frac_pair_for_display(
        pred: torch.Tensor,
        gt: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Per-sample normalization of frac pred & gt using a shared range
        per sample, but with per-tensor mean subtraction so we only match
        range (scale), not absolute offset.
        """
        if pred.ndim != 4 or gt.ndim != 4:
            return pred, gt

        # Subtract per-sample mean from each tensor separately.
        # Shapes: (N,1,H,W) -> means (N,1,1,1)
        pred_flat = pred.reshape(pred.size(0), -1)
        gt_flat = gt.reshape(gt.size(0), -1)
        pred_mean = pred_flat.mean(dim=1).view(-1, 1, 1, 1)
        gt_mean = gt_flat.mean(dim=1).view(-1, 1, 1, 1)
        pred_centered = pred - pred_mean
        gt_centered = gt - gt_mean

        # Stack along channel dim: (N,2,H,W)
        stacked = torch.cat([pred_centered, gt_centered], dim=1)
        flat = stacked.reshape(stacked.size(0), stacked.size(1), -1)
        # Per-sample min/max over both channels (shared range).
        mins = flat.min(dim=2)[0].min(dim=1)[0].view(-1, 1, 1, 1)
        maxs = flat.max(dim=2)[0].max(dim=1)[0].view(-1, 1, 1, 1)
        denom = (maxs - mins).clamp_min(1e-6)
        stacked_norm = (stacked - mins) / denom
        stacked_norm = torch.clamp(stacked_norm, 0.0, 1.0)
        pred_norm = stacked_norm[:, 0:1]
        gt_norm = stacked_norm[:, 1:2]
        return pred_norm, gt_norm
 
    def evaluate_and_visualize(global_step: int) -> None:
        nonlocal test_gt_logged
        model.eval()
        test_losses: List[float] = []
        test_cos_losses: List[float] = []
        test_frac_mag_losses: List[float] = []
        test_frac_dir_losses: List[float] = []
        vis_images: List[torch.Tensor] = []
        vis_cos_pred: List[torch.Tensor] = []
        vis_cos_gt: List[torch.Tensor] = []
        vis_frac: List[torch.Tensor] = []            # underlying frac field
        vis_frac_mag_pred: List[torch.Tensor] = []   # magnitude predictions
        vis_frac_mag_gt: List[torch.Tensor] = []     # magnitude GT
        vis_dir_pred: List[torch.Tensor] = []        # direction predictions
        vis_dir_gt: List[torch.Tensor] = []          # direction GT
        vis_outer_idx: List[torch.Tensor] = []
 
        with torch.no_grad():
            for images, labels, _ids, frac_pos_batch, outer_cc_idx_batch, mono_batch in eval_loader:
                # Move full batch (inputs, labels, geometry) to device as soon as it is available.
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                frac_pos_batch = frac_pos_batch.to(device, non_blocking=True)
                outer_cc_idx_batch = outer_cc_idx_batch.to(device, non_blocking=True)
                mono_batch = mono_batch.to(device, non_blocking=True)
 
                images = images.float() / 255.0
 
                pred = model(images)
                cos_pred = pred[:, 0:1]
                mag_pred = pred[:, 1:2]
                dir_pred = pred[:, 2:3]
 
                cos_gt, frac_norm, frac_mask = compute_geom_targets(
                    frac_pos_batch,
                    outer_cc_idx_batch,
                    mono_batch,
                )
                frac_mask_eroded = erode_mask(frac_mask)
                # Compute GT gradients on raw frac_norm; apply masking only in losses/vis.
                mag_gt_vis, dir_gt_vis = compute_frac_mag_dir(frac_norm)
 
                loss, loss_cos, loss_frac_mag, loss_frac_dir = compute_geom_losses(
                    pred,
                    cos_gt,
                    frac_norm,
                    frac_mask,
                    outer_cc_idx_batch,
                    cos_scale_loss,
                    geom_scale_loss,
                    w_cos,
                    w_mag,
                    w_dir,
                )
                test_losses.append(float(loss.item()))
                test_cos_losses.append(float(loss_cos.item()))
                test_frac_mag_losses.append(float(loss_frac_mag.item()))
                test_frac_dir_losses.append(float(loss_frac_dir.item()))
 
                if len(vis_images) < 4:
                    # Mask gradient vis by the eroded frac mask so edges at the masked boundary disappear.
                    mag_pred_masked = mag_pred * frac_mask_eroded
                    mag_gt_vis_masked = mag_gt_vis * frac_mask_eroded
                    dir_pred_masked = dir_pred * frac_mask_eroded
                    dir_gt_vis_masked = dir_gt_vis * frac_mask_eroded
                    frac_vis_masked = frac_norm * frac_mask_eroded
 
                    vis_images.append(images.cpu())
                    vis_cos_pred.append(cos_pred.cpu())
                    vis_cos_gt.append(cos_gt.cpu())
                    vis_frac.append(frac_vis_masked.cpu())
                    vis_frac_mag_pred.append(mag_pred_masked.cpu())
                    vis_frac_mag_gt.append(mag_gt_vis_masked.cpu())
                    vis_dir_pred.append(dir_pred_masked.cpu())
                    vis_dir_gt.append(dir_gt_vis_masked.cpu())
                    vis_outer_idx.append(outer_cc_idx_batch.cpu())
 
        if test_losses:
            mean_loss = sum(test_losses) / len(test_losses)
            writer.add_scalar("test/loss", mean_loss, global_step)
            mean_cos = sum(test_cos_losses) / len(test_cos_losses)
            mean_frac_mag = sum(test_frac_mag_losses) / len(test_frac_mag_losses)
            mean_frac_dir = sum(test_frac_dir_losses) / len(test_frac_dir_losses)
            writer.add_scalar("test/cos_loss", mean_cos, global_step)
            writer.add_scalar("test/frac_mag_loss", mean_frac_mag, global_step)
            writer.add_scalar("test/frac_dir_loss", mean_frac_dir, global_step)
 
        if vis_images:
            images_grid = torch.cat(vis_images, dim=0)
            cos_pred_grid = torch.cat(vis_cos_pred, dim=0)
            cos_gt_grid = torch.cat(vis_cos_gt, dim=0)
            frac_grid = torch.cat(vis_frac, dim=0)
            frac_mag_pred_grid = torch.cat(vis_frac_mag_pred, dim=0)
            frac_mag_gt_grid = torch.cat(vis_frac_mag_gt, dim=0)
            dir_pred_grid = torch.cat(vis_dir_pred, dim=0)
            dir_gt_grid = torch.cat(vis_dir_gt, dim=0)
            outer_idx_grid = torch.cat(vis_outer_idx, dim=0).unsqueeze(1).float()
 
            frac_grid_disp = _normalize_for_display(frac_grid)
            frac_mag_pred_grid_disp, frac_mag_gt_grid_disp = _normalize_frac_pair_for_display(
                frac_mag_pred_grid, frac_mag_gt_grid
            )
            dir_pred_grid_disp, dir_gt_grid_disp = _normalize_frac_pair_for_display(
                dir_pred_grid, dir_gt_grid
            )
            outer_idx_disp = _normalize_for_display(outer_idx_grid)
 
            writer.add_images("test/input", images_grid, global_step)
            writer.add_images("test/cos_pred", cos_pred_grid, global_step)
            writer.add_images("test/cos_gt", cos_gt_grid, global_step)
            writer.add_images("test/frac", frac_grid_disp, global_step)
            writer.add_images("test/frac_mag_pred", frac_mag_pred_grid_disp, global_step)
            writer.add_images("test/frac_mag_gt", frac_mag_gt_grid_disp, global_step)
            writer.add_images("test/frac_dir_pred", dir_pred_grid_disp, global_step)
            writer.add_images("test/frac_dir_gt", dir_gt_grid_disp, global_step)
            writer.add_images("test/outer_idx", outer_idx_disp, global_step)
            if not test_gt_logged:
                test_gt_logged = True
 
            if (
                train_vis_images_raw is not None
                and train_vis_images is not None
                and train_vis_preds is not None
                and train_vis_targets is not None
                and train_vis_ids is not None
            ):
                n_test = images_grid.size(0)
                n_train = train_vis_images.size(0)
                n = min(n_test, n_train)
                if n > 0:
                    writer.add_images("train/input_raw", train_vis_images_raw[:n], global_step)
                    writer.add_images("train/input", train_vis_images[:n], global_step)
                    writer.add_images("train/pred", train_vis_preds[:n], global_step)
                    writer.add_images("train/target", train_vis_targets[:n], global_step)
 
                    if train_vis_cos_gt is not None and train_vis_cos_pred is not None:
                        writer.add_images("train/cos_gt", train_vis_cos_gt[:n], global_step)
                        writer.add_images("train/cos_pred", train_vis_cos_pred[:n], global_step)
                    if train_vis_frac_gt is not None:
                        frac_train_disp = _normalize_for_display(train_vis_frac_gt[:n])
                        writer.add_images("train/frac", frac_train_disp, global_step)
                    if train_vis_frac_mag_gt is not None and train_vis_frac_mag_pred is not None:
                        frac_pred_train_disp, frac_gt_train_disp = _normalize_frac_pair_for_display(
                            train_vis_frac_mag_pred[:n],
                            train_vis_frac_mag_gt[:n],
                        )
                        writer.add_images("train/frac_mag_gt", frac_gt_train_disp, global_step)
                        writer.add_images("train/frac_mag_pred", frac_pred_train_disp, global_step)
                    if train_vis_dir_gt is not None and train_vis_dir_pred is not None:
                        dir_pred_train_disp, dir_gt_train_disp = _normalize_frac_pair_for_display(
                            train_vis_dir_pred[:n],
                            train_vis_dir_gt[:n],
                        )
                        writer.add_images("train/frac_dir_gt", dir_gt_train_disp, global_step)
                        writer.add_images("train/frac_dir_pred", dir_pred_train_disp, global_step)
                    if train_vis_outer_idx is not None:
                        outer_idx_train = train_vis_outer_idx[:n].unsqueeze(1).float()
                        outer_idx_train_disp = _normalize_for_display(outer_idx_train)
                        writer.add_images("train/outer_idx", outer_idx_train_disp, global_step)
 
                    id_lines = []
                    for i in range(n):
                        if i < len(train_vis_ids):
                            id_lines.append(f"{i}: {train_vis_ids[i]}")
                    if id_lines:
                        writer.add_text("train/vis_ids", "\n".join(id_lines), global_step)
 
        torch.save(model.state_dict(), run_dir / "unet_current.pt")
 
        model.train()

    global_step = 0
    model.train()
    total_pixels = 0
    total_time = 0.0
    window_pixels = 0
    window_time = 0.0
    corrupt_samples = 0
    seen_samples = 0
    corrupt_ids: Set[str] = set()
    clean_ids: Set[str] = set()
    last_report_t = time.perf_counter() if bench else 0.0
    for epoch in range(epochs):
        print("starting training with epoch ", epoch)
        for batch_idx, (images, labels_cpu, sample_ids, frac_pos_batch, outer_cc_idx_batch, mono_batch) in enumerate(train_loader):
            step_start = time.perf_counter() if bench else 0.0
    
            # Move full batch (inputs, labels, geometry) to device as soon as it is available.
            images = images.to(device, non_blocking=True)
            labels = labels_cpu.to(device, non_blocking=True)  # (N,H,W) with values {0,1,2}
            frac_pos_batch = frac_pos_batch.to(device, non_blocking=True)
            outer_cc_idx_batch = outer_cc_idx_batch.to(device, non_blocking=True)
            mono_batch = mono_batch.to(device, non_blocking=True)

            seen_samples += images.size(0)
 
            # Detect saturated raw inputs (>=255 on >50% of pixels) and ignore them via labels==2.
            with torch.no_grad():
                high_mask = (images >= 255).view(images.size(0), -1)
                frac_high = high_mask.float().mean(dim=1)
                bad_mask = frac_high > 0.5
                if bad_mask.any():
                    bad_list = bad_mask.tolist()
                    for i, is_bad in enumerate(bad_list):
                        sid = sample_ids[i]
                        img_id = sid.split("[", 1)[0]
                        if is_bad:
                            corrupt_samples += 1
                            labels[i].fill_(2)
                            corrupt_ids.add(img_id)
                            print(
                                f"[corrupt] ignoring sample {sid} "
                                f"({corrupt_samples}/{seen_samples} images ignored so far; "
                                f"unique_corrupt={len(corrupt_ids)}, unique_clean={len(clean_ids)})"
                            )
                        else:
                            clean_ids.add(img_id)
 
            images = images.float() / 255.0  # (N,1,H,W) normalized to [0,1]
 
            # Spatial augmentations (flips & 90° rotations) applied jointly to images, labels & geometry maps.
            if torch.rand(1).item() < 0.5:
                images = torch.flip(images, dims=[3])
                labels = torch.flip(labels, dims=[2])
                frac_pos_batch = torch.flip(frac_pos_batch, dims=[2])
                outer_cc_idx_batch = torch.flip(outer_cc_idx_batch, dims=[2])
                mono_batch = torch.flip(mono_batch, dims=[2])
            if torch.rand(1).item() < 0.5:
                images = torch.flip(images, dims=[2])
                labels = torch.flip(labels, dims=[1])
                frac_pos_batch = torch.flip(frac_pos_batch, dims=[1])
                outer_cc_idx_batch = torch.flip(outer_cc_idx_batch, dims=[1])
                mono_batch = torch.flip(mono_batch, dims=[1])
            k = int(torch.randint(0, 4, (1,)).item())
            if k:
                images = torch.rot90(images, k, dims=(2, 3))
                labels = torch.rot90(labels, k, dims=(1, 2))
                frac_pos_batch = torch.rot90(frac_pos_batch, k, dims=(1, 2))
                outer_cc_idx_batch = torch.rot90(outer_cc_idx_batch, k, dims=(1, 2))
                mono_batch = torch.rot90(mono_batch, k, dims=(1, 2))
 
            raw_images = images.clone()
            images = aug(images)
            images = torch.clamp(images, 0.0, 1.0)

            # Geometry targets and model predictions on augmented tensors.
            cos_gt, frac_norm, frac_mask = compute_geom_targets(
                frac_pos_batch,
                outer_cc_idx_batch,
                mono_batch,
            )
            frac_mask_eroded = erode_mask(frac_mask)
            # Compute GT gradients on raw frac_norm; apply masking only in losses/vis.
            mag_gt_vis, dir_gt_vis = compute_frac_mag_dir(frac_norm)
 
            pred = model(images)
            cos_pred = pred[:, 0:1]
            mag_pred = pred[:, 1:2]
            dir_pred = pred[:, 2:3]
 
            loss, loss_cos, loss_frac_mag, loss_frac_dir = compute_geom_losses(
                pred,
                cos_gt,
                frac_norm,
                frac_mask,
                outer_cc_idx_batch,
                cos_scale_loss,
                geom_scale_loss,
                w_cos,
                w_mag,
                w_dir,
            )
 
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if bench:
                elapsed = time.perf_counter() - step_start
                pixels = images.size(0) * images.size(2) * images.size(3)
                total_pixels += pixels
                total_time += elapsed
                window_pixels += pixels
                window_time += elapsed
                now = time.perf_counter()
                if now - last_report_t >= 1.0 and window_time > 0.0:
                    if isinstance(device, str) and device.startswith("cuda"):
                        torch.cuda.synchronize()
                    inst_throughput = (window_pixels / 1e6) / window_time
                    avg_throughput = (total_pixels / 1e6) / total_time if total_time > 0.0 else 0.0
                    print(
                        f"[bench] epoch {epoch+1} batch {batch_idx}: "
                        f"inst={inst_throughput:.2f} Mpix/s, avg={avg_throughput:.2f} Mpix/s"
                    )
                    window_pixels = 0
                    window_time = 0.0
                    last_report_t = now

            writer.add_scalar("train/loss", float(loss.item()), global_step)
            writer.add_scalar("train/cos_loss", float(loss_cos.item()), global_step)
            writer.add_scalar("train/frac_mag_loss", float(loss_frac_mag.item()), global_step)
            writer.add_scalar("train/frac_dir_loss", float(loss_frac_dir.item()), global_step)
 
            if global_step % 100 == 0:
                with torch.no_grad():
                    n_vis = min(4, images.size(0))
                    frac_mask_vis = frac_mask_eroded[:n_vis]
                    frac_gt_vis_masked = frac_norm[:n_vis] * frac_mask_vis
                    mag_gt_vis_masked = mag_gt_vis[:n_vis] * frac_mask_vis
                    mag_pred_masked = mag_pred[:n_vis] * frac_mask_vis
                    dir_gt_vis_masked = dir_gt_vis[:n_vis] * frac_mask_vis
                    dir_pred_masked = dir_pred[:n_vis] * frac_mask_vis
 
                    train_vis_images_raw = raw_images[:n_vis].detach().cpu()
                    train_vis_images = images[:n_vis].detach().cpu()
                    train_vis_preds = cos_pred[:n_vis].detach().cpu()
                    train_vis_targets = cos_gt[:n_vis].detach().cpu()
                    train_vis_cos_gt = cos_gt[:n_vis].detach().cpu()
                    train_vis_cos_pred = cos_pred[:n_vis].detach().cpu()
                    train_vis_frac_gt = frac_gt_vis_masked.detach().cpu()
                    train_vis_frac_mag_gt = mag_gt_vis_masked.detach().cpu()
                    train_vis_frac_mag_pred = mag_pred_masked.detach().cpu()
                    train_vis_dir_gt = dir_gt_vis_masked.detach().cpu()
                    train_vis_dir_pred = dir_pred_masked.detach().cpu()
                    train_vis_outer_idx = outer_cc_idx_batch[:n_vis].detach().cpu()
                    train_vis_ids = list(sample_ids[:n_vis])
                evaluate_and_visualize(global_step)

            global_step += 1

        torch.save(model.state_dict(), run_dir / "unet_current.pt")
 
    if seen_samples > 0:
        pct = 100.0 * corrupt_samples / seen_samples
        print(
            f"[corrupt] total ignored samples: "
            f"{corrupt_samples}/{seen_samples} ({pct:.2f}%)"
        )
        print(f"[corrupt] unique corrupted samples: {len(corrupt_ids)}")
        print(f"[corrupt] unique non-corrupted samples: {len(clean_ids)}")
    else:
        print("[corrupt] no training samples seen, nothing to report")
 
    writer.close()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train U-Net on multi-layer TIFF data.")
    parser.add_argument("--images-dir", type=str, required=True, help="Directory with image TIFFs.")
    parser.add_argument("--labels-dir", type=str, required=True, help="Directory with label TIFFs.")
    parser.add_argument("--log-dir", type=str, default="runs/unet", help="Base directory for TensorBoard logs & checkpoints.")
    parser.add_argument("--run-name", type=str, required=True, help="Short name used to prefix the TensorBoard run subdirectory.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--w-cos", type=float, default=1.0, help="Weight for cosine branch loss term.")
    parser.add_argument("--w-mag", type=float, default=100.0, help="Weight for gradient-magnitude loss term.")
    parser.add_argument("--w-dir", type=float, default=1.0, help="Weight for gradient-direction loss term.")
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--device", type=str, default=None, help='Device string, e.g. "cuda" or "cpu".')
    parser.add_argument("--bench", action="store_true", help="If set, run data-loading benchmark instead of training.")
    parser.add_argument("--weights", type=str, default=None, help="Path to checkpoint file to load before training.")
 
    args = parser.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    train(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        log_dir=args.log_dir,
        run_name=args.run_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        w_cos=args.w_cos,
        w_mag=args.w_mag,
        w_dir=args.w_dir,
        num_workers=args.num_workers,
        device=device,
        bench=args.bench,
        weights=args.weights,
    )


if __name__ == "__main__":
    main()
