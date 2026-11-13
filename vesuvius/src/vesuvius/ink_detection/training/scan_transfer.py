"""Pooled-to-native scan-type augmentation for flat ink training.

Most labelled 9 um training data are renders pooled from 2.4 um scans (``public_2p4_level2_zmean4``), while the
unread scrolls are scanned natively at 8.6-9.4 um. A registered pooled/native pair of the same surface (PHerc 0814
segment 46527) shows that, to first order, the native render is the pooled one blurred over about three depth planes
and slightly in-plane, plus smooth low-frequency intensity differences. ``scan_transfer_fit.py`` fits that linear
kernel by least squares; this module applies it as a training augmentation.

For a random ``probability`` share of the training samples in each batch (never validation), the normalized image is
cross-correlated with the kernel (replicate padding), restored to its own mean and standard deviation, and a smooth
Gaussian random field (standard deviation ``noise_amplitude`` times the sample's) is added. Enable it with::

    "scan_transfer": {"enabled": true, "kernel": "builtin:pooled_to_native_pherc0814", "probability": 0.5}

Leave-one-scroll-out on PHerc 0139 (trained on 1667, Paris 4 and 0814, scored on 0139's native 9.362 um segments):
mean ROC-AUC 0.743 -> 0.785 for the same seed, with no loss on pooled renders (docs/ink_cross_scroll.md).
"""
from __future__ import annotations

import json
import math
from importlib import resources
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

BUILTIN_PREFIX = "builtin:"


def load_kernel(spec: str) -> np.ndarray:
    """A (kz, ky, kx) kernel from ``builtin:<name>``, a JSON file ({"kernel": nested list}) or an NPZ file."""

    if spec.startswith(BUILTIN_PREFIX):
        name = spec[len(BUILTIN_PREFIX):]
        text = (
            resources.files("vesuvius.ink_detection")
            .joinpath("configs", "scan_transfer", f"{name}.json")
            .read_text(encoding="utf-8")
        )
        kernel = np.asarray(json.loads(text)["kernel"], dtype=np.float32)
    else:
        path = Path(spec)
        if path.suffix == ".npz":
            kernel = np.asarray(np.load(path)["kernel"], dtype=np.float32)
        else:
            kernel = np.asarray(json.loads(path.read_text(encoding="utf-8"))["kernel"], dtype=np.float32)
    if kernel.ndim != 3 or any(size % 2 == 0 for size in kernel.shape):
        raise ValueError(f"scan_transfer kernel must be 3D with odd sizes, got {kernel.shape}")
    return kernel


def _gauss1d(sigma: float, device) -> torch.Tensor:
    radius = max(1, int(math.ceil(3 * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


class ScanTransfer:
    """Callable batch augmentation: ``image`` (B, 1, Z, Y, X) -> converted copy."""

    def __init__(
        self,
        kernel: np.ndarray,
        probability: float = 0.5,
        noise_amplitude: float = 0.5,
        noise_sigma_xy: float = 4.0,
        noise_sigma_z: float = 1.5,
        device=None,
        generator: torch.Generator | None = None,
    ):
        k = torch.as_tensor(np.asarray(kernel, dtype=np.float32), device=device)
        self.kernel = k[None, None]
        kz, ky, kx = k.shape
        self.pad = (kx // 2, kx // 2, ky // 2, ky // 2, kz // 2, kz // 2)
        self.p = float(probability)
        self.amp = float(noise_amplitude)
        self.gxy = _gauss1d(noise_sigma_xy, device)
        self.gz = _gauss1d(noise_sigma_z, device)
        self.generator = generator

    @classmethod
    def from_config(cls, config, device) -> "ScanTransfer":
        return cls(
            load_kernel(config.kernel),
            probability=config.probability,
            noise_amplitude=config.noise_amplitude,
            noise_sigma_xy=config.noise_sigma_xy,
            noise_sigma_z=config.noise_sigma_z,
            device=device,
        )

    def _smooth_noise(self, shape, device) -> torch.Tensor:
        noise = torch.randn(shape, device=device, generator=self.generator)
        for dim, g in ((2, self.gz), (3, self.gxy), (4, self.gxy)):
            k = g.view([1, 1] + [len(g) if d == dim else 1 for d in (2, 3, 4)])
            pad = [0, 0, 0, 0, 0, 0]
            pad[(4 - dim) * 2] = pad[(4 - dim) * 2 + 1] = len(g) // 2
            noise = F.conv3d(F.pad(noise, pad, mode="replicate"), k)
        return noise / (noise.flatten(1).std(1).view(-1, 1, 1, 1, 1) + 1e-6)

    @torch.no_grad()
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 5 or image.shape[1] != 1:
            raise ValueError(f"scan_transfer expects (B, 1, Z, Y, X), got {tuple(image.shape)}")
        selected = torch.rand(image.shape[0], device=image.device, generator=self.generator) < self.p
        if not bool(selected.any()):
            return image
        x = image[selected].float()
        mean = x.flatten(1).mean(1).view(-1, 1, 1, 1, 1)
        std = x.flatten(1).std(1).view(-1, 1, 1, 1, 1) + 1e-6
        y = F.conv3d(F.pad(x, self.pad, mode="replicate"), self.kernel)
        y = (y - y.flatten(1).mean(1).view(-1, 1, 1, 1, 1)) / (y.flatten(1).std(1).view(-1, 1, 1, 1, 1) + 1e-6)
        y = y * std + mean
        if self.amp > 0:
            y = y + self.amp * std * self._smooth_noise(y.shape, y.device)
        out = image.clone()
        out[selected] = y.to(image.dtype)
        return out
