# structure_tensor.py
"""
Shared utilities for computing structure tensors using Pavel Holoborodko
derivative kernels. This provides the canonical implementation used by
StructureTensorInferer and other subsystems.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_LAYOUT_3D: Sequence[str] = ("Jzz", "Jzy", "Jzx", "Jyy", "Jyx", "Jxx")
DEFAULT_LAYOUT_2D: Sequence[str] = ("Jyy", "Jyx", "Jxx")


def _cache_key(device: torch.device, dtype: torch.dtype, extra: Iterable[int] | None = None) -> tuple:
    """Build a hashable cache key for kernel builders."""
    base = (device.type, device.index if device.index is not None else -1, str(dtype))
    if not extra:
        return base
    return base + tuple(extra)


@lru_cache(maxsize=None)
def _get_gaussian_kernel_3d(device: torch.device, dtype: torch.dtype, sigma: float) -> tuple[torch.Tensor, int]:
    radius = int(3 * sigma)
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    g1 = torch.exp(-coords**2 / (2 * sigma * sigma))
    g1 = g1 / g1.sum()
    g3 = g1[:, None, None] * g1[None, :, None] * g1[None, None, :]
    kernel = g3.unsqueeze(0).unsqueeze(0).contiguous()
    return kernel, radius


@lru_cache(maxsize=None)
def _get_gaussian_kernel_2d(device: torch.device, dtype: torch.dtype, sigma: float) -> tuple[torch.Tensor, int]:
    radius = int(3 * sigma)
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    g1 = torch.exp(-coords**2 / (2 * sigma * sigma))
    g1 = g1 / g1.sum()
    g2 = g1[:, None] * g1[None, :]
    kernel = g2.unsqueeze(0).unsqueeze(0).contiguous()
    return kernel, radius


@lru_cache(maxsize=None)
def _get_pavel_kernels_3d(device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    d = torch.tensor([2.0, 1.0, -16.0, -27.0, 0.0, 27.0, 16.0, -1.0, -2.0], device=device, dtype=dtype)
    s = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], device=device, dtype=dtype)

    kz = (d.view(9, 1, 1) * s.view(1, 5, 1) * s.view(1, 1, 5)) / (96 * 16 * 16)
    ky = (s.view(5, 1, 1) * d.view(1, 9, 1) * s.view(1, 1, 5)) / (96 * 16 * 16)
    kx = (s.view(5, 1, 1) * s.view(1, 5, 1) * d.view(1, 1, 9)) / (96 * 16 * 16)

    return (
        kz.unsqueeze(0).unsqueeze(0).contiguous(),
        ky.unsqueeze(0).unsqueeze(0).contiguous(),
        kx.unsqueeze(0).unsqueeze(0).contiguous(),
    )


@lru_cache(maxsize=None)
def _get_pavel_kernels_2d(device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    kz3d, ky3d, kx3d = _get_pavel_kernels_3d(device, dtype)
    # slice the central plane along the orthogonal dimension to obtain 2D kernels
    ky2d = ky3d[0, 0, 2, :, :].unsqueeze(0).unsqueeze(0).contiguous()
    kx2d = kx3d[0, 0, 2, :, :].unsqueeze(0).unsqueeze(0).contiguous()
    return ky2d, kx2d


def _ensure_tensor(data: torch.Tensor | np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        tensor = data.to(device=device, dtype=dtype)
    else:
        tensor = torch.as_tensor(data, device=device, dtype=dtype)
    return tensor


def _components_to_matrix_structure(components: torch.Tensor, layout: Sequence[str]) -> torch.Tensor:
    """Convert flattened components into symmetric matrices."""
    if components.dim() < 2:
        raise ValueError("components tensor must have channel dimension")
    ch = components.shape[1]
    if ch == 6:
        Jzz, Jzy, Jzx, Jyy, Jyx, Jxx = components.unbind(dim=1)
        mats = torch.stack(
            [
                torch.stack([Jxx, Jyx, Jzx], dim=-1),
                torch.stack([Jyx, Jyy, Jzy], dim=-1),
                torch.stack([Jzx, Jzy, Jzz], dim=-1),
            ],
            dim=-2,
        )
        return mats
    if ch == 3:
        Jyy, Jyx, Jxx = components.unbind(dim=1)
        mats = torch.stack(
            [
                torch.stack([Jxx, Jyx], dim=-1),
                torch.stack([Jyx, Jyy], dim=-1),
            ],
            dim=-2,
        )
        return mats
    raise ValueError(f"Unsupported number of channels for structure tensor: {ch}")


def components_to_matrix(components: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Convert flattened components to symmetric matrices."""
    is_numpy = isinstance(components, np.ndarray)
    tensor = torch.as_tensor(components)
    layout = DEFAULT_LAYOUT_3D if tensor.shape[1] == 6 else DEFAULT_LAYOUT_2D
    mats = _components_to_matrix_structure(tensor, layout)
    if is_numpy:
        return mats.detach().cpu().numpy()
    return mats


def eigendecompose(
    components: torch.Tensor | np.ndarray,
    smallest_first: bool = True,
) -> tuple[torch.Tensor | np.ndarray, torch.Tensor | np.ndarray]:
    """
    Compute eigenvalues/vectors for structure tensor components.
    Returns eigenvalues and eigenvectors, leveraging torch/numpy as needed.
    """
    if isinstance(components, np.ndarray):
        tensor = torch.as_tensor(components)
        layout = DEFAULT_LAYOUT_3D if tensor.shape[1] == 6 else DEFAULT_LAYOUT_2D
        if tensor.shape[1] == 3:
            return _eigendecompose_2d_numpy(tensor, smallest_first=smallest_first)
        mats = _components_to_matrix_structure(tensor, layout).cpu().numpy()
        w, v = np.linalg.eigh(mats)
        if smallest_first:
            order = np.argsort(w, axis=-1)
            w = np.take_along_axis(w, order, axis=-1)
            v = np.take_along_axis(v, order[..., None], axis=-1)
        return w, v
    if components.shape[1] == 3:
        return _eigendecompose_2d_torch(components, smallest_first=smallest_first)
    layout = DEFAULT_LAYOUT_3D if components.shape[1] == 6 else DEFAULT_LAYOUT_2D
    mats = _components_to_matrix_structure(components, layout)
    w, v = torch.linalg.eigh(mats)
    if smallest_first:
        w, idx = torch.sort(w, dim=-1)
        v = torch.gather(v, dim=-1, index=idx.unsqueeze(-1).expand_as(v))
    return w, v


def _eigendecompose_2d_torch(
    components: torch.Tensor,
    *,
    smallest_first: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    Jyy, Jyx, Jxx = components.unbind(dim=1)
    trace = Jxx + Jyy
    diff = Jxx - Jyy
    alpha = torch.sqrt(diff * diff + 4.0 * (Jyx * Jyx))
    lam1 = 0.5 * (trace - alpha)
    lam2 = 0.5 * (trace + alpha)
    if not smallest_first:
        lam1, lam2 = lam2, lam1
    w = torch.stack([lam1, lam2], dim=-1)

    # Eigenvectors
    v1x = torch.where(
        torch.abs(Jyx) > torch.abs(lam1 - Jxx),
        -(lam1 - Jxx),
        -Jyx,
    )
    v1y = torch.where(
        torch.abs(Jyx) > torch.abs(lam1 - Jxx),
        Jyx,
        Jxx - lam1,
    )
    norm = torch.sqrt(v1x * v1x + v1y * v1y).clamp_min(1e-12)
    v1x = v1x / norm
    v1y = v1y / norm
    v2x = -v1y
    v2y = v1x
    if not smallest_first:
        v1x, v2x = v2x, v1x
        v1y, v2y = v2y, v1y
    v = torch.stack(
        [
            torch.stack([v1x, v2x], dim=-1),
            torch.stack([v1y, v2y], dim=-1),
        ],
        dim=-2,
    )
    return w, v


def _eigendecompose_2d_numpy(
    components: torch.Tensor,
    *,
    smallest_first: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    tensor = components.cpu().numpy()
    Jyy, Jyx, Jxx = tensor
    trace = Jxx + Jyy
    diff = Jxx - Jyy
    alpha = np.sqrt(diff * diff + 4.0 * (Jyx * Jyx))
    lam1 = 0.5 * (trace - alpha)
    lam2 = 0.5 * (trace + alpha)
    if not smallest_first:
        lam1, lam2 = lam2, lam1
    w = np.stack([lam1, lam2], axis=-1)

    v1x = np.where(
        np.abs(Jyx) > np.abs(lam1 - Jxx),
        -(lam1 - Jxx),
        -Jyx,
    )
    v1y = np.where(
        np.abs(Jyx) > np.abs(lam1 - Jxx),
        Jyx,
        Jxx - lam1,
    )
    norm = np.sqrt(v1x * v1x + v1y * v1y) + 1e-12
    v1x = v1x / norm
    v1y = v1y / norm
    v2x = -v1y
    v2y = v1x
    if not smallest_first:
        v1x, v2x = v2x, v1x
        v1y, v2y = v2y, v1y
    v = np.stack(
        [
            np.stack([v1x, v2x], axis=-1),
            np.stack([v1y, v2y], axis=-1),
        ],
        axis=-2,
    )
    return w, v


@dataclass
class StructureTensorComputer:
    sigma: float = 1.0
    component_sigma: float | None = None
    smooth_components: bool = False
    device: torch.device | str | None = None
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        resolved_device = torch.device(self.device) if self.device is not None else torch.device("cpu")
        if resolved_device.type == "cuda" and not torch.cuda.is_available():
            resolved_device = torch.device("cpu")
        self.device = resolved_device

    def compute(
        self,
        volume: torch.Tensor | np.ndarray,
        *,
        sigma: float | None = None,
        component_sigma: float | None = None,
        smooth_components: bool | None = None,
        device: torch.device | str | None = None,
        as_numpy: bool = False,
        spatial_dims: int | None = None,
    ) -> torch.Tensor | np.ndarray:
        """Compute the structure tensor components for a 2D or 3D scalar field."""
        sigma_val = float(self.sigma if sigma is None else sigma)
        smooth_components_val = self.smooth_components if smooth_components is None else bool(smooth_components)

        component_sigma_val: float | None = (
            float(component_sigma) if component_sigma is not None else self.component_sigma
        )
        if smooth_components_val:
            if component_sigma_val is None:
                component_sigma_val = sigma_val
        else:
            component_sigma_val = 0.0

        target_device = torch.device(device) if device is not None else self.device
        if target_device.type == "cuda" and not torch.cuda.is_available():
            target_device = torch.device("cpu")

        x = _ensure_tensor(volume, target_device, self.dtype)

        if spatial_dims is not None and spatial_dims not in (2, 3):
            raise ValueError("spatial_dims must be 2 or 3 when specified.")

        if spatial_dims is None:
            if x.dim() >= 5:
                inferred_dims = 3
            elif x.dim() == 4:
                inferred_dims = 2 if x.shape[-3] == 1 else 3
            elif x.dim() == 3:
                inferred_dims = 2 if x.shape[-3] == 1 else 3
            elif x.dim() == 2:
                inferred_dims = 2
            else:
                raise ValueError("Unsupported input dimensionality for structure tensor computation.")
        else:
            inferred_dims = spatial_dims

        remove_batch_dim = False

        if inferred_dims == 3:
            if x.dim() == 5:
                if x.shape[1] != 1:
                    raise ValueError("Expected single-channel input for 3D structure tensor.")
            elif x.dim() == 4:
                if x.shape[1] != 1:
                    x = x.unsqueeze(1)
            elif x.dim() == 3:
                x = x.unsqueeze(0).unsqueeze(0)
                remove_batch_dim = True
            elif x.dim() == 2:
                x = x.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                remove_batch_dim = True
            else:
                raise ValueError("Unable to broadcast input to `[batch,1,D,H,W]`.")
            components = self._compute_3d(x, sigma_val, component_sigma_val or 0.0)
        else:
            # 2D case
            if x.dim() == 4:
                if x.shape[1] != 1:
                    x = x.unsqueeze(1)
            elif x.dim() == 3:
                if x.shape[0] == 1:
                    x = x.unsqueeze(0)
                    remove_batch_dim = True
                else:
                    x = x.unsqueeze(1)
            elif x.dim() == 2:
                x = x.unsqueeze(0).unsqueeze(0)
                remove_batch_dim = True
            else:
                raise ValueError("Unable to broadcast input to `[batch,1,H,W]`.")
            components = self._compute_2d(x, sigma_val, component_sigma_val or 0.0)

        if remove_batch_dim and components.shape[0] == 1:
            components = components.squeeze(0)

        if as_numpy:
            return components.detach().cpu().numpy()
        return components

    def _compute_3d(self, x: torch.Tensor, sigma: float, component_sigma: float) -> torch.Tensor:
        device = x.device
        dtype = x.dtype

        if sigma > 0:
            kernel, pad = _get_gaussian_kernel_3d(device, dtype, sigma)
            x = F.conv3d(x, kernel, padding=(pad, pad, pad))

        kz, ky, kx = _get_pavel_kernels_3d(device, dtype)
        gz = F.conv3d(x, kz, padding=(4, 2, 2))
        gy = F.conv3d(x, ky, padding=(2, 4, 2))
        gx = F.conv3d(x, kx, padding=(2, 2, 4))

        Jxx = gx * gx
        Jyx = gx * gy
        Jzx = gx * gz
        Jyy = gy * gy
        Jzy = gy * gz
        Jzz = gz * gz

        J = torch.cat([Jzz, Jzy, Jzx, Jyy, Jyx, Jxx], dim=1)

        if component_sigma > 0:
            kernel, pad = _get_gaussian_kernel_3d(device, dtype, component_sigma)
            kernel = kernel.expand(J.shape[1], 1, -1, -1, -1).contiguous()
            J = F.conv3d(J, kernel, padding=(pad, pad, pad), groups=J.shape[1])
        return J

    def _compute_2d(self, x: torch.Tensor, sigma: float, component_sigma: float) -> torch.Tensor:
        device = x.device
        dtype = x.dtype

        if sigma > 0:
            kernel, pad = _get_gaussian_kernel_2d(device, dtype, sigma)
            x = F.conv2d(x, kernel, padding=pad)

        ky, kx = _get_pavel_kernels_2d(device, dtype)
        gy = F.conv2d(x, ky, padding=(4, 2))
        gx = F.conv2d(x, kx, padding=(2, 4))

        Jxx = gx * gx
        Jyx = gx * gy
        Jyy = gy * gy
        J = torch.cat([Jyy, Jyx, Jxx], dim=1)

        if component_sigma > 0:
            kernel, pad = _get_gaussian_kernel_2d(device, dtype, component_sigma)
            kernel = kernel.expand(J.shape[1], 1, -1, -1).contiguous()
            J = F.conv2d(J, kernel, padding=pad, groups=J.shape[1])
        return J
