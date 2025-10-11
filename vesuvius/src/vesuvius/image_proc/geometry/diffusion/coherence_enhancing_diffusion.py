from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm is optional
    tqdm = None

from vesuvius.image_proc.geometry.structure_tensor import StructureTensorComputer

EPS = 2**-52
GAMMA = 0.01
CM = 7.2848


def _make_progress_iterator(steps: int, show_progress: bool) -> Iterable[int]:
    if show_progress and tqdm is not None:
        return tqdm(range(steps), desc="Diffusion steps", leave=False)
    return range(steps)


def _structure_tensor_components_2d(
    img: torch.Tensor, st_computer: StructureTensorComputer
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute 2D structure tensor components using the shared StructureTensorComputer."""
    if st_computer.device != img.device:
        st_computer.device = img.device
    if st_computer.dtype != img.dtype:
        st_computer.dtype = img.dtype
    comps = st_computer.compute(img, device=img.device, spatial_dims=2)
    s22 = comps[:, 0:1]  # Jyy
    s12 = comps[:, 1:2]  # Jyx
    s11 = comps[:, 2:3]  # Jxx
    return s11, s12, s22


def _prepare_tensor(
    data: torch.Tensor | np.ndarray,
    *,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor], bool, np.dtype | torch.dtype]:
    """Convert input data to a 4D float32 tensor and provide a restorer."""
    from_numpy = isinstance(data, np.ndarray)

    if from_numpy:
        np_array = np.asarray(data)
        original_dtype = np_array.dtype
        tensor = torch.from_numpy(np_array.astype(np.float32, copy=False))
    elif isinstance(data, torch.Tensor):
        tensor = data
        original_dtype = data.dtype
    else:
        raise TypeError("coherence_enhancing_diffusion expects torch.Tensor or np.ndarray input.")

    original_shape = tensor.shape

    if tensor.ndim == 4:
        pass
    elif tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    else:
        raise ValueError("Input must have 2, 3, or 4 dimensions representing [B,C,H,W] data.")

    if device is not None:
        tensor = tensor.to(device=device)

    tensor = tensor.to(dtype=torch.float32)

    def restore_fn(result: torch.Tensor) -> torch.Tensor:
        restored = result
        if len(original_shape) == 2:
            restored = restored.squeeze(0).squeeze(0)
        elif len(original_shape) == 3:
            restored = restored.squeeze(0)
        return restored

    return tensor, restore_fn, from_numpy, original_dtype


def compute_alpha(s11: torch.Tensor, s12: torch.Tensor, s22: torch.Tensor) -> torch.Tensor:
    """Compute eigenvalue measure alpha."""
    a = s11 - s22
    b = s12
    return torch.sqrt(a * a + 4.0 * b * b)


def compute_c2(alpha: torch.Tensor, lambda_param: float, m: float) -> torch.Tensor:
    """Compute diffusivity function c2."""
    h1 = (alpha + EPS) / lambda_param
    h2 = h1 if abs(m - 1.0) < 1e-10 else torch.pow(h1, m)
    h3 = torch.exp(-CM / h2)
    return GAMMA + (1.0 - GAMMA) * h3


def compute_diffusion_tensor(
    s11: torch.Tensor,
    s12: torch.Tensor,
    s22: torch.Tensor,
    alpha: torch.Tensor,
    c2: torch.Tensor,
    *,
    c1: float = GAMMA,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute diffusion tensor components D11, D12, D22."""
    dd = (c2 - c1) * (s11 - s22) / (alpha + EPS)
    d11 = 0.5 * (c1 + c2 + dd)
    d12 = (c1 - c2) * s12 / (alpha + EPS)
    d22 = 0.5 * (c1 + c2 - dd)
    return d11, d12, d22


def _diffusion_step(
    img: torch.Tensor, d11: torch.Tensor, d12: torch.Tensor, d22: torch.Tensor, step_size: float
) -> torch.Tensor:
    """Single explicit diffusion step matching the Java implementation."""
    img_pad = F.pad(img, (1, 1, 1, 1), mode="replicate")
    d11_pad = F.pad(d11, (1, 1, 1, 1), mode="replicate")
    d12_pad = F.pad(d12, (1, 1, 1, 1), mode="replicate")
    d22_pad = F.pad(d22, (1, 1, 1, 1), mode="replicate")

    img_c = img_pad[:, :, 1:-1, 1:-1]
    img_n = img_pad[:, :, 0:-2, 1:-1]
    img_s = img_pad[:, :, 2:, 1:-1]
    img_w = img_pad[:, :, 1:-1, 0:-2]
    img_e = img_pad[:, :, 1:-1, 2:]
    img_nw = img_pad[:, :, 0:-2, 0:-2]
    img_ne = img_pad[:, :, 0:-2, 2:]
    img_sw = img_pad[:, :, 2:, 0:-2]
    img_se = img_pad[:, :, 2:, 2:]

    d11_c = d11_pad[:, :, 1:-1, 1:-1]
    d11_n = d11_pad[:, :, 0:-2, 1:-1]
    d11_s = d11_pad[:, :, 2:, 1:-1]

    d22_c = d22_pad[:, :, 1:-1, 1:-1]
    d22_w = d22_pad[:, :, 1:-1, 0:-2]
    d22_e = d22_pad[:, :, 1:-1, 2:]

    d12_c = d12_pad[:, :, 1:-1, 1:-1]
    d12_n = d12_pad[:, :, 0:-2, 1:-1]
    d12_s = d12_pad[:, :, 2:, 1:-1]
    d12_w = d12_pad[:, :, 1:-1, 0:-2]
    d12_e = d12_pad[:, :, 1:-1, 2:]

    c_cop = d22_c + d22_w
    a_amo = d11_s + d11_c
    a_apo = d11_n + d11_c
    c_com = d22_c + d22_e

    first_deriv = (
        c_cop * img_w
        + a_amo * img_s
        - (a_amo + a_apo + c_com + c_cop) * img_c
        + a_apo * img_n
        + c_com * img_e
    )

    bmo = d12_s
    bop = d12_w
    bpo = d12_n
    bom = d12_e

    second_deriv = (
        -((bmo + bop) * img_sw + (bpo + bom) * img_ne)
        + (bpo + bop) * img_nw
        + (bmo + bom) * img_se
    )

    return img + step_size * (0.5 * first_deriv + 0.25 * second_deriv)


def coherence_enhancing_diffusion(
    data: torch.Tensor | np.ndarray,
    config: Mapping[str, float],
    *,
    show_progress: bool = False,
    device: torch.device | None = None,
    return_numpy: bool | None = None,
) -> torch.Tensor | np.ndarray:
    """
    Run coherence-enhancing diffusion on a tensor or numpy array.

    Args:
        data: Input in shape (B,C,H,W), (C,H,W), or (H,W).
        config: Mapping with 'lambda', 'sigma', 'rho', 'step_size', 'm', and 'num_steps'.
        show_progress: Display tqdm progress if available.
        device: Optional device override for tensor inputs or numpy conversion.
        return_numpy: Force numpy output (True/False). Defaults to matching input type.
    """
    tensor, restore_fn, from_numpy, original_dtype = _prepare_tensor(data, device=device)

    rho = float(config["rho"])
    sigma = float(config["sigma"])
    lambda_param = float(config["lambda"])
    step_size = float(config["step_size"])
    m = float(config["m"])
    num_steps = int(config["num_steps"])

    component_sigma = rho if rho > 0 else None
    st_computer = StructureTensorComputer(
        sigma=sigma,
        component_sigma=component_sigma,
        smooth_components=component_sigma is not None,
        device=tensor.device,
        dtype=tensor.dtype,
    )

    iterator = _make_progress_iterator(num_steps, show_progress)

    for _ in iterator:
        s11, s12, s22 = _structure_tensor_components_2d(tensor, st_computer)
        alpha = compute_alpha(s11, s12, s22)
        c2 = compute_c2(alpha, lambda_param, m)
        d11, d12, d22 = compute_diffusion_tensor(s11, s12, s22, alpha, c2)
        tensor = _diffusion_step(tensor, d11, d12, d22, step_size)

    result_tensor = restore_fn(tensor)

    if return_numpy is None:
        return_numpy = from_numpy

    if return_numpy:
        out = result_tensor.detach().cpu().numpy()
        if isinstance(original_dtype, np.dtype):
            out = out.astype(original_dtype, copy=False)
        return out

    return result_tensor.to(dtype=original_dtype) if isinstance(original_dtype, torch.dtype) else result_tensor
