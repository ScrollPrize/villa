# optimize_fields/utils.py

import torch
import torch.nn.functional as F


def rotate_by_quaternion(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    q: (...,4) unit quaternion with components [w,z,y,x]
    v: (...,3)      vector field to rotate
    returns v_rot(...,3)
    """
    # unpack
    w = q[..., 0].unsqueeze(-1)     # (...,1)
    zyx = q[..., 1:]                # (...,3)
    # cross products
    uv  = torch.cross(    zyx, v, dim=-1)        # (...,3)
    uuv = torch.cross(zyx, uv, dim=-1)           # (...,3)
    return v + 2*(w*uv + uuv)

def gradient(field: torch.Tensor) -> torch.Tensor:
    """
    Finite‐difference gradient of a 3‐component field.

    Args:
      field: Tensor of shape (3, D, H, W)
    Returns:
      grad: Tensor of shape (3, 3, D, H, W),
            where grad[c, i, ...] = ∂_{axis i} field[c, ...]
    """
    # batch‐ify channels: (1,3,D,H,W)
    f = field.unsqueeze(0)

    device = f.device
    dtype = f.dtype

    # central difference kernel [-0.5, 0, +0.5]
    k = torch.tensor([-0.5, 0.0, +0.5], device=device, dtype=dtype)

    # build 5D kernels for each axis, replicated per‐channel
    # shape should be (3,1,3,3,3) → then groups=3
    # ∂x (pad only width)
    kx = k.view(1,1,1,1,3).repeat(3, 1, 1, 1, 1)
    # ∂y (pad only height)
    ky = k.view(1,1,1,3,1).repeat(3, 1, 1, 1, 1)
    # ∂z (pad only depth)
    kz = k.view(1,1,3,1,1).repeat(3, 1, 1, 1, 1)

    # perform grouped conv3d: each of the 3 input channels uses its own 1×3 filter
    dx = F.conv3d(f, kx, padding=(0, 0, 1), groups=3)  # → (1,3,D,H,W)
    dy = F.conv3d(f, ky, padding=(0, 1, 0), groups=3)
    dz = F.conv3d(f, kz, padding=(1, 0, 0), groups=3)

    # remove batch dim: (3,D,H,W)
    dx = dx.squeeze(0)
    dy = dy.squeeze(0)
    dz = dz.squeeze(0)

    # stack into (3,3,D,H,W): first dim = channel, second = derivative axis
    grad = torch.stack((dz, dy, dx), dim=1)
    return grad


def divergence(field: torch.Tensor) -> torch.Tensor:
    """
    Divergence of a 3‐component field via its gradient.

    Args:
      field: Tensor of shape (3, D, H, W)
    Returns:
      div: Tensor of shape (D, H, W)
    """
    # grad has shape (3,3,D,H,W): grad[c, i, ...]
    g = gradient(field)
    # trace: ∂x u_x + ∂y u_y + ∂z u_z
    # which is g[0,0] + g[1,1] + g[2,2]
    return g[0,0] + g[1,1] + g[2,2]
