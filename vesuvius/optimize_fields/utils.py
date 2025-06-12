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
    return (v + 2*(w*uv + uuv)).clone()

#@torch.compile(fullgraph=True, mode="max-autotune", dynamic=True)
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

#@torch.compile(fullgraph=True, mode="max-autotune", dynamic=True)
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

def matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of rotation matrices R[..., 3, 3] into quaternions q[..., 4],
    using the “scalar‐first” convention (w, x, y, z) and ensuring w >= 0.

    Algorithm (stable version):
      Let
        t0 = 1 + R[0,0] + R[1,1] + R[2,2]
        t1 = 1 + R[0,0] - R[1,1] - R[2,2]
        t2 = 1 - R[0,0] + R[1,1] - R[2,2]
        t3 = 1 - R[0,0] - R[1,1] + R[2,2]

      Find the index k = argmax([t0, t1, t2, t3]) along last‐but‐one axis.
      Then compute the quaternion elements appropriately:

        if k == 0:
          w = 0.5 * sqrt(t0)
          x = (R[2,1] - R[1,2]) / (4*w)
          y = (R[0,2] - R[2,0]) / (4*w)
          z = (R[1,0] - R[0,1]) / (4*w)
        elif k == 1:
          x = 0.5 * sqrt(t1)
          w = (R[2,1] - R[1,2]) / (4*x)
          y = (R[0,1] + R[1,0]) / (4*x)
          z = (R[2,0] + R[0,2]) / (4*x)
        elif k == 2:
          y = 0.5 * sqrt(t2)
          w = (R[0,2] - R[2,0]) / (4*y)
          x = (R[0,1] + R[1,0]) / (4*y)
          z = (R[1,2] + R[2,1]) / (4*y)
        else:  # k == 3
          z = 0.5 * sqrt(t3)
          w = (R[1,0] - R[0,1]) / (4*z)
          x = (R[2,0] + R[0,2]) / (4*z)
          y = (R[1,2] + R[2,1]) / (4*z)

    Finally force w ≥ 0 by possibly flipping the sign of (w, x, y, z).

    This implementation is fully batched over the leading dims of R.
    """
    # R: shape (..., 3, 3).  We’ll output q: shape (..., 4).
    *batch_dims, _, _ = R.shape
    # Compute the four “traces” t0,...,t3:
    diag = torch.stack([R[..., 0, 0], R[..., 1, 1], R[..., 2, 2]], dim=-1)  # (..., 3)
    t0 = 1.0 + diag.sum(dim=-1)                                                     # (...,)
    t1 = 1.0 + R[..., 0, 0] - R[..., 1, 1] - R[..., 2, 2]                           # (...,)
    t2 = 1.0 - R[..., 0, 0] + R[..., 1, 1] - R[..., 2, 2]                           # (...,)
    t3 = 1.0 - R[..., 0, 0] - R[..., 1, 1] + R[..., 2, 2]                           # (...,)

    # Stack them and pick the index of the maximum:
    t_stack = torch.stack([t0, t1, t2, t3], dim=-1)  # (..., 4)
    k = torch.argmax(t_stack, dim=-1)                # (...,) in {0,1,2,3}

    # Prepare output tensor:
    q = torch.empty((*batch_dims, 4), device=R.device, dtype=R.dtype)

    # Helper to gather a boolean mask of where k == some value:
    def mask_idx(val):
        return (k == val)

    # Compute each case separately:
    # CASE k == 0  (largest trace = R[0,0]+R[1,1]+R[2,2] )
    m0 = mask_idx(0)
    if m0.any():
        # w = 0.5 * sqrt(t0)
        w0 = 0.5 * torch.sqrt(torch.clamp(t0[m0], min=1e-16))
        inv_4w = 1.0 / (4.0 * w0)
        x0 = (R[..., 2, 1][m0] - R[..., 1, 2][m0]) * inv_4w
        y0 = (R[..., 0, 2][m0] - R[..., 2, 0][m0]) * inv_4w
        z0 = (R[..., 1, 0][m0] - R[..., 0, 1][m0]) * inv_4w
        q[m0, 0] = w0
        q[m0, 1] = x0
        q[m0, 2] = y0
        q[m0, 3] = z0

    # CASE k == 1
    m1 = mask_idx(1)
    if m1.any():
        x1 = 0.5 * torch.sqrt(torch.clamp(t1[m1], min=1e-16))
        inv_4x = 1.0 / (4.0 * x1)
        w1 = (R[..., 2, 1][m1] - R[..., 1, 2][m1]) * inv_4x
        y1 = (R[..., 0, 1][m1] + R[..., 1, 0][m1]) * inv_4x
        z1 = (R[..., 2, 0][m1] + R[..., 0, 2][m1]) * inv_4x
        q[m1, 0] = w1
        q[m1, 1] = x1
        q[m1, 2] = y1
        q[m1, 3] = z1

    # CASE k == 2
    m2 = mask_idx(2)
    if m2.any():
        y2 = 0.5 * torch.sqrt(torch.clamp(t2[m2], min=1e-16))
        inv_4y = 1.0 / (4.0 * y2)
        w2 = (R[..., 0, 2][m2] - R[..., 2, 0][m2]) * inv_4y
        x2 = (R[..., 0, 1][m2] + R[..., 1, 0][m2]) * inv_4y
        z2 = (R[..., 1, 2][m2] + R[..., 2, 1][m2]) * inv_4y
        q[m2, 0] = w2
        q[m2, 1] = x2
        q[m2, 2] = y2
        q[m2, 3] = z2

    # CASE k == 3
    m3 = mask_idx(3)
    if m3.any():
        z3 = 0.5 * torch.sqrt(torch.clamp(t3[m3], min=1e-16))
        inv_4z = 1.0 / (4.0 * z3)
        w3 = (R[..., 1, 0][m3] - R[..., 0, 1][m3]) * inv_4z
        x3 = (R[..., 2, 0][m3] + R[..., 0, 2][m3]) * inv_4z
        y3 = (R[..., 1, 2][m3] + R[..., 2, 1][m3]) * inv_4z
        q[m3, 0] = w3
        q[m3, 1] = x3
        q[m3, 2] = y3
        q[m3, 3] = z3

    # Finally, ensure the scalar part w >= 0.  If w < 0, flip the entire quaternion
    w_sign = (q[..., 0] < 0).to(R.dtype).unsqueeze(-1)  # shape (..., 1), 1 where w<0
    q = q * (1.0 - 2.0 * w_sign)  # multiply entire (w,x,y,z) by -1 where w<0

    return q  # shape (..., 4)


if hasattr(torch, "compile"):
    gradient = torch.compile(
        gradient,
        fullgraph=True,
        mode="max-autotune",
        dynamic=True,
    )
    divergence = torch.compile(
        divergence,
        fullgraph=True,
        mode="max-autotune",
        dynamic=True,
    )