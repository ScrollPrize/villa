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

# ─── Pavel Holoborodko kernels ────────────────────────────────────────
# 1D derivative and smoothing kernels (float32, CPU)
_d = torch.tensor([2.,  1., -16., -27., 0., 27., 16., -1., -2.], dtype=torch.float32)
_s = torch.tensor([1., 4., 6., 4., 1.], dtype=torch.float32)

# build 3D “derivative-of-Gaussian” kernels
_kz = (_d.view(9,1,1) * _s.view(1,5,1) * _s.view(1,1,5)) / (96*16*16)  # (9,5,5)
_ky = (_s.view(5,1,1) * _d.view(1,9,1) * _s.view(1,1,5)) / (96*16*16)  # (5,9,5)
_kx = (_s.view(5,1,1) * _s.view(1,5,1) * _d.view(1,1,9)) / (96*16*16)  # (5,5,9)

# compute per-kernel padding (half‐sizes)
_pad_kz = ( _kz.shape[0]//2, _kz.shape[1]//2, _kz.shape[2]//2 )  # (4,2,2)
_pad_ky = ( _ky.shape[0]//2, _ky.shape[1]//2, _ky.shape[2]//2 )  # (2,4,2)
_pad_kx = ( _kx.shape[0]//2, _kx.shape[1]//2, _kx.shape[2]//2 )  # (2,2,4)

def gradient(field: torch.Tensor) -> torch.Tensor:
    """
    Pavel Holoborodko 3D gradient via 9×5×5 derivative-of-Gaussian filters.
    Args:
      field: (3, D, H, W)
    Returns:
      grad:  (3, 3, D, H, W), axes = (z, y, x)
    """
    # batchify
    f = field.unsqueeze(0)                # → (1,3,D,H,W)
    device = f.device

    # move the base kernels onto the same device as `f`
    kz5 = _kz.view(1,1,9,5,5).repeat(3,1,1,1,1).to(device)
    ky5 = _ky.view(1,1,5,9,5).repeat(3,1,1,1,1).to(device)
    kx5 = _kx.view(1,1,5,5,9).repeat(3,1,1,1,1).to(device)

    # three grouped conv3d calls, each padded so output = (D,H,W)
    dz = F.conv3d(f, kz5, padding=_pad_kz, groups=3).squeeze(0)
    dy = F.conv3d(f, ky5, padding=_pad_ky, groups=3).squeeze(0)
    dx = F.conv3d(f, kx5, padding=_pad_kx, groups=3).squeeze(0)

    # stack → (3 channels, 3 derivative-axes, D, H, W)
    return torch.stack((dz, dy, dx), dim=1)

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
