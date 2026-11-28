"""Dense uint8 3D grid_sample in torch ops: MPS without the Metal kernels (LASAGNA_METAL=0, or Metal unavailable).

Follows grid_sample_3d_u8_kernel.cu and grid_sample_3d_u8_diff_kernel.cu term by term:
p = (grid - offset) * inv_scale per axis, corners floor(p) and floor(p) + 1, trilinear weights from the
fractions, corners outside the (Z, Y, X) volume contribute nothing, accumulated in the kernels' order.
The uint8 variant rounds half away from zero and clamps to [0, 255]; the backward differentiates the weights
and applies inv_scale. A sample with a non-finite coordinate reads 0 and has a zero gradient (the kernels'
integer conversion of NaN or inf is undefined).

MPS has no grid_sampler_3d_backward, so F.grid_sample can't serve the differentiable path there.
"""
from __future__ import annotations

import torch

_BATCH = 1 << 20


def _check(volume, grid, offset, inv_scale):
    if volume.dtype != torch.uint8 or volume.dim() != 4:
        raise ValueError(f"volume must be uint8 (C, Z, Y, X), got {volume.dtype} {tuple(volume.shape)}")
    if grid.dtype != torch.float32 or grid.dim() != 4 or grid.size(3) != 3:
        raise ValueError(f"grid must be float32 (D, H, W, 3), got {grid.dtype} {tuple(grid.shape)}")
    for name, t in (("offset", offset), ("inv_scale", inv_scale)):
        if t.dtype != torch.float32 or t.numel() != 3:
            raise ValueError(f"{name} must be float32 (3,)")
    for name, t in (("grid", grid), ("offset", offset), ("inv_scale", inv_scale)):
        if t.device != volume.device:
            raise ValueError(f"{name} is on {t.device}, the volume on {volume.device}")


def _locate(points, shape_zyx, offset, inv_scale):
    """Corner offsets (n, 8) into one channel, their validity (n, 8) and the fractions, kernel order."""
    Z, Y, X = shape_zyx
    p = (points - offset.view(1, 3)) * inv_scale.view(1, 3)
    finite = torch.isfinite(p).all(dim=1)
    p = torch.where(finite.unsqueeze(1), p, torch.zeros_like(p))
    f0 = torch.floor(p)
    frac = p - f0
    i0 = f0.long()
    i1 = i0 + 1
    ok = []
    idx = []
    for k in range(8):  # bit 0: x, bit 1: y, bit 2: z -> order 000, 100, 010, 110, 001, 101, 011, 111
        ix = i1[:, 0] if k & 1 else i0[:, 0]
        iy = i1[:, 1] if k & 2 else i0[:, 1]
        iz = i1[:, 2] if k & 4 else i0[:, 2]
        valid = finite & (ix >= 0) & (ix < X) & (iy >= 0) & (iy < Y) & (iz >= 0) & (iz < Z)
        ok.append(valid)
        flat = (iz * Y + iy) * X + ix
        idx.append(torch.where(valid, flat, torch.zeros_like(flat)))
    return torch.stack(idx, 1), torch.stack(ok, 1), frac[:, 0], frac[:, 1], frac[:, 2]


def _corner_values(flat_c, idx, ok):
    v = flat_c[idx].to(torch.float32)
    return torch.where(ok, v, torch.zeros_like(v))


def _sample(volume, grid, offset, inv_scale, *, as_uint8):
    _check(volume, grid, offset, inv_scale)
    C, Z, Y, X = (int(v) for v in volume.shape)
    D, H, W = (int(v) for v in grid.shape[:3])
    N = D * H * W
    dtype = torch.uint8 if as_uint8 else torch.float32
    out = torch.zeros((C, N), dtype=dtype, device=volume.device)
    if N == 0 or volume.numel() == 0:
        return out.reshape(C, D, H, W)
    vol = volume.contiguous().reshape(C, -1)
    points = grid.reshape(N, 3)
    for start in range(0, N, _BATCH):
        stop = min(N, start + _BATCH)
        idx, ok, fx, fy, fz = _locate(points[start:stop], (Z, Y, X), offset, inv_scale)
        ofx, ofy, ofz = 1.0 - fx, 1.0 - fy, 1.0 - fz
        weights = (ofx * ofy * ofz, fx * ofy * ofz, ofx * fy * ofz, fx * fy * ofz,
                   ofx * ofy * fz, fx * ofy * fz, ofx * fy * fz, fx * fy * fz)
        for c in range(C):
            v = _corner_values(vol[c], idx, ok)
            val = torch.zeros_like(fx)
            for k in range(8):  # skipped corners add nothing, as `if (ok) val += w * v`
                val = torch.where(ok[:, k], val + weights[k] * v[:, k], val)
            if as_uint8:
                t = torch.floor(val)
                val = (t + (val - t >= 0.5).to(val.dtype)).clamp(0.0, 255.0)
            out[c, start:stop] = val.to(dtype)
    return out.reshape(C, D, H, W)


def sample_u8(volume, grid, offset, inv_scale):
    """grid_sample_3d_u8: (C, D, H, W) uint8."""
    return _sample(volume, grid, offset, inv_scale, as_uint8=True)


def sample_f32(volume, grid, offset, inv_scale):
    """grid_sample_3d_u8_diff's forward: (C, D, H, W) float32."""
    return _sample(volume, grid, offset, inv_scale, as_uint8=False)


def sample_grad(volume, grid, offset, inv_scale, grad_output):
    """grid_sample_3d_u8_diff's backward: (D, H, W, 3) float32 gradient w.r.t. the full-resolution grid."""
    _check(volume, grid, offset, inv_scale)
    C, Z, Y, X = (int(v) for v in volume.shape)
    D, H, W = (int(v) for v in grid.shape[:3])
    N = D * H * W
    grad = torch.zeros((N, 3), dtype=torch.float32, device=volume.device)
    if N == 0 or volume.numel() == 0:
        return grad.reshape(D, H, W, 3)
    vol = volume.contiguous().reshape(C, -1)
    points = grid.reshape(N, 3)
    go_all = grad_output.reshape(C, N)
    isx, isy, isz = inv_scale.reshape(3).unbind(0)
    for start in range(0, N, _BATCH):
        stop = min(N, start + _BATCH)
        idx, ok, fx, fy, fz = _locate(points[start:stop], (Z, Y, X), offset, inv_scale)
        ofx, ofy, ofz = 1.0 - fx, 1.0 - fy, 1.0 - fz
        dgx = torch.zeros_like(fx)
        dgy = torch.zeros_like(fx)
        dgz = torch.zeros_like(fx)
        for c in range(C):
            go = go_all[c, start:stop]
            v000, v100, v010, v110, v001, v101, v011, v111 = _corner_values(vol[c], idx, ok).unbind(1)
            dfx = ofy * ofz * (v100 - v000) + fy * ofz * (v110 - v010) + ofy * fz * (v101 - v001) + fy * fz * (v111 - v011)
            dfy = ofx * ofz * (v010 - v000) + fx * ofz * (v110 - v100) + ofx * fz * (v011 - v001) + fx * fz * (v111 - v101)
            dfz = ofx * ofy * (v001 - v000) + fx * ofy * (v101 - v100) + ofx * fy * (v011 - v010) + fx * fy * (v111 - v110)
            dgx = dgx + go * dfx * isx
            dgy = dgy + go * dfy * isy
            dgz = dgz + go * dfz * isz
        grad[start:stop] = torch.stack([dgx, dgy, dgz], dim=1)
    return grad.reshape(D, H, W, 3)


class GridSample3DU8Diff(torch.autograd.Function):
    @staticmethod
    def forward(ctx, volume, grid, offset, inv_scale):
        out = sample_f32(volume, grid, offset, inv_scale)
        ctx.save_for_backward(grid)
        ctx.volume = volume
        ctx.offset = offset
        ctx.inv_scale = inv_scale
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grid, = ctx.saved_tensors
        return None, sample_grad(ctx.volume, grid, ctx.offset, ctx.inv_scale, grad_output.contiguous()), None, None


class _TorchDense:
    """The dense samplers' MPS entry points, in torch ops (same names as grid_sample_3d_u8_metal)."""

    @staticmethod
    def grid_sample_3d_u8(volume, grid, offset, inv_scale):
        return sample_u8(volume, grid, offset, inv_scale)

    @staticmethod
    def grid_sample_3d_u8_diff(volume, grid, offset, inv_scale):
        return GridSample3DU8Diff.apply(volume, grid, offset, inv_scale)


_metal_dense = None
_metal_dense_error: str | None = None
_reported: set[str] = set()


def _load_metal_dense():
    global _metal_dense, _metal_dense_error
    if _metal_dense is None and _metal_dense_error is None:
        try:
            import importlib.util
            from pathlib import Path
            spec = importlib.util.spec_from_file_location(
                "grid_sample_3d_u8_metal", Path(__file__).with_name("grid_sample_3d_u8_metal.py"))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module._lib()  # compile now, so a failure falls back here rather than mid-fit
            _metal_dense = module
        except Exception as exc:
            _metal_dense_error = f"{type(exc).__name__}: {exc}"
    return _metal_dense


def mps_dense_backend():
    """(implementation, label) of the dense uint8 samplers on MPS, following LASAGNA_METAL as the sparse cache does:
    unset = the Metal kernels when they compile, else torch ops; "0" = torch ops; "1" = Metal or an error."""
    from sparse_pool_cache import metal_mode

    mode = metal_mode()
    if mode == "off":
        impl, label = _TorchDense, "torch (LASAGNA_METAL=0)"
    else:
        metal = _load_metal_dense()
        if metal is not None:
            impl, label = metal, "metal"
        elif mode == "required":
            raise RuntimeError(f"LASAGNA_METAL=1 but the dense Metal sampler is unavailable: {_metal_dense_error}")
        else:
            impl, label = _TorchDense, f"torch (metal unavailable: {_metal_dense_error})"
    if label not in _reported:
        _reported.add(label)
        print(f"[grid_sample_3d_u8] MPS dense sampler: {label}", flush=True)
    return impl, label
