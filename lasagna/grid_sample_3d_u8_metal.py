"""Metal uint8 3D grid_sample for Apple GPUs (MPS).

Ports of grid_sample_3d_u8_kernel.cu (uint8 output) and
grid_sample_3d_u8_diff_kernel.cu (float output, gradients w.r.t. the sample
positions), with the same signatures and the same arithmetic: trilinear
weights from floor/frac of ``(grid - offset) * inv_scale``, corners outside the
volume contribute zero, the uint8 variant rounds half away from zero and
clamps to [0, 255], and the backward differentiates the trilinear weights and
applies ``inv_scale`` (no atomics: every sample writes its own gradient). A
sample with a non-finite position reads 0 and has a zero gradient, as in
grid_sample_3d_u8_ops (the CUDA kernels' integer conversion of NaN or inf is
undefined).

Without them (LASAGNA_METAL=0, or when they do not compile), MPS uses the torch
ops in grid_sample_3d_u8_ops. ``F.grid_sample`` can't serve the differentiable
path on MPS: its 3D backward (``aten::grid_sampler_3d_backward``) is not
implemented there.

Compiled once per process with torch.mps.compile_shader (PyTorch's safe math
mode, so isfinite survives).
"""
import torch

_GROUP = 256
_LIB = None

_SOURCE = r'''
#include <metal_stdlib>
using namespace metal;

// ip: C, Z, Y, X, N, W_grid, H_grid, stride_d, stride_h, stride_w, stride_c
struct Sample {
    float fx, fy, fz;
    long ix0, iy0, iz0, ix1, iy1, iz1;
    bool x0_ok, x1_ok, y0_ok, y1_ok, z0_ok, z1_ok;
};

inline Sample locate(device const float* grid, constant long* ip, long n,
                     device const float* offset, device const float* inv_scale) {
    const long W = ip[5], H = ip[6];
    const long w = n % W;
    const long tmp = n / W;
    const long h = tmp % H;
    const long d = tmp / H;
    const long base = d * ip[7] + h * ip[8] + w * ip[9];
    const long sc = ip[10];
    float px = (grid[base + 0 * sc] - offset[0]) * inv_scale[0];
    float py = (grid[base + 1 * sc] - offset[1]) * inv_scale[1];
    float pz = (grid[base + 2 * sc] - offset[2]) * inv_scale[2];
    // A non-finite position reads 0 and has a zero gradient, as in grid_sample_3d_u8_ops: no corner
    // is valid, and p becomes 0 so that no NaN or infinity reaches the fractions or an integer
    // conversion (undefined for those values, in the CUDA kernels too).
    const bool finite = isfinite(px) && isfinite(py) && isfinite(pz);
    if (!finite) {
        px = 0.0f;
        py = 0.0f;
        pz = 0.0f;
    }
    const float fx0 = floor(px);
    const float fy0 = floor(py);
    const float fz0 = floor(pz);
    Sample s;
    s.fx = px - fx0;
    s.fy = py - fy0;
    s.fz = pz - fz0;
    s.ix0 = long(fx0); s.iy0 = long(fy0); s.iz0 = long(fz0);
    s.ix1 = s.ix0 + 1; s.iy1 = s.iy0 + 1; s.iz1 = s.iz0 + 1;
    const long X = ip[3], Y = ip[2], Z = ip[1];
    s.x0_ok = finite && s.ix0 >= 0 && s.ix0 < X; s.x1_ok = finite && s.ix1 >= 0 && s.ix1 < X;
    s.y0_ok = finite && s.iy0 >= 0 && s.iy0 < Y; s.y1_ok = finite && s.iy1 >= 0 && s.iy1 < Y;
    s.z0_ok = finite && s.iz0 >= 0 && s.iz0 < Z; s.z1_ok = finite && s.iz1 >= 0 && s.iz1 < Z;
    return s;
}

// The 8 corner values of one channel (0 outside the volume).
struct Corners8 { float v000, v100, v010, v110, v001, v101, v011, v111; };

inline Corners8 corners(device const uchar* vol_c, thread const Sample& s,
                        long stride_z, long stride_y) {
    Corners8 c;
    c.v000 = (s.z0_ok && s.y0_ok && s.x0_ok) ? float(vol_c[s.iz0 * stride_z + s.iy0 * stride_y + s.ix0]) : 0.0f;
    c.v100 = (s.z0_ok && s.y0_ok && s.x1_ok) ? float(vol_c[s.iz0 * stride_z + s.iy0 * stride_y + s.ix1]) : 0.0f;
    c.v010 = (s.z0_ok && s.y1_ok && s.x0_ok) ? float(vol_c[s.iz0 * stride_z + s.iy1 * stride_y + s.ix0]) : 0.0f;
    c.v110 = (s.z0_ok && s.y1_ok && s.x1_ok) ? float(vol_c[s.iz0 * stride_z + s.iy1 * stride_y + s.ix1]) : 0.0f;
    c.v001 = (s.z1_ok && s.y0_ok && s.x0_ok) ? float(vol_c[s.iz1 * stride_z + s.iy0 * stride_y + s.ix0]) : 0.0f;
    c.v101 = (s.z1_ok && s.y0_ok && s.x1_ok) ? float(vol_c[s.iz1 * stride_z + s.iy0 * stride_y + s.ix1]) : 0.0f;
    c.v011 = (s.z1_ok && s.y1_ok && s.x0_ok) ? float(vol_c[s.iz1 * stride_z + s.iy1 * stride_y + s.ix0]) : 0.0f;
    c.v111 = (s.z1_ok && s.y1_ok && s.x1_ok) ? float(vol_c[s.iz1 * stride_z + s.iy1 * stride_y + s.ix1]) : 0.0f;
    return c;
}

// Same accumulation order as the CUDA kernels: w000, w100, w010, w110,
// w001, w101, w011, w111 (skipped corners add nothing).
inline float interpolate(thread const Sample& s, thread const Corners8& c) {
    const float fx = s.fx, fy = s.fy, fz = s.fz;
    const float w000 = (1.0f - fx) * (1.0f - fy) * (1.0f - fz);
    const float w100 = fx          * (1.0f - fy) * (1.0f - fz);
    const float w010 = (1.0f - fx) * fy          * (1.0f - fz);
    const float w110 = fx          * fy          * (1.0f - fz);
    const float w001 = (1.0f - fx) * (1.0f - fy) * fz;
    const float w101 = fx          * (1.0f - fy) * fz;
    const float w011 = (1.0f - fx) * fy          * fz;
    const float w111 = fx          * fy          * fz;
    float val = 0.0f;
    if (s.z0_ok && s.y0_ok && s.x0_ok) val += w000 * c.v000;
    if (s.z0_ok && s.y0_ok && s.x1_ok) val += w100 * c.v100;
    if (s.z0_ok && s.y1_ok && s.x0_ok) val += w010 * c.v010;
    if (s.z0_ok && s.y1_ok && s.x1_ok) val += w110 * c.v110;
    if (s.z1_ok && s.y0_ok && s.x0_ok) val += w001 * c.v001;
    if (s.z1_ok && s.y0_ok && s.x1_ok) val += w101 * c.v101;
    if (s.z1_ok && s.y1_ok && s.x0_ok) val += w011 * c.v011;
    if (s.z1_ok && s.y1_ok && s.x1_ok) val += w111 * c.v111;
    return val;
}

kernel void gs3d_u8(device const uchar* volume [[buffer(0)]],
                    device const float* grid [[buffer(1)]],
                    device const float* offset [[buffer(2)]],
                    device const float* inv_scale [[buffer(3)]],
                    device uchar* out [[buffer(4)]],
                    constant long* ip [[buffer(5)]],
                    uint gid [[thread_position_in_grid]]) {
    const long n = long(gid);
    const long N = ip[4];
    if (n >= N) {
        return;
    }
    const Sample s = locate(grid, ip, n, offset, inv_scale);
    const long stride_y = ip[3];
    const long stride_z = ip[2] * ip[3];
    for (long c = 0; c < ip[0]; ++c) {
        const Corners8 v = corners(volume + c * ip[1] * stride_z, s, stride_z, stride_y);
        const float val = interpolate(s, v);
        const int ival = clamp(int(round(val)), 0, 255);
        out[c * N + n] = uchar(ival);
    }
}

kernel void gs3d_u8_diff_fwd(device const uchar* volume [[buffer(0)]],
                             device const float* grid [[buffer(1)]],
                             device const float* offset [[buffer(2)]],
                             device const float* inv_scale [[buffer(3)]],
                             device float* out [[buffer(4)]],
                             constant long* ip [[buffer(5)]],
                             uint gid [[thread_position_in_grid]]) {
    const long n = long(gid);
    const long N = ip[4];
    if (n >= N) {
        return;
    }
    const Sample s = locate(grid, ip, n, offset, inv_scale);
    const long stride_y = ip[3];
    const long stride_z = ip[2] * ip[3];
    for (long c = 0; c < ip[0]; ++c) {
        const Corners8 v = corners(volume + c * ip[1] * stride_z, s, stride_z, stride_y);
        out[c * N + n] = interpolate(s, v);
    }
}

kernel void gs3d_u8_diff_bwd(device const uchar* volume [[buffer(0)]],
                             device const float* grid [[buffer(1)]],
                             device const float* offset [[buffer(2)]],
                             device const float* inv_scale [[buffer(3)]],
                             device const float* grad_output [[buffer(4)]],
                             device float* grad_grid [[buffer(5)]],
                             constant long* ip [[buffer(6)]],
                             uint gid [[thread_position_in_grid]]) {
    const long n = long(gid);
    const long N = ip[4];
    if (n >= N) {
        return;
    }
    const Sample s = locate(grid, ip, n, offset, inv_scale);
    const float isx = inv_scale[0], isy = inv_scale[1], isz = inv_scale[2];
    const float fx = s.fx, fy = s.fy, fz = s.fz;
    const long stride_y = ip[3];
    const long stride_z = ip[2] * ip[3];
    float dgx = 0.0f, dgy = 0.0f, dgz = 0.0f;
    for (long c = 0; c < ip[0]; ++c) {
        const float go = grad_output[c * N + n];
        const Corners8 v = corners(volume + c * ip[1] * stride_z, s, stride_z, stride_y);
        const float dfx = (1.0f - fy) * (1.0f - fz) * (v.v100 - v.v000)
                        + fy          * (1.0f - fz) * (v.v110 - v.v010)
                        + (1.0f - fy) * fz          * (v.v101 - v.v001)
                        + fy          * fz          * (v.v111 - v.v011);
        const float dfy = (1.0f - fx) * (1.0f - fz) * (v.v010 - v.v000)
                        + fx          * (1.0f - fz) * (v.v110 - v.v100)
                        + (1.0f - fx) * fz          * (v.v011 - v.v001)
                        + fx          * fz          * (v.v111 - v.v101);
        const float dfz = (1.0f - fx) * (1.0f - fy) * (v.v001 - v.v000)
                        + fx          * (1.0f - fy) * (v.v101 - v.v100)
                        + (1.0f - fx) * fy          * (v.v011 - v.v010)
                        + fx          * fy          * (v.v111 - v.v110);
        dgx += go * dfx * isx;
        dgy += go * dfy * isy;
        dgz += go * dfz * isz;
    }
    grad_grid[n * 3 + 0] = dgx;
    grad_grid[n * 3 + 1] = dgy;
    grad_grid[n * 3 + 2] = dgz;
}
'''


def _lib():
    global _LIB
    if _LIB is None:
        _LIB = torch.mps.compile_shader(_SOURCE)
    return _LIB


def _check(volume, grid, offset, inv_scale):
    if volume.device.type != "mps" or volume.dtype != torch.uint8 or volume.dim() != 4:
        raise ValueError("volume must be an MPS uint8 (C, Z, Y, X) tensor")
    if grid.device.type != "mps" or grid.dtype != torch.float32 or grid.dim() != 4 or grid.size(3) != 3:
        raise ValueError("grid must be an MPS float32 (D, H, W, 3) tensor")
    for name, t in (("offset", offset), ("inv_scale", inv_scale)):
        if t.device.type != "mps" or t.dtype != torch.float32 or t.numel() != 3:
            raise ValueError(f"{name} must be an MPS float32 (3,) tensor")


def _params(volume, grid):
    C, Z, Y, X = volume.shape
    D, H, W = grid.shape[:3]
    # Strides in floats, so a non-contiguous grid needs no copy (as in CUDA);
    # the kernel indexes from the grid's storage offset.
    return [int(C), int(Z), int(Y), int(X), int(D * H * W), int(W), int(H),
            *(int(s) for s in grid.stride())]


def grid_sample_3d_u8(volume, grid, offset, inv_scale):
    """grid_sample_3d_u8 on MPS: (C, Z, Y, X) uint8 volume, (D, H, W, 3)
    full-resolution (x, y, z) grid -> (C, D, H, W) uint8."""
    _check(volume, grid, offset, inv_scale)
    volume = volume.contiguous()
    C = volume.shape[0]
    D, H, W = grid.shape[:3]
    out = torch.empty((C, D * H * W), dtype=torch.uint8, device=volume.device)
    if out.shape[1] > 0:
        _lib().gs3d_u8(volume, grid, offset.contiguous(), inv_scale.contiguous(), out,
                       _params(volume, grid), threads=out.shape[1], group_size=_GROUP)
    return out.reshape(C, D, H, W)


class _GridSample3DU8DiffMetal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, volume, grid, offset, inv_scale):
        volume = volume.contiguous()
        offset = offset.contiguous()
        inv_scale = inv_scale.contiguous()
        C = volume.shape[0]
        D, H, W = grid.shape[:3]
        out = torch.empty((C, D * H * W), dtype=torch.float32, device=volume.device)
        if out.shape[1] > 0:
            _lib().gs3d_u8_diff_fwd(volume, grid, offset, inv_scale, out,
                                    _params(volume, grid), threads=out.shape[1],
                                    group_size=_GROUP)
        ctx.save_for_backward(grid)
        ctx.volume = volume
        ctx.offset = offset
        ctx.inv_scale = inv_scale
        return out.reshape(C, D, H, W)

    @staticmethod
    def backward(ctx, grad_output):
        grid, = ctx.saved_tensors
        D, H, W = grid.shape[:3]
        grad_grid = torch.empty((D * H * W, 3), dtype=torch.float32, device=grid.device)
        if grad_grid.shape[0] > 0:
            _lib().gs3d_u8_diff_bwd(ctx.volume, grid, ctx.offset, ctx.inv_scale,
                                    grad_output.contiguous(), grad_grid,
                                    _params(ctx.volume, grid), threads=grad_grid.shape[0],
                                    group_size=_GROUP)
        return None, grad_grid.reshape(D, H, W, 3), None, None


def grid_sample_3d_u8_diff(volume, grid, offset, inv_scale):
    """grid_sample_3d_u8_diff on MPS: (C, D, H, W) float32 raw interpolated
    values, differentiable w.r.t. the grid positions."""
    _check(volume, grid, offset, inv_scale)
    return _GridSample3DU8DiffMetal.apply(volume, grid, offset, inv_scale)
