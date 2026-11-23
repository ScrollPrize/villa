"""PatchTokens' cell sampling on Apple GPUs as one gather kernel over a channels-last feature map

PatchTokens samples the CNN lattice at points fixed by the model (the same for every decision), so each token's
point features are a fixed linear combination of lattice cells: per sub-point the eight trilinear corners of
``F.grid_sample(padding_mode='border', align_corners=True)``, zero outside the crop, then the sub-point mean. The
combination is built once per lattice shape on the CPU in float64 and applied by one kernel that reads the map
channels-last (the layout MetalCropCNN produces), instead of grid_sample on a permuted view. The support column of
each point is a constant of the table. Anything else (several levels, other devices, autograd) runs the stock
forward.
"""
from __future__ import annotations

import torch

from vesuvius.neural_tracing.tracers.models.follower import crop_support, feature_grid
from vesuvius.neural_tracing.tracers.models.patch_tokens import PatchTokens

_SOURCE = r'''
#include <metal_stdlib>
using namespace metal;

// out [B, R, C + 1] = per row r: sum_j w[r, j] * feat[b, idx[r, j], :], then the row's support constant.
// feat is channels-last [B, V, C] as float4 groups; ints: R, J, G (= C / 4), V. Grid (R * G, B)
kernel void gather_rows(device const float4* feat [[buffer(0)]], device const int* idx [[buffer(1)]],
                        device const float* w [[buffer(2)]], device const float* support [[buffer(3)]],
                        device float* out [[buffer(4)]], device const int* ints [[buffer(5)]],
                        uint2 gid [[thread_position_in_grid]]) {
    const uint R = uint(ints[0]), J = uint(ints[1]), G = uint(ints[2]), V = uint(ints[3]);
    if (gid.x >= R * G) return;
    const uint r = gid.x / G, g = gid.x - r * G, b = gid.y;
    device const float4* base = feat + ulong(b) * V * G + g;
    float4 acc = 0.0f;
    for (uint j = 0; j < J; ++j) {
        const float wj = w[r * J + j];
        acc += wj * base[ulong(idx[r * J + j]) * G];
    }
    const uint C = 4 * G;
    device float* o = out + (ulong(b) * R + r) * (C + 1);
    o[4 * g] = acc.x; o[4 * g + 1] = acc.y; o[4 * g + 2] = acc.z; o[4 * g + 3] = acc.w;
    if (g == 0) o[C] = support[r];
}
'''
_LIBRARY = {}


def _library():
    if 'lib' not in _LIBRARY:
        _LIBRARY['lib'] = torch.mps.compile_shader(_SOURCE)
    return _LIBRARY['lib']


def gather_table(points, weights, crop, stride, shape):
    """The fixed combination of one PatchTokens part: (idx [R, J] int32, w [R, J] float32, support [R] float32) for
    rows R = tokens * points and J = 8 corners per sub-point, over a lattice of ``shape`` (D, H, W). ``points``
    (T * P * S, 3) and ``weights`` (T, P, S) are the part's buffers; float64 throughout, as grid_sample's border,
    align_corners=True trilinear sampling and PatchTokens' sub-point mean (a single sub-point is taken as is)"""
    t, p, s = weights.shape
    points = points.detach().cpu().double()  # MPS has no float64: move first
    supported = crop_support(points, crop)
    grid = feature_grid(torch.where(supported[..., None], points, 0.), crop, shape, stride)
    d, h, w_ = shape
    sizes = torch.tensor((w_, h, d), dtype=torch.float64)
    # align_corners=True unnormalization, then the border clamp
    coords = ((grid+1)/2*(sizes-1)).clamp(min=torch.zeros(3, dtype=torch.float64), max=sizes-1)
    lo = coords.floor()
    frac = coords-lo
    lo = lo.long()
    sub = weights.detach().cpu().double().reshape(-1) if s > 1 else torch.ones(t*p*s, dtype=torch.float64)
    scale = sub*supported.double()
    idx, wts = [], []
    for corner in range(8):
        offset = torch.tensor(((corner & 1), (corner >> 1) & 1, (corner >> 2) & 1))  # x, y, z
        at = lo+offset
        weight = torch.where(offset.bool(), frac, 1-frac).prod(-1)*scale
        inside = (at <= (sizes-1).long()).all(-1)  # a clamped coordinate's upper corner has weight 0
        at = torch.where(inside[:, None], at, lo)
        idx.append((at[:, 2]*h+at[:, 1])*w_+at[:, 0])
        wts.append(torch.where(inside, weight, 0.))
    # per point, its sub-points' corners side by side: (T * P, S * 8)
    idx = torch.stack(idx, -1).reshape(t*p, s*8)
    wts = torch.stack(wts, -1).reshape(t*p, s*8)
    support = (sub*supported.double()).reshape(t*p, s).sum(-1)
    return idx.int(), wts.float(), support.float()


class MetalPatchTokens(PatchTokens):
    """PatchTokens with the cell sampling of a single lattice as one Metal gather (``gather_rows``); same parameters"""

    @staticmethod
    def suits(module):
        return type(module) is PatchTokens and len(module.levels) == 1

    def forward(self, features):
        lattice = features if torch.is_tensor(features) else (features[0] if len(features) == 1 else None)
        if (lattice is None or lattice.device.type != 'mps' or lattice.dtype != torch.float32
                or torch.is_grad_enabled() or lattice.shape[1] % 4):
            return super().forward(features)
        return self.table_forward(lattice, metal_gather)

    def table_forward(self, lattice, gather):
        """The stock forward of one lattice (B, C, D, H, W) through the fixed tables, with ``gather(rows, idx, w,
        support, ints)`` -> (B, R, C + 1) over channels-last ``rows`` (B, D, H, W, C)"""
        rows = lattice.permute(0, 2, 3, 4, 1)  # channels-last storage when MetalCropCNN made it, else one copy
        if not rows.is_contiguous():
            rows = rows.contiguous()
        c = lattice.shape[1]
        shape = tuple(lattice.shape[2:])
        tables = self.__dict__.setdefault('_metal_tables', {})
        crop, stride = self.levels[0]
        tokens = []
        for name in self.groups:
            parts = []
            for part, (_, (t, p, s)) in enumerate(self.shapes[name]):
                key = (name, part, shape, lattice.device.type)
                if key not in tables:
                    idx, w, support = gather_table(getattr(self, f'{name}_points_{part}'),
                                                   getattr(self, f'{name}_weights_{part}'), crop, stride, shape)
                    ints = torch.tensor((t*p, idx.shape[1], c//4, shape[0]*shape[1]*shape[2]), dtype=torch.int32)
                    tables[key] = tuple(x.to(lattice.device) for x in (idx, w, support, ints))
                parts.append(gather(rows, *tables[key]).reshape(len(lattice), t, -1))
            order = getattr(self, f'{name}_order')
            values = parts[0] if len(parts) == 1 else torch.cat(parts, 1)[:, order]
            tokens.append(getattr(self, self.projection_name(name, 0))(values))
        return torch.cat(tokens, 1)


def metal_gather(rows, idx, w, support, ints):
    """``gather_rows`` on MPS: (B, R, C + 1)"""
    b, c = len(rows), rows.shape[-1]
    r = idx.shape[0]
    out = torch.empty((b, r, c+1), dtype=torch.float32, device=rows.device)
    _library().gather_rows(rows, idx, w, support, out, ints, threads=(r*(c//4), b), group_size=(256, 1))
    return out


def reference_gather(rows, idx, w, support, ints):
    """``gather_rows`` in torch, for tests: (B, R, C + 1)"""
    b, c = len(rows), rows.shape[-1]
    flat = rows.reshape(b, -1, c)
    values = (flat[:, idx.long()]*w[None, ..., None]).sum(2)
    return torch.cat((values, support[None, :, None].expand(b, -1, 1)), -1)
