"""Apple GPU (PyTorch MPS) speed-ups for the tracers' models that keep their weights and outputs

``optimize_for_mps(model)`` changes layers in place after the checkpoint is loaded. Parameters, buffers and
``state_dict`` keys are unchanged, so a model can still be saved or moved back to another device.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from vesuvius.models.build.resblocks import BasicBlockD
from vesuvius.models.build.simple_conv_blocks import ConvDropoutNormReLU


def folded_empty(b, c, d, h, w, device):
    """A (B, C, D, H, W) float32 view into a new (B, D + 2, C, H, W) buffer whose first and last depth slice of
    every sample are zero: the layout :class:`DepthTapConv3d` reads without copies. Write the view, never the pads"""
    base = torch.empty((b, d+2, c, h, w), dtype=torch.float32, device=device)
    base[:, 0].zero_()
    base[:, d+1].zero_()
    base._zero_depth_pads = True
    return base[:, 1:d+1].transpose(1, 2)


def _folded_rows(x):
    """The (B * (D + 2), C, H, W) rows behind ``x`` when it is a :func:`folded_empty` view, else None"""
    base = x._base
    if base is None or not getattr(base, '_zero_depth_pads', False):
        return None
    b, c, d, h, w = x.shape
    if (tuple(base.shape) != (b, d+2, c, h, w) or x.stride() != ((d+2)*c*h*w, h*w, c*h*w, w, 1)
            or x.storage_offset() != base.storage_offset()+c*h*w):
        return None
    return base.view(b*(d+2), c, h, w)


class DepthTapConv3d(nn.Conv3d):
    """A 3x3x3, padding-1, stride-1 Conv3d run as the sum of three conv2d over its depth taps, with the depth
    slices folded into the batch.

    The input is laid out as :func:`folded_empty` rows, sample-major with a zero depth slice either side of each
    sample, so tap ``dz`` reads one contiguous row range and output row ``b * (D + 2) + i`` is depth ``i`` of sample
    ``b``; the two rows per sample that straddle samples are computed and dropped. An input already in that layout
    (the fused blocks write their norms into it) costs no copy, any other one copy. On an M4 Pro this runs the strip
    model's CropCNN convs 2.0-2.5x faster than MPS conv3d and matches float64 more closely (max error about 1e-6
    against conv3d's 1e-5). Each tap weight is made contiguous on every call: MPS conv2d gives wrong results for a
    weight view with a non-zero storage offset (``weight[:, :, 1]``).
    """

    @staticmethod
    def suits(module):
        return (type(module) is nn.Conv3d and module.kernel_size == (3, 3, 3) and module.stride == (1, 1, 1)
                and module.padding == (1, 1, 1) and module.dilation == (1, 1, 1) and module.groups == 1
                and module.padding_mode == 'zeros')

    def forward(self, x):
        b, c, d, h, w = x.shape
        rows = _folded_rows(x)
        if rows is None:
            folded = folded_empty(b, c, d, h, w, x.device)
            folded.copy_(x)
            rows = _folded_rows(folded)
        n = rows.shape[0]-2
        total = None
        for dz in range(3):
            bias = self.bias if dz == 0 else None
            y = F.conv2d(rows[dz:dz+n], self.weight[:, :, dz].contiguous(), bias, padding=1)
            total = y if total is None else total.add_(y)
        o = total.shape[1]
        return total.as_strided((b, o, d, h, w), ((d+2)*o*h*w, h*w, o*h*w, w, 1))


class EvenAdaptiveAvgPool3d(nn.AdaptiveAvgPool3d):
    """AdaptiveAvgPool3d, which MPS does not implement: avg_pool3d where every input size is a multiple of the
    output's (the same windows), the CPU otherwise."""

    def forward(self, x):
        out = nn.modules.utils._triple(self.output_size)
        sizes = x.shape[-3:]
        if x.device.type == 'mps' and all(o and s % o == 0 for s, o in zip(sizes, out)):
            window = tuple(s//o for s, o in zip(sizes, out))
            return F.avg_pool3d(x, window, window)
        if x.device.type == 'mps':
            return super().forward(x.cpu()).to(x.device)
        return super().forward(x)


def optimize_for_mps(model):
    """Swap layers of ``model`` in place for MPS ones with the same parameters and outputs: a CropCNN the Metal kernels
    cover for :class:`~vesuvius.neural_tracing.tracers.models.mps_cnn.MetalCropCNN` (macOS 26 and later, whose Metal
    has the tensor ops it needs); single-lattice PatchTokens for
    :class:`~vesuvius.neural_tracing.tracers.models.mps_tokens.MetalPatchTokens`; stride-1 3x3x3 Conv3d for
    :class:`DepthTapConv3d` (or :class:`DirectConv3d` when it has few channels), AdaptiveAvgPool3d for
    :class:`EvenAdaptiveAvgPool3d` and InstanceNorm/LeakyReLU ResNet-D blocks for :class:`FusedBasicBlockD`, which
    also serve MetalCropCNN's fallback. Returns how many."""
    from vesuvius.neural_tracing.tracers.models.mps_cnn import MetalCropCNN  # imports blocks, which imports this
    from vesuvius.neural_tracing.tracers.models.mps_tokens import MetalPatchTokens
    swapped = 0
    for module in model.modules():
        if MetalCropCNN.suits(module):
            module.__class__ = MetalCropCNN
            swapped += 1
        if MetalPatchTokens.suits(module):
            module.__class__ = MetalPatchTokens
            swapped += 1
        if FusedBasicBlockD.suits(module):
            module.__class__ = FusedBasicBlockD
            swapped += 1
        if DirectConv3d.suits(module):
            module.__class__ = DirectConv3d
            swapped += 1
        elif DepthTapConv3d.suits(module):
            module.__class__ = DepthTapConv3d
            swapped += 1
        elif type(module) is nn.AdaptiveAvgPool3d and all(o is not None for o in nn.modules.utils._triple(module.output_size)):
            module.__class__ = EvenAdaptiveAvgPool3d
            swapped += 1
    return swapped


# Gaussian tube rendering (sheet_strip/model.py render_tube) without a host sync per crop: one thread per point and
# offset, each sample keeping its largest weight by an atomic max on the weight's bits (weights are positive, so their
# float and unsigned orders agree, and the result does not depend on thread order)
_TUBE = r'''
#include <metal_stdlib>
using namespace metal;
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)

// ints: B, P, K, D, H, W
kernel void tube_max(device atomic_uint* out [[buffer(0)]],
                     device const float* index [[buffer(1)]],
                     device const uchar* valid [[buffer(2)]],
                     device const int* offsets [[buffer(3)]],
                     device const int* ints [[buffer(4)]],
                     constant float& denominator [[buffer(5)]],
                     uint gid [[thread_position_in_grid]]) {
    const int P = ints[1], K = ints[2], D = ints[3], H = ints[4], W = ints[5];
    const uint point = gid / uint(K), k = gid - point * uint(K);
    if (point >= uint(ints[0] * P) || !valid[point]) return;
    const uint b = point / uint(P);
    float centre[3], q = 0.f;
    int sample[3];
    const int size[3] = {D, H, W};
    for (int a = 0; a < 3; ++a) {
        centre[a] = index[point * 3 + a];
        sample[a] = int(rint(centre[a])) + offsets[k * 3 + a];  // torch.round: half to even
        if (sample[a] < 0 || sample[a] >= size[a]) return;
    }
    for (int a = 0; a < 3; ++a) {
        const float d = float(sample[a]) - centre[a];
        q += d * d;
    }
    const float weight = precise::exp(-q / denominator);
    const uint linear = ((b * uint(D) + uint(sample[0])) * uint(H) + uint(sample[1])) * uint(W) + uint(sample[2]);
    atomic_fetch_max_explicit(out + linear, as_type<uint>(weight), memory_order_relaxed);
}
'''

_TUBE_LIBRARY = None


def render_tube_max(index, valid, offsets, sigma, out):
    """``render_tube``'s per-crop loop on the GPU: ``out`` (B, 1, D, H, W) float32, zeroed, takes at each sample the
    largest Gaussian weight of the valid crop-index points ``index`` (B, P, 3; d, h, w) over ``offsets`` (K, 3)."""
    global _TUBE_LIBRARY
    if _TUBE_LIBRARY is None:
        _TUBE_LIBRARY = torch.mps.compile_shader(_TUBE)
    b, p, _ = index.shape
    k = len(offsets)
    ints = torch.tensor([b, p, k, *out.shape[2:]], dtype=torch.int32, device=out.device)
    if b*p*k:
        _TUBE_LIBRARY.tube_max(out.view(torch.int32), index.float().contiguous(), valid.contiguous().view(torch.uint8),
                               offsets.to(device=out.device, dtype=torch.int32).contiguous(), ints,
                               float(torch.tensor(2*sigma*sigma, dtype=torch.float32)), threads=b*p*k)
    return out


# Fused InstanceNorm for ResNet-D basic blocks: per-plane statistics (two passes in one kernel), then one pass that
# normalizes, adds the residual (itself normalized for a projection skip) and applies LeakyReLU. Inputs and outputs
# are read and written through 5D strides, so a DepthTapConv3d output (a transposed view) needs no copy
_NORM = r'''
#include <metal_stdlib>
using namespace metal;

constant constexpr uint T = 256;

// ints: B, C, D, H, W, then element strides b, c, d, h, w. One threadgroup per (b, c) plane: mean, then the mean of
// squared deviations (biased variance, as InstanceNorm), out[2 g] = mean, out[2 g + 1] = 1 / sqrt(var + eps)
kernel void plane_stats(device const float* x [[buffer(0)]],
                        device const long* ints [[buffer(1)]],
                        device float* out [[buffer(2)]],
                        constant float& eps [[buffer(3)]],
                        uint g [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
    const long C = ints[1], D = ints[2], H = ints[3], W = ints[4];
    const long b = long(g) / C, c = long(g) % C;
    device const float* p = x + b * ints[5] + c * ints[6];
    const long sd = ints[7], sh = ints[8], sw = ints[9];
    threadgroup float part[T / 32];
    threadgroup float shared_mean;
    float acc = 0.f;
    for (long hw = tid; hw < H * W; hw += T) {
        const long h = hw / W, w = hw - h * W;
        device const float* q = p + h * sh + w * sw;
        for (long d = 0; d < D; ++d) acc += q[d * sd];
    }
    acc = simd_sum(acc);
    if (lane == 0) part[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.f;
        for (uint i = 0; i < T / 32; ++i) total += part[i];
        shared_mean = total / float(D * H * W);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float mean = shared_mean;
    acc = 0.f;
    for (long hw = tid; hw < H * W; hw += T) {
        const long h = hw / W, w = hw - h * W;
        device const float* q = p + h * sh + w * sw;
        for (long d = 0; d < D; ++d) {
            const float e = q[d * sd] - mean;
            acc += e * e;
        }
    }
    acc = simd_sum(acc);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) part[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.f;
        for (uint i = 0; i < T / 32; ++i) total += part[i];
        out[2 * g] = mean;
        out[2 * g + 1] = 1.f / sqrt(total / float(D * H * W) + eps);
    }
}

// ints: B, C, D, H, W, mode (0: none, 1: residual as is, 2: residual normalized), activation (0/1), then element
// strides (b, c, d, h, w) of x, residual and out. Grid (H * W, D, B * C)
kernel void norm_apply(device const float* x [[buffer(0)]],
                       device const float* stats [[buffer(1)]],
                       device const float* gamma [[buffer(2)]],
                       device const float* beta [[buffer(3)]],
                       device const float* residual [[buffer(4)]],
                       device const float* rstats [[buffer(5)]],
                       device const float* rgamma [[buffer(6)]],
                       device const float* rbeta [[buffer(7)]],
                       device float* out [[buffer(8)]],
                       device const long* ints [[buffer(9)]],
                       constant float& slope [[buffer(10)]],
                       uint3 gid [[thread_position_in_grid]]) {
    const long C = ints[1], H = ints[3], W = ints[4];
    if (long(gid.x) >= H * W) return;
    const long h = long(gid.x) / W, w = long(gid.x) - h * W, d = gid.y, g = gid.z;
    const long b = g / C, c = g - b * C;
    const long offset = 7;
    const long xi = b * ints[offset] + c * ints[offset + 1] + d * ints[offset + 2] + h * ints[offset + 3] + w * ints[offset + 4];
    float y = (x[xi] - stats[2 * g]) * stats[2 * g + 1] * gamma[c] + beta[c];
    const long mode = ints[5];
    if (mode) {
        const long ri = b * ints[offset + 5] + c * ints[offset + 6] + d * ints[offset + 7] + h * ints[offset + 8] + w * ints[offset + 9];
        float r = residual[ri];
        if (mode == 2) r = (r - rstats[2 * g]) * rstats[2 * g + 1] * rgamma[c] + rbeta[c];
        y += r;
    }
    if (ints[6] && y < 0.f) y *= slope;
    out[b * ints[offset + 10] + c * ints[offset + 11] + d * ints[offset + 12] + h * ints[offset + 13] + w * ints[offset + 14]] = y;
}
'''

_NORM_LIBRARY = None


def _norm_library():
    global _NORM_LIBRARY
    if _NORM_LIBRARY is None:
        _NORM_LIBRARY = torch.mps.compile_shader(_NORM)
    return _NORM_LIBRARY


def plane_stats(x, eps):
    """Per (b, c) plane of the MPS float32 (B, C, D, H, W) ``x`` (any strides): (B * C, 2) mean and
    1 / sqrt(biased variance + eps), as InstanceNorm3d computes them."""
    b, c = x.shape[:2]
    ints = torch.tensor([*x.shape, *x.stride()], dtype=torch.int64, device=x.device)
    out = torch.empty((b*c, 2), dtype=torch.float32, device=x.device)
    _norm_library().plane_stats(x, ints, out, float(eps), threads=b*c*256, group_size=256)
    return out


def norm_apply(x, stats, norm, *, residual=None, residual_stats=None, residual_norm=None, slope=None, out=None):
    """``leaky_relu(norm(x) + r)``: ``norm`` an affine InstanceNorm3d with ``stats`` (:func:`plane_stats`), ``r`` the
    ``residual`` as is, or normalized by ``residual_norm`` with ``residual_stats``, or absent; no activation when
    ``slope`` is None. Writes ``out`` (default: a new contiguous tensor), any strides."""
    out = torch.empty(x.shape, dtype=torch.float32, device=x.device) if out is None else out
    mode = 0 if residual is None else 1 if residual_norm is None else 2
    r = x if residual is None else residual
    rs = stats if residual_stats is None else residual_stats
    rn = norm if residual_norm is None else residual_norm
    ints = torch.tensor([*x.shape, mode, int(slope is not None), *x.stride(), *r.stride(), *out.stride()],
                        dtype=torch.int64, device=x.device)
    b, c, d, h, w = x.shape
    _norm_library().norm_apply(x, stats, norm.weight, norm.bias, r, rs, rn.weight, rn.bias, out, ints,
                               float(0. if slope is None else slope), threads=(h*w, d, b*c),
                               group_size=(min(h*w, 256), 1, 1))
    return out


def _instance_norm(module):
    return (type(module) is nn.InstanceNorm3d and module.affine and not module.track_running_stats
            and module.weight is not None)


def _plain_conv_norm(block, activation):
    """A ConvDropoutNormReLU of conv, affine InstanceNorm3d and (``activation``) LeakyReLU, nothing else"""
    if type(block) is not ConvDropoutNormReLU or hasattr(block, 'dropout') or not _instance_norm(getattr(block, 'norm', None)):
        return False
    return type(getattr(block, 'nonlin', None)) is nn.LeakyReLU if activation else not hasattr(block, 'nonlin')


class FusedBasicBlockD(BasicBlockD):
    """A ResNet-D basic block (conv, InstanceNorm, LeakyReLU; conv, InstanceNorm; plus the skip; LeakyReLU) whose
    norms, residual add and activations run as two Metal kernels per norm on MPS without autograd
    (:func:`plane_stats`, :func:`norm_apply`): the same arithmetic as the modules, without their separate passes"""

    @staticmethod
    def suits(module):
        if type(module) is not BasicBlockD or module.apply_se or module.apply_stochastic_depth:
            return False
        if not (_plain_conv_norm(module.conv1, True) and _plain_conv_norm(module.conv2, False)
                and type(module.nonlin2) is nn.LeakyReLU):
            return False
        if not isinstance(module.skip, nn.Sequential):
            return True
        *pools, last = module.skip
        return (all(type(m) is nn.AvgPool3d for m in pools)
                and (type(last) is nn.AvgPool3d or _plain_conv_norm(last, False)))

    def forward(self, x):
        if x.device.type != 'mps' or x.dtype != torch.float32 or torch.is_grad_enabled():
            return super().forward(x)
        c1, c2 = self.conv1, self.conv2
        y = c1.conv(x)
        # both norms write the zero-padded depth layout, so a following DepthTapConv3d reads them without a copy
        hidden = norm_apply(y, plane_stats(y, c1.norm.eps), c1.norm, slope=c1.nonlin.negative_slope,
                            out=folded_empty(*y.shape, y.device))
        y = c2.conv(hidden)
        stats = plane_stats(y, c2.norm.eps)
        slope = self.nonlin2.negative_slope
        out = folded_empty(*y.shape, y.device)
        if not isinstance(self.skip, nn.Sequential):
            return norm_apply(y, stats, c2.norm, residual=x, slope=slope, out=out)
        residual = x
        for module in self.skip:
            if type(module) is nn.AvgPool3d:
                residual = module(residual)
        last = self.skip[-1]
        if type(last) is nn.AvgPool3d:
            return norm_apply(y, stats, c2.norm, residual=residual, slope=slope, out=out)
        residual = last.conv(residual)
        return norm_apply(y, stats, c2.norm, residual=residual, residual_stats=plane_stats(residual, last.norm.eps),
                          residual_norm=last.norm, slope=slope, out=out)


def start_queued_work():
    """Commit the MPS stream's queued work so the GPU starts it now: PyTorch MPS otherwise holds encoded work until
    something waits for it, leaving the GPU idle while the host builds the next batch. Recording an event commits
    the command buffer without waiting for it"""
    torch.mps.Event().record()


# Direct convolution for layers with few channels, where MPS conv3d leaves the GPU mostly idle (the strip model's
# first stage: 2 -> 12 channels at stride 2 on 144 x 128 x 128 crops, 12 -> 12 at stride 1, the 2 -> 12 skip). One
# thread per output voxel computes every output channel; the channel counts, kernel size and stride are compile-time
# constants, so the loops unroll. A kernel is built per layer shape, once
_DIRECT_CONV = r'''
#include <metal_stdlib>
using namespace metal;
constant constexpr int CI = {ci}, CO = {co}, K = {k}, S = {s}, P = {p};

// x (B, CI, D, H, W) and out (B, CO, Do, Ho, Wo) contiguous; w (CO, CI, K, K, K); ints: B, D, H, W, Do, Ho, Wo
kernel void direct_conv(device const float* x [[buffer(0)]],
                        device const float* w [[buffer(1)]],
                        device const float* bias [[buffer(2)]],
                        device float* out [[buffer(3)]],
                        device const int* ints [[buffer(4)]],
                        constant int& has_bias [[buffer(5)]],
                        uint3 gid [[thread_position_in_grid]]) {
    const int D = ints[1], H = ints[2], W = ints[3], Do = ints[4], Ho = ints[5], Wo = ints[6];
    const int ow = int(gid.x), oh = int(gid.y);
    if (ow >= Wo || oh >= Ho) return;
    const int b = int(gid.z) / Do, od = int(gid.z) - b * Do;
    float acc[CO];
    for (int o = 0; o < CO; ++o) acc[o] = has_bias ? bias[o] : 0.f;
    const long plane = long(D) * H * W;
    device const float* xb = x + long(b) * CI * plane;
    for (int kd = 0; kd < K; ++kd) {
        const int id = od * S - P + kd;
        if (id < 0 || id >= D) continue;
        for (int kh = 0; kh < K; ++kh) {
            const int ih = oh * S - P + kh;
            if (ih < 0 || ih >= H) continue;
            for (int kw = 0; kw < K; ++kw) {
                const int iw = ow * S - P + kw;
                if (iw < 0 || iw >= W) continue;
                const long at = (long(id) * H + ih) * W + iw;
                for (int i = 0; i < CI; ++i) {
                    const float v = xb[i * plane + at];
                    for (int o = 0; o < CO; ++o) acc[o] = fma(v, w[((o * CI + i) * K + kd) * K * K + kh * K + kw], acc[o]);
                }
            }
        }
    }
    const long oplane = long(Do) * Ho * Wo;
    device float* ob = out + long(b) * CO * oplane + (long(od) * Ho + oh) * Wo + ow;
    for (int o = 0; o < CO; ++o) ob[o * oplane] = acc[o];
}
'''

_DIRECT_LIBRARIES = {}


class DirectConv3d(nn.Conv3d):
    """A Conv3d with few channels (both at most 16), cube kernel 1 or 3 with padding k // 2 and equal strides, as a
    direct Metal kernel on MPS without autograd (module comment above); the stock forward elsewhere."""

    @staticmethod
    def suits(module):
        if type(module) is not nn.Conv3d:
            return False
        k, s = module.kernel_size, module.stride
        return (module.in_channels <= 16 and module.out_channels <= 16
                and module.groups == 1 and module.dilation == (1, 1, 1) and module.padding_mode == 'zeros'
                and len(set(k)) == 1 and k[0] in (1, 3) and len(set(s)) == 1 and s[0] in (1, 2)
                and module.padding == (k[0]//2,)*3)

    def forward(self, x):
        if x.device.type != 'mps' or x.dtype != torch.float32 or torch.is_grad_enabled():
            return super().forward(x)
        k, s = self.kernel_size[0], self.stride[0]
        key = (self.in_channels, self.out_channels, k, s)
        if key not in _DIRECT_LIBRARIES:
            _DIRECT_LIBRARIES[key] = torch.mps.compile_shader(
                _DIRECT_CONV.replace('{ci}', str(key[0])).replace('{co}', str(key[1])).replace('{k}', str(k))
                .replace('{s}', str(s)).replace('{p}', str(k//2)))
        x = x.contiguous()
        b, _, d, h, w = x.shape
        size = [(n+2*(k//2)-k)//s+1 for n in (d, h, w)]
        out = torch.empty((b, self.out_channels, *size), dtype=torch.float32, device=x.device)
        ints = torch.tensor([b, d, h, w, *size], dtype=torch.int32, device=x.device)
        bias = self.bias if self.bias is not None else self.weight
        _DIRECT_LIBRARIES[key].direct_conv(x, self.weight.contiguous(), bias, out, ints, int(self.bias is not None),
                                           threads=(size[2], size[1], b*size[0]),
                                           group_size=(min(size[2], 64), max(1, 256//min(size[2], 64)), 1))
        return out
