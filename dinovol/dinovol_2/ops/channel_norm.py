"""Channel normalization for the CNN, with bounded, stage-local backward storage.

The opaque custom-op boundary is intentional: Inductor must not defer affine
gradient reductions and fuse them with other stages, keeping volume-sized tensors
alive across convolution backwards. Both gradients finish before this op returns.
No input is mutated, and no per-voxel FP32 statistics are retained.
"""

from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


def supported(x, weight, bias, residual=None):
    return (
        triton is not None and torch.version.hip is None and x.is_cuda
        and x.ndim == 5 and 0 < x.shape[1] <= 256 and x.numel() > 0
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and weight.dtype == bias.dtype == torch.float32
        and weight.device == bias.device == x.device
        and weight.is_contiguous() and bias.is_contiguous()
        and weight.shape == bias.shape == (x.shape[1],)
        and (residual is None or (
            residual.shape == x.shape and residual.dtype == x.dtype and residual.device == x.device
        ))
        and torch.cuda.get_device_capability(x.device)[0] >= 8
    )


def reference(x, weight, bias, eps, residual=None, relu=False):
    """Original FP32 statistics/affine and explicit activation rounding boundaries."""
    dtype = x.dtype
    xf = x.float()
    centered = xf - xf.mean(1, keepdim=True)
    normalized = centered / torch.sqrt(centered.square().mean(1, keepdim=True) + eps)
    idx = (None, slice(None), *([None] * (x.ndim - 2)))
    out = (weight[idx] * normalized + bias[idx]).to(dtype)
    if residual is not None:
        out = out + residual
    return torch.relu(out) if relu else out


if triton is not None:
    @triton.jit
    def _offset(rows, cols, D: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
                S: tl.constexpr):
        # Strided views and large volumes can exceed 32-bit element offsets.
        rows = rows.to(tl.int64)
        return (
            (rows // (D * H * W))[:, None] * S[0]
            + cols[None, :] * S[1]
            + ((rows // (H * W)) % D)[:, None] * S[2]
            + ((rows // W) % H)[:, None] * S[3]
            + (rows % W)[:, None] * S[4]
        )

    @triton.jit
    def _normalize(X, Weight, Bias, rows, cols, M, C: tl.constexpr,
                   D: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
                   XS: tl.constexpr, EPS: tl.constexpr):
        mask = (rows[:, None] < M) & (cols[None, :] < C)
        x = tl.load(X + _offset(rows, cols, D, H, W, XS), mask, other=0).to(tl.float32)
        mean = tl.sum(x, 1) / C
        centered = tl.where(cols[None, :] < C, x - mean[:, None], 0.)
        var = tl.sum(centered * centered, 1) / C
        std = tl.sqrt(var + EPS)
        normalized = centered / std[:, None]
        weight = tl.load(Weight + cols, cols < C, other=0.)
        bias = tl.load(Bias + cols, cols < C, other=0.)
        affine = normalized * weight[None, :] + bias[None, :]
        return normalized, std, weight, affine

    @triton.jit
    def _forward(X, Weight, Bias, Residual, Y, M, C: tl.constexpr,
                 D: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
                 XS: tl.constexpr, RS: tl.constexpr, EPS: tl.constexpr,
                 RESIDUAL: tl.constexpr, RELU: tl.constexpr,
                 BR: tl.constexpr, BC: tl.constexpr):
        rows = tl.program_id(0) * BR + tl.arange(0, BR)
        cols = tl.arange(0, BC)
        _, _, _, affine = _normalize(X, Weight, Bias, rows, cols, M, C, D, H, W, XS, EPS)
        # Match norm.to(x.dtype), then residual addition in that dtype.
        y = affine.to(Y.dtype.element_ty).to(tl.float32)
        if RESIDUAL:
            residual = tl.load(Residual + _offset(rows, cols, D, H, W, RS),
                               (rows[:, None] < M) & (cols[None, :] < C), other=0).to(tl.float32)
            y = (y + residual).to(Y.dtype.element_ty).to(tl.float32)
        if RELU:
            y = tl.maximum(y, 0.)
        tl.store(Y + rows.to(tl.int64)[:, None] * C + cols[None, :], y,
                 (rows[:, None] < M) & (cols[None, :] < C))

    @triton.jit
    def _backward(X, Weight, Bias, Residual, DY, DX, DR, PartialW, PartialB,
                  M, C: tl.constexpr, D: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
                  XS: tl.constexpr, RS: tl.constexpr, GS: tl.constexpr, EPS: tl.constexpr,
                  RESIDUAL: tl.constexpr, RELU: tl.constexpr,
                  BR: tl.constexpr, BC: tl.constexpr, GROUPS: tl.constexpr):
        cols = tl.arange(0, BC)
        part_w = tl.full((BR, BC), 0., tl.float32)
        part_b = tl.full((BR, BC), 0., tl.float32)
        # Fixed partitions and a separate final reduction avoid atomics/locks.
        for start in range(tl.program_id(0) * BR, M, GROUPS * BR):
            rows = start + tl.arange(0, BR)
            mask = (rows[:, None] < M) & (cols[None, :] < C)
            normalized, std, weight, affine = _normalize(
                X, Weight, Bias, rows, cols, M, C, D, H, W, XS, EPS)
            dy = tl.load(DY + _offset(rows, cols, D, H, W, GS), mask, other=0).to(tl.float32)
            if RELU:
                y = affine.to(X.dtype.element_ty).to(tl.float32)
                if RESIDUAL:
                    residual = tl.load(Residual + _offset(rows, cols, D, H, W, RS),
                                       mask, other=0).to(tl.float32)
                    y = (y + residual).to(X.dtype.element_ty).to(tl.float32)
                dy = tl.where(y <= 0., 0., dy)
            if RESIDUAL:
                tl.store(DR + rows.to(tl.int64)[:, None] * C + cols[None, :], dy, mask)
            dy = tl.where(mask, dy, 0.)
            weighted = dy * weight[None, :]
            mean_dy = tl.sum(weighted, 1) / C
            mean_dyx = tl.sum(weighted * normalized, 1) / C
            dx = (weighted - mean_dy[:, None] - normalized * mean_dyx[:, None]) / std[:, None]
            tl.store(DX + rows.to(tl.int64)[:, None] * C + cols[None, :], dx, mask)
            part_w += dy * normalized
            part_b += dy
        tl.store(PartialW + tl.program_id(0) * C + cols, tl.sum(part_w, 0), cols < C)
        tl.store(PartialB + tl.program_id(0) * C + cols, tl.sum(part_b, 0), cols < C)

    @triton.jit
    def _affine_reduce(PW, PB, DW, DB, C: tl.constexpr, GROUPS: tl.constexpr,
                       BG: tl.constexpr, BC: tl.constexpr):
        groups = tl.arange(0, BG)
        cols = tl.program_id(0) * BC + tl.arange(0, BC)
        offset = groups[:, None] * C + cols[None, :]
        mask = (groups[:, None] < GROUPS) & (cols[None, :] < C)
        w = tl.sum(tl.load(PW + offset, mask, other=0.), 0)
        b = tl.sum(tl.load(PB + offset, mask, other=0.), 0)
        tl.store(DW + cols, w, cols < C)
        tl.store(DB + cols, b, cols < C)


def _empty_activation(x):
    return torch.empty_like(x, memory_format=torch.channels_last_3d)


@torch.library.custom_op("dinovol::channel_norm", mutates_args=())
def channel_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                 eps: float, residual: Optional[torch.Tensor], relu: bool) -> torch.Tensor:
    y = _empty_activation(x)
    n, c, d, h, w = x.shape
    rows, cols = 128, triton.next_power_of_2(c)
    with torch.cuda.device(x.device):
        _forward[(triton.cdiv(n * d * h * w, rows),)](
            x, weight, bias, residual, y, n * d * h * w, c, d, h, w,
            x.stride(), residual.stride() if residual is not None else x.stride(),
            eps, residual is not None, relu, rows, cols, num_warps=4, enable_fp_fusion=False,
        )
    return y


@channel_norm.register_fake
def _fake_forward(x, weight, bias, eps, residual, relu):
    return _empty_activation(x)


@torch.library.custom_op("dinovol::channel_norm_backward", mutates_args=())
def _channel_norm_backward(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                           eps: float, residual: Optional[torch.Tensor], relu: bool,
                           dy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n, c, d, h, w = x.shape
    rows, cols = 128, triton.next_power_of_2(c)
    m = n * d * h * w
    groups = min(triton.cdiv(m, rows), 2048)
    dx = _empty_activation(x)
    dr = _empty_activation(x) if residual is not None else x.new_empty(0)
    dw, db = torch.empty_like(weight), torch.empty_like(bias)
    pw = torch.empty((groups, c), device=x.device, dtype=torch.float32)
    pb = torch.empty_like(pw)
    with torch.cuda.device(x.device):
        _backward[(groups,)](
            x, weight, bias, residual, dy, dx, dr, pw, pb, m, c, d, h, w,
            x.stride(), residual.stride() if residual is not None else x.stride(), dy.stride(),
            eps, residual is not None, relu, rows, cols, groups, num_warps=4, enable_fp_fusion=False,
        )
        _affine_reduce[(triton.cdiv(c, 16),)](
            pw, pb, dw, db, c, groups, triton.next_power_of_2(groups), 16,
            num_warps=8, enable_fp_fusion=False,
        )
    return dx, dw, db, dr


@_channel_norm_backward.register_fake
def _fake_backward(x, weight, bias, eps, residual, relu, dy):
    return (_empty_activation(x), torch.empty_like(weight), torch.empty_like(bias),
            _empty_activation(x) if residual is not None else x.new_empty(0))


def _setup_context(ctx, inputs, output):
    x, weight, bias, eps, residual, relu = inputs
    ctx.save_for_backward(x, weight, bias, residual)
    ctx.eps, ctx.relu = eps, relu


def _autograd_backward(ctx, dy):
    x, weight, bias, residual = ctx.saved_tensors
    dx, dw, db, dr = _channel_norm_backward(x, weight, bias, ctx.eps, residual, ctx.relu, dy)
    return dx, dw, db, None, dr if residual is not None else None, None


channel_norm.register_autograd(_autograd_backward, setup_context=_setup_context)
