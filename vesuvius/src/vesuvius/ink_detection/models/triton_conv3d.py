"""Triton fp16 kernels for the 16->16 3x3x3 stem convolution on ROCm.

The forward and input-gradient passes are one implicit-GEMM kernel over
channels-last rows [B*D*H*W, C]; the input gradient is the same convolution
with the kernel flipped in space and Cin/Cout swapped. The weight gradient
splits the sum over rows and accumulates with fp32 atomics. Any case the
kernels do not cover runs the ordinary ``nn.Conv3d`` forward.
"""

from __future__ import annotations

import threading
import warnings

import torch
from torch import nn

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only, macOS and Windows installs
    triton = None


SUPPORTED_CHANNELS = 16

# The autotuners keep per-call state, so DataParallel replicas must not launch concurrently.
_launch_lock = threading.Lock()
_device_works: dict[torch.device, bool] = {}

if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BM": bm, "BN": bn}, num_warps=w, num_stages=st)
            for bm in (64, 128, 256)
            for bn in (32, 64, 128)
            for w in (4, 8)
            for st in (1, 2)
        ],
        key=["Cin", "Cout", "D", "H", "W"],
    )
    @triton.jit
    def _conv3d_k3_kernel(
        x, w, y, B, D, H, W, Cin, Cout,
        BK: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
    ):
        pm = tl.program_id(0)
        pn = tl.program_id(1)
        m = pm * BM + tl.arange(0, BM)
        n = pn * BN + tl.arange(0, BN)
        M = B * D * H * W
        wi = m % W
        hi = (m // W) % H
        di = (m // (W * H)) % D
        bi = m // (W * H * D)
        acc = tl.zeros((BM, BN), tl.float32)
        for t in range(27):
            dd = di + t // 9 - 1
            hh = hi + (t // 3) % 3 - 1
            ww = wi + t % 3 - 1
            ok = (m < M) & (dd >= 0) & (dd < D) & (hh >= 0) & (hh < H) & (ww >= 0) & (ww < W)
            r = (bi * D + dd) * H * W + hh * W + ww
            part = tl.zeros((BM, BN), tl.float32)
            for c0 in range(0, Cin, BK):
                c = c0 + tl.arange(0, BK)
                a = tl.load(x + r[:, None] * Cin + c[None, :], mask=ok[:, None] & (c[None, :] < Cin), other=0.0)
                b = tl.load(w + (t * Cin + c)[:, None] * Cout + n[None, :], mask=(c[:, None] < Cin) & (n[None, :] < Cout), other=0.0)
                part += tl.dot(a, b)
            acc += part
        om = (m[:, None] < M) & (n[None, :] < Cout)
        tl.store(y + m[:, None] * Cout + n[None, :], acc.to(tl.float16), mask=om)

    @triton.autotune(
        configs=[
            triton.Config({"BM": bm, "BC": bc, "BN": bn}, num_warps=w)
            for bm in (64, 128)
            for bc in (32, 64)
            for bn in (32, 64)
            for w in (4, 8)
        ],
        key=["Cin", "Cout", "D", "H", "W"],
        reset_to_zero=["dw"],
    )
    @triton.jit
    def _conv3d_k3_wgrad_kernel(
        x, g, dw, B, D, H, W, Cin, Cout, MCH,
        BM: tl.constexpr, BC: tl.constexpr, BN: tl.constexpr,
    ):
        t = tl.program_id(0)
        pc = tl.program_id(1)
        pn = tl.program_id(2) % tl.cdiv(Cout, BN)
        ks = tl.program_id(2) // tl.cdiv(Cout, BN)
        c = pc * BC + tl.arange(0, BC)
        n = pn * BN + tl.arange(0, BN)
        M = B * D * H * W
        acc = tl.zeros((BC, BN), tl.float32)
        for m0 in range(ks * MCH, tl.minimum(ks * MCH + MCH, M), BM):
            m = m0 + tl.arange(0, BM)
            wi = m % W
            hi = (m // W) % H
            di = (m // (W * H)) % D
            bi = m // (W * H * D)
            dd = di + t // 9 - 1
            hh = hi + (t // 3) % 3 - 1
            ww = wi + t % 3 - 1
            ok = (m < M) & (dd >= 0) & (dd < D) & (hh >= 0) & (hh < H) & (ww >= 0) & (ww < W)
            r = (bi * D + dd) * H * W + hh * W + ww
            a = tl.load(x + r[None, :] * Cin + c[:, None], mask=ok[None, :] & (c[:, None] < Cin), other=0.0)
            gv = tl.load(g + m[:, None] * Cout + n[None, :], mask=(m[:, None] < M) & (n[None, :] < Cout), other=0.0).to(tl.float32)
            acc += tl.dot(a, gv.to(tl.float16))
        tl.atomic_add(dw + (t * Cin + c)[:, None] * Cout + n[None, :], acc, mask=(c[:, None] < Cin) & (n[None, :] < Cout))


def _conv_rows(x2, wm, B, D, H, W, Cin, Cout):
    # x2: [B*D*H*W, Cin] fp16 rows, wm: [27*Cin, Cout] fp16 -> [B*D*H*W, Cout] fp16
    y = torch.empty(B * D * H * W, Cout, device=x2.device, dtype=torch.float16)
    BK = max(16, min(64, triton.next_power_of_2(Cin)))
    grid = lambda META: (triton.cdiv(B * D * H * W, META["BM"]), triton.cdiv(Cout, META["BN"]))
    with _launch_lock, torch.cuda.device(x2.device):
        try:
            _conv3d_k3_kernel[grid](x2, wm, y, B, D, H, W, Cin, Cout, BK=BK)
        finally:
            # The autotuner keeps the arguments, and so the tensors, when a launch raises.
            _conv3d_k3_kernel.nargs = None
    return y


def _wgrad_rows(x2, g2, B, D, H, W, Cin, Cout):
    # returns dW in Conv3d layout [Cout, Cin, 3, 3, 3], fp32
    M = B * D * H * W
    dw = torch.zeros(27 * Cin, Cout, device=x2.device, dtype=torch.float32)
    splits = max(1, min(256, M // 8192))
    MCH = triton.cdiv(triton.cdiv(M, splits), 128) * 128
    grid = lambda META: (27, triton.cdiv(Cin, META["BC"]), triton.cdiv(Cout, META["BN"]) * triton.cdiv(M, MCH))
    with _launch_lock, torch.cuda.device(x2.device):
        _conv3d_k3_wgrad_kernel[grid](x2, g2, dw, B, D, H, W, Cin, Cout, MCH)
    return dw.view(3, 3, 3, Cin, Cout).permute(4, 3, 0, 1, 2).contiguous()


class _Conv3dK3Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        B, Cin, D, H, W = x.shape
        Cout = weight.shape[0]
        # A no-op for fp16 channels_last_3d input.
        x2 = x.half().permute(0, 2, 3, 4, 1).contiguous().view(-1, Cin)
        wm = weight.detach().float().permute(2, 3, 4, 1, 0).reshape(-1, Cout)
        y2 = _conv_rows(x2, wm.half().contiguous(), B, D, H, W, Cin, Cout)
        ctx.save_for_backward(x2, weight)
        ctx.shape = (B, D, H, W, Cin, Cout)
        return y2.view(B, D, H, W, Cout).permute(0, 4, 1, 2, 3)

    @staticmethod
    def backward(ctx, grad_output):
        x2, weight = ctx.saved_tensors
        B, D, H, W, Cin, Cout = ctx.shape
        g2 = grad_output.half().permute(0, 2, 3, 4, 1).contiguous().view(-1, Cout)
        # The input gradient is the forward convolution with the kernel
        # flipped in space and the channel axes swapped.
        wt = weight.detach().float().flip(2, 3, 4).permute(2, 3, 4, 0, 1).reshape(-1, Cin)
        dx2 = _conv_rows(g2, wt.half().contiguous(), B, D, H, W, Cout, Cin)
        dw = _wgrad_rows(x2, g2, B, D, H, W, Cin, Cout).to(weight.dtype)
        return dx2.view(B, D, H, W, Cin).permute(0, 4, 1, 2, 3), dw


class TritonConv3d(nn.Conv3d):
    """``nn.Conv3d`` that runs the fp16 16->16 3x3x3 case through Triton on ROCm.

    Parameters and state-dict keys are those of ``nn.Conv3d``, so checkpoints
    load with either backend.
    """

    def _uses_triton(self, input: torch.Tensor) -> bool:
        if triton is None or not torch.version.hip or not input.is_cuda:
            return False
        fp16 = input.dtype == torch.float16 or (
            torch.is_autocast_enabled("cuda")
            and torch.get_autocast_dtype("cuda") == torch.float16
        )
        return (
            fp16
            and self.in_channels == SUPPORTED_CHANNELS
            and self.out_channels == SUPPORTED_CHANNELS
            and self.kernel_size == (3, 3, 3)
            and self.stride == (1, 1, 1)
            and self.padding == (1, 1, 1)
            and self.dilation == (1, 1, 1)
            and self.groups == 1
            and self.bias is None
            and self.padding_mode == "zeros"
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not self._uses_triton(input) or _device_works.get(input.device) is False:
            return super().forward(input)
        if input.device in _device_works:
            return _Conv3dK3Function.apply(input, self.weight)
        try:
            output = _Conv3dK3Function.apply(input, self.weight)
        except RuntimeError as exc:
            # e.g. a Triton build that cannot compile for this GPU architecture
            warnings.warn(f"Triton stem convolution failed on {input.device}, using torch: {exc}")
            _device_works[input.device] = False
            return super().forward(input)
        _device_works[input.device] = True
        return output
