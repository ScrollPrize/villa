from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import flow_grad_smoothing
import flow_triton


def sample_field(normalised_zyx, field_for_grid_sample):
    # normalised_zyx :: *, zyx in [0, 1]; field_for_grid_sample :: zyx, z, y, x
    orig_shape = normalised_zyx.shape
    zyx = (normalised_zyx * 2. - 1.).view(1, -1, 1, 1, 3)
    field_samples = F.grid_sample(
        input=field_for_grid_sample[None],
        grid=zyx.flip(-1),
        align_corners=True,
        mode='bilinear',
        padding_mode='border',
    )  # 1, zyx, n, 1, 1
    return field_samples.squeeze(0).squeeze(-2).squeeze(-1).T.view(*orig_shape[:-1], 3)  # *, zyx


_BSPLINE_CONSTS_CACHE = {}


def _bspline_consts(field, pts_dtype):
    # Per-(shape, device, dtype) constants for sample_field_bspline. The
    # sampler runs for every RK4 stage of every transform call; rebuilding
    # these tiny tensors each call would cost a pageable host->device copy
    # plus a stream sync per tensor.
    key = (tuple(field.shape[1:]), field.device, pts_dtype)
    consts = _BSPLINE_CONSTS_CACHE.get(key)
    if consts is None:
        shape_m1 = torch.tensor(
            [s - 1 for s in field.shape[1:]], device=field.device, dtype=pts_dtype)
        consts = (shape_m1, torch.zeros_like(shape_m1))
        _BSPLINE_CONSTS_CACHE[key] = consts
    return consts


def sample_field_bspline(field, normalised_zyx):
    # field :: 3, Z, Y, X -- control points of a tricubic uniform B-spline,
    #   placed at the same align_corners=True lattice positions sample_field
    #   uses, with edge control points replicated outside the lattice
    #   (matching padding_mode='border'). Every dimension must be >= 2.
    # normalised_zyx :: *, 3 in [0, 1]
    #
    # Uses the two-fetches-per-axis decomposition (Sigg & Hadwiger 2005,
    # Ruijters et al. 2008): the four cubic B-spline weights along one axis
    # collapse into two linear interpolations at offset positions, so the
    # 4x4x4 stencil is evaluated exactly as 8 trilinear fetches. Both offset
    # positions along each axis stay inside a single lattice cell (or in the
    # replicated border region), so each trilinear fetch reproduces its pair
    # of stencil taps exactly. All 8 fetches run as ONE sample_field call on
    # an 8x-widened point batch: one grid_sample kernel per lattice, and --
    # since grid_sample's backward materialises a dense field-sized gradient
    # per call -- one such transient instead of eight.
    orig_shape = normalised_zyx.shape
    pts = normalised_zyx.reshape(-1, 3)
    shape_m1, zeros3 = _bspline_consts(field, pts.dtype)
    x = (pts * shape_m1).clamp(min=zeros3, max=shape_m1)
    lo = x.floor()
    f = x - lo
    f2 = f * f
    f3 = f2 * f
    w0 = (1. - f) ** 3 / 6.
    w1 = (3. * f3 - 6. * f2 + 4.) / 6.
    w2 = (-3. * f3 + 3. * f2 + 3. * f + 1.) / 6.
    w3 = f3 / 6.
    # Per axis: weights w0..w3 over taps lo-1..lo+2 become weights (g0, g1) on
    # linear fetches at h0 in [lo-1, lo] and h1 in [lo+1, lo+2]. g0 and g1 are
    # bounded in [1/6, 5/6], so the divisions are safe.
    g0 = w0 + w1
    g1 = w2 + w3
    h0 = (lo - 1. + w1 / g0) / shape_m1
    h1 = (lo + 1. + w3 / g1) / shape_m1
    corner_is_hi = _corner_bits(pts.device).bool()[:, None, :]  # 8, 1, 3
    pos = torch.where(corner_is_hi, h1[None], h0[None])  # 8, n, 3
    weight = torch.where(corner_is_hi, g1[None], g0[None]).prod(-1)  # 8, n
    fetched = sample_field(pos, field)  # 8, n, 3
    return (fetched * weight.unsqueeze(-1)).sum(0).view(*orig_shape[:-1], 3)


def _bspline_w4(f):
    # Uniform cubic B-spline weights over taps lo-1 .. lo+2 for fractional
    # offsets f, stacked on a new leading axis :: 4, *f.shape. Same basis as
    # sample_field_bspline's w0..w3 (which keeps them unstacked to feed the
    # two-fetch collapse).
    f2 = f * f
    f3 = f2 * f
    return torch.stack([
        (1. - f) ** 3 / 6.,
        (3. * f3 - 6. * f2 + 4.) / 6.,
        (-3. * f3 + 3. * f2 + 3. * f + 1.) / 6.,
        f3 / 6.,
    ])


_CORNER_BITS_CACHE = {}


def _corner_bits(device):
    t = _CORNER_BITS_CACHE.get(device)
    if t is None:
        t = torch.tensor(
            [[dz, dy, dx] for dz in (0, 1) for dy in (0, 1) for dx in (0, 1)],
            device=device,
            dtype=torch.int64,
        )
        _CORNER_BITS_CACHE[device] = t
    return t


_FIELD_CONSTS_CACHE = {}


def _field_consts(field, pts_dtype):
    # Per-(shape, device, dtype) constant tensors used by the sampler backward.
    # That backward runs for every RK4 stage of every transform call (hundreds
    # of times per training step), and rebuilding these tiny tensors each call
    # costs a pageable host->device copy plus a stream sync per tensor. The
    # cached values are identical constants, so results are unchanged.
    key = (tuple(field.shape), field.device, pts_dtype)
    consts = _FIELD_CONSTS_CACHE.get(key)
    if consts is None:
        device = field.device
        shape = torch.tensor(field.shape[1:], device=device, dtype=pts_dtype)
        Y, X = field.shape[2], field.shape[3]
        corners = _corner_bits(device)
        strides_t = torch.tensor((Y * X, X, 1), device=device, dtype=torch.int64)
        consts = {
            'shape': shape,
            'zeros': torch.zeros_like(shape),
            'shape_m1': shape - 1,
            'shape_m2': shape - 2,
            # (corners * strides).sum(-1) is constant integer math; precompute.
            'corner_offsets': (corners * strides_t).sum(-1),
            'signs': (2 * corners - 1).to(pts_dtype),
        }
        _FIELD_CONSTS_CACHE[key] = consts
    return consts


@torch.jit.script
def _sparse_backward_impl(
    grad_out,
    pts,
    low_field,
    high_field,
    high_scale: float,
    acc: Optional[torch.Tensor],
    shape_m1,
    zeros3,
    shape_m2,
    corner_offsets,
    signs,
    corners,
    Y: int,
    X: int,
):
    # Body of _SparseAccumTrilinearSample.backward. This runs for every RK4
    # stage of every transform call (hundreds of times per training step); the
    # ~25 aten calls are unchanged (same kernels, same order -> same values),
    # but TorchScript dispatches them from C++, removing most of the Python
    # per-op overhead that otherwise makes the backward CPU-bound.
    coord_raw = pts * shape_m1
    coord = coord_raw.clamp(min=zeros3, max=shape_m1)
    lo = torch.nan_to_num(coord, nan=0.0).floor().clamp(
        min=zeros3,
        max=shape_m2,
    ).to(torch.int64)
    frac = coord - lo.to(coord.dtype)
    base = (lo[:, 0] * Y + lo[:, 1]) * X + lo[:, 2]

    flat_low = low_field.reshape(3, -1)
    flat_high = high_field.reshape(3, -1)

    idx_all = base[:, None] + corner_offsets[None, :]

    f = torch.stack([1 - frac, frac], dim=-1)
    fz = f[:, 0, corners[:, 0]]
    fy = f[:, 1, corners[:, 1]]
    fx = f[:, 2, corners[:, 2]]
    w_all = fz * fy * fx

    vals = (grad_out[:, None, :] * w_all[..., None]).permute(2, 0, 1).reshape(3, -1)
    if acc is not None:
        acc_flat = acc.reshape(3, -1)
        acc_flat.index_add_(1, idx_all.reshape(-1), vals)

    flat_indices = idx_all.reshape(-1)
    gathered = (
        flat_low[:, flat_indices]
        + flat_high[:, flat_indices] * high_scale
    ).view(3, idx_all.shape[0], idx_all.shape[1])
    v_dot_g = (gathered * grad_out.T[:, :, None]).sum(0)
    grad_coord = torch.stack([
        (v_dot_g * signs[:, 0] * fy * fx).sum(1),
        (v_dot_g * signs[:, 1] * fz * fx).sum(1),
        (v_dot_g * signs[:, 2] * fz * fy).sum(1),
    ], dim=-1)
    unclipped = (coord_raw >= 0) & (coord_raw <= shape_m1)
    return grad_coord * unclipped.to(grad_coord.dtype) * shape_m1


class _SparseAccumTrilinearSample(torch.autograd.Function):
    # Trilinear sampling matching sample_field / grid_sample(align_corners=True,
    # padding_mode='border'), with sparse field-gradient accumulation into a
    # caller-owned dense buffer. This avoids materialising one full field-sized
    # gradient tensor per sampler call in the RK4 integration loop.

    @staticmethod
    def forward(ctx, pts, low_field, high_field, high_scale, acc):
        ctx.set_materialize_grads(False)
        # Keep the two full-resolution sources separate.  Materialising their
        # sum costs another complete Cartesian field (about 1.49 GiB with the
        # default full-scroll configuration), while sparse sampling is linear.
        out = (
            sample_field(pts, low_field)
            + sample_field(pts, high_field) * high_scale
        )
        ctx.save_for_backward(pts, low_field, high_field)
        ctx.high_scale = high_scale
        ctx.acc = acc
        return out

    @staticmethod
    def backward(ctx, grad_out):
        if grad_out is None:
            return None, None, None, None, None
        pts, low_field, high_field = ctx.saved_tensors
        consts = _field_consts(low_field, pts.dtype)
        if consts['signs'].dtype == grad_out.dtype:
            signs = consts['signs']
        else:
            signs = (2 * _corner_bits(pts.device) - 1).to(grad_out.dtype)
        grad_pts = _sparse_backward_impl(
            grad_out,
            pts,
            low_field,
            high_field,
            float(ctx.high_scale),
            ctx.acc,
            consts['shape_m1'],
            consts['zeros'],
            consts['shape_m2'],
            consts['corner_offsets'],
            signs,
            _corner_bits(pts.device),
            low_field.shape[2],
            low_field.shape[3],
        )
        return grad_pts, None, None, None, None


class _RK4SparseFlowIntegrate(torch.autograd.Function):
    # The whole time-invariant RK4 integration as a single autograd node.
    #
    # The eager sampler-based loop creates ~42 autograd nodes per transform
    # call (12 sampler Functions plus the stage arithmetic), and the engine's
    # per-node dispatch dominates step time. This Function runs the identical
    # forward arithmetic, saves the 12 stage points (the same tensors the
    # eager graph would keep alive via the sampler nodes' save_for_backward),
    # and hand-writes the adjoint sweep in backward.
    #
    # The backward reproduces the eager graph's exact floating-point operation
    # and gradient-accumulation order (verified bitwise in
    # tests/test_speedup_equivalence.py):
    #   k_bar4 = g*(h/6)
    #   k_bar3 = (g*(h/6))*2 + x_bar4*h
    #   k_bar2 = (g*(h/6))*2 + x_bar3*(h/2)
    #   k_bar1 =  g*(h/6)    + x_bar2*(h/2)
    #   g_prev = (((g + x_bar4) + x_bar3) + x_bar2) + x_bar1
    # where x_bar_i is the sampler backward at stage point x_i (which also
    # accumulates the field gradient into `acc`, in the same k4..k1 order the
    # autograd engine used).

    @staticmethod
    def forward(ctx, y0, low_field, high_field, high_scale, acc, h, n_steps):
        ctx.set_materialize_grads(False)
        stage_points = []
        y = y0
        for _ in range(n_steps):
            k1 = sample_field(y, low_field) + sample_field(y, high_field) * high_scale
            x2 = y + (h / 2) * k1
            k2 = sample_field(x2, low_field) + sample_field(x2, high_field) * high_scale
            x3 = y + (h / 2) * k2
            k3 = sample_field(x3, low_field) + sample_field(x3, high_field) * high_scale
            x4 = y + h * k3
            k4 = sample_field(x4, low_field) + sample_field(x4, high_field) * high_scale
            stage_points.extend([y, x2, x3, x4])
            y = y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        ctx.save_for_backward(low_field, high_field, *stage_points)
        ctx.high_scale = float(high_scale)
        ctx.acc = acc
        ctx.h = float(h)
        ctx.n_steps = int(n_steps)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        if grad_y is None:
            return None, None, None, None, None, None, None
        saved = ctx.saved_tensors
        low_field, high_field = saved[0], saved[1]
        stage_points = saved[2:]
        h = ctx.h
        scale = ctx.high_scale
        acc = ctx.acc
        consts = _field_consts(low_field, grad_y.dtype)
        corners = _corner_bits(grad_y.device)
        if consts['signs'].dtype == grad_y.dtype:
            signs = consts['signs']
        else:
            signs = (2 * corners - 1).to(grad_y.dtype)
        Y, X = low_field.shape[2], low_field.shape[3]

        def stage_grad(g, pts):
            return _sparse_backward_impl(
                g, pts, low_field, high_field, scale, acc,
                consts['shape_m1'], consts['zeros'], consts['shape_m2'],
                consts['corner_offsets'], signs, corners, Y, X,
            )

        for step in range(ctx.n_steps - 1, -1, -1):
            x1, x2, x3, x4 = stage_points[4 * step: 4 * step + 4]
            g6 = grad_y * (h / 6)
            xb4 = stage_grad(g6, x4)
            xb3 = stage_grad(g6 * 2 + xb4 * h, x3)
            xb2 = stage_grad(g6 * 2 + xb3 * (h / 2), x2)
            xb1 = stage_grad(g6 + xb2 * (h / 2), x1)
            grad_y = grad_y + xb4
            grad_y = grad_y + xb3
            grad_y = grad_y + xb2
            grad_y = grad_y + xb1
        return grad_y, None, None, None, None, None, None



def _low_res_width(sigma_hr_cells, low_res_sigma_hr_cells):
    # The low-resolution lattice's own along-sheet width when one is set,
    # else the shared physical width.
    if low_res_sigma_hr_cells is not None and float(low_res_sigma_hr_cells) > 0.0:
        return float(low_res_sigma_hr_cells)
    return float(sigma_hr_cells)



def _rk4_sampler_loop(y, sampler, h, n_steps):
    # Plain RK4 over one stationary velocity sampler; the eager reference for
    # every fused path and the no-grad evaluation path (no adjoint sweep will
    # run, so nothing is retained).
    for _ in range(n_steps):
        k1 = sampler(y)
        k2 = sampler(y + (h / 2) * k1)
        k3 = sampler(y + (h / 2) * k2)
        k4 = sampler(y + h * k3)
        y = y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return y


def _slab_order(num_stages, reverse):
    return range(num_stages - 1, -1, -1) if reverse else range(num_stages)


class CartesianFlowField(nn.Module):
    """Piecewise-stationary velocity field on two Cartesian lattices.

    ``flows[0]`` is the low-resolution lattice and ``flows[1]`` the
    high-resolution one; both are ``[num_stages, 3, Z, Y, X]``. Slab ``t`` is
    a stationary velocity field integrated for unit time, and the integrator
    applies the slabs in order (in reverse order, each backwards, for the
    inverse), so ``num_stages`` slabs are the sequential composition of
    ``num_stages`` stationary flows. ``num_stages == 1`` is the original
    single-field model with identical parameters and state_dict keys.
    """

    def __init__(self, resolution, spatial_scale_factor=6, num_stages=1, direct_lr=False):
        super().__init__()
        self.num_stages = int(num_stages)
        assert self.num_stages >= 1
        self.direct_lr = direct_lr
        self.spatial_scale_factor = int(spatial_scale_factor)
        resolution = [int(s) for s in resolution]
        self.flows = nn.ParameterList([
            nn.Parameter(torch.zeros([self.num_stages, 3, *shape]))
            for shape in [
                [s // spatial_scale_factor for s in resolution],
                resolution,
            ]
        ])
        self._pending_lr_upsampled = None
        self._field_grad_acc = None
        self._lr_grad_acc = None
        self._pending_direct = False

    def _prepare_upsampled_fields(self):
        # Shared state for get_sampler / get_integrator in upsampled-LR mode.
        # Returns (low_field, high_field, acc), each [num_stages, 3, Z, Y, X];
        # acc is None outside training (no grad, or frozen fields).
        lr_flow, hr_flow = self.flows[0], self.flows[1]
        hr_shape = tuple(hr_flow.shape[2:])
        # HR flow is already at the target resolution, so only LR interpolates.
        lr_upsampled = F.interpolate(lr_flow, size=hr_shape, mode='trilinear')
        training_field = torch.is_grad_enabled() and (lr_flow.requires_grad or hr_flow.requires_grad)
        if not training_field:
            return lr_upsampled, hr_flow.detach(), None

        # Sampling keeps the detached LR-upsampled and HR fields separate and
        # accumulates dL/d(their effective sum) in one caller-owned buffer.
        # Keep only the LR interpolation graph; after all point losses have
        # backpropagated, its gradient is propagated normally while the same
        # full-resolution accumulator becomes the HR parameter gradient.
        low_field = lr_upsampled.detach()
        high_field = hr_flow.detach()
        if self._field_grad_acc is None or self._field_grad_acc.shape != high_field.shape:
            self._field_grad_acc = torch.zeros_like(high_field)
        else:
            self._field_grad_acc.zero_()
        self._pending_lr_upsampled = lr_upsampled
        return low_field, high_field, self._field_grad_acc

    def _get_direct_integrator(self):
        # Direct-LR mode: sample the LR lattice at query points instead of
        # upsampling it to HR every step (FIT_SPIRAL_DIRECT_LR=1). Field
        # gradients accumulate into per-lattice buffers.
        lr_flow, hr_flow = self.flows[0], self.flows[1]
        low, high = lr_flow, hr_flow
        training_field = torch.is_grad_enabled() and (
            lr_flow.requires_grad or hr_flow.requires_grad)
        acc_lo = acc_hi = None
        if training_field:
            low, high = low.detach(), high.detach()
            if self._lr_grad_acc is None or self._lr_grad_acc.shape != low.shape:
                self._lr_grad_acc = torch.zeros_like(low)
            else:
                self._lr_grad_acc.zero_()
            if self._field_grad_acc is None or self._field_grad_acc.shape != high.shape:
                self._field_grad_acc = torch.zeros_like(high)
            else:
                self._field_grad_acc.zero_()
            acc_lo, acc_hi = self._lr_grad_acc, self._field_grad_acc
            # Only one pending-gradient record may be armed per iteration; a
            # stale upsampled-mode record would fire on the shared
            # accumulator after the direct branch consumed it.
            assert self._pending_lr_upsampled is None
            self._pending_direct = True
        low_c, high_c = low.contiguous(), high.contiguous()

        def integrate(y_flat, h, n_steps, reverse=False):
            return flow_triton.rk4_direct_integrate(
                y_flat, low_c, high_c, 1.0, 1.0, acc_lo, acc_hi, h, n_steps,
                reverse)

        return integrate

    def get_integrator(self):
        """Return ``integrate(y_flat, h, n_steps, reverse=False) -> y_flat``.

        Runs every slab's RK4 integration (``n_steps`` steps of size ``h``
        each) in order, or in reverse order with ``reverse``. On CUDA the
        whole walk is one autograd node (see flow_triton); the eager fallback
        composes one _RK4SparseFlowIntegrate node per slab.
        """
        if (flow_triton.direct_lr_enabled(self.direct_lr)
                and flow_triton.rk4_triton_available(self.flows[0], self.flows[1])):
            return self._get_direct_integrator()
        low_field, high_field, acc = self._prepare_upsampled_fields()

        if flow_triton.rk4_triton_available(low_field, high_field, acc):
            low_c, high_c = low_field.contiguous(), high_field.contiguous()

            def integrate(y_flat, h, n_steps, reverse=False):
                return flow_triton.rk4_integrate(
                    y_flat, low_c, high_c, 1.0, acc, h, n_steps, reverse,
                )

            return integrate

        num_stages = self.num_stages

        def integrate(y_flat, h, n_steps, reverse=False):
            y = y_flat
            for slab in _slab_order(num_stages, reverse):
                low, high = low_field[slab], high_field[slab]
                if not (torch.is_grad_enabled() and y.requires_grad):
                    # No adjoint sweep will run; the plain sampler loop avoids
                    # the Function's transient retention of all 4*n_steps
                    # stage points (matters for large no-grad evals like
                    # previews/satisfaction).
                    y = _rk4_sampler_loop(
                        y, lambda p: sample_field(p, low) + sample_field(p, high),
                        h, n_steps)
                else:
                    y = _RK4SparseFlowIntegrate.apply(
                        y, low, high, 1.0,
                        None if acc is None else acc[slab], h, n_steps,
                    )
            return y

        return integrate

    def get_sampler(self, stage=0):
        # Returns a callable mapping normalised zyx points in [0, 1] to the
        # velocity of slab `stage`. Materialises the slab as a [3, Z, Y, X]
        # cartesian tensor of zyx vector components once and reuses it across
        # the (e.g. RK4) integrator's many sample calls. In training this
        # arms the field-gradient accumulator like get_integrator does.
        stage = int(stage)
        low_field, high_field, acc = self._prepare_upsampled_fields()
        low, high = low_field[stage], high_field[stage]
        if acc is None:
            return lambda y: sample_field(y, low) + sample_field(y, high)
        acc_slab = acc[stage]

        def sample(normalised_zyx):
            flat = normalised_zyx.reshape(-1, 3)
            return _SparseAccumTrilinearSample.apply(
                flat,
                low,
                high,
                1.0,
                acc_slab,
            ).view(*normalised_zyx.shape[:-1], 3)

        return sample

    def apply_accumulated_field_grad(self):
        if self._pending_direct:
            self._pending_direct = False
            for param, grad in (
                (self.flows[0], self._lr_grad_acc),
                (self.flows[1], self._field_grad_acc),
            ):
                # Reuse the accumulator storage as the parameter gradient.
                if param.grad is None:
                    param.grad = grad
                else:
                    param.grad.add_(grad)
            return

        if self._pending_lr_upsampled is None:
            return

        # dL/dLR is the interpolation backward of dL/dfield.
        self._pending_lr_upsampled.backward(gradient=self._field_grad_acc)
        self._pending_lr_upsampled = None

        # Reuse the accumulator storage as the HR parameter's gradient instead
        # of materialising another [T,3,Z,Y,X] tensor.
        hr_param = self.flows[1]
        hr_grad = self._field_grad_acc
        if hr_param.grad is None:
            hr_param.grad = hr_grad
        else:
            hr_param.grad.add_(hr_grad)

    def smooth_grad_(self, sigma_hr_cells, across_sigma_hr_cells=0.0,
                     low_res_sigma_hr_cells=0.0):
        """Gaussian-smooth both lattices' gradients in place (isotropic).

        ``sigma_hr_cells`` is the width in nominal high-resolution cells;
        coarse cells are ``spatial_scale_factor`` times larger, so they see
        the same flow-frame width in their own cell units unless
        ``low_res_sigma_hr_cells`` (also in high-resolution cells, 0 = same
        as ``sigma_hr_cells``) gives it a width of its own.
        ``across_sigma_hr_cells`` is accepted for signature parity with the
        cylindrical lattice and ignored: a Cartesian lattice has no
        across-winding axis to smooth differently.
        """
        for level, flow in enumerate(self.flows):
            if flow.grad is None:
                continue
            scale = self.spatial_scale_factor if level == 0 else 1
            along = _low_res_width(sigma_hr_cells, low_res_sigma_hr_cells) if level == 0 else sigma_hr_cells
            flow_grad_smoothing.smooth_cartesian_(flow.grad, float(along) / scale)


def _adopt_accumulated_grads_(pairs):
    # Make each fused-kernel gradient accumulator its parameter's gradient,
    # reusing the accumulator storage. After the first iteration param.grad
    # usually IS the accumulator (same storage), so only a foreign gradient
    # buffer is added into.
    for param, grad in pairs:
        if param.grad is None:
            param.grad = grad
        elif (param.grad.untyped_storage().data_ptr()
              != grad.untyped_storage().data_ptr()):
            param.grad.add_(grad)


class _DirectSampledFlowField(nn.Module):

    # Base for flow fields whose parameter lattices are sampled directly at
    # query points (no upsampled full-resolution intermediate): provides the
    # streamed-backward support and the eager slab-by-slab integrator such
    # fields share. The samplers / integrators are cached by the
    # diffeomorphism and shared across every loss family in an iteration, so
    # subclasses pass their per-iteration field tensors through
    # _maybe_cut_field_leaves to cut the field graphs at detached leaves;
    # each family's backward then owns its whole graph (no retain_graph), and
    # the accumulated leaf gradients flow to the parameters when the training
    # loop calls apply_accumulated_field_grad.
    #
    # Subclasses provide num_stages, _iteration_fields() -> (lr_field,
    # hr_field) (each [num_stages, ...], leaf-cut in training) and
    # _make_sampler(lr_field, hr_field, stage) -> velocity sampler for one
    # slab.

    def __init__(self):
        super().__init__()
        # (field graph output, detached leaf) pairs armed by
        # _maybe_cut_field_leaves in training and consumed by
        # apply_accumulated_field_grad.
        self._pending_field_graphs = None

    def _maybe_cut_field_leaves(self, fields):
        if not (torch.is_grad_enabled() and any(f.requires_grad for f in fields)):
            return tuple(fields)
        # Overwrites any previous pending record, matching CartesianFlowField's
        # one-armed-record-per-iteration discipline.
        leaves = [field.detach().requires_grad_(True) for field in fields]
        self._pending_field_graphs = list(zip(fields, leaves))
        return tuple(leaves)

    def get_sampler(self, stage=0):
        # Returns a callable mapping normalised zyx points in [0, 1] to the
        # velocity of slab `stage`, sampling the lattices directly at each
        # query point. The closure captures this iteration's (leaf-cut)
        # lattices so those one-time costs amortise across the integrator's
        # sample calls.
        lr_field, hr_field = self._iteration_fields()
        return self._make_sampler(lr_field, hr_field, int(stage))

    def _eager_integrator(self):
        # Slab-by-slab RK4 over the eager samplers: the reference for every
        # fused path and the CPU path. The samplers are built on first use so
        # the leaf-cutting record is armed only when this path actually runs.
        num_stages = self.num_stages
        samplers = None

        def integrate(y_flat, h, n_steps, reverse=False):
            nonlocal samplers
            if samplers is None:
                lr_field, hr_field = self._iteration_fields()
                samplers = [
                    self._make_sampler(lr_field, hr_field, stage)
                    for stage in range(num_stages)]
            y = y_flat
            for slab in _slab_order(num_stages, reverse):
                y = _rk4_sampler_loop(y, samplers[slab], h, n_steps)
            return y

        return integrate

    def apply_accumulated_field_grad(self):
        pending, self._pending_field_graphs = self._pending_field_graphs, None
        if not pending:
            return
        outputs = [output for output, leaf in pending if leaf.grad is not None]
        if outputs:
            torch.autograd.backward(
                outputs,
                [leaf.grad for _, leaf in pending if leaf.grad is not None],
            )


def _cylindrical_query_coords(normalised_zyx, nz, nr):
    # Shared query-coordinate mapping for the cylindrical lattice samplers
    # (trilinear and b-spline). normalised_zyx :: *, 3 in [0, 1] (cartesian
    # box-relative). Returns flat (n,) tensors: continuous z and r lattice
    # indices (align_corners=True style, z clamped to the lattice and r to the
    # inscribed disk) plus the query angle, both wrapped to [0, 2pi) for
    # lattice indexing and raw in (-pi, pi] for the local-basis rotation.
    pts = normalised_zyx.reshape(-1, 3) * 2. - 1.  # n, 3 in [-1, 1] cartesian
    z_n, y_n, x_n = pts[:, 0], pts[:, 1], pts[:, 2]
    # The cylindrical basis is singular exactly on the axis: sqrt(0) and atan2(0, 0)
    # have finite forward values but undefined gradients. Use a fixed +x basis there.
    axis_eps = torch.finfo(pts.dtype).eps
    on_axis = (y_n.abs() <= axis_eps) & (x_n.abs() <= axis_eps)
    safe_y_n = torch.where(on_axis, torch.zeros_like(y_n), y_n)
    safe_x_n = torch.where(on_axis, torch.ones_like(x_n), x_n)
    rr = torch.sqrt(safe_y_n ** 2 + safe_x_n ** 2).clamp(max=1.)  # inscribed-disk clamp
    rr = torch.where(on_axis, torch.zeros_like(rr), rr)
    phi = torch.atan2(safe_y_n, safe_x_n)  # in (-pi, pi]

    z_cont = ((z_n + 1.) * 0.5 * (nz - 1)).clamp(0., float(nz - 1))
    r_cont = rr * (nr - 1)
    phi_in_2pi = phi % (2. * np.pi)  # in [0, 2pi)
    return z_cont, r_cont, phi_in_2pi, phi


def _cylindrical_local_to_cartesian(sampled, phi, orig_shape):
    # sampled :: 3, n in the local (z, radial, tangential) basis at query
    # angle phi :: n. phi = atan2(y, x), so outward-radial in (y, x) is
    # (sin(phi), cos(phi)) and tangential (d/dphi unit) is (cos(phi),
    # -sin(phi)). Rotate local (r, phi) components into (y, x).
    z_c, r_c, p_c = sampled[0], sampled[1], sampled[2]
    sin_phi, cos_phi = torch.sin(phi), torch.cos(phi)
    y_c = r_c * sin_phi + p_c * cos_phi
    x_c = r_c * cos_phi - p_c * sin_phi
    return torch.stack([z_c, y_c, x_c], dim=-1).view(*orig_shape)


class CylindricalFlowField(_DirectSampledFlowField):

    # Flow field with parameters on a cylindrical lattice (z, r, phi). The cylinder axis lies
    # along z at the centre of the y, x box; the lattice spans z=[0,Z) and the inscribed disk in
    # y, x (radius<=1 in normalised cartesian; corners outside the disk are clamped on r). Stored
    # per-cell vectors are in the local (z, radial, tangential) basis: component 1 points outward
    # radially, component 2 in the direction of increasing phi (right-hand rule about +z). The
    # integrator samples the lattice directly at cartesian query points and rotates the (r, phi)
    # components into (y, x) on the fly using the local basis at each query point.
    #
    # Rings have *varying* numbers of angular cells: ring r holds num_phi[r] = max(1, round(2*pi*r))
    # cells (= circumference / lattice radial spacing), so inner rings are coarse and outer rings
    # fine. All rings are packed end-to-end along the last (phi) axis of the parameter tensor,
    # which is therefore "ragged"; sampling does explicit per-query gathers (one per surrounding
    # corner of the (z, r, phi) trilinear stencil).
    #
    # Like CartesianFlowField, both lattices carry a leading slab axis of length num_stages: slab
    # t is a stationary velocity field integrated for unit time, and the slabs compose in order.
    #
    # Note: near r=0 the cylindrical basis is degenerate; ring 0 holds a single cell that is
    # pinned to zero.

    # Interpolant the fused kernel evaluates (flow_triton CUBIC switch);
    # BSplineCylindricalFlowField flips it.
    _cubic = False

    def __init__(self, resolution, spatial_scale_factor=6, num_stages=1, direct_lr=False):
        # resolution is interpreted as the equivalent cartesian (Z, Y, X) voxel shape; the
        # cylindrical lattice sizes are derived from it. direct_lr is accepted for
        # constructor parity with CartesianFlowField and ignored: the ragged
        # cylindrical lattice is always sampled directly (never upsampled).
        super().__init__()
        self.num_stages = int(num_stages)
        assert self.num_stages >= 1
        self.spatial_scale_factor = int(spatial_scale_factor)
        Z, Y, X = (int(s) for s in resolution)

        nz_hr = Z
        nr_hr = max(2, min(Y, X) // 2)
        nz_lr = max(2, nz_hr // spatial_scale_factor)
        nr_lr = max(2, nr_hr // spatial_scale_factor)

        # The lr lattice has spatial_scale_factor-wider rings, so its ring r covers the same
        # circumference as the hr ring r*spatial_scale_factor; the factors cancel in
        # "cells per (sub-)ring unit", so the same 2*pi*r formula applies to both lattices.
        def compute_num_phi(nr):
            return torch.tensor(
                [1 if r == 0 else max(1, int(round(2 * np.pi * r))) for r in range(nr)],
                dtype=torch.long,
            )

        lr_num_phi = compute_num_phi(nr_lr)
        hr_num_phi = compute_num_phi(nr_hr)
        lr_offsets = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(lr_num_phi, dim=0)])
        hr_offsets = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(hr_num_phi, dim=0)])
        self.register_buffer('_lr_num_phi', lr_num_phi)
        self.register_buffer('_hr_num_phi', hr_num_phi)
        self.register_buffer('_lr_offsets', lr_offsets)
        self.register_buffer('_hr_offsets', hr_offsets)

        self.flows = nn.ParameterList([
            nn.Parameter(torch.zeros([self.num_stages, 3, nz_lr, int(lr_offsets[-1])])),
            nn.Parameter(torch.zeros([self.num_stages, 3, nz_hr, int(hr_offsets[-1])])),
        ])
        # Fused-backward gradient buffers in the parameter layout; they
        # become the parameters' gradients (apply_accumulated_field_grad).
        self._lr_grad_acc = None
        self._hr_grad_acc = None
        # The cubic kernels scatter into padded channels-last buffers
        # ([num_stages, nz, total_phi, 4], flow_triton's cubic lattice
        # section) that are permuted into the above at apply time.
        self._lr_scatter_acc = None
        self._hr_scatter_acc = None
        self._pending_fused = False

    @staticmethod
    def _sample_lattice(field, ring_num_phi, ring_offsets, normalised_zyx):
        # field :: 3, nz, total_phi -- rings packed end-to-end along the last axis
        # ring_num_phi :: nr (long) -- per-ring phi cell counts
        # ring_offsets :: nr+1 (long) -- cumulative ring start offsets in the flat phi axis
        # normalised_zyx :: *, 3 in [0, 1] (cartesian box-relative)
        # Returns: *, 3 with components in cartesian (z, y, x).
        nz = field.shape[1]
        nr = ring_num_phi.shape[0]
        orig_shape = normalised_zyx.shape
        z_cont, r_cont, phi_in_2pi, phi = _cylindrical_query_coords(
            normalised_zyx, nz, nr)

        z_lo = torch.floor(z_cont).clamp(max=nz - 2).long()
        z_hi = z_lo + 1
        frac_z = (z_cont - z_lo.to(z_cont.dtype)).unsqueeze(0)  # 1, n

        r_lo = torch.floor(r_cont).clamp(max=nr - 2).long()
        r_hi = r_lo + 1
        frac_r = (r_cont - r_lo.to(r_cont.dtype)).unsqueeze(0)  # 1, n

        def sample_at_ring(r_idx):
            # r_idx :: n (long). Returns 3, n -- bilinear in (z, phi) at this integer ring.
            num_phi_r = ring_num_phi[r_idx]
            offset_r = ring_offsets[r_idx]
            phi_cont = phi_in_2pi * (num_phi_r.to(phi_in_2pi.dtype) / (2. * np.pi))
            phi_lo_floor = torch.floor(phi_cont)
            phi_lo = phi_lo_floor.long() % num_phi_r
            phi_hi = (phi_lo + 1) % num_phi_r  # cyclic wrap
            frac_phi = (phi_cont - phi_lo_floor).unsqueeze(0)
            flat_lo = offset_r + phi_lo
            flat_hi = offset_r + phi_hi
            v00 = field[:, z_lo, flat_lo]
            v01 = field[:, z_lo, flat_hi]
            v10 = field[:, z_hi, flat_lo]
            v11 = field[:, z_hi, flat_hi]
            v0 = v00 + (v01 - v00) * frac_phi
            v1 = v10 + (v11 - v10) * frac_phi
            return v0 + (v1 - v0) * frac_z

        v_rlo = sample_at_ring(r_lo)
        v_rhi = sample_at_ring(r_hi)
        sampled = v_rlo + (v_rhi - v_rlo) * frac_r  # 3, n in (z, r, phi) local components
        return _cylindrical_local_to_cartesian(sampled, phi, orig_shape)

    def _iteration_fields(self):
        # The axis-pinned LR & HR lattices, [num_stages, 3, nz, total_phi],
        # shared by every eager sampler built from one get_sampler /
        # get_integrator call. Pins the r=0 ring (axis singularity) to zero
        # by replacing its flat-phi slice with a constant zero, so no gradient
        # flows to those parameters; they stay zero indefinitely.
        assert not self._pending_fused
        lr_field = self.flows[0]
        hr_field = self.flows[1]
        n0_lr = int(self._lr_num_phi[0])
        n0_hr = int(self._hr_num_phi[0])
        lr_field = torch.cat([torch.zeros_like(lr_field[..., :n0_lr]), lr_field[..., n0_lr:]], dim=-1)
        hr_field = torch.cat([torch.zeros_like(hr_field[..., :n0_hr]), hr_field[..., n0_hr:]], dim=-1)
        return self._maybe_cut_field_leaves((lr_field, hr_field))

    def _make_sampler(self, lr_field, hr_field, stage):
        # Velocity sampler for one slab of the pinned fields: normalised zyx
        # points in [0, 1] -> cartesian (z, y, x) velocity.
        lr_slab = lr_field[stage]
        hr_slab = hr_field[stage]
        sample_lattice = self._sample_lattice
        lr_num_phi = self._lr_num_phi
        lr_offsets = self._lr_offsets
        hr_num_phi = self._hr_num_phi
        hr_offsets = self._hr_offsets

        def sample(normalised_zyx):
            return (
                sample_lattice(lr_slab, lr_num_phi, lr_offsets, normalised_zyx)
                + sample_lattice(hr_slab, hr_num_phi, hr_offsets, normalised_zyx)
            )
        return sample

    def get_integrator(self):
        """Return a lazy slab-by-slab RK4 integrator with a fused CUDA path."""
        fused_lattices = flow_triton.rk4_triton_available(
            self.flows[0], self.flows[1])
        mode = None
        eager_integrate = self._eager_integrator()
        fused_fields = fused_accs = None

        def integrate(y_flat, h, n_steps, reverse=False):
            nonlocal mode, fused_fields, fused_accs
            can_fuse = (
                fused_lattices
                and flow_triton.rk4_triton_available(y_flat)
                and y_flat.device == self.flows[0].device
                and self._lr_num_phi.is_cuda and self._hr_num_phi.is_cuda
                and self._lr_offsets.is_cuda and self._hr_offsets.is_cuda)
            if mode is None:
                mode = 'fused' if can_fuse else 'eager'
            if mode == 'eager':
                return eager_integrate(y_flat, h, n_steps, reverse)
            if not can_fuse:
                raise RuntimeError(
                    'a cylindrical integrator cannot mix fused and eager tensor types')

            if fused_fields is None:
                low, high = self.flows[0], self.flows[1]
                training_field = torch.is_grad_enabled() and (
                    self.flows[0].requires_grad or self.flows[1].requires_grad)
                acc_low = acc_high = None
                if training_field:
                    assert self._pending_field_graphs is None
                    acc_low, acc_high = self._fused_accumulators(low, high)
                    self._pending_fused = True
                if self._cubic:
                    # Channels-last copies for the cubic kernels' read path,
                    # one per lattice per get_integrator call (amortised
                    # across the integrator's calls, like the accumulators).
                    fused_fields = (
                        low.permute(0, 2, 3, 1).contiguous(),
                        high.permute(0, 2, 3, 1).contiguous())
                else:
                    fused_fields = (low.contiguous(), high.contiguous())
                fused_accs = (acc_low, acc_high)

            low, high = fused_fields
            acc_low, acc_high = fused_accs
            return flow_triton.rk4_cylindrical_integrate(
                y_flat, low, self._lr_num_phi, self._lr_offsets,
                high, self._hr_num_phi, self._hr_offsets,
                acc_low, acc_high, h, n_steps, reverse, cubic=self._cubic)

        return integrate

    @staticmethod
    def _zeroed_buffer(current, shape, like, zero=True):
        if current is None or tuple(current.shape) != tuple(shape):
            return torch.zeros(shape, dtype=like.dtype, device=like.device)
        if zero:
            current.zero_()
        return current

    def _fused_accumulators(self, low, high):
        # The (zeroed) buffers the fused backward scatters this iteration's
        # field gradients into. Trilinear kernels scatter straight into the
        # parameter-layout buffers; cubic kernels into the padded
        # channels-last ones, in which case the parameter-layout buffers
        # are only the permute targets (overwritten at apply time, so not
        # zeroed here).
        cubic = self._cubic
        self._lr_grad_acc = self._zeroed_buffer(
            self._lr_grad_acc, low.shape, low, zero=not cubic)
        self._hr_grad_acc = self._zeroed_buffer(
            self._hr_grad_acc, high.shape, high, zero=not cubic)
        if not cubic:
            return self._lr_grad_acc, self._hr_grad_acc
        self._lr_scatter_acc = self._zeroed_buffer(
            self._lr_scatter_acc, (*low.shape[:1], *low.shape[2:], 4), low)
        self._hr_scatter_acc = self._zeroed_buffer(
            self._hr_scatter_acc, (*high.shape[:1], *high.shape[2:], 4), high)
        return self._lr_scatter_acc, self._hr_scatter_acc

    def apply_accumulated_field_grad(self):
        if self._pending_fused:
            self._pending_fused = False
            if self._cubic:
                for grad, scatter in ((self._lr_grad_acc, self._lr_scatter_acc),
                                      (self._hr_grad_acc, self._hr_scatter_acc)):
                    grad.copy_(scatter[..., :3].permute(0, 3, 1, 2))
            _adopt_accumulated_grads_((
                (self.flows[0], self._lr_grad_acc),
                (self.flows[1], self._hr_grad_acc)))
            return
        super().apply_accumulated_field_grad()

    def smooth_grad_(self, sigma_hr_cells, across_sigma_hr_cells=0.0,
                     low_res_sigma_hr_cells=0.0):
        """Gaussian-smooth both lattices' gradients in place: ``sigma_hr_cells``
        along z and around each ring (approximating along-sheet directions)
        and ``across_sigma_hr_cells`` across rings; see
        flow_grad_smoothing.smooth_cylindrical_.

        All widths are in nominal high-resolution cells (ring arc spacing
        is approximate because angular cell counts are rounded). Coarse
        cells are ``spatial_scale_factor`` times larger, so they see the same
        flow-frame widths in their own cell units, except that
        ``low_res_sigma_hr_cells`` (0 = same as ``sigma_hr_cells``) gives it
        an along-sheet width of its own.
        """
        tables = ((self._lr_num_phi, self._lr_offsets), (self._hr_num_phi, self._hr_offsets))
        for level, (flow, (num_phi, offsets)) in enumerate(zip(self.flows, tables)):
            if flow.grad is None:
                continue
            scale = self.spatial_scale_factor if level == 0 else 1
            along = _low_res_width(sigma_hr_cells, low_res_sigma_hr_cells) if level == 0 else sigma_hr_cells
            flow_grad_smoothing.smooth_cylindrical_(
                flow.grad, num_phi, offsets, float(along) / scale,
                float(across_sigma_hr_cells) / scale)


class BSplineFlowField(_DirectSampledFlowField):

    # Flow field whose parameters are control points of a tricubic uniform
    # B-spline lattice (see sample_field_bspline) instead of a trilinearly
    # interpolated voxel grid. The C2 basis represents smooth deformations
    # with a coarser lattice, so this is normally paired with a larger
    # model_flow_voxel_resolution; note that at equal resolution it is a
    # smoothed (approximating, not interpolating) version of the trilinear
    # field. Keeps CartesianFlowField's two-level (low-res + high-res)
    # parameter structure and leading slab axis ([num_stages, 3, Z, Y, X])
    # so optimizer grouping, the high-res LR schedule, checkpoint shape
    # checks and the flow-stage migration apply unchanged. Both lattices are
    # sampled directly at query points; integration runs in the fused Triton
    # kernel (flow_triton.rk4_bspline_integrate) where available, and
    # otherwise falls back to the sampler-based RK4 loop over
    # sample_field_bspline (also the CPU/test reference). direct_lr is
    # accepted for constructor parity and ignored.

    def __init__(self, resolution, spatial_scale_factor=6, num_stages=1, direct_lr=False):
        super().__init__()
        self.num_stages = int(num_stages)
        assert self.num_stages >= 1
        self.spatial_scale_factor = int(spatial_scale_factor)
        # Direct sampling needs >= 2 control points per axis
        # (sample_field_bspline divides by shape-1), unlike the cartesian LR
        # lattice, which is only ever an F.interpolate input.
        hr_shape = [max(2, int(s)) for s in resolution]
        lr_shape = [max(2, s // self.spatial_scale_factor) for s in hr_shape]
        self.flows = nn.ParameterList([
            nn.Parameter(torch.zeros([self.num_stages, 3, *lr_shape])),
            nn.Parameter(torch.zeros([self.num_stages, 3, *hr_shape])),
        ])
        self._lr_grad_acc = None
        self._hr_grad_acc = None
        self._pending_fused = False

    def _iteration_fields(self):
        # The LR & HR control-point lattices, [num_stages, 3, Z, Y, X], cut
        # at detached leaves in training.
        assert not self._pending_fused
        return self._maybe_cut_field_leaves((self.flows[0], self.flows[1]))

    @staticmethod
    def _make_sampler(lr_field, hr_field, stage):
        # Velocity sampler for one slab: normalised zyx points in [0, 1] ->
        # cartesian (z, y, x) velocity, evaluating both B-spline lattices
        # directly at each query point.
        lr_slab = lr_field[stage]
        hr_slab = hr_field[stage]

        def sample(normalised_zyx):
            return (
                sample_field_bspline(lr_slab, normalised_zyx)
                + sample_field_bspline(hr_slab, normalised_zyx)
            )
        return sample

    def get_integrator(self):
        """Return ``integrate(y_flat, h, n_steps, reverse=False) -> y_flat``.

        On CUDA every slab's RK4 walk (forward and adjoint each) runs as one
        fused Triton kernel launch, with field gradients accumulated into
        per-lattice buffers; otherwise the eager sampler loop runs, one
        autograd graph per sampler call.
        """
        lr_flow, hr_flow = self.flows[0], self.flows[1]
        if not flow_triton.rk4_triton_available(lr_flow, hr_flow):
            return self._eager_integrator()

        training_field = torch.is_grad_enabled() and (
            lr_flow.requires_grad or hr_flow.requires_grad)
        acc_lo = acc_hi = None
        if training_field:
            # Only one pending-gradient record may be armed per iteration; a
            # stale sampler-path record would double-apply on the parameters
            # after the fused branch consumed its accumulators.
            assert self._pending_field_graphs is None
            if self._lr_grad_acc is None or self._lr_grad_acc.shape != lr_flow.shape:
                self._lr_grad_acc = torch.zeros_like(lr_flow)
            else:
                self._lr_grad_acc.zero_()
            if self._hr_grad_acc is None or self._hr_grad_acc.shape != hr_flow.shape:
                self._hr_grad_acc = torch.zeros_like(hr_flow)
            else:
                self._hr_grad_acc.zero_()
            acc_lo, acc_hi = self._lr_grad_acc, self._hr_grad_acc
            self._pending_fused = True
        # Channels-last copies ([num_stages, Z, Y, X, 3]) for the kernel's
        # read path: each stencil tap then reads its 3 components from
        # consecutive addresses instead of three planes a full channel apart.
        # One copy per lattice per iteration, amortised across the
        # integrator's calls; the gradient accumulators keep the parameter
        # layout.
        low_cl = lr_flow.detach().permute(0, 2, 3, 4, 1).contiguous()
        high_cl = hr_flow.detach().permute(0, 2, 3, 4, 1).contiguous()

        def integrate(y_flat, h, n_steps, reverse=False):
            return flow_triton.rk4_bspline_integrate(
                y_flat, low_cl, high_cl, acc_lo, acc_hi, h, n_steps, reverse)

        return integrate

    def apply_accumulated_field_grad(self):
        if self._pending_fused:
            self._pending_fused = False
            # Both lattices enter the velocity sum unscaled, so the
            # accumulators are exactly the parameter gradients.
            _adopt_accumulated_grads_((
                (self.flows[0], self._lr_grad_acc),
                (self.flows[1], self._hr_grad_acc)))
            return
        super().apply_accumulated_field_grad()

    def smooth_grad_(self, sigma_hr_cells, across_sigma_hr_cells=0.0,
                     low_res_sigma_hr_cells=0.0):
        """Gaussian-smooth both control-point lattices' gradients in place
        (isotropic); same contract as CartesianFlowField.smooth_grad_, the
        lattices being Cartesian grids of control points.
        """
        for level, flow in enumerate(self.flows):
            if flow.grad is None:
                continue
            scale = self.spatial_scale_factor if level == 0 else 1
            along = _low_res_width(sigma_hr_cells, low_res_sigma_hr_cells) if level == 0 else sigma_hr_cells
            flow_grad_smoothing.smooth_cartesian_(flow.grad, float(along) / scale)


class BSplineCylindricalFlowField(CylindricalFlowField):

    # CylindricalFlowField's lattice (same packed ragged parameter layout,
    # per-ring phi counts, local (z, radial, tangential) basis, slab axis and
    # pinned r=0 ring) sampled with tricubic uniform B-spline weights instead
    # of trilinear: cubic along z and r with border-replicated taps, periodic
    # cubic along phi within each ring. Each of the four r-stencil rings is
    # evaluated with its own phi parameterisation, so the interpolant is a C2
    # blend of per-ring bicubic splines rather than one tensor-product
    # spline; every axis' weights still sum to 1, so constants are reproduced
    # exactly away from the axis. Because the cubic stencil spans rings
    # r-1 .. r+2 and below-axis taps clamp to the pinned zero ring, the axis
    # pin suppresses the field over roughly two rings rather than one.
    # Integration reuses CylindricalFlowField's machinery: the fused Triton
    # kernels compiled with CUBIC=True (flow_triton._cylb_* lattice samplers,
    # fed channels-last lattice copies and padded channels-last gradient
    # accumulators) on CUDA, the eager sampler loop over _sample_lattice
    # elsewhere.

    _cubic = True

    @staticmethod
    def _sample_lattice(field, ring_num_phi, ring_offsets, normalised_zyx):
        # Same contract as CylindricalFlowField._sample_lattice.
        nz = field.shape[1]
        nr = ring_num_phi.shape[0]
        orig_shape = normalised_zyx.shape
        z_cont, r_cont, phi_in_2pi, phi = _cylindrical_query_coords(
            normalised_zyx, nz, nr)
        tap_offsets = torch.arange(-1, 3, device=field.device)  # 4

        z_lo = z_cont.floor()
        wz = _bspline_w4(z_cont - z_lo)  # 4z, n
        z_taps = (z_lo.long()[None] + tap_offsets[:, None]).clamp(0, nz - 1)  # 4z, n

        r_lo = r_cont.floor()
        wr = _bspline_w4(r_cont - r_lo)  # 4r, n
        r_taps = (r_lo.long()[None] + tap_offsets[:, None]).clamp(0, nr - 1)  # 4r, n

        # Each r-stencil ring has its own phi resolution, hence its own
        # continuous phi index, fraction and (cyclic) tap indices.
        num_phi_r = ring_num_phi[r_taps]  # 4r, n
        offsets_r = ring_offsets[r_taps]  # 4r, n
        phi_cont = phi_in_2pi[None] * (num_phi_r.to(phi_in_2pi.dtype) / (2. * np.pi))
        phi_lo = phi_cont.floor()
        wp = _bspline_w4(phi_cont - phi_lo)  # 4p, 4r, n
        phi_taps = (phi_lo.long()[None] + tap_offsets[:, None, None]) % num_phi_r[None]
        flat_taps = offsets_r[None] + phi_taps  # 4p, 4r, n

        values = field[:, z_taps[:, None, None, :], flat_taps[None]]  # 3, 4z, 4p, 4r, n
        weights = wz[:, None, None, :] * wp[None] * wr[None, None]  # 4z, 4p, 4r, n
        sampled = (values * weights[None]).sum(dim=(1, 2, 3))  # 3, n local components
        return _cylindrical_local_to_cartesian(sampled, phi, orig_shape)
