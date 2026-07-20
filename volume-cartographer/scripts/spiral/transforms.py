import os

import numpy as np
import scipy.ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro.distributions
from einops import rearrange
from torchdiffeq import odeint

import gap_triton
import sample_spiral
from flow_fields import CartesianFlowField, CylindricalFlowField
from geom_utils import expm_2x2, interp1d
from sample_spiral import get_bounding_windings, get_theta_and_radii


class IntegratedFlowDiffeomorphism(pyro.distributions.transforms.Transform):

    domain = pyro.distributions.constraints.real_vector
    codomain = domain

    def __init__(self, flow_field, flow_min_corner_zyx, flow_max_corner_zyx, num_steps, solver, truncate_at_step=None, event_dim=0, cache_size=0):
        super().__init__(cache_size=cache_size)
        self.flow_field = flow_field
        self.flow_min_corner_zyx = flow_min_corner_zyx
        self.flow_max_corner_zyx = flow_max_corner_zyx
        self.num_steps = num_steps
        self.solver = solver
        self.truncate_at_step = truncate_at_step
        self._event_dim = event_dim
        self._flow_range_zyx = self.flow_max_corner_zyx - self.flow_min_corner_zyx
        self.num_flow_timesteps = getattr(flow_field, 'num_flow_timesteps', 1)
        # Cached sampler/integrator closure at t=0 for the num_flow_timesteps==1 fast path.
        # Built once per diffeomorphism instance (one per training iteration), shared across
        # forward and inverse calls so per-iteration setup (e.g. trilinear LR->HR upsampling)
        # is amortised. A closure built under no_grad cannot route gradients to the field
        # parameters, so it is upgraded (rebuilt) if a later call arrives with grad enabled.
        self._cached_sampler = None
        self._cached_integrator = None
        self._cached_sampler_grad_mode = False

    def _velocity(self, t_int, current_zyx_scaled):
        # t_int is a scalar in [0, 1]; flow_field expects t in [-1, 1]
        t_flow = t_int * 2 - 1
        return self.flow_field.get_sampler(t_flow)(current_zyx_scaled)

    def _call(self, input_zyx, inverse=False):
        # ODE integration of the temporally-varying flow to give a diffeomorphism.
        # The flow & diffeomorphism represent shifts in normalised units [0,1] over the flow region.
        y = (input_zyx - self.flow_min_corner_zyx) / self._flow_range_zyx
        n_steps = self.num_steps if self.truncate_at_step is None else self.truncate_at_step
        t_span = n_steps / self.num_steps
        h = (-t_span if inverse else t_span) / n_steps
        if self.num_flow_timesteps == 1:
            # Time-invariant flow: skip torchdiffeq's per-step dispatch overhead.
            assert self.solver == 'rk4'
            get_integrator = getattr(self.flow_field, 'get_time_invariant_integrator', None)
            if get_integrator is not None:
                # Cartesian fast path: the whole RK4 integration runs as ONE
                # autograd node (identical arithmetic and gradient values; see
                # _RK4SparseFlowIntegrate), instead of ~42 nodes per call.
                if self._cached_integrator is None or (torch.is_grad_enabled() and not self._cached_sampler_grad_mode):
                    self._cached_integrator = get_integrator()
                    self._cached_sampler_grad_mode = torch.is_grad_enabled()
                if n_steps > 0:
                    orig_shape = y.shape
                    y = self._cached_integrator(y.reshape(-1, 3), h, n_steps).view(orig_shape)
            else:
                # e.g. CylindricalFlowField: build the sampler once and inline a manual rk4 loop.
                if self._cached_sampler is None or (torch.is_grad_enabled() and not self._cached_sampler_grad_mode):
                    self._cached_sampler = self.flow_field.get_sampler(0.0)
                    self._cached_sampler_grad_mode = torch.is_grad_enabled()
                sampler = self._cached_sampler
                for _ in range(n_steps):
                    k1 = sampler(y)
                    k2 = sampler(y + (h / 2) * k1)
                    k3 = sampler(y + (h / 2) * k2)
                    k4 = sampler(y + h * k3)
                    y = y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            t0 = 1. if inverse else 0.
            ts = torch.linspace(t0, t0 + h * n_steps, n_steps + 1, device=y.device)
            y = odeint(self._velocity, y, ts, method=self.solver)[-1]
        return y * self._flow_range_zyx + self.flow_min_corner_zyx

    def _inverse(self, input_yx):
        return self._call(input_yx, inverse=True)


class GapExpanderParams(nn.Module):

    def __init__(self, resolution, min_z, max_z, num_windings, dr_per_winding):
        super().__init__()
        self.num_by_winding = (2 * torch.pi * (torch.arange(1, num_windings) + 0.5) * dr_per_winding / resolution + 0.5).to(torch.int64)
        self.num_z = int((max_z - min_z) / resolution)
        self.logits = nn.Parameter(torch.zeros([1, 1, self.num_z, sum(self.num_by_winding)]))
        self.register_buffer('winding_first_logit_idx', torch.cat([torch.zeros([1]), torch.cumsum(self.num_by_winding, dim=0)]))


class GapExpandingTransform(pyro.distributions.transforms.Transform):

    domain = pyro.distributions.constraints.real_vector
    codomain = domain

    def __init__(self, params, dr_per_winding, min_z, max_z, gap_expander_lr_scale, truncate_frac=None, event_dim=0, cache_size=0):
        super().__init__(cache_size=cache_size)
        self.params = params
        self.dr_per_winding = dr_per_winding
        self.min_z = min_z
        self.max_z = max_z
        self.gap_expander_lr_scale = gap_expander_lr_scale
        self.truncate_frac = truncate_frac
        # One transform instance exists per training iteration, and the logits
        # parameter does not change within an iteration. Build the pinned+scaled
        # logits once and share it across every _call/_inverse; each grid_sample
        # then saves a reference to this single tensor instead of a fresh
        # full-size cat+mul copy per transform invocation.
        self._pinned_scaled_logits = None

    def _get_pinned_scaled_logits(self):
        # Rebuild the cache if it was first created under no_grad but the
        # current call needs gradients: a detached cache would silently
        # disconnect the gap logits for every later use of this transform
        # instance (e.g. when a no_grad step-estimation pre-pass is the
        # instance's first call).
        needs_grad = self.params.logits.requires_grad and torch.is_grad_enabled()
        cached = self._pinned_scaled_logits
        if cached is None or (needs_grad and not cached.requires_grad):
            # Pin the 0th logit (i.e. theta=0 on 1th winding) to be zero, to avoid a jump going from winding #0 to #1
            logits = torch.cat([torch.zeros_like(self.params.logits[..., :1]), self.params.logits[..., 1:]], dim=-1)
            self._pinned_scaled_logits = logits * self.gap_expander_lr_scale
        return self._pinned_scaled_logits

    def _triton_consts(self):
        # Constants for the fused gap_triton kernels, cached on the (long
        # lived) params module so float() conversions of tensor scalars do not
        # sync the GPU once per training iteration.
        consts = getattr(self.params, '_triton_consts', None)
        if consts is None:
            idx = self.params.winding_first_logit_idx.to(torch.float32).contiguous()
            consts = {
                'idx': idx,
                'idx_total': float(idx[-1]),
                'min_z': float(self.min_z),
                'max_z': float(self.max_z),
            }
            self.params._triton_consts = consts
        return consts

    def _use_triton(self, input_zyx, dr):
        return isinstance(dr, torch.Tensor) and gap_triton.gap_triton_available(
            input_zyx, self.params.logits, dr)

    def get_transformed_winding_radii(self, theta, z):
        # This returns the sequence of winding radii (true, not shifted) for the radials given by theta and z
        theta_normalised = theta / (2 * torch.pi)
        logits_by_winding = self.get_logits_by_winding(theta, z)
        scales_by_winding = torch.exp(logits_by_winding * 2.e2)
        if self.truncate_frac is not None:
            scales_by_winding = torch.lerp(torch.ones([], device=scales_by_winding.device), scales_by_winding, self.truncate_frac)
        inter_winding_distances = self.dr_per_winding * scales_by_winding
        winding_zero_radii = self.dr_per_winding * theta_normalised
        winding_radii = winding_zero_radii[..., None] + torch.cat([torch.zeros_like(inter_winding_distances[..., :1]), torch.cumsum(inter_winding_distances, dim=-1)[..., :-1]], dim=-1)
        return winding_radii

    def get_logits_by_winding(self, theta, z):
        """Pinned, interpolated native gap logits at ``(theta, z)``.

        The returned values include ``gap_expander_lr_scale`` but deliberately
        exclude the fixed ``2e2`` exponent scale. Keeping this interpolation in
        one method makes the minimum-gap barrier use exactly the same pinning,
        grid coordinates, and bilinear sampling as the forward transform.
        """
        num_windings = len(self.params.num_by_winding)
        winding_first_logit_idx = self.params.winding_first_logit_idx
        theta_normalised = theta / (2 * torch.pi)
        winding_coords = torch.lerp(
            winding_first_logit_idx[:-1], winding_first_logit_idx[1:],
            theta_normalised[..., None])
        winding_coords_normalised = (
            winding_coords / winding_first_logit_idx[-1] * 2 - 1)
        z_normalised = (z - self.min_z) / (self.max_z - self.min_z) * 2 - 1
        return F.grid_sample(
            self._get_pinned_scaled_logits(),
            torch.stack([winding_coords_normalised, z_normalised[..., None].expand(*theta.shape, num_windings)], dim=-1).view(1, -1, num_windings, 2),
            mode='bilinear',
            padding_mode='border',
            align_corners=True,
        ).squeeze(1).squeeze(0).view(*theta.shape, num_windings)

    def get_native_log_gaps(self, winding_idx, theta, z):
        """Log native inter-winding distances before exponentiation.

        ``dr_per_winding`` contributes to the value but is detached so a
        minimum-gap violation can only update the sampled local gap logits.
        This method is intentionally defined only for the untruncated transform:
        warm-up truncation interpolates distances after exponentiation and has
        no equivalent pre-exponentiation log gap.
        """
        if self.truncate_frac is not None:
            raise ValueError('native log gaps are defined only for the untruncated transform')
        logits_by_winding = self.get_logits_by_winding(theta, z)
        winding_idx = winding_idx.to(torch.long).clamp(
            min=0, max=logits_by_winding.shape[-1] - 1)
        sampled = torch.gather(
            logits_by_winding, -1, winding_idx[..., None]).squeeze(-1)
        return torch.log(self.dr_per_winding.detach()) + sampled * 2.e2

    def _call(self, input_zyx):
        theta, original_radius, inner_winding, _ = get_bounding_windings(input_zyx[..., 1:], self.dr_per_winding)
        num_windings = len(self.params.num_by_winding)
        inner_winding_clipped = inner_winding.to(torch.int64).clip(min=0, max=num_windings - 2)
        if self._use_triton(input_zyx, self.dr_per_winding):
            consts = self._triton_consts()
            transformed_inner_radius, transformed_outer_radius = gap_triton.gap_bracketing_radii(
                theta, input_zyx[..., 0], self._get_pinned_scaled_logits(),
                consts['idx'], consts['idx_total'], self.dr_per_winding,
                inner_winding_clipped, self.truncate_frac, consts['min_z'], consts['max_z'])
        else:
            transformed_winding_radii = self.get_transformed_winding_radii(theta, input_zyx[..., 0])
            transformed_inner_radius = torch.gather(transformed_winding_radii, dim=-1, index=inner_winding_clipped[..., None]).squeeze(-1)
            transformed_outer_radius = torch.gather(transformed_winding_radii, dim=-1, index=(inner_winding_clipped + 1)[..., None]).squeeze(-1)
        original_inner_radius = (inner_winding_clipped + theta / (2 * torch.pi)) * self.dr_per_winding
        original_outer_radius = original_inner_radius + self.dr_per_winding
        frac = (original_radius - original_inner_radius) / (original_outer_radius - original_inner_radius)
        transformed_radius = torch.lerp(transformed_inner_radius, transformed_outer_radius, frac)
        delta_radius = transformed_radius - original_radius
        outward_direction = torch.cat([torch.zeros_like(input_zyx[..., :1]), F.normalize(input_zyx[..., 1:], dim=-1)], dim=-1)
        transformed_zyx = input_zyx + outward_direction * delta_radius[..., None]
        return transformed_zyx

    def _inverse(self, input_zyx):
        theta, transformed_radius, _ = get_theta_and_radii(input_zyx[..., 1:], self.dr_per_winding)
        if self._use_triton(input_zyx, self.dr_per_winding):
            consts = self._triton_consts()
            transformed_inner_radius, transformed_outer_radius, inner_winding_clipped = gap_triton.gap_search_radii(
                theta, input_zyx[..., 0], self._get_pinned_scaled_logits(),
                consts['idx'], consts['idx_total'], self.dr_per_winding,
                transformed_radius, self.truncate_frac, consts['min_z'], consts['max_z'])
        else:
            transformed_winding_radii = self.get_transformed_winding_radii(theta, input_zyx[..., 0])
            inner_winding_indices = torch.searchsorted(transformed_winding_radii, transformed_radius[..., None]).squeeze(-1) - 1
            # If shifted_radius is exactly zero, avoid this being -1.
            inner_winding_clipped = inner_winding_indices.clip(min=0, max=transformed_winding_radii.shape[-1] - 2)

            transformed_inner_radius = torch.gather(transformed_winding_radii, dim=-1, index=inner_winding_clipped[..., None]).squeeze(-1)
            transformed_outer_radius = torch.gather(transformed_winding_radii, dim=-1, index=(inner_winding_clipped + 1)[..., None]).squeeze(-1)
        original_inner_radius = (inner_winding_clipped + theta / (2 * torch.pi)) * self.dr_per_winding
        original_outer_radius = original_inner_radius + self.dr_per_winding
        frac = (transformed_radius - transformed_inner_radius) / (transformed_outer_radius - transformed_inner_radius)
        original_radius = torch.lerp(original_inner_radius, original_outer_radius, frac)
        delta_radius = original_radius - transformed_radius
        outward_direction = torch.cat([torch.zeros_like(input_zyx[..., :1]), F.normalize(input_zyx[..., 1:], dim=-1)], dim=-1)
        transformed_zyx = input_zyx + outward_direction * delta_radius[..., None]

        return transformed_zyx


class VaryingLinearTransform(pyro.distributions.transforms.Transform):

    # This applies a z-dependent 2x2 linear transform M(z) on yx, parametrised
    # as M(z) = expm(L(z)) where L(z) is an unconstrained 2x2 matrix.
    # det(M) = exp(tr(L)) > 0, so M is always invertible and orientation-preserving.

    domain = pyro.distributions.constraints.real_vector
    codomain = domain

    def __init__(self, logits, min_z, max_z, truncate_frac=None, event_dim=0, cache_size=0):
        super().__init__(cache_size=cache_size)
        self.min_z = min_z
        self.max_z = max_z
        self.logits = logits
        self.truncate_frac = truncate_frac

    def _call(self, input_zyx, inverse=False):
        zs = input_zyx[..., :1]
        if os.environ.get('FIT_SPIRAL_FAST_LINEAR', '1') != '0':
            # Explicit 1-D lerp over z + elementwise 2x2 apply. The
            # grid_sample this replaces is a pure z-interpolation on a
            # [Z, 2, 2] table (W=1, align_corners=True, border padding), but
            # its backward kernel serializes scattering millions of points
            # into the tiny table; batched [N,2,2]@[N,2,1] matmul likewise
            # dispatches to pathological tiny-gemm cublas launches. Same
            # arithmetic per point (fp-association tolerance class).
            Z = self.logits.shape[0]
            zn = (zs.view(-1) - self.min_z) / (self.max_z - self.min_z)
            coord = (zn * 2 - 1 + 1) / 2 * (Z - 1)
            coord = coord.clamp(min=0., max=float(Z - 1))
            lo = coord.detach().floor().clamp(max=float(Z - 2) if Z > 1 else 0.)
            frac = (coord - lo)[..., None]
            lo = lo.to(torch.int64)
            flat = self.logits.reshape(Z, 4)
            if Z > 1:
                logits = torch.lerp(
                    F.embedding(lo, flat), F.embedding(lo + 1, flat), frac)
            else:
                logits = F.embedding(lo, flat)
            logits = logits.view(*input_zyx.shape[:-1], 2, 2)
        else:
            normalised_zs = (zs.view(-1) - self.min_z) / (self.max_z - self.min_z) * 2 - 1
            logits = F.grid_sample(
                rearrange(self.logits, 'z r c -> 1 (r c) z 1'),
                torch.stack([torch.zeros_like(normalised_zs), normalised_zs], dim=-1)[None, None],
                padding_mode='border',
                align_corners=True
            ).squeeze(2).squeeze(0).T.view(*input_zyx.shape[:-1], 2, 2)
        if inverse:
            logits = -logits
        if self.truncate_frac is not None:
            # In log-space, scaling by truncate_frac gives a geodesic interpolation
            # towards the identity at frac=0
            logits = logits * self.truncate_frac
        M = expm_2x2(logits)
        y, x = input_zyx[..., 1], input_zyx[..., 2]
        yx_out = torch.stack([
            M[..., 0, 0] * y + M[..., 0, 1] * x,
            M[..., 1, 0] * y + M[..., 1, 1] * x,
        ], dim=-1)
        return torch.cat([zs, yx_out], dim=-1)

    def _inverse(self, input_zyx):
        return self._call(input_zyx, inverse=True)


class UmbilicusTransform(pyro.distributions.transforms.Transform):

    # This translates in the yx plane by a z-dependent value (i.e. shears the volume) s.t. the origin is moved to the umbilicus

    domain = pyro.distributions.constraints.real_vector
    codomain = domain

    def __init__(self, umbilicus_zyx, event_dim=2, cache_size=0):
        super().__init__(cache_size=cache_size)
        self._event_dim = event_dim
        yx_filtered = scipy.ndimage.gaussian_filter1d(umbilicus_zyx[:, 1:].cpu().numpy(), sigma=75., axis=0, mode='nearest')
        self._yx = torch.from_numpy(yx_filtered).to(umbilicus_zyx.device).contiguous()
        self._z = umbilicus_zyx[:, :1].contiguous()

    def _call(self, input_zyx, inverse=False):
        centre_yx = interp1d(input_zyx[..., 0].contiguous(), self._z, self._yx)
        return input_zyx + torch.cat([torch.zeros_like(centre_yx[..., :1]), centre_yx], dim=-1) * (-1 if inverse else 1)

    def _inverse(self, input_zyx):
        return self._call(input_zyx, inverse=True)


def ray_gap_enabled():
    # Per-ray specialization of the gap-expander stage for radial-ray sample
    # batches (phase-bundle polylines / registration targets). The generic
    # inverse chain recomputes the whole winding-radius walk per SAMPLE while
    # every sample on a ray shares (theta, z); computing the [rays, windings]
    # radii table once per ray and gathering per sample does the same
    # arithmetic ~2 orders of magnitude fewer times. Tolerance class: equal to
    # the eager per-point pipeline up to fp association (the fused gap_triton
    # kernels it replaces already differ from eager by scan order / FMA).
    return os.environ.get('FIT_SPIRAL_RAY_GAP', '1') != '0'


def ray_specialized_spiral_to_scroll(
    slice_to_spiral_transform, radii, theta, z, pair_id, sin_t, cos_t,
):
    """Spiral->scroll mapping for radial-ray samples, per-ray gap stage.

    Equivalent to ``slice_to_spiral_transform.inv(spiral_poly)`` for
    ``spiral_poly = [z[pair_id], sin_t[pair_id]*radii, cos_t[pair_id]*radii]``
    when the transform is the production chain
    ``Compose([gap, (flip,) diffeo, linear, umbilicus]).inv``. The gap
    expander's transformed-winding-radii table is built once per ray
    ([rays, windings], differentiable) instead of once per sample; the
    per-sample part is a gather + lerp. The flow / linear / umbilicus stages
    see post-gap (and post-flow) coordinates whose z varies per sample, so
    they stay generic per-sample calls.

    Returns the mapped points, or ``None`` when the chain does not match the
    production shape (caller falls back to the generic transform).
    """
    # slice_to_spiral is Compose(parts).inv; depending on the torch/pyro
    # version that is either a ComposeTransform of per-part inverses (in
    # reversed order) or an _InverseTransform wrapping the forward compose.
    # Recover the forward parts without relying on the weakref inv cache.
    inv_parts = getattr(slice_to_spiral_transform, 'parts', None)
    if inv_parts is not None:
        try:
            parts = [p.inv for p in reversed(inv_parts)]
        except AttributeError:
            return None
    else:
        base = getattr(slice_to_spiral_transform, '_inv', None)
        parts = list(getattr(base, 'parts', None) or []) or None
    if not parts or not isinstance(parts[0], GapExpandingTransform):
        return None
    gap, rest = parts[0], parts[1:]
    flip = None
    if rest and isinstance(rest[0], pyro.distributions.transforms.AffineTransform):
        flip, rest = rest[0], rest[1:]
    if len(rest) != 3 or not (
            isinstance(rest[0], IntegratedFlowDiffeomorphism)
            and isinstance(rest[1], VaryingLinearTransform)
            and isinstance(rest[2], UmbilicusTransform)):
        return None
    diffeo, linear, umbilicus = rest

    dr = gap.dr_per_winding
    theta_norm = theta / (2 * torch.pi)
    # Per-ray transformed winding radii (differentiable through logits + dr;
    # includes the truncate_frac warm-up lerp exactly like the eager path).
    table = gap.get_transformed_winding_radii(theta, z)
    num_windings = table.shape[-1]

    # Eager _call per-sample pipeline, with per-ray quantities gathered.
    tn_s = theta_norm[pair_id]
    shifted = (radii - tn_s * dr).clamp(min=0.)
    inner = torch.floor(shifted / dr).to(torch.int64).clip(
        min=0, max=num_windings - 2)
    # Flat per-sample gather from the per-ray table; never materialize the
    # [samples, windings] expansion. F.embedding rather than plain indexing:
    # index backward is a pathological _index_put_impl_ accumulate here,
    # embedding_dense_backward is the fused gather-accumulate kernel.
    flat_table = table.reshape(-1, 1)
    flat_idx = pair_id * num_windings + inner
    r_in = F.embedding(flat_idx, flat_table).squeeze(-1)
    r_out = F.embedding(flat_idx + 1, flat_table).squeeze(-1)
    original_inner = (inner + tn_s) * dr
    original_outer = original_inner + dr
    frac = (radii - original_inner) / (original_outer - original_inner)
    transformed_radius = torch.lerp(r_in, r_out, frac)
    sin_s, cos_s = sin_t[pair_id], cos_t[pair_id]
    x_sign = -1.0 if flip is not None else 1.0
    pts = torch.stack([
        z[pair_id],
        sin_s * transformed_radius,
        (cos_s * transformed_radius) * x_sign,
    ], dim=-1)
    pts = diffeo._call(pts)
    pts = linear._call(pts)
    return umbilicus._call(pts)


class SpiralAndTransform(nn.Module):

    def __init__(self, flow_integration_steps, flow_integration_solver, flow_min_corner_zyx, flow_max_corner_zyx, umbilicus_zyx, config, spiral_outward_sense='CW'):

        super().__init__()

        self.cfg = config
        self.spiral_outward_sense = spiral_outward_sense
        self.flow_integration_steps = flow_integration_steps
        self.flow_integration_solver = flow_integration_solver
        self.flow_min_corner_zyx = flow_min_corner_zyx
        self.flow_max_corner_zyx = flow_max_corner_zyx
        self.spiral_intensity = 200 / 255
        self.dr_per_winding_scale = 12.  # larger value increases effective learning rate
        self.linear_logits_scale = 40.  # larger value increases effective learning rate

        self.umbilicus_transform = UmbilicusTransform(umbilicus_zyx)
        self.dr_per_winding_logit = nn.Parameter(torch.tensor(config['initial_dr_per_winding'] / self.dr_per_winding_scale, dtype=torch.float32))

        flow_resolution = (flow_max_corner_zyx - flow_min_corner_zyx) // config['flow_voxel_resolution']
        if config['flow_field_type'] == 'cylindrical':
            self.flow_field = CylindricalFlowField(
                flow_resolution,
                lr_scale_factor=config['flow_field_high_res_lr_scale_initial'],
                num_flow_timesteps=config['num_flow_timesteps'],
            )
        else:
            self.flow_field = CartesianFlowField(
                flow_resolution,
                lr_scale_factor=config['flow_field_high_res_lr_scale_initial'],
                num_flow_timesteps=config['num_flow_timesteps'],
            )

        self.linear_logits = nn.Parameter(torch.zeros([int(flow_max_corner_zyx[0] - flow_min_corner_zyx[0]) // config['linear_z_resolution'], 2, 2], dtype=torch.float32))

        self.gap_expander_params = GapExpanderParams(
            resolution=config['gap_expander_logit_resolution'],
            min_z=flow_min_corner_zyx[0],
            max_z=flow_max_corner_zyx[0],
            num_windings=config['gap_expander_num_windings'],
            dr_per_winding=config['initial_dr_per_winding'],  # this is a nominal (fixed) winding spacing which we only use to calculate the number of logits
        )

    @property
    def device(self):
        return self.linear_logits.device

    def _get_transform_parts(self, truncate_at_step=None):
        truncate_frac = None if truncate_at_step is None else truncate_at_step / (self.flow_integration_steps - 1)
        diffeomorphism = IntegratedFlowDiffeomorphism(self.flow_field, self.flow_min_corner_zyx, self.flow_max_corner_zyx, num_steps=self.flow_integration_steps, solver=self.flow_integration_solver, truncate_at_step=truncate_at_step)
        gap_expander = GapExpandingTransform(
            self.gap_expander_params,
            self.get_dr_per_winding(),
            self.flow_min_corner_zyx[0],
            self.flow_max_corner_zyx[0],
            self.cfg['gap_expander_lr_scale'],
            truncate_frac,
        )
        if self.spiral_outward_sense == 'CW':
            maybe_flip = []
        else:
            assert self.spiral_outward_sense == 'ACW'
            # To make spiral go anticlockwise in slice space (going outwards from the centre), flip it horizontally
            maybe_flip = [pyro.distributions.transforms.AffineTransform(loc=0., scale=torch.tensor([1., 1., -1.], device=self.device))]
        return gap_expander, maybe_flip, diffeomorphism, truncate_frac

    def get_slice_to_spiral_transform(self, truncate_at_step=None):
        gap_expander, maybe_flip, diffeomorphism, truncate_frac = self._get_transform_parts(truncate_at_step)
        return pyro.distributions.transforms.ComposeTransform([
            gap_expander,
            *maybe_flip,
            diffeomorphism,
            VaryingLinearTransform(self.linear_logits * self.linear_logits_scale, self.flow_min_corner_zyx[0], self.flow_max_corner_zyx[0], truncate_frac),
            self.umbilicus_transform,
        ]).inv

    def get_flowbox_to_spiral_transform(self, include_diffeomorphism=True):
        # Maps positions expressed in the flow lattice's coordinate frame (the
        # spiral-side intermediate space in which the diffeomorphism integrates)
        # back to canonical spiral space. With include_diffeomorphism a lattice
        # position is treated as an integration-trajectory *end* point; without,
        # as a trajectory *start* point. The two differ by at most the flow
        # displacement, so evaluating both brackets the material coordinates a
        # flow voxel can influence.
        gap_expander, maybe_flip, diffeomorphism, _ = self._get_transform_parts()
        parts = [gap_expander, *maybe_flip] + ([diffeomorphism] if include_diffeomorphism else [])
        return pyro.distributions.transforms.ComposeTransform(parts).inv

    def get_dr_per_winding(self):
        return F.softplus(self.dr_per_winding_logit * self.dr_per_winding_scale)

    def get_native_log_gaps(self, winding_idx, theta, z):
        """Exact pre-exponentiation log gap for the native gap expander."""
        gap_expander, _, _, truncate_frac = self._get_transform_parts()
        assert truncate_frac is None
        return gap_expander.get_native_log_gaps(winding_idx, theta, z)

    def get_spiral_density(self, spiral_zyx, winding_range=None):
        if winding_range is None:
            winding_range = (self.cfg['output_first_winding'], float('inf'))
        return sample_spiral.get_spiral_density(
            spiral_zyx[..., 1:],
            dr_per_winding=self.get_dr_per_winding(),
            sigma=1.,
            winding_range=winding_range,
        ) * self.spiral_intensity
