import os

import numpy as np
import scipy.ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro.distributions
from einops import rearrange

import flow_grad_smoothing
import gap_triton
import pins as pins_module
import sample_spiral
from flow_fields import (
    BSplineCylindricalFlowField,
    BSplineFlowField,
    CartesianFlowField,
    CylindricalFlowField,
)
from gap_parameterization import (
    calibrated_gap_softplus_scale,
    initial_dr_logit,
    lower_bounded_dr,
    lower_bounded_gap,
)
from geom_utils import expm_2x2, interp1d
from sample_spiral import get_bounding_windings, get_theta_and_radii


class IntegratedFlowDiffeomorphism(pyro.distributions.transforms.Transform):
    """The diffeomorphism a piecewise-stationary flow field integrates to.

    Each slab of the flow field is a stationary velocity field integrated for
    unit time with ``num_steps`` RK4 steps; the slabs compose in order (spiral
    -> slice), and the inverse runs them backwards in reverse order. The flow
    and diffeomorphism represent shifts in normalised units [0, 1] over the
    flow region.
    """

    domain = pyro.distributions.constraints.real_vector
    codomain = domain

    def __init__(self, flow_field, flow_min_corner_zyx, flow_max_corner_zyx, num_steps, solver, truncate_at_step=None, event_dim=0, cache_size=0):
        super().__init__(cache_size=cache_size)
        assert solver == 'rk4', solver
        self.flow_field = flow_field
        self.flow_min_corner_zyx = flow_min_corner_zyx
        self.flow_max_corner_zyx = flow_max_corner_zyx
        self.num_steps = num_steps
        self.solver = solver
        self.truncate_at_step = truncate_at_step
        self._event_dim = event_dim
        self._flow_range_zyx = self.flow_max_corner_zyx - self.flow_min_corner_zyx
        # Cached integrator closure. Built once per diffeomorphism instance
        # (one per training iteration), shared across forward and inverse
        # calls so per-iteration setup (e.g. trilinear LR->HR upsampling) is
        # amortised. A closure built under no_grad cannot route gradients to
        # the field parameters, so it is upgraded (rebuilt) if a later call
        # arrives with grad enabled.
        self._cached_integrator = None
        self._cached_integrator_grad_mode = False

    def _call(self, input_zyx, inverse=False):
        y = (input_zyx - self.flow_min_corner_zyx) / self._flow_range_zyx
        # truncate_at_step integrates only the first steps of every slab (the
        # warm-up ramp); the step size is always that of the full schedule.
        n_steps = self.num_steps if self.truncate_at_step is None else self.truncate_at_step
        h = (-1.0 if inverse else 1.0) / self.num_steps
        if self._cached_integrator is None or (torch.is_grad_enabled() and not self._cached_integrator_grad_mode):
            self._cached_integrator = self.flow_field.get_integrator()
            self._cached_integrator_grad_mode = torch.is_grad_enabled()
        if n_steps > 0:
            # Every slab's RK4 walk runs as ONE autograd node on CUDA (see
            # flow_triton); the eager fallback composes one node per slab.
            orig_shape = y.shape
            y = self._cached_integrator(y.reshape(-1, 3), h, n_steps, reverse=inverse).view(orig_shape)
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

    def __init__(self, params, dr_per_winding, min_z, max_z,
                 gap_expander_lr_scale, min_gap=1.0, softplus_bias=4.0,
                 softplus_scale=None, truncate_frac=None, event_dim=0,
                 cache_size=0):
        super().__init__(cache_size=cache_size)
        self.params = params
        self.dr_per_winding = dr_per_winding
        self.min_z = min_z
        self.max_z = max_z
        self.gap_expander_lr_scale = gap_expander_lr_scale
        self.min_gap = float(min_gap)
        self.softplus_bias = float(softplus_bias)
        self.softplus_scale = float(
            softplus_scale
            if softplus_scale is not None
            else calibrated_gap_softplus_scale(
                float(dr_per_winding.detach()), self.min_gap,
                self.softplus_bias))
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
            # Pin the 0th logit (i.e. theta=0 on 1th winding) to be zero, to avoid a jump going from winding #0 to #1.
            # Keep this in sync with SpiralAndTransform.get_shared_transform_tensors,
            # which precomputes the same tensor for injection as a detached leaf.
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
                'min_gap': self.min_gap,
                'softplus_bias': self.softplus_bias,
                'softplus_denominator': float(F.softplus(
                    torch.tensor(self.softplus_bias))),
                'softplus_scale': self.softplus_scale,
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
        inter_winding_distances = lower_bounded_gap(
            logits_by_winding, self.dr_per_winding, self.min_gap,
            self.softplus_bias, self.softplus_scale)
        if self.truncate_frac is not None:
            inter_winding_distances = torch.lerp(
                self.dr_per_winding, inter_winding_distances,
                self.truncate_frac)
        winding_zero_radii = self.dr_per_winding * theta_normalised
        winding_radii = winding_zero_radii[..., None] + torch.cat([torch.zeros_like(inter_winding_distances[..., :1]), torch.cumsum(inter_winding_distances, dim=-1)[..., :-1]], dim=-1)
        return winding_radii

    def get_logits_by_winding(self, theta, z):
        """Pinned, interpolated native gap logits at ``(theta, z)``.

        The returned values include ``gap_expander_lr_scale`` but deliberately
        exclude the calibrated softplus scale. Keeping this interpolation in
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
        """Log native lower-bounded inter-winding distances.

        ``dr_per_winding`` contributes to the value but is detached so a
        minimum-gap violation can only update the sampled local gap logits.
        This method is intentionally defined only for the untruncated transform:
        warm-up truncation interpolates distances afterwards and has no
        equivalent native log gap.
        """
        if self.truncate_frac is not None:
            raise ValueError('native log gaps are defined only for the untruncated transform')
        logits_by_winding = self.get_logits_by_winding(theta, z)
        winding_idx = winding_idx.to(torch.long).clamp(
            min=0, max=logits_by_winding.shape[-1] - 1)
        sampled = torch.gather(
            logits_by_winding, -1, winding_idx[..., None]).squeeze(-1)
        gaps = lower_bounded_gap(
            sampled, self.dr_per_winding.detach(), self.min_gap,
            self.softplus_bias, self.softplus_scale)
        return torch.log(gaps)

    def _call(self, input_zyx):
        theta, original_radius, inner_winding, _ = get_bounding_windings(input_zyx[..., 1:], self.dr_per_winding)
        num_windings = len(self.params.num_by_winding)
        inner_winding_clipped = inner_winding.to(torch.int64).clip(min=0, max=num_windings - 2)
        if self._use_triton(input_zyx, self.dr_per_winding):
            consts = self._triton_consts()
            transformed_inner_radius, transformed_outer_radius = gap_triton.gap_bracketing_radii(
                theta, input_zyx[..., 0], self._get_pinned_scaled_logits(),
                consts['idx'], consts['idx_total'], self.dr_per_winding,
                inner_winding_clipped, self.truncate_frac,
                consts['min_z'], consts['max_z'], consts['min_gap'],
                consts['softplus_bias'], consts['softplus_denominator'],
                consts['softplus_scale'])
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
                transformed_radius, self.truncate_frac,
                consts['min_z'], consts['max_z'], consts['min_gap'],
                consts['softplus_bias'], consts['softplus_denominator'],
                consts['softplus_scale'])
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


class PinnedGapExpandingTransform(GapExpandingTransform):
    """Gap expander whose winding radii are pinned.

    The parent's winding radii are blended with constraints from ``pin_table``
    (:class:`pins.PinTable`) in intermediate-radius space. The resulting
    radius deformation is composed with the free map, with exact full-strength
    pins unless constraints conflict. The fused Triton path is bypassed:
    every call runs the eager pinned arithmetic,
    chunked over points to bound the ``[chunk, windings]`` intermediates.
    """

    def __init__(self, *args, pin_table, chunk_size=16384, **kwargs):
        super().__init__(*args, **kwargs)
        self.pin_table = pin_table
        self.chunk_size = int(chunk_size)

    def _use_triton(self, input_zyx, dr):
        return False

    def ray_map(self, theta, z):
        """Build the free map and pin-only radius deformation per query ray."""
        pre_pin_winding_radii = self.get_transformed_winding_radii(theta, z)
        R, S, w, valid = self.pin_table.anchors(theta, z)
        return pins_module.build_pinned_ray_map(
            pre_pin_winding_radii, self.dr_per_winding, theta / (2 * torch.pi),
            R, S, w, valid, self.min_gap)

    def _radial(self, input_zyx, inverse):
        flat = input_zyx.reshape(-1, 3)
        pieces = []
        for start in range(0, flat.shape[0], self.chunk_size):
            chunk = flat[start:start + self.chunk_size]
            theta, radius, _ = get_theta_and_radii(chunk[..., 1:], self.dr_per_winding)
            z = chunk[..., 0]
            ray_map = self.ray_map(theta, z)
            if inverse:
                # intermediate -> canonical
                mapped = pins_module.pinned_map_forward(radius, ray_map)
            else:
                # canonical -> intermediate
                mapped = pins_module.pinned_map_inverse(radius, ray_map)
            delta_radius = mapped - radius
            outward_direction = torch.cat([
                torch.zeros_like(chunk[..., :1]),
                F.normalize(chunk[..., 1:], dim=-1)], dim=-1)
            pieces.append(chunk + outward_direction * delta_radius[..., None])
        if not pieces:
            return input_zyx
        return torch.cat(pieces, dim=0).reshape(input_zyx.shape)

    def _call(self, input_zyx):
        return self._radial(input_zyx, inverse=False)

    def _inverse(self, input_zyx):
        return self._radial(input_zyx, inverse=True)

    @torch.no_grad()
    def pin_diagnostics(self, pins, chunk_size=None):
        """Evaluate the pinned map at every pin's own ray.

        ``pins`` is the ``[P, 4]`` ``(z, theta, r, dr (T + n))`` tensor the
        table was built from. Returns exactness residuals, guard-violation
        counts split by cause, and the minimum effective winding gap.
        """
        chunk_size = chunk_size or self.chunk_size
        residuals = []
        order_violations = 0
        min_rise_violations = 0
        rays_with_violation = 0
        anchors_total = 0
        min_effective_gap = float('inf')
        for start in range(0, pins.shape[0], chunk_size):
            chunk = pins[start:start + chunk_size]
            z, theta, r, r_target_shifted = chunk.unbind(-1)
            ray_map = self.ray_map(theta, z)
            theta_norm = theta / (2 * torch.pi)
            s = pins_module.pinned_map_forward(r, ray_map)
            residuals.append((s - (r_target_shifted + self.dr_per_winding * theta_norm)).abs())
            order_violations += int(ray_map.order_violations().sum())
            min_rise_violations += int(ray_map.min_rise_violations().sum())
            rays_with_violation += int(((ray_map.order_violations() + ray_map.min_rise_violations()) > 0).sum())
            anchors_total += int(ray_map.anchor_valid.sum())
            min_effective_gap = min(
                min_effective_gap, float(ray_map.minimum_winding_gap(self.dr_per_winding).min()))
        residual = torch.cat(residuals) if residuals else pins.new_zeros([0])
        num_pins = int(pins.shape[0])
        dr = float(self.dr_per_winding.detach())
        return {
            'pin_residual_max': float(residual.max()) if residual.numel() else 0.0,
            'pin_residual_mean': float(residual.mean()) if residual.numel() else 0.0,
            # Pins missed by more than a tenth of a winding: the exactness failures.
            'pin_inexact_fraction': float((residual > 0.1 * dr).float().mean()) if residual.numel() else 0.0,
            # Anchor-level sums (one ray can carry many) and their denominators.
            'pin_order_violations': order_violations,
            'pin_min_rise_violations': min_rise_violations,
            'pin_anchors_evaluated': anchors_total,
            'pin_rays_evaluated': num_pins,
            'pin_rays_with_violation_fraction': rays_with_violation / max(num_pins, 1),
            'pin_min_effective_gap': min_effective_gap,
        }


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
        # Add the centre to the yx columns directly rather than materialising
        # a zero-padded [..., 3] offset; z passes through untouched (it only
        # gained an exact +0 before) and the yx additions are identical.
        yx = (input_zyx[..., 1:] - centre_yx if inverse
              else input_zyx[..., 1:] + centre_yx)
        return torch.cat([input_zyx[..., :1], yx], dim=-1)

    def _inverse(self, input_zyx):
        return self._call(input_zyx, inverse=True)


def pinned_gap_stage(slice_to_spiral_transform):
    """The PinnedGapExpandingTransform of a production chain, or None."""
    inv_parts = getattr(slice_to_spiral_transform, 'parts', None)
    if inv_parts is not None:
        try:
            parts = [p.inv for p in reversed(inv_parts)]
        except AttributeError:
            return None
    else:
        base = getattr(slice_to_spiral_transform, '_inv', None)
        parts = list(getattr(base, 'parts', None) or [])
    if parts and isinstance(parts[0], PinnedGapExpandingTransform):
        return parts[0]
    return None


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
    if getattr(parts[0], 'pin_table', None) is not None:
        # The pinned gap stage is not a per-ray table gather; use the generic
        # (pinned) transform instead.
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
    pre_pin_winding_radii = gap.get_transformed_winding_radii(theta, z)
    num_windings = pre_pin_winding_radii.shape[-1]

    # Eager _call per-sample pipeline, with per-ray quantities gathered.
    tn_s = theta_norm[pair_id]
    shifted = (radii - tn_s * dr).clamp(min=0.)
    inner = torch.floor(shifted / dr).to(torch.int64).clip(
        min=0, max=num_windings - 2)
    # Flat per-sample gather from the per-ray table; never materialize the
    # [samples, windings] expansion. F.embedding rather than plain indexing:
    # index backward is a pathological _index_put_impl_ accumulate here,
    # embedding_dense_backward is the fused gather-accumulate kernel.
    flat_table = pre_pin_winding_radii.reshape(-1, 1)
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
        self.gap_min_gap = float(config.get(
            'model_gap_expander_min_gap', 1.0))
        self.gap_softplus_bias = float(
            config.get('model_gap_expander_softplus_bias', 4.0))
        self.gap_softplus_scale = calibrated_gap_softplus_scale(
            float(config['model_initial_dr_per_winding']), self.gap_min_gap,
            self.gap_softplus_bias)
        self.dr_per_winding_logit = nn.Parameter(initial_dr_logit(
            float(config['model_initial_dr_per_winding']), self.gap_min_gap))

        flow_resolution = (flow_max_corner_zyx - flow_min_corner_zyx) // config['model_flow_voxel_resolution']
        flow_field_cls = {
            'cartesian': CartesianFlowField,
            'cylindrical': CylindricalFlowField,
            'bspline': BSplineFlowField,
            'bspline_cylindrical': BSplineCylindricalFlowField,
        }[config['model_flow_field_type']]

        # num_flow_stages: number of stationary velocity fields whose integrated diffeomorphisms
        # are composed sequentially (phi = exp(v_N) o ... o exp(v_1) in the spiral->slice
        # direction; the inverse runs them backwards in reverse order). They are the slabs of
        # one flow field's lattices ([num_flow_stages, 3, ...]), integrated by one
        # IntegratedFlowDiffeomorphism, so num_flow_stages == 1 is exactly the original
        # single-field model with identical parameters and state_dict keys.
        self.num_flow_stages = int(config.get('model_num_flow_stages', 1) or 1)
        assert self.num_flow_stages >= 1
        self.flow_field = flow_field_cls(
            flow_resolution,
            num_stages=self.num_flow_stages,
            direct_lr=config.get('model_flow_field_direct_lr', False),
        )

        self.linear_logits = nn.Parameter(torch.zeros([int(flow_max_corner_zyx[0] - flow_min_corner_zyx[0]) // config['model_linear_z_resolution'], 2, 2], dtype=torch.float32))

        self.gap_expander_params = GapExpanderParams(
            resolution=config['model_gap_expander_logit_resolution'],
            min_z=flow_min_corner_zyx[0],
            max_z=flow_max_corner_zyx[0],
            num_windings=config.get(
                'model_gap_expander_capacity_windings',
                config['model_gap_expander_num_windings']),
            dr_per_winding=config['model_initial_dr_per_winding'],  # this is a nominal (fixed) winding spacing which we only use to calculate the number of logits
        )

        # Pinned winding radii. pin_targets is
        # the per-component fractional winding coordinate T; it is created by
        # init_pin_targets() once the constraint graph's component count is
        # known, so a model without pins has no such parameter. The registry
        # (pins.PinRegistry) and the active flag are runtime state set by the
        # fitter; while inactive every transform is the unpinned one.
        self.pin_targets = None
        self.pin_registry = None
        self.pins_active = False
        self._pin_slots = None
        self._pin_groups = None
        self._pin_rebuilds = 0
        self.last_pins = None

    @property
    def device(self):
        return self.linear_logits.device

    # -- pins ----------------------------------------------------------------

    def init_pin_targets(self, num_components):
        """Create ``T`` (``pin_targets``) for ``num_components`` components."""
        self.pin_targets = nn.Parameter(
            torch.zeros([int(num_components)], dtype=torch.float32, device=self.device))
        return self.pin_targets

    def set_pin_registry(self, registry, *, reset_targets=True):
        """Attach a finalised registry; optionally load its ``T`` estimate."""
        self.pin_registry = registry.to(self.device) if registry is not None else None
        self._pin_slots = None
        self._pin_groups = None
        if registry is not None:
            if self.pin_targets is None or self.pin_targets.numel() != registry.num_components:
                self.init_pin_targets(registry.num_components)
            if reset_targets:
                with torch.no_grad():
                    self.pin_targets.copy_(registry.initial_T.to(self.device))

    def effective_pin_targets(self):
        registry = self.pin_registry
        return torch.where(registry.fixed_T, registry.fixed_T_value, self.pin_targets)

    def get_unpinned_slice_to_spiral_transform(self):
        """The full scroll -> spiral transform with the free gap expander."""
        active = self.pins_active
        self.pins_active = False
        try:
            return self.get_slice_to_spiral_transform()
        finally:
            self.pins_active = active

    @torch.no_grad()
    def estimate_pin_targets(self, chunk_size=262144):
        """``T_g`` = median over the component's pins of the canonical
        shifted winding ``s_free(r_i)/dr - n_i`` under the current unpinned
        model; fixed components keep their value."""
        registry = self.pin_registry
        dr = self.get_dr_per_winding()
        transform = self.get_unpinned_slice_to_spiral_transform()
        zyx = registry.zyx
        spiral = torch.cat([
            transform(zyx[start:start + chunk_size])
            for start in range(0, zyx.shape[0], chunk_size)], dim=0) \
            if zyx.shape[0] else zyx.new_zeros([0, 3])
        theta, _, shifted = get_theta_and_radii(spiral[..., 1:], dr)
        estimate = shifted / dr - registry.adjusted_n(theta).to(shifted.dtype)
        T = torch.zeros([registry.num_components], dtype=torch.float32, device=self.device)
        for component in range(registry.num_components):
            mask = registry.component == component
            if mask.any():
                T[component] = estimate[mask].median()
        return torch.where(registry.fixed_T, registry.fixed_T_value, T)

    def get_slice_to_intermediate_transform(self, shared=None, truncate_at_step=None):
        """Scroll -> intermediate space: the chain without the gap expander."""
        _, maybe_flip, diffeomorphism, truncate_frac = self._get_transform_parts(
            truncate_at_step, shared, with_pins=False)
        scaled_linear_logits = (
            shared[1] if shared is not None
            else self.linear_logits * self.linear_logits_scale)
        return pyro.distributions.transforms.ComposeTransform([
            *maybe_flip,
            diffeomorphism,
            VaryingLinearTransform(scaled_linear_logits, self.flow_min_corner_zyx[0], self.flow_max_corner_zyx[0], truncate_frac),
            self.umbilicus_transform,
        ]).inv

    def sample_pin_subset(self, sample_count):
        """Stratified random subset of the registry for one training step.

        Every component keeps at least one pin and otherwise a share
        proportional to its size. Returns ``(indices, eps_theta, eps_z)``
        where the footprints are widened for the thinning (own-object
        spacing under uniform random sampling: by ``1/sqrt(f)``
        for 2-D patch grids, ``1/f`` for chains), capped like the registry.
        """
        registry = self.pin_registry
        num_pins = registry.num_pins
        sample_count = int(sample_count or 0)
        if sample_count <= 0 or num_pins <= sample_count:
            return None
        device = registry.zyx.device
        counts = torch.bincount(registry.component, minlength=registry.num_components).to(torch.float32)
        fraction = sample_count / num_pins
        quota = torch.clamp(torch.floor(counts * fraction), min=1.0)
        quota = torch.where(counts > 0, torch.minimum(quota, counts), torch.zeros_like(quota))
        # Rank each pin within its component by a random key.
        key = torch.rand([num_pins], device=device)
        order = torch.argsort(registry.component.to(torch.float64) * 2.0 + key.to(torch.float64))
        component_sorted = registry.component[order]
        starts = torch.cumsum(counts, dim=0) - counts
        rank = torch.arange(num_pins, device=device, dtype=torch.float32) - starts[component_sorted]
        keep_sorted = rank < quota[component_sorted]
        indices = order[keep_sorted]
        scale_component = counts / quota.clamp(min=1.0)
        scale = scale_component[registry.component[indices]]
        kind = registry.kind[indices]
        widen = torch.where(kind == pins_module.PIN_KIND_PATCH, torch.sqrt(scale),
                            torch.where(kind == pins_module.PIN_KIND_CHAIN, scale, torch.ones_like(scale)))
        rule = self._pin_footprint_caps()
        eps_theta = (registry.eps_theta[indices] * widen).clamp(max=rule[0])
        eps_z = (registry.eps_z[indices] * widen).clamp(max=rule[1])
        return indices, eps_theta, eps_z

    def _pin_footprint_caps(self):
        return (float(self.cfg.get('model_pin_kernel_max_theta_radians', 0.25)),
                float(self.cfg.get('model_pin_kernel_max_z_voxels', 200.0)))

    def compute_pins(self, shared=None, registry=None, chunk_size=None, subsample=False):
        """Push the registry through the flow chain.

        Returns ``pins`` ``[P, 4]`` = ``(z, theta, r, dr (T + n))`` in
        intermediate space with the graph attached (flow, linear, ``dr``,
        ``T``), and stores the pins' winding slots for the table build.
        ``chunk_size`` bounds the flow-integration batch under ``no_grad``
        (with the graph attached the whole registry goes through at once).
        With ``subsample`` (the training step) only the stratified subset of
        ``sample_count_pins`` registry pins is pushed; export and
        diagnostics use the full registry.
        """
        registry = self.pin_registry if registry is None else registry
        dr = shared[0] if shared is not None else self.get_dr_per_winding()
        transform = self.get_slice_to_intermediate_transform(shared)
        subset = self.sample_pin_subset(self.cfg.get('sample_count_pins', 0)) if subsample else None
        if subset is None:
            self._pin_view = {
                'indices': None, 'component': registry.component,
                'eps_theta': registry.eps_theta, 'eps_z': registry.eps_z,
                'kind': registry.kind, 'n0': registry.n0, 'theta0': registry.theta0,
                'local_gap': registry.local_gap, 'num_pins': registry.num_pins}
            zyx = registry.zyx
        else:
            indices, eps_theta, eps_z = subset
            self._pin_view = {
                'indices': indices, 'component': registry.component[indices],
                'eps_theta': eps_theta, 'eps_z': eps_z, 'kind': registry.kind[indices],
                'n0': registry.n0[indices], 'theta0': registry.theta0[indices],
                'local_gap': registry.local_gap[indices], 'num_pins': int(indices.numel())}
            zyx = registry.zyx[indices]
        view = self._pin_view
        if chunk_size is None and not torch.is_grad_enabled():
            chunk_size = 1 << 20
        if chunk_size is not None and not torch.is_grad_enabled():
            intermediate = torch.cat([
                transform(zyx[start:start + chunk_size])
                for start in range(0, zyx.shape[0], chunk_size)], dim=0) \
                if zyx.shape[0] else zyx.new_zeros([0, 3])
        else:
            intermediate = transform(zyx)
        theta, radius, _ = get_theta_and_radii(intermediate[..., 1:], dr)
        z = intermediate[..., 0]
        n = (view['n0'] + torch.round((view['theta0'] - theta.detach()) / (2 * torch.pi)).to(torch.int32)).to(torch.float32)
        T_eff = self.effective_pin_targets()
        # Shifted target dr (T + n); the lookup adds the query ray's angular
        # term so anchors are continuous across the theta = 0 seam.
        target = dr * (T_eff[view['component']] + n)
        pins = torch.stack([z, theta, radius, target], dim=-1)
        self._pin_slots_next = torch.round(T_eff.detach()[view['component']] + n).to(torch.int64)
        self._pin_n_next = n.to(torch.int32)
        self.last_pins = pins
        return pins

    def pin_groups_stale(self, settings_key, rebin_interval):
        """True when the coincidence groups must be recomputed: the pin set,
        any pin's slot or the coincidence settings changed, or the groups are
        ``rebin_interval`` steps old (pins move every step, so previously
        coincident pins can separate while staying merged)."""
        if getattr(self, '_pin_groups_key', None) != settings_key:
            return True
        if getattr(self, '_pin_groups_age', 0) >= max(int(rebin_interval), 1):
            return True
        return self.pin_slots_changed()

    def pin_slots_changed(self):
        """True when the pin set or any pin's slot differs from the cached table."""
        if self._pin_slots is None or self._pin_slots.shape != self._pin_slots_next.shape:
            return True
        indices = self._pin_view.get('indices')
        cached = getattr(self, '_pin_slots_indices', None)
        if (indices is None) != (cached is None):
            return True
        if indices is not None and (indices.shape != cached.shape or bool((indices != cached).any())):
            return True
        return bool((self._pin_slots != self._pin_slots_next).any())

    def build_pin_table(self, pins, coincidence_frac, conflict_tolerance, dr=None,
                        rebin_interval=1):
        """Rasterise ``pins`` into a :class:`pins.PinTable`.

        The slot assignment and coincidence groups are rebuilt whenever any
        pin's slot changed; otherwise the cached groups are reused
        and only the differentiable values and the cells are refreshed.
        """
        registry = self.pin_registry
        view = self._pin_view
        dr = self.get_dr_per_winding() if dr is None else dr
        num_slots = len(self.gap_expander_params.num_by_winding)
        min_z = float(self.flow_min_corner_zyx[0])
        max_z = float(self.flow_max_corner_zyx[0])
        settings_key = (float(coincidence_frac), float(conflict_tolerance))
        if self.pin_groups_stale(settings_key, rebin_interval):
            self._pin_slots = self._pin_slots_next
            self._pin_slots_indices = view.get('indices')
            self._pin_groups_key = settings_key
            self._pin_groups_age = 0
            with torch.no_grad():
                self._pin_groups = pins_module.compute_coincidence_groups(
                    pins[:, 1].detach(), pins[:, 0].detach(), self._pin_slots, view['component'],
                    self._pin_n_next, pins[:, 2].detach(),
                    view['eps_theta'], view['eps_z'], min_z, max_z,
                    view['local_gap'],
                    coincidence_frac=coincidence_frac,
                    conflict_tolerance=conflict_tolerance,
                    kinds=view['kind'])
            self._pin_rebuilds += 1
        self._pin_groups_age = getattr(self, '_pin_groups_age', 0) + 1
        return pins_module.PinTable(
            pins[:, 0], pins[:, 1], pins[:, 2], pins[:, 3], self._pin_slots,
            view['eps_theta'], view['eps_z'], num_slots, min_z, max_z, dr,
            groups=self._pin_groups, num_registry_pins=view['num_pins'])

    @property
    def pin_conflicts(self):
        return self._pin_groups.conflicts if self._pin_groups is not None else []

    def merged_pin_mask(self):
        """Registry pins that the coincidence pass merged with another pin
        (their rasterised pin is the group mean, so they are not individually
        exact)."""
        registry = self.pin_registry
        if self._pin_groups is None:
            return torch.zeros([registry.num_pins], dtype=torch.bool, device=registry.zyx.device)
        counts = torch.bincount(self._pin_groups.group, minlength=self._pin_groups.num_groups)
        merged = counts[self._pin_groups.group] > 1
        indices = self._pin_view.get('indices') if getattr(self, '_pin_view', None) else None
        if indices is None:
            return merged
        full = torch.zeros([registry.num_pins], dtype=torch.bool, device=registry.zyx.device)
        full[indices] = merged
        return full

    def _pins_enabled(self):
        return self.pins_active and self.pin_registry is not None and self.pin_targets is not None

    def _get_transform_parts(self, truncate_at_step=None, shared=None, with_pins=True):
        truncate_frac = None if truncate_at_step is None else truncate_at_step / (self.flow_integration_steps - 1)
        diffeomorphism = IntegratedFlowDiffeomorphism(self.flow_field, self.flow_min_corner_zyx, self.flow_max_corner_zyx, num_steps=self.flow_integration_steps, solver=self.flow_integration_solver, truncate_at_step=truncate_at_step)
        gap_args = (
            self.gap_expander_params,
            shared[0] if shared is not None else self.get_dr_per_winding(),
            self.flow_min_corner_zyx[0],
            self.flow_max_corner_zyx[0],
            self.cfg['model_gap_expander_lr_scale'],
            self.gap_min_gap,
            self.gap_softplus_bias,
            self.gap_softplus_scale,
            truncate_frac,
        )
        if with_pins and self._pins_enabled():
            # Pins are the fourth shared leaf when the step supplies them;
            # otherwise compute them now from the live parameters.
            if shared is not None and len(shared) > 3 and shared[3] is not None:
                pins = shared[3]
            else:
                pins = self.compute_pins(shared)
            pin_table = self.build_pin_table(
                pins, float(self.cfg.get('model_pin_coincidence_frac', 0.05)),
                float(self.cfg.get('model_pin_conflict_tolerance', 0.1)), dr=gap_args[1],
                rebin_interval=int(self.cfg.get('model_pin_rebin_interval', 1) or 1))
            gap_expander = PinnedGapExpandingTransform(*gap_args, pin_table=pin_table)
        else:
            gap_expander = GapExpandingTransform(*gap_args)
        if shared is not None:
            gap_expander._pinned_scaled_logits = shared[2]
        if self.spiral_outward_sense == 'CW':
            maybe_flip = []
        else:
            assert self.spiral_outward_sense == 'ACW'
            # To make spiral go anticlockwise in slice space (going outwards from the centre), flip it horizontally
            maybe_flip = [pyro.distributions.transforms.AffineTransform(loc=0., scale=torch.tensor([1., 1., -1.], device=self.device))]
        return gap_expander, maybe_flip, diffeomorphism, truncate_frac

    def get_slice_to_spiral_transform(self, truncate_at_step=None, shared=None):
        # `shared` optionally supplies the (dr_per_winding, scaled_linear_logits,
        # pinned_scaled_gap_logits) triple from get_shared_transform_tensors(),
        # typically as detached leaves so many separate loss backwards can run
        # through one transform instance without retain_graph.
        gap_expander, maybe_flip, diffeomorphism, truncate_frac = self._get_transform_parts(truncate_at_step, shared)
        scaled_linear_logits = (
            shared[1] if shared is not None
            else self.linear_logits * self.linear_logits_scale)
        return pyro.distributions.transforms.ComposeTransform([
            gap_expander,
            *maybe_flip,
            diffeomorphism,
            VaryingLinearTransform(scaled_linear_logits, self.flow_min_corner_zyx[0], self.flow_max_corner_zyx[0], truncate_frac),
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
        parts = [gap_expander, *maybe_flip]
        if include_diffeomorphism:
            parts.append(diffeomorphism)
        return pyro.distributions.transforms.ComposeTransform(parts).inv

    def get_dr_per_winding(self):
        return lower_bounded_dr(
            self.dr_per_winding_logit, self.gap_min_gap)

    def smooth_flow_grad_(self, sigma_voxels, across_sigma_voxels=0.0,
                          low_res_sigma_voxels=0.0):
        """Gaussian-smooth the flow lattices' gradients in place.

        ``sigma_voxels`` is the z/around-ring width for cylindrical lattices
        (isotropic for Cartesian), ``across_sigma_voxels`` the across-ring
        width (cylindrical only), and ``low_res_sigma_voxels`` a coarse-lattice
        override for the first width (0 = same as ``sigma_voxels``). Directions
        approximate along/across-sheet directions; see flow_grad_smoothing.
        All widths use scroll-voxel units of the flow frame, not distances
        measured on the deformed sheet. Convert using the nominal fine cell
        width model_flow_voxel_resolution; each field scales for its coarse
        lattice.
        Applies to every flow stage. Call after apply_accumulated_field_grad
        (and after any all-reduce) and before the optimizer step.
        """
        cell_voxels = float(self.cfg['model_flow_voxel_resolution'])
        self.flow_field.smooth_grad_(
            float(sigma_voxels) / cell_voxels,
            float(across_sigma_voxels) / cell_voxels,
            float(low_res_sigma_voxels or 0.0) / cell_voxels)

    def describe_flow_grad_smoothing(self, sigma_voxels, across_sigma_voxels=0.0,
                                     low_res_sigma_voxels=0.0):
        """The effective smoothing widths per lattice, for the startup log."""
        return flow_grad_smoothing.describe_widths(
            sigma_voxels, across_sigma_voxels,
            float(self.cfg['model_flow_voxel_resolution']),
            self.flow_field.spatial_scale_factor,
            self.cfg['model_flow_field_type'],
            low_res_along_voxels=low_res_sigma_voxels)

    def get_shared_transform_tensors(self):
        """The tiny graph paths every evaluation of one transform instance
        shares: the dr-per-winding softplus, the scaled linear logits, and the
        pinned+scaled gap logits (kept in sync with
        GapExpandingTransform._get_pinned_scaled_logits). The training loop
        passes detached leaf copies to get_slice_to_spiral_transform(shared=...)
        so each loss family's backward owns its whole graph and needs no
        retain_graph, then propagates the accumulated leaf gradients through
        these outputs once per step."""
        gap_logits = self.gap_expander_params.logits
        pinned_scaled_gap_logits = torch.cat(
            [torch.zeros_like(gap_logits[..., :1]), gap_logits[..., 1:]], dim=-1,
        ) * self.cfg['model_gap_expander_lr_scale']
        outputs = (
            self.get_dr_per_winding(),
            self.linear_logits * self.linear_logits_scale,
            pinned_scaled_gap_logits,
        )
        if self._pins_enabled():
            # The pins tensor is a fourth shared leaf: each loss family's
            # backward accumulates into it and the step propagates once through
            # compute_pins (flow, linear, dr and T). It must be propagated
            # before the flow field's accumulated gradient is flushed. The
            # training step pushes the sampled subset (sample_count_pins).
            outputs = outputs + (self.compute_pins(subsample=True),)
        return outputs

    def get_native_log_gaps(self, winding_idx, theta, z):
        """Exact log gap for the native lower-bounded gap expander."""
        # The free gap table only; the pinned stage is irrelevant here.
        gap_expander, _, _, truncate_frac = self._get_transform_parts(with_pins=False)
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
