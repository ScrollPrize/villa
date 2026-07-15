"""Localized influence regions for interactive (ephemeral) inputs.

When enabled, each input added to a resident interactive session defines a
region of spiral space -- the anisotropic dilation of the input's actual point
footprint by user-set maxima along z (voxels), winding (across wraps), and
theta (along the wrap, circular) -- inside which the fit may move, with the
ability to move decaying gaussian-like towards the region boundary. Everything
outside the union of all such regions is held in place by two mechanisms:

1. Gradient masking: each spatially-addressable parameter grid element
   (flow-field voxels, gap-expander logits) has its gradient scaled by the
   influence weight at its spiral-space location; fully-outside elements are
   exactly frozen (grad, Adam momentum, and weight decay all zeroed, so the
   AdamW update is bitwise zero). The globally-acting parameters
   (linear_logits, dr_per_winding_logit) are frozen entirely.
2. Anchor loss: a fixed bank of scroll-space points has its spiral-space
   images cached at each incorporation; a per-step loss penalizes displacement
   from those targets, weighted by (1 - w) ** ramp_power so it ramps up
   rapidly outside the influence region.

Regions accumulate across incorporations as a pointwise max (union) of the
per-input weight fields; anchor targets are refreshed to the current state at
each incorporation.
"""

import math
import time

import numpy as np
import torch

from sample_spiral import get_theta_and_radii


def spiral_zst(spiral_zyx, dr_per_winding):
    """Spiral-space zyx -> (z, s, theta): s is the continuous winding
    coordinate (shifted radius in winding units), theta in [0, 2pi)."""
    theta, _, shifted_radius = get_theta_and_radii(spiral_zyx[..., 1:], dr_per_winding)
    s = shifted_radius / dr_per_winding
    return torch.stack([spiral_zyx[..., 0], s, theta], dim=-1)


def influence_weight(query_zst, footprint_zst, limits, sigma, chunk_elements=2 ** 27):
    """Influence weight in [0, 1] for each query against a footprint point set.

    d(q)^2 = min_f ((z_q-z_f)/L_z)^2 + ((s_q-s_f)/L_k)^2 + (dtheta_circ/L_t)^2
    w(q)   = exp(-d(q)^2 / (2 sigma^2)) where d(q) < 1, else 0 (hard cutoff).

    limits = (L_z, L_k, L_theta) with L_theta in radians; theta distance is
    circular. sigma is in normalized (cutoff-relative) units.
    """
    limit_z, limit_k, limit_theta = (float(v) for v in limits)
    num_queries = query_zst.shape[0]
    weights = torch.zeros([num_queries], dtype=torch.float32, device=query_zst.device)
    if footprint_zst.shape[0] == 0 or num_queries == 0:
        return weights
    footprint_zst = footprint_zst.to(query_zst.device, torch.float32)

    # Pre-cull: outside the footprint's z/s bounding box dilated by the hard
    # extents, the weight is exactly zero (theta is circular, so no theta cull).
    z_lo = footprint_zst[:, 0].min() - limit_z
    z_hi = footprint_zst[:, 0].max() + limit_z
    s_lo = footprint_zst[:, 1].min() - limit_k
    s_hi = footprint_zst[:, 1].max() + limit_k
    candidate = (
        (query_zst[:, 0] >= z_lo) & (query_zst[:, 0] <= z_hi)
        & (query_zst[:, 1] >= s_lo) & (query_zst[:, 1] <= s_hi)
    )
    candidate_idx = candidate.nonzero(as_tuple=True)[0]
    if candidate_idx.numel() == 0:
        return weights

    fz = footprint_zst[:, 0] / limit_z
    fs = footprint_zst[:, 1] / limit_k
    ftheta = footprint_zst[:, 2]
    chunk = max(1024, chunk_elements // footprint_zst.shape[0])
    two_pi = 2. * math.pi
    for start in range(0, candidate_idx.numel(), chunk):
        idx = candidate_idx[start:start + chunk]
        q = query_zst[idx].to(torch.float32)
        d2 = (q[:, 0, None] / limit_z - fz[None]) ** 2
        d2 += (q[:, 1, None] / limit_k - fs[None]) ** 2
        dtheta = (q[:, 2, None] - ftheta[None]).abs() % two_pi
        dtheta = torch.minimum(dtheta, two_pi - dtheta)
        d2 += (dtheta / limit_theta) ** 2
        d2_min = d2.min(dim=1).values
        w = torch.exp(-d2_min / (2. * sigma ** 2))
        weights[idx] = torch.where(d2_min < 1., w, torch.zeros_like(w))
    return weights


def subsample_rows(points, max_points, generator):
    if points.shape[0] <= max_points:
        return points
    perm = torch.randperm(points.shape[0], generator=generator)
    return points[perm[:max_points]]


def _apply_transform_chunked(transform, points, chunk=1_000_000):
    outputs = [transform(points[start:start + chunk]) for start in range(0, points.shape[0], chunk)]
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)


def _gap_logit_zst(gap_params, min_z, max_z, device):
    """Spiral-space (z, s, theta) of every gap-expander logit, as row (z) and
    column (s, theta) coordinate vectors: returns (z_rows [num_z],
    s_cols [total], theta_cols [total])."""
    num_z = gap_params.num_z
    first_idx = gap_params.winding_first_logit_idx.to(device, torch.float32)
    num_by_winding = gap_params.num_by_winding.to(device, torch.float32)
    total = int(round(float(first_idx[-1])))
    z_rows = torch.linspace(float(min_z), float(max_z), num_z, device=device)
    # grid_sample uses winding_coords_normalised = wc / total * 2 - 1 with
    # align_corners=True over `total` columns, so column x is sampled at
    # wc = x * total / (total - 1) (transforms.py get_transformed_winding_radii).
    wc = torch.arange(total, device=device, dtype=torch.float32) * (total / (total - 1))
    k = torch.searchsorted(first_idx, wc, right=True) - 1
    k = k.clamp(0, num_by_winding.shape[0] - 1)
    theta_cols = ((wc - first_idx[k]) / num_by_winding[k] * (2. * math.pi)) % (2. * math.pi)
    # Logit column in bucket k adjusts the gap between windings k and k+1.
    s_cols = k.to(torch.float32) + 0.5
    return z_rows, s_cols, theta_cols


class InteractiveInfluenceState:
    """Union of influence regions plus the machinery enforcing them.

    Lives on the fitter thread; all tensors on the fit device. Persisted in
    checkpoints via state_dict()/from_state_dict() -- masks cannot be
    regenerated later (they were evaluated against transforms at past
    incorporation times), so they are stored in full (fp16).
    """

    def __init__(self, limits, sigma, ramp_power, device):
        self.limits = tuple(float(v) for v in limits)  # (L_z, L_k, L_theta)
        self.sigma = float(sigma)
        self.ramp_power = float(ramp_power)
        self.device = device
        self.masks = {}  # 'flow_lr'/'flow_hr' [Z,Y,X] fp16, 'gap' [num_z,total] fp16
        self.footprints = []  # per input: {'input_id', 'kind', 'zst' cpu fp32} (diagnostics)
        self.anchor_scroll = None  # [A,3] fp32
        self.anchor_target = None  # [A,3] fp32, spiral-space
        self.anchor_w = None  # [A] fp32, accumulated influence weight
        self.anchor_loss_weight = None  # [A] fp32, (1-w)**ramp_power
        self.saved_gap_weight_decay = None
        self.num_incorporations = 0

    @property
    def active(self):
        return self.num_incorporations > 0

    # ---------------------------------------------------------------- setup

    def _generator(self, cfg):
        generator = torch.Generator()
        generator.manual_seed(int(cfg['random_seed']) + self.num_incorporations)
        return generator

    @torch.no_grad()
    def _build_footprint(self, new_patches, new_collections, slice_to_spiral_transform,
                         dr_per_winding, cfg, z_begin, z_end, generator):
        """Per-input scroll-space point sets -> pooled spiral (z, s, theta)."""
        per_input = []
        max_points = int(cfg['interactive_influence_footprint_points'])
        for input_id, patch in new_patches.items():
            zyx = patch.zyxs[patch.valid_vertex_mask].reshape(-1, 3).to(torch.float32)
            per_input.append((input_id, 'patch', zyx))
        for input_id, pcl in new_collections.items():
            zyx = torch.from_numpy(np.stack(
                [point['zyx'] for point in pcl['points'].values()], axis=0)).to(torch.float32)
            per_input.append((str(input_id), 'pcl', zyx))
        footprint_parts = []
        for input_id, kind, zyx in per_input:
            # copy=True: the subsample may alias the input's own tensor, and the
            # z-clamp below must not mutate it.
            zyx = subsample_rows(zyx, max_points, generator).to(self.device, copy=True)
            zyx[:, 0] = zyx[:, 0].clamp(float(z_begin), float(z_end - 1))
            zst = spiral_zst(_apply_transform_chunked(slice_to_spiral_transform, zyx),
                             dr_per_winding).to(torch.float32)
            self.footprints.append({'input_id': input_id, 'kind': kind, 'zst': zst.cpu()})
            footprint_parts.append(zst)
        return torch.cat(footprint_parts, dim=0) if footprint_parts else \
            torch.empty([0, 3], dtype=torch.float32, device=self.device)

    def _find_param_group(self, optimiser, param):
        for group in optimiser.param_groups:
            if any(p is param for p in group['params']):
                return group
        raise RuntimeError('parameter not found in any optimizer param group')

    @torch.no_grad()
    def _allocate_masks(self, spiral_and_transform):
        flow_field = spiral_and_transform.flow_field
        if not hasattr(flow_field, 'flows') or len(flow_field.flows) != 2:
            raise RuntimeError(
                'interactive influence regions require the cartesian flow field '
                '(low-res + high-res lattices)')
        lr_flow, hr_flow = flow_field.flows
        gap_logits = spiral_and_transform.gap_expander_params.logits
        self.masks = {
            'flow_lr': torch.zeros(lr_flow.shape[2:], dtype=torch.float16, device=self.device),
            'flow_hr': torch.zeros(hr_flow.shape[2:], dtype=torch.float16, device=self.device),
            'gap': torch.zeros(gap_logits.shape[2:], dtype=torch.float16, device=self.device),
        }

    # ------------------------------------------------------- mask evaluation

    @torch.no_grad()
    def _update_gap_mask(self, spiral_and_transform, footprint_zst):
        gap_params = spiral_and_transform.gap_expander_params
        z_rows, s_cols, theta_cols = _gap_logit_zst(
            gap_params,
            spiral_and_transform.flow_min_corner_zyx[0],
            spiral_and_transform.flow_max_corner_zyx[0],
            self.device,
        )
        limit_z = self.limits[0]
        row_active = (
            (z_rows >= footprint_zst[:, 0].min() - limit_z)
            & (z_rows <= footprint_zst[:, 0].max() + limit_z)
        ).nonzero(as_tuple=True)[0]
        mask = self.masks['gap']
        total = s_cols.shape[0]
        rows_per_chunk = max(1, 2_000_000 // total)
        for start in range(0, row_active.numel(), rows_per_chunk):
            rows = row_active[start:start + rows_per_chunk]
            query = torch.stack([
                z_rows[rows][:, None].expand(-1, total),
                s_cols[None].expand(rows.numel(), -1),
                theta_cols[None].expand(rows.numel(), -1),
            ], dim=-1).reshape(-1, 3)
            w = influence_weight(query, footprint_zst, self.limits, self.sigma)
            mask[rows] = torch.maximum(mask[rows], w.view(rows.numel(), total).to(torch.float16))

    @torch.no_grad()
    def _flow_z_displacement_bound(self, spiral_and_transform):
        flow_field = spiral_and_transform.flow_field
        lr_flow, hr_flow = flow_field.flows
        v_max = (lr_flow[:, 0].abs().max() * float(flow_field.flow_scales[0])
                 + hr_flow[:, 0].abs().max() * float(flow_field.flow_scales[1]))
        z_range = float(spiral_and_transform.flow_max_corner_zyx[0]
                        - spiral_and_transform.flow_min_corner_zyx[0])
        return float(v_max) * z_range

    @torch.no_grad()
    def _update_flow_mask(self, spiral_and_transform, mask_key, footprint_zst, dr_per_winding):
        mask = self.masks[mask_key]
        min_corner = spiral_and_transform.flow_min_corner_zyx.to(torch.float32)
        max_corner = spiral_and_transform.flow_max_corner_zyx.to(torch.float32)
        shape = mask.shape
        axis_coords = [
            torch.linspace(float(min_corner[axis]), float(max_corner[axis]), shape[axis],
                           device=self.device)
            for axis in range(3)
        ]
        # A lattice position brackets the material coordinates it influences
        # between the integration-trajectory start (no diffeomorphism) and end
        # (with); evaluate both and take the max weight.
        transforms = [
            spiral_and_transform.get_flowbox_to_spiral_transform(include_diffeomorphism=False),
            spiral_and_transform.get_flowbox_to_spiral_transform(include_diffeomorphism=True),
        ]
        limit_z = self.limits[0]
        z_safety = self._flow_z_displacement_bound(spiral_and_transform)
        z_lo = footprint_zst[:, 0].min() - limit_z - z_safety
        z_hi = footprint_zst[:, 0].max() + limit_z + z_safety
        row_active = ((axis_coords[0] >= z_lo) & (axis_coords[0] <= z_hi)).nonzero(as_tuple=True)[0]
        row_points = shape[1] * shape[2]
        rows_per_chunk = max(1, 2_000_000 // row_points)
        for start in range(0, row_active.numel(), rows_per_chunk):
            rows = row_active[start:start + rows_per_chunk]
            lattice = torch.stack(torch.meshgrid(
                axis_coords[0][rows], axis_coords[1], axis_coords[2], indexing='ij',
            ), dim=-1).reshape(-1, 3)
            w = None
            for transform in transforms:
                zst = spiral_zst(transform(lattice), dr_per_winding)
                w_variant = influence_weight(zst, footprint_zst, self.limits, self.sigma)
                w = w_variant if w is None else torch.maximum(w, w_variant)
            w = w.view(rows.numel(), shape[1], shape[2]).to(torch.float16)
            mask[rows] = torch.maximum(mask[rows], w)

    # ------------------------------------------------------------ anchor bank

    @torch.no_grad()
    def _build_anchor_bank(self, spiral_and_transform, slice_to_spiral_transform,
                           dr_per_winding, anchor_geometry_zyx, cfg, z_begin, z_end, generator):
        parts = []
        if anchor_geometry_zyx is not None and anchor_geometry_zyx.shape[0] > 0:
            geometry = subsample_rows(
                anchor_geometry_zyx.to(torch.float32).cpu(),
                int(cfg['interactive_influence_anchor_geometry_points']), generator)
            parts.append(geometry.to(self.device))
        num_lattice = int(cfg['interactive_influence_anchor_lattice_points'])
        outer_winding = int(cfg['shell_outer_winding_idx'])
        if num_lattice > 0 and outer_winding > 1:
            winding_indices = torch.arange(1, outer_winding, dtype=torch.float32)
            per_winding = max(1, math.ceil(num_lattice / winding_indices.shape[0]))
            theta = torch.rand([winding_indices.shape[0], per_winding],
                               generator=generator) * (2. * math.pi)
            z = torch.rand([winding_indices.shape[0], per_winding],
                           generator=generator) * float(z_end - 1 - z_begin) + float(z_begin)
            theta, z = theta.to(self.device), z.to(self.device)
            radius = (winding_indices.to(self.device)[:, None]
                      + theta / (2. * math.pi)) * dr_per_winding
            lattice_spiral = torch.stack(
                [z, torch.sin(theta) * radius, torch.cos(theta) * radius], dim=-1).reshape(-1, 3)
            parts.append(_apply_transform_chunked(slice_to_spiral_transform.inv, lattice_spiral))
        if not parts:
            raise RuntimeError('no anchor points available for the influence anchor loss')
        self.anchor_scroll = torch.cat(parts, dim=0).to(torch.float32)
        self.anchor_w = torch.zeros([self.anchor_scroll.shape[0]],
                                    dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------ public API

    @torch.no_grad()
    def activate_or_extend_(self, *, new_patches, new_collections, spiral_and_transform,
                            optimiser, cfg, z_begin, z_end, anchor_geometry_zyx):
        """Union the new inputs' influence regions in; runs at a pause boundary."""
        if not new_patches and not new_collections:
            return
        started = time.monotonic()
        slice_to_spiral_transform = spiral_and_transform.get_slice_to_spiral_transform()
        dr_per_winding = spiral_and_transform.get_dr_per_winding()
        generator = self._generator(cfg)
        first_activation = not self.active
        if first_activation:
            self._allocate_masks(spiral_and_transform)
            self._build_anchor_bank(spiral_and_transform, slice_to_spiral_transform,
                                    dr_per_winding, anchor_geometry_zyx, cfg,
                                    z_begin, z_end, generator)
        footprint_zst = self._build_footprint(
            new_patches, new_collections, slice_to_spiral_transform, dr_per_winding,
            cfg, z_begin, z_end, generator)
        if footprint_zst.shape[0] == 0:
            raise RuntimeError('influence footprint is empty for the incorporated inputs')

        self._update_gap_mask(spiral_and_transform, footprint_zst)
        self._update_flow_mask(spiral_and_transform, 'flow_lr', footprint_zst, dr_per_winding)
        self._update_flow_mask(spiral_and_transform, 'flow_hr', footprint_zst, dr_per_winding)

        # Refresh anchor targets to the current state and union the new
        # region's weight at each anchor.
        anchor_spiral = _apply_transform_chunked(slice_to_spiral_transform, self.anchor_scroll)
        self.anchor_target = anchor_spiral.to(torch.float32)
        anchor_zst = spiral_zst(anchor_spiral, dr_per_winding).to(torch.float32)
        w_new = influence_weight(anchor_zst, footprint_zst, self.limits, self.sigma)
        self.anchor_w = torch.maximum(self.anchor_w, w_new)
        self.anchor_loss_weight = (1. - self.anchor_w).clamp(0., 1.) ** self.ramp_power

        self._apply_optimizer_surgery(spiral_and_transform, optimiser)
        self.num_incorporations += 1
        print(f'influence region updated in {time.monotonic() - started:.1f}s: '
              f'{footprint_zst.shape[0]} footprint points, mask coverage '
              f'gap {float((self.masks["gap"] > 0).float().mean()):.4f} / '
              f'flow_hr {float((self.masks["flow_hr"] > 0).float().mean()):.4f}, '
              f'anchors held: {int((self.anchor_loss_weight > 0.5).sum())}/{self.anchor_w.shape[0]}')

    @torch.no_grad()
    def _apply_optimizer_surgery(self, spiral_and_transform, optimiser):
        # Adam momentum would keep moving masked-out elements after their
        # gradients are zeroed; scale it by the mask (zero where fully masked).
        lr_flow, hr_flow = spiral_and_transform.flow_field.flows
        gap_logits = spiral_and_transform.gap_expander_params.logits
        for param, mask in ((lr_flow, self.masks['flow_lr']),
                            (hr_flow, self.masks['flow_hr']),
                            (gap_logits, self.masks['gap'])):
            state = optimiser.state.get(param)
            if state and 'exp_avg' in state:
                state['exp_avg'].mul_(mask)
        for param in (spiral_and_transform.linear_logits,
                      spiral_and_transform.dr_per_winding_logit):
            state = optimiser.state.get(param)
            if state and 'exp_avg' in state:
                state['exp_avg'].zero_()
        # Decoupled AdamW weight decay also moves parameters regardless of
        # gradient; disable it on the gap group and re-emulate it inside the
        # region in apply_masked_gap_decay_.
        gap_group = self._find_param_group(optimiser, gap_logits)
        if self.saved_gap_weight_decay is None:
            self.saved_gap_weight_decay = float(gap_group['weight_decay'])
        gap_group['weight_decay'] = 0.0

    def reapply_optimizer_overrides_(self, spiral_and_transform, optimiser):
        """After resuming a checkpoint with an active influence state, the
        freshly-constructed optimizer has the default gap weight decay again."""
        if not self.active:
            return
        gap_group = self._find_param_group(optimiser, spiral_and_transform.gap_expander_params.logits)
        if self.saved_gap_weight_decay is None:
            self.saved_gap_weight_decay = float(gap_group['weight_decay'])
        gap_group['weight_decay'] = 0.0

    @torch.no_grad()
    def apply_grad_masks_(self, spiral_and_transform):
        lr_flow, hr_flow = spiral_and_transform.flow_field.flows
        gap_logits = spiral_and_transform.gap_expander_params.logits
        for param, mask in ((lr_flow, self.masks['flow_lr']),
                            (hr_flow, self.masks['flow_hr']),
                            (gap_logits, self.masks['gap'])):
            if param.grad is not None:
                param.grad.mul_(mask)
        for param in (spiral_and_transform.linear_logits,
                      spiral_and_transform.dr_per_winding_logit):
            if param.grad is not None:
                param.grad.zero_()

    @torch.no_grad()
    def apply_masked_gap_decay_(self, spiral_and_transform, optimiser):
        """Emulate the disabled decoupled weight decay, restricted to the region."""
        if not self.saved_gap_weight_decay:
            return
        gap_logits = spiral_and_transform.gap_expander_params.logits
        gap_group = self._find_param_group(optimiser, gap_logits)
        step_size = float(gap_group['lr']) * self.saved_gap_weight_decay
        gap_logits.data.mul_(1. - step_size * self.masks['gap'].to(torch.float32))

    def get_anchor_loss(self, slice_to_spiral_transform, dr_per_winding, num_samples):
        idx = torch.randint(self.anchor_scroll.shape[0], [num_samples],
                            device=self.anchor_scroll.device)
        weight = self.anchor_loss_weight[idx]
        current = slice_to_spiral_transform(self.anchor_scroll[idx])
        displacement = torch.linalg.norm(current - self.anchor_target[idx], dim=-1) / dr_per_winding
        return (weight * displacement).sum() / weight.sum().clamp(min=1.)

    # ---------------------------------------------------------- persistence

    def state_dict(self):
        return {
            'limits': list(self.limits),
            'sigma': self.sigma,
            'ramp_power': self.ramp_power,
            'masks': {key: mask.cpu() for key, mask in self.masks.items()},
            'footprints': [{'input_id': f['input_id'], 'kind': f['kind'], 'zst': f['zst']}
                           for f in self.footprints],
            'anchor_scroll': self.anchor_scroll.cpu() if self.anchor_scroll is not None else None,
            'anchor_target': self.anchor_target.cpu() if self.anchor_target is not None else None,
            'anchor_w': self.anchor_w.cpu() if self.anchor_w is not None else None,
            'saved_gap_weight_decay': self.saved_gap_weight_decay,
            'num_incorporations': self.num_incorporations,
        }

    @classmethod
    def from_state_dict(cls, payload, device):
        if payload is None:
            return None
        state = cls(payload['limits'], payload['sigma'], payload['ramp_power'], device)
        state.masks = {key: mask.to(device) for key, mask in payload['masks'].items()}
        state.footprints = list(payload['footprints'])
        for key in ('anchor_scroll', 'anchor_target', 'anchor_w'):
            value = payload[key]
            setattr(state, key, value.to(device) if value is not None else None)
        state.saved_gap_weight_decay = payload['saved_gap_weight_decay']
        state.num_incorporations = int(payload['num_incorporations'])
        if state.anchor_w is not None:
            state.anchor_loss_weight = (1. - state.anchor_w).clamp(0., 1.) ** state.ramp_power
        return state

    def assert_matches_model(self, spiral_and_transform):
        lr_flow, hr_flow = spiral_and_transform.flow_field.flows
        expected = {
            'flow_lr': tuple(lr_flow.shape[2:]),
            'flow_hr': tuple(hr_flow.shape[2:]),
            'gap': tuple(spiral_and_transform.gap_expander_params.logits.shape[2:]),
        }
        for key, shape in expected.items():
            if tuple(self.masks[key].shape) != shape:
                raise RuntimeError(
                    f'restored influence mask {key!r} has shape {tuple(self.masks[key].shape)}, '
                    f'expected {shape}; the checkpoint was created with different model-shaping config')


def make_influence_state(cfg, device):
    limits = (
        float(cfg['interactive_influence_z']),
        float(cfg['interactive_influence_windings']),
        float(cfg['interactive_influence_theta_frac']) * 2. * math.pi,
    )
    return InteractiveInfluenceState(
        limits=limits,
        sigma=float(cfg['interactive_influence_sigma']),
        ramp_power=float(cfg['interactive_influence_anchor_ramp_power']),
        device=device,
    )
