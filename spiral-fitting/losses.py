import numpy as np
import torch
import torch.nn.functional as F

import geom_utils
import prefetch
from dt_targets import patch_dt_target_in_sample_frame
from loss_maps import diagnostics_enabled, record_loss_samples
from sample_spiral import (
    canonical_winding_samples,
    get_radial_covector_in_scroll_space,
    get_theta_and_radii,
    radius_from_unwrapped_shifted,
)
from spiral_helpers import _huber_abs


def _masked_mean(values, mask):
    mask_f = mask.to(values.dtype)
    return (values * mask_f).sum() / mask_f.sum().clamp(min=1.)


_pinned_to_device = geom_utils.pinned_to_device
_cached_scalar_tensor = geom_utils.cached_scalar_tensor


def get_shell_outer_loss(shell_map, slice_to_spiral_transform, dr_per_winding, outer_winding_idx, *, cfg, z_begin, z_end, with_metrics=True):
    # with_metrics=False skips the residual summary block: its data-dependent
    # `valid.any()` branch synchronises the CPU on all queued GPU work, so
    # callers that only log every N steps request metrics on those steps
    # alone.
    device = dr_per_winding.device
    zero = torch.zeros([], device=device)
    if shell_map is None or outer_winding_idx is None:
        return zero, {}

    num_samples = max(1, int(cfg['sample_count_shell_samples']))
    huber_delta = _cached_scalar_tensor(
        cfg['shell_huber_delta'], device, dtype=torch.float32)

    outer_spiral = canonical_winding_samples([outer_winding_idx], num_samples, dr_per_winding, device, z_begin, z_end)[0]
    outer_scan = slice_to_spiral_transform.inv(outer_spiral)

    target_r, scan_r, confidence, valid = shell_map.lookup(outer_scan)
    residual = scan_r - target_r
    shell_outer_loss = _masked_mean(_huber_abs(residual, huber_delta), valid)

    metrics = {}
    if with_metrics:
        with torch.no_grad():
            if valid.any():
                abs_residual = residual[valid].abs()
                metrics = {
                    'shell_outer_error_mean': abs_residual.mean(),
                    'shell_outer_error_p95': torch.quantile(abs_residual, 0.95),
                    'shell_confidence_mean': confidence[valid].mean(),
                }

    return shell_outer_loss, metrics



def _sample_patch_points(patch_indices, cap, rng, patch_atlas):
    """Sample patch quads through the required vc sampling binding."""
    sampling_atlas = getattr(patch_atlas, 'sampling_atlas', None)
    if sampling_atlas is None or not hasattr(sampling_atlas, 'sample_patch_points'):
        raise RuntimeError(
            'Patch sampling requires '
            'vc_spiral.spiral_sampling.PatchSamplingAtlas.sample_patch_points; '
            'rebuild and install the Spiral native extensions')
    seed = int(rng.randint(0, np.iinfo(np.int64).max))
    sampled = sampling_atlas.sample_patch_points(
        np.ascontiguousarray(patch_indices, dtype=np.int64), cap, seed)
    return (
        np.asarray(sampled['ijs'], dtype=np.float32),
        np.asarray(sampled['counts'], dtype=np.int64),
        np.asarray(sampled['node_ordinals'], dtype=np.int64)
        if 'node_ordinals' in sampled else None,
    )




def _aggregate_dt_track_losses(track_losses, across_p):
    # Power-mean across tracks/patches: ((sum x^p) / n)^(1/p); a zero scalar
    # when there are none.
    if track_losses.numel() == 0:
        return torch.zeros([], device=track_losses.device)
    return ((track_losses ** across_p).sum() / track_losses.numel()) ** (1 / across_p)


def _sample_patch_batch(key, patches, sampling_probabilities, num_to_sample,
                        point_cap, cfg, patch_atlas=None, crossing_map=None):
    """Return padded uniform 2D samples and their validity/topology metadata."""
    if num_to_sample <= 0:
        raise ValueError('Expected at least one patch index')

    def build(rng):
        patch_indices = rng.choice(len(patches), num_to_sample,
                                   p=sampling_probabilities, replace=True)
        ijs_np, counts_np, node_ordinals_np = _sample_patch_points(
            patch_indices, point_cap, rng, patch_atlas)
        row_indices = np.broadcast_to(
            np.asarray(patch_indices, dtype=np.int64)[:, None],
            (num_to_sample, point_cap),
        )
        node_ids_np = (
            patch_atlas.theta_node_ids_from_ordinals(node_ordinals_np)
            if (node_ordinals_np is not None
                and hasattr(patch_atlas, 'theta_node_ids_from_ordinals'))
            else patch_atlas.theta_node_ids(row_indices, ijs_np)
        )
        ijs_cpu = torch.from_numpy(ijs_np)
        idx_cpu = torch.from_numpy(
            np.ascontiguousarray(patch_indices, dtype=np.int64))
        node_ids_cpu = torch.from_numpy(
            np.ascontiguousarray(node_ids_np, dtype=np.int64))
        counts_cpu = torch.from_numpy(
            np.ascontiguousarray(counts_np, dtype=np.int64))
        sample_mask_cpu = (
            torch.arange(point_cap)[None, :] < counts_cpu[:, None])
        target_device = patch_atlas.device
        # Upload before the atlas lookup so it receives device tensors and
        # skips its own (pageable, synchronising) transfers.
        ijs_gpu = _pinned_to_device(ijs_cpu, target_device)
        idx_gpu = _pinned_to_device(idx_cpu, target_device)
        node_ids_gpu = _pinned_to_device(node_ids_cpu, target_device)
        sample_mask_gpu = _pinned_to_device(sample_mask_cpu, target_device)
        slice_zyxs_gpu = patch_atlas.lookup(
            idx_gpu[:, None].expand(num_to_sample, point_cap), ijs_gpu)
        return (ijs_gpu, idx_gpu, slice_zyxs_gpu, node_ids_gpu,
                sample_mask_gpu)

    if prefetch.prefetch_enabled() and torch.device(patch_atlas.device).type == 'cuda':
        pf = prefetch.get_prefetcher()
        rng = pf.np_rng(key)
        return pf.pop_or_run((key, id(crossing_map), num_to_sample,
                              point_cap),
                             lambda: build(rng))
    return build(prefetch.LegacyNumpyRandom)


def _sample_patch_tracks(slice_to_spiral_transform, dr_per_winding, patches,
                         patch_atlas, batch, crossing_map, extra_zyxs=None):
    # The bilinear atlas gather already ran on the CPU at batch-build time
    # (see _sample_patch_batch); the batch carries the interpolated points.
    (combined_ijs_gpu, patch_indices_gpu, all_slice_zyxs,
     sample_node_ids, sample_mask) = batch

    # When the caller has extra points (umbilicus, shell, ...), pack them into the same
    # forward ODE call to amortise the per-call overhead.
    patches_flat = all_slice_zyxs.reshape(-1, 3)
    if extra_zyxs is not None:
        combined_spiral = slice_to_spiral_transform(torch.cat([patches_flat, extra_zyxs], dim=0))
        n_patch_pts = patches_flat.shape[0]
        all_spiral_zyxs = combined_spiral[:n_patch_pts].reshape(*all_slice_zyxs.shape)
        extra_spiral = combined_spiral[n_patch_pts:]
    else:
        all_spiral_zyxs = slice_to_spiral_transform(patches_flat).reshape(*all_slice_zyxs.shape)
        extra_spiral = None

    all_theta, _, all_shifted_radii = get_theta_and_radii(all_spiral_zyxs[..., 1:], dr_per_winding)
    all_crossing_adjustments = crossing_map.adjustments_from_potentials(
        sample_node_ids, all_theta, dr_per_winding)
    all_shifted_radii = all_shifted_radii + all_crossing_adjustments

    return (
        combined_ijs_gpu,
        all_slice_zyxs,
        all_spiral_zyxs,
        all_theta,
        all_shifted_radii,
        all_crossing_adjustments,
        sample_mask,
        extra_spiral,
    )




def _patch_radius_and_dt_losses(
    slice_to_spiral_transform, dr_per_winding,
    all_slice_zyxs, all_spiral_zyxs, all_theta, all_shifted_radii,
    all_crossing_adjustments,
    num_patches_for_radius, num_patches_for_dt, compute_dt,
    radius_loss_margin, radius_loss_inv, radius_within_norm_p,
    dt_loss_margin, dt_norm_p, dt_within_patch_norm_p,
    patch_indices=None, sample_ijs=None, dt_target_cache=None, sample_mask=None,
    diagnostic_prefix='patch',
):
    # Shared radius + DT patch losses, operating on padded uniform 2D samples
    # (all_*; see _sample_patch_tracks). Pulled out of get_patch_and_umbilicus_losses so the
    # sampling and loss stages stay separable. Returns (mean_radius_deviation, patch_dt_loss).
    # `dt_target_cache` is the whole-object DT target cache (see dt_targets.py) or None
    # in legacy strip-median mode; `patch_indices` maps the sampled tracks to cache rows.
    radius_hinge_margin = dr_per_winding.detach() * radius_loss_margin
    dt_hinge_margin = dr_per_winding.detach() * dt_loss_margin
    if sample_mask is None:
        sample_mask = torch.ones_like(all_shifted_radii, dtype=torch.bool)

    radius_shifted_radii = all_shifted_radii[:num_patches_for_radius]
    radius_slice_zyxs = all_slice_zyxs[:num_patches_for_radius]
    radius_spiral_zyxs = all_spiral_zyxs[:num_patches_for_radius]
    radius_theta = all_theta[:num_patches_for_radius]
    radius_crossing_adjustments = all_crossing_adjustments[:num_patches_for_radius]
    radius_mask = sample_mask[:num_patches_for_radius]
    radius_counts = radius_mask.sum(dim=-1, keepdim=True).clamp(min=1)
    mean_shifted_radii = (
        (radius_shifted_radii * radius_mask).sum(dim=-1, keepdim=True)
        / radius_counts)
    radius_target_spiral_zyxs = None
    if radius_loss_inv or diagnostics_enabled():
        radius_target_radii = radius_from_unwrapped_shifted(
            radius_theta,
            mean_shifted_radii,
            radius_crossing_adjustments,
            dr_per_winding,
        )
        radius_target_spiral_zyxs = torch.stack([
            radius_spiral_zyxs[..., 0],
            torch.sin(radius_theta) * radius_target_radii,
            torch.cos(radius_theta) * radius_target_radii,
        ], dim=-1).detach()

    if radius_loss_inv:
        # Express the loss in scroll space like the DT loss below: construct target
        # spiral-space points at the track's mean shifted-radius (continuous, not snapped
        # to an integer winding) but with each point's own z and theta, transform back to
        # scroll space, and penalise the distance from the original sampled points.
        radius_target_scroll_zyxs = slice_to_spiral_transform.inv(radius_target_spiral_zyxs.reshape(-1, 3)).reshape(*radius_target_spiral_zyxs.shape)

        radius_point_distances = torch.linalg.norm(radius_slice_zyxs - radius_target_scroll_zyxs, dim=-1)
        radius_point_residuals = F.relu(radius_point_distances - radius_hinge_margin)
        mean_radius_deviation = _masked_mean(
            radius_point_residuals, radius_mask)
        record_loss_samples(
            f'{diagnostic_prefix}_radius', radius_spiral_zyxs,
            radius_point_residuals, radius_mask,
            display_spiral_zyx=radius_target_spiral_zyxs,
        )
    else:
        # Penalise deviation from the track's mean shifted-radius directly in spiral space.
        radius_deviations = (radius_shifted_radii - mean_shifted_radii).abs()
        radius_deviations_hinge = F.relu(radius_deviations - radius_hinge_margin)
        if radius_within_norm_p == 1.0:
            mean_radius_deviation = _masked_mean(
                radius_deviations_hinge, radius_mask)
        else:
            d = radius_deviations_hinge + 1.e-5
            per_track = (
                ((d ** radius_within_norm_p) * radius_mask).sum(dim=-1)
                / radius_counts.squeeze(-1)
            ) ** (1.0 / radius_within_norm_p)
            mean_radius_deviation = per_track.mean()
        record_loss_samples(
            f'{diagnostic_prefix}_radius', radius_spiral_zyxs,
            radius_deviations_hinge, radius_mask,
            display_spiral_zyx=radius_target_spiral_zyxs,
        )

    if compute_dt:
        dt_slice_zyxs = all_slice_zyxs[:num_patches_for_dt]
        dt_spiral_zyxs = all_spiral_zyxs[:num_patches_for_dt]
        dt_theta = all_theta[:num_patches_for_dt]
        dt_shifted_radii = all_shifted_radii[:num_patches_for_dt]
        dt_crossing_adjustments = all_crossing_adjustments[:num_patches_for_dt]
        dt_mask = sample_mask[:num_patches_for_dt]

        # Define the DT target winding (see patch_dt_target_in_sample_frame: whole-patch cached
        # target when available, else the track's own snapped median). Every sampled
        # point on the track is then pulled towards that target winding.
        target_shifted_radii = patch_dt_target_in_sample_frame(
            dt_shifted_radii,
            sample_ijs[:num_patches_for_dt] if sample_ijs is not None else None,
            dt_theta,
            dt_crossing_adjustments,
            dr_per_winding,
            dt_target_cache,
            patch_indices[:num_patches_for_dt] if patch_indices is not None else None,
            sample_mask=dt_mask,
        )
        target_radii = radius_from_unwrapped_shifted(
            dt_theta,
            target_shifted_radii,
            dt_crossing_adjustments,
            dr_per_winding,
        )
        target_spiral_zyxs = torch.stack([
            dt_spiral_zyxs[..., 0],
            torch.sin(dt_theta) * target_radii,
            torch.cos(dt_theta) * target_radii,
        ], dim=-1).detach()

        target_scroll_zyxs = slice_to_spiral_transform.inv(target_spiral_zyxs.reshape(-1, 3)).reshape(*target_spiral_zyxs.shape)

        point_distances = torch.linalg.norm(dt_slice_zyxs - target_scroll_zyxs, dim=-1)
        point_distances = F.relu(point_distances - dt_hinge_margin) + 1.e-5  # epsilon to avoid NaN in p-norm backward
        dt_counts = dt_mask.sum(dim=-1).clamp(min=1)
        track_losses = (
            ((point_distances ** dt_within_patch_norm_p) * dt_mask).sum(dim=-1)
            / dt_counts
        ) ** (1 / dt_within_patch_norm_p)
        patch_dt_loss = _aggregate_dt_track_losses(track_losses, dt_norm_p)
        record_loss_samples(
            f'{diagnostic_prefix}_dt', dt_spiral_zyxs,
            point_distances, dt_mask,
            display_spiral_zyx=target_spiral_zyxs,
        )
    else:
        patch_dt_loss = torch.zeros([], device=dr_per_winding.device)

    return mean_radius_deviation, patch_dt_loss



def get_patch_and_umbilicus_losses(slice_to_spiral_transform, dr_per_winding, num_patches_for_radius, num_patches_for_dt, patches, patch_atlas, patch_sampling_probabilities, umbilicus_zyx, compute_dt=True, dt_target_cache=None, *, crossing_map, cfg):

    extra_zyxs = umbilicus_zyx

    if len(patches) == 0:
        # supervision-free (disable_patches) fits: the umbilicus anchor
        # still applies; the patch radius/DT terms are inert zeros
        extra_spiral = slice_to_spiral_transform(extra_zyxs)
        mean_radius_deviation = torch.zeros([], device=dr_per_winding.device)
        patch_dt_loss = torch.zeros([], device=dr_per_winding.device)
    else:
        # Sample once and share the tracks between the radius and DT losses; the loss using
        # fewer patches takes a prefix of the larger sample.
        num_patches_to_sample = max(num_patches_for_radius, num_patches_for_dt) if compute_dt else num_patches_for_radius
        batch = _sample_patch_batch(
            'verified_patches', patches, patch_sampling_probabilities,
            num_patches_to_sample, cfg['sample_count_points_per_patch'],
            cfg, patch_atlas, crossing_map)

        (
            sample_ijs,
            all_slice_zyxs,
            all_spiral_zyxs,
            all_theta,
            all_shifted_radii,
            all_crossing_adjustments,
            sample_mask,
            extra_spiral,
        ) = _sample_patch_tracks(
            slice_to_spiral_transform,
            dr_per_winding,
            patches,
            patch_atlas,
            batch,
            crossing_map,
            extra_zyxs,
        )

        mean_radius_deviation, patch_dt_loss = _patch_radius_and_dt_losses(
            slice_to_spiral_transform, dr_per_winding,
            all_slice_zyxs, all_spiral_zyxs, all_theta, all_shifted_radii,
            all_crossing_adjustments,
            num_patches_for_radius, num_patches_for_dt, compute_dt,
            cfg['patch_radius_loss_margin'], cfg['patch_radius_loss_inv'], cfg['patch_radius_within_norm_p'],
            cfg['patch_dt_loss_margin'], cfg['patch_dt_norm_p'], cfg['patch_dt_within_patch_norm_p'],
            patch_indices=batch[1], sample_ijs=sample_ijs, dt_target_cache=dt_target_cache,
            sample_mask=sample_mask,
            diagnostic_prefix='patch',
        )

    # Umbilicus should map to the spiral origin (yx ≈ 0)
    umbilicus_loss = extra_spiral[..., 1:].abs().mean()

    return mean_radius_deviation, umbilicus_loss, patch_dt_loss


def _decode_uint8_normal_component(value):
    return (value - 128.0) / 127.0


def _decode_uint8_normal(nx_u8, ny_u8):
    # Decode the uint8 nx/ny normal encoding to a unit zyx direction, plus a mask
    # marking samples that carry a direction at all (both components unset = no data).
    valid = ((nx_u8 != 0) | (ny_u8 != 0)).float()
    nx = _decode_uint8_normal_component(nx_u8.float())
    ny = _decode_uint8_normal_component(ny_u8.float())
    nz = torch.sqrt((1. - nx * nx - ny * ny).clamp(min=0.))
    return F.normalize(torch.stack([nz, ny, nx], dim=-1), dim=-1), valid



def get_radial_normal_in_scroll_space(slice_to_spiral_transform, scroll_zyx, spiral_zyx=None, epsilon=6.0):
    # The normalised scroll-space sheet normal at each point: the spiral-space radial
    # covector pulled back through the transform (sample_spiral.
    # get_radial_covector_in_scroll_space, which documents the J^T transport and
    # the gradient path). Returns (num_points, 3) unit directions in zyx.
    return F.normalize(get_radial_covector_in_scroll_space(
        slice_to_spiral_transform, scroll_zyx, spiral_zyx=spiral_zyx, epsilon=epsilon), dim=-1)



def get_fiber_direction_loss(transform, samples, num_points, epsilon, device):
    """Penalize fiber axes that leave the fitted winding tangent plane."""
    if samples is None or num_points <= 0:
        return torch.zeros([], device=device)
    count = len(samples["position_zyx"])
    # Sampling with replacement avoids constructing a permutation of a
    # potentially multi-million-point host pool every optimizer step.
    indices = np.random.randint(0, count, size=num_points)
    position = torch.as_tensor(samples["position_zyx"][indices], device=device)
    direction, valid = _decode_uint8_normal(
        torch.as_tensor(samples["nx"][indices], device=device),
        torch.as_tensor(samples["ny"][indices], device=device))
    confidence = valid * torch.as_tensor(
        samples["presence"][indices], device=device).float() / 255.0
    normal = get_radial_normal_in_scroll_space(
        transform, position, epsilon=epsilon)
    residual = (normal * direction).sum(dim=-1).square()
    if diagnostics_enabled():
        with torch.no_grad():
            spiral = transform(position)
        record_loss_samples('fiber_directions', spiral, residual)
    return (residual * confidence).sum() / confidence.sum().clamp(min=1e-8)



def sample_spiral_surface_frame(dr_per_winding, outer_winding_idx, num_points, z_begin, z_end):
    # Sample points from discrete spiral windings embedded in spiral yx (over the z-ROI) and return
    # each point's orthonormal in-surface frame in spiral space: e1 = z-axis, e2 = the winding tangent.
    # Winding indices are sampled with probability proportional to their approximate circumference,
    # which is the simple large-radius approximation to uniform area over the wound surface. The inner
    # core is excluded because there is no scroll surface there.
    # Returns (spiral_zyx, e1, e2), each (num_points, 3) in zyx.
    device = dr_per_winding.device
    winding_weights = torch.arange(1, int(outer_winding_idx), device=device, dtype=dr_per_winding.dtype) + 0.5
    winding_idx = torch.multinomial(winding_weights, num_points, replacement=True).to(dr_per_winding.dtype) + 1.0
    theta = torch.rand([num_points], device=device) * (2 * torch.pi)
    radius = (winding_idx + theta / (2 * torch.pi)) * dr_per_winding.detach()
    z = torch.empty([num_points], device=device).uniform_(float(z_begin), float(z_end - 1))
    spiral_zyx = torch.stack([z, torch.sin(theta) * radius, torch.cos(theta) * radius], dim=-1)

    dr_dtheta = dr_per_winding.detach() / (2 * torch.pi)
    tangent_y = torch.cos(theta) * radius + torch.sin(theta) * dr_dtheta
    tangent_x = -torch.sin(theta) * radius + torch.cos(theta) * dr_dtheta
    tangential_yx = F.normalize(torch.stack([tangent_y, tangent_x], dim=-1), dim=-1)
    e1 = F.pad(torch.zeros_like(tangential_yx), (1, 0), value=1.)  # (1, 0, 0) -> z-axis
    e2 = F.pad(tangential_yx, (1, 0), value=0.)  # (0, ty, tx)
    return spiral_zyx, e1, e2



def iter_lasagna_losses(slice_to_spiral_transform, dr_per_winding, lasagna_volume, outer_winding_idx, num_points, epsilon=None, compute_spacing=True, compute_normals=True, *, cfg, z_begin, z_end):
    # Sample points uniformly over the spiral cylinder (a disk of radius
    # dr_per_winding * outer_winding_idx in spiral yx, over the z-ROI). Two losses are computed:
    #   (normals) the spiral radial covector at each sample is pulled back to scroll space via
    #             central-difference J^T (a normal is a covector, not a finite-length displacement)
    #             and matched in direction to the precomputed nx/ny scroll-space normal.
    #   (spacing) [the dense_spacing_mode='grad_mag' objective; the production
    #             mode is 'winding_model' (winding_supervision.py), and
    #             compute_spacing=False skips this entirely] at each sample,
    #             shift inward and outward by dr_per_winding/2
    #             along the spiral radial direction (so the two endpoints span exactly one
    #             winding in spiral space), map both endpoints to scroll space, and
    #             integrate the winding-density field (grad_mag, windings per voxel) along
    #             the scroll-space segment between them. grad_mag is a density, not a
    #             distance, so the number of windings the segment actually crosses is the
    #             line integral of that density along it; for a correct fit the integral
    #             equals 1 (one winding). The density is decoded from grad_mag in windings
    #             per full-resolution voxel.
    device = dr_per_winding.device
    zero = torch.zeros([], device=device)
    if lasagna_volume is None or outer_winding_idx is None:
        if compute_spacing:
            yield 'dense_spacing', zero
        if compute_normals:
            yield 'dense_normals', zero
        return

    backend = lasagna_volume.get('backend', 'dense_test')
    volume = lasagna_volume.get('volume')  # dense: 3 (nx, ny, grad_mag), z, y, x uint8
    z_size, y_size, x_size = lasagna_volume['shape']
    z_origin = lasagna_volume['z_origin']
    y_origin = lasagna_volume.get('y_origin', 0)
    x_origin = lasagna_volume.get('x_origin', 0)
    lasagna_scale = lasagna_volume['lasagna_scale']
    if epsilon is None:
        epsilon = cfg['dense_normals_finite_difference_epsilon']

    dr = dr_per_winding.detach()
    r_max = dr * float(outer_winding_idx)
    r_min = dr  # inner endpoint sits at radius - dr/2 >= dr/2 > 0
    theta = torch.rand([num_points], device=device) * (2 * torch.pi)
    radius = torch.sqrt(torch.rand([num_points], device=device) * (r_max ** 2 - r_min ** 2) + r_min ** 2)
    z = torch.empty([num_points], device=device).uniform_(float(z_begin), float(z_end - 1))
    sin_theta, cos_theta = torch.sin(theta), torch.cos(theta)
    spiral_zyx = torch.stack([z, sin_theta * radius, cos_theta * radius], dim=-1)
    radius_inner = radius - dr / 2
    radius_outer = radius + dr / 2
    spiral_inner = torch.stack([z, sin_theta * radius_inner, cos_theta * radius_inner], dim=-1)
    spiral_outer = torch.stack([z, sin_theta * radius_outer, cos_theta * radius_outer], dim=-1)

    scroll_samples = slice_to_spiral_transform.inv(torch.cat([spiral_inner, spiral_outer, spiral_zyx], dim=0))
    scroll_inner, scroll_outer, scroll_center = scroll_samples.chunk(3, dim=0)
    scroll_displacement = scroll_outer - scroll_inner  # spans exactly one winding in spiral space
    scroll_segment_length = torch.linalg.norm(scroll_displacement, dim=-1).clamp(min=1.e-8)

    # Look up the precomputed scroll-space targets at the midpoint of the displacement (the
    # geometric centre of the one-winding step in scroll space).
    scroll_mid = ((scroll_inner + scroll_outer) / 2).detach()
    sample_zyx = (scroll_mid / lasagna_scale).round().long()
    zi = sample_zyx[:, 0] - z_origin
    yi = sample_zyx[:, 1] - y_origin
    xi = sample_zyx[:, 2] - x_origin
    in_bounds = (zi >= 0) & (zi < z_size) & (yi >= 0) & (yi < y_size) & (xi >= 0) & (xi < x_size)
    zi = zi.clamp(0, z_size - 1)
    yi = yi.clamp(0, y_size - 1)
    xi = xi.clamp(0, x_size - 1)

    # Build both sparse requests before touching the shared CUDA cache.
    if compute_spacing:
        density_decode = cfg['dense_grad_mag_factor'] / cfg['dense_grad_mag_encode_scale'] * lasagna_scale
        num_steps = int(cfg['dense_spacing_integration_steps'])
        step_frac = (torch.arange(num_steps, device=device).float() + 0.5) / num_steps
        integration_zyx = scroll_inner[:, None, :] + step_frac[None, :, None] * scroll_displacement[:, None, :]
        int_idx = (integration_zyx.detach() / lasagna_scale).round().long()
        izi = int_idx[..., 0] - z_origin
        iyi = int_idx[..., 1] - y_origin
        ixi = int_idx[..., 2] - x_origin
        int_in_bounds = (izi >= 0) & (izi < z_size) & (iyi >= 0) & (iyi < y_size) & (ixi >= 0) & (ixi < x_size)
        izi = izi.clamp(0, z_size - 1)
        iyi = iyi.clamp(0, y_size - 1)
        ixi = ixi.clamp(0, x_size - 1)
    else:
        integration_zyx = None

    if backend == 'sparse_cuda':
        normal_indices = (
            torch.stack([zi, yi, xi], dim=-1)
            if compute_normals
            else torch.zeros([0, 3], dtype=torch.int64, device=device)
        )
        if compute_spacing:
            grad_indices = torch.stack([izi, iyi, ixi], dim=-1)
        else:
            grad_indices = torch.zeros([0, 3], dtype=torch.int64, device=device)
        normal_u8, grad_mag_u8 = lasagna_volume['store'].gather_pair(
            normal_indices, grad_indices, device)
        if compute_normals:
            nx_u8, ny_u8 = normal_u8.unbind(dim=-1)
        if compute_spacing:
            grad_mag_u8 = grad_mag_u8.reshape(izi.shape)
    elif backend in ('dense', 'dense_test'):
        if compute_normals:
            nx_u8 = volume[0, zi, yi, xi]
            ny_u8 = volume[1, zi, yi, xi]
        grad_mag_u8 = volume[2, izi, iyi, ixi] if compute_spacing else None
    else:
        raise ValueError(f'unsupported lasagna backend {backend!r}')
    if compute_normals:
        target_normal, valid_normal = _decode_uint8_normal(nx_u8, ny_u8)  # zyx
        normal_weight = valid_normal * in_bounds.float()

    if compute_spacing:
        # grad_mag encodes a winding density (windings per base-volume voxel); the decode factor below
        # also rescales it to current-grid windings/voxel. The number of windings actually crossed by
        # the one-winding scroll-space segment (scroll_inner -> scroll_outer) is the line integral of
        # this density along it, so we sample the density at evenly spaced midpoints along the segment
        # and accumulate density * dl (a midpoint Riemann sum). For a correct fit the integral equals 1.
        sample_valid = (grad_mag_u8 != 0) & int_in_bounds
        density = grad_mag_u8.float() * density_decode  # current-grid windings/voxel
        # dl is the per-step scroll-space length (current-grid voxels); gradient flows through it so the
        # loss can stretch/compress the mapping until the integrated winding count matches.
        dl = scroll_segment_length / num_steps
        integrated_windings = (density * sample_valid.float()).sum(dim=-1) * dl
        # Only score samples whose whole segment lies inside the valid field; a partially covered path
        # would under-integrate and unfairly compare against 1.
        spacing_weight = sample_valid.all(dim=-1).float()
        spacing_residual = (integrated_windings - 1.).abs()
        spacing_loss = (spacing_residual * spacing_weight).sum() / spacing_weight.sum().clamp(min=1)
        record_loss_samples('dense_spacing', spiral_zyx, spacing_residual,
                            spacing_weight.bool())

    scroll_center_detached = scroll_center.detach()
    spiral_zyx_detached = spiral_zyx.detach()
    if compute_spacing:
        yield 'dense_spacing', spacing_loss
        del spacing_loss

    # The caller has released the endpoint/integration graph.  Dense normals
    # use detached sample positions and build their own finite-difference graph,
    # so the two large transform graphs never need to coexist.
    del scroll_samples, scroll_inner, scroll_outer, scroll_center
    del scroll_displacement, scroll_segment_length, integration_zyx
    if not compute_normals:
        return
    scroll_normal = get_radial_normal_in_scroll_space(
        slice_to_spiral_transform,
        scroll_center_detached,
        spiral_zyx=spiral_zyx_detached,
        epsilon=epsilon,
    )
    normals_residual = 1. - (scroll_normal * target_normal).sum(dim=-1).abs()
    normals_loss = (normals_residual * normal_weight).sum() / normal_weight.sum().clamp(min=1)
    record_loss_samples('dense_normals', spiral_zyx_detached, normals_residual,
                        normal_weight.bool())
    yield 'dense_normals', normals_loss


def get_symmetric_dirichlet_loss(slice_to_spiral_transform, dr_per_winding, outer_winding_idx, num_points, epsilon=None, *, cfg, z_begin, z_end):
    # In-surface symmetric Dirichlet energy of the spiral<->scroll map, evaluated at points sampled
    # uniformly over the spiral cylinder (see sample_spiral_surface_frame).
    # At each point we take the orthonormal in-surface frame (e1, e2) in spiral space, map it to scroll
    # space through the inverse transform by finite differences to get its scroll-space image (a, b), and
    # form the 2x2 induced metric G = [[a.a, a.b], [a.b, b.b]]. The energy ||J||_F^2 + ||J^{-1}||_F^2 =
    # tr(G) + tr(G^{-1}) = (s1^2 + s2^2) + (1/s1^2 + 1/s2^2) is minimised (value 4) at an in-surface
    # isometry and diverges as the map degenerates (singular value -> 0 or inf), acting as a barrier
    # against in-surface collapse / element flips. We subtract 4 so the reported value is 0 at rest.
    device = dr_per_winding.device
    if outer_winding_idx is None:
        return torch.zeros([], device=device)
    if epsilon is None:
        epsilon = cfg['model_sym_dirichlet_finite_difference_epsilon']

    spiral_zyx, e1, e2 = sample_spiral_surface_frame(dr_per_winding, outer_winding_idx, num_points, z_begin, z_end)

    spiral_shift_1 = spiral_zyx + e1 * epsilon
    spiral_shift_2 = spiral_zyx + e2 * epsilon
    combined_spiral = torch.cat([spiral_zyx, spiral_shift_1, spiral_shift_2], dim=0)
    combined_scroll = slice_to_spiral_transform.inv(combined_spiral)
    scroll_zyx, scroll_shift_1, scroll_shift_2 = combined_scroll.chunk(3, dim=0)

    a = (scroll_shift_1 - scroll_zyx) / epsilon
    b = (scroll_shift_2 - scroll_zyx) / epsilon
    g11 = (a * a).sum(dim=-1)
    g22 = (b * b).sum(dim=-1)
    g12 = (a * b).sum(dim=-1)
    trace_g = g11 + g22
    det_g = g11 * g22 - g12 * g12
    # Energy is tr(G) + tr(G^{-1}) = (s1^2 + s2^2) + (1/s1^2 + 1/s2^2), regularised per-eigenvalue so a
    # vanishing singular value contributes a finite-but-large 1/(lambda+eps) barrier. We compute the
    # regularised inverse-eigenvalue sum directly from trace_g, det_g via the algebraic identity
    #   1/(l1+eps) + 1/(l2+eps) = ((l1+eps) + (l2+eps)) / ((l1+eps)(l2+eps))
    #                           = (trace_g + 2*eps) / (det_g + eps*trace_g + eps**2)
    inverse_eps = 1e-3
    inverse_term = (trace_g + 2.0 * inverse_eps) / (det_g + inverse_eps * trace_g + inverse_eps ** 2)
    energy = (trace_g + inverse_term - 4.0).clamp(min=0.0)
    # Per-sample cap so a single near-degenerate sample doesn't dominate the batch mean / gradient.
    energy = energy.clamp(max=1.e2)
    record_loss_samples('sym_dirichlet', spiral_zyx, energy)
    return energy.mean()


def _sample_windings_by_circumference(
    lowest, highest, num_samples, device, generator=None,
):
    """Integer windings in [lowest, highest] with probability proportional to
    winding circumference (weight k + 0.5). ``highest`` may be a per-sample
    tensor. Uses the analytic inverse CDF: cum-mass to k is ((k+1)^2 - a^2)/2."""
    a = float(lowest)
    if torch.is_tensor(highest):
        b = highest.to(device=device, dtype=torch.float32)
    else:
        # A pageable scalar upload would stall on the whole GPU queue.
        b = _cached_scalar_tensor(highest, device)
    total_mass = ((b + 1.0) ** 2 - a * a) / 2.0
    u = torch.rand(num_samples, device=device, generator=generator)
    k = torch.ceil(torch.sqrt(2.0 * u * total_mass + a * a) - 1.0)
    return torch.minimum(k.clamp(min=a), b)


def get_min_spacing_loss(
    spiral_and_transform, outer_winding_idx, cfg, z_begin, z_end, *,
    generator=None, with_metrics=True,
):
    """Squared hinge on sampled native log gaps before exponentiation: the
    asset-independent exact-collapse recovery barrier.

    Every float()/.item() in the metrics dict is a device synchronization,
    so callers that log on a cadence pass ``with_metrics=False`` on the other
    steps; the loss and its gradient are identical either way.
    """
    device = spiral_and_transform.device
    zero = torch.zeros([], device=device)
    if outer_winding_idx is None:
        return zero, {}
    num_samples = int(cfg['sample_count_minimum_spacing_independent_samples'])
    # Gaps are indexed by their inner winding; winding 1 is the innermost by
    # convention, so the sampled gaps are [1, outer_winding_idx - 2].
    inner, outer = 1, int(outer_winding_idx) - 1
    if outer <= inner or num_samples <= 0:
        return zero, {}
    winding = _sample_windings_by_circumference(
        inner, outer - 1, num_samples, device, generator=generator).long()
    theta = torch.rand(
        num_samples, device=device, generator=generator) * (2 * np.pi)
    z = torch.empty(num_samples, device=device).uniform_(
        float(z_begin), float(z_end - 1), generator=generator)
    ell_gap = spiral_and_transform.get_native_log_gaps(winding, theta, z)
    ell_min = float(np.log(float(cfg['dense_min_spacing_d_min_wv'])))
    deficiency = F.relu(ell_min - ell_gap)
    loss = deficiency.square().mean()
    if not with_metrics:
        return loss, {}
    with torch.no_grad():
        gap = torch.exp(ell_gap)
        q = torch.quantile(gap, torch.tensor([0.1, 0.5, 0.9], device=device))
        active = deficiency > 0
        metrics = {
            'min_spacing_active_fraction': float(active.float().mean().item()),
            'min_spacing_gap_p10': float(q[0].item()),
            'min_spacing_gap_p50': float(q[1].item()),
            'min_spacing_gap_p90': float(q[2].item()),
            'min_spacing_ell_gap_mean': float(ell_gap.mean().item()),
            'min_spacing_violation_depth_mean': float(
                deficiency[active].mean().item()) if active.any() else 0.0,
        }
    return loss, metrics
