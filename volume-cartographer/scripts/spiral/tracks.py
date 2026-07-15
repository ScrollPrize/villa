import colorsys
import dbm
import pickle

import kornia
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
from tqdm import tqdm

import prefetch
import geom_utils
from sample_spiral import get_theta_and_radii


def load_tracks_from_dbm(path, z_lo, z_hi):
    # Load tracks written by extract_surface_tracks.py. Each DBM value is a
    # pickled list of (N, 3) int32 zyx arrays; keep only tracks that lie entirely
    # within the full-resolution [z_lo, z_hi) ROI.
    tracks = []
    with dbm.open(path, 'r') as db:
        for key in tqdm(db.keys(), desc='loading tracks'):
            entries = pickle.loads(db[key])
            if not entries:
                continue
            # Vectorize the per-track z min/max across the whole key: concatenate
            # every non-empty track's z column and reduce per segment, rather
            # than calling .min()/.max() once per track.
            idx = [i for i in range(len(entries)) if len(entries[i])]
            if not idx:
                continue
            lengths = np.fromiter((len(entries[i]) for i in idx), dtype=np.intp, count=len(idx))
            zcat = np.concatenate([entries[i][:, 0] for i in idx])
            offsets = np.zeros(len(idx), dtype=np.intp)
            np.cumsum(lengths[:-1], out=offsets[1:])
            zmins = np.minimum.reduceat(zcat, offsets)
            zmaxs = np.maximum.reduceat(zcat, offsets)
            keep = (zmins >= z_lo) & (zmaxs < z_hi)
            for j in np.nonzero(keep)[0]:
                tracks.append(entries[idx[j]].astype(np.float32))
    return tracks


@geom_utils.maybe_compile
def _unwrap_track_shifted_radii(theta, shifted_radii, dr_per_winding):
    if theta.shape[-1] <= 1:
        return shifted_radii

    theta_diffs = theta.detach()[..., 1:] - theta.detach()[..., :-1]
    step_adjustments = (
        (theta_diffs > np.pi).to(shifted_radii.dtype)
        - (theta_diffs < -np.pi).to(shifted_radii.dtype)
    ) * dr_per_winding.detach()
    adjustments = torch.cat([
        torch.zeros([*theta.shape[:-1], 1], device=shifted_radii.device, dtype=shifted_radii.dtype),
        torch.cumsum(step_adjustments, dim=-1),
    ], dim=-1)
    return shifted_radii + adjustments


def _aggregate_dt_track_losses(track_losses, across_p, active_mask=None):
    if active_mask is not None:
        track_losses = track_losses[active_mask]
    if track_losses.numel() == 0:
        return torch.zeros([], device=track_losses.device)
    return ((track_losses ** across_p).sum() / track_losses.numel()) ** (1 / across_p)


def _progressive_dt_active_mask(snapped_winding, dr_per_winding, dt_max_winding):
    if dt_max_winding is None:
        return None
    winding_idx = (snapped_winding / dr_per_winding).detach()
    return winding_idx <= dt_max_winding


def _build_track_flat_bundle(tracks, device):
    valid_track_indices = [i for i, track in enumerate(tracks) if len(track) >= 2]
    if not valid_track_indices:
        return None, torch.zeros(0, dtype=torch.int64), valid_track_indices

    pairs = [
        (
            np.asarray(tracks[i], dtype=np.float32),
            np.zeros(len(tracks[i]), dtype=np.float32),
        )
        for i in valid_track_indices
    ]
    lengths_np = np.fromiter((len(z) for z, _ in pairs), dtype=np.int64, count=len(pairs))
    starts_np = np.empty(len(pairs) + 1, dtype=np.int64)
    starts_np[0] = 0
    np.cumsum(lengths_np, out=starts_np[1:])
    total = int(starts_np[-1])
    if total == 0:
        return None, torch.from_numpy(lengths_np), valid_track_indices

    zyxs_flat = np.concatenate([z for z, _ in pairs], axis=0).astype(np.float32, copy=False)
    windings_flat = np.concatenate([w for _, w in pairs], axis=0).astype(np.float32, copy=False)
    strip_id_np = np.repeat(np.arange(len(pairs), dtype=np.int64), lengths_np)
    flat = {
        'zyxs': torch.from_numpy(zyxs_flat).to(device=device),
        'windings': torch.from_numpy(windings_flat).to(device=device),
        'strip_id': torch.from_numpy(strip_id_np).to(device=device),
        'starts': torch.from_numpy(starts_np).to(device=device),
        'lengths': torch.from_numpy(lengths_np).to(device=device),
        'lengths_cpu': torch.from_numpy(lengths_np),
        'total': total,
    }
    return flat, flat['lengths_cpu'], valid_track_indices


def _build_track_spiral_context(slice_to_spiral_transform, dr_per_winding, flat, num_tracks, metrics_config):
    spiral_tolerance = dr_per_winding.detach() * metrics_config['satisfaction_radius_tolerance']
    scan_tolerance = metrics_config['satisfaction_distance_tolerance']
    dr = dr_per_winding.detach()
    device = dr_per_winding.device

    if flat is None or flat['total'] == 0:
        lengths_cpu = flat['lengths_cpu'] if flat is not None else torch.zeros(num_tracks, dtype=torch.int64)
        return None, lengths_cpu, num_tracks

    chunk = 65536

    def transform_in_chunks(zyxs, fn):
        if zyxs.shape[0] <= chunk:
            return fn(zyxs)
        pieces = []
        for st in range(0, zyxs.shape[0], chunk):
            pieces.append(fn(zyxs[st:st + chunk]))
        return torch.cat(pieces, dim=0)

    zyxs = flat['zyxs']
    windings = flat['windings']
    track_id = flat['strip_id']
    starts = flat['starts']
    lengths = flat['lengths']
    lengths_cpu = flat['lengths_cpu']
    total = flat['total']

    with torch.no_grad():
        spiral_zyxs = transform_in_chunks(zyxs, slice_to_spiral_transform)
        theta, _, shifted_radii = get_theta_and_radii(spiral_zyxs[..., 1:], dr_per_winding)

        if total > 1:
            theta_d = theta.detach()
            diffs = theta_d[1:] - theta_d[:-1]
            same_track = track_id[1:] == track_id[:-1]
            step_adj = (
                (diffs > np.pi).to(shifted_radii.dtype)
                - (diffs < -np.pi).to(shifted_radii.dtype)
            ) * dr
            step_adj = torch.where(same_track, step_adj, torch.zeros_like(step_adj))
            cumsum_inner = torch.cumsum(step_adj, dim=0)
            cumsum_flat = torch.cat([
                torch.zeros(1, device=device, dtype=cumsum_inner.dtype),
                cumsum_inner,
            ], dim=0)
            adjustments = cumsum_flat - cumsum_flat[starts[:-1][track_id]]
        else:
            adjustments = torch.zeros_like(shifted_radii)
        unwrapped_shifted = shifted_radii + adjustments
        normalised_radii = unwrapped_shifted - windings * dr

    return {
        'spiral_tolerance': spiral_tolerance,
        'scan_tolerance': scan_tolerance,
        'dr': dr,
        'device': device,
        'num_tracks': num_tracks,
        'slice_to_spiral_transform': slice_to_spiral_transform,
        'transform_in_chunks': transform_in_chunks,
        'zyxs': zyxs,
        'windings': windings,
        'track_id': track_id,
        'lengths_cpu': lengths_cpu,
        'spiral_zyxs': spiral_zyxs,
        'theta': theta,
        'adjustments': adjustments,
        'unwrapped_shifted': unwrapped_shifted,
        'normalised_radii': normalised_radii,
    }, lengths_cpu, num_tracks


def _mode_winding_per_track(track_id, winding_idx_per_point, num_tracks, device):
    mode_winding_per_track = torch.zeros(num_tracks, dtype=torch.int64, device=device)
    if winding_idx_per_point.numel() == 0:
        return mode_winding_per_track

    w_min = winding_idx_per_point.min()
    w_max = winding_idx_per_point.max()
    w_span = (w_max - w_min + 1).to(torch.int64)
    composite = track_id.to(torch.int64) * w_span + (winding_idx_per_point - w_min).to(torch.int64)
    sorted_comp, _ = torch.sort(composite)
    unique_comp, counts = torch.unique_consecutive(sorted_comp, return_counts=True)
    u_track = unique_comp // w_span
    u_widx = (unique_comp % w_span) + w_min

    counts_max = counts.max().to(torch.int64)
    widx_min = u_widx.min().to(torch.int64)
    widx_max = u_widx.max().to(torch.int64)
    widx_span = (widx_max - widx_min + 1).to(torch.int64)
    key = (
        u_track * ((counts_max + 1) * widx_span)
        + (counts_max - counts.to(torch.int64)) * widx_span
        + (u_widx.to(torch.int64) - widx_min)
    )
    order = torch.argsort(key)
    sorted_track = u_track[order]
    sorted_widx = u_widx[order]
    new_track = torch.cat([
        torch.ones(1, dtype=torch.bool, device=device),
        sorted_track[1:] != sorted_track[:-1],
    ])
    first_idx = torch.nonzero(new_track, as_tuple=False).squeeze(-1)
    mode_winding_per_track[sorted_track[first_idx]] = sorted_widx[first_idx].to(torch.int64)
    return mode_winding_per_track


def _track_satisfaction_from_target(ctx, target_normalised_per_track):
    dr = ctx['dr']
    device = ctx['device']
    num_tracks = ctx['num_tracks']
    track_id = ctx['track_id']
    windings = ctx['windings']
    theta = ctx['theta']
    adjustments = ctx['adjustments']
    unwrapped_shifted = ctx['unwrapped_shifted']
    spiral_zyxs = ctx['spiral_zyxs']
    zyxs = ctx['zyxs']
    lengths_cpu = ctx['lengths_cpu']
    spiral_tolerance = ctx['spiral_tolerance']
    scan_tolerance = ctx['scan_tolerance']
    transform_in_chunks = ctx['transform_in_chunks']
    slice_to_spiral_transform = ctx['slice_to_spiral_transform']

    with torch.no_grad():
        target_normalised = target_normalised_per_track[track_id]
        target_shifted = target_normalised + windings * dr
        spiral_in_band = (unwrapped_shifted - target_shifted).abs() <= spiral_tolerance

        target_radii = target_shifted - adjustments + theta / (2 * np.pi) * dr
        target_spiral_zyxs = torch.stack([
            spiral_zyxs[..., 0],
            torch.sin(theta) * target_radii,
            torch.cos(theta) * target_radii,
        ], dim=-1)
        target_scroll_zyxs = transform_in_chunks(target_spiral_zyxs, slice_to_spiral_transform.inv)
        scan_distances = torch.linalg.norm(target_scroll_zyxs - zyxs, dim=-1)
        scan_in_band = scan_distances <= scan_tolerance

        satisfied = spiral_in_band & scan_in_band
        satisfied_counts_dev = torch.zeros(num_tracks, dtype=torch.int64, device=device)
        satisfied_counts_dev.scatter_add_(0, track_id, satisfied.to(torch.int64))
        satisfied_counts = satisfied_counts_dev.cpu()
        per_point_satisfaction = list(torch.split(satisfied.cpu(), lengths_cpu.tolist()))

    return satisfied_counts, per_point_satisfaction


def get_track_satisfied_counts(slice_to_spiral_transform, dr_per_winding, tracks, metrics_config):
    device = dr_per_winding.device
    if not tracks:
        empty = torch.zeros(0, dtype=torch.int64)
        return empty, empty, empty, [], empty

    flat, lengths_cpu, valid_track_indices = _build_track_flat_bundle(tracks, device)
    if not valid_track_indices:
        empty = torch.zeros(0, dtype=torch.int64)
        return empty, empty, empty, [], empty

    num_tracks = len(valid_track_indices)
    ctx, lengths_cpu, num_tracks = _build_track_spiral_context(
        slice_to_spiral_transform, dr_per_winding, flat, num_tracks, metrics_config,
    )
    valid_track_indices_t = torch.tensor(valid_track_indices, dtype=torch.int64)
    if ctx is None:
        per_point = [torch.zeros([int(n.item())], dtype=torch.bool) for n in lengths_cpu]
        return valid_track_indices_t, torch.zeros(num_tracks, dtype=torch.int64), lengths_cpu.clone(), per_point, torch.zeros(num_tracks, dtype=torch.int64)

    dr = ctx['dr']
    track_id = ctx['track_id']
    normalised_radii = ctx['normalised_radii']

    with torch.no_grad():
        winding_idx_per_point = torch.round(normalised_radii / dr).to(torch.int64)
        mode_winding_per_track = _mode_winding_per_track(track_id, winding_idx_per_point, num_tracks, device)
        target_normalised_per_track = mode_winding_per_track.to(dr.dtype) * dr

    satisfied_counts, per_point_satisfaction = _track_satisfaction_from_target(ctx, target_normalised_per_track)
    return (
        valid_track_indices_t,
        satisfied_counts,
        lengths_cpu.clone(),
        per_point_satisfaction,
        mode_winding_per_track.cpu(),
    )


def get_track_satisfied_counts_in_chunks(slice_to_spiral_transform, dr_per_winding, tracks, metrics_config, chunk_size=500_000):
    sat_parts, tot_parts = [], []
    for start in range(0, len(tracks), chunk_size):
        chunk = tracks[start:start + chunk_size]
        _, sat, tot, _, _ = get_track_satisfied_counts(
            slice_to_spiral_transform, dr_per_winding, chunk, metrics_config,
        )
        sat_parts.append(sat)
        tot_parts.append(tot)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if not sat_parts:
        empty = torch.zeros(0, dtype=torch.int64)
        return empty, empty
    return torch.cat(sat_parts), torch.cat(tot_parts)


def _build_anchor_kdtree(anchor_zyx):
    if anchor_zyx is None:
        return None
    if isinstance(anchor_zyx, torch.Tensor):
        anchor_np = anchor_zyx.detach().cpu().numpy()
    else:
        anchor_np = np.asarray(anchor_zyx)
    if anchor_np.shape[0] == 0:
        return None
    return cKDTree(np.ascontiguousarray(anchor_np, dtype=np.float32))


def _track_points_far_from_anchors_mask(track_zyx, anchor_tree, threshold):
    if isinstance(track_zyx, torch.Tensor):
        track_np = track_zyx.detach().cpu().numpy()
    else:
        track_np = np.asarray(track_zyx)
    track_np = np.ascontiguousarray(track_np, dtype=np.float32)
    if threshold <= 0 or anchor_tree is None:
        return np.ones(track_np.shape[0], dtype=bool)
    dist, _ = anchor_tree.query(track_np, k=1, distance_upper_bound=float(threshold), workers=-1)
    return np.isinf(dist)


def prepare_main_phase_tracks(tracks, anchor_scroll_zyxs, exclusion_radius, device, anchor_tree=None):
    if not tracks:
        return None
    print('removing tracks near patches')
    if anchor_tree is None:
        anchor_tree = _build_anchor_kdtree(anchor_scroll_zyxs)

    # The common configuration has no exclusion radius.  The generic path
    # below creates several point-count-sized int64 arrays and stable-sorts the
    # already grouped tracks; none of that changes the result when every point
    # is kept.  Concatenate only the surviving (length >= 2) tracks directly.
    if exclusion_radius <= 0 or anchor_tree is None:
        surviving_tracks = [
            np.asarray(track, dtype=np.float32)
            for track in tracks
            if len(track) >= 2
        ]
        print(f'kept {len(surviving_tracks)} / {len(tracks)} tracks')
        if not surviving_tracks:
            return None
        lengths_new = np.fromiter(
            (len(track) for track in surviving_tracks),
            dtype=np.int64,
            count=len(surviving_tracks),
        )
        flat_zyx_np = np.concatenate(surviving_tracks, axis=0)
        offsets_new = np.empty(len(lengths_new) + 1, dtype=np.int64)
        offsets_new[0] = 0
        np.cumsum(lengths_new, out=offsets_new[1:])
        print(
            f'track radius loss: {len(surviving_tracks)}/{len(tracks)} tracks survive exclusion '
            f'(radius {exclusion_radius:.1f}); {int(lengths_new.sum())} points retained'
        )
        return {
            'flat_zyx_cpu': torch.from_numpy(flat_zyx_np).contiguous(),
            'offsets': torch.from_numpy(offsets_new).to(device=device),
            'lengths': torch.from_numpy(lengths_new).to(device=device),
            'device': torch.device(device),
            'staging': None,
        }

    flat_zyx_np = np.concatenate([t.astype(np.float32) for t in tracks], axis=0)
    track_id_np = np.concatenate([
        np.full(len(t), i, dtype=np.int64) for i, t in enumerate(tracks)
    ])
    keep_np = _track_points_far_from_anchors_mask(flat_zyx_np, anchor_tree, exclusion_radius)
    flat_zyx_np = flat_zyx_np[keep_np]
    track_id_np = track_id_np[keep_np]
    num_tracks_orig = len(tracks)
    new_lengths = np.bincount(track_id_np, minlength=num_tracks_orig)
    surviving = np.where(new_lengths >= 2)[0]
    print(f'kept {len(surviving)} / {len(tracks)} tracks')
    if len(surviving) == 0:
        return None
    old_to_new = -np.ones(num_tracks_orig, dtype=np.int64)
    old_to_new[surviving] = np.arange(len(surviving))
    new_id = old_to_new[track_id_np]
    keep2 = new_id >= 0
    flat_zyx_np = flat_zyx_np[keep2]
    new_id = new_id[keep2]
    sort_idx = np.argsort(new_id, kind='stable')
    flat_zyx_np = flat_zyx_np[sort_idx]
    lengths_new = new_lengths[surviving].astype(np.int64)
    offsets_new = np.concatenate([[0], np.cumsum(lengths_new)]).astype(np.int64)
    print(
        f'track radius loss: {len(surviving)}/{num_tracks_orig} tracks survive exclusion '
        f'(radius {exclusion_radius:.1f}); {int(lengths_new.sum())} points retained'
    )
    return {
        # The full point cloud can be tens of millions of points, while a step
        # consumes only a sampled batch. Keep coordinates in host RAM and stage
        # that batch below; offsets/lengths remain on CUDA so sampling stays fast
        # and retains the existing CUDA RNG sequence.
        'flat_zyx_cpu': torch.from_numpy(flat_zyx_np).contiguous(),
        'offsets': torch.from_numpy(offsets_new).to(device=device),
        'lengths': torch.from_numpy(lengths_new).to(device=device),
        'device': torch.device(device),
        'staging': None,
    }


def _draw_track_sample(prepared_tracks, k, num_points_per_track, generator=None):
    # GPU index math, one forced D2H of the flat indices, the big host-side
    # coordinate gather, and the H2D upload. Under prefetch this runs on the
    # worker thread for the next step, so the syncs stall the worker rather
    # than the training loop.
    flat_zyx_cpu = prepared_tracks['flat_zyx_cpu']
    offsets = prepared_tracks['offsets']
    lengths = prepared_tracks['lengths']
    device = prepared_tracks['device']
    num_tracks = int(lengths.numel())
    track_idx = torch.randint(num_tracks, (k,), device=device, generator=generator)
    track_lengths_sample = lengths[track_idx]
    track_offsets_sample = offsets[track_idx]
    point_idx_within = (
        torch.rand([k, num_points_per_track], device=device, generator=generator)
        * track_lengths_sample[:, None].to(torch.float32)
    ).to(torch.int64)
    point_idx_within, _ = torch.sort(point_idx_within, dim=-1)
    flat_idx = (track_offsets_sample[:, None] + point_idx_within).reshape(-1)
    flat_idx_cpu = flat_idx.to(device='cpu')
    sampled_cpu = flat_zyx_cpu[flat_idx_cpu].view(k, num_points_per_track, 3)
    if device.type == 'cuda':
        staging = prepared_tracks.get('staging')
        if staging is None or staging.shape != sampled_cpu.shape:
            staging = torch.empty(sampled_cpu.shape, dtype=torch.float32, pin_memory=True)
            prepared_tracks['staging'] = staging
        staging.copy_(sampled_cpu)
        # Synchronous reuse of one pinned buffer avoids a lifetime hazard between
        # consecutive iterations. The transfer is only the sampled batch (~19 MB
        # with the default full-range settings), not the complete track database.
        sampled_scroll = staging.to(device=device, non_blocking=False)
    else:
        sampled_scroll = sampled_cpu.to(device=device)
    return track_idx, flat_idx, sampled_scroll


def _sample_prepared_track_points(prepared_tracks, num_tracks_per_step, num_points_per_track):
    lengths = prepared_tracks['lengths']
    device = prepared_tracks['device']
    num_tracks = int(lengths.numel())
    if num_tracks == 0 or num_tracks_per_step <= 0 or num_points_per_track <= 0:
        return None
    k = min(int(num_tracks_per_step), num_tracks)

    if prefetch.prefetch_enabled() and device.type == 'cuda':
        pf = prefetch.get_prefetcher()
        generator = pf.torch_rng('tracks', device)
        return pf.pop_or_run(
            ('tracks', k, num_points_per_track),
            lambda: _draw_track_sample(prepared_tracks, k, num_points_per_track, generator),
        )
    return _draw_track_sample(prepared_tracks, k, num_points_per_track)


@geom_utils.maybe_compile
def _same_radius_loss_tensor(
    shifted_radii,
    dr_per_winding,
    use_median_target,
    radius_loss_margin,
    within_p,
):
    radius_hinge_margin = dr_per_winding.detach() * radius_loss_margin
    if use_median_target:
        radius_target_per_track = shifted_radii.median(dim=-1, keepdim=True).values
    else:
        radius_target_per_track = shifted_radii.mean(dim=-1, keepdim=True)
    deviations = (shifted_radii - radius_target_per_track).abs()
    hinged = F.relu(deviations - radius_hinge_margin)
    if within_p == 1.0:
        return hinged.mean()
    per_track = ((hinged + 1.e-5) ** within_p).mean(dim=-1) ** (1.0 / within_p)
    return per_track.mean()


def _same_radius_loss_for_shifted_radii(shifted_radii, dr_per_winding, cfg):
    target = cfg['track_radius_target']
    if target not in ('mean', 'median'):
        raise ValueError(f"track_radius_target must be 'mean' or 'median', got {target!r}")
    return _same_radius_loss_tensor(
        shifted_radii,
        dr_per_winding,
        target == 'median',
        cfg['track_radius_loss_margin'],
        cfg['track_radius_within_norm_p'],
    )


def iter_track_losses(slice_to_spiral_transform, dr_per_winding, prepared_tracks, cfg, compute_dt=True, dt_max_winding=None):
    """Yield radius then DT losses so the caller can backward them separately.

    The DT target is detached before its inverse transform, so its graph does
    not depend on the radius-loss forward graph.  Yielding at that boundary
    prevents both large transform graphs from being resident together.
    """
    device = dr_per_winding.device
    zero = torch.zeros([], device=device)
    if prepared_tracks is None:
        yield 'track_radius', zero
        yield 'track_dt', zero
        return
    sample = _sample_prepared_track_points(
        prepared_tracks,
        cfg['track_num_per_step'],
        cfg['track_num_points_per_step'],
    )
    if sample is None:
        yield 'track_radius', zero
        yield 'track_dt', zero
        return
    _, _, sampled_scroll = sample
    k = sampled_scroll.shape[0]
    num_points = sampled_scroll.shape[1]
    sampled_spiral = slice_to_spiral_transform(sampled_scroll.reshape(-1, 3)).reshape(k, num_points, 3)
    theta, _, shifted_radii = get_theta_and_radii(sampled_spiral[..., 1:], dr_per_winding)
    shifted_radii = _unwrap_track_shifted_radii(theta, shifted_radii, dr_per_winding)
    dt_hinge_margin = dr_per_winding.detach() * cfg['track_dt_loss_margin']
    radius_loss = _same_radius_loss_for_shifted_radii(shifted_radii, dr_per_winding, cfg)

    if not compute_dt:
        yield 'track_radius', radius_loss
        del radius_loss, sampled_spiral, theta, shifted_radii
        yield 'track_dt', zero
        return

    target_shifted_radii = torch.round(shifted_radii.median(dim=-1, keepdim=True).values / dr_per_winding) * dr_per_winding
    target_radii = target_shifted_radii + theta / (2 * np.pi) * dr_per_winding
    target_spiral_zyxs = torch.stack([
        sampled_spiral[..., 0],
        torch.sin(theta) * target_radii,
        torch.cos(theta) * target_radii,
    ], dim=-1).detach()
    active_mask = _progressive_dt_active_mask(target_shifted_radii.squeeze(-1), dr_per_winding, dt_max_winding)

    yield 'track_radius', radius_loss
    # The caller has now released the radius graph.  Keep only detached DT
    # inputs before constructing the inverse-transform graph.
    del radius_loss, sampled_spiral, theta, shifted_radii
    del target_radii, target_shifted_radii
    target_scroll_zyxs = slice_to_spiral_transform.inv(target_spiral_zyxs.reshape(-1, 3)).reshape(*target_spiral_zyxs.shape)

    within_p = cfg['track_dt_within_track_norm_p']
    across_p = cfg['track_dt_norm_p']
    point_distances = torch.linalg.norm(sampled_scroll - target_scroll_zyxs, dim=-1)
    point_distances = F.relu(point_distances - dt_hinge_margin) + 1.e-5
    track_losses = (point_distances ** within_p).mean(dim=-1) ** (1 / within_p)
    dt_loss = _aggregate_dt_track_losses(track_losses, across_p, active_mask)

    yield 'track_dt', dt_loss


def render_spiral_on_tracks_for_slice(
    spiral_zyx, spiral_density, dr_per_winding,
    slice_z, all_tracks, snapped_tracks,
    out_path, name_suffix,
    render_volume_scale=1,
):
    z_window = 20
    point_radius = 1
    target_ids = {id(t) for t in snapped_tracks}

    def track_colour(track, is_target):
        hue = ((id(track) * 2654435761) & 0xFFFFFFFF) / 2 ** 32
        sat, val = (0.9, 1.0) if is_target else (0.35, 0.75)
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        return (int(r * 255), int(g * 255), int(b * 255))

    _, _, shifted_radius = get_theta_and_radii(spiral_zyx[..., 1:], dr_per_winding)
    winding_idx = (shifted_radius / dr_per_winding).round().to(torch.int64).clamp_min(0)
    num_winding_hues = 6
    hue_min, hue_max = 1.5 / 6, 5.25 / 6
    hue_fraction = hue_min + (winding_idx % num_winding_hues).to(torch.float32) / num_winding_hues * (hue_max - hue_min)
    hue = hue_fraction * 2 * np.pi
    hsv = torch.stack([hue, torch.full_like(hue, 0.5), torch.ones_like(hue)])
    spiral_colours = kornia.color.hsv_to_rgb(hsv).permute(1, 2, 0) * 255
    canvas = (spiral_colours * spiral_density[..., None]).to(torch.uint8).cpu().numpy()
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)

    for is_target in (False, True):
        for track in all_tracks:
            if (id(track) in target_ids) != is_target:
                continue
            zs = track[:, 0]
            in_slab = np.abs(zs.astype(np.float32) - float(slice_z)) <= z_window
            if not in_slab.any():
                continue
            colour = track_colour(track, is_target)
            for idx in np.nonzero(in_slab)[0]:
                y = float(track[idx, 1]) / render_volume_scale
                x = float(track[idx, 2]) / render_volume_scale
                draw.ellipse(
                    [x - point_radius, y - point_radius, x + point_radius, y + point_radius],
                    fill=colour,
                )
    image.save(f'{out_path}/spiral_on_tracks_s{int(slice_z):05}_{name_suffix}.png', compress_level=3)
