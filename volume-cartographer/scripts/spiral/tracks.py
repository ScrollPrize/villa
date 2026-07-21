import colorsys
import dbm
import itertools
import math
import pickle

import kornia
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
from tqdm import tqdm

import prefetch
from dt_targets import strip_dt_target_in_sample_frame
from loss_maps import diagnostics_enabled, record_loss_samples
import geom_utils
from sample_spiral import (
    get_theta_and_radii,
    get_theta_crossing_step_adjustments,
    radius_from_unwrapped_shifted,
    unwrap_shifted_radii,
)


def load_tracks_from_dbm(path, z_lo, z_hi, return_families=False):
    # Load tracks written by extract_surface_tracks.py. Each DBM value is a
    # pickled list of (N, 3) int32 zyx arrays; keep only tracks that lie entirely
    # within the full-resolution [z_lo, z_hi) ROI.
    tracks = []
    families = []
    with dbm.open(path, 'r') as db:
        for key in tqdm(db.keys(), desc='loading tracks'):
            family = None
            if return_families:
                prefix = key.decode().split(':', 1)[0]
                family = 'horizontal' if prefix == 'h' else (
                    'vertical' if prefix in ('vx', 'vy') else None)
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
                if return_families:
                    families.append(family)
    if return_families:
        return tracks, families
    return tracks


def validate_track_sampling_config(config):
    """Validate and normalize the optional session-scoped track policies."""
    weights = config.get('track_length_bin_weights')
    if weights is not None:
        if not isinstance(weights, (list, tuple)) or len(weights) != 3:
            raise ValueError('track_length_bin_weights must be null or [short, medium, long]')
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(float(value)) or float(value) < 0
               for value in weights):
            raise ValueError('track_length_bin_weights values must be finite and non-negative')
        weights = np.asarray(weights, dtype=np.float64)
        if float(weights.sum()) <= 0:
            raise ValueError('track_length_bin_weights must contain at least one positive value')
        weights = weights / weights.sum()

    max_tortuosity = config.get('track_max_tortuosity')
    if max_tortuosity is not None:
        if (isinstance(max_tortuosity, bool)
                or not isinstance(max_tortuosity, (int, float))
                or not math.isfinite(float(max_tortuosity))
                or float(max_tortuosity) < 1.0):
            raise ValueError('track_max_tortuosity must be null or a finite number >= 1')
        max_tortuosity = float(max_tortuosity)

    max_crossings = config.get('max_track_crossing_per_step', 0)
    if (isinstance(max_crossings, bool) or not isinstance(max_crossings, (int, float))
            or not math.isfinite(float(max_crossings))
            or not float(max_crossings).is_integer() or int(max_crossings) < 0):
        raise ValueError('max_track_crossing_per_step must be a non-negative integer')

    return {
        'length_bin_weights': weights,
        'max_tortuosity': max_tortuosity,
        'max_crossings': int(max_crossings),
    }


def _polyline_arclengths(tracks):
    return np.asarray([
        np.linalg.norm(np.diff(np.asarray(track, dtype=np.float64), axis=0), axis=1).sum()
        if len(track) >= 2 else 0.0
        for track in tracks
    ], dtype=np.float64)


def _track_tortuosities(tracks, arclengths):
    chords = np.asarray([
        np.linalg.norm(np.asarray(track[-1], dtype=np.float64)
                       - np.asarray(track[0], dtype=np.float64))
        if len(track) >= 2 else 0.0
        for track in tracks
    ], dtype=np.float64)
    result = np.full(len(tracks), np.inf, dtype=np.float64)
    np.divide(arclengths, chords, out=result, where=chords > 0)
    return result


def _length_bin_probabilities(arclengths, weights, device):
    edges = np.quantile(arclengths, [1 / 3, 2 / 3])
    bin_ids = (arclengths > edges[0]).astype(np.int64)
    bin_ids += (arclengths > edges[1]).astype(np.int64)
    counts = np.bincount(bin_ids, minlength=3)
    available_weights = np.where(counts > 0, weights, 0.0)
    if available_weights.sum() <= 0:
        raise ValueError(
            'track_length_bin_weights assign no probability to the non-empty length bins')
    available_weights /= available_weights.sum()
    probabilities = available_weights[bin_ids] / counts[bin_ids]
    print(
        'track length bins: '
        f'short <= {edges[0]:.1f} ({counts[0]}), '
        f'medium <= {edges[1]:.1f} ({counts[1]}), long ({counts[2]}); '
        f'effective weights {available_weights.tolist()}'
    )
    return torch.as_tensor(probabilities, dtype=torch.float32, device=device)


def _track_tangent(track, raw_index, radius_voxels=12.0):
    point = np.asarray(track[raw_index], dtype=np.float64)
    left = raw_index
    while left > 0 and np.linalg.norm(np.asarray(track[left], dtype=np.float64) - point) < radius_voxels:
        left -= 1
    right = raw_index
    while (right + 1 < len(track)
           and np.linalg.norm(np.asarray(track[right], dtype=np.float64) - point) < radius_voxels):
        right += 1
    vector = np.asarray(track[right], dtype=np.float64) - np.asarray(track[left], dtype=np.float64)
    norm = np.linalg.norm(vector)
    return vector / norm if norm else None


def _pack_track_points(points):
    integer_points = np.asarray(points, dtype=np.int64)
    if np.any(integer_points < 0) or np.any(integer_points >= (1 << 20)):
        raise ValueError('track coordinates must lie in [0, 2**20) for crossing pairing')
    packed = integer_points.astype(np.uint64, copy=False)
    return ((packed[:, 0] << np.uint64(40))
            | (packed[:, 1] << np.uint64(20)) | packed[:, 2])


def _select_spaced_crossing_partners(candidates, maximum):
    """Choose distinct partners, spreading their crossings along the primary."""
    if not candidates or maximum <= 0:
        return []
    remaining = list(candidates)
    if maximum == 1 or len(remaining) == 1:
        return [max(remaining, key=lambda item: (item[2], -item[0]))[0]]

    # Seed the set with the widest-separated pair, then use maximin spacing
    # for any remaining slots. Clearance and partner id make ties stable.
    first = min(remaining, key=lambda item: (item[1], -item[2], item[0]))
    selected = [first]
    remaining.remove(first)
    second = max(
        remaining,
        key=lambda item: (abs(item[1] - first[1]), item[2], -item[0]),
    )
    selected.append(second)
    remaining.remove(second)
    while remaining and len(selected) < maximum:
        positions = [item[1] for item in selected]
        choice = max(
            remaining,
            key=lambda item: (
                min(abs(item[1] - position) for position in positions),
                item[2],
                -item[0],
            ),
        )
        selected.append(choice)
        remaining.remove(choice)
    return [partner for partner, _, _ in selected]


def _build_crossing_partner_table(tracks, families, maximum, device):
    """Mirror grow_track_grids exact crossings for the opt-in sampler."""
    if maximum <= 0:
        return None
    if families is None or len(families) != len(tracks):
        raise ValueError('crossing-track sampling requires DBM track-family provenance')
    if not tracks:
        return torch.empty((0, maximum), dtype=torch.int64, device=device)

    lengths = np.fromiter((len(track) for track in tracks), dtype=np.int64, count=len(tracks))
    points = np.concatenate(tracks, axis=0)
    track_ids = np.repeat(np.arange(len(tracks), dtype=np.int64), lengths)
    local_indices = np.concatenate([
        np.arange(length, dtype=np.int64) for length in lengths
    ])
    packed = _pack_track_points(points)
    order = np.argsort(packed, kind='stable')
    packed = packed[order]
    track_ids = track_ids[order]
    local_indices = local_indices[order]
    boundaries = np.flatnonzero(packed[1:] != packed[:-1]) + 1
    starts = np.r_[0, boundaries]
    stops = np.r_[boundaries, len(packed)]
    shared = stops - starts > 1
    starts = starts[shared]
    stops = stops[shared]
    del points, packed, order, boundaries, shared

    angle_cutoff = math.cos(math.radians(30.0))
    tangent_cache = {}
    raw_events = {}
    for start, stop in zip(starts, stops):
        unique = {}
        for position in range(int(start), int(stop)):
            unique.setdefault(int(track_ids[position]), int(local_indices[position]))
        if len(unique) < 2:
            continue
        for first, second in itertools.combinations(unique, 2):
            if (families[first] is None or families[second] is None
                    or families[first] == families[second]):
                continue
            first_index, second_index = unique[first], unique[second]
            first_key, second_key = (first, first_index), (second, second_index)
            if first_key not in tangent_cache:
                tangent_cache[first_key] = _track_tangent(tracks[first], first_index)
            if second_key not in tangent_cache:
                tangent_cache[second_key] = _track_tangent(tracks[second], second_index)
            first_tangent, second_tangent = tangent_cache[first_key], tangent_cache[second_key]
            if first_tangent is None or second_tangent is None:
                continue
            if abs(float(np.dot(first_tangent, second_tangent))) > angle_cutoff:
                continue
            raw_events.setdefault((first, second), []).append((first_index, second_index))

    cumulative = [np.r_[0.0, np.cumsum(np.linalg.norm(
        np.diff(np.asarray(track, dtype=np.float64), axis=0), axis=1))]
        for track in tracks]
    adjacency = [[] for _ in tracks]
    accepted_events = 0
    for (first, second), events in raw_events.items():
        events.sort()
        clusters = []
        cluster = []
        for event in events:
            if (cluster and abs(event[0] - cluster[-1][0]) <= 4
                    and abs(event[1] - cluster[-1][1]) <= 4):
                cluster.append(event)
            else:
                if cluster:
                    clusters.append(cluster)
                cluster = [event]
        if cluster:
            clusters.append(cluster)
        representatives = [cluster[len(cluster) // 2] for cluster in clusters]
        best = None
        for first_index, second_index in representatives:
            first_position = float(cumulative[first][first_index])
            second_position = float(cumulative[second][second_index])
            clearance = min(
                first_position, cumulative[first][-1] - first_position,
                second_position, cumulative[second][-1] - second_position,
            )
            candidate = (clearance, first_position, second_position)
            if best is None or candidate > best:
                best = candidate
        clearance, first_position, second_position = best
        adjacency[first].append((second, first_position, clearance))
        adjacency[second].append((first, second_position, clearance))
        accepted_events += len(representatives)

    table = np.full((len(tracks), maximum), -1, dtype=np.int64)
    for track_id, candidates in enumerate(adjacency):
        chosen = _select_spaced_crossing_partners(candidates, maximum)
        table[track_id, :len(chosen)] = chosen
    paired_tracks = int(np.count_nonzero(np.any(table >= 0, axis=1)))
    partner_slots = int(np.count_nonzero(table >= 0))
    print(
        f'track crossings: {accepted_events} exact crossing events, '
        f'{paired_tracks}/{len(tracks)} tracks have partners, '
        f'{partner_slots} partner slots retained (max {maximum} per primary)'
    )
    return torch.from_numpy(table).to(device=device)


def _unwrap_track_shifted_radii(theta, shifted_radii, dr_per_winding):
    # Compatibility wrapper for callers/tests that only need the unwrapped values.
    return unwrap_shifted_radii(theta, shifted_radii, dr_per_winding)[0]


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
            same_track = track_id[1:] == track_id[:-1]
            step_adj = get_theta_crossing_step_adjustments(theta, dr)
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

        target_radii = radius_from_unwrapped_shifted(
            theta, target_shifted, adjustments, dr,
        )
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


def prepare_main_phase_tracks(
        tracks, anchor_scroll_zyxs, exclusion_radius, device, anchor_tree=None,
        sampling_config=None, track_families=None):
    if not tracks:
        return None
    if sampling_config is not None and 'length_bin_weights' in sampling_config:
        policy = sampling_config
    else:
        policy = validate_track_sampling_config(sampling_config or {})
    weights = policy['length_bin_weights']
    max_tortuosity = policy['max_tortuosity']
    max_crossings = policy['max_crossings']

    input_track_count = len(tracks)
    working_tracks = list(tracks)
    working_families = list(track_families) if track_families is not None else None
    arclengths = None
    if weights is not None or max_tortuosity is not None:
        arclengths = _polyline_arclengths(working_tracks)
    if max_tortuosity is not None:
        tortuosities = _track_tortuosities(working_tracks, arclengths)
        keep = np.flatnonzero(tortuosities <= max_tortuosity)
        working_tracks = [working_tracks[index] for index in keep]
        if working_families is not None:
            working_families = [working_families[index] for index in keep]
        arclengths = arclengths[keep]
        print(
            f'track tortuosity <= {max_tortuosity:g}: kept '
            f'{len(working_tracks)} / {input_track_count} tracks'
        )
        if not working_tracks:
            return None

    print('removing tracks near patches')
    if anchor_tree is None:
        anchor_tree = _build_anchor_kdtree(anchor_scroll_zyxs)

    def finish_prepared(flat_zyx_np, lengths_new, surviving_indices):
        offsets_new = np.empty(len(lengths_new) + 1, dtype=np.int64)
        offsets_new[0] = 0
        np.cumsum(lengths_new, out=offsets_new[1:])
        prepared = {
            # The full point cloud can be tens of millions of points, while a
            # step consumes only a sampled batch. Keep coordinates in host RAM.
            'flat_zyx_cpu': torch.from_numpy(flat_zyx_np).contiguous(),
            'offsets': torch.from_numpy(offsets_new).to(device=device),
            'lengths': torch.from_numpy(lengths_new).to(device=device),
            'device': torch.device(device),
            'staging': None,
        }
        if weights is not None:
            eligible_arclengths = arclengths[surviving_indices]
            prepared['sampling_probabilities'] = _length_bin_probabilities(
                eligible_arclengths, weights, device)
        if max_crossings > 0:
            eligible_tracks = [working_tracks[index] for index in surviving_indices]
            eligible_families = (
                [working_families[index] for index in surviving_indices]
                if working_families is not None else None)
            prepared['crossing_partners'] = _build_crossing_partner_table(
                eligible_tracks, eligible_families, max_crossings, device)
        return prepared

    # The common configuration has no exclusion radius.  The generic path
    # below creates several point-count-sized int64 arrays and stable-sorts the
    # already grouped tracks; none of that changes the result when every point
    # is kept.  Concatenate only the surviving (length >= 2) tracks directly.
    if exclusion_radius <= 0 or anchor_tree is None:
        surviving = np.asarray([
            index for index, track in enumerate(working_tracks) if len(track) >= 2
        ], dtype=np.int64)
        surviving_tracks = [
            np.asarray(working_tracks[index], dtype=np.float32)
            for index in surviving
        ]
        print(f'kept {len(surviving_tracks)} / {len(working_tracks)} tracks')
        if not surviving_tracks:
            return None
        lengths_new = np.fromiter(
            (len(track) for track in surviving_tracks),
            dtype=np.int64,
            count=len(surviving_tracks),
        )
        flat_zyx_np = np.concatenate(surviving_tracks, axis=0)
        print(
            f'track radius loss: {len(surviving_tracks)}/{len(working_tracks)} tracks survive exclusion '
            f'(radius {exclusion_radius:.1f}); {int(lengths_new.sum())} points retained'
        )
        return finish_prepared(flat_zyx_np, lengths_new, surviving)

    flat_zyx_np = np.concatenate([t.astype(np.float32) for t in working_tracks], axis=0)
    track_id_np = np.concatenate([
        np.full(len(t), i, dtype=np.int64) for i, t in enumerate(working_tracks)
    ])
    keep_np = _track_points_far_from_anchors_mask(flat_zyx_np, anchor_tree, exclusion_radius)
    flat_zyx_np = flat_zyx_np[keep_np]
    track_id_np = track_id_np[keep_np]
    num_tracks_orig = len(working_tracks)
    new_lengths = np.bincount(track_id_np, minlength=num_tracks_orig)
    surviving = np.where(new_lengths >= 2)[0]
    print(f'kept {len(surviving)} / {len(working_tracks)} tracks')
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
    print(
        f'track radius loss: {len(surviving)}/{num_tracks_orig} tracks survive exclusion '
        f'(radius {exclusion_radius:.1f}); {int(lengths_new.sum())} points retained'
    )
    return finish_prepared(flat_zyx_np, lengths_new, surviving)


def _draw_unique_track_point_indices(track_lengths, num_points, generator=None):
    """Uniformly draw fixed-size point sets without replacement per track."""
    sample_count = int(track_lengths.numel())
    device = track_lengths.device
    random_values = torch.rand(
        [sample_count, num_points], device=device, generator=generator)
    selected = torch.empty(
        [sample_count, num_points], dtype=torch.int64, device=device)

    # Batched Floyd sampling. For column c, draw from [0, L-k+c]. If that
    # value is already present in the row, use the new upper endpoint instead.
    # This produces a uniform k-subset using O(k) random values per track.
    first_upper = track_lengths - num_points
    for column in range(num_points):
        upper = first_upper + column
        candidate = (
            random_values[:, column] * (upper + 1).to(torch.float32)
        ).to(torch.int64)
        if column:
            duplicate = (selected[:, :column] == candidate[:, None]).any(dim=1)
            candidate = torch.where(duplicate, upper, candidate)
        selected[:, column] = candidate
    return torch.sort(selected, dim=-1).values


def _draw_track_sample(
        prepared_tracks, eligible_track_indices, k, num_points_per_track,
        generator=None):
    # GPU index math, one forced D2H of the flat indices, the big host-side
    # coordinate gather, and the H2D upload. Under prefetch this runs on the
    # worker thread for the next step, so the syncs stall the worker rather
    # than the training loop.
    flat_zyx_cpu = prepared_tracks['flat_zyx_cpu']
    offsets = prepared_tracks['offsets']
    lengths = prepared_tracks['lengths']
    device = prepared_tracks['device']
    num_eligible = int(eligible_track_indices.numel())
    sampling_probabilities = prepared_tracks.get('sampling_probabilities')
    if sampling_probabilities is None:
        eligible_choice = torch.randint(
            num_eligible, (k,), device=device, generator=generator)
        track_idx = eligible_track_indices[eligible_choice]
    else:
        eligible_probabilities = sampling_probabilities[eligible_track_indices]
        probability_sum = eligible_probabilities.sum()
        if float(probability_sum) <= 0:
            raise ValueError(
                'track_length_bin_weights assign no probability to tracks long enough '
                'for track_num_points_per_step')
        track_idx = torch.multinomial(
            eligible_probabilities / probability_sum,
            k, replacement=True, generator=generator)
        track_idx = eligible_track_indices[track_idx]

    crossing_partners = prepared_tracks.get('crossing_partners')
    if crossing_partners is not None and crossing_partners.shape[1] > 0:
        partners = crossing_partners[track_idx]
        partners = partners[partners >= 0]
        partners = partners[lengths[partners] >= num_points_per_track]
        if partners.numel() > 0:
            track_idx = torch.cat([track_idx, partners])

    sample_count = int(track_idx.numel())
    track_lengths_sample = lengths[track_idx]
    track_offsets_sample = offsets[track_idx]
    point_idx_within = _draw_unique_track_point_indices(
        track_lengths_sample, num_points_per_track, generator)
    flat_idx = (track_offsets_sample[:, None] + point_idx_within).reshape(-1)
    flat_idx_cpu = flat_idx.to(device='cpu')
    sampled_cpu = flat_zyx_cpu[flat_idx_cpu].view(sample_count, num_points_per_track, 3)
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
    return track_idx, point_idx_within, sampled_scroll


def _sample_prepared_track_points(prepared_tracks, num_tracks_per_step, num_points_per_track):
    lengths = prepared_tracks['lengths']
    device = prepared_tracks['device']
    num_tracks = int(lengths.numel())
    if num_tracks == 0 or num_tracks_per_step <= 0 or num_points_per_track <= 0:
        return None
    eligibility_cache = prepared_tracks.setdefault('eligibility_cache', {})
    eligible_track_indices = eligibility_cache.get(int(num_points_per_track))
    if eligible_track_indices is None:
        eligible_track_indices = torch.nonzero(
            lengths >= int(num_points_per_track), as_tuple=False).squeeze(-1)
        eligibility_cache[int(num_points_per_track)] = eligible_track_indices
        if int(eligible_track_indices.numel()) != num_tracks:
            print(
                f'track point sampling: {int(eligible_track_indices.numel())}/{num_tracks} '
                f'tracks have at least {int(num_points_per_track)} unique points'
            )
    num_eligible = int(eligible_track_indices.numel())
    if num_eligible == 0:
        return None
    k = min(int(num_tracks_per_step), num_eligible)

    if prefetch.prefetch_enabled() and device.type == 'cuda':
        pf = prefetch.get_prefetcher()
        generator = pf.torch_rng('tracks', device)
        return pf.pop_or_run(
            ('tracks', k, num_points_per_track),
            lambda: _draw_track_sample(
                prepared_tracks, eligible_track_indices, k,
                num_points_per_track, generator),
        )
    return _draw_track_sample(
        prepared_tracks, eligible_track_indices, k, num_points_per_track)


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


def iter_track_losses(slice_to_spiral_transform, dr_per_winding, prepared_tracks, cfg, compute_dt=True, dt_max_winding=None, dt_target_cache=None):
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
    track_idx, sample_local_idx, sampled_scroll = sample
    k = sampled_scroll.shape[0]
    num_points = sampled_scroll.shape[1]
    sampled_spiral = slice_to_spiral_transform(sampled_scroll.reshape(-1, 3)).reshape(k, num_points, 3)
    theta, _, shifted_radii = get_theta_and_radii(sampled_spiral[..., 1:], dr_per_winding)
    shifted_radii, crossing_adjustments = unwrap_shifted_radii(
        theta, shifted_radii, dr_per_winding,
    )
    dt_hinge_margin = dr_per_winding.detach() * cfg['track_dt_loss_margin']
    radius_loss = _same_radius_loss_for_shifted_radii(shifted_radii, dr_per_winding, cfg)
    if diagnostics_enabled():
        if cfg['track_radius_target'] == 'median':
            diagnostic_radius_target = shifted_radii.median(dim=-1, keepdim=True).values
        else:
            diagnostic_radius_target = shifted_radii.mean(dim=-1, keepdim=True)
        diagnostic_radius = F.relu(
            (shifted_radii - diagnostic_radius_target).abs()
            - dr_per_winding.detach() * cfg['track_radius_loss_margin'])
        diagnostic_target_radii = radius_from_unwrapped_shifted(
            theta, diagnostic_radius_target, crossing_adjustments,
            dr_per_winding,
        )
        diagnostic_target_spiral = torch.stack([
            sampled_spiral[..., 0],
            torch.sin(theta) * diagnostic_target_radii,
            torch.cos(theta) * diagnostic_target_radii,
        ], dim=-1).detach()
        record_loss_samples(
            'track_radius', sampled_spiral, diagnostic_radius,
            display_spiral_zyx=diagnostic_target_spiral,
        )

    if not compute_dt:
        yield 'track_radius', radius_loss
        del radius_loss, sampled_spiral, theta, shifted_radii, crossing_adjustments
        yield 'track_dt', zero
        return

    target_shifted_radii = strip_dt_target_in_sample_frame(
        shifted_radii, sample_local_idx, theta, crossing_adjustments,
        dr_per_winding, dt_target_cache, track_idx,
    )
    target_radii = radius_from_unwrapped_shifted(
        theta, target_shifted_radii, crossing_adjustments, dr_per_winding,
    )
    target_spiral_zyxs = torch.stack([
        sampled_spiral[..., 0],
        torch.sin(theta) * target_radii,
        torch.cos(theta) * target_radii,
    ], dim=-1).detach()
    active_mask = _progressive_dt_active_mask(target_shifted_radii.squeeze(-1), dr_per_winding, dt_max_winding)

    yield 'track_radius', radius_loss
    # The caller has now released the radius graph.  Keep only detached DT
    # inputs before constructing the inverse-transform graph.
    del radius_loss, sampled_spiral, theta, shifted_radii, crossing_adjustments
    del target_radii, target_shifted_radii
    target_scroll_zyxs = slice_to_spiral_transform.inv(target_spiral_zyxs.reshape(-1, 3)).reshape(*target_spiral_zyxs.shape)

    within_p = cfg['track_dt_within_track_norm_p']
    across_p = cfg['track_dt_norm_p']
    point_distances = torch.linalg.norm(sampled_scroll - target_scroll_zyxs, dim=-1)
    point_distances = F.relu(point_distances - dt_hinge_margin) + 1.e-5
    track_losses = (point_distances ** within_p).mean(dim=-1) ** (1 / within_p)
    dt_loss = _aggregate_dt_track_losses(track_losses, across_p, active_mask)
    record_loss_samples(
        'track_dt', target_spiral_zyxs, point_distances,
        active_mask[..., None] if active_mask is not None else None,
    )

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
