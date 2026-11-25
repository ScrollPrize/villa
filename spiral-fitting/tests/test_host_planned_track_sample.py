"""tracks._host_planned_track_sample must give the eager batch exactly.

On MPS the track batch is planned on the host and built by one Metal
dispatch (track_loss_metal.build_track_sample) instead of ~15 eager ops.
A grid of crossing horizontal and vertical tracks of several lengths gives
batches with crossing partners, rows of different lengths and repeated
target positions; with the same RNG state, every field of every draw must
equal the eager path's, integer for integer and bit for bit.
"""
import numpy as np
import pytest
import torch

from tracks import (_load_native_track_crossings,
                    _sample_prepared_track_points, prepare_main_phase_tracks)

pytestmark = [
    pytest.mark.skipif(not torch.backends.mps.is_available(), reason='needs MPS'),
    pytest.mark.skipif(_load_native_track_crossings() is None,
                       reason='needs the native vc_spiral.track_crossings extension'),
]

ROWS = 6


def _grid():
    """Horizontal tracks along x at y = 4 + 9i and vertical ones along y at
    x = 3 + 11j, all at z = 10, with the reciprocal crossing CSR of a
    TrackGraph (rows are tracks in source-id order)."""
    horizontal = [(4 + 9 * i, 30 + 17 * i) for i in range(ROWS)]   # (y, length)
    vertical = [(3 + 11 * j, 25 + 13 * j) for j in range(ROWS)]    # (x, length)
    tracks = []
    for y, length in horizontal:
        points = np.zeros((length + 1, 3), dtype=np.int32)
        points[:, 0], points[:, 1], points[:, 2] = 10, y, np.arange(length + 1)
        tracks.append(points)
    for x, length in vertical:
        points = np.zeros((length + 1, 3), dtype=np.int32)
        points[:, 0], points[:, 1], points[:, 2] = 10, np.arange(length + 1), x
        tracks.append(points)
    records = [[] for _ in tracks]
    for i, (y, h_length) in enumerate(horizontal):
        for j, (x, v_length) in enumerate(vertical):
            if x <= h_length and y <= v_length:
                h, v = i, ROWS + j
                records[h].append((v, x, y))
                records[v].append((h, y, x))
    offsets = np.zeros(len(tracks) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(r) for r in records])
    flat = [entry for row in records for entry in sorted(row)]
    graph = {
        'source_ids': np.arange(1, len(tracks) + 1, dtype=np.uint64),
        'offsets': offsets,
        'partners': np.asarray([p for p, _, _ in flat], dtype=np.int32),
        'self_local': np.asarray([s for _, s, _ in flat], dtype=np.int32),
        'partner_local': np.asarray([q for _, _, q in flat], dtype=np.int32),
    }
    return tracks, graph


def _prepare():
    tracks, graph = _grid()
    prepared = prepare_main_phase_tracks(
        tracks, None, 0.0, 'mps',
        sampling_config={
            'track_crossing_precompute_max': 3,
            'track_max_track_crossing_per_step': 3,
        },
        track_source_ids=graph['source_ids'], track_graph=graph)
    prepared['sampling_probabilities'] = torch.linspace(
        1.0, 2.0, len(tracks), device='mps')
    return prepared


def _draws(prepared, monkeypatch, eager):
    monkeypatch.setenv('FIT_SPIRAL_METAL_TRACKS', '0' if eager else '1')
    torch.manual_seed(11)
    out = []
    for _ in range(6):
        sample = _sample_prepared_track_points(
            prepared, 9, 7, min_sample_spacing=2.0, max_sample_spacing=5.0)
        out.append({key: value.cpu() if torch.is_tensor(value) else value
                    for key, value in sample.items()})
    return out


def test_host_planned_batches_equal_eager(monkeypatch):
    prepared = _prepare()
    assert prepared.get('crossing_index') is not None
    eager = _draws(prepared, monkeypatch, eager=True)
    planned = _draws(prepared, monkeypatch, eager=False)
    partners = 0
    for left, right in zip(eager, planned):
        assert 'partner_offsets' in right, 'the host-planned path did not run'
        assert set(right) - set(left) == {'partner_offsets'}
        # the fused loss walks group g's partner rows over
        # [partner_offsets[g], partner_offsets[g + 1]), so they must be
        # contiguous and in group order
        # track_idx covers partner rows too; the first num_groups rows are the primaries
        k = int(left['num_groups'])
        partner_group = left['group_id'][k:].to(torch.int64)
        assert torch.all(partner_group[1:] >= partner_group[:-1])
        expected = torch.searchsorted(
            partner_group, torch.arange(k + 1, dtype=torch.int64), side='left')
        assert torch.equal(right['partner_offsets'], expected)
        for key, value in left.items():
            other = right[key]
            if torch.is_tensor(value):
                assert value.shape == other.shape, key
                assert value.dtype == other.dtype or key in (
                    'primary_cross_flat', 'partner_cross_flat'), key
                assert torch.equal(value.to(other.dtype), other), key
            else:
                assert value == other, key
        partners += int(left['partner_rows'].numel())
    # the comparison covered crossing partners, not only primary rows
    assert partners > 0
