import dbm
import pickle
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from build_track_crossings import build_cache
from tracks import (
    _load_native_track_crossings,
    _mps_shape_bucket,
    _sample_prepared_track_points,
    _transform_shape_bucketed,
    load_track_crossing_cache,
    load_tracks_from_dbm,
    prepare_main_phase_tracks,
)


def line_track(length, *, z=10, y=10, axis=2):
    points = np.zeros((int(length) + 1, 3), dtype=np.int32)
    points[:, 0] = z
    points[:, 1] = y
    points[:, axis] = np.arange(int(length) + 1, dtype=np.int32)
    return points


@unittest.skipUnless(
    torch.backends.mps.is_available(), 'requires Apple-silicon MPS')
@unittest.skipIf(
    _load_native_track_crossings() is None,
    'requires the native vc_spiral.track_crossings extension')
class MpsTrackSamplingTests(unittest.TestCase):
    """Weighted track sampling with crossings must work on MPS.

    On torch 2.11 MPS, ``torch.multinomial`` over a fit-sized track set
    aborts the process (it asks Metal for an ~83 GB buffer), and
    ``torch.randint(0, 2**63 - 1)`` returns values wrapped to int32, about
    half of them negative, which the native sampler's ``uint64_t seed``
    rejects with a TypeError.
    """

    def prepare(self, root):
        path = Path(root) / 'tracks.dbm'
        horizontal = line_track(20, z=10, y=10, axis=2)
        vertical = line_track(20, z=10, y=0, axis=1)
        vertical[:, 2] = 10
        with dbm.open(str(path), 'c') as database:
            database[b'h:0'] = pickle.dumps([horizontal])
            database[b'vy:0'] = pickle.dumps([vertical])
        build_cache(path, show_progress=False)
        tracks, families, source_ids = load_tracks_from_dbm(
            path, 0, 20, return_families=True,
            return_source_ids=True, show_progress=False)
        prepared = prepare_main_phase_tracks(
            tracks, None, 0.0, 'mps',
            sampling_config={
                'track_crossing_precompute_max': 1,
                'track_max_track_crossing_per_step': 1,
            },
            track_families=families,
            track_source_ids=source_ids,
            crossing_cache=load_track_crossing_cache(path),
        )
        return prepared, families

    def test_weighted_sampling_with_crossings_on_mps(self):
        with tempfile.TemporaryDirectory() as temporary:
            prepared, families = self.prepare(temporary)
            horizontal = families.index('horizontal')
            vertical = families.index('vertical')
            probabilities = torch.zeros(len(families), device='mps')
            probabilities[horizontal] = 1
            prepared['sampling_probabilities'] = probabilities
            torch.manual_seed(1)
            # Each draw takes a fresh crossing seed; without the fix about
            # half of them are negative, so 16 draws all but guarantee one.
            for _ in range(16):
                sample = _sample_prepared_track_points(prepared, 1, 4)
                self.assertEqual(sample['track_idx'].device.type, 'mps')
                self.assertEqual(
                    sample['track_idx'].tolist(), [horizontal, vertical])

    def test_mps_weighted_sampling_is_reproducible(self):
        with tempfile.TemporaryDirectory() as temporary:
            prepared, families = self.prepare(temporary)
            prepared['sampling_probabilities'] = torch.full(
                (len(families),), 0.5, device='mps')
            draws = []
            for _ in range(2):
                torch.manual_seed(7)
                draws.append([
                    _sample_prepared_track_points(prepared, 8, 4)[
                        'track_idx'][:8].tolist()
                    for _ in range(4)])
            self.assertEqual(draws[0], draws[1])


@unittest.skipUnless(
    torch.backends.mps.is_available(), 'requires Apple-silicon MPS')
@unittest.skipIf(
    _load_native_track_crossings() is None,
    'requires the native vc_spiral.track_crossings extension')
class HostPlannedSampleTests(unittest.TestCase):
    """tracks._host_planned_track_sample must give the eager batch exactly.

    A grid of crossing horizontal and vertical tracks of several lengths, so
    batches have partners, rows of different lengths and repeated target
    positions; every field of every draw must be equal, integer for integer
    and bit for bit, with the same RNG state.
    """

    def prepare(self, root):
        path = Path(root) / 'tracks.dbm'
        with dbm.open(str(path), 'c') as database:
            for index in range(6):
                horizontal = line_track(30 + 17 * index, z=10, y=4 + 9 * index, axis=2)
                database[f'h:{index}'.encode()] = pickle.dumps([horizontal])
                vertical = line_track(25 + 13 * index, z=10, y=0, axis=1)
                vertical[:, 2] = 3 + 11 * index
                database[f'vy:{index}'.encode()] = pickle.dumps([vertical])
        build_cache(path, show_progress=False)
        tracks, families, source_ids = load_tracks_from_dbm(
            path, 0, 20, return_families=True,
            return_source_ids=True, show_progress=False)
        return prepare_main_phase_tracks(
            tracks, None, 0.0, 'mps',
            sampling_config={
                'track_crossing_precompute_max': 3,
                'track_max_track_crossing_per_step': 3,
            },
            track_families=families,
            track_source_ids=source_ids,
            crossing_cache=load_track_crossing_cache(path),
        )

    def draws(self, prepared, eager):
        import os
        previous = os.environ.get('FIT_SPIRAL_METAL_TRACKS')
        os.environ['FIT_SPIRAL_METAL_TRACKS'] = '0' if eager else '1'
        try:
            torch.manual_seed(11)
            out = []
            for _ in range(6):
                sample = _sample_prepared_track_points(
                    prepared, 9, 7, min_sample_spacing=2.0,
                    max_sample_spacing=5.0)
                out.append({key: value.cpu() if torch.is_tensor(value) else value
                            for key, value in sample.items()})
            return out
        finally:
            if previous is None:
                del os.environ['FIT_SPIRAL_METAL_TRACKS']
            else:
                os.environ['FIT_SPIRAL_METAL_TRACKS'] = previous

    def test_host_planned_batches_equal_eager(self):
        with tempfile.TemporaryDirectory() as temporary:
            prepared = self.prepare(temporary)
            prepared['sampling_probabilities'] = torch.linspace(
                1.0, 2.0, int(prepared['lengths'].numel()), device='mps')
            eager = self.draws(prepared, eager=True)
            planned = self.draws(prepared, eager=False)
        partners = 0
        for left, right in zip(eager, planned):
            for key, value in left.items():
                other = right[key]
                if torch.is_tensor(value):
                    self.assertEqual(value.shape, other.shape, key)
                    self.assertTrue(torch.equal(value.to(other.dtype), other), key)
                else:
                    self.assertEqual(value, other, key)
            partners += int(left['partner_rows'].numel())
        # the comparison covered crossing partners, not only primary rows
        self.assertGreater(partners, 0)


class ShapeBucketTests(unittest.TestCase):
    def test_buckets_bound_padding_and_distinct_shapes(self):
        for count in [1, 1024, 1025, 4097, 199_240, 206_869, 10**7]:
            bucket = _mps_shape_bucket(count)
            self.assertGreaterEqual(bucket, count)
            if count > 1024:
                self.assertLessEqual(bucket - count, count / 16)
        # A real fit draws ~200k track points per step with a few % jitter;
        # all of those must map onto a handful of shapes.
        sizes = {_mps_shape_bucket(n) for n in range(190_000, 215_000)}
        self.assertLessEqual(len(sizes), 4)

    def test_non_mps_points_pass_through_unpadded(self):
        seen = []

        def transform(points):
            seen.append(points.shape[0])
            return points * 2

        points = torch.randn(1500, 3)
        torch.testing.assert_close(
            _transform_shape_bucketed(transform, points), points * 2)
        self.assertEqual(seen, [1500])

    @unittest.skipUnless(
        torch.backends.mps.is_available(), 'requires Apple-silicon MPS')
    def test_mps_padding_is_invisible_to_values_and_gradients(self):
        seen = []

        def transform(points):
            seen.append(points.shape[0])
            return torch.sin(points) * points.sum(dim=-1, keepdim=True)

        cpu = torch.randn(1500, 3, dtype=torch.float32)
        points = cpu.to('mps').requires_grad_(True)
        out = _transform_shape_bucketed(transform, points)
        self.assertEqual(seen, [_mps_shape_bucket(1500)])
        self.assertEqual(tuple(out.shape), (1500, 3))
        out.square().sum().backward()

        reference = cpu.clone().requires_grad_(True)
        expected = transform(reference)
        expected.square().sum().backward()
        torch.testing.assert_close(out.detach().cpu(), expected.detach())
        torch.testing.assert_close(points.grad.cpu(), reference.grad)


if __name__ == '__main__':
    unittest.main()
