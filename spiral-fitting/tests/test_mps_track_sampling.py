import unittest

import numpy as np
import torch

from tracks import (
    _mps_shape_bucket,
    _sample_prepared_track_points,
    _transform_shape_bucketed,
    prepare_main_phase_tracks,
)


def line_track(length, *, z=10, y=10, axis=2):
    points = np.zeros((int(length) + 1, 3), dtype=np.float32)
    points[:, 0] = z
    points[:, 1] = y
    points[:, axis] = np.arange(int(length) + 1, dtype=np.float32)
    return points


@unittest.skipUnless(
    torch.backends.mps.is_available(), 'requires Apple-silicon MPS')
class MpsTrackSamplingTests(unittest.TestCase):
    """Weighted track sampling must work on MPS.

    On torch 2.11 MPS, ``torch.multinomial`` over a fit-sized track set
    aborts the process (it asks Metal for an ~83 GB buffer), so the draw
    runs on the CPU and follows the globally seeded CPU RNG.
    """

    def test_mps_weighted_sampling_is_reproducible(self):
        tracks = [line_track(20, y=10), line_track(20, y=20)]
        prepared = prepare_main_phase_tracks(tracks, None, 0.0, 'mps')
        prepared['sampling_probabilities'] = torch.full(
            (len(tracks),), 0.5, device='mps')
        draws = []
        for _ in range(2):
            torch.manual_seed(7)
            draws.append([
                _sample_prepared_track_points(prepared, 8, 4)[
                    'track_idx'][:8].tolist()
                for _ in range(4)])
        self.assertEqual(draws[0], draws[1])


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
