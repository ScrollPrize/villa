import unittest
import numpy as np
import torch
from tracks import (
    _pack_track_points,
    _sample_prepared_track_points,
    configure_prepared_track_sampling,
    iter_track_losses,
    prepare_main_phase_tracks,
)
from tracks import _crossing_row_alignments


def test_track_walk_samples_complete_crossing_tracks_deterministically():
    tracks = []
    for axis in (1, 2):
        for fixed in (10, 40, 70):
            points = np.zeros((81, 3), dtype=np.float32)
            points[:, 0] = 10
            points[:, axis] = np.arange(81)
            points[:, 3 - axis] = fixed
            tracks.append(points)
    prepared = prepare_main_phase_tracks(
        tracks, None, 0.0, 'cpu',
        track_families=['vertical'] * 3 + ['horizontal'] * 3,
        sampling_config={
            'track_crossing_mode': 'track_walk',
            'track_min_walk_steps_per_track': 24,
            'track_max_walk_steps_per_track': 40,
        },
    )
    torch.manual_seed(123)
    first = _sample_prepared_track_points(prepared, 2, 12)
    torch.manual_seed(123)
    repeated = _sample_prepared_track_points(prepared, 2, 12)
    for key in ('track_idx', 'sampled_scroll', 'group_id'):
        torch.testing.assert_close(first[key], repeated[key], rtol=0, atol=0)
    for group in range(2):
        selected = first['track_idx'][first['group_id'] == group]
        assert 3 <= len(selected) <= 5
        assert len(selected.unique()) == len(selected)
    for row, track in enumerate(first['track_idx']):
        samples = first['sampled_scroll'][first['row_id'] == row]
        torch.testing.assert_close(samples[0], torch.from_numpy(tracks[int(track)][0]))
        torch.testing.assert_close(samples[-1], torch.from_numpy(tracks[int(track)][-1]))
    torch.testing.assert_close(
        first['sampled_scroll'][first['primary_cross_flat']],
        first['sampled_scroll'][first['partner_cross_flat']],
    )


class CpuTrackStorageTests(unittest.TestCase):
    @staticmethod
    def _line_track(length, *, z=10, y=10, axis=2):
        points = np.zeros((int(length) + 1, 3), dtype=np.float32)
        points[:, 0] = z
        points[:, 1] = y
        points[:, axis] = np.arange(int(length) + 1, dtype=np.float32)
        return points

    def test_complete_track_sample_stays_between_20_and_60_voxel_spacing(self):
        track = self._line_track(125, y=10)
        prepared = prepare_main_phase_tracks([track], None, 0.0, 'cpu', sampling_config={'track_crossing_mode': 'count'})

        sample = _sample_prepared_track_points(
            prepared, 1, 24,
            min_sample_spacing=20.0, max_sample_spacing=60.0)
        points = sample['sampled_scroll']
        spacing = torch.linalg.norm(torch.diff(points, dim=0), dim=-1)

        torch.testing.assert_close(points[0], torch.from_numpy(track[0]))
        torch.testing.assert_close(points[-1], torch.from_numpy(track[-1]))
        self.assertLess(len(points), 24)
        self.assertGreaterEqual(float(spacing.min()), 20.0)
        self.assertLessEqual(float(spacing.max()), 60.0)

    def test_length_bin_weights_are_distributed_within_tertiles(self):
        tracks = [self._line_track(length, y=length * 2) for length in range(1, 10)]
        prepared = prepare_main_phase_tracks(
            tracks, None, 0.0, 'cpu',
            sampling_config={'track_crossing_mode': 'count',
                'track_length_bin_weights': [0.15, 0.25, 0.60],
                'track_max_tortuosity': None,
                'track_max_track_crossing_per_step': 0,
            },
        )

        probabilities = prepared['sampling_probabilities'].numpy()
        np.testing.assert_allclose([
            probabilities[:3].sum(),
            probabilities[3:6].sum(),
            probabilities[6:].sum(),
        ], [0.15, 0.25, 0.60], rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(probabilities[:3], np.full(3, 0.05))

    def test_tortuosity_filter_is_opt_in_and_uses_arclength_over_chord(self):
        straight = np.array([
            [10, 10, 0], [10, 10, 5], [10, 10, 10],
        ], dtype=np.float32)
        tortuous = np.array([
            [10, 10, 0], [10, 13, 0], [10, 13, 4], [10, 10, 4],
        ], dtype=np.float32)

        unfiltered = prepare_main_phase_tracks(
            [straight, tortuous], None, 0.0, 'cpu',
            sampling_config={'track_crossing_mode': 'count', 'track_max_tortuosity': None},
        )
        filtered = prepare_main_phase_tracks(
            [straight, tortuous], None, 0.0, 'cpu',
            sampling_config={'track_crossing_mode': 'count', 'track_max_tortuosity': 2.0},
        )

        self.assertEqual(unfiltered['lengths'].numel(), 2)
        self.assertEqual(filtered['lengths'].numel(), 1)
        np.testing.assert_array_equal(filtered['flat_zyx_cpu'].numpy(), straight)

    def test_crossing_partners_are_sampled_from_all_exact_partners(self):
        primary = self._line_track(20, z=10, y=10, axis=2)

        def vertical_at(x):
            track = np.zeros((21, 3), dtype=np.float32)
            track[:, 0] = 10
            track[:, 1] = np.arange(21, dtype=np.float32)
            track[:, 2] = x
            return track

        tracks = [
            primary,
            vertical_at(4),
            vertical_at(10),
            vertical_at(16),
            primary.copy(),  # Opposite provenance, but parallel: reject it.
        ]
        prepared = prepare_main_phase_tracks(
            tracks, None, 0.0, 'cpu',
            sampling_config={'track_crossing_mode': 'count',
                'track_length_bin_weights': None,
                'track_max_tortuosity': None,
                'track_crossing_precompute_max': 2,
                'track_max_track_crossing_per_step': 2,
            },
            track_families=['horizontal', 'vertical', 'vertical', 'vertical', 'vertical'],
        )

        self.assertIn('crossing_index', prepared)
        self.assertEqual(
            int(prepared['crossing_index_stats']['directed_crossings']), 6)

        configure_prepared_track_sampling(prepared, {
            'track_max_track_crossing_per_step': 1,
        })

        # Force primary track zero so the first draw uses the Run-scoped limit.
        prepared['sampling_probabilities'] = torch.tensor([1., 0., 0., 0., 0.])
        torch.manual_seed(123)
        first = _sample_prepared_track_points(prepared, 1, 4)
        torch.manual_seed(123)
        repeated = _sample_prepared_track_points(prepared, 1, 4)
        np.testing.assert_array_equal(
            first['track_idx'].numpy(), repeated['track_idx'].numpy())
        self.assertEqual(first['track_idx'][0], 0)
        self.assertIn(int(first['track_idx'][1]), {1, 2, 3})
        sample = first
        self.assertEqual(sample['row_lengths'].shape, (2,))
        self.assertEqual(sample['group_id'].tolist(), [0, 0])
        self.assertEqual(sample['group_width'], 2)

        configure_prepared_track_sampling(prepared, {
            'track_max_track_crossing_per_step': 2,
        })
        sample = _sample_prepared_track_points(prepared, 1, 4)
        self.assertEqual(sample['track_idx'][0], 0)
        self.assertEqual(len(set(sample['track_idx'][1:].tolist())), 2)
        self.assertTrue(set(sample['track_idx'][1:].tolist()) <= {1, 2, 3})
        self.assertEqual(sample['group_width'], 3)
        for primary_flat, partner_flat in zip(
                sample['primary_cross_flat'], sample['partner_cross_flat']):
            torch.testing.assert_close(
                sample['sampled_scroll'][primary_flat],
                sample['sampled_scroll'][partner_flat],
            )

        configure_prepared_track_sampling(prepared, {
            'track_max_track_crossing_per_step': 1,
        })
        observed = set()
        for seed in range(32):
            torch.manual_seed(seed)
            draw = _sample_prepared_track_points(prepared, 1, 4)
            observed.add(int(draw['track_idx'][1]))
        self.assertEqual(observed, {1, 2, 3})

    def test_track_point_packing_is_chunk_independent(self):
        points = np.array([
            [0, 0, 0],
            [1, 2, 3],
            [(1 << 20) - 1, (1 << 20) - 2, (1 << 20) - 3],
        ], dtype=np.float32)
        expected = (
            points[:, 0].astype(np.uint64) << np.uint64(40)
            | points[:, 1].astype(np.uint64) << np.uint64(20)
            | points[:, 2].astype(np.uint64)
        )
        np.testing.assert_array_equal(
            _pack_track_points(points, chunk_size=1), expected)
        np.testing.assert_array_equal(
            _pack_track_points(points, chunk_size=len(points)), expected)

    def test_staged_track_backward_matches_combined_backward(self):
        class Translation:
            def __init__(self, parameter, sign=1.0):
                self.parameter = parameter
                self.sign = sign

            def __call__(self, points):
                return points + self.parameter * self.sign

            @property
            def inv(self):
                return Translation(self.parameter, -self.sign)

        tracks = [
            np.array([[1, 5, 8], [2, 6, 9], [3, 7, 10], [4, 8, 11]], dtype=np.float32),
            np.array([[5, 9, 12], [6, 10, 13], [7, 11, 14], [8, 12, 15]], dtype=np.float32),
        ]
        prepared = prepare_main_phase_tracks(tracks, None, 0.0, 'cpu', sampling_config={'track_crossing_mode': 'count'})
        config = {
            'sample_count_tracks_per_step': 2,
            'sample_count_track_points_per_step': 4,
            'track_radius_loss_margin': 0.025,
            'track_radius_target': 'mean',
            'track_radius_within_norm_p': 3.0,
            'track_dt_loss_margin': 0.025,
            'track_dt_within_track_norm_p': 3.0,
            'track_dt_norm_p': 0.5,
        }
        dr = torch.tensor(10.0)

        combined_parameter = torch.tensor(0.2, requires_grad=True)
        torch.manual_seed(12)
        combined_parts = list(iter_track_losses(
            Translation(combined_parameter), dr, prepared, config, compute_dt=True,
        ))
        sum(value for _, value in combined_parts).backward()

        staged_parameter = torch.tensor(0.2, requires_grad=True)
        torch.manual_seed(12)
        staged_parts = []
        for name, value in iter_track_losses(
            Translation(staged_parameter), dr, prepared, config, compute_dt=True,
        ):
            staged_parts.append((name, value.detach()))
            value.backward()

        self.assertEqual([name for name, _ in staged_parts], ['track_radius', 'track_dt'])
        torch.testing.assert_close(
            torch.stack([value for _, value in staged_parts]),
            torch.stack([value.detach() for _, value in combined_parts]),
        )
        torch.testing.assert_close(staged_parameter.grad, combined_parameter.grad)


class TrackWalkConfigurationTests(unittest.TestCase):
    def test_chain_alignment_propagates_through_every_hop(self):
        radii = torch.tensor([10.0, 2.0, 5.0, 1.0, 7.0, 3.0])
        alignment = _crossing_row_alignments(
            radii,
            torch.tensor([0, 2, 4]),
            torch.tensor([1, 3, 5]),
            torch.tensor([1, 2, 3]),
            row_count=4,
            chain=True,
            edge_group_id=torch.tensor([0, 0, 0]),
            edge_slot=torch.tensor([0, 1, 2]),
            group_count=1,
            maximum_hops=3)
        # +8 aligns row 1; row 2 then sees 5+8 at its source crossing,
        # and row 3 likewise inherits the full preceding-chain offset.
        torch.testing.assert_close(
            alignment, torch.tensor([0.0, 8.0, 12.0, 16.0]))

    def test_chain_alignment_vectorizes_independent_groups(self):
        radii = torch.tensor([
            10.0, 2.0, 5.0, 1.0,
            20.0, 19.0, 30.0, 27.0,
        ])
        alignment = _crossing_row_alignments(
            radii,
            torch.tensor([0, 2, 4, 6]),
            torch.tensor([1, 3, 5, 7]),
            torch.tensor([1, 2, 4, 5]),
            row_count=6,
            chain=True,
            edge_group_id=torch.tensor([0, 0, 1, 1]),
            edge_slot=torch.tensor([0, 1, 0, 1]),
            group_count=2,
            maximum_hops=2)
        torch.testing.assert_close(
            alignment, torch.tensor([0.0, 8.0, 12.0, 0.0, 1.0, 4.0]))
