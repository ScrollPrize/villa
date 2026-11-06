import math
import unittest

import torch

from dinovol_2.model.rope import CompassRopePositionEmbedding


class CompassInitializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_initial_directional_energy_has_no_systematic_axis_bias(self):
        # Average across heads: individual heads need not be isotropic. The old
        # radius/azimuth coupling survives averaging and exceeds this tolerance.
        with torch.random.fork_rng(devices=[]):
            for ndim in (2, 3):
                for head_dim in (48, 64):
                    for use_dft in (False, True):
                        with self.subTest(ndim=ndim, head_dim=head_dim, use_dft=use_dft):
                            torch.manual_seed(0)
                            rope = CompassRopePositionEmbedding(
                                head_dim, ndim=ndim, num_heads=512,
                                dtype=torch.float64, use_dft=use_dft,
                            )
                            # Recover effective frequencies through public phases.
                            # Small coordinates keep all phases within (-pi, pi).
                            step = 0.01
                            coords = torch.eye(ndim, dtype=torch.float64) * step
                            with torch.no_grad():
                                sin, cos = rope.get_embed_from_coords(coords)
                            frequencies = (
                                torch.atan2(sin, cos)[..., :head_dim // 2]
                                / (2 * math.pi * step)
                            ).transpose(-1, -2)
                            gram = frequencies.transpose(-1, -2) @ frequencies
                            energy = gram.diagonal(dim1=-2, dim2=-1).sum(-1)
                            mean_gram = (gram / energy[:, None, None]).mean(0)
                            torch.testing.assert_close(
                                mean_gram, torch.eye(ndim, dtype=torch.float64) / ndim,
                                rtol=0, atol=0.05,
                            )

    def test_initialization_preserves_periods_and_angular_stratification(self):
        with torch.random.fork_rng(devices=[]):
            for ndim in (2, 3):
                with self.subTest(ndim=ndim):
                    torch.manual_seed(0)
                    rope = CompassRopePositionEmbedding(
                        48, ndim=ndim, num_heads=8, dtype=torch.float64,
                        radial_jitter=0,
                    )
                    torch.testing.assert_close(
                        rope.radii, rope.periods.reciprocal().expand_as(rope.radii),
                        rtol=0, atol=0,
                    )
                    expected = torch.arange(rope.num_pairs).expand(rope.num_heads, -1)
                    azimuth_bins = (rope.angles[..., 0] * rope.num_pairs / (2 * math.pi)).long()
                    torch.testing.assert_close(azimuth_bins.sort(-1).values, expected)
                    if ndim == 3:
                        polar_bins = ((rope.angles[..., 1].cos() + 1) * rope.num_pairs / 2).long()
                        torch.testing.assert_close(polar_bins.sort(-1).values, expected)


if __name__ == "__main__":
    unittest.main()
