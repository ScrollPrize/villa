import types
import unittest
import numpy as np
import torch


class DevicePatchAtlasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # fit_spiral is import-heavy (wandb, zarr, ...); load it once for the
        # class rather than at test-module import.
        from fit_spiral import PatchAtlas
        cls.PatchAtlas = PatchAtlas

    @staticmethod
    def _fake_patch(height, width, seed):
        generator = torch.Generator().manual_seed(seed)
        return types.SimpleNamespace(
            zyxs=torch.rand(height, width, 3, generator=generator) * 100.,
            _sampling_valid_quad_mask_np=np.ones((height - 1, width - 1), dtype=bool),
            _sampling_2d_path=None,
        )

    @staticmethod
    def _manual_bilinear(grid, i, j):
        i0, j0 = int(np.floor(i)), int(np.floor(j))
        di, dj = i - i0, j - j0
        top = grid[i0, j0] * (1 - dj) + grid[i0, j0 + 1] * dj
        bottom = grid[i0 + 1, j0] * (1 - dj) + grid[i0 + 1, j0 + 1] * dj
        return top * (1 - di) + bottom * di

    def test_cpu_fallback_lookup_matches_manual_bilinear(self):
        patches = {'a': self._fake_patch(5, 7, 0), 'b': self._fake_patch(9, 4, 1)}
        atlas = self.PatchAtlas(patches, device='cpu')
        self.assertIsNone(atlas.zyxs_flat)
        self.assertEqual(atlas.offsets.device.type, 'cpu')

        idx = torch.tensor([0, 1, 1, 0])
        ijs = torch.tensor([[0.25, 0.75], [3.5, 1.25], [7.0, 2.0], [3.25, 5.5]])
        out = atlas.lookup(idx, ijs)
        expected = torch.stack([
            self._manual_bilinear(patches[key].zyxs, float(ij[0]), float(ij[1]))
            for key, ij in zip(['a', 'b', 'b', 'a'], ijs)
        ])
        torch.testing.assert_close(out, expected)
        self.assertIsNone(atlas.zyxs_flat)
        atlas.materialize()
        self.assertEqual(atlas.zyxs_flat.device.type, 'cpu')
        torch.testing.assert_close(atlas.lookup(idx, ijs), expected)

    def test_replacement_reorders_shared_geometry_and_removes_patches(self):
        a, b, c = [self._fake_patch(5, width, seed)
                   for width, seed in [(7, 71), (4, 72), (6, 73)]]
        original = self.PatchAtlas({'a': a, 'b': b}, device='cpu').materialize()
        original = original.replaced({'a': a, 'b': b, 'c': c})
        replacement = self._fake_patch(6, 8, 74)
        candidate = original.replaced({'c': c, 'a': a, 'new': replacement})
        self.assertEqual(original.id_to_idx, {'a': 0, 'b': 1, 'c': 2})
        self.assertIs(candidate._geometry_chunks[0]['zyxs_flat'],
                      original._geometry_chunks[1]['zyxs_flat'])
        self.assertIs(candidate._geometry_chunks[1]['zyxs_flat'],
                      original._geometry_chunks[0]['zyxs_flat'])
        for atlas, patches in [(candidate, [c, a, replacement]),
                               (candidate.replaced({'a': a, 'c': c}), [a, c])]:
            indices = torch.arange(len(patches))
            ijs = torch.tensor([[1.5, 2.25]] * len(patches))
            expected = torch.stack([self._manual_bilinear(p.zyxs, 1.5, 2.25)
                                    for p in patches])
            torch.testing.assert_close(atlas.lookup(indices, ijs), expected)
            vertex_ids = atlas.offsets[:-1] + atlas.widths + 2
            torch.testing.assert_close(atlas.vertex_zyxs(vertex_ids),
                                       torch.stack([p.zyxs[1, 2] for p in patches]))
        empty = candidate.replaced({})
        empty = empty.replaced({'b': b})
        torch.testing.assert_close(
            empty.lookup(torch.tensor([0]), torch.tensor([[1.5, 2.25]]))[0],
            self._manual_bilinear(b.zyxs, 1.5, 2.25))

    def test_patch_atlas_registers_potential_for_every_valid_quad(self):
        from theta_crossing_map import ThetaCrossingMap

        patch = self._fake_patch(6, 8, 13)
        # A connected ragged mask exercises the DFS tree rather than relying on
        # a rectangular row walk. Geometry uses a smooth theta ramp so every
        # non-tree edge agrees with the cached lift.
        mask = np.ones((5, 7), dtype=bool)
        mask[0, 5:] = False
        mask[1, 6] = False
        patch._sampling_valid_quad_mask_np = mask
        theta = torch.linspace(5.5, 7.2, patch.zyxs.shape[1])
        radius = torch.full_like(theta, 30.0)
        patch.zyxs[..., 0] = torch.arange(
            patch.zyxs.shape[0], dtype=torch.float32)[:, None]
        patch.zyxs[..., 1] = torch.sin(theta)[None, :] * radius
        patch.zyxs[..., 2] = torch.cos(theta)[None, :] * radius

        atlas = self.PatchAtlas({'p': patch}, device='cpu')
        crossing_map = ThetaCrossingMap('cpu', chunk_size=4)
        atlas.register_theta_topology(crossing_map)
        crossing_map.force_refresh(lambda value: value)

        node_ids = atlas.theta_node_ids(
            np.zeros(int(mask.sum()), dtype=np.int64), np.argwhere(mask))
        potentials = crossing_map.winding_potentials(node_ids)
        self.assertEqual(potentials.numel(), int(mask.sum()))
        self.assertTrue(bool((potentials != crossing_map._unset_potential).all()))
        self.assertEqual(crossing_map.potential_consistency()['inconsistent_edges'], 0)
