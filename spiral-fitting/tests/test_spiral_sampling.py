import unittest
from types import SimpleNamespace
import numpy as np
import dt_targets
from spiral_sampling import load_spiral_sampling
import losses
import torch

spiral_sampling = load_spiral_sampling()


class PatchSamplingBindingTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(
            spiral_sampling, 'vc_spiral.spiral_sampling is required for patch sampling')
        self.mask = np.ones((31, 37), dtype=bool)
        self.mask[8:14, 10:22] = False
        self.atlas = spiral_sampling.PatchSamplingAtlas([self.mask])

    def test_patch_points_are_deterministic_distinct_and_valid(self):
        indices = np.zeros(24, dtype=np.int64)
        first = self.atlas.sample_patch_points(indices, 40, 1234)
        second = self.atlas.sample_patch_points(indices, 40, 1234)
        np.testing.assert_array_equal(first['ijs'], second['ijs'])
        np.testing.assert_array_equal(first['counts'], second['counts'])
        ijs = np.asarray(first['ijs'])
        counts = np.asarray(first['counts'])
        self.assertEqual(ijs.shape, (24, 40, 2))
        np.testing.assert_array_equal(counts, 40)
        floors = np.floor(ijs).astype(np.int64)
        self.assertTrue(self.mask[floors[..., 0], floors[..., 1]].all())
        for row in floors:
            self.assertEqual(len(np.unique(row, axis=0)), 40)

    def test_small_patch_uses_every_cell_and_pads_with_valid_geometry(self):
        mask = np.zeros((4, 5), dtype=bool)
        mask[[0, 1, 3], [1, 4, 2]] = True
        atlas = spiral_sampling.PatchSamplingAtlas([mask])
        result = atlas.sample_patch_points(np.array([0]), 8, 77)
        ijs = np.asarray(result['ijs'])[0]
        self.assertEqual(int(np.asarray(result['counts'])[0]), 3)
        cells = np.floor(ijs).astype(np.int64)
        self.assertEqual(len(np.unique(cells[:3], axis=0)), 3)
        self.assertTrue(mask[cells[:, 0], cells[:, 1]].all())
        np.testing.assert_array_equal(ijs[3:], np.repeat(ijs[:1], 5, axis=0))

    def test_ragged_tree_and_streamed_neighbors_match_legacy_graph(self):
        import scipy.sparse

        mask = np.ones((8, 10), dtype=bool)
        mask[0, 7:] = False
        mask[2:5, 4] = False
        mask[6:, :2] = False
        node_map = np.full(mask.shape, -1, dtype=np.int64)
        node_map[mask] = np.arange(mask.sum())
        edges = []
        for di, dj in ((0, 1), (1, -1), (1, 0), (1, 1)):
            a = node_map[:mask.shape[0] - di,
                         max(0, -dj):mask.shape[1] - max(0, dj)]
            b = node_map[di:,
                         max(0, dj):mask.shape[1] - max(0, -dj)]
            valid = (a >= 0) & (b >= 0)
            if valid.any():
                edges.append(np.stack([a[valid], b[valid]], axis=1))
        expected_edges = np.concatenate(edges)
        graph = scipy.sparse.csr_matrix(
            (np.ones(len(expected_edges), dtype=np.int8),
             (expected_edges[:, 0], expected_edges[:, 1])),
            shape=(int(mask.sum()),) * 2)
        expected_order = scipy.sparse.csgraph.depth_first_order(
            graph, 0, directed=False, return_predecessors=False)

        atlas = spiral_sampling.PatchSamplingAtlas([mask])
        tree = atlas.tree_chunk(0, int(mask.sum()))
        np.testing.assert_array_equal(tree['node_ordinals'], expected_order)
        streamed = atlas.neighbor_chunk(0, int(mask.sum()) * 4)
        actual_edges = np.asarray(streamed['node_pairs'])
        np.testing.assert_array_equal(
            actual_edges[np.lexsort((actual_edges[:, 1], actual_edges[:, 0]))],
            expected_edges[np.lexsort((expected_edges[:, 1], expected_edges[:, 0]))])


@unittest.skipUnless(spiral_sampling is not None, "vc_spiral.spiral_sampling is not built")
class DtTargetBindingTests(unittest.TestCase):
    def setUp(self):
        self.previous_binding = dt_targets._spiral_sampling

    def tearDown(self):
        dt_targets._spiral_sampling = self.previous_binding

    def test_dt_sample_preparation_matches_python(self):
        mask = np.ones((53, 71), dtype=bool)
        mask[7:19, 21:36] = False
        mask[40:, :12] = False
        patch = SimpleNamespace(
            _sampling_valid_quad_mask_np=mask,
            scale=np.array([0.25, 0.5]),
        )
        dt_targets._spiral_sampling = None
        dt_targets.prepare_patch_dt_target_samples([patch], 256, 128)
        expected = (
            patch._dt_target_ijs.copy(),
            patch._dt_target_block_rc.copy(),
            patch._dt_target_block_shape,
            patch._dt_target_anchor_max_dist_sq,
        )
        dt_targets._spiral_sampling = spiral_sampling
        dt_targets.prepare_patch_dt_target_samples([patch], 256, 128)
        np.testing.assert_array_equal(patch._dt_target_ijs, expected[0])
        np.testing.assert_array_equal(patch._dt_target_block_rc, expected[1])
        self.assertEqual(patch._dt_target_block_shape, expected[2])
        self.assertEqual(patch._dt_target_anchor_max_dist_sq, expected[3])

    def test_block_unwrap_matches_python(self):
        rows, columns = 17, 19
        all_rc = np.stack(np.unravel_index(
            np.arange(rows * columns), (rows, columns)), axis=1).astype(np.int32)
        keep = ~((all_rc[:, 0] == 8) & (all_rc[:, 1] > 3))
        block_rc = all_rc[keep]
        rng = np.random.default_rng(12)
        theta = rng.uniform(-np.pi, np.pi, len(block_rc)).astype(np.float32)
        dt_targets._spiral_sampling = None
        expected = dt_targets._unwrap_block_samples(
            theta, block_rc, (rows, columns))
        dt_targets._spiral_sampling = spiral_sampling
        actual = dt_targets._unwrap_block_samples(
            theta, block_rc, (rows, columns))
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])


def _cfg(*, stratified=True, weights=None):
    return {
        'pcl_stratified_pcl_sampling': stratified,
        'pcl_sampling_weights': weights,
    }


def test_legacy_stratification_draws_equally_from_each_group():
    cfg = _cfg()
    strata = losses.build_pcl_sampling_strata(['a', 'a', 'b', 'b', 'c', 'c'], cfg)
    np.random.seed(1)
    chosen = losses._choose_pcl_indices(strata, 3, cfg)
    assert sorted(index // 2 for index in chosen) == [0, 1, 2]


def test_explicit_weights_take_precedence_and_can_disable_groups():
    cfg = _cfg(stratified=False, weights={'a': 0, 'b': 1})
    strata = losses.build_pcl_sampling_strata(['a', 'a', 'b', 'b'], cfg)
    assert strata['groups'] == ['b']
    assert np.array_equal(strata['all'], np.array([2, 3]))


def test_component_member_count_allows_repeated_draws():
    cfg = _cfg(stratified=False)
    strata = losses.build_pcl_sampling_strata(
        ['fibers'], cfg, member_weights=[4])

    assert strata['effective_size'] == 4
    chosen = losses._choose_pcl_indices(strata, 4, cfg)
    assert np.array_equal(chosen, np.zeros(4, dtype=np.int64))


class _Atlas:
    device = torch.device('cpu')

    def __init__(self, masks):
        if spiral_sampling is None:
            raise RuntimeError('vc_spiral.spiral_sampling is required by this test')
        self.sampling_atlas = spiral_sampling.PatchSamplingAtlas(masks)
        self.node_maps = []
        start = 0
        for mask in masks:
            node_map = np.full(mask.shape, -1, dtype=np.int64)
            node_map[mask] = start + np.arange(mask.sum())
            self.node_maps.append(node_map)
            start += int(mask.sum())

    def theta_node_ids(self, patch_indices, ijs):
        patch_indices = np.asarray(patch_indices)
        cells = np.floor(np.asarray(ijs)).astype(np.int64)
        out = np.empty(cells.shape[:-1], dtype=np.int64)
        for patch_idx in np.unique(patch_indices):
            selected = patch_indices == patch_idx
            ij = cells[selected]
            out[selected] = self.node_maps[int(patch_idx)][ij[:, 0], ij[:, 1]]
        return out

    def lookup(self, patch_indices, ijs):
        return torch.cat([
            torch.zeros((*ijs.shape[:-1], 1)), ijs.to(torch.float32)], dim=-1)


def _patch(mask):
    return SimpleNamespace(
        _sampling_valid_quad_mask_np=mask,
        _sampling_valid_quad_indices_np=np.argwhere(mask).astype(np.int64),
    )


def test_patch_batch_returns_node_ids_and_explicit_padding_mask():
    masks = [np.ones((2, 2), dtype=bool), np.ones((5, 5), dtype=bool)]
    patches = [_patch(mask) for mask in masks]
    atlas = _Atlas(masks)
    np.random.seed(3)
    batch = losses._sample_patch_batch(
        'uniform_2d', patches, np.array([1.0, 0.0]), 2, 7, {},
        patch_atlas=atlas, crossing_map=object())
    ijs, patch_indices, zyxs, node_ids, sample_mask = batch
    assert ijs.shape == (2, 7, 2)
    assert zyxs.shape == (2, 7, 3)
    assert node_ids.shape == (2, 7)
    assert sample_mask.shape == (2, 7)
    assert sample_mask.sum(dim=-1).tolist() == [4, 4]
    assert patch_indices.tolist() == [0, 0]
    expected = atlas.theta_node_ids(
        np.zeros((2, 7), dtype=np.int64), ijs.numpy())
    np.testing.assert_array_equal(node_ids.numpy(), expected)
