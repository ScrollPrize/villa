"""A written mesh must not declare a node outside the scroll as surface.

``save_mesh`` transforms a spiral grid into scroll coordinates and keeps
every node whose z stays inside the fitted band. A node can leave the volume
sideways as well, and the transform can return a node that is not finite:
both were written raw into the place where the surface goes, and one
non-finite node also put the bare literal ``NaN`` into ``meta.json``, which
RFC 8259 does not allow and a strict reader refuses.

The fixture is a verbatim crop of a fitted winding of PHerc0826, a scroll
8169 voxels wide: 90 of its 110 nodes are valid and 36 of those have y
beyond that width, so they name a place the scroll does not have.
"""
import json
from pathlib import Path
from unittest import mock

import numpy as np
import torch

import spiral_helpers
from sample_spiral import get_spiral_yxs
from tifxyz import load_tifxyz, save_tifxyz

FIXTURE = Path(__file__).parent / 'data' / 'PHerc0826-w207-crop.json'
# A spiral wide enough that the windings the export writes carry more columns
# than the fixture has nodes, so every node of the crop is written somewhere.
DR = 200.0
STEP = 20


def fixture():
    return json.loads(FIXTURE.read_text())


def valid_nodes(document):
    """The crop's nodes that the writer counts as surface, as [N, 3] zyx."""
    grid = np.asarray(document['zyxs'], dtype=np.float32)
    return grid[np.any(grid != -1.0, axis=-1)]


def inside_the_scroll(nodes, base_shape_zyx):
    y, x = nodes[..., 1], nodes[..., 2]
    return ((y >= 0) & (y < base_shape_zyx[1])
            & (x >= 0) & (x < base_shape_zyx[2]))


class ReplayTransform:
    """Lands the fixture's nodes wherever the transformed grid would land."""

    def __init__(self, nodes):
        self.nodes = torch.as_tensor(nodes)
        self.consumed = 0

    def inv(self, points):
        index = (torch.arange(points.shape[0]) + self.consumed) % self.nodes.shape[0]
        self.consumed += points.shape[0]
        return self.nodes[index]


def export(nodes, out_path, document, **kwargs):
    """save_mesh over a grid whose every node is one of the fixture's."""
    cfg = {'shell_outer_winding_idx': 4, 'output_step_size': STEP,
           'model_flow_bounds_z_margin': 0}
    written = {}

    def capture(winding_zyxs, out_dir, uuid, **unused):
        written[uuid] = np.asarray(winding_zyxs)
        return True

    # The splice reads patch geometry, which is a different question from
    # what the transform put on the grid, so it is out of this test.
    with mock.patch.object(spiral_helpers, 'save_tifxyz', side_effect=capture), \
            mock.patch.object(spiral_helpers, '_build_spliced_overlay'), \
            mock.patch.object(
                spiral_helpers, 'get_spiral_yxs',
                side_effect=lambda *args, **named: get_spiral_yxs(
                    *args, **named, device='cpu')):
        spiral_helpers.save_mesh(
            ReplayTransform(nodes), torch.tensor(DR), [], [], str(out_path), cfg,
            document['z_begin'], document['z_end'], document['voxel_size_um'],
            lambda *unused: None,
            winding_range=(2, 4),
            patch_satisfaction_evaluation=None,
            patch_atlas=None,
            **kwargs)
    assert written, 'the export wrote no winding'
    return written


def written_nodes(grids):
    stacked = np.concatenate([grid.reshape(-1, 3) for grid in grids.values()])
    return stacked[np.any(stacked != -1.0, axis=-1)]


def test_a_node_that_is_not_finite_is_not_a_valid_vertex(tmp_path):
    document = fixture()
    grid = np.asarray(document['zyxs'], dtype=np.float32)
    assert np.isfinite(grid).all(), 'the fixture itself must start finite'
    grid[5, 5] = np.nan
    save_tifxyz(grid, str(tmp_path), uuid='w207',
                step_size=document['step_size'],
                voxel_size_um=document['voxel_size_um'], source='test')
    written = tmp_path / 'w207'
    raw = (written / 'meta.json').read_text()
    assert 'NaN' not in raw, (
        'meta.json carries the literal NaN, which RFC 8259 does not allow: ' + raw)
    bbox = np.asarray(json.loads(raw)['bbox'], dtype=np.float64)
    assert np.isfinite(bbox).all(), f'the bounding box is not finite: {bbox.tolist()}'
    assert torch.isfinite(load_tifxyz(str(written)).zyxs).all(), (
        'a reader of the written mesh sees a node that is not finite')


def test_nodes_outside_the_scroll_are_not_written_as_surface(tmp_path):
    document = fixture()
    base_shape_zyx = document['base_shape_zyx']
    nodes = valid_nodes(document)
    inside = inside_the_scroll(nodes, base_shape_zyx)
    assert int((~inside).sum()) == document['nodes_with_y_beyond_the_scroll']

    stated = written_nodes(
        export(nodes, tmp_path / 'stated', document, base_shape_zyx=base_shape_zyx))
    outside = ~inside_the_scroll(stated, base_shape_zyx)
    assert not outside.any(), (
        f'{int(outside.sum())} of {len(stated)} written nodes lie outside the '
        f'{base_shape_zyx[2]} by {base_shape_zyx[1]} volume of this scroll')
    assert ({tuple(node) for node in stated}
            == {tuple(node) for node in nodes[inside]}), (
        'the nodes inside the scroll must be written exactly as the transform '
        'returned them')

    # The writer cannot invent the volume. A dataset whose scroll spec does
    # not state base_shape_zyx keeps those nodes, which is also what makes
    # the assertion above a statement about the bound and not about a crop
    # that happens to have nothing outside it.
    unstated = written_nodes(export(nodes, tmp_path / 'unstated', document))
    assert not inside_the_scroll(unstated, base_shape_zyx).all()


def test_a_node_that_is_not_finite_is_never_written_as_surface(tmp_path):
    document = fixture()
    nodes = valid_nodes(document)
    nodes[7, 1] = np.nan          # what a non-finite transform output looks like
    written = written_nodes(export(nodes, tmp_path, document))
    not_finite = ~np.isfinite(written).all(axis=-1)
    assert not not_finite.any(), (
        f'{int(not_finite.sum())} of {len(written)} written nodes are not finite')
