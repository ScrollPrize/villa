import json
import os
import tempfile
import unittest

import numpy as np

from surface_export import new_surface
from vc_reader import tifxyz


def _points():
    rows, cols = np.indices((6, 5), dtype=np.float32)
    points = np.stack(
        [
            3000.0 + cols * 25.0,
            2500.0 + rows * 20.0,
            8000.0 + rows * 20.0,
        ],
        axis=-1,
    )
    points[0, 0] = -1.0
    points[3, 2] = -1.0
    return points


def _meta(path):
    with open(os.path.join(path, "meta.json")) as stream:
        return json.load(stream)


def _write(points, path, *, layout_metadata=None):
    tifxyz.write(path, new_surface(points, 20.0, 7.91, "test", layout_metadata=layout_metadata),
                 compression="none", tile=None, mask=False, overwrite=True)


class MergeConcatRunsTests(unittest.TestCase):
    def test_bbox_excludes_sentinel_and_is_xyz_ordered(self):
        points = _points()
        with tempfile.TemporaryDirectory() as temporary:
            _write(points, temporary)
            bbox = _meta(temporary)["bbox"]

        valid = points[np.any(points != -1, axis=-1)]
        expected = [valid.min(axis=0).tolist(), valid.max(axis=0).tolist()]
        for actual_row, expected_row in zip(bbox, expected):
            for actual, expected_value in zip(actual_row, expected_row):
                self.assertAlmostEqual(actual, expected_value, places=5)
        self.assertNotIn(-1.0, bbox[0] + bbox[1])

    def test_all_invalid_grid_has_vc_reader_empty_bbox(self):
        points = np.full((6, 5, 3), -1.0, dtype=np.float32)
        with tempfile.TemporaryDirectory() as temporary:
            _write(points, temporary)
            metadata = _meta(temporary)

        self.assertEqual(
            metadata["bbox"],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        )
        self.assertEqual(metadata["area_vx2"], 0)


def test_writes_preserve_layout_and_assign_distinct_uuids(tmp_path):
    from uuid import UUID
    from surface_orientation import METADATA_KEY

    points = _points()
    layout = {'test_layout': 'preserved'}
    merged = tmp_path / 'merged'
    reference = tmp_path / 'reference'
    _write(points, merged, layout_metadata=layout)
    _write(points, reference, layout_metadata=layout)
    first = _meta(merged)
    second = _meta(reference)
    assert str(UUID(first['uuid'])) == first['uuid']
    assert str(UUID(second['uuid'])) == second['uuid']
    assert first['uuid'] != second['uuid']
    assert first[METADATA_KEY] == second[METADATA_KEY] == layout


def test_main_merges_runs_into_one_surface(tmp_path, monkeypatch):
    import sys

    import merge_concat_runs

    points = _points()
    for run, offset in (('a_baseline', 0.0), ('b', 2.0)):
        moved = np.where(points == -1, -1, points + offset)
        _write(moved.astype(np.float32), tmp_path / 'runs' / run / 'meshes' / 'm' / 'concat' / 'w000')
    out = tmp_path / 'out'
    monkeypatch.setattr(sys, 'argv', ['merge', str(tmp_path / 'runs'), '--out', str(out), '--method', 'mean'])
    merge_concat_runs.main()
    merged = tifxyz.read(out / 'w000')
    assert merged.valid.sum() == points.shape[0] * points.shape[1] - 2
    np.testing.assert_array_equal(merged.xyz[merged.valid], (points + 1)[merged.valid])
    assert _meta(out / 'w000')['format'] == 'tifxyz'
    assert sorted(os.listdir(out / 'w000')) == ['meta.json', 'x.tif', 'y.tif', 'z.tif']
