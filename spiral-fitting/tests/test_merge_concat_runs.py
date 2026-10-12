import json
import os
import tempfile
import unittest

import numpy as np

import merge_concat_runs
import tifxyz


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


class MergeConcatRunsTests(unittest.TestCase):
    def test_bbox_excludes_sentinel_and_is_xyz_ordered(self):
        points = _points()
        with tempfile.TemporaryDirectory() as temporary:
            merge_concat_runs.save_tifxyz(
                points, temporary, "w000", 20.0, 7.91, "test")
            bbox = _meta(temporary)["bbox"]

        valid = points[np.any(points != -1, axis=-1)]
        expected = [valid.min(axis=0).tolist(), valid.max(axis=0).tolist()]
        for actual_row, expected_row in zip(bbox, expected):
            for actual, expected_value in zip(actual_row, expected_row):
                self.assertAlmostEqual(actual, expected_value, places=5)
        self.assertNotIn(-1.0, bbox[0] + bbox[1])

    def test_bbox_matches_reference_writer(self):
        points = _points()
        with tempfile.TemporaryDirectory() as temporary:
            merge_path = os.path.join(temporary, "merge")
            reference_path = os.path.join(temporary, "reference")
            merge_concat_runs.save_tifxyz(
                points, merge_path, "w000", 20.0, 7.91, "test")
            tifxyz.save_tifxyz(
                points[..., ::-1], reference_path, "w000", 20.0, 7.91, "test")
            merge_meta = _meta(merge_path)
            reference_meta = _meta(os.path.join(reference_path, "w000"))

        for actual_row, expected_row in zip(
            merge_meta["bbox"], reference_meta["bbox"]):
            for actual, expected in zip(actual_row, expected_row):
                self.assertAlmostEqual(actual, expected, places=5)
        self.assertEqual(merge_meta["area_vx2"], reference_meta["area_vx2"])

    def test_all_invalid_grid_keeps_sentinel_bbox(self):
        points = np.full((6, 5, 3), -1.0, dtype=np.float32)
        with tempfile.TemporaryDirectory() as temporary:
            merge_concat_runs.save_tifxyz(
                points, temporary, "w000", 20.0, 7.91, "test")
            metadata = _meta(temporary)

        self.assertEqual(
            metadata["bbox"],
            [[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]],
        )
        self.assertEqual(metadata["area_vx2"], 0)
