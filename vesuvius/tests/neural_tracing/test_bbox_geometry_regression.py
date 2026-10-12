"""Bbox optimization output must remain byte-stable for fixed input and seed."""
import hashlib
import json
import unittest

import numpy as np

from vesuvius.neural_tracing.inference.generate_segment_cover_bboxes import (
    _generate_segment_cover_records,
    _points_within_bbox,
)

EXPECTED_SHA = {
    (0.0, False): "012b5a8a57eafcb13a54cccb00431893703fed24ebe4db8ec97c118212667ff5",
    (0.0, True): "012b5a8a57eafcb13a54cccb00431893703fed24ebe4db8ec97c118212667ff5",
    (0.25, False): "b788c4d855eb866b58a3ff3c69e04798150d254347a29be68919791823bd0dcd",
    (0.25, True): "b788c4d855eb866b58a3ff3c69e04798150d254347a29be68919791823bd0dcd",
    (0.5, False): "4fa9a9d4f60c8bcaaf3f5455afb8f131d7e0be0872b9a52bedb1afcfe72f4648",
    (0.5, True): "510802c56a268e1fd44dee81d134d3ffe81883d6bc327e0b4dc48e412d1f335d",
}

class BboxGeometryRegression(unittest.TestCase):
    def test_bounding_boxes_remain_identical(self):
        rows, cols = np.indices((19, 23))
        pts = np.stack(
            [13000 + 3 * rows + 2 * cols, 2500 + 11 * rows, 1800 + 13 * cols],
            axis=-1,
        ).reshape(-1, 3).astype(float)

        for (overlap, prune), checksum_expected in EXPECTED_SHA.items():
            with self.subTest(overlap=overlap, prune=prune):
                result = _generate_segment_cover_records(
                    pts, (128, 128, 128), overlap=overlap,
                    prune_bboxes=prune, band_workers=1,
                )
                records = [
                    {
                        "grid_index": list(item["grid_index"]),
                        "bbox": list(item["bbox"]),
                        "shift_axis": item["shift_axis"],
                        "shift_voxels": int(item["shift_voxels"]),
                    }
                    for item in result["final_records"]
                ]
                checksum = hashlib.sha256(
                    json.dumps(records, sort_keys=True).encode()
                ).hexdigest()
                self.assertEqual(checksum, checksum_expected)
                covered = np.zeros(len(pts), dtype=bool)
                for item in result["final_records"]:
                    covered |= _points_within_bbox(pts, item["bbox"])
                self.assertTrue(covered.all())

if __name__ == "__main__":
    unittest.main()
