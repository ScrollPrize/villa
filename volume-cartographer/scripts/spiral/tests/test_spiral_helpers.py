import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from spiral_helpers import load_fiber_point_collection


class FiberPointCollectionTests(unittest.TestCase):
    def _write_fiber(self, directory, data):
        path = Path(directory) / "fiber.json"
        path.write_text(json.dumps(data))
        return path

    def test_loads_control_points_instead_of_line_points(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self._write_fiber(temporary, {
                "control_points": [[4, 8, 12], [20, 24, 28]],
                "line_points": [[400, 800, 1200]],
            })

            collection = load_fiber_point_collection(
                path, collection_id=7, min_point_spacing=0)

            points = [point["p"] for point in collection["points"].values()]
            np.testing.assert_array_equal(points, [[1, 2, 3], [5, 6, 7]])

    def test_does_not_fall_back_to_line_points(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self._write_fiber(temporary, {
                "line_points": [[4, 8, 12]],
            })

            collection = load_fiber_point_collection(path, collection_id=7)

            self.assertIsNone(collection)


if __name__ == "__main__":
    unittest.main()
