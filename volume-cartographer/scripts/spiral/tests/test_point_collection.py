import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout

from point_collection import load_point_collection


class LoadPointCollectionTests(unittest.TestCase):
    def test_loads_valid_collection(self):
        payload = {
            "vc_pointcollections_json_version": "1",
            "collections": {
                "0": {
                    "name": "col-a",
                    "points": {
                        "0": {"p": [1.0, 2.0, 3.0], "wind_a": 1.5},
                        "1": {"p": [4.0, 5.0, 6.0]},
                    },
                },
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = os.path.join(temporary, "abs_winding.json")
            with open(path, "w", encoding="utf-8") as stream:
                json.dump(payload, stream)
            loaded = load_point_collection(path)

        self.assertIsNotNone(loaded)
        self.assertEqual(list(loaded.keys()), [0])
        collection = loaded[0]
        self.assertEqual(collection["name"], "col-a")
        self.assertEqual(collection["points"][0]["p"], [1.0, 2.0, 3.0])
        self.assertEqual(collection["points"][0]["winding_annotation"], 1.5)
        # Absent winding annotations become NaN, not None.
        self.assertNotEqual(
            collection["points"][1]["winding_annotation"],
            collection["points"][1]["winding_annotation"],
        )

    def test_missing_file_warns_and_skips(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = os.path.join(temporary, "drawn_control_points.json")
            output = io.StringIO()
            with redirect_stdout(output):
                loaded = load_point_collection(path)

        self.assertIsNone(loaded)
        message = output.getvalue()
        self.assertIn("not found", message)
        self.assertIn(path, message)
        self.assertNotIn("Error", message)

    def test_malformed_file_still_reports_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = os.path.join(temporary, "abs_winding.json")
            with open(path, "w", encoding="utf-8") as stream:
                stream.write("{not valid json")
            output = io.StringIO()
            with redirect_stdout(output):
                loaded = load_point_collection(path)

        self.assertIsNone(loaded)
        self.assertIn("Error loading point collection", output.getvalue())

    def test_unsupported_version_still_reports_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = os.path.join(temporary, "abs_winding.json")
            with open(path, "w", encoding="utf-8") as stream:
                json.dump({"vc_pointcollections_json_version": "2"}, stream)
            output = io.StringIO()
            with redirect_stdout(output):
                loaded = load_point_collection(path)

        self.assertIsNone(loaded)
        self.assertIn("Unsupported JSON version", output.getvalue())


if __name__ == "__main__":
    unittest.main()
