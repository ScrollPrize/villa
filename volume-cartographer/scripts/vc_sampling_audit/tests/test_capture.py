import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from vc_sampling_audit.capture import read_capture, report_capture


class RendererCaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "capture.json"
        y, x = np.meshgrid([-.25, .25], [-.25, .25], indexing="ij")
        q = np.stack([x, y, np.zeros_like(x)], axis=-1)
        # Unit directions: projected (1-d)(1-2d) is positive at
        # endpoints 0,2 but negative between 0.5 and 1.
        n = np.stack([-x, -2*y, np.full_like(x, np.sqrt(11)/4)], axis=-1)
        self.data = {"schema": "vc-sampling-grid-v1",
            "stage": "final-base-and-dirs-before-readMultiSlice",
            "pixel_order": "row-major-baseXYZ-directionXYZ", "units": "level-index-voxel",
            "between_pixel_interpolation": "NOT_SPECIFIED_BY_RENDERER",
            "shape_hw": [2,2], "level": 2, "crop_xy": [17,29], "offsets": [0,2],
            "pixels": np.concatenate([q,n], axis=-1).astype(np.float32).reshape(-1,6).tolist(),
            "invalid_pixels": 0}

    def save(self):
        self.path.write_text(json.dumps(self.data))

    def rejected(self):
        self.save()
        with self.assertRaises((ValueError, TypeError, OverflowError)):
            read_capture(self.path)

    def test_roundtrip_original_float32(self):
        self.save()
        _, q, n, _, _ = read_capture(self.path)
        expected = np.array(self.data["pixels"], dtype=np.float32).reshape(2,2,6)
        np.testing.assert_array_equal(np.concatenate([q,n], axis=-1), expected.astype(np.float64))

    def test_interior_failure_and_canvas_witness(self):
        self.save()
        report = report_capture(self.path, [0,0,1])
        for item in report["diagonals"].values():
            self.assertEqual(item["interior_projected_failures_missed_by_endpoints"], 2)
            self.assertEqual(item["failure_witnesses"][0]["canvas_cell_xy"], [17,29])

    def test_signed_zero_bits_preserved(self):
        self.data["pixels"][0][0] = -0.0
        self.data["offsets"] = [-0.0]
        self.save()
        _, q, _, offsets, _ = read_capture(self.path)
        self.assertTrue(np.signbit(q[0,0,0]))
        self.assertTrue(np.signbit(offsets[0]))

    def test_exact_offset_extrema_not_symmetric_recentering(self):
        self.data["offsets"] = [0.75,-0.5,0.5,-0.25]
        self.save()
        report = report_capture(self.path, [0,0,1])
        self.assertEqual(report["diagonals"]["AC"]["depth_interval_input_units"], [-.5,.75])
        self.assertEqual(report["capture"]["offset_count"], 4)

    def test_one_layer_is_zero_width_interval(self):
        self.data["offsets"] = [0.25]
        self.save()
        report = report_capture(self.path, [0,0,1])
        self.assertEqual(report["diagonals"]["AC"]["depth_interval_input_units"], [.25,.25])

    def test_normal_reversal_does_not_reverse_base_chart_reference(self):
        self.save()
        original = report_capture(self.path, [0,0,1])
        for pixel in self.data["pixels"]:
            pixel[3:] = [-value for value in pixel[3:]]
        self.data["offsets"] = [-value for value in self.data["offsets"]]
        self.save()
        reversed_depth = report_capture(self.path, [0,0,1])
        for diagonal in ("AC", "BD"):
            for field in ("projected_nonpositive_triangles", "volume_nonpositive_triangles"):
                self.assertEqual(original["diagonals"][diagonal][field],
                                 reversed_depth["diagonals"][diagonal][field])
        with self.assertRaisesRegex(ValueError, "nonpositive oriented triangle"):
            report_capture(self.path, [0,0,-1])

    def test_no_anatomy_or_renderer_execution_attestation(self):
        self.save()
        report = report_capture(self.path, [0,0,1])
        self.assertEqual(report["anatomical_sheet_identity"], "NOT_ASSESSED")
        self.assertIn("NOT_ATTESTED", report["capture"]["full_renderer_execution"])
        self.assertIn("diagnostic surrogate", report["renderer_interpolation_match"])

    def test_unsupported_metadata(self):
        for key in ("schema", "stage", "pixel_order", "units", "between_pixel_interpolation"):
            with self.subTest(key=key):
                original = self.data[key]
                self.data[key] = "other"
                self.rejected()
                self.data[key] = original

    def test_missing_rays(self):
        self.data["pixels"].pop(); self.rejected()

    def test_invalid_pixels_not_masked(self):
        self.data["invalid_pixels"] = 1; self.rejected()

    def test_null_ray_despite_zero_counter(self):
        self.data["pixels"][0][0] = None; self.rejected()

    def test_nonfinite_offsets(self):
        self.data["offsets"] = [float("inf")]; self.rejected()

    def test_empty_offsets(self):
        self.data["offsets"] = []; self.rejected()

    def test_overlarge_offsets(self):
        self.data["offsets"] = [0]*65537; self.rejected()

    def test_invalid_dimensions(self):
        for shape in ([1,4], [True,4], [513,513], [2,2,3]):
            self.data["shape_hw"] = shape
            self.rejected()

    def test_invalid_crop_and_level(self):
        self.data["crop_xy"] = [-1,0]; self.rejected()
        self.data["crop_xy"] = [0,0]
        self.data["level"] = True; self.rejected()

    def test_float32_overflow(self):
        self.data["pixels"][0][0] = 1e99; self.rejected()

    def test_duplicate_keys_refused(self):
        self.save()
        text = self.path.read_text().replace('"level": 2', '"level": 2, "level": 0')
        self.path.write_text(text)
        with self.assertRaisesRegex(ValueError, "Duplicate capture key"):
            read_capture(self.path)


if __name__ == "__main__":
    unittest.main()
