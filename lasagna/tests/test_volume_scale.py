from __future__ import annotations

import os
import sys
import unittest


ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
	sys.path.insert(0, ROOT)

import volume_scale


class VolumeScaleTest(unittest.TestCase):
	def test_missing_source_shape_is_identity(self) -> None:
		scale = volume_scale.coordinate_scale_to_base(
			base_shape_zyx=(100, 200, 300),
			source_shape_zyx=None,
		)

		self.assertEqual(scale.factor, 1.0)
		self.assertIsNone(scale.source_shape_zyx)

	def test_exact_identity_double_and_half_scales(self) -> None:
		self.assertEqual(
			volume_scale.coordinate_scale_to_base(
				base_shape_zyx=(100, 200, 300),
				source_shape_zyx=(100, 200, 300),
			).factor,
			1.0,
		)
		self.assertEqual(
			volume_scale.coordinate_scale_to_base(
				base_shape_zyx=(100, 200, 300),
				source_shape_zyx=(50, 100, 150),
			).factor,
			2.0,
		)
		self.assertEqual(
			volume_scale.coordinate_scale_to_base(
				base_shape_zyx=(100, 200, 300),
				source_shape_zyx=(200, 400, 600),
			).factor,
			0.5,
		)

	def test_axis_mismatch_errors(self) -> None:
		with self.assertRaisesRegex(ValueError, "anisotropic"):
			volume_scale.coordinate_scale_to_base(
				base_shape_zyx=(100, 200, 300),
				source_shape_zyx=(50, 100, 300),
			)

	def test_non_power_of_two_errors(self) -> None:
		with self.assertRaisesRegex(ValueError, "power-of-two"):
			volume_scale.coordinate_scale_to_base(
				base_shape_zyx=(100, 200, 300),
				source_shape_zyx=(80, 160, 240),
			)

	def test_corr_points_scale_xyz_only(self) -> None:
		corr = {
			"collections": {
				"7": {
					"points": {
						"3": {"p": [1, 2, 3], "wind_a": 9},
					}
				}
			}
		}

		got = volume_scale.scale_corr_points_json(corr, 2.0)

		self.assertEqual(got["collections"]["7"]["points"]["3"]["p"], [2.0, 4.0, 6.0])
		self.assertEqual(got["collections"]["7"]["points"]["3"]["wind_a"], 9)
		self.assertEqual(corr["collections"]["7"]["points"]["3"]["p"], [1, 2, 3])

	def test_tifxyz_meta_scales_scale_bbox_and_point_fields(self) -> None:
		meta = {
			"scale": [0.5, 0.5],
			"bbox": [[1, 2, 3], [4, 5, 6]],
			"points": [{"p": [7, 8, 9], "wind_a": 2}],
		}

		got = volume_scale.scale_tifxyz_meta(
			meta,
			2.0,
			base_shape_zyx=(100, 100, 100),
			lasagna_base_shape_zyx=(100, 100, 100),
		)

		self.assertEqual(got["scale"], [0.25, 0.25])
		self.assertEqual(got["bbox"], [[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]])
		self.assertEqual(got["points"][0]["p"], [14.0, 16.0, 18.0])
		self.assertEqual(got["points"][0]["wind_a"], 2)
		self.assertEqual(got["base_shape_zyx"], [100, 100, 100])
		self.assertEqual(got["lasagna_base_shape_zyx"], [100, 100, 100])


if __name__ == "__main__":
	unittest.main()
