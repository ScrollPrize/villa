import json
import tempfile
import unittest
from pathlib import Path

import fit_data
from lasagna_volume import LasagnaVolume


def _write_manifest(root: Path, *, groups: dict, base_shape=(64, 48, 32)) -> Path:
	path = root / "vol.lasagna.json"
	path.write_text(
		json.dumps(
			{
				"version": 2,
				"source_to_base": 1.0,
				"base_shape_zyx": list(base_shape),
				"grad_mag_encode_scale": 1000.0,
				"grad_mag_factor": 1.0,
				"umbilicus_json": "umbilicus.json",
				"groups": groups,
			}
		)
		+ "\n",
		encoding="utf-8",
	)
	return path


class FitDataManifestLazyTests(unittest.TestCase):
	def test_preprocessed_params_uses_manifest_shape_without_opening_zarrs(self) -> None:
		with tempfile.TemporaryDirectory() as td:
			root = Path(td)
			manifest = _write_manifest(
				root,
				groups={
					"cos": {"zarr": "missing_cos.ome.zarr/0", "scaledown": 0, "channels": ["cos"]},
					"surface": {
						"zarr": "missing_surface.ome.zarr/2",
						"scaledown": 2,
						"channels": ["grad_mag", "nx", "ny"],
					},
				},
			)

			params = fit_data.get_preprocessed_params(str(manifest))

			self.assertEqual(params["scaledown"], 1.0)
			self.assertEqual(params["volume_extent_fullres"], (32, 48, 64))

	def test_opened_zarr_shape_is_checked_against_manifest(self) -> None:
		with tempfile.TemporaryDirectory() as td:
			root = Path(td)
			manifest = _write_manifest(
				root,
				base_shape=(16, 16, 16),
				groups={
					"grad_mag": {
						"zarr": "grad_mag.ome.zarr/1",
						"scaledown": 1,
						"channels": ["grad_mag"],
					},
				},
			)
			vol = LasagnaVolume.load(manifest)
			group = vol.groups["grad_mag"]

			with self.assertRaisesRegex(ValueError, "zarr shape mismatch"):
				fit_data._validate_group_zarr_shape(
					vol=vol,
					group_name="grad_mag",
					group=group,
					zarr_path=str(root / group.zarr_path),
					shape=(7, 8, 8),
				)


if __name__ == "__main__":
	unittest.main()
