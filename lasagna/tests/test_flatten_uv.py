import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import tifffile
import torch

import flatten_uv
import fit
import model


class FlattenUvSidecarTests(unittest.TestCase):
	def _uv(self, h=5, w=7):
		r, c = np.meshgrid(
			np.arange(h, dtype=np.float32),
			np.arange(w, dtype=np.float32), indexing="ij")
		return np.stack((r + 0.03 * c, c + 0.01 * r), axis=-1)

	def test_round_trip_preserves_every_source_vertex_exactly(self):
		uv = self._uv()
		fingerprint = flatten_uv.canonical_grid_fingerprint(
			uv.shape[:2], source_step=20.0,
			winding_column_ranges=[[0, 3], [3, 7]])
		with tempfile.TemporaryDirectory() as temporary:
			metadata = flatten_uv.write_sidecars(
				temporary, uv, fingerprint=fingerprint,
				source_step=20.0, output_step=20.0,
				valid=np.ones(uv.shape[:2], dtype=np.bool_),
				winding_column_ranges=[[0, 3], [3, 7]],
				winding_ids=[10, 11])
			loaded, loaded_valid, loaded_cells, info = flatten_uv.load_sidecars(
				metadata,
				expected_source_step=20.0,
				expected_output_step=20.0,
				expected_winding_ids=[10, 11])
			np.testing.assert_array_equal(loaded, uv)
			self.assertTrue(loaded_valid.all())
			self.assertTrue(loaded_cells.all())
			self.assertTrue(info["covers_complete_source_grid"])
			with self.assertRaisesRegex(flatten_uv.FlattenUvError, "output_step"):
				flatten_uv.load_sidecars(
					metadata,
					expected_output_step=10.0,
					expected_winding_ids=[10, 11])

	def test_current_source_is_projected_onto_stored_spiral_grid(self):
		h, w = 3, 9
		columns = torch.arange(w, dtype=torch.float32).reshape(1, w).expand(h, w)
		rows = torch.arange(h, dtype=torch.float32).reshape(h, 1).expand(h, w)
		xyz = torch.stack((columns, rows, torch.zeros_like(columns)), dim=-1)
		valid = torch.ones(h, w, dtype=torch.bool)
		projected, projected_valid, source_columns = (
			fit._project_flatten_source_to_warm_grid(
				xyz, valid,
				current_ranges=[[0, 4], [4, 9]],
				target_ranges=[[0, 3], [3, 7]],
				winding_ids=[10, 11],
				source_step=20.0,
				current_dr_per_winding=1.1,
				target_dr_per_winding=1.0,
				target_rows=h))
		self.assertEqual(tuple(projected.shape), (h, 7, 3))
		self.assertTrue(bool(projected_valid.all()))
		expected_columns = torch.tensor(
			[0.0, 1.1, 2.2, 4.0, 5.1, 6.2, 7.3])
		torch.testing.assert_close(source_columns, expected_columns)
		torch.testing.assert_close(projected[..., 0], expected_columns.expand(h, 7))

	def test_shape_fingerprint_corruption_nonfinite_and_folds_are_rejected(self):
		uv = self._uv()
		fingerprint = flatten_uv.canonical_grid_fingerprint(
			uv.shape[:2], source_step=20.0)
		with tempfile.TemporaryDirectory() as temporary:
			root = Path(temporary)
			metadata = flatten_uv.write_sidecars(
				root, uv, fingerprint=fingerprint,
				source_step=20.0, output_step=20.0,
				valid=np.ones(uv.shape[:2], dtype=np.bool_))
			original_metadata = json.loads(metadata.read_text(encoding="utf-8"))
			bad_metadata = dict(original_metadata)
			bad_metadata["source_shape"] = [1, 7]
			metadata.write_text(json.dumps(bad_metadata), encoding="utf-8")
			with self.assertRaisesRegex(flatten_uv.FlattenUvError, "shape"):
				flatten_uv.load_sidecars(metadata)
			bad_metadata = dict(original_metadata)
			bad_metadata["canonical_grid_fingerprint"] = "wrong"
			metadata.write_text(json.dumps(bad_metadata), encoding="utf-8")
			with self.assertRaisesRegex(flatten_uv.FlattenUvError, "fingerprint"):
				flatten_uv.load_sidecars(metadata)
			metadata.write_text(json.dumps(original_metadata), encoding="utf-8")

			row = tifffile.imread(root / flatten_uv.ROW_FILENAME)
			row[1, 1] = np.nan
			tifffile.imwrite(root / flatten_uv.ROW_FILENAME, row)
			with self.assertRaisesRegex(flatten_uv.FlattenUvError, "non-finite"):
				flatten_uv.load_sidecars(metadata)

			row = uv[..., 0].copy()
			row[1:, :] *= -1.0
			tifffile.imwrite(root / flatten_uv.ROW_FILENAME, row)
			with self.assertRaisesRegex(flatten_uv.FlattenUvError, "folded"):
				flatten_uv.load_sidecars(metadata)

	def test_corrupt_metadata_is_rejected(self):
		with tempfile.TemporaryDirectory() as temporary:
			path = Path(temporary) / flatten_uv.METADATA_FILENAME
			path.write_text("{not-json", encoding="utf-8")
			with self.assertRaisesRegex(flatten_uv.FlattenUvError, "metadata"):
				flatten_uv.load_sidecars(path)

	def test_schema_v3_sidecars_remain_readable(self):
		uv = self._uv()
		fingerprint = flatten_uv.canonical_grid_fingerprint(
			uv.shape[:2], source_step=20.0)
		with tempfile.TemporaryDirectory() as temporary:
			metadata_path = flatten_uv.write_sidecars(
				temporary, uv, fingerprint=fingerprint,
				source_step=20.0, output_step=20.0,
				valid=np.ones(uv.shape[:2], dtype=np.bool_))
			metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
			metadata["schema_version"] = flatten_uv.LEGACY_SCHEMA_VERSION
			metadata.pop("cell_valid_file")
			metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

			loaded, loaded_valid, loaded_cells, _ = (
				flatten_uv.load_sidecars(metadata_path))

		np.testing.assert_array_equal(loaded, uv)
		self.assertTrue(loaded_valid.all())
		self.assertTrue(loaded_cells.all())

	def test_extrapolated_invalid_cells_may_cross_but_supported_cells_may_not(self):
		uv = self._uv(3, 4)
		valid = np.ones((3, 4), dtype=np.bool_)
		valid[:, -1] = False
		uv[:, -1] = uv[::-1, -2]
		flatten_uv.validate_uv(uv, (3, 4), valid=valid)
		valid[:, -1] = True
		with self.assertRaisesRegex(flatten_uv.FlattenUvError, "folded"):
			flatten_uv.validate_uv(uv, (3, 4), valid=valid)

	def test_second_fixed_diagonal_triangle_fold_is_rejected(self):
		# Triangle (m00,m10,m01) is positive, while exporter triangle
		# (m10,m11,m01) has the opposite winding.
		uv = np.array([
			[[0.0, 0.0], [0.0, 1.0]],
			[[1.0, 0.0], [-1.0, 2.0]],
		], dtype=np.float32)
		with self.assertRaisesRegex(flatten_uv.FlattenUvError, "folded"):
			flatten_uv.validate_uv(uv, (2, 2))

	def test_sidecars_exclude_only_folded_cells(self):
		uv = np.array([
			[[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]],
			[[1.0, 0.0], [1.0, 1.0], [-1.0, 2.0]],
		], dtype=np.float32)
		fingerprint = flatten_uv.canonical_grid_fingerprint(
			uv.shape[:2], source_step=20.0)
		with tempfile.TemporaryDirectory() as temporary:
			metadata_path = flatten_uv.write_sidecars(
				temporary, uv, fingerprint=fingerprint,
				source_step=20.0, output_step=20.0,
				valid=np.ones(uv.shape[:2], dtype=np.bool_))
			loaded, loaded_valid, loaded_cells, metadata = (
				flatten_uv.load_sidecars(metadata_path))

		np.testing.assert_array_equal(loaded, uv)
		self.assertTrue(loaded_valid.all())
		np.testing.assert_array_equal(
			loaded_cells, np.array([[True, False]], dtype=np.bool_))
		self.assertEqual(
			metadata["topology_validation"]["source_supported_cell_count"], 2)
		self.assertEqual(
			metadata["topology_validation"]["excluded_folded_cell_count"], 1)
		self.assertEqual(
			metadata["topology_validation"]["cell_count"], 1)

	def test_explicit_cell_mask_allows_a_folded_unsupported_cell(self):
		uv = np.array([
			[[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]],
			[[1.0, 0.0], [1.0, 1.0], [-1.0, 2.0]],
		], dtype=np.float32)
		validated = flatten_uv.validate_uv(
			uv, uv.shape[:2],
			cell_valid=np.array([[True, False]], dtype=np.bool_))
		np.testing.assert_array_equal(validated, uv)

	def test_model_reconstructs_exact_source_uv_pyramid_and_anchor(self):
		h, w = 5, 7
		r, c = torch.meshgrid(
			torch.arange(h, dtype=torch.float32),
			torch.arange(w, dtype=torch.float32), indexing="ij")
		xyz = torch.stack((c * 20.0, r * 20.0, torch.zeros_like(r)), dim=-1)
		valid = torch.ones(h, w, dtype=torch.bool)
		initial = torch.stack((r + 4.25, c + 8.5), dim=-1)
		mdl = model.Model3D.from_flatten_tifxyz_crop(
			xyz, valid, device=torch.device("cpu"), mesh_step=20.0,
			flatten_direction="forward", flatten_output_step=20.0,
			flatten_initial_uv=initial)
		self.assertEqual(
			len(mdl.flatten_map_ms),
			mdl._scale_count_to_longer_dim_2(h, w))
		torch.testing.assert_close(
			mdl.flatten_base_uv,
			initial.permute(2, 0, 1).unsqueeze(1), rtol=0, atol=0)
		self.assertEqual(int(torch.count_nonzero(mdl.flatten_map_ms[0])), 0)
		torch.testing.assert_close(mdl.flatten_map(), initial, rtol=0, atol=1e-6)
		expected_offset = model.Model3D._flatten_avg_offset(initial, valid)
		torch.testing.assert_close(
			mdl.flatten_initial_avg_offset, expected_offset, rtol=0, atol=1e-6)

	def test_model_uses_fixed_warm_base_and_full_zero_correction_pyramid(self):
		h, w = 17, 33
		r, c = torch.meshgrid(
			torch.arange(h, dtype=torch.float32),
			torch.arange(w, dtype=torch.float32), indexing="ij")
		xyz = torch.stack((c * 20.0, r * 20.0, torch.zeros_like(r)), dim=-1)
		valid = torch.ones(h, w, dtype=torch.bool)
		initial = torch.stack((r + 4.25, c + 8.5), dim=-1)
		mdl = model.Model3D.from_flatten_tifxyz_crop(
			xyz, valid, device=torch.device("cpu"), mesh_step=20.0,
			flatten_direction="forward", flatten_output_step=20.0,
			flatten_initial_uv=initial)
		self.assertEqual(
			len(mdl.flatten_map_ms),
			mdl._scale_count_to_longer_dim_2(h, w))
		self.assertTrue(all(
			int(torch.count_nonzero(level)) == 0
			for level in mdl.flatten_map_ms))
		torch.testing.assert_close(mdl.flatten_map(), initial, rtol=0, atol=1e-6)


if __name__ == "__main__":
	unittest.main()
