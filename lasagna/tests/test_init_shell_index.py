from __future__ import annotations

import math
import os
import sys
import tempfile
import unittest
import argparse
from pathlib import Path

import numpy as np
import torch
import tifffile


ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
	sys.path.insert(0, ROOT)

import fit
import cli_data
import init_shell_index
import lasagna_volume
import model as fit_model


def _write_tifxyz(path: Path, xyz: torch.Tensor) -> None:
	path.mkdir(parents=True, exist_ok=True)
	xyz_np = xyz.detach().cpu().numpy().astype(np.float32)
	tifffile.imwrite(str(path / "x.tif"), xyz_np[..., 0])
	tifffile.imwrite(str(path / "y.tif"), xyz_np[..., 1])
	tifffile.imwrite(str(path / "z.tif"), xyz_np[..., 2])
	(path / "meta.json").write_text("{}\n", encoding="utf-8")


def _wrapped_cylinder(
	*,
	radius: float,
	z_values: list[float],
	width: int = 32,
	center: tuple[float, float] = (0.0, 0.0),
) -> torch.Tensor:
	angles = torch.arange(width, dtype=torch.float32) * (2.0 * math.pi / float(width))
	x = float(center[0]) + float(radius) * torch.cos(angles)
	y = float(center[1]) + float(radius) * torch.sin(angles)
	rows = []
	for z in z_values:
		rows.append(torch.stack([x, y, torch.full_like(x, float(z))], dim=-1))
	unique = torch.stack(rows, dim=0)
	return torch.cat([unique, unique[:, :1]], dim=1).contiguous()


def _wrapped_flat_shell() -> torch.Tensor:
	h, w = 2, 4
	x = torch.arange(w, dtype=torch.float32).view(1, w).expand(h, w)
	y = torch.arange(h, dtype=torch.float32).view(h, 1).expand(h, w)
	z = torch.zeros(h, w, dtype=torch.float32)
	unique = torch.stack([x, y, z], dim=-1)
	return torch.cat([unique, unique[:, :1]], dim=1).contiguous()


def _closest_for_surface(
	surface: init_shell_index.InitShellSurface,
	*,
	h: float,
	w: float,
	xyz: tuple[float, float, float],
) -> init_shell_index.ShellClosestPoint:
	return init_shell_index.ShellClosestPoint(
		shell_id=surface.shell_id,
		shell_index=0,
		shell_path=surface.path,
		quad_row=max(0, min(int(h), int(surface.xyz_wrapped.shape[0]) - 2)),
		quad_col=int(w) % int(surface.unique_w),
		triangle_id=0,
		barycentric=(1.0, 0.0, 0.0),
		closest_xyz=xyz,
		distance=0.0,
		h=float(h),
		w=float(w),
	)


def _grid_center(xyz: torch.Tensor) -> torch.Tensor:
	h_mid = float(int(xyz.shape[0]) - 1) * 0.5
	w_mid = float(int(xyz.shape[1]) - 1) * 0.5
	h0 = int(math.floor(h_mid))
	w0 = int(math.floor(w_mid))
	h1 = min(h0 + 1, int(xyz.shape[0]) - 1)
	w1 = min(w0 + 1, int(xyz.shape[1]) - 1)
	fh = h_mid - float(h0)
	fw = w_mid - float(w0)
	return (
		(1.0 - fh) * (1.0 - fw) * xyz[h0, w0]
		+ fh * (1.0 - fw) * xyz[h1, w0]
		+ (1.0 - fh) * fw * xyz[h0, w1]
		+ fh * fw * xyz[h1, w1]
	)


class InitShellIndexErrorTest(unittest.TestCase):
	def test_missing_manifest_init_shell_dir_key_errors(self) -> None:
		with self.assertRaisesRegex(ValueError, "init_shell_dir"):
			fit._require_manifest_init_shell_dir({})
		with self.assertRaisesRegex(ValueError, "init_shell_dir"):
			fit._require_manifest_init_shell_dir({"init_shell_dir": ""})

	def test_manifest_init_shell_dir_resolves_relative_to_lasagna_json(self) -> None:
		with tempfile.TemporaryDirectory() as td:
			root = Path(td)
			manifest = root / "volume.lasagna.json"
			manifest.write_text(
				'{"version": 2, "umbilicus_json": "umb.json", "init_shell_dir": "init_shells/", "groups": {}}\n',
				encoding="utf-8",
			)

			vol = lasagna_volume.LasagnaVolume.load(manifest)

		self.assertEqual(vol.init_shell_dir, "init_shells/")
		self.assertEqual(vol.init_shell_dir_abs_path(), root.resolve() / "init_shells")

	def test_missing_and_empty_init_shell_dir_errors(self) -> None:
		with tempfile.TemporaryDirectory() as td:
			root = Path(td)
			with self.assertRaisesRegex(ValueError, "init_shell_dir"):
				init_shell_index.InitShellIndex.from_directory(root / "missing")
			with self.assertRaisesRegex(ValueError, "shell_\\*.tifxyz"):
				init_shell_index.InitShellIndex.from_directory(root)

	def test_invalid_shell_vertex_errors(self) -> None:
		with tempfile.TemporaryDirectory() as td:
			root = Path(td)
			xyz = _wrapped_cylinder(radius=10.0, z_values=[0.0, 10.0])
			xyz[0, 0] = -1.0
			_write_tifxyz(root / "shell_0001.tifxyz", xyz)

			with self.assertRaisesRegex(ValueError, "invalid"):
				init_shell_index.InitShellIndex.from_directory(root)

	def test_malformed_wrapped_shell_errors(self) -> None:
		with tempfile.TemporaryDirectory() as td:
			root = Path(td)
			xyz = _wrapped_cylinder(radius=10.0, z_values=[0.0, 10.0])
			xyz[:, -1, 0] += 1.0
			_write_tifxyz(root / "shell_0001.tifxyz", xyz)

			with self.assertRaisesRegex(ValueError, "explicitly wrapped"):
				init_shell_index.InitShellIndex.from_directory(root)


class InitShellIndexLookupTest(unittest.TestCase):
	def test_two_synthetic_cylinders_select_closest_shell(self) -> None:
		with tempfile.TemporaryDirectory() as td:
			root = Path(td)
			_write_tifxyz(root / "shell_0001.tifxyz", _wrapped_cylinder(radius=10.0, z_values=[0.0, 10.0]))
			_write_tifxyz(root / "shell_0002.tifxyz", _wrapped_cylinder(radius=20.0, z_values=[0.0, 10.0]))
			idx = init_shell_index.InitShellIndex.from_directory(root)

			got = idx.closest_point((21.0, 0.0, 5.0), device="cpu")

		self.assertEqual(got.shell_id, "shell_0002.tifxyz")
		self.assertLess(got.distance, 1.1)

	def test_seed_near_wrap_seam_uses_seam_quad(self) -> None:
		width = 16
		angle = -0.05
		seed = (10.0 * math.cos(angle), 10.0 * math.sin(angle), 5.0)
		with tempfile.TemporaryDirectory() as td:
			root = Path(td)
			_write_tifxyz(root / "shell_0001.tifxyz", _wrapped_cylinder(radius=10.0, z_values=[0.0, 10.0], width=width))
			idx = init_shell_index.InitShellIndex.from_directory(root)

			got = idx.closest_point(seed, device="cpu")

		self.assertEqual(got.quad_col, width - 1)
		self.assertGreater(got.w, width - 0.5)
		self.assertLess(got.distance, 0.1)

	def test_continuous_coordinates_for_first_triangle_split(self) -> None:
		with tempfile.TemporaryDirectory() as td:
			root = Path(td)
			_write_tifxyz(root / "shell_0001.tifxyz", _wrapped_flat_shell())
			idx = init_shell_index.InitShellIndex.from_directory(root)

			got = idx.closest_point((0.5, 0.8, 0.0), device="cpu")

		self.assertEqual((got.quad_row, got.quad_col, got.triangle_id), (0, 0, 0))
		self.assertAlmostEqual(got.h, 0.8, delta=1.0e-5)
		self.assertAlmostEqual(got.w, 0.5, delta=1.0e-5)
		self.assertLess(got.distance, 1.0e-5)

	def test_continuous_coordinates_for_second_triangle_split(self) -> None:
		with tempfile.TemporaryDirectory() as td:
			root = Path(td)
			_write_tifxyz(root / "shell_0001.tifxyz", _wrapped_flat_shell())
			idx = init_shell_index.InitShellIndex.from_directory(root)

			got = idx.closest_point((0.8, 0.3, 0.0), device="cpu")

		self.assertEqual((got.quad_row, got.quad_col, got.triangle_id), (0, 0, 1))
		self.assertAlmostEqual(got.h, 0.3, delta=1.0e-5)
		self.assertAlmostEqual(got.w, 0.8, delta=1.0e-5)
		self.assertLess(got.distance, 1.0e-5)


class InitShellCropTest(unittest.TestCase):
	def test_cli_data_accepts_fractional_wrap_width_unit(self) -> None:
		parser = argparse.ArgumentParser()
		cli_data.add_args(parser)
		args = parser.parse_args([
			"--input", "/tmp/input.lasagna.json",
			"--model-w", "1.5",
			"--model-w-unit", "wraps",
		])

		cfg = cli_data.from_args(args)

		self.assertEqual(cfg.model_w, 1.5)
		self.assertEqual(cfg.model_w_unit, "wraps")
		self.assertFalse(cfg.corr_point_roi)
		self.assertEqual(cfg.corr_point_roi_init_margin, 80)
		self.assertEqual(cfg.corr_point_roi_output_radius, 20)

	def test_cli_data_accepts_corr_point_roi_args(self) -> None:
		parser = argparse.ArgumentParser()
		cli_data.add_args(parser)
		args = parser.parse_args([
			"--input", "/tmp/input.lasagna.json",
			"--corr-point-roi",
			"--corr-point-roi-init-margin", "5",
			"--corr-point-roi-output-radius", "2",
		])

		cfg = cli_data.from_args(args)

		self.assertTrue(cfg.corr_point_roi)
		self.assertEqual(cfg.corr_point_roi_init_margin, 5)
		self.assertEqual(cfg.corr_point_roi_output_radius, 2)

	def test_corr_point_roi_init_recenters_and_computes_exact_crop_extents(self) -> None:
		width = 32
		radius = 50.0
		angles = [0.0, 0.25 * math.pi, 0.5 * math.pi]
		points_xyz = [
			(radius * math.cos(angles[0]), radius * math.sin(angles[0]), 10.0),
			(radius * math.cos(angles[1]), radius * math.sin(angles[1]), 20.0),
			(radius * math.cos(angles[2]), radius * math.sin(angles[2]), 20.0),
		]
		with tempfile.TemporaryDirectory() as td:
			root = Path(td)
			_write_tifxyz(
				root / "shell_0001.tifxyz",
				_wrapped_cylinder(radius=radius, z_values=[0.0, 10.0, 20.0, 30.0], width=width),
			)
			idx = init_shell_index.InitShellIndex.from_directory(root)
			corr = fit.fit_data.CorrPoints3D(
				points_xyz_winda=torch.tensor([[*p, 0.0] for p in points_xyz], dtype=torch.float32),
				collection_idx=torch.zeros(3, dtype=torch.int64),
				point_ids=torch.arange(3, dtype=torch.int64),
				is_absolute=torch.zeros(3, dtype=torch.bool),
			)
			normals = torch.tensor(
				[[math.cos(a), math.sin(a), 0.0] for a in angles],
				dtype=torch.float32,
			)

			roi = fit._derive_corr_point_roi_init(
				shell_index=idx,
				corr_points=corr,
				normals_xyz=normals,
				mesh_step=10.0,
				init_margin_grid_points=2,
				device=torch.device("cpu"),
			)

		self.assertEqual(roi.payload["usable_point_count"], 3)
		self.assertEqual(roi.payload["projection_mode"], "normal-line")
		self.assertEqual(roi.payload["depth"], 1)
		self.assertAlmostEqual(roi.payload["projected_h_span_grid"], 1.0, delta=1.0e-3)
		self.assertAlmostEqual(roi.payload["projected_w_span_grid"], 8.0, delta=1.0e-3)
		self.assertAlmostEqual(roi.model_h, 50.0, delta=1.0e-3)
		self.assertAlmostEqual(roi.model_w, 120.0, delta=1.0e-3)
		self.assertEqual([round(v, 3) for v in roi.payload["effective_seed"]], [round(v, 3) for v in points_xyz[1]])

	def test_corr_point_roi_line_projection_selects_anchor_nearest_opposite_hit(self) -> None:
		surface = init_shell_index.InitShellSurface(
			shell_id="shell_0001.tifxyz",
			path=Path("shell_0001.tifxyz"),
			xyz_wrapped=_wrapped_cylinder(radius=10.0, z_values=[0.0, 10.0, 20.0], width=64),
			unique_w=64,
		)
		points = torch.tensor([[10.0, 0.0, 10.0]], dtype=torch.float32)
		normals = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)

		valid, h, w, sign = fit._line_project_points_to_selected_shell(
			surface,
			points,
			normals,
			anchor_hw=(1.0, 32.0),
			device=torch.device("cpu"),
		)

		self.assertTrue(bool(valid[0]))
		self.assertAlmostEqual(float(h[0]), 1.0, delta=1.0e-3)
		self.assertAlmostEqual(float(w[0]), 32.0, delta=1.0e-3)
		self.assertEqual(int(sign[0]), -1)

	def test_corr_point_roi_line_projection_uses_periodic_anchor_distance(self) -> None:
		surface = init_shell_index.InitShellSurface(
			shell_id="shell_0001.tifxyz",
			path=Path("shell_0001.tifxyz"),
			xyz_wrapped=_wrapped_cylinder(radius=10.0, z_values=[0.0, 10.0, 20.0], width=64),
			unique_w=64,
		)
		points = torch.tensor([[10.0, 0.0, 10.0]], dtype=torch.float32)
		normals = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)

		valid, _h, w, _sign = fit._line_project_points_to_selected_shell(
			surface,
			points,
			normals,
			anchor_hw=(1.0, 63.8),
			device=torch.device("cpu"),
		)

		self.assertTrue(bool(valid[0]))
		self.assertLess(float(w[0]), 0.5)

	def test_corr_point_roi_line_projection_prefers_grid_nearest_fold_hit(self) -> None:
		width = 8
		base_angles = (torch.arange(width, dtype=torch.float32) % 4.0) * (2.0 * math.pi / 4.0)
		x = 10.0 * torch.cos(base_angles)
		y = 10.0 * torch.sin(base_angles)
		unique = torch.stack([
			torch.stack([x, y, torch.zeros_like(x)], dim=-1),
			torch.stack([x, y, torch.full_like(x, 10.0)], dim=-1),
		], dim=0)
		surface = init_shell_index.InitShellSurface(
			shell_id="folded.tifxyz",
			path=Path("folded.tifxyz"),
			xyz_wrapped=torch.cat([unique, unique[:, :1]], dim=1).contiguous(),
			unique_w=width,
		)
		points = torch.tensor([[0.0, 10.0, 5.0]], dtype=torch.float32)
		normals = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32)

		valid, h, w, _sign = fit._line_project_points_to_selected_shell(
			surface,
			points,
			normals,
			anchor_hw=(0.5, 5.0),
			device=torch.device("cpu"),
		)

		self.assertTrue(bool(valid[0]))
		self.assertTrue(math.isfinite(float(h[0])))
		self.assertAlmostEqual(float(w[0]), 5.0, delta=1.0e-4)

	def test_corr_point_roi_line_projection_skips_top_bottom_misses(self) -> None:
		surface = init_shell_index.InitShellSurface(
			shell_id="shell_0001.tifxyz",
			path=Path("shell_0001.tifxyz"),
			xyz_wrapped=_wrapped_cylinder(radius=10.0, z_values=[0.0, 10.0, 20.0], width=64),
			unique_w=64,
		)
		corr = fit.fit_data.CorrPoints3D(
			points_xyz_winda=torch.tensor([[10.0, 0.0, 100.0, 0.0]], dtype=torch.float32),
			collection_idx=torch.zeros(1, dtype=torch.int64),
			point_ids=torch.zeros(1, dtype=torch.int64),
			is_absolute=torch.zeros(1, dtype=torch.bool),
		)

		projections, skipped = fit._project_corr_points_to_shell(
			surface,
			corr,
			torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
			anchor_hw=(1.0, 0.0),
			device=torch.device("cpu"),
		)

		self.assertEqual(projections, [])
		self.assertEqual(skipped[0]["reason"], "no_line_shell_intersection")

	def test_full_width_crop_cuts_opposite_anchor_and_is_nonperiodic(self) -> None:
		surface = init_shell_index.InitShellSurface(
			shell_id="shell_0001.tifxyz",
			path=Path("shell_0001.tifxyz"),
			xyz_wrapped=_wrapped_cylinder(radius=10.0, z_values=[0.0, 10.0, 20.0], width=64),
			unique_w=64,
		)
		closest = _closest_for_surface(surface, h=1.0, w=0.0, xyz=(10.0, 0.0, 10.0))

		crop, valid, info = init_shell_index.crop_shell_surface(
			surface,
			closest,
			seed=(10.0, 0.0, 10.0),
			model_w=0.0,
			model_h=20.0,
			mesh_step=5.0,
			device="cpu",
		)
		mdl = fit_model.Model3D.from_tifxyz_crop(
			crop,
			valid,
			device=torch.device("cpu"),
			mesh_step=5,
			winding_step=5,
			subsample_mesh=1,
			subsample_winding=1,
		)

		self.assertTrue(info.full_width)
		self.assertEqual(tuple(crop.shape[:2]), (5, 13))
		self.assertEqual((mdl.depth, mdl.mesh_h, mdl.mesh_w), (1, 5, 13))
		center = crop[2, 6]
		self.assertTrue(torch.allclose(center, torch.tensor([10.0, 0.0, 10.0]), atol=0.25))
		self.assertLess(float(crop[:, 0, 0].mean()), -8.0)
		self.assertLess(float(crop[:, -1, 0].mean()), -8.0)
		self.assertFalse(torch.allclose(crop[:, 0], crop[:, -1], atol=1.0e-4, rtol=1.0e-4))

	def test_wrap_width_unrolls_multiple_circumferences(self) -> None:
		surface = init_shell_index.InitShellSurface(
			shell_id="shell_0001.tifxyz",
			path=Path("shell_0001.tifxyz"),
			xyz_wrapped=_wrapped_cylinder(radius=10.0, z_values=[0.0, 10.0, 20.0], width=64),
			unique_w=64,
		)
		closest = _closest_for_surface(surface, h=1.0, w=0.0, xyz=(10.0, 0.0, 10.0))

		legacy_crop, _legacy_valid, legacy_info = init_shell_index.crop_shell_surface(
			surface,
			closest,
			seed=(10.0, 0.0, 10.0),
			model_w=0.0,
			model_h=20.0,
			mesh_step=5.0,
			device="cpu",
		)
		crop, _valid, info = init_shell_index.crop_shell_surface(
			surface,
			closest,
			seed=(10.0, 0.0, 10.0),
			model_w=2.0,
			model_w_unit="wraps",
			model_h=20.0,
			mesh_step=5.0,
			device="cpu",
		)

		self.assertTrue(legacy_info.full_width)
		self.assertFalse(info.full_width)
		self.assertGreater(crop.shape[1], legacy_crop.shape[1])
		self.assertGreaterEqual((crop.shape[1] - 1) * 5.0, 2.0 * info.circumference)
		self.assertLess((crop.shape[1] - 2) * 5.0, 2.0 * info.circumference)

	def test_voxel_width_larger_than_circumference_unrolls(self) -> None:
		surface = init_shell_index.InitShellSurface(
			shell_id="shell_0001.tifxyz",
			path=Path("shell_0001.tifxyz"),
			xyz_wrapped=_wrapped_cylinder(radius=10.0, z_values=[0.0, 10.0, 20.0], width=64),
			unique_w=64,
		)
		closest = _closest_for_surface(surface, h=1.0, w=0.0, xyz=(10.0, 0.0, 10.0))
		legacy_crop, _legacy_valid, legacy_info = init_shell_index.crop_shell_surface(
			surface,
			closest,
			seed=(10.0, 0.0, 10.0),
			model_w=0.0,
			model_h=20.0,
			mesh_step=5.0,
			device="cpu",
		)

		crop, _valid, info = init_shell_index.crop_shell_surface(
			surface,
			closest,
			seed=(10.0, 0.0, 10.0),
			model_w=1.5 * legacy_info.circumference,
			model_w_unit="voxels",
			model_h=20.0,
			mesh_step=5.0,
			device="cpu",
		)

		self.assertFalse(info.full_width)
		self.assertGreater(crop.shape[1], legacy_crop.shape[1])
		self.assertAlmostEqual(info.requested_width_wraps, 1.5, places=6)

	def test_narrow_crop_is_anchor_centered_and_nonperiodic(self) -> None:
		surface = init_shell_index.InitShellSurface(
			shell_id="shell_0001.tifxyz",
			path=Path("shell_0001.tifxyz"),
			xyz_wrapped=_wrapped_cylinder(radius=10.0, z_values=[0.0, 10.0, 20.0], width=64),
			unique_w=64,
		)
		closest = _closest_for_surface(surface, h=1.0, w=0.0, xyz=(10.0, 0.0, 10.0))

		crop, valid, info = init_shell_index.crop_shell_surface(
			surface,
			closest,
			seed=(10.0, 0.0, 10.0),
			model_w=10.0,
			model_h=20.0,
			mesh_step=5.0,
			device="cpu",
		)

		self.assertFalse(info.full_width)
		self.assertEqual(tuple(crop.shape[:2]), (5, 3))
		self.assertTrue(bool(valid.all()))
		self.assertTrue(torch.allclose(crop[2, 1], torch.tensor([10.0, 0.0, 10.0]), atol=0.25))
		self.assertFalse(torch.allclose(crop[:, 0], crop[:, -1], atol=1.0e-4, rtol=1.0e-4))

	def test_height_crop_drops_out_of_range_rows_instead_of_repeating_boundary(self) -> None:
		surface = init_shell_index.InitShellSurface(
			shell_id="shell_0001.tifxyz",
			path=Path("shell_0001.tifxyz"),
			xyz_wrapped=_wrapped_cylinder(radius=10.0, z_values=[0.0, 10.0, 20.0, 30.0, 40.0], width=64),
			unique_w=64,
		)
		closest = _closest_for_surface(surface, h=3.0, w=0.0, xyz=(10.0, 0.0, 30.0))

		crop, valid, info = init_shell_index.crop_shell_surface(
			surface,
			closest,
			seed=(10.0, 0.0, 30.0),
			model_w=10.0,
			model_h=60.0,
			mesh_step=10.0,
			device="cpu",
		)

		self.assertEqual(info.requested_mesh_h, 7)
		self.assertEqual(info.mesh_h, 3)
		self.assertEqual(info.height_dropped_low, 2)
		self.assertEqual(info.height_dropped_high, 2)
		self.assertEqual(tuple(crop.shape[:2]), (3, 2))
		self.assertTrue(bool(valid.all()))
		self.assertTrue(torch.allclose(_grid_center(crop), torch.tensor(closest.closest_xyz), atol=1.0e-5))
		self.assertGreater(float((crop[1] - crop[0]).norm(dim=-1).mean()), 5.0)
		self.assertGreater(float((crop[2] - crop[1]).norm(dim=-1).mean()), 5.0)

	def test_source_shell_row_quality_trim_removes_bad_full_rows(self) -> None:
		width = 64
		angles = torch.arange(width, dtype=torch.float32) * (2.0 * math.pi / float(width))
		rows = []
		for radius, z in [(5.0, 0.0), (100.0, 10.0), (100.0, 20.0), (100.0, 30.0), (5.0, 40.0)]:
			rows.append(torch.stack([
				float(radius) * torch.cos(angles),
				float(radius) * torch.sin(angles),
				torch.full_like(angles, float(z)),
			], dim=-1))
		unique = torch.stack(rows, dim=0)
		surface = init_shell_index.InitShellSurface(
			shell_id="shell_0001.tifxyz",
			path=Path("shell_0001.tifxyz"),
			xyz_wrapped=torch.cat([unique, unique[:, :1]], dim=1).contiguous(),
			unique_w=width,
			source_step=10.0,
		)

		trimmed, trim_top, trim_bottom = init_shell_index.trim_shell_surface_rows_by_quality(
			surface,
			target_step=10.0,
		)

		self.assertEqual(trim_top, 1)
		self.assertEqual(trim_bottom, 1)
		self.assertEqual(tuple(trimmed.xyz_wrapped.shape[:2]), (3, 65))

	def test_crop_grid_center_uses_exact_closest_xyz_not_resampled_hw(self) -> None:
		unique = torch.tensor(
			[
				[
					[0.0, 0.0, 0.0],
					[10.0, 0.0, 0.0],
					[20.0, 0.0, 0.0],
					[30.0, 0.0, 0.0],
				],
				[
					[0.0, 10.0, 0.0],
					[10.0, 10.0, 80.0],
					[20.0, 10.0, 0.0],
					[30.0, 10.0, 0.0],
				],
				[
					[0.0, 20.0, 0.0],
					[10.0, 20.0, 0.0],
					[20.0, 20.0, 0.0],
					[30.0, 20.0, 0.0],
				],
			],
			dtype=torch.float32,
		)
		surface = init_shell_index.InitShellSurface(
			shell_id="shell_0001.tifxyz",
			path=Path("shell_0001.tifxyz"),
			xyz_wrapped=torch.cat([unique, unique[:, :1]], dim=1),
			unique_w=4,
		)
		closest = _closest_for_surface(
			surface,
			h=0.5,
			w=0.3,
			xyz=(3.0, 5.0, 24.0),
		)

		crop, _valid, _info = init_shell_index.crop_shell_surface(
			surface,
			closest,
			seed=(3.0, 5.0, 24.0),
			model_w=10.0,
			model_h=10.0,
			mesh_step=5.0,
			device="cpu",
		)

		self.assertTrue(torch.allclose(_grid_center(crop), torch.tensor(closest.closest_xyz), atol=1.0e-5))


if __name__ == "__main__":
	unittest.main()
