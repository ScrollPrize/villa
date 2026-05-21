from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import tifffile


ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
	sys.path.insert(0, ROOT)

import fit
import fit2tifxyz
import model as fit_model
import opt_loss_flatten
import optimizer


def _flat_grid(h: int, w: int, *, sx: float = 1.0, sy: float = 1.0) -> torch.Tensor:
	yy = torch.arange(h, dtype=torch.float32).view(h, 1).expand(h, w)
	xx = torch.arange(w, dtype=torch.float32).view(1, w).expand(h, w)
	zz = torch.zeros(h, w, dtype=torch.float32)
	return torch.stack([xx * sx, yy * sy, zz], dim=-1)


def _make_flatten_model(
	xyz: torch.Tensor,
	valid: torch.Tensor | None = None,
	*,
	mesh_step: int = 1,
	flatten_filter_source_angles: bool = False,
	flatten_filter_angle_deg: float = 90.0,
	flatten_filter_radius: int = 2,
) -> fit_model.Model3D:
	if valid is None:
		valid = torch.ones(xyz.shape[:2], dtype=torch.bool)
	return fit_model.Model3D.from_flatten_tifxyz_crop(
		xyz,
		valid,
		device=torch.device("cpu"),
		mesh_step=mesh_step,
		winding_step=1,
		subsample_mesh=1,
		subsample_winding=1,
		flatten_filter_source_angles=flatten_filter_source_angles,
		flatten_filter_angle_deg=flatten_filter_angle_deg,
		flatten_filter_radius=flatten_filter_radius,
	)


def _set_flatten_map(mdl: fit_model.Model3D, map_yx: torch.Tensor) -> None:
	flat = map_yx.permute(2, 0, 1).unsqueeze(1).contiguous()
	mdl.flatten_map_ms = fit_model.Model3D._construct_pyramid_from_flat_3d(
		flat,
		len(mdl.flatten_map_ms),
		pyramid_d=False,
	)


class FlattenLossTest(unittest.TestCase):
	def test_flatten_stage_defaults_disable_volume_losses(self) -> None:
		stages = optimizer.load_stages_cfg({
			"args": {"model-init": "flatten"},
			"base": {
				"flatten_sdir": 1.0,
				"flatten_map_step": 0.001,
				"flatten_avg_offset": 1.0,
				"flatten_orient": 0.001,
			},
			"stages": [{
				"name": "flatten",
				"global_opt": {
					"steps": 1,
					"lr": 0.1,
					"params": ["flatten_map_ms"],
				},
			}],
		})

		self.assertEqual(stages[0].global_opt.eff["normal"], 0.0)
		self.assertEqual(stages[0].global_opt.eff["flatten_sdir"], 1.0)
		self.assertEqual(stages[0].global_opt.eff["flatten_map_step"], 0.001)
		self.assertEqual(stages[0].global_opt.eff["flatten_avg_offset"], 1.0)
		self.assertEqual(stages[0].global_opt.eff["flatten_orient"], 0.001)

	def test_auto_steps_config_uses_max_cap_for_progress_budget(self) -> None:
		stages = optimizer.load_stages_cfg({
			"args": {"model-init": "flatten"},
			"base": {"flatten_avg_offset": 1.0},
			"stages": [{
				"name": "flatten",
				"global_opt": {
					"steps": "auto",
					"lr": 0.0,
					"params": ["flatten_map_ms"],
					"args": {
						"auto_steps_max": 12,
						"auto_steps_window": 3,
						"auto_steps_min": 3,
						"auto_steps_rel_threshold": 1.0e-6,
					},
				},
			}],
		})

		self.assertTrue(stages[0].global_opt.steps_auto)
		self.assertEqual(stages[0].global_opt.steps, 12)
		self.assertEqual(optimizer.total_steps_for_stages(stages), 12)

	def test_auto_steps_relative_improvement_uses_best_before_and_recent_window(self) -> None:
		history = [10.0, 1.0, 9.0, 8.0, 0.9]

		rel = optimizer._auto_steps_relative_improvement(history, window=2)

		self.assertAlmostEqual(rel, 0.1, places=6)

	def test_auto_steps_stops_when_window_improvement_is_small(self) -> None:
		mdl = _make_flatten_model(_flat_grid(5, 5), mesh_step=1)
		stages = optimizer.load_stages_cfg({
			"args": {"model-init": "flatten"},
			"base": {"flatten_avg_offset": 1.0},
			"stages": [{
				"name": "flatten",
				"global_opt": {
					"steps": "auto",
					"lr": 0.0,
					"params": ["flatten_map_ms"],
					"args": {
						"auto_steps_max": 10,
						"auto_steps_window": 3,
						"auto_steps_min": 3,
						"auto_steps_rel_threshold": 1.0e-6,
						"status_interval": 0,
						"flatten_max_update": 0.0,
					},
				},
			}],
		})
		progress_steps: list[int] = []

		optimizer.optimize(
			model=mdl,
			data=fit._dummy_flatten_data(),
			stages=stages,
			snapshot_interval=0,
			snapshot_fn=lambda **_kw: None,
			progress_fn=lambda **kw: progress_steps.append(int(kw["step"])),
		)

		self.assertEqual(progress_steps[-1], 3)
		self.assertLess(progress_steps[-1], stages[0].global_opt.steps)

	def test_flatten_pyramid_reaches_two_in_longer_dimension(self) -> None:
		mdl = _make_flatten_model(_flat_grid(5, 9), mesh_step=1)
		shapes = [(int(p.shape[2]), int(p.shape[3])) for p in mdl.flatten_map_ms]

		self.assertEqual(shapes, [(6, 11), (3, 6), (2, 3), (2, 2)])
		self.assertEqual(max(shapes[-1]), 2)

	def test_flatten_init_uses_centered_twenty_percent_larger_output_canvas(self) -> None:
		mdl = _make_flatten_model(_flat_grid(11, 21), mesh_step=1)
		map_yx = mdl.flatten_map().detach()

		self.assertEqual(tuple(map_yx.shape), (13, 25, 2))
		self.assertEqual(mdl.mesh_h, 13)
		self.assertEqual(mdl.mesh_w, 25)
		self.assertAlmostEqual(float(map_yx[0, 0, 0]), -1.0, places=6)
		self.assertAlmostEqual(float(map_yx[0, 0, 1]), -2.0, places=6)
		self.assertAlmostEqual(float(map_yx[-1, -1, 0]), 11.0, places=6)
		self.assertAlmostEqual(float(map_yx[-1, -1, 1]), 22.0, places=6)

	def test_bilinear_validity_rejects_cells_with_any_invalid_corner(self) -> None:
		xyz = _flat_grid(4, 4)
		valid = torch.ones(4, 4, dtype=torch.bool)
		valid[1, 1] = False
		cell_valid = fit_model.Model3D._source_cell_valid(valid)
		map_yx = torch.tensor([[[0.25, 0.25], [1.25, 1.25]]], dtype=torch.float32)

		_sampled, point_valid = fit_model.Model3D._flatten_sample_map(xyz, cell_valid, map_yx)

		self.assertFalse(bool(point_valid[0, 0]))
		self.assertFalse(bool(point_valid[0, 1]))

	def test_source_angle_filter_punches_bad_source_cells(self) -> None:
		xyz = _flat_grid(5, 5)
		xyz[2, :, 1] = 0.0
		mdl = _make_flatten_model(
			xyz,
			mesh_step=1,
			flatten_filter_source_angles=True,
			flatten_filter_angle_deg=90.0,
			flatten_filter_radius=0,
		)
		stats = mdl.flatten_source_filter_stats

		self.assertGreater(stats["bad_pairs"], 0.0)
		self.assertGreater(stats["bad_cells"], 0.0)
		self.assertLess(stats["cell_valid_after"], stats["cell_valid_before"])

		map_yx = torch.tensor([[[1.25, 1.25], [3.25, 1.25]]], dtype=torch.float32)
		_sampled, point_valid = fit_model.Model3D._flatten_sample_map(
			mdl.flatten_source_xyz,
			mdl.flatten_source_cell_valid,
			map_yx,
		)

		self.assertFalse(bool(point_valid[0, 0]))
		self.assertTrue(bool(point_valid[0, 1]))

	def test_identity_flat_regular_grid_has_near_zero_sdir(self) -> None:
		mdl = _make_flatten_model(_flat_grid(5, 5), mesh_step=1)
		res = mdl(fit._dummy_flatten_data(), needs=fit_model.ModelForwardNeeds(flatten=True))

		loss, _lms, _masks = opt_loss_flatten.flatten_sdir_loss(res=res)

		self.assertLess(float(loss.detach()), 1.0e-6)
		self.assertGreater(int(res.flatten_quad_mask.sum()), 0)

	def test_flatten_target_step_is_measured_not_mesh_step(self) -> None:
		mdl = _make_flatten_model(_flat_grid(5, 5, sx=3.0, sy=3.0), mesh_step=20)
		res = mdl(fit._dummy_flatten_data(), needs=fit_model.ModelForwardNeeds(flatten=True))

		loss, _lms, _masks = opt_loss_flatten.flatten_sdir_loss(res=res)

		self.assertAlmostEqual(float(res.flatten_target_step.detach()), 3.0, places=5)
		self.assertLess(float(loss.detach()), 1.0e-6)

	def test_anisotropic_deformation_increases_sdir(self) -> None:
		mdl = _make_flatten_model(_flat_grid(7, 7), mesh_step=1)
		map_yx = mdl.flatten_map().detach().clone()
		map_yx[..., 1] = map_yx[..., 1] * 0.5
		_set_flatten_map(mdl, map_yx)
		res = mdl(fit._dummy_flatten_data(), needs=fit_model.ModelForwardNeeds(flatten=True))

		loss, _lms, _masks = opt_loss_flatten.flatten_sdir_loss(res=res)

		self.assertGreater(float(loss.detach()), 0.5)

	def test_map_step_regularizer_rejects_checkerboard(self) -> None:
		mdl = _make_flatten_model(_flat_grid(6, 6), mesh_step=1)
		res = mdl(fit._dummy_flatten_data(), needs=fit_model.ModelForwardNeeds(flatten=True))
		loss_id, _lms, _masks = opt_loss_flatten.flatten_map_step_loss(res=res)
		self.assertLess(float(loss_id.detach()), 1.0e-6)

		map_yx = mdl.flatten_map().detach().clone()
		H, W = int(map_yx.shape[0]), int(map_yx.shape[1])
		yy = torch.arange(H, dtype=torch.float32).view(H, 1).expand(H, W)
		xx = torch.arange(W, dtype=torch.float32).view(1, W).expand(H, W)
		checker = ((yy.long() + xx.long()) % 2).to(dtype=torch.float32) * 0.25
		map_yx[..., 0] = map_yx[..., 0] + checker
		map_yx[..., 1] = map_yx[..., 1] - checker
		_set_flatten_map(mdl, map_yx)
		res = mdl(fit._dummy_flatten_data(), needs=fit_model.ModelForwardNeeds(flatten=True))
		loss_checker, _lms, _masks = opt_loss_flatten.flatten_map_step_loss(res=res)

		self.assertGreater(float(loss_checker.detach()), 0.05)

	def test_avg_offset_regularizer_keeps_initial_valid_mean_offset(self) -> None:
		mdl = _make_flatten_model(_flat_grid(6, 6), mesh_step=1)
		res = mdl(fit._dummy_flatten_data(), needs=fit_model.ModelForwardNeeds(flatten=True))
		loss_id, _lms, _masks = opt_loss_flatten.flatten_avg_offset_loss(res=res)
		self.assertLess(float(loss_id.detach()), 1.0e-6)

		map_yx = mdl.flatten_map().detach().clone()
		map_yx[..., 0] = map_yx[..., 0] + 0.5
		map_yx[..., 1] = map_yx[..., 1] - 0.25
		_set_flatten_map(mdl, map_yx)
		res = mdl(fit._dummy_flatten_data(), needs=fit_model.ModelForwardNeeds(flatten=True))
		loss_shifted, _lms, _masks = opt_loss_flatten.flatten_avg_offset_loss(res=res)

		self.assertAlmostEqual(float(loss_shifted.detach()), 0.5 * 0.5 + 0.25 * 0.25, places=5)

	def test_flatten_update_clamp_limits_each_scale(self) -> None:
		params = [
			torch.nn.Parameter(torch.zeros(2, 1, 4, 4)),
			torch.nn.Parameter(torch.zeros(2, 1, 2, 2)),
		]
		before = [p.detach().clone() for p in params]
		with torch.no_grad():
			params[0].add_(1.0)
			params[1].add_(1.0)

		optimizer._clamp_flatten_map_ms_update(params, before, base_step=0.1)

		self.assertLessEqual(float(torch.linalg.vector_norm((params[0] - before[0]).detach(), dim=0).max()), 0.10001)
		self.assertLessEqual(float(torch.linalg.vector_norm((params[1] - before[1]).detach(), dim=0).max()), 0.20001)

	def test_orient_regularizer_allows_positive_area_stretch(self) -> None:
		opt_loss_flatten.configure(orient_min_det=0.0, reset_history=True)
		mdl = _make_flatten_model(_flat_grid(6, 6), mesh_step=1)
		map_yx = mdl.flatten_map().detach().clone()
		map_yx[..., 0] = map_yx[..., 0] * 0.05
		map_yx[..., 1] = map_yx[..., 1] * 12.0
		_set_flatten_map(mdl, map_yx)
		res = mdl(fit._dummy_flatten_data(), needs=fit_model.ModelForwardNeeds(flatten=True))

		loss, _lms, _masks = opt_loss_flatten.flatten_orient_loss(res=res)
		stats = opt_loss_flatten.last_stats()

		self.assertLess(float(loss.detach()), 1.0e-6)
		self.assertEqual(stats["flatten_orient_fold_frac"], 0.0)
		self.assertEqual(stats["flatten_orient_lowdet_frac"], 0.0)
		self.assertGreater(stats["flatten_orient_min_det"], 0.0)

	def test_orient_regularizer_rejects_negative_area_fold(self) -> None:
		opt_loss_flatten.configure(orient_min_det=0.0, reset_history=True)
		mdl = _make_flatten_model(_flat_grid(6, 6), mesh_step=1)
		map_yx = mdl.flatten_map().detach().clone()
		map_yx[..., 1] = -map_yx[..., 1]
		_set_flatten_map(mdl, map_yx)
		res = mdl(fit._dummy_flatten_data(), needs=fit_model.ModelForwardNeeds(flatten=True))
		loss, _lms, masks = opt_loss_flatten.flatten_orient_loss(res=res)
		stats = opt_loss_flatten.last_stats()

		self.assertGreater(float(loss.detach()), 20.0)
		self.assertEqual(stats["flatten_orient_fold_frac"], 1.0)
		self.assertEqual(stats["flatten_orient_lowdet_frac"], 1.0)
		self.assertLess(stats["flatten_orient_min_det"], 0.0)
		self.assertEqual(int(masks[0].sum().detach()), 36)

	def test_flatten_stats_track_validity_transitions(self) -> None:
		opt_loss_flatten.configure(reset_history=True)
		mdl = _make_flatten_model(_flat_grid(5, 5), mesh_step=1)
		res = mdl(fit._dummy_flatten_data(), needs=fit_model.ModelForwardNeeds(flatten=True))
		opt_loss_flatten.flatten_sdir_loss(res=res)

		map_yx = mdl.flatten_map().detach().clone()
		map_yx[1, 1] = torch.tensor([-1.0, -1.0])
		map_yx[0, 0] = torch.tensor([0.5, 0.5])
		_set_flatten_map(mdl, map_yx)
		res = mdl(fit._dummy_flatten_data(), needs=fit_model.ModelForwardNeeds(flatten=True))
		opt_loss_flatten.flatten_sdir_loss(res=res)
		stats = opt_loss_flatten.last_stats()

		self.assertAlmostEqual(stats["flatten_valid_to_invalid"], 1.0 / 36.0, places=6)
		self.assertAlmostEqual(stats["flatten_invalid_to_valid"], 1.0 / 36.0, places=6)
		self.assertIn("flatten_sdir_no_new", stats)

	def test_flatten_export_writes_invalid_points_as_minus_one(self) -> None:
		xyz = _flat_grid(4, 4)
		valid = torch.ones(4, 4, dtype=torch.bool)
		valid[1, 1] = False
		mdl = _make_flatten_model(xyz, valid, mesh_step=1)
		with tempfile.TemporaryDirectory() as td:
			out = Path(td)
			fit._export_flatten_result(
				mdl=mdl,
				data=fit._dummy_flatten_data(),
				out_dir=out,
				scale=1.0,
				voxel_size_um=None,
				fit_config={},
				model_source=None,
			)
			self.assertFalse((out / "map_y.tif").exists())
			self.assertFalse((out / "map_x.tif").exists())
			x = tifffile.imread(str(out / "flatten.tifxyz" / "x.tif"))
			y = tifffile.imread(str(out / "flatten.tifxyz" / "y.tif"))
			z = tifffile.imread(str(out / "flatten.tifxyz" / "z.tif"))
			self.assertTrue(bool(((x == -1.0) & (y == -1.0) & (z == -1.0)).any()))

	def test_fit2tifxyz_exports_flatten_checkpoint(self) -> None:
		xyz = _flat_grid(4, 4)
		valid = torch.ones(4, 4, dtype=torch.bool)
		valid[1, 1] = False
		mdl = _make_flatten_model(xyz, valid, mesh_step=1)
		with tempfile.TemporaryDirectory() as td:
			root = Path(td)
			model_path = root / "flatten_model.pt"
			out = root / "out"
			fit._save_flatten_model(
				str(model_path),
				mdl=mdl,
				data=fit._dummy_flatten_data(),
				fit_config={"args": {"model-init": "flatten"}},
			)

			fit2tifxyz.main([
				"--input", str(model_path),
				"--output", str(out),
				"--output-name", "vc3d_name.tifxyz",
			])

			self.assertFalse((out / "map_y.tif").exists())
			self.assertFalse((out / "map_x.tif").exists())
			self.assertTrue((out / "vc3d_name.tifxyz" / "meta.json").exists())
			x = tifffile.imread(str(out / "vc3d_name.tifxyz" / "x.tif"))
			y = tifffile.imread(str(out / "vc3d_name.tifxyz" / "y.tif"))
			z = tifffile.imread(str(out / "vc3d_name.tifxyz" / "z.tif"))
			self.assertTrue(bool(((x == -1.0) & (y == -1.0) & (z == -1.0)).any()))


if __name__ == "__main__":
	unittest.main()
