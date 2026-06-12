from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
	sys.path.insert(0, ROOT)

import fit_data
import model as fit_model
import optimizer
import opt_loss_atlas_line


def _fit_data(lines: fit_data.AtlasLines3D) -> fit_data.FitData3D:
	return fit_data.FitData3D(
		cos=None,
		grad_mag=None,
		nx=None,
		ny=None,
		pred_dt=None,
		corr_points=None,
		winding_volume=None,
		origin_fullres=(0.0, 0.0, 0.0),
		spacing=(1.0, 1.0, 1.0),
		atlas_lines=lines,
		_vol_size=(1, 1, 1),
	)


def _normal_grid(xyz: torch.Tensor, *, sign: float = 1.0) -> torch.Tensor:
	n = torch.zeros_like(xyz)
	n[..., 2] = float(sign)
	return n


def _result(
	xyz: torch.Tensor,
	lines: fit_data.AtlasLines3D,
	*,
	normals: torch.Tensor | None = None,
) -> fit_model.FitResult3D:
	D, H, W, _ = xyz.shape
	if normals is None:
		normals = _normal_grid(xyz)
	return fit_model.FitResult3D(
		xyz_lr=xyz,
		xyz_hr=None,
		data=_fit_data(lines),
		data_s=None,
		data_lr=None,
		target_plain=None,
		target_mod=None,
		amp_lr=torch.ones(D, 1, H, W, dtype=xyz.dtype),
		bias_lr=torch.zeros(D, 1, H, W, dtype=xyz.dtype),
		mask_hr=None,
		mask_lr=None,
		normals=normals,
		xy_conn=None,
		mask_conn=None,
		sign_conn=None,
		params=fit_model.ModelParams3D(
			mesh_step=1,
			winding_step=1,
			subsample_mesh=1,
			subsample_winding=1,
			scaledown=1.0,
			z_step_eff=1,
			volume_extent=None,
			pyramid_d=False,
		),
	)


def _plane_grid() -> torch.Tensor:
	H, W = 3, 4
	y = torch.arange(H, dtype=torch.float32).view(H, 1).expand(H, W)
	x = torch.arange(W, dtype=torch.float32).view(1, W).expand(H, W)
	z = torch.zeros_like(x)
	return torch.stack([x, y, z], dim=-1).unsqueeze(0).contiguous()


class AtlasLineLossTest(unittest.TestCase):
	def test_debug_payload_is_only_generated_when_requested(self) -> None:
		opt_loss_atlas_line.reset_state()
		xyz = _plane_grid().requires_grad_(True)
		lines = fit_data.AtlasLines3D(
			target_xyz=torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
			normal_xyz=torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32),
			model_h=torch.tensor([1.0], dtype=torch.float32),
			model_w=torch.tensor([1.0], dtype=torch.float32),
		)

		opt_loss_atlas_line.atlas_line_loss(res=_result(xyz, lines))
		self.assertIsNone(opt_loss_atlas_line.last_debug_payload())

		opt_loss_atlas_line.atlas_line_loss(res=_result(xyz, lines), debug_payload=True)
		payload = opt_loss_atlas_line.last_debug_payload()
		self.assertIsNotNone(payload)
		self.assertEqual(tuple(payload.valid.shape), (1, 1))

	def test_atlas_control_points_results_from_debug_payload(self) -> None:
		opt_loss_atlas_line.reset_state()
		xyz = _plane_grid().requires_grad_(True)
		lines = fit_data.AtlasLines3D(
			target_xyz=torch.tensor([
				[1.0, 1.0, 1.0],
				[2.0, 1.0, 0.0],
				[1.0, 1.0, 2.0],
			], dtype=torch.float32),
			normal_xyz=torch.tensor([
				[0.0, 0.0, -1.0],
				[0.0, 0.0, -1.0],
				[0.0, 0.0, -1.0],
			], dtype=torch.float32),
			model_h=torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32),
			model_w=torch.tensor([1.0, 2.0, 1.0], dtype=torch.float32),
			object_ids=("fiber_a", "fiber_a", "fiber_a"),
			source_indices=(7, 8, 7),
			is_control_point=torch.tensor([True, False, False]),
			is_snap_point=torch.tensor([False, False, True]),
		)

		opt_loss_atlas_line.atlas_line_loss(
			res=_result(xyz, lines),
			stage_eff={"atlas_line_control": 1.0, "atlas_line_other": 1.0, "atlas_line_snap": 1.0},
			debug_payload=True,
		)
		results = opt_loss_atlas_line.atlas_control_points_results(lines=lines)
		self.assertIsNotNone(results)
		self.assertEqual(results["summary"]["total_count"], 1)
		self.assertEqual(results["summary"]["valid_count"], 1)
		record = results["records"][0]
		self.assertEqual(record["fiber_id"], "fiber_a")
		self.assertEqual(record["source_index"], 7)
		self.assertEqual(record["control_index"], 0)
		self.assertTrue(record["valid"])
		self.assertAlmostEqual(record["distance"], 1.0, places=6)
		self.assertAlmostEqual(record["signed_delta"], 1.0, places=6)
		self.assertEqual(record["target_xyz"], [1.0, 1.0, 1.0])
		self.assertEqual(record["mesh_xyz"], [1.0, 1.0, 0.0])
		self.assertAlmostEqual(record["model_h"], 1.0, places=6)
		self.assertAlmostEqual(record["model_w"], 1.0, places=6)
		self.assertTrue(record["snap_valid"])
		self.assertEqual(record["snap_direction"], "fw")
		self.assertEqual(record["snap_status"], "valid fw")
		self.assertAlmostEqual(record["snap_signed_delta"], 2.0, places=6)
		self.assertEqual(record["snap_target_xyz"], [1.0, 1.0, 2.0])

	def test_debug_payload_splits_control_other_and_normal_proxy(self) -> None:
		opt_loss_atlas_line.reset_state()
		xyz = _plane_grid().requires_grad_(True)
		lines = fit_data.AtlasLines3D(
			target_xyz=torch.tensor([
				[1.0, 1.0, 1.0],
				[2.0, 1.0, 2.0],
			], dtype=torch.float32),
			normal_xyz=torch.tensor([
				[0.0, 0.0, -1.0],
				[0.0, 0.0, -1.0],
			], dtype=torch.float32),
			model_h=torch.tensor([1.0, 1.0], dtype=torch.float32),
			model_w=torch.tensor([1.0, 2.0], dtype=torch.float32),
			is_control_point=torch.tensor([True, False]),
		)

		opt_loss_atlas_line.atlas_line_loss(res=_result(xyz, lines), debug_payload=True)
		payload = opt_loss_atlas_line.last_debug_payload()
		self.assertIsNotNone(payload)
		self.assertTrue(bool(payload.valid[0, 0]))
		self.assertTrue(bool(payload.valid[0, 1]))
		self.assertTrue(bool(payload.is_control[0, 0]))
		self.assertFalse(bool(payload.is_control[0, 1]))
		self.assertEqual(tuple(payload.normal_proxy_valid.shape), (1, 3, 4))
		self.assertGreater(int(payload.normal_proxy_valid.sum()), int(payload.valid.sum()))
		self.assertTrue(torch.isfinite(payload.normal_proxy_target_xyz[payload.normal_proxy_valid]).all().item())

	def test_atlas_line_snap_weight_selects_only_snap_samples(self) -> None:
		opt_loss_atlas_line.reset_state()
		xyz = _plane_grid().requires_grad_(True)
		lines = fit_data.AtlasLines3D(
			target_xyz=torch.tensor([
				[1.0, 1.0, 1.0],
				[2.0, 1.0, 2.0],
				[1.0, 2.0, 3.0],
			], dtype=torch.float32),
			normal_xyz=torch.tensor([
				[0.0, 0.0, -1.0],
				[0.0, 0.0, -1.0],
				[0.0, 0.0, -1.0],
			], dtype=torch.float32),
			model_h=torch.tensor([1.0, 1.0, 2.0], dtype=torch.float32),
			model_w=torch.tensor([1.0, 2.0, 1.0], dtype=torch.float32),
			is_control_point=torch.tensor([True, False, False]),
			is_snap_point=torch.tensor([False, False, True]),
		)

		result = opt_loss_atlas_line.atlas_line_loss(
			res=_result(xyz, lines),
			stage_eff={"atlas_line_snap": 1.0},
		)
		stats = opt_loss_atlas_line.last_stats()
		self.assertAlmostEqual(stats["atlas_line_control_valid"], 0.0, delta=1.0e-12)
		self.assertAlmostEqual(stats["atlas_line_other_valid"], 0.0, delta=1.0e-12)
		self.assertAlmostEqual(stats["atlas_line_snap_valid"], 1.0, delta=1.0e-12)
		self.assertGreater(float(result["atlas_line_snap"][0].detach()), 0.0)
		self.assertEqual(float(result["atlas_line_control"][0].detach()), 0.0)
		self.assertEqual(float(result["atlas_line_other"][0].detach()), 0.0)

	def test_debug_payload_omits_invalid_samples(self) -> None:
		opt_loss_atlas_line.reset_state()
		xyz = _plane_grid().requires_grad_(True)
		lines = fit_data.AtlasLines3D(
			target_xyz=torch.tensor([
				[1.0, 1.0, 1.0],
				[float("nan"), 1.0, 2.0],
			], dtype=torch.float32),
			normal_xyz=torch.tensor([
				[0.0, 0.0, -1.0],
				[0.0, 0.0, -1.0],
			], dtype=torch.float32),
			model_h=torch.tensor([1.0, 1.0], dtype=torch.float32),
			model_w=torch.tensor([1.0, 2.0], dtype=torch.float32),
			is_control_point=torch.tensor([True, False]),
		)

		opt_loss_atlas_line.atlas_line_loss(res=_result(xyz, lines), debug_payload=True)
		payload = opt_loss_atlas_line.last_debug_payload()
		self.assertIsNotNone(payload)
		self.assertEqual(payload.valid.tolist(), [[True, False]])

	def test_write_debug_objs_uses_working_directory_and_expected_line_sets(self) -> None:
		opt_loss_atlas_line.reset_state()
		xyz = _plane_grid().requires_grad_(True)
		lines = fit_data.AtlasLines3D(
			target_xyz=torch.tensor([
				[1.0, 1.0, 1.0],
				[1.0, 2.0, 1.5],
				[2.0, 1.0, 2.0],
			], dtype=torch.float32),
			normal_xyz=torch.tensor([
				[0.0, 0.0, -1.0],
				[0.0, 0.0, -1.0],
				[0.0, 0.0, -1.0],
			], dtype=torch.float32),
			model_h=torch.tensor([1.0, 2.0, 1.0], dtype=torch.float32),
			model_w=torch.tensor([1.0, 1.0, 2.0], dtype=torch.float32),
			object_ids=("fibers/a.json", "fibers/b.json", "fibers/a.json"),
			is_control_point=torch.tensor([True, True, False]),
		)
		opt_loss_atlas_line.atlas_line_loss(res=_result(xyz, lines), debug_payload=True)

		old_cwd = Path.cwd()
		with tempfile.TemporaryDirectory() as td:
			try:
				os.chdir(td)
				root = opt_loss_atlas_line.write_debug_objs(stage="atlas_init_relax", step=0)
			finally:
				os.chdir(old_cwd)
			self.assertEqual(root, Path(td) / "atlas_debug_objs" / "atlas_init_relax_step000000")
			wdir = root / "winding_000"
			self.assertTrue((wdir / "model_surface.obj").exists())
			self.assertIn("\nf ", (wdir / "model_surface.obj").read_text(encoding="utf-8"))
			self.assertEqual((wdir / "control_connections.obj").read_text(encoding="utf-8").count("\nl "), 2)
			self.assertEqual((wdir / "other_connections.obj").read_text(encoding="utf-8").count("\nl "), 1)
			normal_obj = (wdir / "normal_proxy.obj").read_text(encoding="utf-8")
			self.assertEqual(normal_obj.count("\nl "), 12)
			self.assertIn("v 0 0 0", normal_obj)
			by_fiber = root / "control_connections_by_fiber"
			self.assertEqual((by_fiber / "fibers_a.json.obj").read_text(encoding="utf-8").count("\nl "), 1)
			self.assertEqual((by_fiber / "fibers_b.json.obj").read_text(encoding="utf-8").count("\nl "), 1)

	def test_write_debug_objs_splits_atlas_wraps_by_model_columns(self) -> None:
		opt_loss_atlas_line.reset_state()
		H, W = 3, 8
		y = torch.arange(H, dtype=torch.float32).view(H, 1).expand(H, W)
		x = torch.arange(W, dtype=torch.float32).view(1, W).expand(H, W)
		xyz = torch.stack([x, y, torch.zeros_like(x)], dim=-1).unsqueeze(0).requires_grad_(True)
		lines = fit_data.AtlasLines3D(
			target_xyz=torch.tensor([
				[1.0, 1.0, 1.0],
				[5.0, 1.0, 2.0],
			], dtype=torch.float32),
			normal_xyz=torch.tensor([
				[0.0, 0.0, -1.0],
				[0.0, 0.0, -1.0],
			], dtype=torch.float32),
			model_h=torch.tensor([1.0, 1.0], dtype=torch.float32),
			model_w=torch.tensor([1.0, 5.0], dtype=torch.float32),
			is_control_point=torch.tensor([True, False]),
			atlas_winding_model_ranges=((0, 0.0, 4.0), (1, 4.0, 8.0)),
		)
		opt_loss_atlas_line.atlas_line_loss(res=_result(xyz, lines), debug_payload=True)

		old_cwd = Path.cwd()
		with tempfile.TemporaryDirectory() as td:
			try:
				os.chdir(td)
				root = opt_loss_atlas_line.write_debug_objs(stage="atlas_init_relax", step=0)
			finally:
				os.chdir(old_cwd)
			self.assertEqual(root, Path(td) / "atlas_debug_objs" / "atlas_init_relax_step000000")
			self.assertFalse((root / "winding_000").exists())
			w0 = root / "atlas_winding_+000"
			w1 = root / "atlas_winding_+001"
			self.assertTrue((w0 / "model_surface.obj").exists())
			self.assertTrue((w1 / "model_surface.obj").exists())
			self.assertEqual((w0 / "control_connections.obj").read_text(encoding="utf-8").count("\nl "), 1)
			self.assertEqual((w0 / "other_connections.obj").read_text(encoding="utf-8").count("\nl "), 0)
			self.assertEqual((w1 / "control_connections.obj").read_text(encoding="utf-8").count("\nl "), 0)
			self.assertEqual((w1 / "other_connections.obj").read_text(encoding="utf-8").count("\nl "), 1)
			self.assertGreater((w0 / "normal_proxy.obj").read_text(encoding="utf-8").count("\nl "), 0)
			self.assertGreater((w1 / "normal_proxy.obj").read_text(encoding="utf-8").count("\nl "), 0)

	def test_optimizer_atlas_debug_objs_initial_eval_and_one_step(self) -> None:
		class TinyAtlasModel(torch.nn.Module):
			def __init__(self) -> None:
				super().__init__()
				self.xyz = torch.nn.Parameter(_plane_grid().clone())
				self.mesh_h = int(self.xyz.shape[1])
				self.mesh_w = int(self.xyz.shape[2])
				self.depth = int(self.xyz.shape[0])

			def opt_params(self):
				return {"mesh_ms": [self.xyz]}

			def update_conn_offsets(self) -> None:
				return None

			def update_ext_conn_offsets(self) -> None:
				return None

			def forward(self, data: fit_data.FitData3D, needs=None):
				return _result(self.xyz, data.atlas_lines, normals=_normal_grid(self.xyz))

		opt_loss_atlas_line.reset_state()
		lines = fit_data.AtlasLines3D(
			target_xyz=torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
			normal_xyz=torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32),
			model_h=torch.tensor([1.0], dtype=torch.float32),
			model_w=torch.tensor([1.0], dtype=torch.float32),
			is_control_point=torch.tensor([True]),
		)
		stages = optimizer.load_stages_cfg({
			"base": {"normal": 0.0, "atlas_line_control": 1.0, "atlas_line_other": 0.0},
			"stages": [{
				"name": "atlas_init_relax",
				"opt": {
					"steps": 1,
					"lr": 0.01,
					"params": ["mesh_ms"],
					"args": {"atlas_debug_objs": True, "atlas_debug_obj_interval": 1},
					"w_fac": {"atlas_line_control": 1.0},
				},
			}],
		})
		data = _fit_data(lines)
		model = TinyAtlasModel()

		old_cwd = Path.cwd()
		with tempfile.TemporaryDirectory() as td:
			try:
				os.chdir(td)
				optimizer.optimize(
					model=model,
					data=data,
					stages=stages,
					snapshot_interval=0,
					snapshot_fn=lambda **_kw: None,
				)
			finally:
				os.chdir(old_cwd)
			for step in (0, 1):
				wdir = Path(td) / "atlas_debug_objs" / f"atlas_init_relax_step{step:06d}" / "winding_000"
				self.assertTrue((wdir / "model_surface.obj").exists())
				self.assertIn("\nf ", (wdir / "model_surface.obj").read_text(encoding="utf-8"))
				self.assertEqual((wdir / "control_connections.obj").read_text(encoding="utf-8").count("\nl "), 1)
				self.assertEqual((wdir / "normal_proxy.obj").read_text(encoding="utf-8").count("\nl "), 12)

	def test_atlas_line_updates_at_most_one_neighboring_quad_and_clamps(self) -> None:
		opt_loss_atlas_line.reset_state()
		xyz = _plane_grid().requires_grad_(True)
		lines = fit_data.AtlasLines3D(
			target_xyz=torch.tensor([[2.2, 1.2, 1.0]], dtype=torch.float32),
			normal_xyz=torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32),
			model_h=torch.tensor([1.2], dtype=torch.float32),
			model_w=torch.tensor([1.2], dtype=torch.float32),
			object_ids=("fiber",),
			source_indices=(0,),
		)

		result = opt_loss_atlas_line.atlas_line_loss(res=_result(xyz, lines))
		loss, maps, masks = result["atlas_line"]
		loss.backward()

		self.assertAlmostEqual(float(loss.detach()), 1.0, delta=1.0e-5)
		self.assertEqual(tuple(maps[0].shape), (1, 1, 3, 4))
		self.assertAlmostEqual(float(masks[0].sum()), 1.0, delta=1.0e-5)
		self.assertEqual(int(opt_loss_atlas_line._cols[0, 0]), 2)
		self.assertAlmostEqual(float(opt_loss_atlas_line._frac_w[0, 0]), 0.2, delta=1.0e-5)
		self.assertTrue(torch.isfinite(xyz.grad).all().item())

	def test_atlas_line_skips_invalid_samples(self) -> None:
		opt_loss_atlas_line.reset_state()
		xyz = _plane_grid().requires_grad_(True)
		lines = fit_data.AtlasLines3D(
			target_xyz=torch.tensor([[float("nan"), 1.0, 1.0]], dtype=torch.float32),
			normal_xyz=torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32),
			model_h=torch.tensor([1.0], dtype=torch.float32),
			model_w=torch.tensor([1.0], dtype=torch.float32),
		)

		result = opt_loss_atlas_line.atlas_line_loss(res=_result(xyz, lines))
		loss, _maps, masks = result["atlas_line"]

		self.assertEqual(float(loss.detach()), 0.0)
		self.assertEqual(float(masks[0].sum()), 0.0)

	def test_atlas_normals_are_ignored(self) -> None:
		xyz = _plane_grid().requires_grad_(True)
		base_kwargs = {
			"target_xyz": torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
			"model_h": torch.tensor([1.0], dtype=torch.float32),
			"model_w": torch.tensor([1.0], dtype=torch.float32),
		}
		lines_a = fit_data.AtlasLines3D(
			normal_xyz=torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32),
			**base_kwargs,
		)
		lines_b = fit_data.AtlasLines3D(
			normal_xyz=torch.tensor([[float("nan"), float("nan"), float("nan")]], dtype=torch.float32),
			**base_kwargs,
		)

		opt_loss_atlas_line.reset_state()
		loss_a = opt_loss_atlas_line.atlas_line_loss(res=_result(xyz, lines_a))["atlas_line"][0]
		opt_loss_atlas_line.reset_state()
		loss_b = opt_loss_atlas_line.atlas_line_loss(res=_result(xyz, lines_b))["atlas_line"][0]

		self.assertAlmostEqual(float(loss_a.detach()), float(loss_b.detach()), delta=1.0e-6)
		self.assertTrue(torch.isfinite(loss_b).item())

	def test_signed_correction_scalar_follows_model_normal_sign(self) -> None:
		xyz = _plane_grid().requires_grad_(True)
		lines = fit_data.AtlasLines3D(
			target_xyz=torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
			normal_xyz=torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32),
			model_h=torch.tensor([1.0], dtype=torch.float32),
			model_w=torch.tensor([1.0], dtype=torch.float32),
		)

		opt_loss_atlas_line.reset_state()
		opt_loss_atlas_line.atlas_line_loss(res=_result(xyz, lines, normals=_normal_grid(xyz, sign=1.0)))
		pos = opt_loss_atlas_line.last_stats()["atlas_line_signed_delta_mean"]
		opt_loss_atlas_line.reset_state()
		opt_loss_atlas_line.atlas_line_loss(res=_result(xyz, lines, normals=_normal_grid(xyz, sign=-1.0)))
		neg = opt_loss_atlas_line.last_stats()["atlas_line_signed_delta_mean"]

		self.assertAlmostEqual(pos, 1.0, delta=1.0e-5)
		self.assertAlmostEqual(neg, -1.0, delta=1.0e-5)

	def test_tangential_target_offset_has_no_normal_displacement_loss(self) -> None:
		opt_loss_atlas_line.reset_state()
		xyz = _plane_grid().requires_grad_(True)
		lines = fit_data.AtlasLines3D(
			target_xyz=torch.tensor([[2.0, 1.0, 0.0]], dtype=torch.float32),
			normal_xyz=torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32),
			model_h=torch.tensor([1.0], dtype=torch.float32),
			model_w=torch.tensor([1.0], dtype=torch.float32),
		)

		loss = opt_loss_atlas_line.atlas_line_loss(res=_result(xyz, lines))["atlas_line"][0]

		self.assertEqual(float(loss.detach()), 0.0)

	def test_gaussian_splat_affects_neighboring_vertices(self) -> None:
		opt_loss_atlas_line.reset_state()
		xyz = _plane_grid().requires_grad_(True)
		lines = fit_data.AtlasLines3D(
			target_xyz=torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
			normal_xyz=torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32),
			model_h=torch.tensor([1.0], dtype=torch.float32),
			model_w=torch.tensor([1.0], dtype=torch.float32),
		)

		loss, _maps, masks = opt_loss_atlas_line.atlas_line_loss(res=_result(xyz, lines))["atlas_line"]
		loss.backward()

		self.assertGreater(int((masks[0] > 0.0).sum()), 4)
		self.assertGreater(int((xyz.grad[..., 2].abs() > 0.0).sum()), 4)

	def test_atlas_line_exposes_control_and_other_components_from_one_batch(self) -> None:
		opt_loss_atlas_line.reset_state()
		xyz = _plane_grid().requires_grad_(True)
		lines = fit_data.AtlasLines3D(
			target_xyz=torch.tensor([
				[1.0, 1.0, 1.0],
				[2.0, 1.0, 2.0],
			], dtype=torch.float32),
			normal_xyz=torch.tensor([
				[0.0, 0.0, -1.0],
				[0.0, 0.0, -1.0],
			], dtype=torch.float32),
			model_h=torch.tensor([1.0, 1.0], dtype=torch.float32),
			model_w=torch.tensor([1.0, 2.0], dtype=torch.float32),
			is_control_point=torch.tensor([True, False]),
		)

		result = opt_loss_atlas_line.atlas_line_loss(res=_result(xyz, lines))
		loss_all = result["atlas_line"][0]
		loss_control = result["atlas_line_control"][0]
		loss_other = result["atlas_line_other"][0]

		self.assertGreater(float(loss_all.detach()), 0.0)
		self.assertAlmostEqual(float(loss_control.detach()), 1.0, delta=1.0e-5)
		self.assertAlmostEqual(float(loss_other.detach()), 4.0, delta=1.0e-5)
		stats = opt_loss_atlas_line.last_stats()
		self.assertEqual(stats["atlas_line_valid"], 2.0)
		self.assertEqual(stats["atlas_line_control_valid"], 1.0)
		self.assertEqual(stats["atlas_line_other_valid"], 1.0)
		self.assertAlmostEqual(float(result["atlas_line_control"][2][0].sum()), 1.0, delta=1.0e-5)
		self.assertAlmostEqual(float(result["atlas_line_other"][2][0].sum()), 1.0, delta=1.0e-5)

	def test_atlas_line_stage_weights_process_control_only(self) -> None:
		opt_loss_atlas_line.reset_state()
		xyz = _plane_grid().requires_grad_(True)
		lines = fit_data.AtlasLines3D(
			target_xyz=torch.tensor([[1.0, 1.0, 1.0], [2.0, 1.0, 2.0]], dtype=torch.float32),
			normal_xyz=torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]], dtype=torch.float32),
			model_h=torch.tensor([1.0, 1.0], dtype=torch.float32),
			model_w=torch.tensor([1.0, 2.0], dtype=torch.float32),
			is_control_point=torch.tensor([True, False]),
		)

		result = opt_loss_atlas_line.atlas_line_loss(
			res=_result(xyz, lines),
			stage_eff={"atlas_line": 0.0, "atlas_line_control": 1.0, "atlas_line_other": 0.0},
		)

		self.assertAlmostEqual(float(result["atlas_line"][2][0].sum()), 1.0, delta=1.0e-5)
		self.assertAlmostEqual(float(result["atlas_line_control"][2][0].sum()), 1.0, delta=1.0e-5)
		self.assertEqual(float(result["atlas_line_other"][2][0].sum()), 0.0)
		self.assertEqual(opt_loss_atlas_line.last_stats()["atlas_line_other_valid"], 0.0)

	def test_atlas_line_stage_weights_process_other_only(self) -> None:
		opt_loss_atlas_line.reset_state()
		xyz = _plane_grid().requires_grad_(True)
		lines = fit_data.AtlasLines3D(
			target_xyz=torch.tensor([[1.0, 1.0, 1.0], [2.0, 1.0, 2.0]], dtype=torch.float32),
			normal_xyz=torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]], dtype=torch.float32),
			model_h=torch.tensor([1.0, 1.0], dtype=torch.float32),
			model_w=torch.tensor([1.0, 2.0], dtype=torch.float32),
			is_control_point=torch.tensor([True, False]),
		)

		result = opt_loss_atlas_line.atlas_line_loss(
			res=_result(xyz, lines),
			stage_eff={"atlas_line": 0.0, "atlas_line_control": 0.0, "atlas_line_other": 1.0},
		)

		self.assertAlmostEqual(float(result["atlas_line"][2][0].sum()), 1.0, delta=1.0e-5)
		self.assertEqual(float(result["atlas_line_control"][2][0].sum()), 0.0)
		self.assertAlmostEqual(float(result["atlas_line_other"][2][0].sum()), 1.0, delta=1.0e-5)
		self.assertEqual(opt_loss_atlas_line.last_stats()["atlas_line_control_valid"], 0.0)

	def test_atlas_line_legacy_weight_processes_both_groups(self) -> None:
		opt_loss_atlas_line.reset_state()
		xyz = _plane_grid().requires_grad_(True)
		lines = fit_data.AtlasLines3D(
			target_xyz=torch.tensor([[1.0, 1.0, 1.0], [2.0, 1.0, 2.0]], dtype=torch.float32),
			normal_xyz=torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]], dtype=torch.float32),
			model_h=torch.tensor([1.0, 1.0], dtype=torch.float32),
			model_w=torch.tensor([1.0, 2.0], dtype=torch.float32),
			is_control_point=torch.tensor([True, False]),
		)

		result = opt_loss_atlas_line.atlas_line_loss(
			res=_result(xyz, lines),
			stage_eff={"atlas_line": 1.0, "atlas_line_control": 0.0, "atlas_line_other": 0.0},
		)

		self.assertGreater(float(result["atlas_line"][2][0].sum()), 1.0)
		self.assertAlmostEqual(float(result["atlas_line_control"][2][0].sum()), 1.0, delta=1.0e-5)
		self.assertAlmostEqual(float(result["atlas_line_other"][2][0].sum()), 1.0, delta=1.0e-5)
		self.assertEqual(opt_loss_atlas_line.last_stats()["atlas_line_valid"], 2.0)

	def test_atlas_line_all_zero_weights_skip_intersections(self) -> None:
		opt_loss_atlas_line.reset_state()
		xyz = _plane_grid().requires_grad_(True)
		lines = fit_data.AtlasLines3D(
			target_xyz=torch.tensor([[1.0, 1.0, 1.0], [2.0, 1.0, 2.0]], dtype=torch.float32),
			normal_xyz=torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]], dtype=torch.float32),
			model_h=torch.tensor([1.0, 1.0], dtype=torch.float32),
			model_w=torch.tensor([1.0, 2.0], dtype=torch.float32),
			is_control_point=torch.tensor([True, False]),
		)
		original = opt_loss_atlas_line._intersect_quad
		calls = 0

		def counted(*args, **kwargs):
			nonlocal calls
			calls += 1
			return original(*args, **kwargs)

		opt_loss_atlas_line._intersect_quad = counted
		try:
			result = opt_loss_atlas_line.atlas_line_loss(
				res=_result(xyz, lines),
				stage_eff={"atlas_line": 0.0, "atlas_line_control": 0.0, "atlas_line_other": 0.0},
			)
		finally:
			opt_loss_atlas_line._intersect_quad = original

		self.assertEqual(calls, 0)
		self.assertEqual(float(result["atlas_line"][0].detach()), 0.0)
		result["atlas_line"][0].backward()
		self.assertTrue(torch.equal(xyz.grad, torch.zeros_like(xyz)))
		self.assertEqual(float(result["atlas_line"][2][0].sum()), 0.0)
		self.assertEqual(tuple(result["atlas_line"][2][0].shape), (1, 1, 3, 4))


if __name__ == "__main__":
	unittest.main()
