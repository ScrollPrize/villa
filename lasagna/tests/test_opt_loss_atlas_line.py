from __future__ import annotations

import os
import sys
import unittest

import torch


ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
	sys.path.insert(0, ROOT)

import fit_data
import model as fit_model
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
