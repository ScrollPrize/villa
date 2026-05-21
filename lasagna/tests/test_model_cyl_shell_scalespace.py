from __future__ import annotations

import os
import sys
import unittest

import torch


ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
	sys.path.insert(0, ROOT)

import fit
import model as fit_model
import opt_loss_cyl
import optimizer


class _UmbilicusData:
	sparse_caches: dict = {}

	def umbilicus_xy_at_z(self, z: torch.Tensor) -> torch.Tensor:
		return torch.zeros(*z.shape, 2, device=z.device, dtype=z.dtype)


class _CurvedUmbilicusData:
	sparse_caches: dict = {}

	def umbilicus_xy_at_z(self, z: torch.Tensor) -> torch.Tensor:
		return torch.stack([0.01 * z * z, torch.zeros_like(z)], dim=-1)


class CylinderShellScaleSpaceTest(unittest.TestCase):
	def _model(self) -> fit_model.Model3D:
		return fit_model.Model3D(
			device=torch.device("cpu"),
			depth=1,
			mesh_h=5,
			mesh_w=7,
			mesh_step=50,
			winding_step=50,
			subsample_mesh=1,
			subsample_winding=1,
			init_mode="cylinder_seed",
			pyramid_d=False,
		)

	def test_shell_delta_pyramid_reconstructs_and_gets_gradients(self) -> None:
		mdl = self._model()
		mdl.cyl_shell_mode = True
		y = torch.linspace(-1.0, 1.0, 5).view(5, 1)
		x = torch.linspace(-2.0, 2.0, 7).view(1, 7)
		delta = torch.stack(
			[
				10.0 + 3.0 * x.expand(5, 7) + y.expand(5, 7),
				-5.0 + x.expand(5, 7) - 2.0 * y.expand(5, 7),
				0.25 * x.expand(5, 7) + 0.5 * y.expand(5, 7),
			],
			dim=-1,
		)

		mdl._set_shell_delta_xyz_params(delta)
		got = mdl._shell_delta_xyz_params()

		self.assertEqual(len(mdl.cyl_shell_delta_ms), mdl.cyl_shell_n_scales)
		self.assertTrue(torch.allclose(got, delta, atol=1.0e-5, rtol=1.0e-5))

		loss = got.square().mean()
		loss.backward()
		self.assertTrue(all(p.grad is not None for p in mdl.cyl_shell_delta_ms))

	def test_shell_cyl_params_expose_scale_levels(self) -> None:
		mdl = self._model()
		mdl.cyl_shell_mode = True
		mdl._set_shell_delta_xyz_params(torch.zeros(5, 7, 3))

		params = mdl.opt_params()["cyl_params"]
		self.assertEqual(len(params), mdl.cyl_param_scale_count())
		self.assertIs(params[0], mdl.cyl_shell_delta_ms[0])
		self.assertEqual(
			[optimizer._lr_scalespace(lr=[10.0, 1.0, 0.1, 0.01, 0.001], scale_i=i) for i in range(5)],
			[0.001, 0.01, 0.1, 1.0, 10.0],
		)

	def test_cylinder_seed_shell_params_are_xyz(self) -> None:
		mdl = self._model()
		self.assertEqual(mdl._first_shell_radius(), 2000.0)
		mdl.init_cylinder_seed(
			seed=(0.0, 0.0, 10.0),
			model_w=100.0,
			model_h=100.0,
			volume_extent_fullres=None,
		)

		self.assertEqual(tuple(mdl.cyl_params.shape), (5, 7, 3))

		mdl.prepare_umbilicus_tube_init(_UmbilicusData())
		delta = mdl._shell_delta_xyz_params()

		self.assertEqual(delta.shape[-1], 3)
		self.assertEqual(tuple(mdl.cyl_params.shape), tuple(delta.shape))
		self.assertTrue(all(int(p.shape[0]) == 3 for p in mdl.cyl_shell_delta_ms))
		self.assertTrue(torch.allclose(delta[..., 2], torch.zeros_like(delta[..., 2])))

	def test_cylinder_seed_init_height_quadruples_and_clamps_to_volume(self) -> None:
		mdl = self._model()
		mdl.cyl_shell_z_step = 25.0
		mdl.init_cylinder_seed(
			seed=(0.0, 0.0, 50.0),
			model_w=100.0,
			model_h=100.0,
			volume_extent_fullres=(200, 200, 120),
		)
		mdl.prepare_umbilicus_tube_init(_UmbilicusData())
		shell = mdl.current_cylinder_shell_xyz().detach()

		self.assertAlmostEqual(float(shell[0, :, 2].mean()), 0.0, delta=1.0e-5)
		self.assertAlmostEqual(float(shell[-1, :, 2].mean()), 119.0, delta=1.0e-5)
		self.assertAlmostEqual(float(mdl.cyl_shell_z_center_target), 59.5, delta=1.0e-5)
		self.assertAlmostEqual(float(mdl.cyl_shell_model_h), 119.0, delta=1.0e-5)

	def test_cylinder_seed_init_uses_umbilicus_xy_centers_per_z(self) -> None:
		mdl = self._model()
		mdl.cyl_shell_z_step = 25.0
		mdl.cyl_shell_width_target_step = 25.0
		mdl.cyl_shell_current_width_step = 25.0
		mdl.init_cylinder_seed(
			seed=(0.0, 0.0, 0.0),
			model_w=100.0,
			model_h=100.0,
			volume_extent_fullres=None,
		)
		mdl.prepare_umbilicus_tube_init(_CurvedUmbilicusData())
		shell = mdl.current_cylinder_shell_xyz().detach()
		base_center = mdl.cyl_shell_base[..., :2].mean(dim=1)
		mid_i = int(base_center.shape[0]) // 2
		h_len = (shell[1:] - shell[:-1]).norm(dim=-1)
		res = mdl(_CurvedUmbilicusData(), needs=fit_model.ModelForwardNeeds(cyl_samples=True))
		step_loss, _, _ = opt_loss_cyl.cyl_step_loss(res=res)

		self.assertAlmostEqual(float(base_center[0, 0]), 400.0, delta=1.0e-4)
		self.assertAlmostEqual(float(base_center[mid_i, 0]), 0.0, delta=1.0e-4)
		self.assertAlmostEqual(float(_CurvedUmbilicusData().umbilicus_xy_at_z(torch.tensor([0.0]))[0, 0]), 0.0)
		self.assertTrue(torch.allclose(
			h_len,
			torch.full_like(h_len, float(mdl.cyl_shell_z_step)),
			atol=1.0,
			rtol=0.0,
		))
		self.assertLess(float(step_loss.detach()), 0.1)

	def test_fit_result_height_step_uses_model_step_not_measured_spacing(self) -> None:
		mdl = self._model()
		mdl.cyl_shell_z_step = 25.0
		mdl.init_cylinder_seed(
			seed=(0.0, 0.0, 0.0),
			model_w=100.0,
			model_h=100.0,
			volume_extent_fullres=None,
		)
		mdl.prepare_umbilicus_tube_init(_CurvedUmbilicusData())
		mdl.cyl_shell_current_height_step = 123.0

		res = mdl(_CurvedUmbilicusData(), needs=fit_model.ModelForwardNeeds(cyl_samples=True))

		self.assertAlmostEqual(float(res.cyl_shell_height_step), 25.0, delta=1.0e-6)

	def test_cylinder_init_uses_first_stage_model_step_before_prepare(self) -> None:
		mdl = self._model()
		stages = optimizer.load_stages_cfg({
			"args": {"init-mode": "cylinder_seed"},
			"base": {"cyl_normal": 1.0},
			"stages": [
				{
					"name": "cyl_init",
					"steps": 0,
					"lr": 1.0,
					"params": ["cyl_params"],
					"args": {"model-step": 200.0},
				},
			],
		})
		mdl.init_cylinder_seed(
			seed=(0.0, 0.0, 0.0),
			model_w=8000.0,
			model_h=1000.0,
			volume_extent_fullres=None,
		)

		fit._apply_cylinder_prepare_model_step(mdl, fit._first_cylinder_stage_model_step(stages))
		mdl.prepare_umbilicus_tube_init(_UmbilicusData())

		w_avg = mdl._shell_width_step_stats()[0]
		self.assertAlmostEqual(float(mdl.cyl_shell_width_target_step), 200.0, delta=1.0e-5)
		self.assertAlmostEqual(float(mdl.cyl_shell_z_step), 200.0, delta=1.0e-5)
		self.assertAlmostEqual(float(w_avg), 200.0, delta=2.0)
		self.assertAlmostEqual(float(mdl.cyl_shell_target_radius), 2000.0, delta=1.0e-5)
		self.assertAlmostEqual(float(mdl.cyl_shell_current_radius), 2000.0, delta=1.0e-5)

	def test_nonzero_z_residual_changes_current_shell_z(self) -> None:
		mdl = self._model()
		mdl.init_cylinder_seed(
			seed=(0.0, 0.0, 10.0),
			model_w=100.0,
			model_h=100.0,
			volume_extent_fullres=None,
		)
		mdl.prepare_umbilicus_tube_init(_UmbilicusData())
		z_before = mdl.current_cylinder_shell_xyz()[..., 2].detach()
		delta = mdl._shell_delta_xyz_params().detach()
		z_residual = torch.linspace(-2.0, 2.0, delta.numel() // 3).reshape(delta.shape[:2])
		delta[..., 2] = z_residual

		mdl._set_shell_delta_xyz_params(delta)
		z_after = mdl.current_cylinder_shell_xyz()[..., 2].detach()

		self.assertTrue(torch.allclose(z_after, z_before + z_residual, atol=1.0e-5, rtol=1.0e-5))

	def test_width_resampling_preserves_z_variation(self) -> None:
		data = _UmbilicusData()
		mdl = self._model()
		mdl.init_cylinder_seed(
			seed=(0.0, 0.0, 10.0),
			model_w=100.0,
			model_h=100.0,
			volume_extent_fullres=None,
		)
		mdl.prepare_umbilicus_tube_init(data)
		delta = mdl._shell_delta_xyz_params().detach()
		z_residual = torch.linspace(-4.0, 4.0, int(delta.shape[1])).view(1, -1).expand(delta.shape[:2])
		delta[..., 2] = z_residual
		mdl._set_shell_delta_xyz_params(delta)
		before = mdl.current_cylinder_shell_xyz().detach()
		old_w = int(before.shape[1])
		target_w = max(old_w + 1, int(torch.ceil(torch.tensor(float(old_w) * mdl.cyl_shell_growth_factor)).item()))
		expected = mdl._resample_shell_width(before, target_w)

		mdl.resample_current_cylinder_shell_width_for_growth(data)
		got = mdl.current_cylinder_shell_xyz().detach()

		self.assertEqual(int(got.shape[1]), target_w)
		self.assertTrue(torch.allclose(got, expected, atol=1.0e-5, rtol=1.0e-5))
		self.assertGreater(float(got[..., 2].std()), 0.0)


if __name__ == "__main__":
	unittest.main()
