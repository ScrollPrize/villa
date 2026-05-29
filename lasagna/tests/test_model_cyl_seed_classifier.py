from __future__ import annotations

import math
import os
import sys
import unittest
from types import SimpleNamespace

import torch


ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
	sys.path.insert(0, ROOT)

import model as fit_model
import opt_loss_station


def _square_shell(*, half_by_row: list[float], z_by_row: list[float]) -> torch.Tensor:
	rows = []
	for half, z in zip(half_by_row, z_by_row, strict=True):
		rows.append(torch.tensor(
			[
				[-half, -half, z],
				[half, -half, z],
				[half, half, z],
				[-half, half, z],
			],
			dtype=torch.float32,
		))
	return torch.stack(rows, dim=0)


def _cylinder_shell(*, radius: float, z_by_row: torch.Tensor, width: int) -> torch.Tensor:
	angles = torch.arange(width, dtype=torch.float32) * (2.0 * math.pi / float(width))
	x = radius * torch.cos(angles)
	y = radius * torch.sin(angles)
	rows = []
	for z in z_by_row:
		rows.append(torch.stack([x, y, torch.full_like(x, float(z))], dim=-1))
	return torch.stack(rows, dim=0)


def _folded_seed_column_cylinder_shell(*, radius: float, width: int = 64) -> torch.Tensor:
	shell = _cylinder_shell(
		radius=radius,
		z_by_row=torch.tensor([-100.0, -20.0, 20.0, 100.0]),
		width=width,
	)
	shell[1, 0, 2] = 100.0
	shell[2, 0, 2] = -100.0
	return shell


class _StationData:
	sparse_caches: dict = {}

	def grid_sample_fullres(self, query, *, channels, diff=False):
		shape = query.shape[:-1]
		nx = torch.ones(shape, device=query.device, dtype=query.dtype) if "nx" in channels else None
		ny = torch.zeros(shape, device=query.device, dtype=query.dtype) if "ny" in channels else None
		return SimpleNamespace(nx=nx, ny=ny)

	def has_channel(self, name: str) -> bool:
		return False


class CylinderSeedShellClassifierTest(unittest.TestCase):
	def test_seed_inside_shell_row(self) -> None:
		shell = _square_shell(half_by_row=[1.0], z_by_row=[0.0])

		got = fit_model.Model3D.classify_seed_against_shell(shell, (0.0, 0.0, 0.0))
		metrics = fit_model.Model3D.measure_seed_against_shell(shell, (0.0, 0.0, 0.0))

		self.assertEqual(got, "inside")
		self.assertLess(metrics.signed_distance, 0.0)

	def test_seed_outside_shell_row(self) -> None:
		shell = _square_shell(half_by_row=[1.0], z_by_row=[0.0])

		got = fit_model.Model3D.classify_seed_against_shell(shell, (1.5, 0.0, 0.0))
		metrics = fit_model.Model3D.measure_seed_against_shell(shell, (1.5, 0.0, 0.0))

		self.assertEqual(got, "outside")
		self.assertGreater(metrics.signed_distance, 0.0)

	def test_seed_on_edge_counts_as_edge(self) -> None:
		shell = _square_shell(half_by_row=[1.0], z_by_row=[0.0])

		got = fit_model.Model3D.classify_seed_against_shell(shell, (1.0, 0.0, 0.0))
		metrics = fit_model.Model3D.measure_seed_against_shell(shell, (1.0, 0.0, 0.0))

		self.assertEqual(got, "edge")
		self.assertEqual(metrics.signed_distance, 0.0)

	def test_seed_near_edge_counts_as_edge(self) -> None:
		shell = _square_shell(half_by_row=[1.0], z_by_row=[0.0])

		got = fit_model.Model3D.classify_seed_against_shell(
			shell,
			(1.00005, 0.0, 0.0),
			edge_tolerance=1.0e-3,
		)

		self.assertEqual(got, "edge")

	def test_seed_classification_uses_seed_z_cross_section(self) -> None:
		shell = _square_shell(half_by_row=[1.0, 10.0], z_by_row=[0.0, 10.0])

		got = fit_model.Model3D.classify_seed_against_shell(shell, (3.0, 0.0, 5.0))
		metrics = fit_model.Model3D.measure_seed_against_shell(shell, (3.0, 0.0, 5.0))

		self.assertEqual(got, "inside")
		self.assertEqual(metrics.row_index, 0)

	def test_seed_classification_handles_multiple_z_crossings(self) -> None:
		shell = _square_shell(half_by_row=[1.0, 10.0, 1.0], z_by_row=[-1.0, 1.0, -1.0])

		got = fit_model.Model3D.classify_seed_against_shell(shell, (0.0, 0.0, 0.0))
		metrics = fit_model.Model3D.measure_seed_against_shell(shell, (0.0, 0.0, 0.0))

		self.assertEqual(got, "inside")
		self.assertLess(metrics.signed_distance, 0.0)

	def test_seed_bracketing_requires_openings_to_surround_seed_z(self) -> None:
		mdl = fit_model.Model3D(
			device=torch.device("cpu"),
			depth=1,
			mesh_h=2,
			mesh_w=3,
			mesh_step=50,
			winding_step=50,
			subsample_mesh=1,
			subsample_winding=1,
			init_mode="cylinder_seed",
			pyramid_d=False,
		)
		mdl.cyl_seed_xyz = torch.tensor([0.0, 0.0, 0.0])
		ok_shell = _square_shell(half_by_row=[1.0, 1.0], z_by_row=[-10.0, 10.0])
		bad_shell = _square_shell(half_by_row=[1.0, 1.0], z_by_row=[1.0, 10.0])

		mdl.assert_cylinder_shell_brackets_seed(ok_shell, label="test shell")
		with self.assertRaisesRegex(ValueError, "no longer brackets seed z"):
			mdl.assert_cylinder_shell_brackets_seed(bad_shell, label="test shell")


class CylinderSeedShellBakeTest(unittest.TestCase):
	def test_bake_extracts_seed_patch_at_regular_mesh_step(self) -> None:
		mdl = fit_model.Model3D(
			device=torch.device("cpu"),
			depth=1,
			mesh_h=2,
			mesh_w=3,
			mesh_step=50,
			winding_step=50,
			subsample_mesh=1,
			subsample_winding=1,
			init_mode="cylinder_seed",
			pyramid_d=False,
		)
		mdl.init_cylinder_seed(
			seed=(1000.0, 0.0, 0.0),
			model_w=200.0,
			model_h=200.0,
			volume_extent_fullres=None,
		)
		mdl.cyl_shell_completed = [
			_cylinder_shell(
				radius=1000.0,
				z_by_row=torch.linspace(-100.0, 100.0, 5),
				width=256,
			)
		]

		mdl.bake_cylinder_into_mesh(None)

		self.assertFalse(mdl.cylinder_enabled)
		self.assertFalse(mdl.cyl_shell_mode)
		self.assertEqual((mdl.depth, mdl.mesh_h, mdl.mesh_w), (1, 5, 5))
		xyz = mdl.mesh_coarse().detach().permute(1, 2, 3, 0)
		center = xyz[0, 2, 2]
		self.assertTrue(torch.allclose(center, torch.tensor([1000.0, 0.0, 0.0]), atol=1.0))
		h_step = (xyz[0, 3, 2] - xyz[0, 2, 2]).norm()
		w_step = (xyz[0, 2, 3] - xyz[0, 2, 2]).norm()
		self.assertAlmostEqual(float(h_step), 50.0, delta=1.0)
		self.assertAlmostEqual(float(w_step), 50.0, delta=1.0)

	def test_bake_seed_phase_handles_multiple_column_crossings(self) -> None:
		mdl = fit_model.Model3D(
			device=torch.device("cpu"),
			depth=1,
			mesh_h=2,
			mesh_w=3,
			mesh_step=50,
			winding_step=50,
			subsample_mesh=1,
			subsample_winding=1,
			init_mode="cylinder_seed",
			pyramid_d=False,
		)
		seed = torch.tensor([1000.0, 0.0, 0.0])
		mdl.init_cylinder_seed(
			seed=tuple(float(v) for v in seed),
			model_w=200.0,
			model_h=100.0,
			volume_extent_fullres=None,
		)
		shell = _folded_seed_column_cylinder_shell(radius=1000.0)

		phase = mdl._seed_phase_on_shell(shell)
		mdl.cyl_shell_completed = [shell]
		mdl.bake_cylinder_into_mesh(None)

		self.assertLess(min(abs(phase), abs(phase - 1.0)), 1.0e-4)
		xyz = mdl.mesh_coarse().detach().permute(1, 2, 3, 0)
		center = xyz[0, 1, 2]
		self.assertTrue(torch.allclose(center, seed, atol=1.0))

	def test_baked_seed_patch_starts_with_zero_station_loss(self) -> None:
		mdl = fit_model.Model3D(
			device=torch.device("cpu"),
			depth=1,
			mesh_h=2,
			mesh_w=3,
			mesh_step=50,
			winding_step=50,
			subsample_mesh=1,
			subsample_winding=1,
			init_mode="cylinder_seed",
			pyramid_d=False,
		)
		seed = torch.tensor([1000.0, 0.0, 0.0])
		mdl.init_cylinder_seed(
			seed=tuple(float(v) for v in seed),
			model_w=200.0,
			model_h=200.0,
			volume_extent_fullres=None,
		)
		mdl.cyl_shell_completed = [
			_cylinder_shell(
				radius=1000.0,
				z_by_row=torch.linspace(-100.0, 100.0, 5),
				width=256,
			)
		]
		mdl.bake_cylinder_into_mesh(None)
		data = _StationData()
		try:
			opt_loss_station.set_seed(seed, data, Hm=mdl.mesh_h, Wm=mdl.mesh_w, D=mdl.depth)
			res = mdl(data, needs=fit_model.ModelForwardNeeds())
			losses = opt_loss_station.station_loss(res=res)
			self.assertLess(float(losses["station_t"][0].detach()), 1.0e-6)
			self.assertLess(float(losses["station_n"][0].detach()), 1.0e-6)
		finally:
			opt_loss_station.reset()

	def test_baked_multiwrap_seed_patch_anchors_center_station_copy(self) -> None:
		mdl = fit_model.Model3D(
			device=torch.device("cpu"),
			depth=1,
			mesh_h=2,
			mesh_w=3,
			mesh_step=50,
			winding_step=50,
			subsample_mesh=1,
			subsample_winding=1,
			init_mode="cylinder_seed",
			pyramid_d=False,
		)
		seed = torch.tensor([1000.0, 0.0, 0.0])
		circumference = 2.0 * math.pi * 1000.0
		mdl.init_cylinder_seed(
			seed=tuple(float(v) for v in seed),
			model_w=5.0 * circumference,
			model_h=200.0,
			volume_extent_fullres=None,
		)
		mdl.cyl_shell_completed = [
			_cylinder_shell(
				radius=1000.0,
				z_by_row=torch.linspace(-100.0, 100.0, 5),
				width=4096,
			)
		]
		mdl.bake_cylinder_into_mesh(None)
		data = _StationData()
		try:
			opt_loss_station.set_seed(seed, data, Hm=mdl.mesh_h, Wm=mdl.mesh_w, D=mdl.depth)
			res = mdl(data, needs=fit_model.ModelForwardNeeds())
			losses = opt_loss_station.station_loss(res=res)
			self.assertLess(float(losses["station_t"][0].detach()), 1.0e-6)
			self.assertLess(float(losses["station_n"][0].detach()), 1.0e-6)
		finally:
			opt_loss_station.reset()


if __name__ == "__main__":
	unittest.main()
