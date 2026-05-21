from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import model as fit_model
import opt_loss_cyl


def _make_shell(h: int = 4, w: int = 5, *, requires_grad: bool = True) -> torch.Tensor:
	z = torch.linspace(-2.0, 2.0, h).view(h, 1).expand(h, w)
	theta = torch.linspace(0.0, 2.0 * torch.pi * (1.0 - 1.0 / w), w).view(1, w).expand(h, w)
	radius = 10.0 + 0.2 * z
	xyz = torch.stack([radius * torch.cos(theta), radius * torch.sin(theta), z], dim=-1).contiguous()
	if requires_grad:
		xyz.requires_grad_(True)
	return xyz


def _make_planar_grid(h: int = 5, w: int = 6) -> torch.Tensor:
	yy = torch.arange(h, dtype=torch.float32).view(h, 1).expand(h, w)
	xx = torch.arange(w, dtype=torch.float32).view(1, w).expand(h, w)
	zz = torch.zeros_like(xx)
	return torch.stack([xx, yy, zz], dim=-1).contiguous()


def _sampled_target(xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
	raw = torch.stack(
		[
			0.3 * xyz[..., 0] + 1.0,
			-0.2 * xyz[..., 1] + 0.5,
			0.1 * xyz[..., 2] + 0.25,
		],
		dim=-1,
	)
	target = F.normalize(raw, dim=-1, eps=1.0e-8).detach()
	mask = ((xyz[..., 0].sin() + xyz[..., 1].cos()) > -2.0).to(dtype=xyz.dtype).detach()
	return target, mask


def _assert_close(test: unittest.TestCase, actual: torch.Tensor, expected: torch.Tensor, name: str) -> None:
	if torch.allclose(actual, expected, atol=1.0e-6, rtol=1.0e-6):
		return
	max_diff = float((actual.detach() - expected.detach()).abs().max())
	test.fail(f"{name} mismatch: max={max_diff}")


class _FakeData:
	def grid_sample_fullres(self, xyz_fullres: torch.Tensor, *, channels=None, diff: bool = False):
		target, mask = _sampled_target(xyz_fullres.detach()[0])
		return SimpleNamespace(
			normal_3d=target.unsqueeze(0),
			grad_mag=mask.view(1, 1, *mask.shape),
		)

	def umbilicus_xy_at_z(self, z: torch.Tensor) -> torch.Tensor:
		return torch.zeros(*z.shape, 2, device=z.device, dtype=z.dtype)


class _ConstantNormalData:
	def __init__(self, normal: tuple[float, float, float]) -> None:
		self.normal = normal

	def grid_sample_fullres(self, xyz_fullres: torch.Tensor, *, channels=None, diff: bool = False):
		xyz = xyz_fullres.detach()[0]
		normal = torch.tensor(self.normal, device=xyz.device, dtype=xyz.dtype)
		normal = F.normalize(normal, dim=0, eps=1.0e-8)
		target = normal.view(1, 1, 3).expand(*xyz.shape[:2], 3)
		mask = torch.ones(*xyz.shape[:2], device=xyz.device, dtype=xyz.dtype)
		return SimpleNamespace(
			normal_3d=target.unsqueeze(0),
			grad_mag=mask.view(1, 1, *mask.shape),
		)

	def umbilicus_xy_at_z(self, z: torch.Tensor) -> torch.Tensor:
		return torch.zeros(*z.shape, 2, device=z.device, dtype=z.dtype)


class _RadialNormalData:
	def __init__(self, sign: float = 1.0) -> None:
		self.sign = float(sign)

	def grid_sample_fullres(self, xyz_fullres: torch.Tensor, *, channels=None, diff: bool = False):
		xyz = xyz_fullres.detach()[0]
		xy = xyz[..., :2]
		normal_xy = float(self.sign) * F.normalize(xy, dim=-1, eps=1.0e-8)
		target = torch.cat([normal_xy, torch.zeros(*xy.shape[:-1], 1, device=xyz.device, dtype=xyz.dtype)], dim=-1)
		mask = torch.ones(*xyz.shape[:2], device=xyz.device, dtype=xyz.dtype)
		return SimpleNamespace(
			normal_3d=target.unsqueeze(0),
			grad_mag=mask.view(1, 1, *mask.shape),
		)

	def umbilicus_xy_at_z(self, z: torch.Tensor) -> torch.Tensor:
		return torch.zeros(*z.shape, 2, device=z.device, dtype=z.dtype)


class _RecordingConstantNormalData(_ConstantNormalData):
	def __init__(self, normal: tuple[float, float, float]) -> None:
		super().__init__(normal)
		self.sample_shapes: list[tuple[int, ...]] = []

	def grid_sample_fullres(self, xyz_fullres: torch.Tensor, *, channels=None, diff: bool = False):
		self.sample_shapes.append(tuple(int(v) for v in xyz_fullres.shape))
		return super().grid_sample_fullres(xyz_fullres, channels=channels, diff=diff)


class _FitDataShapeNormalData(_ConstantNormalData):
	def grid_sample_fullres(self, xyz_fullres: torch.Tensor, *, channels=None, diff: bool = False):
		xyz = xyz_fullres.detach()[0]
		normal = torch.tensor(self.normal, device=xyz.device, dtype=xyz.dtype)
		normal = F.normalize(normal, dim=0, eps=1.0e-8)
		target = normal.view(1, 1, 1, 3).expand(1, *xyz.shape[:2], 3)
		mask = torch.ones(1, *xyz.shape[:2], device=xyz.device, dtype=xyz.dtype)
		return SimpleNamespace(
			normal_3d=target,
			grad_mag=mask.view(1, 1, *mask.shape),
		)


class _MaskOnlyData:
	def __init__(self, mask: torch.Tensor) -> None:
		self.mask = mask.detach().clone()

	def grid_sample_fullres(self, xyz_fullres: torch.Tensor, *, channels=None, diff: bool = False):
		xyz = xyz_fullres.detach()[0]
		mask = self.mask.to(device=xyz.device, dtype=xyz.dtype)
		if tuple(mask.shape) != tuple(xyz.shape[:2]):
			mask = mask.expand(*xyz.shape[:2])
		normal = torch.zeros(*xyz.shape[:2], 3, device=xyz.device, dtype=xyz.dtype)
		normal[..., 2] = 1.0
		return SimpleNamespace(
			normal_3d=None if channels == {"grad_mag"} else normal.unsqueeze(0),
			grad_mag=mask.view(1, 1, *mask.shape),
		)

	def umbilicus_xy_at_z(self, z: torch.Tensor) -> torch.Tensor:
		return torch.zeros(*z.shape, 2, device=z.device, dtype=z.dtype)


def _make_shell_result(
	xyz: torch.Tensor,
	*,
	data=None,
	cyl_seed_z: float = 0.0,
	cyl_seed_signed_distance: float | None = None,
	cyl_z_center_target: float = 0.0,
	cyl_shell_width_step: float = 0.0,
	cyl_shell_height_step: float = 1.0,
	cyl_shell_step: float = 500.0,
	mesh_step: int = 1,
	cyl_shell_index: int = 0,
	cyl_outside_volume: torch.Tensor | None = None,
	cyl_outside_origin: tuple[float, float, float] | None = None,
	cyl_outside_spacing: tuple[float, float, float] | None = None,
	cyl_outside_shape: tuple[int, int, int] | None = None,
	cyl_outside_depth_max: float = 0.0,
	cyl_outside_sample_factor: int = 2,
	cyl_outside_model_step: float | None = None,
) -> fit_model.FitResult3D:
	return fit_model.FitResult3D(
		xyz_lr=xyz,
		xyz_hr=None,
		data=_FakeData() if data is None else data,
		data_s=None,
		data_lr=None,
		target_plain=None,
		target_mod=None,
		amp_lr=torch.zeros(1, 1, 1, 1),
		bias_lr=torch.zeros(1, 1, 1, 1),
		mask_hr=None,
		mask_lr=None,
		normals=None,
		xy_conn=None,
		mask_conn=None,
			sign_conn=None,
			params=fit_model.ModelParams3D(
				mesh_step=mesh_step,
				winding_step=1,
				subsample_mesh=1,
			subsample_winding=1,
			scaledown=1.0,
			z_step_eff=1,
			volume_extent=None,
			pyramid_d=False,
		),
		gt_normal_lr=None,
		ext_conn=None,
		cyl_xyz=xyz.unsqueeze(0),
		cyl_normals=None,
		cyl_centers=None,
		cyl_axes=None,
		cyl_params=None,
		cyl_count=1,
		cyl_shell_mode=True,
		cyl_shell_step=cyl_shell_step,
		cyl_shell_width_step=cyl_shell_width_step,
		cyl_seed_z=cyl_seed_z,
		cyl_seed_signed_distance=cyl_seed_signed_distance,
		cyl_z_center_target=cyl_z_center_target,
		cyl_shell_height_step=cyl_shell_height_step,
		cyl_shell_index=cyl_shell_index,
		cyl_outside_volume=cyl_outside_volume,
		cyl_outside_origin=cyl_outside_origin,
		cyl_outside_spacing=cyl_outside_spacing,
		cyl_outside_shape=cyl_outside_shape,
		cyl_outside_depth_max=cyl_outside_depth_max,
		cyl_outside_sample_factor=cyl_outside_sample_factor,
		cyl_outside_model_step=cyl_outside_model_step,
	)


class CylZCenterLossTests(unittest.TestCase):
	def tearDown(self) -> None:
		opt_loss_cyl.reset_candidate_terms()

	def test_z_center_loss_uses_configured_target(self) -> None:
		xyz = _make_shell(requires_grad=False) + torch.tensor([0.0, 0.0, 3.0])
		target = float(xyz[..., 2].mean())

		loss0, _, _ = opt_loss_cyl.cyl_z_center_loss(
			res=_make_shell_result(
				xyz,
				cyl_seed_z=0.0,
				cyl_z_center_target=target,
				cyl_shell_height_step=2.0,
			)
		)
		loss1, _, _ = opt_loss_cyl.cyl_z_center_loss(
			res=_make_shell_result(
				xyz,
				cyl_seed_z=0.0,
				cyl_z_center_target=target + 2.0,
				cyl_shell_height_step=2.0,
			)
		)

		self.assertAlmostEqual(float(loss0.detach()), 0.0, delta=1.0e-8)
		self.assertAlmostEqual(float(loss1.detach()), 1.0, delta=1.0e-6)

	def test_z_center_loss_gradient_follows_height_direction(self) -> None:
		xyz = _make_shell(requires_grad=True)
		target = float(xyz[..., 2].mean().detach()) + 2.0
		loss, _, _ = opt_loss_cyl.cyl_z_center_loss(
			res=_make_shell_result(
				xyz,
				cyl_z_center_target=target,
				cyl_shell_height_step=2.0,
			)
		)
		loss.backward()

		with torch.no_grad():
			h_vec = torch.empty_like(xyz)
			h_vec[0] = xyz[1] - xyz[0]
			h_vec[1:-1] = 0.5 * (xyz[2:] - xyz[:-2])
			h_vec[-1] = xyz[-1] - xyz[-2]
			h_dir = F.normalize(h_vec, dim=-1, eps=1.0e-8)
			update_dir = -xyz.grad

		self.assertGreater(float((update_dir * h_dir).sum(dim=-1).mean()), 0.0)
		self.assertGreater(float(update_dir[..., :2].abs().sum()), 0.0)


class CylStepPushLossTests(unittest.TestCase):
	def tearDown(self) -> None:
		opt_loss_cyl.reset_candidate_terms()

	def test_step_push_loss_is_zero_without_width_target(self) -> None:
		xyz = _make_shell(requires_grad=True)
		loss, _, _ = opt_loss_cyl.cyl_step_push_loss(
			res=_make_shell_result(
				xyz,
				data=_MaskOnlyData(torch.ones((int(xyz.shape[0]) - 1) * 4 + 1, int(xyz.shape[1]) * 4)),
				cyl_shell_width_step=0.0,
			)
		)

		self.assertAlmostEqual(float(loss.detach()), 0.0, delta=1.0e-8)

	def test_step_push_loss_uses_mesh_normals_for_width_growth(self) -> None:
		h = 4
		w = 8
		z = torch.linspace(-1.0, 1.0, h).view(h, 1).expand(h, w)
		theta = torch.linspace(0.0, 2.0 * torch.pi * (1.0 - 1.0 / w), w).view(1, w).expand(h, w)
		xyz = torch.stack([10.0 * torch.cos(theta), 10.0 * torch.sin(theta), z], dim=-1).contiguous()
		xyz.requires_grad_(True)
		radial = F.normalize(xyz.detach()[..., :2], dim=-1, eps=1.0e-8)
		shell_n = opt_loss_cyl._unit_normals_for_shell_xyz(xyz.detach())
		mask = torch.ones((int(xyz.shape[0]) - 1) * 4 + 1, int(xyz.shape[1]) * 4)
		geom = opt_loss_cyl._shell_geometry(res=_make_shell_result(xyz, data=_MaskOnlyData(mask)), factor=4)
		observed = opt_loss_cyl._valid_width_step_avg(geom["xyz"], mask, step_scale=4.0)

		loss, _, _ = opt_loss_cyl.cyl_step_push_loss(
			res=_make_shell_result(
				xyz,
				data=_MaskOnlyData(mask),
				cyl_shell_width_step=float(observed.detach()) + 4.0,
				mesh_step=2,
			)
		)
		loss.backward()

		update = -xyz.grad.detach()
		self.assertLess(float((shell_n[..., :2] * radial).sum(dim=-1).mean()), 0.0)
		self.assertGreater(float((update[..., :2] * radial).sum(dim=-1).mean()), 0.0)
		self.assertAlmostEqual(float(update[..., 2].abs().max()), 0.0, delta=1.0e-7)

	def test_step_push_loss_uses_valid_width_average_only(self) -> None:
		xyz = _make_shell(h=3, w=8, requires_grad=True)
		with torch.no_grad():
			xyz[:, 0, :2] *= 5.0
		mask = torch.zeros((int(xyz.shape[0]) - 1) * 4 + 1, int(xyz.shape[1]) * 4)
		mask[:, 12:16] = 1.0
		geom = opt_loss_cyl._shell_geometry(res=_make_shell_result(xyz, data=_MaskOnlyData(mask)), factor=4)
		observed = opt_loss_cyl._valid_width_step_avg(geom["xyz"], mask, step_scale=4.0)

		loss, _, _ = opt_loss_cyl.cyl_step_push_loss(
			res=_make_shell_result(
				xyz,
				data=_MaskOnlyData(mask),
				cyl_shell_width_step=float(observed.detach()),
				mesh_step=2,
			)
		)

		self.assertAlmostEqual(float(loss.detach()), 0.0, delta=1.0e-6)

	def test_step_push_loss_masks_invalid_samples(self) -> None:
		h = 3
		w = 4
		xyz = _make_shell(h=h, w=w, requires_grad=True)
		mask = torch.zeros((h - 1) * 4 + 1, w * 4)
		mask[:, 4:8] = 1.0
		geom = opt_loss_cyl._shell_geometry(res=_make_shell_result(xyz, data=_MaskOnlyData(mask)), factor=4)
		observed = opt_loss_cyl._valid_width_step_avg(geom["xyz"], mask, step_scale=4.0)

		loss, lms, masks = opt_loss_cyl.cyl_step_push_loss(
			res=_make_shell_result(
				xyz,
				data=_MaskOnlyData(mask),
				cyl_shell_width_step=float(observed.detach()) + 2.0,
				mesh_step=2,
			)
		)
		loss.backward()

		self.assertAlmostEqual(float(masks[0].sum()), float(mask.sum()), delta=1.0e-6)
		self.assertEqual(tuple(lms[0].shape[-2:]), tuple(mask.shape))
		self.assertGreater(float(xyz.grad.detach().abs().sum()), 0.0)


class CylStepMaskTests(unittest.TestCase):
	def tearDown(self) -> None:
		opt_loss_cyl.reset_candidate_terms()

	def test_width_edge_stats_report_valid_invalid_split(self) -> None:
		xyz = torch.tensor(
			[
				[
					[0.0, 0.0, 0.0],
					[2.0, 0.0, 0.0],
					[5.0, 0.0, 0.0],
					[9.0, 0.0, 0.0],
				],
			],
			requires_grad=True,
		)
		mask = torch.tensor([[1.0, 1.0, 0.0, 1.0]])

		stats = opt_loss_cyl.cyl_shell_width_edge_stats(
			res=_make_shell_result(xyz, data=_MaskOnlyData(mask))
		)

		self.assertIsNotNone(stats)
		self.assertAlmostEqual(float(stats["valid_avg_vx"]), 5.5, delta=1.0e-6)
		self.assertAlmostEqual(float(stats["invalid_avg_vx"]), 3.5, delta=1.0e-6)
		self.assertAlmostEqual(float(stats["invalid_frac"]), 0.5, delta=1.0e-6)

	def test_step_loss_uses_endpoint_validity_not_8_connected_erosion(self) -> None:
		xyz = torch.zeros(2, 3, 3, requires_grad=True)
		mask = torch.ones(2, 3)
		mask[1, 1] = 0.0

		loss, _, _ = opt_loss_cyl.cyl_step_loss(
			res=_make_shell_result(
				xyz,
				data=_MaskOnlyData(mask),
				cyl_shell_width_step=1.0,
				cyl_shell_height_step=1.0,
			)
		)

		self.assertGreater(float(loss.detach()), 0.5)

	def test_step_loss_replaces_invalid_step_terms_with_smoothing(self) -> None:
		xyz = torch.tensor(
			[
				[[0.0, 0.0, 0.0]],
				[[1.0, 0.0, 1.0]],
				[[0.0, 0.0, 2.0]],
			],
			requires_grad=True,
		)
		mask = torch.tensor([[1.0], [0.0], [1.0]])

		loss, _, _ = opt_loss_cyl.cyl_step_loss(
			res=_make_shell_result(
				xyz,
				data=_MaskOnlyData(mask),
				cyl_shell_width_step=1.0,
				cyl_shell_height_step=1.0,
			)
		)
		loss.backward()

		self.assertGreater(float(loss.detach()), 0.0)
		self.assertGreater(float(xyz.grad.detach().abs().sum()), 0.0)

	def test_step_loss_uses_grad_mask_without_normal_channels(self) -> None:
		xyz = torch.tensor(
			[
				[[0.0, 0.0, 0.0]],
				[[0.0, 0.0, 1.0]],
				[[0.0, 0.0, 2.0]],
			],
			requires_grad=True,
		)

		valid_loss, _, _ = opt_loss_cyl.cyl_step_loss(
			res=_make_shell_result(
				xyz,
				data=_MaskOnlyData(torch.ones(3, 1)),
				cyl_shell_width_step=1.0,
				cyl_shell_height_step=1.0,
			)
		)
		opt_loss_cyl.reset_candidate_terms()
		invalid_loss, _, _ = opt_loss_cyl.cyl_step_loss(
			res=_make_shell_result(
				xyz,
				data=_MaskOnlyData(torch.tensor([[1.0], [0.0], [1.0]])),
				cyl_shell_width_step=1.0,
				cyl_shell_height_step=1.0,
			)
		)

		self.assertGreater(float(valid_loss.detach()), 0.1)
		self.assertLess(float(invalid_loss.detach()), float(valid_loss.detach()))


class CylNormalInterpolationTests(unittest.TestCase):
	def tearDown(self) -> None:
		opt_loss_cyl.reset_candidate_terms()

	def test_shell_sample_gt_squeezes_fitdata_singleton_depth_mask(self) -> None:
		xyz = _make_shell(h=3, w=4, requires_grad=False)
		res = _make_shell_result(xyz, data=_FitDataShapeNormalData((0.0, 0.0, 1.0)))

		target, mask = opt_loss_cyl._sample_shell_gt(res=res, xyz=xyz)

		self.assertEqual(tuple(target.shape), tuple(xyz.shape))
		self.assertEqual(tuple(mask.shape), tuple(xyz.shape[:2]))

	def test_shell_normal_target_h_bracket_fills_with_half_weight(self) -> None:
		xyz = _make_planar_grid(h=5, w=5)
		target = torch.zeros(*xyz.shape[:2], 3)
		mask = torch.zeros(*xyz.shape[:2])
		target[1, 2] = torch.tensor([0.0, 0.0, 1.0])
		target[3, 2] = torch.tensor([0.0, 0.0, 1.0])
		mask[1, 2] = 1.0
		mask[3, 2] = 1.0

		out_target, out_mask = opt_loss_cyl._prepare_shell_normal_targets(xyz=xyz, target=target, mask=mask)

		self.assertAlmostEqual(float(out_mask[2, 2]), 0.5, delta=1.0e-6)
		self.assertAlmostEqual(float(out_mask[1, 2]), 1.0, delta=1.0e-6)
		self.assertGreater(float(out_target[2, 2, 2]), 0.99)

	def test_shell_normal_target_h_one_sided_does_not_fill(self) -> None:
		xyz = _make_planar_grid(h=5, w=5)
		target = torch.zeros(*xyz.shape[:2], 3)
		mask = torch.zeros(*xyz.shape[:2])
		target[1, 2] = torch.tensor([0.0, 0.0, 1.0])
		mask[1, 2] = 1.0

		_out_target, out_mask = opt_loss_cyl._prepare_shell_normal_targets(xyz=xyz, target=target, mask=mask)

		self.assertAlmostEqual(float(out_mask[2, 2]), 0.0, delta=1.0e-6)

	def test_shell_normal_target_w_wrap_fills_across_seam(self) -> None:
		xyz = _make_planar_grid(h=5, w=8)
		target = torch.zeros(*xyz.shape[:2], 3)
		mask = torch.zeros(*xyz.shape[:2])
		target[2, 7] = torch.tensor([0.0, 0.0, -1.0])
		target[2, 1] = torch.tensor([0.0, 0.0, -1.0])
		mask[2, 7] = 1.0
		mask[2, 1] = 1.0

		_out_target, out_mask = opt_loss_cyl._prepare_shell_normal_targets(xyz=xyz, target=target, mask=mask)

		self.assertAlmostEqual(float(out_mask[2, 0]), 0.5, delta=1.0e-6)

	def test_shell_normal_target_blend_favors_shorter_index_distance(self) -> None:
		xyz = _make_planar_grid(h=5, w=5)
		h_dir = F.normalize(torch.tensor([1.0, 0.0, 1.0]), dim=0, eps=1.0e-8)
		w_dir = F.normalize(torch.tensor([0.0, 1.0, 1.0]), dim=0, eps=1.0e-8)
		target = torch.zeros(*xyz.shape[:2], 3)
		mask = torch.zeros(*xyz.shape[:2])
		target[0, 2] = h_dir
		target[4, 2] = h_dir
		target[2, 1] = w_dir
		target[2, 3] = w_dir
		mask[0, 2] = 1.0
		mask[4, 2] = 1.0
		mask[2, 1] = 1.0
		mask[2, 3] = 1.0

		out_target, out_mask = opt_loss_cyl._prepare_shell_normal_targets(xyz=xyz, target=target, mask=mask)

		self.assertAlmostEqual(float(out_mask[2, 2]), 0.5, delta=1.0e-6)
		self.assertGreater(float((out_target[2, 2] * w_dir).sum()), float((out_target[2, 2] * h_dir).sum()))

	def test_shell_normal_target_sign_aligns_before_interpolation(self) -> None:
		xyz = _make_planar_grid(h=5, w=5)
		target = torch.zeros(*xyz.shape[:2], 3)
		mask = torch.zeros(*xyz.shape[:2])
		target[1, 2] = torch.tensor([0.0, 0.0, 1.0])
		target[3, 2] = torch.tensor([0.0, 0.0, -1.0])
		mask[1, 2] = 1.0
		mask[3, 2] = 1.0

		out_target, out_mask = opt_loss_cyl._prepare_shell_normal_targets(xyz=xyz, target=target, mask=mask)

		self.assertAlmostEqual(float(out_mask[2, 2]), 0.5, delta=1.0e-6)
		self.assertGreater(float(out_target[2, 2, 2]), 0.99)

	def test_shell_normal_target_all_invalid_stays_masked(self) -> None:
		xyz = _make_planar_grid(h=5, w=5)
		target = torch.zeros(*xyz.shape[:2], 3)
		mask = torch.zeros(*xyz.shape[:2])

		_out_target, out_mask = opt_loss_cyl._prepare_shell_normal_targets(xyz=xyz, target=target, mask=mask)

		self.assertAlmostEqual(float(out_mask.sum()), 0.0, delta=1.0e-6)


class CylNormalCompileCoreTests(unittest.TestCase):
	def tearDown(self) -> None:
		opt_loss_cyl.configure_compile(shell_normal=False)
		opt_loss_cyl.reset_candidate_terms()

	def test_direct_core_matches_compiled_wrapper_for_outputs_geometry_and_gradients(self) -> None:
		xyz_e = _make_shell()
		with torch.no_grad():
			geom_e, _ = opt_loss_cyl._shell_normal_geometry_core(
				xyz_e, None, None, None, factor=4, has_base=False)
		target, mask = _sampled_target(geom_e)
		err_e, lm_e, dot_e = opt_loss_cyl._shell_normal_compute_core(
			xyz_e, None, None, None, target, mask, 4, False)
		err_e.backward()
		grad_e = xyz_e.grad.detach().clone()

		xyz_c = _make_shell()
		opt_loss_cyl.configure_compile(shell_normal=True, backend="eager")
		err_c, lm_c, dot_c = opt_loss_cyl._run_shell_normal_compute_core(
			xyz_c, None, None, None, target, mask, factor=4, has_base=False)
		err_c.backward()
		with torch.no_grad():
			geom_c, _ = opt_loss_cyl._shell_normal_geometry_core(
				xyz_c, None, None, None, factor=4, has_base=False)

		_assert_close(self, err_c, err_e, "err")
		_assert_close(self, lm_c, lm_e, "lm")
		_assert_close(self, dot_c, dot_e, "dot_abs")
		_assert_close(self, geom_c, geom_e, "geometry")
		_assert_close(self, xyz_c.grad, grad_e, "xyz grad")

	def test_connected_core_matches_compiled_wrapper_for_delta_gradients(self) -> None:
		torch.manual_seed(1)
		xyz_ref = _make_shell(requires_grad=False)
		base = (xyz_ref + torch.tensor([0.5, -0.25, 0.0])).detach()
		offsets_e = (0.15 * torch.randn(xyz_ref.shape[:2])).requires_grad_(True)
		delta_xy = 0.05 * torch.randn(*xyz_ref.shape[:2], 2)
		delta_e = torch.cat([delta_xy, torch.zeros(*xyz_ref.shape[:2], 1)], dim=-1).requires_grad_(True)
		with torch.no_grad():
			geom_e, base_e = opt_loss_cyl._shell_normal_geometry_core(
				xyz_ref, base, offsets_e, delta_e, factor=4, has_base=True)
		target, mask = _sampled_target(geom_e)
		err_e, lm_e, dot_e = opt_loss_cyl._shell_normal_compute_core(
			xyz_ref, base, offsets_e, delta_e, target, mask, 4, True)
		err_e.backward()
		offsets_grad_e = offsets_e.grad.detach().clone()
		delta_grad_e = delta_e.grad.detach().clone()

		offsets_c = offsets_e.detach().clone().requires_grad_(True)
		delta_c = delta_e.detach().clone().requires_grad_(True)
		opt_loss_cyl.configure_compile(shell_normal=True, backend="eager")
		err_c, lm_c, dot_c = opt_loss_cyl._run_shell_normal_compute_core(
			xyz_ref, base, offsets_c, delta_c, target, mask, factor=4, has_base=True)
		err_c.backward()
		with torch.no_grad():
			geom_c, base_c = opt_loss_cyl._shell_normal_geometry_core(
				xyz_ref, base, offsets_c, delta_c, factor=4, has_base=True)

		_assert_close(self, err_c, err_e, "connected err")
		_assert_close(self, lm_c, lm_e, "connected lm")
		_assert_close(self, dot_c, dot_e, "connected dot_abs")
		_assert_close(self, geom_c, geom_e, "connected geometry")
		_assert_close(self, base_c, base_e, "connected base geometry")
		_assert_close(self, offsets_c.grad, offsets_grad_e, "connected offsets grad")
		_assert_close(self, delta_c.grad, delta_grad_e, "connected delta grad")

	def test_shell_cyl_normal_loss_matches_compiled_wrapper(self) -> None:
		xyz_e = _make_shell()
		opt_loss_cyl.configure_compile(shell_normal=False)
		loss_e, lm_e, mask_e = opt_loss_cyl.cyl_normal_loss(res=_make_shell_result(xyz_e))
		loss_e.backward()
		grad_e = xyz_e.grad.detach().clone()

		xyz_c = _make_shell()
		opt_loss_cyl.reset_candidate_terms()
		opt_loss_cyl.configure_compile(shell_normal=True, backend="eager")
		loss_c, lm_c, mask_c = opt_loss_cyl.cyl_normal_loss(res=_make_shell_result(xyz_c))
		loss_c.backward()

		_assert_close(self, loss_c, loss_e, "full loss")
		_assert_close(self, lm_c[0], lm_e[0], "full lm")
		_assert_close(self, mask_c[0], mask_e[0], "full mask")
		_assert_close(self, xyz_c.grad, grad_e, "full xyz grad")


class CylOutsideLossTests(unittest.TestCase):
	def tearDown(self) -> None:
		opt_loss_cyl.reset_candidate_terms()

	def test_outside_loss_uses_u8_diff_sampler_masks_grad_mag_and_backprops(self) -> None:
		xyz = torch.tensor(
			[
				[[16.0, 0.0, 0.0], [24.0, 0.0, 0.0], [32.0, 0.0, 0.0]],
				[[40.0, 0.0, 1.0], [48.0, 0.0, 1.0], [56.0, 0.0, 1.0]],
			],
			requires_grad=True,
		)
		mask = torch.ones(2, 3)
		mask[0, 1] = 0.0
		calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []

		def _fake_sampler(volume, grid, offset, inv_scale):
			calls.append((volume, grid, offset, inv_scale))
			return (grid[..., 0] * 4.0).clamp(max=255.0).unsqueeze(0)

		res = _make_shell_result(
			xyz,
			data=_MaskOnlyData(mask),
			mesh_step=5,
			cyl_shell_index=1,
			cyl_outside_volume=torch.zeros(1, 4, 4, 4, dtype=torch.uint8),
			cyl_outside_origin=(0.0, 0.0, 0.0),
			cyl_outside_spacing=(1.0, 1.0, 1.0),
			cyl_outside_shape=(4, 4, 4),
			cyl_outside_depth_max=10.0,
			cyl_outside_sample_factor=1,
			cyl_outside_model_step=5.0,
		)

		with mock.patch.object(opt_loss_cyl, "grid_sample_3d_u8_diff", side_effect=_fake_sampler):
			loss, _lms, masks = opt_loss_cyl.cyl_outside_loss(res=res)
		loss.backward()

		self.assertEqual(len(calls), 1)
		self.assertEqual(tuple(calls[0][1].shape), (1, 2, 3, 3))
		self.assertAlmostEqual(float(masks[0].sum()), float(mask.sum()), delta=1.0e-6)
		self.assertGreater(float(xyz.grad.detach()[mask.bool()].abs().sum()), 0.0)
		self.assertAlmostEqual(float(xyz.grad.detach()[0, 1].abs().sum()), 0.0, delta=1.0e-7)
		stats = opt_loss_cyl.last_stats()
		self.assertGreater(stats["cyl_outside_pen_frac"], 0.0)
		self.assertGreater(stats["cyl_outside_depth_max"], 0.0)

	def test_outside_loss_is_zero_without_previous_shell_volume(self) -> None:
		xyz = _make_shell(requires_grad=True)
		loss, _, _ = opt_loss_cyl.cyl_outside_loss(
			res=_make_shell_result(xyz, cyl_shell_index=1)
		)

		self.assertAlmostEqual(float(loss.detach()), 0.0, delta=1.0e-8)


if __name__ == "__main__":
	unittest.main()
