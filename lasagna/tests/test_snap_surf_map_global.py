from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from types import SimpleNamespace

import torch
import optimizer
import fit_data

TEST_DIR = os.path.dirname(__file__)
if TEST_DIR not in sys.path:
	sys.path.insert(0, TEST_DIR)

from snap_surf.map_fixture_io import _float_tif, _mask_tif, _write_json, _write_vector_dir, load_map_fixture
from snap_surf.map_global import (
	AffineMapModel,
	BoundarySelfMapRuntime,
	GlobalMapModel,
	GlobalMapRuntime,
	GlobalMapStageConfig,
	GlobalMapConfig,
	MapRuntimeStatusPrinter,
	SelfMapRuntime,
	_LrAutoscaleState,
	_affine_from_seed_ext_quads,
	_affine_multistart_candidates,
	_affine_seed_quad_expansion_rows,
	_affine_seed_grid_candidates,
	_apply_external_quad_health_filter,
	_apply_optimizer_lr_schedule,
	_objective_for_uv,
	_full_active_quad,
	_lr_autoscale_config,
	_run_affine_seed_quad_expansion_reopt,
	_select_affine_seed_grid_candidate,
	_seed_quad_affine_init_result,
	_stage_objective_level,
	_stage_loss_cfg,
	_stage_station_weight,
	_write_map_objs,
	_self_map_objective_for_uv,
	self_map_active_quads,
	self_map_initial_uv,
	self_map_pair_depths,
	optimize_fixture,
	parse_global_map_stage_item,
	parse_global_map_config,
	snap_surf_config_from_global_config,
)
from snap_surf.map_pyramid import _map_init_integrate_dyadic_uv_pyramid, _map_init_uv_pyr_from_dense
from snap_surf import map_global_cli
from snap_surf_test_utils import _normals_2d, _normals_3d, _plane_xyz


def _constant_grad_data(value: float, *, shape: tuple[int, int, int] = (6, 6, 6)) -> fit_data.FitData3D:
	return fit_data.FitData3D(
		cos=None,
		grad_mag=torch.full((1, 1, *shape), float(value), dtype=torch.float32),
		nx=None,
		ny=None,
		pred_dt=None,
		corr_points=None,
		winding_volume=None,
		origin_fullres=(0.0, 0.0, 0.0),
		spacing=(1.0, 1.0, 1.0),
		grad_mag_scale=1.0,
		cuda_gridsample=False,
	)


class _RejectNonFiniteGradData:
	def __init__(self, value: float = 0.5) -> None:
		self.value = float(value)
		self.origin_fullres = (0.0, 0.0, 0.0)
		self.spacing = (1.0, 1.0, 1.0)
		self.channels: list[str] = []
		self.queries: list[torch.Tensor] = []

	def _spacing_for(self, channel: str) -> tuple[float, float, float]:
		self.channels.append(channel)
		return self.spacing

	def grid_sample_fullres(self, xyz_fullres: torch.Tensor, *, channels=None, diff: bool = False):
		self.queries.append(xyz_fullres.detach().clone())
		if not bool(torch.isfinite(xyz_fullres).all().detach().cpu()):
			raise AssertionError("grid_sample_fullres received non-finite coordinates")
		shape = tuple(int(v) for v in xyz_fullres.shape[:-1])
		grad_mag = torch.full((1, 1, *shape), self.value, device=xyz_fullres.device, dtype=xyz_fullres.dtype)
		return SimpleNamespace(grad_mag=grad_mag)


def _write_planar_global_fixture(root: str, *, h: int = 5, w: int = 5, offset_h: float = 0.25, offset_w: float = 0.5) -> torch.Tensor:
	path = Path(root)
	path.mkdir(parents=True, exist_ok=True)
	model_xyz = _plane_xyz(h=h + 2, w=w + 2, z=0.0).unsqueeze(0)
	ext_xyz = _plane_xyz(h=h, w=w, z=0.0, offset_h=offset_h, offset_w=offset_w)
	model_valid = torch.ones(1, h + 2, w + 2, dtype=torch.bool)
	ext_valid = torch.ones(h, w, dtype=torch.bool)
	ext_quad = torch.ones(h - 1, w - 1, dtype=torch.bool)
	model_normals = _normals_3d(1, h + 2, w + 2)
	ext_normals = _normals_2d(h, w)
	hh = torch.arange(h, dtype=torch.float32).view(h, 1).expand(h, w) + float(offset_h)
	ww = torch.arange(w, dtype=torch.float32).view(1, w).expand(h, w) + float(offset_w)
	reference_uv = torch.stack([hh, ww], dim=-1)

	_write_vector_dir(path / "ext_surface", ext_xyz, ext_valid, meta={"kind": "external_surface"})
	_mask_tif(path / "ext_surface" / "quad_valid.tif", ext_quad)
	_write_vector_dir(path / "model_stack", model_xyz, model_valid, meta={"kind": "model_stack"})
	_write_vector_dir(path / "ext_normals", ext_normals, ext_valid, meta={"kind": "external_normals"})
	_write_vector_dir(path / "model_normals", model_normals, model_valid, meta={"kind": "model_normals"})
	(path / "map").mkdir(exist_ok=True)
	_float_tif(path / "map" / "model_x.tif", reference_uv[..., 1])
	_float_tif(path / "map" / "model_y.tif", reference_uv[..., 0])
	_mask_tif(path / "map" / "active_quad.tif", ext_quad)
	_mask_tif(path / "map" / "blocked_quad.tif", torch.zeros_like(ext_quad))
	_write_json(
		path / "fixture.json",
		{
			"schema_version": 1,
			"kind": "snap_surf_map_fixture",
			"seed_xyz": [2.0, 2.0, 0.0],
			"seed_ext_sample_hw": [2, 2],
			"model_depth": 0,
			"sign": 1,
			"snap_surf_config": {"map_init": {"subdiv": 1}},
		},
	)
	return reference_uv


def _xy_normals_like(shape: tuple[int, ...]) -> torch.Tensor:
	n = torch.zeros(*shape, 3, dtype=torch.float32)
	n[..., 0] = 1.0
	return n


def _write_config(root: str, *, affine_steps: int = 8, map_steps: int = 8, write_objs: bool = False) -> str:
	path = Path(root, "cfg.json")
	base = {
		"map_station_t": 0.001,
		"map_init": {
			"subdiv": 2,
			"scale_levels": 4,
			"w_dist": 1.0,
			"w_vec_normal": 0.0,
			"w_surface_normal": 0.0,
			"w_smooth": 0.0,
			"w_bend": 0.0,
			"w_jac": 0.0,
			"w_metric_smooth": 0.0,
			"w_area_smooth": 0.0,
			"max_sample_angle_deg": 180.0,
			"max_step_neighbor_ratio": 0.0,
		},
	}
	if bool(write_objs):
		base["write_objs"] = True
	_write_json(
		path,
		{
			"base": base,
			"stages": [
				{"steps": affine_steps, "lr": 0.05, "params": ["map_surf_affine"], "args": {"subdiv": 2}},
				{
					"steps": map_steps,
					"lr": 0.02,
					"params": ["map_surf_ms"],
					"min_scaledown": 3,
					"args": {"subdiv": 2, "map_station_t": 0.001},
				},
			],
		},
	)
	return str(path)


class SnapSurfMapGlobalTest(unittest.TestCase):
	def test_map_runtime_status_printer_includes_smoothing_losses(self) -> None:
		stdout = StringIO()
		stats = {
			"snaps_map_loss": 1.0,
			"snaps_map_dist": 2.0,
			"snaps_map_vec": 3.0,
			"snaps_map_norm": 4.0,
			"snaps_map_turn": 5.0,
			"snaps_map_smooth": 6.0,
			"snaps_map_bend": 7.0,
			"snaps_map_metric_smooth": 8.0,
			"snaps_map_area_smooth": 9.0,
		}

		with redirect_stdout(stdout):
			MapRuntimeStatusPrinter(label="stage", total_steps=1).print(
				step=0,
				total=1,
				stats=stats,
			)

		out = stdout.getvalue()
		self.assertIn("sm_smo", out)
		self.assertIn("sm_bnd", out)
		self.assertIn("sm_met", out)
		self.assertIn("sm_ar", out)

	def _snap_loss_case(
		self,
		*,
		offset: float,
		model_z: float = 2.0,
		grad_mag: float = 0.5,
		data=None,
		mutate_model_xyz=None,
		mutate_ext_xyz=None,
		runtime_sign: int = 1,
	):
		h, w = 5, 5
		runtime = GlobalMapRuntime()
		runtime.sign = int(runtime_sign)
		model_xyz = _plane_xyz(h=h, w=w, z=model_z).unsqueeze(0)
		model_normals = _normals_3d(1, h, w)
		model_valid = torch.ones(1, h, w, dtype=torch.bool)
		ext_xyz = _plane_xyz(h=h, w=w, z=0.0)
		ext_valid = torch.ones(h, w, dtype=torch.bool)
		ext_normals = _normals_2d(h, w)
		ext_quad = torch.ones(h - 1, w - 1, dtype=torch.bool)
		stdout = StringIO()
		if mutate_model_xyz is not None:
			mutate_model_xyz(model_xyz)
		if mutate_ext_xyz is not None:
			mutate_ext_xyz(ext_xyz)
		if data is None:
			data = _constant_grad_data(grad_mag)

		with redirect_stdout(stdout):
			loss, _lms, _masks, stats = runtime.snap_loss(
				model_xyz=model_xyz,
				model_normals=model_normals,
				model_valid=model_valid,
				ext_xyz=ext_xyz,
				ext_valid=ext_valid,
				ext_normals=ext_normals,
				ext_quad_valid=ext_quad,
				offset=offset,
				data=data,
				strip_samples=4,
			)
		return loss, stats, stdout.getvalue()

	def test_lasagna_stage_parser_routes_model_and_map_params(self) -> None:
		cfg = {
			"base": {"snap_surf_map": 1.0},
			"stages": [
				{"name": "model", "steps": 1, "lr": 0.1, "params": ["mesh_ms"]},
				{"name": "map", "steps": 1, "lr": 0.01, "params": ["map_surf_affine"], "w_fac": 1.0},
			],
		}

		stages = optimizer.load_stages_cfg(cfg)

		self.assertEqual(stages[0].global_opt.kind, "model")
		self.assertEqual(stages[1].global_opt.kind, "map")

	def test_map_objective_can_use_min_scaledown_sampling_grid(self) -> None:
		with tempfile.TemporaryDirectory() as tmp:
			_write_planar_global_fixture(tmp, h=9, w=9)
			fixture = load_map_fixture(tmp)
			cfg = snap_surf_config_from_global_config(parse_global_map_config(_write_config(tmp, affine_steps=0, map_steps=0)))
			affine = AffineMapModel(
				ext_shape=tuple(int(v) for v in fixture.ext_xyz.shape[:2]),
				device=torch.device("cpu"),
				dtype=torch.float32,
			)
			full_uv = affine()
			global_model = GlobalMapModel(full_uv.detach(), levels=4, factor=2)

			_loss0, terms0 = _objective_for_uv(
				uv=global_model(active_level=0),
				fixture=fixture,
				cfg=cfg,
				level=0,
				z_lift=None,
			)
			_loss3, terms3 = _objective_for_uv(
				uv=global_model(active_level=3),
				fixture=fixture,
				cfg=cfg,
				level=3,
				z_lift=None,
			)

			self.assertEqual(int(terms0["sample_total"].detach().cpu()), 8 * 8 * 4)
			self.assertEqual(int(terms3["sample_total"].detach().cpu()), 1 * 1 * 4)

	def test_map_objective_sampling_defaults_to_full_resolution(self) -> None:
		ext_shape = (9, 9)
		stage = GlobalMapStageConfig(params=("map_uv_ms",), min_scaledown=3)
		coarse = GlobalMapStageConfig(
			params=("map_uv_ms",),
			min_scaledown=3,
			args={"use_min_scaledown_sampling": True},
		)
		explicit = GlobalMapStageConfig(
			params=("map_uv_ms",),
			min_scaledown=3,
			args={"objective_min_scaledown": 2},
		)

		self.assertEqual(_stage_objective_level(stage, 3, ext_shape), 0)
		self.assertEqual(_stage_objective_level(coarse, 3, ext_shape), 3)
		self.assertEqual(_stage_objective_level(explicit, 3, ext_shape), 2)

	def test_self_map_mode1_initializes_single_shifted_map(self) -> None:
		uv_out = self_map_initial_uv(
			mode="multi_wrap_full",
			direction="out",
			depth=1,
			height=3,
			width=6,
			model_w_wraps=2.5,
			device=torch.device("cpu"),
			dtype=torch.float32,
		)
		uv_in = self_map_initial_uv(
			mode="multi_wrap_full",
			direction="in",
			depth=1,
			height=3,
			width=6,
			model_w_wraps=2.5,
			device=torch.device("cpu"),
			dtype=torch.float32,
		)

		self.assertEqual(tuple(uv_out.shape), (1, 3, 6, 2))
		self.assertTrue(torch.allclose(uv_out[0, :, :, 0], torch.arange(3, dtype=torch.float32).view(3, 1).expand(3, 6)))
		self.assertAlmostEqual(float(uv_out[0, 0, 0, 1]), 2.0)
		self.assertAlmostEqual(float(uv_in[0, 0, 0, 1]), -2.0)
		self.assertEqual(self_map_pair_depths("multi_wrap_full", "out", 1), ([0], [0]))

	def test_self_map_mode2_initializes_identity_pairs(self) -> None:
		uv = self_map_initial_uv(
			mode="multi_wrap_d",
			direction="out",
			depth=4,
			height=3,
			width=5,
			device=torch.device("cpu"),
			dtype=torch.float32,
		)

		self.assertEqual(tuple(uv.shape), (3, 3, 5, 2))
		self.assertEqual(self_map_pair_depths("multi_wrap_d", "out", 4), ([0, 1, 2], [1, 2, 3]))
		self.assertEqual(self_map_pair_depths("multi_wrap_d", "in", 4), ([1, 2, 3], [0, 1, 2]))
		hh = torch.arange(3, dtype=torch.float32).view(3, 1).expand(3, 5)
		ww = torch.arange(5, dtype=torch.float32).view(1, 5).expand(3, 5)
		self.assertTrue(torch.allclose(uv[2, :, :, 0], hh))
		self.assertTrue(torch.allclose(uv[2, :, :, 1], ww))

	def test_stacked_uv_pyramid_preserves_batch_axis(self) -> None:
		uv = self_map_initial_uv(
			mode="multi_wrap_d",
			direction="out",
			depth=3,
			height=5,
			width=5,
			device=torch.device("cpu"),
			dtype=torch.float32,
		)
		pyr = _map_init_uv_pyr_from_dense(uv, levels=3, factor=2)
		recon = _map_init_integrate_dyadic_uv_pyramid(list(pyr), preserve_batch=True)

		self.assertEqual(tuple(pyr[0].shape), (2, 2, 5, 5))
		self.assertEqual(tuple(pyr[1].shape[:2]), (2, 2))
		self.assertEqual(tuple(recon.shape), tuple(uv.shape))
		self.assertTrue(torch.allclose(recon, uv, atol=1.0e-6))

	def test_self_map_batched_objective_matches_pair_average(self) -> None:
		D, H, W = 3, 4, 4
		model_xyz = torch.stack([_plane_xyz(h=H, w=W, z=float(d)) for d in range(D)], dim=0)
		model_normals = _normals_3d(D, H, W)
		model_valid = torch.ones(D, H, W, dtype=torch.bool)
		uv = self_map_initial_uv(
			mode="multi_wrap_d",
			direction="out",
			depth=D,
			height=H,
			width=W,
			device=torch.device("cpu"),
			dtype=torch.float32,
		)
		cfg = snap_surf_config_from_global_config(
			GlobalMapConfig(base={
				"map_init": {
					"subdiv": 1,
					"w_vec_normal": 0.0,
					"w_surface_normal": 0.0,
					"w_smooth": 0.0,
					"w_bend": 0.0,
					"w_jac": 0.0,
					"w_metric_smooth": 0.0,
					"w_area_smooth": 0.0,
					"w_z_lift": 0.0,
					"max_sample_angle_deg": 180.0,
					"max_step_neighbor_ratio": 0.0,
				}
			}, stages=())
		)
		active = self_map_active_quads(mode="multi_wrap_d", direction="out", model_valid=model_valid, uv=uv)

		loss_b, _terms = _self_map_objective_for_uv(
			uv=uv,
			mode="multi_wrap_d",
			direction="out",
			model_xyz=model_xyz,
			model_normals=model_normals,
			model_valid=model_valid,
			cfg=cfg,
			level=0,
			active_quad=active,
		)
		per_pair = []
		for i in range(D - 1):
			loss_i, _ = _self_map_objective_for_uv(
				uv=uv[i:i + 1],
				mode="multi_wrap_d",
				direction="out",
				model_xyz=model_xyz[i:i + 2],
				model_normals=model_normals[i:i + 2],
				model_valid=model_valid[i:i + 2],
				cfg=cfg,
				level=0,
				active_quad=active[i:i + 1],
			)
			per_pair.append(loss_i)
		expected = torch.stack(per_pair).mean()

		self.assertTrue(torch.allclose(loss_b, expected, atol=1.0e-6))

	def test_self_map_batched_snap_loss_matches_pair_average(self) -> None:
		D, H, W = 3, 4, 4
		model_xyz = torch.stack([_plane_xyz(h=H, w=W, z=float(d)) for d in range(D)], dim=0)
		model_normals = _normals_3d(D, H, W)
		model_valid = torch.ones(D, H, W, dtype=torch.bool)
		runtime = SelfMapRuntime(mode="multi_wrap_d", direction="out")
		loss_b, _lms, _masks, stats_b = runtime.snap_loss(
			model_xyz=model_xyz,
			model_normals=model_normals,
			model_valid=model_valid,
			offset=1.0,
			data=None,
		)
		per_pair = []
		samples = 0.0
		for i in range(D - 1):
			pair_runtime = SelfMapRuntime(mode="multi_wrap_d", direction="out")
			loss_i, _lms_i, _masks_i, stats_i = pair_runtime.snap_loss(
				model_xyz=model_xyz[i:i + 2],
				model_normals=model_normals[i:i + 2],
				model_valid=model_valid[i:i + 2],
				offset=1.0,
				data=None,
			)
			per_pair.append(loss_i)
			samples += stats_i["snaps_map_snap_samples"]

		expected = torch.stack(per_pair).mean()

		self.assertTrue(torch.allclose(loss_b, expected, atol=1.0e-6))
		self.assertEqual(stats_b["snaps_map_snap_samples"], samples)

	def test_self_map_in_direction_uses_negative_signed_offset(self) -> None:
		D, H, W = 2, 4, 4
		model_xyz = torch.stack([_plane_xyz(h=H, w=W, z=float(d)) for d in range(D)], dim=0)
		model_normals = _normals_3d(D, H, W)
		model_valid = torch.ones(D, H, W, dtype=torch.bool)

		for data in (None, _constant_grad_data(1.0)):
			loss_out, _lms_o, _masks_o, stats_out = SelfMapRuntime(
				mode="multi_wrap_d",
				direction="out",
			).snap_loss(
				model_xyz=model_xyz,
				model_normals=model_normals,
				model_valid=model_valid,
				offset=1.0,
				data=data,
			)
			loss_in, _lms_i, _masks_i, stats_in = SelfMapRuntime(
				mode="multi_wrap_d",
				direction="in",
			).snap_loss(
				model_xyz=model_xyz,
				model_normals=model_normals,
				model_valid=model_valid,
				offset=1.0,
				data=data,
			)

			self.assertLess(float(loss_out.detach()), 1.0e-6)
			self.assertLess(float(loss_in.detach()), 1.0e-6)
			self.assertEqual(stats_out["snaps_map_snap_samples"], stats_in["snaps_map_snap_samples"])

	def test_boundary_self_map_snap_loss_identity_maps_both_directions(self) -> None:
		H, W = 4, 4
		fixed_xyz = _plane_xyz(h=H, w=W, z=0.0)
		new_xyz = _plane_xyz(h=H, w=W, z=1.0).unsqueeze(0)
		fixed_normals = _normals_2d(H, W)
		new_normals = _normals_3d(1, H, W)
		fixed_valid = torch.ones(H, W, dtype=torch.bool)
		new_valid = torch.ones(1, H, W, dtype=torch.bool)

		loss_out, _lms_o, _masks_o, stats_out = BoundarySelfMapRuntime(
			mode="multi_wrap_d",
			direction="out",
			fixed_xyz=fixed_xyz,
			fixed_normals=fixed_normals,
			fixed_valid=fixed_valid,
		).snap_loss(
			model_xyz=new_xyz,
			model_normals=new_normals,
			model_valid=new_valid,
			offset=1.0,
			data=None,
		)
		loss_in, _lms_i, _masks_i, stats_in = BoundarySelfMapRuntime(
			mode="multi_wrap_d",
			direction="in",
			fixed_xyz=fixed_xyz,
			fixed_normals=fixed_normals,
			fixed_valid=fixed_valid,
		).snap_loss(
			model_xyz=new_xyz,
			model_normals=new_normals,
			model_valid=new_valid,
			offset=1.0,
			data=None,
		)

		self.assertLess(float(loss_out.detach()), 1.0e-6)
		self.assertLess(float(loss_in.detach()), 1.0e-6)
		self.assertEqual(stats_out["snaps_map_snap_samples"], stats_in["snaps_map_snap_samples"])
		self.assertGreater(stats_out["snaps_map_snap_samples"], 0.0)

	def test_boundary_self_map_in_direction_grads_trainable_source_only(self) -> None:
		H, W = 4, 4
		fixed_xyz = _plane_xyz(h=H, w=W, z=0.0).requires_grad_(True)
		new_xyz = _plane_xyz(h=H, w=W, z=1.25).unsqueeze(0).detach().requires_grad_(True)
		runtime = BoundarySelfMapRuntime(
			mode="multi_wrap_d",
			direction="in",
			fixed_xyz=fixed_xyz,
			fixed_normals=_normals_2d(H, W),
			fixed_valid=torch.ones(H, W, dtype=torch.bool),
		)

		loss, _lms, _masks, stats = runtime.snap_loss(
			model_xyz=new_xyz,
			model_normals=_normals_3d(1, H, W),
			model_valid=torch.ones(1, H, W, dtype=torch.bool),
			offset=1.0,
			data=None,
		)
		loss.backward()

		self.assertGreater(stats["snaps_map_snap_samples"], 0.0)
		self.assertIsNotNone(new_xyz.grad)
		self.assertGreater(float(new_xyz.grad.detach().abs().sum()), 0.0)
		self.assertIsNone(fixed_xyz.grad)

	def test_boundary_self_map_runtime_serializes_like_self_map_state(self) -> None:
		H, W = 5, 5
		runtime = BoundarySelfMapRuntime(
			mode="multi_wrap_d",
			direction="out",
			fixed_xyz=_plane_xyz(h=H, w=W, z=0.0),
			fixed_normals=_normals_2d(H, W),
			fixed_valid=torch.ones(H, W, dtype=torch.bool),
		)
		stage = GlobalMapStageConfig(
			steps=0,
			lr=0.01,
			params=("map_uv_ms",),
			args={"startup_timing": False},
		)

		stats = runtime.run_stage(
			stage=stage,
			model_xyz=_plane_xyz(h=H, w=W, z=1.0).unsqueeze(0),
			model_normals=_normals_3d(1, H, W),
			model_valid=torch.ones(1, H, W, dtype=torch.bool),
		)

		self.assertIsNotNone(runtime.global_model)
		assert runtime.global_model is not None
		state = {
			"preserve_batch": bool(runtime.global_model.preserve_batch),
			"map_uv_ms": [p.detach().cpu() for p in runtime.global_model.map_uv_ms],
			"mode": runtime.mode,
			"direction": runtime.direction,
			"steps_run": runtime.steps_run,
		}
		self.assertTrue(state["preserve_batch"])
		self.assertEqual(state["mode"], "multi_wrap_d")
		self.assertEqual(state["direction"], "out")
		self.assertEqual(state["map_uv_ms"][0].shape[0], 1)
		self.assertIn("snaps_map_loss", stats)

	def test_runtime_map_init_keeps_z_lift_unless_disabled(self) -> None:
		h, w = 5, 5
		angles = torch.linspace(0.0, math.pi, w)
		ext_normals = torch.zeros(h, w, 3)
		ext_normals[..., 0] = torch.cos(angles).view(1, w)
		ext_normals[..., 1] = torch.sin(angles).view(1, w)
		model_normals = torch.zeros(1, h, w, 3)
		model_normals[..., 0] = 1.0
		base = {
			"map_init": {
				"subdiv": 1,
				"w_dist": 0.0,
				"w_vec_normal": 0.0,
				"w_surface_normal": 0.0,
				"w_z_lift": 1.0,
				"w_smooth": 0.0,
				"w_bend": 0.0,
				"w_jac": 0.0,
				"w_metric_smooth": 0.0,
				"w_area_smooth": 0.0,
				"max_sample_angle_deg": 180.0,
				"max_step_neighbor_ratio": 0.0,
			},
		}
		common = dict(
			model_xyz=_plane_xyz(h=h, w=w, z=0.0).unsqueeze(0),
			model_normals=model_normals,
			model_valid=torch.ones(1, h, w, dtype=torch.bool),
			ext_xyz=_plane_xyz(h=h, w=w, z=0.0),
			ext_valid=torch.ones(h, w, dtype=torch.bool),
			ext_normals=ext_normals,
			ext_quad_valid=torch.ones(h - 1, w - 1, dtype=torch.bool),
		)
		stage = GlobalMapStageConfig(steps=0, lr=0.01, params=("affine",), args={"startup_timing": False})
		runtime = GlobalMapRuntime(base=base, seed_xyz=(0.5, 0.5, 0.0))
		stats = runtime.run_stage(stage=stage, **common)

		disabled = GlobalMapRuntime(base=base, seed_xyz=(0.5, 0.5, 0.0))
		disabled_stats = disabled.run_stage(
			stage=GlobalMapStageConfig(steps=0, lr=0.01, params=("affine",), args={"startup_timing": False, "disable_z_lift": True}),
			**common,
		)

		self.assertGreater(stats["snaps_map_turn"], 0.0)
		self.assertGreater(stats["snaps_map_turn_smp"], 0.0)
		self.assertEqual(disabled_stats["snaps_map_turn"], 0.0)
		self.assertEqual(disabled_stats["snaps_map_turn_smp"], 0.0)

	def test_snap_loss_zero_offset_preserves_voxel_residual(self) -> None:
		loss, stats, out = self._snap_loss_case(offset=0.0, model_z=2.0, grad_mag=0.5)

		self.assertAlmostEqual(float(loss.detach()), 4.0, places=6)
		self.assertAlmostEqual(stats["snaps_map_snap_abs"], 2.0, places=6)
		self.assertIn("snap loss offset_mode=voxel offset=0", out)

	def test_snap_loss_nonzero_offset_uses_winding_residual(self) -> None:
		loss, stats, out = self._snap_loss_case(offset=1.0, model_z=2.0, grad_mag=0.5)

		self.assertAlmostEqual(float(loss.detach()), 0.0, places=6)
		self.assertAlmostEqual(stats["snaps_map_snap_abs"], 0.0, places=6)
		self.assertEqual(stats["snaps_map_snap_samples"], 25.0)
		self.assertIn("snap loss offset_mode=winding offset=1", out)

	def test_snap_loss_nonzero_offset_stats_report_winding_error(self) -> None:
		loss, stats, _out = self._snap_loss_case(offset=2.0, model_z=2.0, grad_mag=0.5)

		self.assertAlmostEqual(float(loss.detach()), 1.0, places=6)
		self.assertAlmostEqual(stats["snaps_map_snap_abs"], 1.0, places=6)
		self.assertAlmostEqual(stats["snaps_map_snap_max"], 1.0, places=6)

	def test_snap_loss_prefetch_items_cover_winding_strip_query(self) -> None:
		h, w = 5, 5
		runtime = GlobalMapRuntime()
		model_xyz = _plane_xyz(h=h, w=w, z=2.0).unsqueeze(0)
		model_normals = _normals_3d(1, h, w)
		model_valid = torch.ones(1, h, w, dtype=torch.bool)
		ext_xyz = _plane_xyz(h=h, w=w, z=0.0)
		ext_valid = torch.ones(h, w, dtype=torch.bool)
		ext_normals = _normals_2d(h, w)
		ext_quad = torch.ones(h - 1, w - 1, dtype=torch.bool)
		data = _RejectNonFiniteGradData(value=0.5)

		items = runtime.snap_loss_prefetch_items(
			model_xyz=model_xyz,
			model_normals=model_normals,
			model_valid=model_valid,
			ext_xyz=ext_xyz,
			ext_valid=ext_valid,
			ext_normals=ext_normals,
			ext_quad_valid=ext_quad,
			offset=1.0,
			data=data,
			strip_samples=3,
		)
		runtime.snap_loss(
			model_xyz=model_xyz,
			model_normals=model_normals,
			model_valid=model_valid,
			ext_xyz=ext_xyz,
			ext_valid=ext_valid,
			ext_normals=ext_normals,
			ext_quad_valid=ext_quad,
			offset=1.0,
			data=data,
			strip_samples=3,
		)

		self.assertIn("grad_mag", items)
		self.assertEqual(int(items["grad_mag"].reshape(-1, 3).shape[0]), int(data.queries[0].reshape(-1, 3).shape[0]))
		self.assertTrue(torch.allclose(items["grad_mag"].reshape(-1, 3), data.queries[0].reshape(-1, 3)))

	def test_snap_loss_nonzero_offset_measures_tangential_segment_length(self) -> None:
		h, w = 5, 5
		runtime = GlobalMapRuntime()
		runtime.affine = AffineMapModel(ext_shape=(h, w), device=torch.device("cpu"), dtype=torch.float32)
		model_xyz = _plane_xyz(h=h, w=w, z=2.0).unsqueeze(0)
		model_xyz[..., 0] += 3.0
		model_normals = _normals_3d(1, h, w)
		model_valid = torch.ones(1, h, w, dtype=torch.bool)
		ext_xyz = _plane_xyz(h=h, w=w, z=0.0)
		ext_valid = torch.ones(h, w, dtype=torch.bool)
		ext_normals = _normals_2d(h, w)
		ext_quad = torch.ones(h - 1, w - 1, dtype=torch.bool)
		grad_mag = 0.5
		offset = (13.0 ** 0.5) * grad_mag

		loss, _lms, _masks, stats = runtime.snap_loss(
			model_xyz=model_xyz,
			model_normals=model_normals,
			model_valid=model_valid,
			ext_xyz=ext_xyz,
			ext_valid=ext_valid,
			ext_normals=ext_normals,
			ext_quad_valid=ext_quad,
			offset=offset,
			data=_RejectNonFiniteGradData(grad_mag),
			strip_samples=4,
		)

		self.assertAlmostEqual(float(loss.detach()), 0.0, places=6)
		self.assertAlmostEqual(stats["snaps_map_snap_abs"], 0.0, places=6)
		self.assertEqual(stats["snaps_map_snap_samples"], 25.0)

	def test_snap_loss_nonzero_offset_backprops_only_model_normal_direction(self) -> None:
		h, w = 5, 5
		runtime = GlobalMapRuntime()
		runtime.affine = AffineMapModel(ext_shape=(h, w), device=torch.device("cpu"), dtype=torch.float32)
		model_xyz = _plane_xyz(h=h, w=w, z=2.0).unsqueeze(0).clone()
		model_xyz[..., 0] += 3.0
		model_xyz.requires_grad_()
		model_normals = _normals_3d(1, h, w)
		model_valid = torch.ones(1, h, w, dtype=torch.bool)
		ext_xyz = _plane_xyz(h=h, w=w, z=0.0)
		ext_valid = torch.ones(h, w, dtype=torch.bool)
		ext_normals = _normals_2d(h, w)
		ext_quad = torch.ones(h - 1, w - 1, dtype=torch.bool)

		loss, _lms, _masks, _stats = runtime.snap_loss(
			model_xyz=model_xyz,
			model_normals=model_normals,
			model_valid=model_valid,
			ext_xyz=ext_xyz,
			ext_valid=ext_valid,
			ext_normals=ext_normals,
			ext_quad_valid=ext_quad,
			offset=1.0,
			data=_RejectNonFiniteGradData(0.5),
			strip_samples=4,
		)
		loss.backward()

		grad = model_xyz.grad.detach()
		self.assertLess(float(grad[..., :2].abs().max()), 1.0e-7)
		self.assertGreater(float(grad[..., 2].abs().max()), 1.0e-5)

	def test_snap_loss_nonzero_offset_masks_invalid_grad_mag_strip(self) -> None:
		loss, stats, _out = self._snap_loss_case(offset=1.0, model_z=2.0, grad_mag=0.0)

		self.assertAlmostEqual(float(loss.detach()), 0.0, places=6)
		self.assertEqual(stats["snaps_map_snap_samples"], 0.0)

	def test_snap_loss_nonzero_offset_positive_side_follows_aligned_model_normal(self) -> None:
		loss_pos, stats_pos, _out = self._snap_loss_case(offset=1.0, model_z=2.0, grad_mag=0.5, runtime_sign=1)
		above_ext = lambda ext_xyz: ext_xyz.__setitem__((slice(None), slice(None), 2), 4.0)
		loss_neg, stats_neg, _out = self._snap_loss_case(
			offset=1.0,
			model_z=2.0,
			grad_mag=0.5,
			runtime_sign=-1,
			mutate_ext_xyz=above_ext,
		)
		loss_wrong, stats_wrong, _out = self._snap_loss_case(
			offset=1.0,
			model_z=2.0,
			grad_mag=0.5,
			runtime_sign=1,
			mutate_ext_xyz=above_ext,
		)

		self.assertAlmostEqual(float(loss_pos.detach()), 0.0, places=6)
		self.assertAlmostEqual(float(loss_neg.detach()), 0.0, places=6)
		self.assertAlmostEqual(stats_pos["snaps_map_snap_abs"], 0.0, places=6)
		self.assertAlmostEqual(stats_neg["snaps_map_snap_abs"], 0.0, places=6)
		self.assertGreater(stats_wrong["snaps_map_snap_abs"], 1.9)

	def test_snap_loss_nonzero_offset_sanitizes_nan_external_samples(self) -> None:
		data = _RejectNonFiniteGradData(0.5)

		loss, stats, _out = self._snap_loss_case(
			offset=1.0,
			model_z=2.0,
			data=data,
			mutate_ext_xyz=lambda ext_xyz: ext_xyz.__setitem__((1, 2, 0), float("nan")),
		)

		self.assertGreater(len(data.queries), 0)
		self.assertTrue(bool(torch.isfinite(data.queries[0]).all()))
		self.assertAlmostEqual(float(loss.detach()), 0.0, places=6)
		self.assertEqual(stats["snaps_map_snap_samples"], 24.0)

	def test_snap_loss_nonzero_offset_sanitizes_invalid_model_samples(self) -> None:
		data = _RejectNonFiniteGradData(0.5)

		loss, stats, _out = self._snap_loss_case(
			offset=1.0,
			model_z=2.0,
			data=data,
			mutate_model_xyz=lambda model_xyz: model_xyz.__setitem__((0, 2, 2, 2), float("nan")),
		)

		self.assertGreater(len(data.queries), 0)
		self.assertTrue(bool(torch.isfinite(data.queries[0]).all()))
		self.assertAlmostEqual(float(loss.detach()), 0.0, places=6)
		self.assertLess(stats["snaps_map_snap_samples"], 25.0)
		self.assertGreater(stats["snaps_map_snap_samples"], 0.0)

	def test_lasagna_map_stage_uses_global_map_loss_weights(self) -> None:
		stages = optimizer.load_stages_cfg({
			"base": {
				"map_dist": 2.0,
				"map_smooth": 0.25,
				"map_station_t": 0.5,
			},
			"stages": [
				{
					"name": "map",
					"steps": 1,
					"lr": 0.01,
					"params": ["map_surf_ms"],
					"w_fac": {"map_dist": 0.5},
				},
			],
		})

		opt = stages[0].global_opt
		self.assertEqual(opt.kind, "map")
		self.assertEqual(opt.eff["map_dist"], 1.0)
		self.assertEqual(opt.eff["map_smooth"], 0.25)
		self.assertEqual(opt.eff["map_station_t"], 0.5)

	def test_lasagna_map_stage_rejects_model_loss_weight_override(self) -> None:
		with self.assertRaisesRegex(ValueError, "map stages may only override map loss"):
			optimizer.load_stages_cfg({
				"stages": [
					{
						"name": "map",
						"steps": 1,
						"params": ["map_surf_affine"],
						"w_fac": {"smooth": 0.0},
					},
				],
			})

	def test_global_map_stage_dict_w_fac_multiplies_base_weights(self) -> None:
		stage = parse_global_map_stage_item({
			"name": "map",
			"steps": 1,
			"params": ["map_surf_ms"],
			"w_fac": {"dist": 0.5, "smooth": 3.0, "map_station_t": 0.25},
		})
		cfg = GlobalMapConfig(
			base={
				"map_station_t": 0.8,
				"map_init": {
					"w_dist": 2.0,
					"w_smooth": 0.25,
					"w_jac": 4.0,
				},
			},
			stages=(stage,),
		)
		stage_cfg = _stage_loss_cfg(snap_surf_config_from_global_config(cfg, stage), stage)

		self.assertEqual(stage_cfg.map_init.w_dist, 1.0)
		self.assertEqual(stage_cfg.map_init.w_smooth, 0.75)
		self.assertEqual(stage_cfg.map_init.w_jac, 4.0)
		self.assertEqual(_stage_station_weight(cfg, stage), 0.2)

	def test_global_map_stage_numeric_w_fac_scales_station_weight(self) -> None:
		stage = parse_global_map_stage_item({
			"name": "map",
			"steps": 1,
			"params": ["map_surf_affine"],
			"w_fac": 0.5,
		})
		cfg = GlobalMapConfig(base={"map_station_t": 0.8}, stages=(stage,))

		self.assertEqual(_stage_station_weight(cfg, stage), 0.4)

	def test_global_map_stage_dict_w_fac_rejects_unknown_terms(self) -> None:
		with self.assertRaisesRegex(ValueError, "unknown term"):
			parse_global_map_stage_item({
				"name": "map",
				"steps": 1,
				"params": ["map_surf_ms"],
				"w_fac": {"not_a_loss": 1.0},
			})

	def test_lr_autoscale_scales_inverse_to_recent_max_loss(self) -> None:
		p = torch.nn.Parameter(torch.tensor([0.0]))
		opt = torch.optim.Adam([p], lr=0.2)
		cfg = _lr_autoscale_config({"lr_autoscale": True, "lr_autoscale_window": 2})
		state = _LrAutoscaleState(cfg)

		_apply_optimizer_lr_schedule(
			opt,
			step1=1,
			warmup_steps=0,
			autoscale=state,
			loss=torch.tensor(2.0),
		)
		self.assertAlmostEqual(float(opt.param_groups[0]["lr"]), 0.1)

		_apply_optimizer_lr_schedule(
			opt,
			step1=2,
			warmup_steps=0,
			autoscale=state,
			loss=torch.tensor(0.5),
		)
		self.assertAlmostEqual(float(opt.param_groups[0]["lr"]), 0.1)

		_apply_optimizer_lr_schedule(
			opt,
			step1=3,
			warmup_steps=0,
			autoscale=state,
			loss=torch.tensor(0.5),
		)
		self.assertAlmostEqual(float(opt.param_groups[0]["lr"]), 0.4)

	def test_lr_autoscale_combines_with_warmup(self) -> None:
		p = torch.nn.Parameter(torch.tensor([0.0]))
		opt = torch.optim.Adam([p], lr=0.2)
		state = _LrAutoscaleState(_lr_autoscale_config({"lr_autoscale": {"enabled": True, "window": 10}}))

		_apply_optimizer_lr_schedule(
			opt,
			step1=2,
			warmup_steps=4,
			autoscale=state,
			loss=torch.tensor(0.5),
		)

		self.assertAlmostEqual(float(opt.param_groups[0]["lr"]), 0.2)

	def test_lasagna_stage_parser_rejects_plain_affine_and_mixed_params(self) -> None:
		with self.assertRaisesRegex(ValueError, "map_surf_affine"):
			optimizer.load_stages_cfg({"stages": [{"name": "bad", "params": ["affine"]}]})
		with self.assertRaisesRegex(ValueError, "map_surf_affine"):
			optimizer.load_stages_cfg({"stages": [{"name": "bad", "params": ["map_affine"]}]})
		with self.assertRaisesRegex(ValueError, "map_surf_ms"):
			optimizer.load_stages_cfg({"stages": [{"name": "bad", "params": ["map_uv_ms"]}]})
		with self.assertRaisesRegex(ValueError, "cannot mix model params"):
			optimizer.load_stages_cfg({"stages": [{"name": "bad", "params": ["mesh_ms", "map_surf_ms"]}]})

	def test_nested_map_opt_uses_normal_lasagna_param_names(self) -> None:
		stage = parse_global_map_stage_item(
			{"steps": 1, "lr": 0.01, "params": ["map_surf_affine"], "args": {"subdiv": 2}},
			normal_lasagna=True,
		)
		self.assertEqual(stage.params, ("affine",))
		with self.assertRaisesRegex(ValueError, "map_surf_affine"):
			parse_global_map_stage_item({"params": ["affine"]}, normal_lasagna=True)
		with self.assertRaisesRegex(ValueError, "map_surf_ms"):
			parse_global_map_stage_item({"params": ["map_uv_ms"]}, normal_lasagna=True)

	def test_live_runtime_exports_objective_terms_without_reference_distance(self) -> None:
		with tempfile.TemporaryDirectory() as tmp:
			cfg = parse_global_map_config(_write_config(tmp, affine_steps=0, map_steps=0))
			runtime = GlobalMapRuntime(base=cfg.base)
			h, w = 5, 5
			model_xyz = _plane_xyz(h=h + 2, w=w + 2, z=0.0).unsqueeze(0)
			ext_xyz = _plane_xyz(h=h, w=w, z=0.0, offset_h=0.25, offset_w=0.5)
			model_valid = torch.ones(1, h + 2, w + 2, dtype=torch.bool)
			ext_valid = torch.ones(h, w, dtype=torch.bool)
			ext_quad = torch.ones(h - 1, w - 1, dtype=torch.bool)
			model_normals = _normals_3d(1, h + 2, w + 2)
			ext_normals = _normals_2d(h, w)

			stats = runtime.run_stage(
				stage=GlobalMapStageConfig(steps=0, lr=0.01, params=("affine",), args={"subdiv": 2}),
				model_xyz=model_xyz,
				model_normals=model_normals,
				model_valid=model_valid,
				ext_xyz=ext_xyz,
				ext_valid=ext_valid,
				ext_normals=ext_normals,
				ext_quad_valid=ext_quad,
			)

			self.assertIn("snaps_map_dist", stats)
			self.assertIn("snaps_map_vec", stats)
			self.assertIn("snaps_map_norm", stats)
			self.assertNotIn("snaps_map_avg", stats)
			self.assertNotIn("snaps_map_max", stats)

	def test_live_runtime_disables_z_lift_when_requested(self) -> None:
		runtime = GlobalMapRuntime(base={}, seed_xyz=(2.0, 2.0, 0.0))
		h, w = 5, 5
		model_xyz = _plane_xyz(h=h + 2, w=w + 2, z=0.0).unsqueeze(0)
		ext_xyz = _plane_xyz(h=h, w=w, z=0.0, offset_h=0.25, offset_w=0.5)
		model_valid = torch.ones(1, h + 2, w + 2, dtype=torch.bool)
		ext_valid = torch.ones(h, w, dtype=torch.bool)
		ext_quad = torch.ones(h - 1, w - 1, dtype=torch.bool)
		model_normals = _xy_normals_like((1, h + 2, w + 2))
		ext_normals = _xy_normals_like((h, w))

		stats = runtime.run_stage(
			stage=GlobalMapStageConfig(
				steps=0,
				lr=0.0,
				params=("affine",),
				args={"subdiv": 1, "z_lift_norm_xy_min": 0.01, "w_z_lift": 1.0, "disable_z_lift": True},
			),
			model_xyz=model_xyz,
			model_normals=model_normals,
			model_valid=model_valid,
			ext_xyz=ext_xyz,
			ext_valid=ext_valid,
			ext_normals=ext_normals,
			ext_quad_valid=ext_quad,
			external_surface_index=0,
			mesh_epoch=0,
		)

		self.assertEqual(stats["snaps_map_turn"], 0.0)
		self.assertEqual(stats["snaps_map_turn_smp"], 0.0)
		self.assertEqual(stats["snaps_map_zext_cache_miss"], 0.0)
		self.assertEqual(stats["snaps_map_zmdl_cache_miss"], 0.0)
		self.assertEqual(stats["snaps_map_startup_turn_model_ms"], 0.0)

	def test_live_runtime_auto_stop_callback_stops_map_stage(self) -> None:
		with tempfile.TemporaryDirectory() as tmp:
			cfg = parse_global_map_config(_write_config(tmp, affine_steps=0, map_steps=0))
			runtime = GlobalMapRuntime(base=cfg.base)
			h, w = 5, 5
			model_xyz = _plane_xyz(h=h + 2, w=w + 2, z=0.0).unsqueeze(0)
			ext_xyz = _plane_xyz(h=h, w=w, z=0.0, offset_h=0.25, offset_w=0.5)
			model_valid = torch.ones(1, h + 2, w + 2, dtype=torch.bool)
			ext_valid = torch.ones(h, w, dtype=torch.bool)
			ext_quad = torch.ones(h - 1, w - 1, dtype=torch.bool)
			model_normals = _normals_3d(1, h + 2, w + 2)
			ext_normals = _normals_2d(h, w)
			status_steps: list[int] = []

			def auto_stop(*, history: list[float], step: int) -> bool:
				return int(step) >= 3 and optimizer._auto_steps_should_stop(
					history,
					window=2,
					rel_threshold=1.0e-6,
				)

			stats = runtime.run_stage(
				stage=GlobalMapStageConfig(
					steps=10,
					lr=0.0,
					params=("map_uv_ms",),
					args={"subdiv": 2, "status_interval": 0},
				),
				model_xyz=model_xyz,
				model_normals=model_normals,
				model_valid=model_valid,
				ext_xyz=ext_xyz,
				ext_valid=ext_valid,
				ext_normals=ext_normals,
				ext_quad_valid=ext_quad,
				status_fn=lambda **kw: status_steps.append(int(kw["step"])),
				auto_stop_fn=auto_stop,
			)

			self.assertEqual(stats["snaps_map_stage_steps"], 3.0)
			self.assertEqual(stats["snaps_map_auto_stopped"], 1.0)
			self.assertEqual(status_steps, [0])
			self.assertNotIn(10, status_steps)

	def test_live_runtime_status_zero_reports_current_autoscaled_warmup_lr(self) -> None:
		with tempfile.TemporaryDirectory() as tmp:
			cfg = parse_global_map_config(_write_config(tmp, affine_steps=0, map_steps=0))
			runtime = GlobalMapRuntime(base=cfg.base)
			h, w = 5, 5
			model_xyz = _plane_xyz(h=h + 2, w=w + 2, z=0.0).unsqueeze(0)
			ext_xyz = _plane_xyz(h=h, w=w, z=0.0, offset_h=0.25, offset_w=0.5)
			model_valid = torch.ones(1, h + 2, w + 2, dtype=torch.bool)
			ext_valid = torch.ones(h, w, dtype=torch.bool)
			ext_quad = torch.ones(h - 1, w - 1, dtype=torch.bool)
			model_normals = _normals_3d(1, h + 2, w + 2)
			ext_normals = _normals_2d(h, w)
			rows: list[dict[str, float]] = []

			runtime.run_stage(
				stage=GlobalMapStageConfig(
					steps=1,
					lr=1.0,
					params=("map_uv_ms",),
					args={"subdiv": 2, "status_interval": 0, "lr_autoscale": True, "lr_warmup_steps": 2},
				),
				model_xyz=model_xyz,
				model_normals=model_normals,
				model_valid=model_valid,
				ext_xyz=ext_xyz,
				ext_valid=ext_valid,
				ext_normals=ext_normals,
				ext_quad_valid=ext_quad,
				status_fn=lambda **kw: rows.append(dict(kw["stats"])),
			)

			self.assertEqual(len(rows), 1)
			loss = rows[0]["snaps_map_loss"]
			expected_scale = 1.0 / loss
			self.assertGreater(loss, 0.0)
			self.assertAlmostEqual(rows[0]["snaps_map_lr_autoscale"], expected_scale, places=6)
			self.assertAlmostEqual(rows[0]["snaps_map_lr"], 0.5 * expected_scale, places=6)

	def test_live_runtime_auto_stop_counts_after_lr_warmup(self) -> None:
		with tempfile.TemporaryDirectory() as tmp:
			cfg = parse_global_map_config(_write_config(tmp, affine_steps=0, map_steps=0))
			runtime = GlobalMapRuntime(base=cfg.base)
			h, w = 5, 5
			model_xyz = _plane_xyz(h=h + 2, w=w + 2, z=0.0).unsqueeze(0)
			ext_xyz = _plane_xyz(h=h, w=w, z=0.0, offset_h=0.25, offset_w=0.5)
			model_valid = torch.ones(1, h + 2, w + 2, dtype=torch.bool)
			ext_valid = torch.ones(h, w, dtype=torch.bool)
			ext_quad = torch.ones(h - 1, w - 1, dtype=torch.bool)
			model_normals = _normals_3d(1, h + 2, w + 2)
			ext_normals = _normals_2d(h, w)
			auto_steps: list[int] = []

			def auto_stop(*, history: list[float], step: int) -> bool:
				auto_steps.append(int(step))
				return int(step) >= 3 and optimizer._auto_steps_should_stop(
					history,
					window=2,
					rel_threshold=1.0e-6,
				)

			stats = runtime.run_stage(
				stage=GlobalMapStageConfig(
					steps=10,
					lr=0.0,
					params=("map_uv_ms",),
					args={"subdiv": 2, "status_interval": 0, "lr_warmup_steps": 2},
				),
				model_xyz=model_xyz,
				model_normals=model_normals,
				model_valid=model_valid,
				ext_xyz=ext_xyz,
				ext_valid=ext_valid,
				ext_normals=ext_normals,
				ext_quad_valid=ext_quad,
				auto_stop_fn=auto_stop,
			)

			self.assertEqual(auto_steps, [1, 2, 3])
			self.assertEqual(stats["snaps_map_stage_steps"], 5.0)
			self.assertEqual(stats["snaps_map_auto_stopped"], 1.0)

	def test_cli_default_device_is_auto(self) -> None:
		parser = map_global_cli._build_parser()
		args = parser.parse_args(["optimize-fixture", "fixture", "cfg.json", "--out", "out"])
		self.assertEqual(args.device, "auto")

	def test_affine_outputs_full_size_raw_uv(self) -> None:
		model = AffineMapModel(ext_shape=(3, 4), device=torch.device("cpu"), dtype=torch.float32)
		with torch.no_grad():
			model.affine.copy_(torch.tensor([[2.0, 0.0, 10.0], [0.0, 3.0, 20.0]]))

		uv = model()

		self.assertEqual(tuple(uv.shape), (3, 4, 2))
		self.assertTrue(torch.allclose(uv[2, 3], torch.tensor([14.0, 29.0])))

	def test_global_pyramid_initializes_from_affine_output(self) -> None:
		affine = AffineMapModel(ext_shape=(5, 5), device=torch.device("cpu"), dtype=torch.float32)
		with torch.no_grad():
			affine.affine.copy_(torch.tensor([[1.0, 0.0, 0.25], [0.0, 1.0, 0.5]]))

		global_model = GlobalMapModel(affine().detach(), levels=3)

		self.assertTrue(torch.allclose(global_model(active_level=0), affine(), atol=1.0e-6))

	def test_affine_multistart_candidates_keep_seed_anchor(self) -> None:
		seed_ext = torch.tensor([10.0, 20.0])
		seed_uv = torch.tensor([3.0, 4.0])

		candidates = _affine_multistart_candidates(
			seed_ext_hw=seed_ext,
			seed_model_uv=seed_uv,
			rot_deg=[-15.0, 0.0, 15.0],
			scales=[0.5, 1.0],
		)

		self.assertEqual(len(candidates), 6)
		for _idx, _rot, _scale, affine in candidates:
			mapped = affine[:, :2] @ seed_ext + affine[:, 2]
			self.assertTrue(torch.allclose(mapped, seed_uv, atol=1.0e-6))

	def test_affine_seed_grid_candidates_perturb_seed_affine_and_keep_anchor(self) -> None:
		seed_ext = torch.tensor([2.0, 3.0])
		base = torch.tensor([[2.0, 0.0, 5.0], [0.0, 0.5, -1.0]])
		seed_uv = base[:, :2] @ seed_ext + base[:, 2]

		candidates = _affine_seed_grid_candidates(
			base_affine=base,
			seed_ext_hw=seed_ext,
			rot_deg=[0.0, 90.0],
			scales=[1.0, 2.0],
		)

		self.assertEqual(len(candidates), 4)
		for _idx, _rot, _scale, affine in candidates:
			mapped = affine[:, :2] @ seed_ext + affine[:, 2]
			self.assertTrue(torch.allclose(mapped, seed_uv, atol=1.0e-6))
		_idx, rot, scale, affine = candidates[2]
		self.assertEqual(rot, 0.0)
		self.assertEqual(scale, 2.0)
		self.assertTrue(torch.allclose(affine[:, :2], 2.0 * base[:, :2], atol=1.0e-6))

	def test_affine_seed_grid_selects_best_raw_candidate(self) -> None:
		with tempfile.TemporaryDirectory() as tmp:
			fixture_dir = os.path.join(tmp, "fixture")
			_write_planar_global_fixture(fixture_dir)
			fixture = load_map_fixture(fixture_dir, device=torch.device("cpu"))
			cfg_global = parse_global_map_config(_write_config(tmp, affine_steps=0, map_steps=0))
			stage = GlobalMapStageConfig(
				name="affine_seed_quad_init",
				steps=0,
				lr=0.0,
				params=("affine",),
				args={"affine_seed_quad_init": {"grid_search": {"enabled": True, "rot_deg": [0.0], "scales": [1.0, 2.0]}}},
			)
			stage_cfg = _stage_loss_cfg(snap_surf_config_from_global_config(cfg_global, stage), stage)
			seed_hw = torch.tensor([2.0, 2.0])
			seed_uv = torch.tensor([2.25, 2.5])
			linear = torch.eye(2) * 0.5
			base_affine = torch.cat([linear, (seed_uv - linear @ seed_hw).view(2, 1)], dim=1)
			affine_model = AffineMapModel(
				ext_shape=tuple(int(v) for v in fixture.ext_xyz.shape[:2]),
				device=torch.device("cpu"),
				dtype=torch.float32,
				initial=base_affine,
			)
			stdout = StringIO()

			with redirect_stdout(stdout):
				best, best_loss = _select_affine_seed_grid_candidate(
					base_affine=base_affine,
					stage=stage,
					affine_model=affine_model,
					fixture=fixture,
					stage_cfg=stage_cfg,
					seed_hw=seed_hw,
					station_target=seed_uv,
					w_station=0.0,
				)

			text = stdout.getvalue()
			self.assertIn("affine seed quad grid candidates=2", text)
			self.assertIn("best_idx=0", text)
			self.assertLess(best_loss, 1.0e-6)
			self.assertTrue(torch.allclose(best[:, :2], torch.eye(2), atol=1.0e-5))

	def test_seed_quad_affine_uses_local_fixture_geometry(self) -> None:
		with tempfile.TemporaryDirectory() as tmp:
			_write_planar_global_fixture(tmp)
			fixture_json = Path(tmp, "fixture.json")
			meta = json.loads(fixture_json.read_text(encoding="utf-8"))
			meta["seed_model_quad"] = [0, 1, 1]
			fixture_json.write_text(json.dumps(meta), encoding="utf-8")
			from snap_surf.map_fixture_io import load_map_fixture

			fixture = load_map_fixture(tmp)
			cfg = snap_surf_config_from_global_config(parse_global_map_config(_write_config(tmp, affine_steps=0, map_steps=0)))
			seed_hw = torch.tensor([2.0, 2.0])
			seed_uv = torch.tensor([2.0, 2.0])

			affine = _affine_from_seed_ext_quads(
				fixture=fixture,
				stage_cfg=cfg,
				seed_hw=seed_hw,
				seed_model_uv=seed_uv,
			)

			self.assertIsNotNone(affine)
			assert affine is not None
			pred = torch.tensor([0.0, 0.0, 1.0]) @ affine.transpose(0, 1)
			self.assertTrue(torch.allclose(pred, torch.tensor([0.25, 0.5]), atol=1.0e-4))

	def test_seed_quad_affine_sign_ignores_which_side_model_is_on(self) -> None:
		with tempfile.TemporaryDirectory() as tmp:
			_write_planar_global_fixture(tmp)
			fixture_json = Path(tmp, "fixture.json")
			meta = json.loads(fixture_json.read_text(encoding="utf-8"))
			meta["seed_model_quad"] = [0, 1, 1]
			fixture_json.write_text(json.dumps(meta), encoding="utf-8")
			fixture = load_map_fixture(tmp)
			model_xyz = fixture.model_xyz.clone()
			model_xyz[..., 2] = -1.0
			fixture = replace(fixture, model_xyz=model_xyz)
			cfg = snap_surf_config_from_global_config(parse_global_map_config(_write_config(tmp, affine_steps=0, map_steps=0)))

			result = _seed_quad_affine_init_result(
				fixture=fixture,
				stage_cfg=cfg,
				seed_hw=torch.tensor([2.0, 2.0]),
				seed_model_uv=torch.tensor([4.0, 2.0]),
				raw={"grid_search": False},
			)

			self.assertIsNotNone(result)
			assert result is not None
			self.assertEqual(result.sign, 1)
			self.assertEqual(result.sampled_count, 256)
			self.assertGreaterEqual(result.kept_count, 3)
			self.assertLess(result.seed_vec, 1.0e-4)

	def test_seed_quad_affine_sign_aligns_model_normal_convention(self) -> None:
		with tempfile.TemporaryDirectory() as tmp:
			_write_planar_global_fixture(tmp)
			fixture_json = Path(tmp, "fixture.json")
			meta = json.loads(fixture_json.read_text(encoding="utf-8"))
			meta["seed_model_quad"] = [0, 1, 1]
			fixture_json.write_text(json.dumps(meta), encoding="utf-8")
			fixture = load_map_fixture(tmp)
			model_xyz = fixture.model_xyz.clone()
			model_xyz[..., 2] = 1.0
			fixture = replace(fixture, model_xyz=model_xyz, model_normals=-fixture.model_normals)
			cfg = snap_surf_config_from_global_config(parse_global_map_config(_write_config(tmp, affine_steps=0, map_steps=0)))

			result = _seed_quad_affine_init_result(
				fixture=fixture,
				stage_cfg=cfg,
				seed_hw=torch.tensor([2.0, 2.0]),
				seed_model_uv=torch.tensor([4.0, 2.0]),
				raw={"grid_search": False},
			)

			self.assertIsNotNone(result)
			assert result is not None
			self.assertEqual(result.sign, -1)
			self.assertLess(result.seed_norm, 1.0e-4)
			self.assertLess(result.seed_vec, 1.0e-4)

	def test_seed_quad_affine_expansion_rows_grow_to_full_ext_model(self) -> None:
		with tempfile.TemporaryDirectory() as tmp:
			_write_planar_global_fixture(tmp, h=20, w=20)
			fixture = load_map_fixture(tmp)
			cfg = snap_surf_config_from_global_config(parse_global_map_config(_write_config(tmp, affine_steps=0, map_steps=0)))
			affine = torch.tensor([[1.0, 0.0, 0.25], [0.0, 1.0, 0.5]], dtype=fixture.model_xyz.dtype)

			rows = _affine_seed_quad_expansion_rows(
				affine=affine,
				seed_ext_quad=(9, 9),
				fixture=fixture,
				stage_cfg=cfg,
				sign=1,
			)

			self.assertEqual([int(row["radius"]) for row in rows], [0, 8, 16])
			self.assertEqual([int(row["quads"]) for row in rows], [1, 197, 19 * 19])
			self.assertTrue(all(math.isfinite(float(row["loss"])) for row in rows))
			self.assertTrue(all(int(row["samples"]) > 0 for row in rows))

	def test_seed_quad_affine_expansion_reopt_runs_each_radius(self) -> None:
		with tempfile.TemporaryDirectory() as tmp:
			_write_planar_global_fixture(tmp, h=12, w=12)
			fixture = load_map_fixture(tmp)
			cfg = snap_surf_config_from_global_config(parse_global_map_config(_write_config(tmp, affine_steps=0, map_steps=0)))
			affine = AffineMapModel(
				ext_shape=tuple(int(v) for v in fixture.ext_xyz.shape[:2]),
				device=fixture.model_xyz.device,
				dtype=fixture.model_xyz.dtype,
				initial=torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=fixture.model_xyz.dtype),
			)
			initial = _affine_seed_quad_expansion_rows(
				affine=affine.affine.detach(),
				seed_ext_quad=(5, 5),
				fixture=fixture,
				stage_cfg=cfg,
				sign=1,
			)

			rows, progress_rows = _run_affine_seed_quad_expansion_reopt(
				affine=affine,
				fixture=fixture,
				stage_cfg=cfg,
				seed_ext_quad=(5, 5),
				seed_hw=torch.tensor([5.0, 5.0]),
				station_target=torch.tensor([5.25, 5.5]),
				w_station=0.0,
				steps=2,
				lr=0.05,
				status_interval=1,
				lr_warmup_steps=0,
				stage_idx=0,
				progress_widths_run=None,
				debug_obj_root=Path(tmp, "grow_debug"),
			)

			self.assertEqual([int(row["radius"]) for row in rows], [8, 8, 8])
			self.assertEqual(progress_rows, 0)
			self.assertEqual([int(row["iters"]) for row in rows], [0, 1, 2])
			self.assertAlmostEqual(float(rows[0]["loss"]), float(initial[1]["loss"]), places=5)
			self.assertEqual(float(rows[0]["loss_gain"]), 0.0)
			self.assertEqual(float(rows[0]["lr"]), 0.05)
			self.assertEqual(float(rows[1]["lr"]), 0.05)
			self.assertLess(float(rows[2]["loss"]), float(initial[1]["loss"]))
			self.assertGreater(float(rows[2]["loss_gain"]), 0.0)
			self.assertTrue(torch.isfinite(affine.affine).all())
			self.assertTrue(Path(tmp, "grow_debug", "rad_000008_init", "map_ext_to_model.obj").exists())
			self.assertTrue(Path(tmp, "grow_debug", "rad_000008_final", "map_ext_to_model.obj").exists())
			meta = json.loads(Path(tmp, "grow_debug", "rad_000008_final", "meta.json").read_text(encoding="utf-8"))
			self.assertEqual(meta["phase"], "final")
			self.assertEqual(meta["radius"], 8)

	def test_seed_quad_affine_expansion_reopt_uses_stage_lr_warmup(self) -> None:
		with tempfile.TemporaryDirectory() as tmp:
			_write_planar_global_fixture(tmp, h=12, w=12)
			fixture = load_map_fixture(tmp)
			cfg = snap_surf_config_from_global_config(parse_global_map_config(_write_config(tmp, affine_steps=0, map_steps=0)))
			initial = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=fixture.model_xyz.dtype)
			warm = AffineMapModel(
				ext_shape=tuple(int(v) for v in fixture.ext_xyz.shape[:2]),
				device=fixture.model_xyz.device,
				dtype=fixture.model_xyz.dtype,
				initial=initial,
			)
			half_lr = AffineMapModel(
				ext_shape=tuple(int(v) for v in fixture.ext_xyz.shape[:2]),
				device=fixture.model_xyz.device,
				dtype=fixture.model_xyz.dtype,
				initial=initial,
			)
			common = dict(
				fixture=fixture,
				stage_cfg=cfg,
				seed_ext_quad=(5, 5),
				seed_hw=torch.tensor([5.0, 5.0]),
				station_target=torch.tensor([5.25, 5.5]),
				w_station=0.0,
				steps=1,
				status_interval=1,
				stage_idx=0,
				progress_widths_run=None,
			)

			_run_affine_seed_quad_expansion_reopt(
				affine=warm,
				lr=0.05,
				lr_warmup_steps=2,
				**common,
			)
			rows, _progress_rows = _run_affine_seed_quad_expansion_reopt(
				affine=half_lr,
				lr=0.025,
				lr_warmup_steps=0,
				**common,
			)

			self.assertEqual(float(rows[1]["lr"]), 0.025)
			self.assertTrue(torch.allclose(warm.affine, half_lr.affine, atol=1.0e-6, rtol=1.0e-6))

	def test_live_runtime_seed_quad_init_uses_shared_expansion_reopt(self) -> None:
		with tempfile.TemporaryDirectory() as tmp:
			_write_planar_global_fixture(tmp)
			fixture = load_map_fixture(tmp)
			runtime = GlobalMapRuntime(seed_xyz=(2.0, 2.0, 0.0))
			stage = GlobalMapStageConfig(
				name="affine_seed_quad_init",
				steps=0,
				lr=0.05,
				params=("affine",),
				args={
					"debug_obj_dir": str(Path(tmp, "runtime_debug")),
					"lr_warmup_steps": 4,
					"affine_seed_quad_init": {"expansion_reopt": True, "expansion_reopt_steps": 1, "grid_search": False},
				},
			)
			stdout = StringIO()

			with redirect_stdout(stdout):
				stats = runtime.run_stage(
					stage=stage,
					model_xyz=fixture.model_xyz,
					model_normals=fixture.model_normals,
					model_valid=fixture.model_valid,
					ext_xyz=fixture.ext_xyz,
					ext_valid=fixture.ext_valid,
					ext_normals=fixture.ext_normals,
					ext_quad_valid=fixture.ext_quad_valid,
				)

			text = stdout.getvalue()
			self.assertIn("affine seed quad expansion reopt opts", text)
			self.assertIn("stage_lr=0.05", text)
			self.assertIn("lr=0.05", text)
			self.assertIn("steps_per_radius=1", text)
			self.assertIn("lr_warmup_steps=4", text)
			self.assertIn("grow-r8:1/1", text)
			self.assertIn("  0.0125", text)
			self.assertIn("start_radius=8", text)
			self.assertIn("weights=", text)
			self.assertIn("grow-r", text)
			self.assertIn("affine seed quad expansion debug objs dir=", text)
			self.assertIn("affine seed quad initial debug objs radius=128", text)
			self.assertTrue(Path(tmp, "runtime_debug", "map_global_affine_seed_quad_init", "expansion_reopt", "rad_000128_initial_filtered", "map_ext_to_model.obj").exists())
			self.assertTrue(Path(tmp, "runtime_debug", "map_global_affine_seed_quad_init", "expansion_reopt", "rad_000008_init", "map_ext_to_model.obj").exists())
			self.assertIn("snaps_map_loss", stats)
			self.assertEqual(runtime.sign, 1)

	def test_live_runtime_seed_quad_init_skips_expansion_reopt_by_default(self) -> None:
		with tempfile.TemporaryDirectory() as tmp:
			_write_planar_global_fixture(tmp)
			fixture = load_map_fixture(tmp)
			runtime = GlobalMapRuntime(seed_xyz=(2.0, 2.0, 0.0))
			stage = GlobalMapStageConfig(
				name="affine_seed_quad_init",
				steps=0,
				lr=0.05,
				params=("affine",),
				args={
					"debug_obj_dir": str(Path(tmp, "runtime_debug")),
					"affine_seed_quad_init": {"grid_search": False},
				},
			)
			stdout = StringIO()

			with redirect_stdout(stdout):
				runtime.run_stage(
					stage=stage,
					model_xyz=fixture.model_xyz,
					model_normals=fixture.model_normals,
					model_valid=fixture.model_valid,
					ext_xyz=fixture.ext_xyz,
					ext_valid=fixture.ext_valid,
					ext_normals=fixture.ext_normals,
					ext_quad_valid=fixture.ext_quad_valid,
				)

			text = stdout.getvalue()
			root = Path(tmp, "runtime_debug", "map_global_affine_seed_quad_init", "expansion_reopt")
			self.assertIn("affine seed quad initial debug objs radius=128", text)
			self.assertNotIn("affine seed quad expansion reopt opts", text)
			self.assertNotIn("grow-r", text)
			self.assertTrue((root / "rad_000128_initial_filtered" / "map_ext_to_model.obj").exists())
			self.assertFalse((root / "rad_000008_init").exists())

	def test_expansion_debug_objs_are_scoped_to_active_radius(self) -> None:
		with tempfile.TemporaryDirectory() as tmp:
			uv = _write_planar_global_fixture(tmp, h=5, w=5)
			fixture = load_map_fixture(tmp)
			active = torch.zeros(4, 4, dtype=torch.bool)
			active[1, 1] = True
			out = Path(tmp, "objs")

			_write_map_objs(
				out,
				uv=uv,
				fixture=fixture,
				meta={"phase": "test"},
				active_quad=active,
			)

			def count_prefix(path: Path, prefix: str) -> int:
				return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.startswith(prefix))

			self.assertEqual(count_prefix(out / "ext_surface.obj", "v "), 4)
			self.assertEqual(count_prefix(out / "ext_surface.obj", "f "), 1)
			self.assertEqual(count_prefix(out / "map_ext_to_model.obj", "l "), 4)
			self.assertLess(count_prefix(out / "model_surface.obj", "v "), int(fixture.model_valid.sum()))
			meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
			self.assertEqual(meta["valid_vectors"], 4)

	def test_external_quad_health_filter_rejects_spike_artifact(self) -> None:
		with tempfile.TemporaryDirectory() as tmp:
			_write_planar_global_fixture(tmp, h=6, w=6)
			fixture = load_map_fixture(tmp)
			ext_xyz = fixture.ext_xyz.clone()
			ext_xyz[2, 2, 2] = 1000.0
			fixture = replace(fixture, ext_xyz=ext_xyz)
			cfg = snap_surf_config_from_global_config(
				GlobalMapConfig(
					base={
						"map_init": {
							"ext_mesh_health_filter": True,
							"ext_mesh_health_max_edge_ratio": 10.0,
						}
					}
				)
			)
			stdout = StringIO()

			with redirect_stdout(stdout):
				filtered = _apply_external_quad_health_filter(fixture, cfg, label="debug_values")

			active = _full_active_quad(filtered)
			self.assertLess(int(active.sum()), int(_full_active_quad(fixture).sum()))
			self.assertFalse(bool(active[1, 1]))
			self.assertFalse(bool(active[1, 2]))
			self.assertFalse(bool(active[2, 1]))
			self.assertFalse(bool(active[2, 2]))
			self.assertGreater(filtered.metadata["ext_mesh_health"]["quads_rejected"], 0.0)
			self.assertGreater(filtered.metadata["ext_mesh_health"]["quads_rejected_padding"], 0.0)
			out_text = stdout.getvalue()
			self.assertIn("external quad health filter debug_values", out_text)
			self.assertIn("external quad health reject values debug_values", out_text)
			self.assertIn("aspect count=", out_text)
			self.assertIn("values=", out_text)

	def test_external_quad_health_filter_rejects_twisted_quad(self) -> None:
		with tempfile.TemporaryDirectory() as tmp:
			_write_planar_global_fixture(tmp, h=3, w=3)
			fixture = load_map_fixture(tmp)
			ext_xyz = fixture.ext_xyz.clone()
			ext_xyz[1, 1] = ext_xyz[0, 0]
			fixture = replace(fixture, ext_xyz=ext_xyz)
			cfg = snap_surf_config_from_global_config(
				GlobalMapConfig(
					base={
						"map_init": {
							"ext_mesh_health_filter": True,
							"ext_mesh_health_min_triangle_normal_dot": 0.5,
						}
					}
				)
			)

			filtered = _apply_external_quad_health_filter(fixture, cfg, label="test")

			active = _full_active_quad(filtered)
			self.assertFalse(bool(active[0, 0]))
			self.assertGreater(filtered.metadata["ext_mesh_health"]["quads_rejected"], 0.0)
			self.assertIn("min_triangle_normal_dot", filtered.metadata["ext_mesh_health"])

	def test_external_quad_health_filter_removes_disconnected_islands(self) -> None:
		with tempfile.TemporaryDirectory() as tmp:
			_write_planar_global_fixture(tmp, h=6, w=6)
			fixture = load_map_fixture(tmp)
			ext_quad_valid = fixture.ext_quad_valid.clone()
			ext_quad_valid[:, 3] = False
			fixture = replace(fixture, ext_quad_valid=ext_quad_valid)
			cfg = snap_surf_config_from_global_config(
				GlobalMapConfig(
					base={
						"map_init": {
							"ext_mesh_health_filter": True,
							"ext_mesh_health_reject_radius": 0,
						}
					}
				)
			)

			filtered = _apply_external_quad_health_filter(fixture, cfg, label="test")

			active = _full_active_quad(filtered)
			self.assertTrue(bool(active[2, 2]))
			self.assertFalse(bool(active[2, 4]))
			self.assertEqual(int(active[:, 4].sum()), 0)
			self.assertGreater(filtered.metadata["ext_mesh_health"]["quads_rejected_disconnected"], 0.0)
			self.assertEqual(
				int(filtered.metadata["ext_mesh_health"]["quads_connected_kept"]),
				int(active.sum()),
			)

	def test_config_parses_stage_args(self) -> None:
		with tempfile.TemporaryDirectory() as tmp:
			cfg = parse_global_map_config(_write_config(tmp, affine_steps=1, map_steps=1))
			self.assertEqual(cfg.stages[1].params, ("map_uv_ms",))
			self.assertEqual(cfg.stages[1].min_scaledown, 3)
			parsed = snap_surf_config_from_global_config(cfg, cfg.stages[1])
			self.assertEqual(parsed.map_init.subdiv, 2)
			path = Path(tmp, "named_cfg.json")
			path.write_text(json.dumps({"stages": [{"name": "affine_init_scan", "params": ["map_surf_affine"]}]}), encoding="utf-8")
			named = parse_global_map_config(path)
			self.assertEqual(named.stages[0].name, "affine_init_scan")

	def test_global_map_config_translates_lasagna_weight_aliases(self) -> None:
		stage = GlobalMapStageConfig(params=("map_uv_ms",))
		cfg = GlobalMapConfig(
			base={
				"map_dist": 0.0001,
				"map_vec_normal": 2.0,
				"map_surface_normal": 3.0,
				"map_smooth": 4.0,
				"map_bend": 5.0,
				"map_jac": 6.0,
				"map_metric_smooth": 7.0,
				"map_area_smooth": 8.0,
				"map_dense_prior": 9.0,
			},
			stages=(stage,),
		)

		parsed = snap_surf_config_from_global_config(cfg, stage)

		self.assertEqual(parsed.map_init.w_dist, 0.0001)
		self.assertEqual(parsed.map_init.w_vec_normal, 2.0)
		self.assertEqual(parsed.map_init.w_surface_normal, 3.0)
		self.assertEqual(parsed.map_init.w_smooth, 4.0)
		self.assertEqual(parsed.map_init.w_bend, 5.0)
		self.assertEqual(parsed.map_init.w_jac, 6.0)
		self.assertEqual(parsed.map_init.w_metric_smooth, 7.0)
		self.assertEqual(parsed.map_init.w_area_smooth, 8.0)
		self.assertEqual(parsed.map_init.w_dense_prior, 9.0)

	def test_out_of_bounds_samples_are_skipped_without_clamping(self) -> None:
		with tempfile.TemporaryDirectory() as tmp:
			_write_planar_global_fixture(tmp)
			from snap_surf.map_fixture_io import load_map_fixture

			fixture = load_map_fixture(tmp)
			cfg = snap_surf_config_from_global_config(parse_global_map_config(_write_config(tmp, affine_steps=0, map_steps=0)))
			uv = torch.full((5, 5, 2), -10.0)

			loss, terms = _objective_for_uv(uv=uv, fixture=fixture, cfg=cfg, level=0)

			self.assertTrue(torch.isfinite(loss))
			self.assertEqual(float(terms["samples"]), 0.0)
			self.assertTrue(torch.equal(uv, torch.full((5, 5, 2), -10.0)))

	def test_fixture_cli_writes_final_outputs_without_debug_objs_by_default(self) -> None:
		with tempfile.TemporaryDirectory() as tmp:
			fixture_dir = os.path.join(tmp, "fixture")
			out_dir = os.path.join(tmp, "out")
			_write_planar_global_fixture(fixture_dir)
			cfg_path = _write_config(tmp)

			rc = map_global_cli.main(["optimize-fixture", fixture_dir, cfg_path, "--out", out_dir, "--device", "cpu"])

			self.assertEqual(rc, 0)
			self.assertTrue(os.path.exists(os.path.join(out_dir, "model_x.tif")))
			self.assertTrue(os.path.exists(os.path.join(out_dir, "model_y.tif")))
			self.assertTrue(os.path.exists(os.path.join(out_dir, "meta.json")))
			self.assertTrue(os.path.exists(os.path.join(out_dir, "metrics.json")))
			self.assertFalse(os.path.exists(os.path.join(out_dir, "objs")))
			self.assertFalse(os.path.exists(os.path.join(out_dir, "active_quad.tif")))
			self.assertFalse(os.path.exists(os.path.join(out_dir, "blocked_quad.tif")))
			metrics = json.loads(Path(out_dir, "metrics.json").read_text(encoding="utf-8"))
			self.assertEqual(metrics["common_vertices"], 25)
			self.assertIn("avg_model_quad_distance", metrics)

	def test_runtime_stage_exports_fixture_once_from_stage_args(self) -> None:
		with tempfile.TemporaryDirectory() as tmp:
			fixture_dir = os.path.join(tmp, "fixture")
			export_dir = os.path.join(tmp, "exported_fixture")
			_write_planar_global_fixture(fixture_dir, h=6, w=6)
			fixture = load_map_fixture(fixture_dir)
			ext_xyz = fixture.ext_xyz.clone()
			ext_xyz[2, 2, 2] = 1000.0
			fixture = replace(fixture, ext_xyz=ext_xyz)
			runtime = GlobalMapRuntime(
				seed_xyz=(2.0, 2.0, 0.0),
				base={
					"map_init": {
						"ext_mesh_health_filter": True,
						"ext_mesh_health_reject_radius": 0,
					},
				},
			)
			stage = GlobalMapStageConfig(
				name="export_init",
				steps=0,
				lr=0.05,
				params=("affine",),
				args={
					"subdiv": 1,
					"fixture_export_dir": export_dir,
					"fixture_export_once": True,
					"fixture_export_objs": False,
				},
			)

			stats1 = runtime.run_stage(
				stage=stage,
				model_xyz=fixture.model_xyz,
				model_normals=fixture.model_normals,
				model_valid=fixture.model_valid,
				ext_xyz=fixture.ext_xyz,
				ext_valid=fixture.ext_valid,
				ext_normals=fixture.ext_normals,
				ext_quad_valid=fixture.ext_quad_valid,
			)
			stats2 = runtime.run_stage(
				stage=stage,
				model_xyz=fixture.model_xyz,
				model_normals=fixture.model_normals,
				model_valid=fixture.model_valid,
				ext_xyz=fixture.ext_xyz,
				ext_valid=fixture.ext_valid,
				ext_normals=fixture.ext_normals,
				ext_quad_valid=fixture.ext_quad_valid,
			)

			self.assertEqual(stats1["snaps_map_fixture_exported"], 1.0)
			self.assertEqual(stats2["snaps_map_fixture_exported"], 0.0)
			self.assertTrue(Path(export_dir, "fixture.json").exists())
			self.assertTrue(Path(export_dir, "map", "model_x.tif").exists())
			self.assertTrue(Path(export_dir, "ext_surface", "x.tif").exists())
			self.assertFalse(Path(export_dir, "objs").exists())
			exported = load_map_fixture(export_dir)
			self.assertEqual(tuple(exported.reference_uv.shape), tuple(fixture.reference_uv.shape))
			self.assertEqual(int(exported.ext_quad_valid.sum()), int(fixture.ext_quad_valid.sum()))
			self.assertLess(int(exported.reference_active_quad.sum()), int(_full_active_quad(fixture).sum()))

	def test_fixture_cli_stage_export_writes_fixture_under_out(self) -> None:
		with tempfile.TemporaryDirectory() as tmp:
			fixture_dir = os.path.join(tmp, "fixture")
			out_dir = os.path.join(tmp, "out")
			_write_planar_global_fixture(fixture_dir)
			cfg_path = _write_config(tmp, affine_steps=1, map_steps=1)
			cfg_raw = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
			cfg_raw["stages"][1]["args"]["export_fixture"] = {
				"dir": "rerun_fixture",
				"once": True,
				"objs": True,
			}
			Path(cfg_path).write_text(json.dumps(cfg_raw), encoding="utf-8")

			rc = map_global_cli.main(["benchmark-fixture", fixture_dir, cfg_path, "--out", out_dir, "--device", "cpu"])

			self.assertEqual(rc, 0)
			export_dir = Path(out_dir, "rerun_fixture")
			self.assertTrue(Path(export_dir, "fixture.json").exists())
			self.assertTrue(Path(export_dir, "map", "model_x.tif").exists())
			self.assertTrue(Path(export_dir, "ext_surface", "x.tif").exists())
			self.assertTrue(Path(export_dir, "model_stack", "x.tif").exists())
			self.assertTrue(Path(export_dir, "objs", "map_ext_to_model.obj").exists())
			exported = load_map_fixture(export_dir)
			source = load_map_fixture(fixture_dir)
			self.assertEqual(tuple(exported.reference_uv.shape), tuple(source.reference_uv.shape))

	def test_fixture_cli_writes_debug_objs_when_enabled(self) -> None:
		with tempfile.TemporaryDirectory() as tmp:
			fixture_dir = os.path.join(tmp, "fixture")
			out_dir = os.path.join(tmp, "out")
			_write_planar_global_fixture(fixture_dir)
			cfg_path = _write_config(tmp, write_objs=True)

			rc = map_global_cli.main(["optimize-fixture", fixture_dir, cfg_path, "--out", out_dir, "--device", "cpu"])

			self.assertEqual(rc, 0)
			self.assertTrue(os.path.exists(os.path.join(out_dir, "objs", "stage_000_map_surf_affine", "map_ext_to_model.obj")))
			self.assertTrue(os.path.exists(os.path.join(out_dir, "objs", "stage_001_map_surf_ms", "map_ext_to_model.obj")))
			self.assertTrue(os.path.exists(os.path.join(out_dir, "objs", "final", "map_ext_to_model.obj")))
			self.assertTrue(os.path.exists(os.path.join(out_dir, "objs", "final", "map_ext_to_model_worst_1pct.obj")))
			final_meta = json.loads(Path(out_dir, "objs", "final", "meta.json").read_text(encoding="utf-8"))
			self.assertEqual(final_meta["name"], "final")
			self.assertIn("worst_1pct_vectors", final_meta)
			metrics = json.loads(Path(out_dir, "metrics.json").read_text(encoding="utf-8"))
			self.assertIn("avg_model_quad_distance", metrics["history"][-1])
			self.assertLess(metrics["model_l2_max_delta"], 1.0)

	def test_benchmark_fixture_cli_writes_json_and_passes(self) -> None:
		with tempfile.TemporaryDirectory() as tmp:
			fixture_dir = os.path.join(tmp, "fixture")
			out_dir = os.path.join(tmp, "out")
			_write_planar_global_fixture(fixture_dir)
			cfg_path = _write_config(tmp, affine_steps=2, map_steps=2)

			rc = map_global_cli.main(["benchmark-fixture", fixture_dir, cfg_path, "--out", out_dir, "--device", "cpu"])

			self.assertEqual(rc, 0)
			bench = json.loads(Path(out_dir, "benchmark.json").read_text(encoding="utf-8"))
			self.assertEqual(bench["status"], "pass")
			self.assertTrue(bench["passed"])
			self.assertIn("elapsed_s", bench)
			self.assertIn("optimizer_metrics", bench)
			self.assertEqual(bench["map_deltas"]["common_vertices"], 25)
			self.assertEqual(bench["map_deltas"]["active_quad_diff"], 0)

	def test_benchmark_fixture_cli_strict_threshold_fails(self) -> None:
		with tempfile.TemporaryDirectory() as tmp:
			fixture_dir = os.path.join(tmp, "fixture")
			out_dir = os.path.join(tmp, "out")
			_write_planar_global_fixture(fixture_dir)
			cfg_path = _write_config(tmp, affine_steps=1, map_steps=0)

			rc = map_global_cli.main([
				"benchmark-fixture",
				fixture_dir,
				cfg_path,
				"--out",
				out_dir,
				"--device",
				"cpu",
				"--max-model-abs-delta",
				"1e-12",
				"--max-model-l2-delta",
				"1e-12",
			])

			bench = json.loads(Path(out_dir, "benchmark.json").read_text(encoding="utf-8"))
			if rc == 0:
				self.assertLessEqual(bench["map_deltas"]["model_l2_max_delta"], 1.0e-12)
			else:
				self.assertEqual(rc, 1)
				self.assertEqual(bench["status"], "fail")
				self.assertFalse(bench["passed"])

	def test_benchmark_fixture_cli_accepts_explicit_reference_map_dir(self) -> None:
		with tempfile.TemporaryDirectory() as tmp:
			fixture_dir = os.path.join(tmp, "fixture")
			out_dir = os.path.join(tmp, "out")
			_write_planar_global_fixture(fixture_dir)
			cfg_path = _write_config(tmp, affine_steps=2, map_steps=2)

			rc = map_global_cli.main([
				"benchmark-fixture",
				fixture_dir,
				cfg_path,
				"--out",
				out_dir,
				"--device",
				"cpu",
				"--reference-dir",
				os.path.join(fixture_dir, "map"),
			])

			self.assertEqual(rc, 0)
			bench = json.loads(Path(out_dir, "benchmark.json").read_text(encoding="utf-8"))
			self.assertEqual(bench["status"], "pass")
			self.assertEqual(bench["reference_dir"], os.path.join(fixture_dir, "map"))

	def test_benchmark_fixture_cli_profiles_components_without_changing_parameters(self) -> None:
		with tempfile.TemporaryDirectory() as tmp:
			fixture_dir = os.path.join(tmp, "fixture")
			out_dir = os.path.join(tmp, "out")
			_write_planar_global_fixture(fixture_dir)
			cfg_path = _write_config(tmp, affine_steps=1, map_steps=1)

			rc = map_global_cli.main([
				"benchmark-fixture",
				fixture_dir,
				cfg_path,
				"--out",
				out_dir,
				"--device",
				"cpu",
				"--profile-components",
				"--profile-repeats",
				"1",
			])

			self.assertEqual(rc, 0)
			profile = json.loads(Path(out_dir, "profile_components.json").read_text(encoding="utf-8"))
			components = {row["component"] for row in profile["rows"]}
			self.assertIn("all", components)
			self.assertIn("dist", components)
			self.assertIn("jac", components)
			for row in profile["rows"]:
				self.assertGreaterEqual(row["mean_s"], 0.0)
			bench = json.loads(Path(out_dir, "benchmark.json").read_text(encoding="utf-8"))
			self.assertGreaterEqual(len(bench["profile_components"]), 3)

	def test_optimize_fixture_returns_full_rectangular_map(self) -> None:
		with tempfile.TemporaryDirectory() as tmp:
			fixture_dir = os.path.join(tmp, "fixture")
			out_dir = os.path.join(tmp, "out")
			_write_planar_global_fixture(fixture_dir, h=9, w=9)
			metrics = optimize_fixture(fixture_dir, _write_config(tmp, affine_steps=2, map_steps=2), out_dir=out_dir)

			self.assertEqual(metrics["common_vertices"], 81)
			import tifffile

			model_x = tifffile.imread(os.path.join(out_dir, "model_x.tif"))
			self.assertEqual(tuple(model_x.shape), (9, 9))

	def test_progress_default_status_interval_skips_middle_steps_and_prints_stage_zero(self) -> None:
		with tempfile.TemporaryDirectory() as tmp:
			fixture_dir = os.path.join(tmp, "fixture")
			out_dir = os.path.join(tmp, "out")
			_write_planar_global_fixture(fixture_dir)
			cfg_path = _write_config(tmp, affine_steps=3, map_steps=2)
			stdout = StringIO()

			with redirect_stdout(stdout):
				optimize_fixture(fixture_dir, cfg_path, out_dir=out_dir)

			text = stdout.getvalue()
			self.assertIn("0/3", text)
			self.assertIn("2/3", text)
			self.assertNotIn("1/3", text)
			self.assertNotIn("3/3", text)
			self.assertRegex(text, r"stage1 0/2\s+\S+\s+\S+")
			self.assertIn("sm_los", text)
			self.assertIn("map objective loss", text)
			self.assertIn("sm_dst", text)

	def test_affine_multistart_prints_all_candidate_rows(self) -> None:
		with tempfile.TemporaryDirectory() as tmp:
			fixture_dir = os.path.join(tmp, "fixture")
			out_dir = os.path.join(tmp, "out")
			_write_planar_global_fixture(fixture_dir)
			cfg_path = Path(_write_config(tmp, affine_steps=1, map_steps=0))
			cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
			cfg["stages"][0]["args"]["affine_multistart"] = {
				"enabled": True,
				"rot_deg": [0.0, 15.0],
				"scales": [1.0],
				"steps": 0,
			}
			cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
			stdout = StringIO()

			with redirect_stdout(stdout):
				optimize_fixture(fixture_dir, cfg_path, out_dir=out_dir)

			text = stdout.getvalue()
			self.assertIn("affine multistart progress columns", text)
			self.assertIn("affine multistart candidates=2", text)
			self.assertIn("i00", text)
			self.assertIn("sdet", text)
			self.assertIn("0.0000", text)
			self.assertIn("15.000", text)

	def test_named_affine_init_scan_runs_without_training_stage_steps(self) -> None:
		with tempfile.TemporaryDirectory() as tmp:
			fixture_dir = os.path.join(tmp, "fixture")
			out_dir = os.path.join(tmp, "out")
			_write_planar_global_fixture(fixture_dir)
			cfg_path = Path(tmp, "cfg.json")
			_write_json(
				cfg_path,
				{
					"base": {
						"write_objs": True,
						"map_station_t": 0.001,
						"map_init": {
							"subdiv": 2,
							"w_dist": 1.0,
							"w_vec_normal": 0.0,
							"w_surface_normal": 0.0,
							"w_smooth": 0.0,
							"w_bend": 0.0,
							"w_jac": 0.0,
							"w_metric_smooth": 0.0,
							"w_area_smooth": 0.0,
						},
					},
					"stages": [
						{
							"name": "affine_init_scan",
							"steps": 0,
								"lr": 0.05,
								"params": ["map_surf_affine"],
							"args": {
								"affine_multistart": {
									"enabled": True,
									"rot_deg": [0.0, 15.0],
									"scales": [1.0],
									"steps": 1,
								}
							},
						}
					],
				},
			)
			stdout = StringIO()

			with redirect_stdout(stdout):
				metrics = optimize_fixture(fixture_dir, cfg_path, out_dir=out_dir)

			text = stdout.getvalue()
			self.assertIn("affine multistart candidates=2", text)
			self.assertTrue(os.path.exists(os.path.join(out_dir, "objs", "stage_000_affine_init_scan", "map_ext_to_model.obj")))
			self.assertEqual(metrics["history"][0]["name"], "affine_init_scan")

	def test_named_affine_seed_quad_init_runs_single_seed_fit(self) -> None:
		with tempfile.TemporaryDirectory() as tmp:
			fixture_dir = os.path.join(tmp, "fixture")
			out_dir = os.path.join(tmp, "out")
			_write_planar_global_fixture(fixture_dir)
			fixture_json = Path(fixture_dir, "fixture.json")
			meta = json.loads(fixture_json.read_text(encoding="utf-8"))
			meta["seed_model_quad"] = [0, 1, 1]
			fixture_json.write_text(json.dumps(meta), encoding="utf-8")
			cfg_path = Path(tmp, "cfg.json")
			_write_json(
				cfg_path,
				{
					"base": {
						"write_objs": True,
						"map_station_t": 0.001,
						"map_init": {
							"subdiv": 2,
							"w_dist": 1.0,
							"w_vec_normal": 0.0,
							"w_surface_normal": 0.0,
							"w_smooth": 0.0,
							"w_bend": 0.0,
							"w_jac": 0.0,
							"w_metric_smooth": 0.0,
							"w_area_smooth": 0.0,
						},
					},
					"stages": [
						{
							"name": "affine_seed_quad_init",
							"steps": 1,
							"lr": 0.05,
								"params": ["map_surf_affine"],
							"args": {"affine_seed_quad_init": {"enabled": True}},
						}
					],
				},
			)
			stdout = StringIO()

			with redirect_stdout(stdout):
				metrics = optimize_fixture(fixture_dir, cfg_path, out_dir=out_dir)

			text = stdout.getvalue()
			self.assertIn("affine seed quad prepare", text)
			self.assertIn("0/1", text)
			self.assertNotIn("affine multistart candidates", text)
			self.assertTrue(os.path.exists(os.path.join(out_dir, "objs", "stage_000_affine_seed_quad_init", "map_ext_to_model.obj")))
			self.assertEqual(metrics["history"][0]["name"], "affine_seed_quad_init")


if __name__ == "__main__":
	unittest.main()
