from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO

import torch
import optimizer

TEST_DIR = os.path.dirname(__file__)
if TEST_DIR not in sys.path:
	sys.path.insert(0, TEST_DIR)

from snap_surf.map_fixture_io import _float_tif, _mask_tif, _write_json, _write_vector_dir
from snap_surf.map_global import (
	AffineMapModel,
	GlobalMapModel,
	_affine_from_seed_ext_quads,
	_affine_multistart_candidates,
	_objective_for_uv,
	optimize_fixture,
	parse_global_map_stage_item,
	parse_global_map_config,
	snap_surf_config_from_global_config,
)
from snap_surf import map_global_cli
from snap_surf_test_utils import _normals_2d, _normals_3d, _plane_xyz


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
			"normal_sign": 1,
			"orientation_sign": 1,
			"snap_surf_config": {"map_init": {"subdiv": 1}},
		},
	)
	return reference_uv


def _write_config(root: str, *, affine_steps: int = 8, map_steps: int = 8) -> str:
	path = Path(root, "cfg.json")
	_write_json(
		path,
		{
			"base": {
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
			},
			"stages": [
				{"steps": affine_steps, "lr": 0.05, "params": ["affine"], "args": {"subdiv": 2}},
				{
					"steps": map_steps,
					"lr": 0.02,
					"params": ["map_uv_ms"],
					"min_scaledown": 3,
					"args": {"subdiv": 2, "map_station_t": 0.001},
				},
			],
		},
	)
	return str(path)


class SnapSurfMapGlobalTest(unittest.TestCase):
	def test_lasagna_stage_parser_routes_model_and_map_params(self) -> None:
		cfg = {
			"base": {"snap_surf": 1.0},
			"stages": [
				{"name": "model", "steps": 1, "lr": 0.1, "params": ["mesh_ms"]},
				{"name": "map", "steps": 1, "lr": 0.01, "params": ["map_affine"], "w_fac": 1.0},
			],
		}

		stages = optimizer.load_stages_cfg(cfg)

		self.assertEqual(stages[0].global_opt.kind, "model")
		self.assertEqual(stages[1].global_opt.kind, "map")

	def test_lasagna_stage_parser_rejects_plain_affine_and_mixed_params(self) -> None:
		with self.assertRaisesRegex(ValueError, "map_affine"):
			optimizer.load_stages_cfg({"stages": [{"name": "bad", "params": ["affine"]}]})
		with self.assertRaisesRegex(ValueError, "cannot mix model params"):
			optimizer.load_stages_cfg({"stages": [{"name": "bad", "params": ["mesh_ms", "map_uv_ms"]}]})

	def test_nested_map_opt_uses_normal_lasagna_param_names(self) -> None:
		stage = parse_global_map_stage_item(
			{"steps": 1, "lr": 0.01, "params": ["map_affine"], "args": {"subdiv": 2}},
			normal_lasagna=True,
		)
		self.assertEqual(stage.params, ("affine",))
		with self.assertRaisesRegex(ValueError, "map_affine"):
			parse_global_map_stage_item({"params": ["affine"]}, normal_lasagna=True)

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

	def test_config_parses_stage_args(self) -> None:
		with tempfile.TemporaryDirectory() as tmp:
			cfg = parse_global_map_config(_write_config(tmp, affine_steps=1, map_steps=1))
			self.assertEqual(cfg.stages[1].params, ("map_uv_ms",))
			self.assertEqual(cfg.stages[1].min_scaledown, 3)
			parsed = snap_surf_config_from_global_config(cfg, cfg.stages[1])
			self.assertEqual(parsed.map_init.subdiv, 2)
			path = Path(tmp, "named_cfg.json")
			path.write_text(json.dumps({"stages": [{"name": "affine_init_scan", "params": ["affine"]}]}), encoding="utf-8")
			named = parse_global_map_config(path)
			self.assertEqual(named.stages[0].name, "affine_init_scan")

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

	def test_fixture_cli_writes_global_outputs_without_growth_masks(self) -> None:
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
			self.assertTrue(os.path.exists(os.path.join(out_dir, "objs", "stage_000_affine", "map_ext_to_model.obj")))
			self.assertTrue(os.path.exists(os.path.join(out_dir, "objs", "stage_001_map_uv_ms", "map_ext_to_model.obj")))
			self.assertTrue(os.path.exists(os.path.join(out_dir, "objs", "final", "map_ext_to_model.obj")))
			self.assertTrue(os.path.exists(os.path.join(out_dir, "objs", "final", "map_ext_to_model_worst_ref_1pct.obj")))
			final_meta = json.loads(Path(out_dir, "objs", "final", "meta.json").read_text(encoding="utf-8"))
			self.assertEqual(final_meta["name"], "final")
			self.assertIn("worst_ref_1pct_vectors", final_meta)
			self.assertFalse(os.path.exists(os.path.join(out_dir, "active_quad.tif")))
			self.assertFalse(os.path.exists(os.path.join(out_dir, "blocked_quad.tif")))
			metrics = json.loads(Path(out_dir, "metrics.json").read_text(encoding="utf-8"))
			self.assertEqual(metrics["common_vertices"], 25)
			self.assertIn("avg_model_quad_distance", metrics)
			self.assertIn("avg_model_quad_distance", metrics["history"][-1])
			self.assertLess(metrics["model_l2_max_delta"], 1.0)

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
			self.assertIn("1/3", text)
			self.assertIn("3/3", text)
			self.assertNotIn("2/3", text)
			self.assertRegex(text, r"0/2\s+map_uv_ms")
			self.assertIn("stat", text)
			self.assertIn("station loss", text)
			self.assertIn("dst", text)

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
							"params": ["affine"],
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
							"params": ["affine"],
							"args": {"affine_seed_quad_init": {"enabled": True}},
						}
					],
				},
			)
			stdout = StringIO()

			with redirect_stdout(stdout):
				metrics = optimize_fixture(fixture_dir, cfg_path, out_dir=out_dir)

			text = stdout.getvalue()
			self.assertIn("affine seed quad init", text)
			self.assertIn("0/1", text)
			self.assertIn("1/1", text)
			self.assertNotIn("affine multistart candidates", text)
			self.assertTrue(os.path.exists(os.path.join(out_dir, "objs", "stage_000_affine_seed_quad_init", "map_ext_to_model.obj")))
			self.assertEqual(metrics["history"][0]["name"], "affine_seed_quad_init")


if __name__ == "__main__":
	unittest.main()
