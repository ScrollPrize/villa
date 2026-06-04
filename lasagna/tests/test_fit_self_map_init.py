from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

TEST_DIR = os.path.dirname(__file__)
if TEST_DIR not in sys.path:
	sys.path.insert(0, TEST_DIR)

import fit
import cli_json
import cli_data
import cli_model


class FitSelfMapInitTest(unittest.TestCase):
	def _model_cfg_from_json_args(self, args_cfg: dict) -> cli_model.ModelConfig:
		parser = fit._build_parser()
		cli_json.apply_defaults_from_cfg_args(parser, {"args": args_cfg})
		args = parser.parse_args([])
		return cli_model.from_args(args)

	def test_json_depth_sets_model_depth_for_multi_wrap_full(self) -> None:
		model_cfg = self._model_cfg_from_json_args({"depth": 1})

		self.assertEqual(model_cfg.depth, 1)
		mode = fit._validate_self_map_init_args(
			self_map_init="multi_wrap_full",
			model_init="seed",
			init_mode="shell-dir-crop",
			model_depth=model_cfg.depth,
			model_w=1.5,
			model_w_unit="wraps",
		)
		self.assertEqual(mode, "multi_wrap_full")

	def test_parser_rejects_windings_cli_arg(self) -> None:
		parser = fit._build_parser()

		with self.assertRaises(SystemExit):
			parser.parse_args(["--windings", "1"])

	def test_json_windings_arg_is_removed(self) -> None:
		with self.assertRaisesRegex(ValueError, "args.windings has been removed"):
			fit._reject_removed_windings_arg({"windings": 1})

	def test_fit_main_rejects_json_windings_arg(self) -> None:
		with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8") as f:
			json.dump({"args": {"input": "/tmp/volume.zarr", "device": "cpu", "windings": 1}}, f)
			f.flush()
			with self.assertRaisesRegex(ValueError, "args.windings has been removed"):
				fit.main([f.name])

	def test_cli_data_from_merged_parser_does_not_own_depth(self) -> None:
		parser = fit._build_parser()
		cli_json.apply_defaults_from_cfg_args(
			parser,
			{"args": {"input": "/tmp/volume.zarr", "device": "cpu", "depth": 2}},
		)
		args = parser.parse_args([])

		model_cfg = cli_model.from_args(args)
		data_cfg = cli_data.from_args(args)

		self.assertEqual(model_cfg.depth, 2)
		self.assertFalse(hasattr(data_cfg, "depth"))
		self.assertFalse(hasattr(data_cfg, "windings"))

	def test_multi_wrap_full_accepts_single_depth_wide_wrap_crop(self) -> None:
		mode = fit._validate_self_map_init_args(
			self_map_init="multi_wrap_full",
			model_init="seed",
			init_mode="shell-dir-crop",
			model_depth=1,
			model_w=1.5,
			model_w_unit="wraps",
		)

		self.assertEqual(mode, "multi_wrap_full")

	def test_multi_wrap_d_accepts_multi_depth_subwrap_crop(self) -> None:
		mode = fit._validate_self_map_init_args(
			self_map_init="multi_wrap_d",
			model_init="seed",
			init_mode="shell-dir-crop",
			model_depth=3,
			model_w=0.5,
			model_w_unit="wraps",
		)

		self.assertEqual(mode, "multi_wrap_d")

	def test_multi_wrap_d_accepts_voxel_width_for_late_shell_width_check(self) -> None:
		mode = fit._validate_self_map_init_args(
			self_map_init="multi_wrap_d",
			model_init="seed",
			init_mode="shell-dir-crop",
			model_depth=3,
			model_w=120.0,
			model_w_unit="voxels",
		)

		self.assertEqual(mode, "multi_wrap_d")

	def test_self_map_width_contract_checks_effective_wrap_count(self) -> None:
		fit._validate_self_map_width_contract(mode="multi_wrap_d", model_w_wraps=0.5)
		fit._validate_self_map_width_contract(mode="multi_wrap_full", model_w_wraps=1.5)
		with self.assertRaisesRegex(ValueError, "0 < args.model-w < 1.0 wraps"):
			fit._validate_self_map_width_contract(mode="multi_wrap_d", model_w_wraps=1.5)
		with self.assertRaisesRegex(ValueError, "args.model-w > 1.0 wraps"):
			fit._validate_self_map_width_contract(mode="multi_wrap_full", model_w_wraps=0.5)

	def test_self_map_rejects_non_seed_model_or_non_shell_seed_init(self) -> None:
		with self.assertRaisesRegex(ValueError, "model-init=seed or args.model-init=model"):
			fit._validate_self_map_init_args(
				self_map_init="multi_wrap_full",
				model_init="ext",
				init_mode="shell-dir-crop",
				model_depth=1,
				model_w=1.5,
				model_w_unit="wraps",
			)
		with self.assertRaisesRegex(ValueError, "init-mode=shell-dir-crop"):
			fit._validate_self_map_init_args(
				self_map_init="multi_wrap_full",
				model_init="seed",
				init_mode="cylinder_seed",
				model_depth=1,
				model_w=1.5,
				model_w_unit="wraps",
			)

	def test_self_map_rejects_wrong_depth_or_width_contract(self) -> None:
		with self.assertRaisesRegex(ValueError, "depth=1"):
			fit._validate_self_map_init_args(
				self_map_init="multi_wrap_full",
				model_init="seed",
				init_mode="shell-dir-crop",
				model_depth=2,
				model_w=1.5,
				model_w_unit="wraps",
			)

	def test_init_grow_parses_single_winding_to_target_depth(self) -> None:
		cfg = fit._parse_init_grow_config(
			{
				"init-grow": {
					"enabled": True,
					"initial_depth": 1,
					"order": ["up"],
					"step": 1,
				}
			},
			target_depth=3,
		)

		self.assertTrue(cfg.enabled)
		self.assertEqual(cfg.initial_depth, 1)
		self.assertEqual(cfg.order, ("up",))
		self.assertEqual(cfg.step, 1)

	def test_init_grow_stage_delta_allows_depth_one_request_to_expand(self) -> None:
		raw_cfg = {
			"args": {
				"init-grow": {
					"initial_depth": 1,
				}
			},
			"stages": [
				{
					"name": "expand-z",
					"grow": {"d_pos": 1},
					"stages": [
						{"name": "expand_up", "steps": 0, "lr": 1.0, "params": ["mesh_ms"]}
					],
				}
			],
		}
		args_cfg = raw_cfg["args"]
		target_depth = max(
			1,
			fit._raw_init_grow_initial_depth(args_cfg) + fit._init_grow_stage_depth_delta(raw_cfg),
		)

		cfg = fit._parse_init_grow_config(args_cfg, target_depth=target_depth)

		self.assertEqual(target_depth, 2)
		self.assertTrue(cfg.enabled)
		self.assertEqual(cfg.initial_depth, 1)

	def test_init_grow_rejects_initial_depth_past_target(self) -> None:
		with self.assertRaisesRegex(ValueError, "initial_depth must be <= args.depth"):
			fit._parse_init_grow_config(
				{"init-grow": {"initial_depth": 4}},
				target_depth=3,
			)

	def test_self_map_model_reopt_defers_checkpoint_shape_validation(self) -> None:
		self.assertEqual(
			fit._validate_self_map_init_args(
				self_map_init="multi_wrap_d",
				model_init="model",
				init_mode="cylinder_seed",
				model_depth=None,
				model_w=None,
				model_w_unit="voxels",
				validate_shape_contract=False,
			),
			"multi_wrap_d",
		)
		with self.assertRaisesRegex(ValueError, "depth > 1"):
			fit._validate_self_map_init_args(
				self_map_init="multi_wrap_d",
				model_init="model",
				init_mode="cylinder_seed",
				model_depth=1,
				model_w=None,
				model_w_unit="voxels",
			)

	def test_self_map_rejects_unsupported_reopt_source(self) -> None:
		with self.assertRaisesRegex(ValueError, "model-init=seed or args.model-init=model"):
			fit._validate_self_map_init_args(
				self_map_init="multi_wrap_d",
				model_init="ext",
				init_mode="cylinder_seed",
				model_depth=2,
				model_w=None,
				model_w_unit="voxels",
			)
		with self.assertRaisesRegex(ValueError, "0 < args.model-w < 1.0"):
			fit._validate_self_map_init_args(
				self_map_init="multi_wrap_d",
				model_init="seed",
				init_mode="shell-dir-crop",
				model_depth=3,
				model_w=1.5,
				model_w_unit="wraps",
			)

	def test_self_map_snap_config_does_not_request_external_surface_maps(self) -> None:
		cfg = {
			"args": {"self-map-init": "multi_wrap_d"},
			"base": {"snap_surf_map": 0.1},
			"external_surfaces": [{"path": "/tmp/ref.tifxyz"}],
			"stages": [
				{
					"name": "self_map",
					"steps": 1,
					"lr": 0.1,
					"params": ["mesh_ms"],
					"args": {"snap_surf_map": {"map_opt": {"steps": 1, "params": ["map_surf_ms"]}}},
				}
			],
		}

		self.assertFalse(fit._config_requests_external_surface_maps(cfg, self_map_init="multi_wrap_d"))

	def test_self_d_grow_config_does_not_request_external_surface_maps(self) -> None:
		cfg_path = os.path.join(os.path.dirname(TEST_DIR), "configs", "init_snap_surf_self_d_grow.json")
		with open(cfg_path, "r", encoding="utf-8") as f:
			cfg = json.load(f)

		self.assertFalse(fit._config_requests_external_surface_maps(cfg, self_map_init="multi_wrap_d"))

	def test_global_snap_config_requests_external_surface_maps(self) -> None:
		cfg = {
			"args": {"self-map-init": "off"},
			"base": {"snap_surf_map": 0.1},
			"stages": [
				{"name": "global", "steps": 1, "lr": 0.1, "params": ["mesh_ms"]},
			],
		}

		self.assertTrue(fit._config_requests_external_surface_maps(cfg, self_map_init="off"))

	def test_ext_offset_config_requests_external_surface_maps(self) -> None:
		cfg = {
			"base": {"ext_offset": 1.0},
			"stages": [
				{"name": "ext", "steps": 1, "lr": 0.1, "params": ["mesh_ms"]},
			],
		}

		self.assertTrue(fit._config_requests_external_surface_maps(cfg, self_map_init="multi_wrap_d"))


if __name__ == "__main__":
	unittest.main()
