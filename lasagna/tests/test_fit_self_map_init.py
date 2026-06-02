from __future__ import annotations

import os
import sys
import unittest

TEST_DIR = os.path.dirname(__file__)
if TEST_DIR not in sys.path:
	sys.path.insert(0, TEST_DIR)

import fit
import cli_json
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

	def test_json_windings_alias_sets_model_depth(self) -> None:
		model_cfg = self._model_cfg_from_json_args({"windings": 1})

		self.assertEqual(model_cfg.depth, 1)

	def test_json_depth_and_windings_match_sets_model_depth(self) -> None:
		model_cfg = self._model_cfg_from_json_args({"depth": 1, "windings": 1})

		self.assertEqual(model_cfg.depth, 1)

	def test_json_depth_and_windings_mismatch_is_clear_error(self) -> None:
		with self.assertRaisesRegex(ValueError, "args.depth and args.windings must match"):
			self._model_cfg_from_json_args({"depth": 3, "windings": 1})

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

	def test_self_map_rejects_non_seed_or_non_shell_init(self) -> None:
		with self.assertRaisesRegex(ValueError, "model-init=seed"):
			fit._validate_self_map_init_args(
				self_map_init="multi_wrap_full",
				model_init="model",
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
		with self.assertRaisesRegex(ValueError, "0 < args.model-w < 1.0"):
			fit._validate_self_map_init_args(
				self_map_init="multi_wrap_d",
				model_init="seed",
				init_mode="shell-dir-crop",
				model_depth=3,
				model_w=1.5,
				model_w_unit="wraps",
			)


if __name__ == "__main__":
	unittest.main()
