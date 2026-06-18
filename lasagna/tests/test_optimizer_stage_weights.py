from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path


ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
	sys.path.insert(0, ROOT)

import optimizer


class OptimizerStageWeightsTest(unittest.TestCase):
	def test_atlas_line_config_parses_split_step_regularizers(self) -> None:
		cfg_path = Path(ROOT) / "configs" / "atlas_line.json"
		cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
		stages = optimizer.load_stages_cfg(cfg)

		self.assertEqual(stages[0].name, "atlas_init_relax")
		self.assertTrue(stages[0].global_opt.steps_auto)
		self.assertEqual(stages[0].global_opt.args["atlas_debug_obj_interval"], 1)
		self.assertFalse(stages[0].global_opt.args["atlas_debug_objs"])
		self.assertAlmostEqual(stages[0].global_opt.args["corr_splat_sigma"], 4.0, delta=1.0e-12)
		self.assertIn("smooth_step", cfg["base"])
		self.assertIn("avg_step", cfg["base"])
		self.assertAlmostEqual(
			stages[0].global_opt.base_eff["smooth_step"],
			float(cfg["base"]["smooth_step"]),
			delta=1.0e-12,
		)
		self.assertAlmostEqual(
			stages[0].global_opt.base_eff["avg_step"],
			float(cfg["base"]["avg_step"]),
			delta=1.0e-12,
		)
		self.assertAlmostEqual(
			stages[0].global_opt.base_eff["step"],
			float(cfg["base"].get("step", 0.0)),
			delta=1.0e-12,
		)
		self.assertEqual(stages[1].name, "atlas_line_fit")
		self.assertTrue(stages[1].global_opt.steps_auto)
		self.assertAlmostEqual(stages[1].global_opt.args["corr_splat_sigma"], 4.0, delta=1.0e-12)
		self.assertAlmostEqual(stages[1].global_opt.eff["atlas_line_other"], 0.0, delta=1.0e-12)

	def test_stage_w_fac_does_not_inherit_to_later_stages(self) -> None:
		stages = optimizer.load_stages_cfg({
			"base": {
				"normal": 1.0,
				"smooth": 2.0,
				"corr": 0.25,
			},
			"stages": [
				{"name": "stage0", "steps": 1, "w_fac": {"corr": 0.0}},
				{"name": "stage1", "steps": 1, "w_fac": {"smooth": 3.0}},
				{"name": "stage2", "steps": 1},
			],
		})

		self.assertEqual(stages[0].global_opt.eff["corr"], 0.0)
		self.assertEqual(stages[1].global_opt.eff["corr"], 0.25)
		self.assertEqual(stages[1].global_opt.eff["smooth"], 6.0)
		self.assertEqual(stages[2].global_opt.eff["corr"], 0.25)
		self.assertEqual(stages[2].global_opt.eff["smooth"], 2.0)

	def test_default_mul_does_not_inherit_to_later_stages(self) -> None:
		stages = optimizer.load_stages_cfg({
			"base": {
				"normal": 1.0,
				"smooth": 2.0,
			},
			"stages": [
				{"name": "stage0", "steps": 1, "default_mul": 0.5},
				{"name": "stage1", "steps": 1},
			],
		})

		self.assertEqual(stages[0].global_opt.eff["normal"], 0.5)
		self.assertEqual(stages[0].global_opt.eff["smooth"], 1.0)
		self.assertEqual(stages[1].global_opt.eff["normal"], 1.0)
		self.assertEqual(stages[1].global_opt.eff["smooth"], 2.0)

	def test_numeric_w_fac_scales_applicable_model_losses(self) -> None:
		stages = optimizer.load_stages_cfg({
			"base": {
				"normal": 2.0,
				"smooth": 4.0,
				"map_dist": 8.0,
			},
			"stages": [
				{"name": "stage0", "steps": 1, "params": ["mesh_ms"], "w_fac": 0.25},
			],
		})

		opt = stages[0].global_opt
		self.assertEqual(opt.eff["normal"], 0.5)
		self.assertEqual(opt.eff["smooth"], 1.0)
		self.assertEqual(opt.eff["map_dist"], 8.0)

	def test_atlas_line_split_weights_parse_independently(self) -> None:
		stages = optimizer.load_stages_cfg({
			"base": {
				"atlas_line_control": 0.2,
				"atlas_line_other": 0.5,
			},
			"stages": [
				{"name": "stage0", "steps": 1, "params": ["mesh_ms"], "w_fac": {
					"atlas_line_control": 3.0,
					"atlas_line_other": 0.25,
				}},
			],
		})

		opt = stages[0].global_opt
		self.assertAlmostEqual(opt.eff["atlas_line_control"], 0.6, delta=1.0e-12)
		self.assertAlmostEqual(opt.eff["atlas_line_other"], 0.125, delta=1.0e-12)

	def test_atlas_line_snap_rejects_combined_atlas_line_terms(self) -> None:
		with self.assertRaisesRegex(ValueError, "atlas_line_snap cannot be combined"):
			optimizer.load_stages_cfg({
				"base": {
					"atlas_line_snap": 1.0,
					"atlas_line_control": 0.5,
				},
				"stages": [
					{"name": "stage0", "steps": 1, "params": ["mesh_ms"]},
				],
			})

	def test_atlas_snap_flow_speculative_config_parses_snap_plus_pred_dt(self) -> None:
		cfg_path = Path(ROOT) / "configs" / "atlas_reopt_snap_flow_speculative.json"
		cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
		stages = optimizer.load_stages_cfg(cfg)

		self.assertTrue(cfg["args"].get("tifxyz-flow-gate-channels"))
		self.assertEqual(len(stages), 1)
		opt = stages[0].global_opt
		self.assertGreater(opt.eff["atlas_line_snap"], 0.0)
		self.assertGreater(opt.eff["pred_dt"], 0.0)
		self.assertEqual(opt.eff["atlas_line_control"], 0.0)
		self.assertEqual(opt.eff["atlas_line_other"], 0.0)
		gate = opt.args["pred_dt_flow_gate"]
		self.assertTrue(gate["enabled"])
		self.assertTrue(gate["debug"])
		self.assertEqual(gate["debug_layer_interval"], 0)
		self.assertEqual(gate["debug_vis_interval"], 10)
		self.assertTrue(gate["atlas_snap_seed_enabled"])
		self.assertFalse(gate["corr_seed_enabled"])
		self.assertTrue(gate["anticipatory_pull"]["enabled"])

	def test_step_regularizer_split_weights_parse_independently(self) -> None:
		stages = optimizer.load_stages_cfg({
			"base": {
				"smooth_step": 10.0,
				"avg_step": 0.1,
			},
			"stages": [
				{"name": "stage0", "steps": 1, "params": ["mesh_ms"], "w_fac": {
					"smooth_step": 0.0,
					"avg_step": 5.0,
				}},
				{"name": "stage1", "steps": 1, "params": ["mesh_ms"], "w_fac": {
					"smooth_step": 0.25,
					"avg_step": 0.0,
				}},
			],
		})

		self.assertAlmostEqual(stages[0].global_opt.eff["smooth_step"], 0.0, delta=1.0e-12)
		self.assertAlmostEqual(stages[0].global_opt.eff["avg_step"], 0.5, delta=1.0e-12)
		self.assertAlmostEqual(stages[1].global_opt.eff["smooth_step"], 2.5, delta=1.0e-12)
		self.assertAlmostEqual(stages[1].global_opt.eff["avg_step"], 0.0, delta=1.0e-12)

	def test_numeric_w_fac_scales_applicable_map_losses(self) -> None:
		stages = optimizer.load_stages_cfg({
			"base": {
				"normal": 2.0,
				"map_dist": 8.0,
				"map_station_t": 0.5,
			},
			"stages": [
				{"name": "stage0", "steps": 1, "params": ["map_surf_affine"], "w_fac": 0.25},
			],
		})

		opt = stages[0].global_opt
		self.assertEqual(opt.eff["normal"], 2.0)
		self.assertEqual(opt.eff["map_dist"], 2.0)
		self.assertEqual(opt.eff["map_station_t"], 0.125)

	def test_lasagna_map_stage_adapter_keeps_base_and_modifier_separate(self) -> None:
		stages = optimizer.load_stages_cfg({
			"base": {
				"map_dist": 2.0,
				"map_smooth": 0.25,
				"map_station_t": 0.5,
			},
			"stages": [
				{"name": "stage0", "steps": 1, "params": ["map_surf_ms"], "w_fac": 0.5},
			],
		})

		stage = optimizer._global_map_stage_from_opt_settings(
			name="stage0",
			opt_cfg=stages[0].global_opt,
			args={},
		)

		self.assertEqual(stage.args["map_init"]["w_dist"], 2.0)
		self.assertEqual(stage.args["map_init"]["w_smooth"], 0.25)
		self.assertEqual(stage.w_fac["dist"], 0.5)
		self.assertEqual(stage.w_fac["smooth"], 0.5)
		self.assertEqual(stage.args["map_station_t"], 0.25)


if __name__ == "__main__":
	unittest.main()
