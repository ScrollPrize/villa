from __future__ import annotations

import os
import sys
import unittest


ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
	sys.path.insert(0, ROOT)

import optimizer


class OptimizerStageWeightsTest(unittest.TestCase):
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


if __name__ == "__main__":
	unittest.main()
