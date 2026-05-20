from __future__ import annotations

import os
import sys
import unittest

import torch


ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
	sys.path.insert(0, ROOT)

import fit


class _GridModel:
	def __init__(self, xyz: torch.Tensor) -> None:
		self._xyz = xyz

	def _grid_xyz(self) -> torch.Tensor:
		return self._xyz


class FitSeedSelectionTest(unittest.TestCase):
	def test_model_init_model_uses_grid_center_not_config_seed(self) -> None:
		mdl = _GridModel(torch.tensor([
			[
				[[0.0, 0.0, 0.0], [0.0, 20.0, 0.0]],
				[[10.0, 0.0, 0.0], [10.0, 20.0, 40.0]],
			]
		]))

		seed = fit._optimization_seed_xyz(
			model_init="model",
			config_seed=(999.0, 999.0, 999.0),
			mdl=mdl,
		)

		self.assertEqual(seed, (5.0, 10.0, 10.0))

	def test_non_init_modes_use_grid_center_and_seed_mode_uses_config_seed(self) -> None:
		mdl = _GridModel(torch.tensor([
			[
				[[1.0, 2.0, 3.0], [1.0, 4.0, 3.0]],
				[[3.0, 2.0, 3.0], [3.0, 4.0, 7.0]],
			]
		]))

		self.assertEqual(
			fit._optimization_seed_xyz(
				model_init="ext",
				config_seed=(999.0, 999.0, 999.0),
				mdl=mdl,
			),
			(2.0, 3.0, 4.0),
		)
		self.assertEqual(
			fit._optimization_seed_xyz(
				model_init="seed",
				config_seed=(9.0, 8.0, 7.0),
				mdl=mdl,
			),
			(9.0, 8.0, 7.0),
		)


if __name__ == "__main__":
	unittest.main()
