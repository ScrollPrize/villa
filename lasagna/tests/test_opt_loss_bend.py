from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace

import torch


ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
	sys.path.insert(0, ROOT)

import opt_loss_bend


class BendLossTest(unittest.TestCase):
	def test_regular_bend_includes_diagonal_triplets(self) -> None:
		xyz = torch.tensor(
			[
				[
					[[1.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [-1.0, 1.0, 0.0]],
					[[0.0, -1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
					[[1.0, -1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
				]
			],
			dtype=torch.float32,
		)

		loss, maps, masks = opt_loss_bend.bend_loss(res=SimpleNamespace(xyz_lr=xyz))

		self.assertEqual(tuple(maps[0].shape), (1, 1, 1, 1))
		self.assertEqual(tuple(masks[0].shape), (1, 1, 1, 1))
		# H and W are flat, one diagonal is fully folded: (1 - -0.5)^2 / 4 = 0.5625.
		self.assertAlmostEqual(float(loss.detach()), 0.5625, delta=1.0e-6)


if __name__ == "__main__":
	unittest.main()
