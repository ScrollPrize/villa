from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace

import torch


ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
	sys.path.insert(0, ROOT)

import opt_loss_corr


class CorrHeightSplatTest(unittest.TestCase):
	def test_corr_winding_uses_central_relative_input_convention(self) -> None:
		params = SimpleNamespace(depth_windings=(-2, -1, 0, 1))
		relative = torch.tensor([-1.0, 0.0, 1.5], dtype=torch.float32)

		model_winding = opt_loss_corr._corr_relative_to_model_winding(relative, params)
		roundtrip = opt_loss_corr._corr_model_to_relative_winding(model_winding, params)

		self.assertTrue(torch.allclose(model_winding, torch.tensor([1.0, 2.0, 3.5])))
		self.assertTrue(torch.allclose(roundtrip, relative))

	def test_overlapping_points_average_value_but_max_force_weight(self) -> None:
		H_map = torch.zeros(1, 5, 5, dtype=torch.float32)
		V_map = torch.zeros(1, 5, 5, 3, dtype=torch.float32)
		W_sum_map = torch.zeros_like(H_map)
		W_max_map = torch.zeros_like(H_map)

		opt_loss_corr._height_map_splat(
			d_p=torch.tensor([0, 0], dtype=torch.long),
			h_floor_p=torch.tensor([2, 2], dtype=torch.long),
			w_floor_p=torch.tensor([2, 2], dtype=torch.long),
			h_cont_p=torch.tensor([2.0, 2.0], dtype=torch.float32),
			w_cont_p=torch.tensor([2.0, 2.0], dtype=torch.float32),
			signed_delta_p=torch.tensor([2.0, 4.0], dtype=torch.float32),
			normal_p=torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=torch.float32),
			mask_p=torch.tensor([1.0, 1.0], dtype=torch.float32),
			sigma=1.0,
			D=1,
			Hm=5,
			Wm=5,
			H_map=H_map,
			V_map=V_map,
			W_map=W_sum_map,
			W_max_map=W_max_map,
		)

		active = W_sum_map > 1.0e-8
		avg_delta = H_map[active] / W_sum_map[active]
		avg_disp = V_map[active] / W_sum_map[active].unsqueeze(-1)

		self.assertTrue(torch.allclose(avg_delta, torch.full_like(avg_delta, 3.0)))
		self.assertTrue(torch.allclose(avg_disp, torch.tensor([0.0, 0.0, 3.0]).expand_as(avg_disp)))
		self.assertTrue(torch.allclose(W_max_map[active], 0.5 * W_sum_map[active]))
		self.assertLess(float(W_max_map[0, 2, 2]), float(W_sum_map[0, 2, 2]))

	def test_sparse_strip_prefetch_loads_grad_mag_cache(self) -> None:
		class FakeCache:
			channels = ("grad_mag",)

			def __init__(self) -> None:
				self.prefetch_calls = []
				self.sync_calls = 0

			def prefetch(self, xyz_fullres, origin, spacing) -> None:
				self.prefetch_calls.append((xyz_fullres, origin, spacing))

			def sync(self) -> None:
				self.sync_calls += 1

		class FakeData:
			def __init__(self) -> None:
				self.origin_fullres = (1.0, 2.0, 3.0)
				self.cache = FakeCache()
				self.sparse_caches = {"main": self.cache}

			def _spacing_for(self, channel: str):
				self.spacing_channel = channel
				return (4.0, 5.0, 6.0)

		data = FakeData()
		xyz = torch.zeros(1, 1, 3, 3, dtype=torch.float32)

		opt_loss_corr._prefetch_grad_mag_points(data, xyz)

		self.assertEqual(len(data.cache.prefetch_calls), 1)
		self.assertEqual(data.cache.sync_calls, 1)
		self.assertIs(data.cache.prefetch_calls[0][0], xyz)
		self.assertEqual(data.cache.prefetch_calls[0][1], data.origin_fullres)
		self.assertEqual(data.cache.prefetch_calls[0][2], (4.0, 5.0, 6.0))
		self.assertEqual(data.spacing_channel, "grad_mag")


if __name__ == "__main__":
	unittest.main()
