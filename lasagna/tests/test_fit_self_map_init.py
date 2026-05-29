from __future__ import annotations

import os
import sys
import unittest

TEST_DIR = os.path.dirname(__file__)
if TEST_DIR not in sys.path:
	sys.path.insert(0, TEST_DIR)

import fit


class FitSelfMapInitTest(unittest.TestCase):
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
