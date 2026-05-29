from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path


ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
	sys.path.insert(0, ROOT)


class ApprovalInpaintConfigTest(unittest.TestCase):
	def test_fit_arg_defaults_disable_tifxyz_flow_gate_channels(self) -> None:
		import fit

		parser = fit._build_parser()
		args = parser.parse_args([])
		self.assertFalse(args.tifxyz_flow_gate_channels)

	def test_init_corr_point_roi_enables_tifxyz_flow_gate_channels(self) -> None:
		cfg = json.loads((Path(ROOT) / "configs" / "init_corr_point_roi.json").read_text(encoding="utf-8"))
		args = cfg.get("args", {})
		self.assertTrue(args.get("tifxyz-flow-gate-channels"))

	def test_crop_flow_config_enables_tifxyz_flow_gate_channels(self) -> None:
		cfg = json.loads((Path(ROOT) / "configs" / "init_corr_approval_inpaint_flow.json").read_text(encoding="utf-8"))
		args = cfg.get("args", {})
		self.assertEqual(args.get("init-mode"), "shell-dir-crop")
		self.assertTrue(args.get("tifxyz-flow-gate-channels"))

	def test_approval_inpaint_export_configs_enable_output_mask(self) -> None:
		config_names = [
			"init_corr_approval_inpaint.json",
			"init_corr_approval_inpaint_flow.json",
			"init_corr_snap_approval_inpaint.json",
		]
		config_dir = Path(ROOT) / "configs"
		for name in config_names:
			with self.subTest(name=name):
				cfg = json.loads((config_dir / name).read_text(encoding="utf-8"))
				args = cfg.get("args", {})
				self.assertTrue(args.get("approval-inpaint"))
				self.assertTrue(args.get("approval-inpaint-output-mask"))
				self.assertEqual(args.get("approval-inpaint-output-mask-dilate"), 3)


if __name__ == "__main__":
	unittest.main()
