from __future__ import annotations

import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PANEL_CPP = REPO_ROOT / "volume-cartographer/apps/VC3D/segmentation/panels/SegmentationLasagnaPanel.cpp"


class VC3DLasagnaTransportBoundaryTest(unittest.TestCase):
	def setUp(self) -> None:
		self.source = PANEL_CPP.read_text(encoding="utf-8")

	def test_transport_invariant_is_documented_at_request_assembly(self) -> None:
		self.assertIn("VC3D is transport only", self.source)
		self.assertIn("Config interpretation belongs in fit_service.py / fit.py", self.source)

	def test_vc3d_does_not_branch_on_known_lasagna_config_semantics(self) -> None:
		forbidden_literals = [
			"lasagnaModelInit",
			"modelInit",
			"approval-inpaint",
			"station_",
			"tifxyz-init",
			"model-input",
			"args.remove(QStringLiteral(\"seed\"))",
			"args.remove(QStringLiteral(\"model-w\"))",
			"args.remove(QStringLiteral(\"model-h\"))",
			"args.remove(QStringLiteral(\"windings\"))",
		]

		for literal in forbidden_literals:
			with self.subTest(literal=literal):
				self.assertNotIn(literal, self.source)


if __name__ == "__main__":
	unittest.main()
