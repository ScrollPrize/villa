from __future__ import annotations

import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PANEL_CPP = REPO_ROOT / "volume-cartographer/apps/VC3D/segmentation/panels/SegmentationLasagnaPanel.cpp"
MANAGER_CPP = REPO_ROOT / "volume-cartographer/apps/VC3D/LasagnaServiceManager.cpp"
BATCH_WINDOW_CPP = REPO_ROOT / "volume-cartographer/apps/VC3D/LasagnaBatchWindow.cpp"
FIT_SERVICE_PY = REPO_ROOT / "lasagna/fit_service.py"
OLD_FIT_SERVICE_PY = REPO_ROOT / "lasagna/old_2d/fit_service.py"


class VC3DLasagnaTransportBoundaryTest(unittest.TestCase):
	def setUp(self) -> None:
		self.source = PANEL_CPP.read_text(encoding="utf-8")
		self.manager_source = MANAGER_CPP.read_text(encoding="utf-8")
		self.batch_window_source = BATCH_WINDOW_CPP.read_text(encoding="utf-8")

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

	def test_vc3d_lasagna_requests_use_versioned_request_helper(self) -> None:
		self.assertIn('constexpr const char* kFitServiceApiVersion = "1"', self.manager_source)
		self.assertIn('constexpr const char* kFitServiceApiVersionHeader = "X-Fit-Service-API-Version"', self.manager_source)
		self.assertIn("req.setRawHeader(kFitServiceApiVersionHeader, kFitServiceApiVersion)", self.manager_source)
		self.assertEqual(self.manager_source.count("QNetworkRequest req("), 1)
		for request_line in [
			"QNetworkRequest req = fitServiceRequest(url);",
		]:
			with self.subTest(request_line=request_line):
				self.assertIn(request_line, self.manager_source)

	def test_lasagna_services_enforce_and_return_api_version_header(self) -> None:
		for path in (FIT_SERVICE_PY, OLD_FIT_SERVICE_PY):
			source = path.read_text(encoding="utf-8")
			with self.subTest(path=str(path.relative_to(REPO_ROOT))):
				self.assertIn('_API_VERSION = "1"', source)
				self.assertIn('_API_VERSION_HEADER = "X-Fit-Service-API-Version"', source)
				self.assertIn("self.send_header(_API_VERSION_HEADER, _API_VERSION)", source)
				self.assertIn("def _validate_api_version", source)
				self.assertIn("if not self._validate_api_version():", source)

	def test_batch_queue_table_shows_output_name(self) -> None:
		self.assertIn('tr("Output")', self.batch_window_source)
		self.assertIn('job[QStringLiteral("output_name")]', self.batch_window_source)


if __name__ == "__main__":
	unittest.main()
