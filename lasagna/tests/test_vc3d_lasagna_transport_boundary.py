from __future__ import annotations

import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PANEL_CPP = REPO_ROOT / "volume-cartographer/apps/VC3D/segmentation/panels/SegmentationLasagnaPanel.cpp"
MANAGER_CPP = REPO_ROOT / "volume-cartographer/apps/VC3D/LasagnaServiceManager.cpp"
BATCH_WINDOW_CPP = REPO_ROOT / "volume-cartographer/apps/VC3D/LasagnaBatchWindow.cpp"
FIT_SERVICE_PY = REPO_ROOT / "lasagna/fit_service.py"


class VC3DLasagnaTransportBoundaryTest(unittest.TestCase):
	def setUp(self) -> None:
		self.source = PANEL_CPP.read_text(encoding="utf-8")
		self.manager_source = MANAGER_CPP.read_text(encoding="utf-8")
		self.batch_window_source = BATCH_WINDOW_CPP.read_text(encoding="utf-8")

	def test_transport_invariant_is_documented_at_request_assembly(self) -> None:
		self.assertIn("VC3D is transport only", self.source)
		self.assertIn("Config interpretation belongs in fit_service.py / fit.py", self.source)

	def test_vc3d_sends_active_volume_shape_for_lasagna_scaling(self) -> None:
		self.assertIn("volumeShapeZyxForState", self.source)
		self.assertIn('request[QStringLiteral("volume_shape_zyx")] = volumeShapeZyx', self.source)
		self.assertIn('jobSpec[QStringLiteral("volume_shape_zyx")] = volumeShapeZyx', self.source)

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
			"args[QStringLiteral(\"windings\")]",
		]

		for literal in forbidden_literals:
			with self.subTest(literal=literal):
				self.assertNotIn(literal, self.source)

	def test_vc3d_lasagna_requests_use_versioned_request_helper(self) -> None:
		self.assertIn('constexpr const char* kFitServiceApiVersion = "2"', self.manager_source)
		self.assertIn('constexpr const char* kFitServiceApiVersionHeader = "X-Fit-Service-API-Version"', self.manager_source)
		self.assertIn("req.setRawHeader(kFitServiceApiVersionHeader, kFitServiceApiVersion)", self.manager_source)
		self.assertEqual(self.manager_source.count("QNetworkRequest req("), 1)
		for request_line in [
			"QNetworkRequest req = fitServiceRequest(url);",
		]:
			with self.subTest(request_line=request_line):
				self.assertIn(request_line, self.manager_source)

	def test_vc3d_new_model_sends_depth_not_windings(self) -> None:
		self.assertIn('args[QStringLiteral("depth")] = nmN', self.source)
		self.assertIn("windings/depth=", self.source)
		self.assertIn("Number of windings / model depth layers", self.source)
		self.assertNotIn('args[QStringLiteral("windings")] = nmN', self.source)

	def test_lasagna_service_enforces_and_returns_api_version_header(self) -> None:
		source = FIT_SERVICE_PY.read_text(encoding="utf-8")
		self.assertIn('_API_VERSION = "2"', source)
		self.assertIn('_API_VERSION_HEADER = "X-Fit-Service-API-Version"', source)
		self.assertIn("self.send_header(_API_VERSION_HEADER, _API_VERSION)", source)
		self.assertIn("def _validate_api_version", source)
		self.assertIn("if not self._validate_api_version():", source)

	def test_batch_queue_table_shows_output_name(self) -> None:
		self.assertIn('tr("Output")', self.batch_window_source)
		self.assertIn('job[QStringLiteral("output_name")]', self.batch_window_source)

	def test_vc3d_checks_transport_error_before_api_version(self) -> None:
		self.assertIn("bool isTransportError(const QNetworkReply* reply)", self.manager_source)
		start = self.manager_source.index("void LasagnaServiceManager::handleStatusReply")
		end = self.manager_source.index("void LasagnaServiceManager::downloadResults", start)
		method_source = self.manager_source[start:end]
		self.assertLess(
			method_source.index("isTransportError(reply)"),
			method_source.index('validateApiVersion(reply, tr("Poll status"))'),
		)


if __name__ == "__main__":
	unittest.main()
