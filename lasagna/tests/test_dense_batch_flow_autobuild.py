from __future__ import annotations

import os
import sys
import unittest
import ctypes
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch


ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
	sys.path.insert(0, ROOT)

import dense_batch_flow
import fit_data
import opt_loss_pred_dt


class DenseBatchFlowAutobuildTest(unittest.TestCase):
	def setUp(self) -> None:
		dense_batch_flow._LIB = None

	def tearDown(self) -> None:
		dense_batch_flow._LIB = None

	def test_existing_library_loads_without_autobuild(self) -> None:
		fake_lib = object()
		with (
			mock.patch.object(dense_batch_flow, "_find_library_path", return_value=Path("/tmp/libdense_batch_flow.so")),
			mock.patch.object(dense_batch_flow, "_library_needs_rebuild", return_value=False),
			mock.patch.object(dense_batch_flow, "_auto_build_library") as auto_build,
			mock.patch.object(dense_batch_flow, "_load_library_from_path", return_value=fake_lib) as load_from_path,
		):
			self.assertIs(dense_batch_flow._load_library(), fake_lib)

		auto_build.assert_not_called()
		load_from_path.assert_called_once_with(Path("/tmp/libdense_batch_flow.so"))

	def test_stale_library_autobuilds_then_loads(self) -> None:
		fake_lib = object()
		with (
			mock.patch.object(
				dense_batch_flow,
				"_find_library_path",
				side_effect=[Path("/tmp/libdense_batch_flow.so"), Path("/tmp/libdense_batch_flow.so")],
			),
			mock.patch.object(dense_batch_flow, "_library_needs_rebuild", return_value=True),
			mock.patch.object(dense_batch_flow, "_auto_build_library") as auto_build,
			mock.patch.object(dense_batch_flow, "_load_library_from_path", return_value=fake_lib),
		):
			self.assertIs(dense_batch_flow._load_library(), fake_lib)

		auto_build.assert_called_once_with()

	def test_missing_library_autobuilds_then_loads(self) -> None:
		fake_lib = object()
		with (
			mock.patch.object(
				dense_batch_flow,
				"_find_library_path",
				side_effect=[None, Path("/tmp/libdense_batch_flow.so")],
			),
			mock.patch.object(dense_batch_flow, "_auto_build_library") as auto_build,
			mock.patch.object(dense_batch_flow, "_load_library_from_path", return_value=fake_lib),
		):
			self.assertIs(dense_batch_flow._load_library(), fake_lib)

		auto_build.assert_called_once_with()

	def test_disabled_autobuild_raises_manual_build_message(self) -> None:
		with (
			mock.patch.dict(os.environ, {"LASAGNA_DENSE_BATCH_FLOW_AUTOBUILD": "0"}),
			mock.patch.object(dense_batch_flow, "_find_library_path", return_value=None),
			mock.patch.object(dense_batch_flow, "_auto_build_library") as auto_build,
		):
			with self.assertRaisesRegex(RuntimeError, "Automatic build is disabled"):
				dense_batch_flow._load_library()

		auto_build.assert_not_called()

	def test_build_command_failure_reports_output(self) -> None:
		with mock.patch.object(
			dense_batch_flow.subprocess,
			"run",
			return_value=SimpleNamespace(returncode=1, stdout="cmake exploded\n"),
		):
			with self.assertRaisesRegex(RuntimeError, "cmake exploded"):
				dense_batch_flow._run_build_command(["cmake", "--build", "/missing"])

	def test_compute_flow_grid_passes_local_boost(self) -> None:
		captured: dict[str, float] = {}

		def fake_dense_batch_flow_grid_u8(*args):
			captured["extra_source_count"] = int(args[6])
			captured["grid_step"] = int(args[23])
			captured["backtrack_distance"] = float(args[24].value)
			captured["local_boost"] = float(args[25].value)
			return 0

		fake_lib = SimpleNamespace(
			dense_batch_flow_grid_u8=fake_dense_batch_flow_grid_u8
		)
		with mock.patch.object(dense_batch_flow, "_load_library", return_value=fake_lib):
			dense_batch_flow.compute_flow_grid(
				np.zeros((4, 4), dtype=np.uint8),
				source_xy=(1, 1),
				extra_source_xy=np.array([[2, 1], [2, 2]], dtype=np.int32),
				query_xy=np.array([[1.0, 1.0]], dtype=np.float32),
				grid_step=4,
				backtrack_distance=12.0,
				local_boost=0.5,
			)

		self.assertEqual(captured["extra_source_count"], 2)
		self.assertEqual(captured["grid_step"], 4)
		self.assertEqual(captured["backtrack_distance"], 12.0)
		self.assertEqual(captured["local_boost"], 0.5)

	def test_compute_flow_grid_optionally_returns_gate_components(self) -> None:
		def fake_dense_batch_flow_grid_u8(*args):
			query_count = int(args[8])
			final = np.ctypeslib.as_array(args[9], shape=(query_count,))
			local = np.ctypeslib.as_array(
				ctypes.cast(args[35], ctypes.POINTER(ctypes.c_float)),
				shape=(query_count,),
			)
			normalized = np.ctypeslib.as_array(
				ctypes.cast(args[36], ctypes.POINTER(ctypes.c_float)),
				shape=(query_count,),
			)
			final[:] = [0.25, 0.75]
			local[:] = [0.1, 0.9]
			normalized[:] = [0.2, 0.8]
			return 0

		fake_lib = SimpleNamespace(
			dense_batch_flow_grid_u8=fake_dense_batch_flow_grid_u8
		)
		with mock.patch.object(dense_batch_flow, "_load_library", return_value=fake_lib):
			query_flow, _dense_flow, components = dense_batch_flow.compute_flow_grid(
				np.zeros((4, 4), dtype=np.uint8),
				source_xy=(1, 1),
				query_xy=np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32),
				return_components=True,
			)

		np.testing.assert_allclose(query_flow, [0.25, 0.75])
		np.testing.assert_allclose(components["flow_gate_local_contrast"], [0.1, 0.9])
		np.testing.assert_allclose(components["flow_gate_component_normalized"], [0.2, 0.8])

	def test_flow_gate_debug_intervals_default_to_disabled(self) -> None:
		self.assertEqual(
			opt_loss_pred_dt._debug_interval({}, "debug_layer_interval"),
			0,
		)
		self.assertEqual(
			opt_loss_pred_dt._debug_interval(
				{},
				"debug_vis_interval",
				"debug_jpg_interval",
			),
			0,
		)
		self.assertEqual(
			opt_loss_pred_dt._debug_interval(
				{"debug_vis_interval": 0},
				"debug_vis_interval",
				"debug_jpg_interval",
			),
			0,
		)
		self.assertEqual(
			opt_loss_pred_dt._debug_interval(
				{"debug_jpg_interval": 50},
				"debug_vis_interval",
				"debug_jpg_interval",
			),
			50,
		)
		self.assertEqual(
			opt_loss_pred_dt._debug_interval(
				{"debug_vis_interval": 7, "debug_jpg_interval": 50},
				"debug_vis_interval",
				"debug_jpg_interval",
			),
			7,
		)

	def test_flow_gate_prefetch_skips_anticipatory_pull_until_gate_exists(self) -> None:
		class FakeData:
			def _spacing_for(self, channel: str) -> tuple[float, float, float]:
				return (1.0, 1.0, 1.0)

		cfg = {
			"enabled": True,
			"anticipatory_pull": {
				"enabled": True,
				"samples": 8,
			},
		}
		xyz_hr = torch.zeros((1, 3, 3, 3), dtype=torch.float32)
		xyz_lr = torch.zeros((1, 3, 3, 3), dtype=torch.float32)

		self.assertIsNone(
			opt_loss_pred_dt.flow_gate_prefetch_points(
				data=FakeData(),
				xyz_hr=xyz_hr,
				xyz_lr=xyz_lr,
				cfg=cfg,
			)
		)

	def test_flow_gate_loss_prefetch_skips_anticipatory_pull_until_gate_exists(self) -> None:
		cfg = {
			"enabled": True,
			"anticipatory_pull": {
				"enabled": True,
				"samples": 8,
				"search_steps": 21,
			},
		}
		xyz_lr = torch.zeros((1, 3, 3, 3), dtype=torch.float32)
		gt_normal = torch.zeros_like(xyz_lr)
		gt_normal[..., 2] = 1.0
		res = SimpleNamespace(
			xyz_lr=xyz_lr,
			xyz_hr=torch.zeros((1, 3, 3, 3), dtype=torch.float32),
			gt_normal_lr=gt_normal,
		)

		opt_loss_pred_dt.configure_pred_dt(normal_source="gt")
		try:
			items = opt_loss_pred_dt.flow_gate_prefetch_items_for_result(res=res, cfg=cfg)
		finally:
			opt_loss_pred_dt.configure_pred_dt(normal_source="model")

		self.assertEqual(set(items.keys()), {"pred_dt"})
		self.assertEqual(int(items["pred_dt"].reshape(-1, 3).shape[0]), 9)

	def test_anticipatory_pull_candidate_scoring_is_gate_filtered(self) -> None:
		class ExplodingData:
			sparse_caches = None

			def grid_sample_fullres(self, *args, **kwargs):
				raise AssertionError("inactive anticipatory candidates should not be sampled")

		res = SimpleNamespace(
			xyz_lr=torch.zeros((1, 2, 2, 3), dtype=torch.float32),
			data=ExplodingData(),
			params=SimpleNamespace(mesh_step=1.0),
		)
		candidates = opt_loss_pred_dt._score_anticipatory_pull_candidates(
			res=res,
			cfg={"samples": 4},
			flow_weight=torch.zeros((1, 1, 2, 2), dtype=torch.float32),
			mask_lr=torch.ones((1, 1, 2, 2), dtype=torch.float32),
		)

		self.assertIsNotNone(candidates)
		self.assertEqual(int(candidates["tip_h"].numel()), 0)
		self.assertEqual(candidates["_stats"]["total_candidates"], 12.0)
		self.assertEqual(candidates["_stats"]["gate_candidates"], 0.0)
		self.assertEqual(candidates["_stats"]["scored_candidates"], 0.0)

	def test_local_boost_dilation_is_global_after_region_normalization(self) -> None:
		source_path = Path(ROOT) / "dense_batch_min_cut/src/dense_batch_preprocess.cpp"
		source = source_path.read_text()
		start = source.index("cv::Mat compute_flow_gate_weight_image")
		end = source.index("cv::Mat labels_to_u16", start)
		body = source[start:end]

		self.assertIn("normalize_flow_by_regions(flow, normalization_labels", body)
		self.assertIn("out_component_normalized", body)
		self.assertIn("local_contrast", body)
		self.assertIn("normalized_flow.setTo(1.0f, source_reach_mask)", body)
		self.assertIn("cv::dilate(normalized_flow, local_max, kernel)", body)
		self.assertIn("local_max_scope: global_after_region_normalization", body)
		self.assertNotIn("label_local_max", body)
		self.assertNotIn("normalized_flow.copyTo(label_flow", body)

	def test_flow_bridge_samples_component_channels_separately(self) -> None:
		source_path = Path(ROOT) / "dense_batch_min_cut/src/dense_batch_preprocess.cpp"
		source = source_path.read_text()
		start = source.index('extern "C" int dense_batch_flow_grid_u8')
		end = source.index("return 0;", start)
		body = source[start:end]

		self.assertIn("query_flow_local_contrast", body)
		self.assertIn("query_flow_component_normalized", body)
		self.assertIn("sample_flow(flow_gate_weight", body)
		self.assertIn("query_flow_local_contrast[i] = sample_flow", body)
		self.assertIn("flow_gate_local_contrast", body)
		self.assertIn("query_flow_component_normalized[i] = sample_flow", body)
		self.assertIn("flow_gate_component_normalized", body)

	def test_gate_normalization_uses_pre_merge_component_regions(self) -> None:
		source_path = Path(ROOT) / "dense_batch_min_cut/src/dense_batch_preprocess.cpp"
		source = source_path.read_text()

		self.assertIn("flow_gate_component_regions", source)
		self.assertIn("source_attractor_flow_input", source)
		self.assertIn(
			"white_domain, tree_dense_flow_source_attractor",
			source,
		)
		self.assertIn("source_starts, false", source)
		self.assertIn(
			"dense_flow_result.flow_gate_component_regions,\n"
			"            dense_flow_result.source_reach_mask",
			source,
		)
		self.assertIn(
			"normalize_flow_by_regions(\n"
			"            dense_flow_result.tree_dense_flow_greedy_ascent,\n"
			"            dense_flow_result.flow_gate_component_regions",
			source,
		)

	def test_compute_flow_grid_returns_debug_source_component_mask(self) -> None:
		def fake_dense_batch_flow_grid_u8(*args):
			source_edges = np.ctypeslib.as_array(args[21], shape=(4,))
			source_components = np.ctypeslib.as_array(args[22], shape=(4,))
			source_edges[:] = [0.0, 1.0, 0.0, 1.0]
			source_components[:] = [0.0, 255.0, 255.0, 0.0]
			return 0

		fake_lib = SimpleNamespace(
			dense_batch_flow_grid_u8=fake_dense_batch_flow_grid_u8
		)
		with mock.patch.object(dense_batch_flow, "_load_library", return_value=fake_lib):
			outputs = dense_batch_flow.compute_flow_grid(
				np.zeros((2, 2), dtype=np.uint8),
				source_xy=(1, 1),
				query_xy=np.array([[1.0, 1.0]], dtype=np.float32),
				return_debug=True,
			)

		self.assertEqual(len(outputs), 14)
		np.testing.assert_array_equal(
			outputs[-2], np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
		)
		np.testing.assert_array_equal(
			outputs[-1], np.array([[0.0, 255.0], [255.0, 0.0]], dtype=np.float32)
		)

	def test_multi_source_disconnected_components_get_gate_values(self) -> None:
		height, width = 180, 260
		yy, xx = np.mgrid[:height, :width]
		left_component = (xx - 70) ** 2 + (yy - 90) ** 2 <= 45 ** 2
		right_component = (xx - 190) ** 2 + (yy - 90) ** 2 <= 32 ** 2
		image = np.zeros((height, width), dtype=np.uint8)
		image[left_component | right_component] = 255

		query_xy = np.array([[70.0, 90.0], [190.0, 90.0]], dtype=np.float32)
		outputs = dense_batch_flow.compute_flow_grid(
			image,
			source_xy=(70, 90),
			extra_source_xy=np.array([[190, 90]], dtype=np.int32),
			query_xy=query_xy,
			grid_step=4,
			backtrack_distance=30.0,
			local_boost=0.5,
			return_debug=True,
			return_metadata=True,
		)
		query_flow = outputs[0]
		gate = outputs[1]
		gate_basis = outputs[3]
		source_edges = outputs[-3]
		source_components = outputs[-2]
		metadata = outputs[-1]

		self.assertEqual(metadata["accepted_source_count"], 2)
		self.assertGreater(int(np.count_nonzero(source_components[left_component])), 0)
		self.assertGreater(int(np.count_nonzero(source_components[right_component])), 0)
		self.assertGreater(int(np.count_nonzero(source_edges[left_component])), 0)
		self.assertGreater(int(np.count_nonzero(source_edges[right_component])), 0)
		self.assertGreater(int(np.count_nonzero(gate[left_component])), 0)
		self.assertGreater(int(np.count_nonzero(gate[right_component])), 0)
		self.assertGreater(int(np.count_nonzero(gate_basis[left_component])), 0)
		self.assertGreater(int(np.count_nonzero(gate_basis[right_component])), 0)
		source_edge_pixels = source_edges > 0.5
		self.assertGreater(int(np.count_nonzero(source_edge_pixels)), 0)
		self.assertTrue(np.all(gate[source_edge_pixels] >= 1.0 - 1.0e-6))
		self.assertTrue(np.all(gate_basis[source_edge_pixels] >= 1.0 - 1.0e-6))
		np.testing.assert_allclose(query_flow, np.ones((2,), dtype=np.float32), atol=1.0e-5)

	def test_source_eroded_by_rim_expansion_still_builds_flow(self) -> None:
		image = np.zeros((120, 140), dtype=np.uint8)
		image[20:90, 20:100] = 255
		component = image > 0

		outputs = dense_batch_flow.compute_flow_grid(
			image,
			source_xy=(20, 50),
			query_xy=np.array([[20.0, 50.0], [60.0, 50.0]], dtype=np.float32),
			grid_step=4,
			backtrack_distance=20.0,
			local_boost=0.5,
			return_debug=True,
			return_metadata=True,
		)
		query_flow = outputs[0]
		gate = outputs[1]
		gate_basis = outputs[3]
		source_edges = outputs[-3]
		source_components = outputs[-2]
		metadata = outputs[-1]

		self.assertEqual(metadata["accepted_source_count"], 1)
		self.assertGreater(int(np.count_nonzero(source_components[component])), 0)
		self.assertGreater(int(np.count_nonzero(gate[component])), 0)
		self.assertGreater(int(np.count_nonzero(gate_basis[component])), 0)
		source_edge_pixels = source_edges > 0.5
		self.assertGreater(int(np.count_nonzero(source_edge_pixels)), 0)
		self.assertTrue(np.all(gate[source_edge_pixels] >= 1.0 - 1.0e-6))
		self.assertTrue(np.all(gate_basis[source_edge_pixels] >= 1.0 - 1.0e-6))
		np.testing.assert_allclose(query_flow, np.ones((2,), dtype=np.float32), atol=1.0e-5)

	def test_corr_point_source_xy_filters_by_surface_distance(self) -> None:
		xyz_img = torch.tensor(
			[
				[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
				[[0.0, 10.0, 0.0], [10.0, 10.0, 0.0]],
			],
			dtype=torch.float32,
		)
		points = torch.tensor(
			[
				[0.5, 0.5, 0.0, 0.0],
				[10.0, 10.0, 5.0, 0.0],
			],
			dtype=torch.float32,
		)
		corr = fit_data.CorrPoints3D(
			points_xyz_winda=points,
			collection_idx=torch.zeros((2,), dtype=torch.int64),
			point_ids=torch.arange(2, dtype=torch.int64),
			is_absolute=torch.ones((2,), dtype=torch.bool),
		)
		res = SimpleNamespace(data=SimpleNamespace(corr_points=corr))

		xy, stats, debug = opt_loss_pred_dt._corr_point_source_xy(
			res=res,
			xyz_img=xyz_img,
			cfg={"corr_seed_surface_distance": 2.0},
			sub_h=1,
			sub_w=1,
		)

		np.testing.assert_array_equal(xy, np.array([[0, 0]], dtype=np.int32))
		self.assertEqual(stats["pred_dt_corr_seed_candidates"], 2.0)
		self.assertEqual(stats["pred_dt_corr_seed_valid"], 1.0)
		np.testing.assert_array_equal(debug["xy"], np.array([[0, 0], [1, 1]], dtype=np.int32))
		np.testing.assert_allclose(debug["distance"], np.array([0.70710677, 5.0], dtype=np.float32))
		np.testing.assert_array_equal(debug["valid"], np.array([True, False]))


if __name__ == "__main__":
	unittest.main()
