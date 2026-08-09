from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest
from unittest import mock

import numpy as np
import tifffile

from tifxyz_label_transfer import view_alignment_napari as viewer


class SurfaceVolumeCompositeTests(unittest.TestCase):
    def test_preview_render_uses_ceil_shape_and_separate_cache_name(self) -> None:
        image = np.arange(7 * 10, dtype=np.uint8).reshape(7, 10)

        actual = viewer.preview_render(image, 4)

        self.assertEqual(actual.shape, (2, 3))
        np.testing.assert_array_equal(
            actual, viewer.resize_nearest(image, (2, 3))
        )
        self.assertEqual(
            viewer.preview_cache_name("registered.tif", 4),
            "registered-preview4.tif",
        )
        self.assertEqual(
            viewer.preview_cache_name("registered.tif", 1),
            "registered.tif",
        )

    def test_transferred_result_path_accepts_mask_batch_names(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            results = Path(temp_dir)
            supervision = results / "supervision-mask-updated-2um.tif"
            validation = results / "validation-mask-updated-2um.tif"
            supervision.touch()
            validation.touch()

            self.assertEqual(
                viewer.transferred_result_path(
                    results, "supervision", "updated-2um"
                ),
                supervision,
            )
            self.assertEqual(
                viewer.transferred_result_path(
                    results, "validation", "updated-2um"
                ),
                validation,
            )

    def test_transferred_result_path_prefers_historical_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            results = Path(temp_dir)
            historical = results / "supervision-2.399um.tif"
            semantic = results / "supervision-mask-2.399um.tif"
            historical.touch()
            semantic.touch()

            self.assertEqual(
                viewer.transferred_result_path(
                    results, "supervision", "2.399um"
                ),
                historical,
            )

    def test_updated_stage_accepts_semantic_batch_output_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            results = root / "results"
            results.mkdir()
            (results / "inklabels-updated-2um.report.json").touch()
            case = viewer.Case(
                root=root,
                name="semantic-stage",
                source_tifxyz=root / "source.tifxyz",
                updated_tifxyz=root / "updated.tifxyz",
                target_tifxyz=None,
                affine_path=None,
                source_label=root / "labels.tif",
                source_supervision=root / "supervision.tif",
                source_surface_volume_url="remote:source.zarr",
                surface_volume_urls={},
                results=results,
                updated_resolution=2.399,
                target_resolution=None,
            )

            self.assertEqual(case.updated_stage, "updated-2um")

    def test_prefers_highest_annotation_revision(self) -> None:
        paths = [
            Path("segment_inklabels_v2.tif"),
            Path("segment_inklabels.tif"),
            Path("segment_inklabels_v3.tif"),
        ]
        self.assertEqual(
            viewer.preferred_annotation_image(paths).name,
            "segment_inklabels_v3.tif",
        )

    def test_resized_tiff_reader_matches_nearest_for_strips_and_tiles(self) -> None:
        image = (np.arange(37 * 53).reshape(37, 53) % 251).astype(np.uint8)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for name, options in (
                ("strips", {"rowsperstrip": 2}),
                ("tiles", {"tile": (16, 16)}),
            ):
                path = root / f"{name}.tif"
                tifffile.imwrite(path, image, compression="deflate", **options)
                for shape in ((11, 17), (73, 91), (1, 1)):
                    actual = viewer.read_tiff_nearest(path, shape)
                    expected = viewer.resize_nearest(image, shape)
                    np.testing.assert_array_equal(actual, expected)

    def test_label_display_prefers_existing_zarr_pyramid(self) -> None:
        import zarr

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "labels.tif"
            path.touch()
            group = zarr.open_group(str(path.with_suffix(".zarr")), mode="w")
            fine = group.create_dataset("0", shape=(65, 8, 10), dtype="u1")
            coarse = group.create_dataset("1", shape=(65, 4, 5), dtype="u1")
            values = np.arange(20, dtype=np.uint8).reshape(4, 5)
            fine[32] = viewer.resize_nearest(values, (8, 10))
            coarse[32] = values

            with mock.patch.object(
                viewer,
                "read_tiff_nearest",
                side_effect=AssertionError("unexpected TIFF decode"),
            ):
                actual = viewer.read_label_display(path, (3, 4))

        np.testing.assert_array_equal(
            actual,
            viewer.resize_nearest(values, (3, 4)),
        )

    def test_prepared_exact_center_render_avoids_remote_fetch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            evidence = root / "renders" / "offset-evidence"
            evidence.mkdir(parents=True)
            source = evidence / "source-center.tif"
            expected = np.arange(12, dtype=np.uint8).reshape(3, 4)
            tifffile.imwrite(source, expected)
            (evidence / "manifest.json").write_text(
                json.dumps(
                    {
                        "comparisons": [
                            {
                                "name": "exact-center",
                                "source_render": str(source),
                                "target_render": str(evidence / "target.tif"),
                            }
                        ]
                    }
                )
            )
            case = mock.Mock(root=root)

            actual = viewer.evidence_center_render(case, "source_render")

        np.testing.assert_array_equal(actual, expected)

    def test_cluster_evidence_manifest_resolves_copied_local_render(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            evidence = root / "renders" / "offset-evidence"
            evidence.mkdir(parents=True)
            source = evidence / "source-center.tif"
            expected = np.arange(12, dtype=np.uint8).reshape(3, 4)
            tifffile.imwrite(source, expected, metadata=None)
            (evidence / "manifest.json").write_text(
                json.dumps(
                    {
                        "comparisons": [
                            {
                                "name": "exact-center",
                                "source_render": (
                                    "/data/other-case/"
                                    "source-center.tif"
                                ),
                            }
                        ]
                    }
                )
            )

            actual = viewer.evidence_center_render(
                mock.Mock(root=root), "source_render"
            )

        np.testing.assert_array_equal(actual, expected)

    def test_prepared_matched_max_render_is_selectable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            evidence = root / "renders" / "offset-evidence"
            evidence.mkdir(parents=True)
            source = evidence / "source-annotation-level2.tif"
            expected = np.arange(12, dtype=np.uint8).reshape(3, 4)
            tifffile.imwrite(source, expected, metadata=None)
            (evidence / "manifest.json").write_text(
                json.dumps(
                    {
                        "comparisons": [
                            {
                                "name": "annotation-matched-slab",
                                "source_render": str(source),
                            }
                        ]
                    }
                )
            )

            actual = viewer.evidence_comparison_render(
                mock.Mock(root=root),
                "annotation-matched-slab",
                "source_render",
            )

        np.testing.assert_array_equal(actual, expected)

    def test_legacy_shortened_z_center_evidence_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            evidence = root / "renders" / "offset-evidence"
            evidence.mkdir(parents=True)
            source = evidence / "source-center.tif"
            tifffile.imwrite(source, np.zeros((3, 4), dtype=np.uint8))
            (evidence / "manifest.json").write_text(
                json.dumps(
                    {
                        "source_center": {
                            "selected_level_shape_zyx": [17, 3, 4],
                            "full_resolution_shape_zyx": [65, 12, 16],
                        },
                        "comparisons": [
                            {
                                "name": "exact-center",
                                "source_render": str(source),
                            }
                        ],
                    }
                )
            )

            actual = viewer.evidence_center_render(
                mock.Mock(root=root), "source_render"
            )

        self.assertIsNone(actual)

    def test_discovery_allows_missing_native_target_surface_volume(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "open-data" / "2.4um-updated.tifxyz").mkdir(parents=True)
            (root / "open-data" / "7.91um-target.tifxyz").mkdir()
            (root / "hf" / "labels").mkdir(parents=True)
            (root / "hf" / "supervision-masks").mkdir()
            (root / "hf" / "labels" / "label.tif").touch()
            (root / "hf" / "supervision-masks" / "mask.tif").touch()
            (root / "selection.json").write_text(
                json.dumps(
                    {
                        "segment": {"original_volume_id": "source"},
                        "surface_volumes": [
                            {
                                "resolution_um": 2.4,
                                "path": "scroll/updated.zarr",
                            }
                        ],
                        "source_surface_zarrs": [
                            {"path": "ink/source.zarr"}
                        ],
                    }
                )
            )

            case = viewer.discover_case(root)

        self.assertEqual(case.target_resolution, 7.91)
        self.assertEqual(set(case.surface_volume_urls), {2.4})

    def test_discovery_defaults_need_no_private_rclone_remote(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "open-data" / "2.4um-updated.tifxyz").mkdir(parents=True)
            (root / "hf" / "labels").mkdir(parents=True)
            (root / "hf" / "supervision-masks").mkdir()
            (root / "hf" / "labels" / "label.tif").touch()
            (root / "hf" / "supervision-masks" / "mask.tif").touch()
            (root / "selection.json").write_text(
                json.dumps(
                    {
                        "segment": {"original_volume_id": "source"},
                        "surface_volumes": [
                            {
                                "resolution_um": 2.4,
                                "path": "scroll/updated.zarr",
                            }
                        ],
                        "source_surface_zarrs": [
                            {"path": "ink/scroll-x/source.zarr"}
                        ],
                    }
                )
            )

            default_case = viewer.discover_case(root)
            mirrored_case = viewer.discover_case(
                root, ink_rclone_root="my-mirror:bucket/ink"
            )

        # The private ink dataset has no implicit default; the public
        # open-data root reads anonymously without local rclone config.
        self.assertIsNone(default_case.source_surface_volume_url)
        self.assertTrue(
            default_case.surface_volume_urls[2.4].startswith(":s3,")
        )
        self.assertIn(
            "env_auth=false", default_case.surface_volume_urls[2.4]
        )
        self.assertEqual(
            mirrored_case.source_surface_volume_url,
            "my-mirror:bucket/ink/scroll-x/source.zarr",
        )

    def test_cached_rclone_composite_needs_no_remote_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = Path(temp_dir) / "middle3.tif"
            expected = np.arange(12, dtype=np.uint8).reshape(3, 4)
            tifffile.imwrite(cache, expected)
            with mock.patch.object(
                viewer,
                "inspect_rclone_zarr",
                side_effect=AssertionError("unexpected remote read"),
            ):
                actual = viewer.load_middle_three_max(
                    "remote:bucket/render.zarr", cache, preferred_level=2
                )
        np.testing.assert_array_equal(actual, expected)

    def test_uncompressed_render_cache_is_memory_mapped(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "render.tif"
            expected = np.arange(12, dtype=np.uint8).reshape(3, 4)
            tifffile.imwrite(path, expected, metadata=None)

            actual = viewer.read_render_tiff(path)

            self.assertIsInstance(actual, np.memmap)
            np.testing.assert_array_equal(actual, expected)

    def test_centered_thirteen_plane_max_selects_six_each_side(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = Path(temp_dir) / "middle13.tif"
            info = mock.Mock()
            info.shape = (109, 20, 30)

            def write_composite(_info, indices, output, **_kwargs):
                self.assertEqual(indices, list(range(48, 61)))
                tifffile.imwrite(output, np.zeros((20, 30), np.uint8))

            with (
                mock.patch.object(
                    viewer, "inspect_rclone_zarr", return_value=info
                ),
                mock.patch.object(
                    viewer,
                    "extract_rclone_composite",
                    side_effect=write_composite,
                ),
            ):
                viewer.load_middle_three_max(
                    "remote:bucket/render.zarr",
                    cache,
                    preferred_level=2,
                    plane_count=13,
                )

    def test_rejects_http_urls(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = Path(temp_dir) / "middle3.tif"
            with self.assertRaisesRegex(ValueError, "rclone"):
                viewer.load_middle_three_max(
                    "https://example.test/volume.zarr",
                    cache,
                    preferred_level=0,
                )
            self.assertFalse(cache.exists())


class _FakeLayer:
    def __init__(self, data: np.ndarray, name: str, visible: bool) -> None:
        self.data = np.asarray(data)
        self.name = name
        self.visible = visible


class _FakeViewer:
    def __init__(self) -> None:
        self.layers: list[_FakeLayer] = []

    def _add(self, data: np.ndarray, **kwargs) -> _FakeLayer:
        layer = _FakeLayer(data, kwargs["name"], kwargs.get("visible", True))
        self.layers.append(layer)
        return layer

    def add_image(self, data: np.ndarray, **kwargs) -> _FakeLayer:
        return self._add(data, **kwargs)

    def add_points(self, data: np.ndarray, **kwargs) -> _FakeLayer:
        return self._add(data, **kwargs)

    def add_vectors(self, data: np.ndarray, **kwargs) -> _FakeLayer:
        return self._add(data, **kwargs)


class ViewerLayerSemanticsTests(unittest.TestCase):
    def test_common_canvas_pair_rejects_plain_resize_fallback(self) -> None:
        fake = _FakeViewer()

        with self.assertRaisesRegex(
            ValueError, "plain resize is not a geometric registration"
        ):
            viewer.add_image_pair(
                fake,
                "blue",
                "red",
                "source to target",
                np.zeros((2, 3), dtype=np.uint8),
                np.zeros((3, 2), dtype=np.uint8),
                (1.0, 1.0),
                (0.0, 0.0),
                visible=False,
            )

        self.assertEqual(fake.layers, [])

    def test_register_render_uses_both_surfaces_and_selected_affine(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "source.tifxyz"
            target_path = root / "target.tifxyz"
            source_surface = object()
            target_surface = object()
            source_render = np.arange(6, dtype=np.uint8).reshape(2, 3)
            projected = np.arange(12, dtype=np.uint8).reshape(3, 4)
            validity = np.full(projected.shape, 255, dtype=np.uint8)
            affine = np.asarray(
                [
                    [-1.0, 0.0, 0.0, 10.0],
                    [0.0, -1.0, 0.0, 20.0],
                    [0.0, 0.0, 1.0, 30.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            stats = mock.Mock()
            stats.as_dict.return_value = {"mapping_coverage": 1.0}
            report = {
                "max_distance": 4.5,
                "nearest_vertices": 8,
                "tile_size": 64,
                "query_batch_size": 1024,
            }
            case = mock.Mock(root=root)

            with (
                mock.patch.object(
                    viewer,
                    "load_full_surface",
                    side_effect=[source_surface, target_surface],
                ) as load_surface,
                mock.patch.object(
                    viewer,
                    "transfer_surface_array",
                    return_value=(projected, validity, None, stats),
                ) as transfer,
            ):
                actual = viewer.register_render(
                    case,
                    source_path,
                    target_path,
                    source_render,
                    projected.shape,
                    affine,
                    report,
                    "projected.tif",
                    label_offset_render_yx=(1.25, -2.5),
                )

        np.testing.assert_array_equal(actual, projected)
        self.assertEqual(
            load_surface.call_args_list,
            [mock.call(source_path), mock.call(target_path)],
        )
        args, kwargs = transfer.call_args
        self.assertIs(args[0], source_surface)
        self.assertIs(args[1], target_surface)
        np.testing.assert_array_equal(args[2], source_render)
        np.testing.assert_array_equal(kwargs["affine"], affine)
        self.assertEqual(kwargs["output_shape"], projected.shape)
        self.assertEqual(kwargs["label_offset_yx"], (1.25, -2.5))
        self.assertEqual(kwargs["max_distance"], 4.5)
        self.assertEqual(kwargs["nearest_vertices"], 8)

    def test_replacing_case_layers_loads_only_selected_case(self) -> None:
        fake = _FakeViewer()
        fake.layers.append(_FakeLayer(np.zeros((1, 1)), "old", False))
        case_groups = {"old": {"preset": [fake.layers[0]]}}
        selected = mock.Mock()
        selected.name = "selected"
        prepared = mock.Mock(case=selected)
        groups = {"preset": []}
        prepare = mock.Mock(return_value=prepared)
        add_layers = mock.Mock(return_value=groups)

        with mock.patch.object(viewer.gc, "collect") as collect:
            prepare.side_effect = lambda _case: (
                self.assertEqual(fake.layers, []),
                self.assertEqual(case_groups, {}),
                self.assertTrue(collect.called),
                prepared,
            )[-1]
            actual = viewer.replace_case_layers(
                fake,
                case_groups,
                selected,
                prepare,
                add_layers,
            )

        self.assertIs(actual, groups)
        self.assertEqual(case_groups, {"selected": groups})
        self.assertEqual(fake.layers, [])
        collect.assert_called_once_with()
        prepare.assert_called_once_with(selected)
        add_layers.assert_called_once_with(prepared)

    def test_background_case_load_materialises_only_after_worker_returns(
        self,
    ) -> None:
        class Signal:
            def __init__(self) -> None:
                self.callbacks = []

            def connect(self, callback) -> None:
                self.callbacks.append(callback)

            def emit(self, value) -> None:
                for callback in self.callbacks:
                    callback(value)

        class Worker:
            def __init__(self) -> None:
                self.returned = Signal()
                self.errored = Signal()
                self.started = False

            def start(self) -> None:
                self.started = True

        fake = _FakeViewer()
        fake.layers.append(_FakeLayer(np.zeros((1, 1)), "old", False))
        case_groups = {"old": {"preset": [fake.layers[0]]}}
        case = mock.Mock(name="case")
        case.name = "selected"
        prepared = mock.Mock(case=case)
        groups = {"preset": []}
        worker = Worker()
        worker_factory = mock.Mock(return_value=worker)
        materialise = mock.Mock(return_value=groups)
        loaded = mock.Mock()
        errored = mock.Mock()
        active_workers = {}

        with mock.patch.object(viewer.gc, "collect"):
            actual = viewer.load_case_in_background(
                fake,
                case_groups,
                case,
                worker_factory,
                materialise,
                loaded,
                errored,
                active_workers,
            )

        self.assertIs(actual, worker)
        self.assertTrue(worker.started)
        self.assertEqual(fake.layers, [])
        self.assertEqual(case_groups, {})
        self.assertEqual(active_workers, {"selected": worker})
        materialise.assert_not_called()
        loaded.assert_not_called()

        worker.returned.emit(prepared)

        materialise.assert_called_once_with(prepared)
        loaded.assert_called_once_with("selected", groups)
        errored.assert_not_called()
        self.assertEqual(active_workers, {})

    def test_background_case_load_reports_worker_error(self) -> None:
        class Signal:
            def __init__(self) -> None:
                self.callback = None

            def connect(self, callback) -> None:
                self.callback = callback

            def emit(self, value) -> None:
                self.callback(value)

        worker = mock.Mock()
        worker.returned = Signal()
        worker.errored = Signal()
        case = mock.Mock()
        case.name = "broken"
        errored = mock.Mock()
        active_workers = {}

        with mock.patch.object(viewer.gc, "collect"):
            viewer.load_case_in_background(
                _FakeViewer(),
                {},
                case,
                mock.Mock(return_value=worker),
                mock.Mock(),
                mock.Mock(),
                errored,
                active_workers,
            )
        error = RuntimeError("prepare failed")
        worker.errored.emit(error)

        errored.assert_called_once_with("broken", error)
        self.assertEqual(active_workers, {})

    def test_diagnostic_contact_sheet_packs_sparse_tiles(self) -> None:
        source = np.arange(64, dtype=np.uint8).reshape(8, 8)
        target = source + 1
        bounds = [[0, 2, 0, 2], [6, 8, 6, 8]]

        source_sheet, target_sheet = viewer.diagnostic_contact_sheet(
            source, target, bounds
        )

        self.assertEqual(source_sheet.shape, (2, 4))
        np.testing.assert_array_equal(source_sheet[:, :2], source[:2, :2])
        np.testing.assert_array_equal(source_sheet[:, 2:], source[6:, 6:])
        np.testing.assert_array_equal(target_sheet, source_sheet + 1)

    def test_camera_refits_only_for_forced_or_display_mode_changes(self) -> None:
        self.assertFalse(viewer.should_refit_camera(2, 2, False))
        self.assertFalse(viewer.should_refit_camera(3, 3, False))
        self.assertTrue(viewer.should_refit_camera(2, 3, False))
        self.assertTrue(viewer.should_refit_camera(3, 2, False))
        self.assertTrue(viewer.should_refit_camera(2, 2, True))

    def test_active_2d_camera_ignores_unrelated_layer_extents(self) -> None:
        center, zoom = viewer.calculate_2d_camera(
            [
                np.asarray([[0.0, 0.0], [1_000.0, 900.0]]),
                np.asarray([[0.0, 0.0], [1_000.0, 900.0]]),
            ],
            np.asarray([800.0, 600.0]),
        )

        self.assertEqual(center, (500.0, 450.0))
        self.assertAlmostEqual(zoom, 0.92 * (600.0 / 900.0) * 1.6)

    def test_prepare_case_uses_pipeline_selected_affine_matrix(self) -> None:
        points = np.asarray(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        )
        selected = np.eye(4)
        selected[0, 3] = 5.0
        stage_one = np.eye(4)
        stage_one[0, 3] = 2.0
        source_surface = viewer.Surface(
            points - np.asarray([2.0, 0.0, 0.0]),
            np.arange(4),
            (2, 2),
        )
        updated_surface = viewer.Surface(points, np.arange(4), (2, 2))
        target_surface = viewer.Surface(
            points + np.asarray([5.0, 0.0, 0.0]),
            np.arange(4),
            (2, 2),
        )
        case = viewer.Case(
            root=Path("/not-used"),
            name="inverse-selected",
            source_tifxyz=Path("/not-used/source.tifxyz"),
            updated_tifxyz=Path("/not-used/updated.tifxyz"),
            target_tifxyz=Path("/not-used/target.tifxyz"),
            affine_path=Path("/not-used/raw-affine.json"),
            source_label=Path("/not-used/label.tif"),
            source_supervision=Path("/not-used/mask.tif"),
            source_surface_volume_url="https://example.test/source/",
            surface_volume_urls={},
            results=Path("/not-used/results"),
            updated_resolution=2.399,
            target_resolution=9.362,
        )
        reports = {
            "2.399um": {
                "mapping": {},
                "affine": {
                    "direction": "forward",
                    "matrix": stage_one.tolist(),
                },
            },
            "9.362um": {
                "mapping": {},
                "affine": {
                    "direction": "inverse",
                    "matrix": selected.tolist(),
                },
            },
        }

        with (
            mock.patch.object(
                viewer,
                "load_surface",
                side_effect=[
                    source_surface,
                    updated_surface,
                    target_surface,
                ],
            ),
            mock.patch.object(viewer, "load_reports", return_value=reports),
        ):
            prepared = viewer.prepare_case(
                case,
                max_points=100,
                zarr_level=2,
                include_renders=False,
            )

        np.testing.assert_array_equal(prepared.affine, selected)
        np.testing.assert_array_equal(prepared.stage_one_affine, stage_one)
        self.assertAlmostEqual(
            prepared.summary["updated_to_target_nearest_vertex"]["max"],
            0.0,
        )
        # The old->updated comparison must apply the stage-one frame affine;
        # without it the raw 2-voxel offset would remain.
        self.assertAlmostEqual(
            prepared.summary["old_to_updated_nearest_vertex"]["max"],
            0.0,
        )
        self.assertEqual(
            prepared.summary["stage_one_affine"]["matrix"],
            stage_one.tolist(),
        )

    def test_prepare_preview_downsamples_before_projection_and_skips_residuals(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            label = root / "label.tif"
            supervision = root / "supervision.tif"
            tifffile.imwrite(label, np.ones((8, 12), dtype=np.uint8))
            tifffile.imwrite(
                supervision, np.ones((8, 12), dtype=np.uint8)
            )
            points = np.asarray(
                [
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [0.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                ]
            )
            surface = viewer.Surface(points, np.arange(4), (2, 2))
            case = viewer.Case(
                root=root,
                name="preview",
                source_tifxyz=root / "source.tifxyz",
                updated_tifxyz=root / "updated.tifxyz",
                target_tifxyz=None,
                affine_path=None,
                source_label=label,
                source_supervision=supervision,
                source_surface_volume_url="remote:source.zarr",
                surface_volume_urls={2.399: "remote:updated.zarr"},
                results=root / "results",
                updated_resolution=2.399,
                target_resolution=None,
            )
            reports = {
                "2.399um": {
                    "mapping": {},
                    "affine": {
                        "direction": "identity",
                        "matrix": np.eye(4).tolist(),
                    },
                }
            }
            source_render = np.arange(8 * 12, dtype=np.uint8).reshape(8, 12)
            updated_render = source_render + 1

            def projected(*args, **_kwargs):
                return np.zeros(args[4], dtype=np.uint8)

            with (
                mock.patch.object(
                    viewer, "load_surface", side_effect=[surface, surface]
                ),
                mock.patch.object(viewer, "load_reports", return_value=reports),
                mock.patch.object(
                    viewer,
                    "evidence_comparison_render",
                    side_effect=[source_render, updated_render],
                ),
                mock.patch.object(
                    viewer, "_registered_render_cache_spec", return_value={}
                ),
                mock.patch.object(
                    viewer, "_load_registered_render_cache", return_value=None
                ),
                mock.patch.object(
                    viewer, "register_render", side_effect=projected
                ) as register,
                mock.patch.object(viewer, "measure_render_shift") as residual,
                mock.patch.object(
                    viewer, "load_diagnostic_renders"
                ) as diagnostics,
            ):
                prepared = viewer.prepare_case(
                    case,
                    max_points=100,
                    zarr_level=2,
                    include_renders=True,
                    preview_factor=4,
                )

        self.assertEqual(prepared.source_render.shape, (2, 3))
        self.assertEqual(prepared.renders["updated"].shape, (2, 3))
        self.assertEqual(prepared.registered_hf_to_updated_render.shape, (2, 3))
        self.assertEqual(prepared.summary["preview"]["downsample_factor"], 4)
        self.assertTrue(
            all("preview4" in call.args[7] for call in register.call_args_list)
        )
        residual.assert_not_called()
        diagnostics.assert_not_called()

    def test_stage_ct_is_below_its_matching_annotation_layers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "results"
            results.mkdir()
            source_label = root / "source-label.tif"
            source_supervision = root / "source-supervision.tif"
            source_values = np.asarray([[1, 0], [0, 0]], dtype=np.uint8)
            updated_values = np.asarray([[0, 1], [0, 0]], dtype=np.uint8)
            target_values = np.asarray([[0, 0], [1, 0]], dtype=np.uint8)
            tifffile.imwrite(source_label, source_values)
            tifffile.imwrite(source_supervision, source_values)
            for name, values in (
                ("inklabels-2.399um.tif", updated_values),
                ("supervision-2.399um.tif", updated_values),
                ("inklabels-2.399um.valid.tif", np.full((2, 2), 255, np.uint8)),
                ("inklabels-9.362um.tif", target_values),
                ("supervision-9.362um.tif", target_values),
                ("inklabels-9.362um.valid.tif", np.full((2, 2), 255, np.uint8)),
            ):
                tifffile.imwrite(results / name, values)

            points = np.asarray(
                [
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [0.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                ]
            )
            surface = viewer.Surface(points, np.arange(4), (2, 2))
            case = viewer.Case(
                root=root,
                name="example",
                source_tifxyz=root / "source.tifxyz",
                updated_tifxyz=root / "updated.tifxyz",
                target_tifxyz=root / "target.tifxyz",
                affine_path=root / "affine.json",
                source_label=source_label,
                source_supervision=source_supervision,
                source_surface_volume_url="https://example.test/source/",
                surface_volume_urls={},
                results=results,
                updated_resolution=2.399,
                target_resolution=9.362,
            )
            prepared = viewer.PreparedCase(
                case=case,
                summary={},
                surfaces={
                    "source": surface,
                    "updated": surface,
                    "target": surface,
                },
                affine=np.eye(4),
                source_render=np.full((4, 4), 10, dtype=np.uint8),
                renders={
                    "updated": np.full((4, 4), 20, dtype=np.uint8),
                    "target": np.full((2, 2), 30, dtype=np.uint8),
                },
                registered_hf_to_updated_render=np.full(
                    (4, 4), 15, dtype=np.uint8
                ),
                registered_hf_label_to_updated=np.full(
                    (4, 4), 17, dtype=np.uint8
                ),
                registered_hf_supervision_to_updated=np.full(
                    (4, 4), 18, dtype=np.uint8
                ),
                registered_source_render=np.full(
                    (2, 2), 25, dtype=np.uint8
                ),
                diagnostics={
                    "self_center": (
                        np.full((4, 4), 12, dtype=np.uint8),
                        np.full((4, 4), 13, dtype=np.uint8),
                    )
                },
                stage_one_affine=np.asarray(
                    [
                        [1.0, 0.0, 0.0, 2.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
            )
            fake = _FakeViewer()

            groups = viewer.add_case_layers(
                fake,
                prepared,
                _case_index=0,
                vector_count=2,
                colormaps={
                    "blue": "blue",
                    "red": "red",
                    "green": "green",
                    "orange": "orange",
                    "ink": "ink",
                    "supervision": "supervision",
                },
            )

        for key in (
            "aligned_original",
            "aligned_updated",
            "aligned_final",
        ):
            ct, ink, mask = groups[key]
            self.assertLess(fake.layers.index(ct), fake.layers.index(ink))
            self.assertLess(fake.layers.index(ct), fake.layers.index(mask))
        self.assertIn("native HF", groups["aligned_original"][0].name)
        self.assertIn("transferred", groups["aligned_updated"][1].name)
        self.assertIn("updated 2.399um", groups["label_same"][0].name)
        self.assertIn("transferred", groups["label_same"][1].name)
        self.assertIn("updated 2.399um", groups["validity_same"][0].name)
        self.assertIn("old→updated", groups["validity_same"][1].name)
        self.assertIn("native 9.362um", groups["validity_cross"][0].name)
        self.assertIn("composed", groups["validity_cross"][1].name)
        np.testing.assert_array_equal(
            groups["render_same"][0].data,
            prepared.registered_hf_to_updated_render,
        )
        self.assertEqual(len(groups["render_same"]), 6)
        self.assertIn("LABEL transferred", groups["render_same"][2].name)
        self.assertIn(
            "SUPERVISION transferred", groups["render_same"][3].name
        )
        self.assertIn(
            "LABEL original HF projected", groups["render_same"][4].name
        )
        self.assertIn(
            "SUPERVISION original HF projected",
            groups["render_same"][5].name,
        )
        self.assertIn("self_center", groups)
        # Stage-one 3D comparisons must put the source points into the
        # updated volume frame before displaying surfaces and annotations.
        np.testing.assert_array_equal(
            groups["surface_same"][0].data[0], [1.0, 0.0, 2.0]
        )
        np.testing.assert_array_equal(
            groups["label3d_same"][0].data[0], [1.0, 0.0, 2.0]
        )

    def test_stage_one_only_case_skips_target_layers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "results"
            results.mkdir()
            source_label = root / "source-label.tif"
            source_supervision = root / "source-supervision.tif"
            values = np.asarray([[1, 0], [0, 0]], dtype=np.uint8)
            tifffile.imwrite(source_label, values)
            tifffile.imwrite(source_supervision, values)
            for name in (
                "inklabels-updated-2um.tif",
                "supervision-mask-updated-2um.tif",
                "inklabels-updated-2um.valid.tif",
            ):
                tifffile.imwrite(results / name, values)
            (results / "inklabels-updated-2um.report.json").write_text("{}")

            points = np.asarray(
                [
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [0.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                ]
            )
            surface = viewer.Surface(points, np.arange(4), (2, 2))
            case = viewer.Case(
                root=root,
                name="stage-one-only",
                source_tifxyz=root / "source.tifxyz",
                updated_tifxyz=root / "updated.tifxyz",
                target_tifxyz=None,
                affine_path=None,
                source_label=source_label,
                source_supervision=source_supervision,
                source_surface_volume_url="https://example.test/source/",
                surface_volume_urls={},
                results=results,
                updated_resolution=2.4,
                target_resolution=None,
            )
            self.assertEqual(case.updated_stage, "updated-2um")
            self.assertIsNone(case.target_stage)
            prepared = viewer.PreparedCase(
                case=case,
                summary={},
                surfaces={"source": surface, "updated": surface},
                affine=None,
                source_render=np.full((4, 4), 10, dtype=np.uint8),
                renders={"updated": np.full((4, 4), 20, dtype=np.uint8)},
                registered_hf_to_updated_render=np.full(
                    (4, 4), 15, dtype=np.uint8
                ),
                registered_source_render=None,
            )
            fake = _FakeViewer()

            groups = viewer.add_case_layers(
                fake,
                prepared,
                _case_index=0,
                vector_count=2,
                colormaps={
                    "blue": "blue",
                    "red": "red",
                    "green": "green",
                    "orange": "orange",
                    "ink": "ink",
                    "supervision": "supervision",
                },
            )

        for key in (
            "render_same",
            "aligned_original",
            "aligned_updated",
            "label_same",
            "supervision_same",
            "validity_same",
            "surface_same",
            "label3d_same",
            "supervision3d_same",
        ):
            self.assertIn(key, groups)
        for key in (
            "render_registered",
            "aligned_final",
            "render_label",
            "label_cross",
            "supervision_cross",
            "validity_cross",
            "surface_affine",
            "label3d_affine",
            "supervision3d_affine",
        ):
            self.assertNotIn(key, groups)
        self.assertEqual(len(groups["render_same"]), 4)
        self.assertIn("LABEL transferred", groups["render_same"][2].name)
        self.assertIn(
            "SUPERVISION transferred", groups["render_same"][3].name
        )
        self.assertIn("updated updated-2um", groups["label_same"][0].name)


if __name__ == "__main__":
    unittest.main()
