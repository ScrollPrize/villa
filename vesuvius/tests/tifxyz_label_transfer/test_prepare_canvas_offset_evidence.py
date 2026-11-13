from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import tifffile
from vc_reader import images
from vc_reader.reader import ZarrWriter, create_group

from vesuvius.tifxyz_label_transfer.estimate_canvas_offset_evidence import (
    assess_evidence,
    main as estimate_evidence_main,
)
from vesuvius.tifxyz_label_transfer.prepare_canvas_offset_evidence import (
    DEFAULT_AWS_CREDENTIALS_FILE,
    DEFAULT_INK_ROOT,
    DEFAULT_OPEN_DATA_ROOT,
    ZarrLevel,
    _copy_annotation_render,
    _load_aws_credentials,
    _preserves_z_planes,
    _source_dataset_remote,
    _source_resolution_um,
    _surface_z_step_um,
    build_parser as build_prepare_parser,
    extract_composite,
    inspect_zarr,
    main as prepare_evidence_main,
)


class CompositeTests(unittest.TestCase):
    def test_exact_center_requires_z_plane_preserving_level(self) -> None:
        preserved = SimpleNamespace(
            shape=(65, 20, 30),
            level_zero_metadata={"shape": [65, 80, 120]},
        )
        shortened = SimpleNamespace(
            shape=(17, 20, 30),
            level_zero_metadata={"shape": [65, 80, 120]},
        )

        self.assertTrue(_preserves_z_planes(preserved))
        self.assertFalse(_preserves_z_planes(shortened))

    def test_annotation_copy_takes_the_one_shipped_max(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            dataset.mkdir()
            (dataset / "seg_max_10_20.tif").write_bytes(b"render")
            (dataset / "other.tif").write_bytes(b"no")
            output, indices = _copy_annotation_render(str(dataset), root / "out", False)
            self.assertEqual(indices, (10, 20))
            self.assertEqual(output.read_bytes(), b"render")
            self.assertEqual(list((root / "out").glob("*.partial")), [])

    def test_source_resolution_uses_source_zarr_provenance(self) -> None:
        source = ZarrLevel(
            remote="s3://remote/source",
            level="2",
            attrs={"source_zarr": "/volumes/scan_2.403um_data.zarr"},
            metadata={"shape": [1, 1, 1], "chunks": [1, 1, 1]},
            level_zero_metadata={"shape": [1, 1, 1]},
            scale_zyx=(1.0, 4.0, 4.0),
        )
        resolution, provenance = _source_resolution_um(
            {"segment": {"original_volume_id": "wrong"}},
            source,
            9.362,
            None,
        )
        self.assertEqual(resolution, 2.403)
        self.assertEqual(provenance, "source surface Zarr provenance")

    def test_surface_z_step_distinguishes_physical_and_legacy_metadata(self) -> None:
        physical = ZarrLevel(
            remote="s3://remote/physical",
            level="2",
            attrs={
                "multiscales": [
                    {"axes": [{"name": "z", "unit": "micrometer"}]}
                ],
                "slice_step": 3.0,
            },
            metadata={"shape": [1, 1, 1], "chunks": [1, 1, 1]},
            level_zero_metadata={"shape": [1, 1, 1]},
            scale_zyx=(2.399, 9.596, 9.596),
        )
        legacy = ZarrLevel(
            **{
                **physical.__dict__,
                "attrs": {"slice_step": 2.0},
                "scale_zyx": (1.0, 4.0, 4.0),
            }
        )
        self.assertEqual(_surface_z_step_um(physical, 2.399), 2.399)
        self.assertEqual(_surface_z_step_um(legacy, 2.399), 4.798)

    def test_surface_z_step_distinguishes_level_frames(self) -> None:
        def multiscale(unit: str | None, zero_z: float, level_z: float):
            axes = [{"name": "z"}]
            if unit:
                axes = [{"name": "z", "unit": unit}]
            return {
                "multiscales": [
                    {
                        "axes": axes,
                        "datasets": [
                            {
                                "path": "0",
                                "coordinateTransformations": [
                                    {
                                        "type": "scale",
                                        "scale": [zero_z, 1.0, 1.0],
                                    }
                                ],
                            },
                            {
                                "path": "2",
                                "coordinateTransformations": [
                                    {
                                        "type": "scale",
                                        "scale": [level_z, 4.0, 4.0],
                                    }
                                ],
                            },
                        ],
                    }
                ]
            }

        physical = ZarrLevel(
            remote="s3://remote/physical",
            level="2",
            attrs=multiscale("micrometer", 2.399, 4.798),
            metadata={"shape": [1, 1, 1], "chunks": [1, 1, 1]},
            level_zero_metadata={"shape": [1, 1, 1]},
            scale_zyx=(4.798, 9.596, 9.596),
        )
        # Annotation provenance indices are level-0; the matched slab
        # thickness must use the level-0 step, not the selected level's.
        self.assertEqual(
            _surface_z_step_um(physical, 2.399, level_zero=True), 2.399
        )
        self.assertEqual(_surface_z_step_um(physical, 2.399), 4.798)

        legacy = ZarrLevel(
            **{
                **physical.__dict__,
                "attrs": {
                    **multiscale(None, 1.0, 2.0),
                    "slice_step": 3.0,
                },
                "scale_zyx": (2.0, 4.0, 4.0),
            }
        )
        # slice_step is a level-0 quantity; the selected-level step scales
        # by the declared Z downsample factor.
        self.assertEqual(
            _surface_z_step_um(legacy, 2.0, level_zero=True), 6.0
        )
        self.assertEqual(_surface_z_step_um(legacy, 2.0), 12.0)

    def test_composite_is_the_maximum_of_the_selected_layers(self) -> None:
        volume = np.arange(5 * 3 * 4, dtype=np.uint8).reshape(5, 3, 4)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "render.zarr"
            create_group(root, attrs={
                "multiscales": [{"datasets": [{"path": "0"}]}],
            })
            for level in ("0",):
                writer = ZarrWriter.create(root / level, shape=volume.shape, chunks=(5, 2, 2), dtype=np.uint8, auth=None)
                writer[:] = volume
            info = inspect_zarr(str(root), 0)
            output = Path(directory) / "composite.tif"
            with mock.patch(
                "vesuvius.tifxyz_label_transfer.prepare_canvas_offset_evidence."
                "images.memmap",
                wraps=images.memmap,
            ) as memmap:
                audit = extract_composite(info, [1, 2, 3], output, workers=1, overwrite=False)
            np.testing.assert_array_equal(tifffile.imread(output), volume[1:4].max(axis=0))
            self.assertEqual(audit["transport"], "vc_reader")
            memmap.assert_called_once()
            self.assertEqual(list(Path(directory).glob("*.partial.tif")), [])

            cached = extract_composite(info, [1, 2, 3], output, workers=1, overwrite=False)
            self.assertEqual(cached, audit)


class EvidenceAssessmentTests(unittest.TestCase):
    def test_requires_agreement_between_independent_evidence(self) -> None:
        good = {
            "converged": True,
            "translation_model_valid": True,
        }
        approved = assess_evidence(
            [
                {
                    **good,
                    "offset_yx_full_resolution_px": [20.0, -12.0],
                },
                {
                    **good,
                    "offset_yx_full_resolution_px": [20.5, -12.5],
                },
            ],
            minimum_evidence=2,
            maximum_disagreement_full_px=2.0,
        )
        self.assertTrue(approved["approved"])
        self.assertAlmostEqual(
            approved["maximum_evidence_disagreement_full_px"],
            np.sqrt(0.5),
        )

        rejected = assess_evidence(
            [
                {
                    **good,
                    "offset_yx_full_resolution_px": [20.0, -12.0],
                },
                {
                    **good,
                    "offset_yx_full_resolution_px": [25.0, -12.0],
                },
            ],
            minimum_evidence=2,
            maximum_disagreement_full_px=2.0,
        )
        self.assertFalse(rejected["approved"])

    def test_unusable_comparison_does_not_hide_later_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            case_dir = Path(directory) / "case"
            (case_dir / "hf" / "source.tifxyz").mkdir(parents=True)
            (case_dir / "open-data" / "2.399um-volume.tifxyz").mkdir(
                parents=True
            )
            manifest = case_dir / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "comparisons": [
                            {
                                "name": "unusable",
                                "source_render": "source-a.tif",
                                "target_render": "target-a.tif",
                            },
                            {
                                "name": "usable",
                                "source_render": "source-b.tif",
                                "target_render": "target-b.tif",
                            },
                        ]
                    }
                )
            )
            output = case_dir / "offset.json"
            surface = SimpleNamespace(full_resolution_shape=(100, 200))
            success = {
                "converged": True,
                "translation_model_valid": True,
                "offset_yx_render_px": [1.0, -2.0],
            }
            with (
                mock.patch(
                    "vesuvius.tifxyz_label_transfer.estimate_canvas_offset_evidence."
                    "load_surface",
                    return_value=surface,
                ),
                mock.patch(
                    "vesuvius.tifxyz_label_transfer.estimate_canvas_offset_evidence."
                    "read_image",
                    return_value=np.zeros((25, 50), dtype=np.uint8),
                ),
                mock.patch(
                    "vesuvius.tifxyz_label_transfer.estimate_canvas_offset_evidence."
                    "estimate_canvas_offset",
                    side_effect=[ValueError("no usable tiles"), success],
                ),
            ):
                result = estimate_evidence_main(
                    [
                        "--case-dir",
                        str(case_dir),
                        "--manifest",
                        str(manifest),
                        "--output",
                        str(output),
                    ]
                )

            self.assertEqual(result, 1)
            document = json.loads(output.read_text())
            self.assertEqual(document["accepted_evidence"], 1)
            self.assertEqual(document["evidence"][0]["error"], "no usable tiles")
            np.testing.assert_allclose(
                document["canvas_offset_yx_full_resolution_px"], [4.0, -8.0]
            )


class EvidencePreparationModeTests(unittest.TestCase):
    def test_exact_center_only_skips_annotation_slab_fetch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            case_dir = Path(directory) / "case"
            case_dir.mkdir()
            (case_dir / "selection.json").write_text(
                json.dumps(
                    {
                        "source_surface_zarrs": [
                            {"path": "dataset/source.zarr"}
                        ]
                    }
                )
            )
            zarr = SimpleNamespace(
                shape=(17, 20, 30),
                level=2,
                level_zero_metadata={"shape": [17, 80, 120]},
            )
            with (
                mock.patch(
                    "vesuvius.tifxyz_label_transfer.prepare_canvas_offset_evidence."
                    "_load_aws_credentials",
                    return_value=None,
                ),
                mock.patch(
                    "vesuvius.tifxyz_label_transfer.prepare_canvas_offset_evidence."
                    "_source_dataset_remote",
                    return_value="s3://remote/dataset",
                ),
                mock.patch(
                    "vesuvius.tifxyz_label_transfer.prepare_canvas_offset_evidence."
                    "_target_zarr_remote",
                    return_value="s3://remote/target.zarr",
                ),
                mock.patch(
                    "vesuvius.tifxyz_label_transfer.prepare_canvas_offset_evidence."
                    "_surface_resolution_for_remote",
                    return_value=2.399,
                ),
                mock.patch(
                    "vesuvius.tifxyz_label_transfer.prepare_canvas_offset_evidence."
                    "inspect_zarr",
                    return_value=zarr,
                ),
                mock.patch(
                    "vesuvius.tifxyz_label_transfer.prepare_canvas_offset_evidence."
                    "extract_composite",
                    return_value={"cached": False},
                ) as extract,
                mock.patch(
                    "vesuvius.tifxyz_label_transfer.prepare_canvas_offset_evidence."
                    "_copy_annotation_render"
                ) as copy_annotation,
            ):
                result = prepare_evidence_main(
                    [
                        "--case-dir",
                        str(case_dir),
                        "--ink-root",
                        "test-remote:ink-dataset",
                        "--exact-center-only",
                    ]
                )

            self.assertEqual(result, 0)
            self.assertEqual(extract.call_count, 2)
            copy_annotation.assert_not_called()
            manifest = json.loads(
                (
                    case_dir / "renders" / "offset-evidence" / "manifest.json"
                ).read_text()
            )
            self.assertEqual(
                [item["name"] for item in manifest["comparisons"]],
                ["exact-center"],
            )
            self.assertIsNone(manifest["annotation_render"])
            self.assertIsNone(manifest["target_matched_max"])

    def test_shortened_z_level_is_not_claimed_as_exact_center(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            case_dir = Path(directory) / "case"
            case_dir.mkdir()
            (case_dir / "selection.json").write_text(
                json.dumps(
                    {
                        "source_surface_zarrs": [
                            {"path": "dataset/source.zarr"}
                        ]
                    }
                )
            )
            source = SimpleNamespace(
                shape=(17, 20, 30),
                level=2,
                level_zero_metadata={"shape": [65, 80, 120]},
            )
            target = SimpleNamespace(
                shape=(109, 20, 30),
                level=2,
                level_zero_metadata={"shape": [109, 80, 120]},
            )
            with (
                mock.patch(
                    "vesuvius.tifxyz_label_transfer.prepare_canvas_offset_evidence."
                    "_load_aws_credentials",
                    return_value=None,
                ),
                mock.patch(
                    "vesuvius.tifxyz_label_transfer.prepare_canvas_offset_evidence."
                    "_source_dataset_remote",
                    return_value="s3://remote/dataset",
                ),
                mock.patch(
                    "vesuvius.tifxyz_label_transfer.prepare_canvas_offset_evidence."
                    "_target_zarr_remote",
                    return_value="s3://remote/target.zarr",
                ),
                mock.patch(
                    "vesuvius.tifxyz_label_transfer.prepare_canvas_offset_evidence."
                    "_surface_resolution_for_remote",
                    return_value=2.399,
                ),
                mock.patch(
                    "vesuvius.tifxyz_label_transfer.prepare_canvas_offset_evidence."
                    "inspect_zarr",
                    side_effect=[source, target],
                ),
                mock.patch(
                    "vesuvius.tifxyz_label_transfer.prepare_canvas_offset_evidence."
                    "extract_composite",
                    return_value={"cached": False},
                ) as extract,
            ):
                result = prepare_evidence_main(
                    [
                        "--case-dir",
                        str(case_dir),
                        "--ink-root",
                        "test-remote:ink-dataset",
                        "--exact-center-only",
                    ]
                )

            self.assertEqual(result, 0)
            self.assertEqual(extract.call_count, 1)
            manifest = json.loads(
                (
                    case_dir / "renders" / "offset-evidence" / "manifest.json"
                ).read_text()
            )
            self.assertEqual(manifest["comparisons"], [])
            self.assertFalse(manifest["source_center"]["available"])


class PortabilityTests(unittest.TestCase):
    def test_credentials_default_is_none_and_nothing_implicit_is_read(
        self,
    ) -> None:
        self.assertIsNone(DEFAULT_AWS_CREDENTIALS_FILE)
        args = build_prepare_parser().parse_args(["--case-dir", "case"])
        self.assertIsNone(args.aws_credentials_file)
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertIsNone(_load_aws_credentials(None))

    def test_open_data_default_is_the_public_bucket(self) -> None:
        self.assertEqual(DEFAULT_OPEN_DATA_ROOT, "s3://vesuvius-challenge-open-data")

    def test_private_ink_root_has_no_default_and_errors_actionably(
        self,
    ) -> None:
        self.assertIsNone(DEFAULT_INK_ROOT)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(SystemExit) as caught:
                prepare_evidence_main(["--case-dir", directory])
        self.assertIn("--ink-root", str(caught.exception))

    def test_source_dataset_remote_joins_the_root(self) -> None:
        selection = {
            "source_surface_zarrs": [
                {"path": "ink/scroll-x/source.zarr"}
            ]
        }
        self.assertEqual(
            _source_dataset_remote(
                selection, "s3://bucket/datasets/ink"
            ),
            "s3://bucket/datasets/ink/scroll-x",
        )


if __name__ == "__main__":
    unittest.main()
