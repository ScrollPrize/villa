from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import tifffile

from tifxyz_label_transfer.estimate_canvas_offset_evidence import (
    assess_evidence,
    main as estimate_evidence_main,
)
from tifxyz_label_transfer.prepare_canvas_offset_evidence import (
    ANNOTATION_COPY_TIMEOUT_SECONDS,
    DEFAULT_AWS_CREDENTIALS_FILE,
    DEFAULT_INK_ROOT,
    DEFAULT_OPEN_DATA_ROOT,
    RCLONE_PROCESS_TIMEOUT_SECONDS,
    ZarrLevel,
    _batched_range_payloads,
    _bulk_copy_compressed_chunks,
    _copy_annotation_render,
    _load_aws_credentials,
    _preserves_z_planes,
    _source_dataset_remote,
    _source_resolution_um,
    _surface_z_step_um,
    build_parser as build_prepare_parser,
    extract_composite,
    main as prepare_evidence_main,
    _run_rclone,
)


class RcloneCompositeTests(unittest.TestCase):
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

    def test_rclone_process_timeout_allows_slow_range_reads(self) -> None:
        completed = mock.Mock(returncode=0, stdout=b"payload", stderr=b"")
        with (
            mock.patch("subprocess.run", return_value=completed) as run,
            mock.patch(
                "tifxyz_label_transfer.prepare_canvas_offset_evidence."
                "shutil.which",
                return_value="/usr/bin/setpriv",
            ),
            mock.patch(
                "tifxyz_label_transfer.prepare_canvas_offset_evidence."
                "sys.platform",
                "linux",
            ),
        ):
            self.assertEqual(_run_rclone(["cat", "remote:path"]), b"payload")
        self.assertEqual(
            run.call_args.kwargs["timeout"], RCLONE_PROCESS_TIMEOUT_SECONDS
        )
        self.assertEqual(
            run.call_args.args[0][:5],
            [
                "/usr/bin/setpriv",
                "--pdeathsig",
                "TERM",
                "rclone",
                "cat",
            ],
        )

    def test_large_annotation_copy_uses_extended_process_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with (
                mock.patch(
                    "tifxyz_label_transfer.prepare_canvas_offset_evidence."
                    "_list_files",
                    return_value=["segment_max_22_42.tif"],
                ),
                mock.patch(
                    "tifxyz_label_transfer.prepare_canvas_offset_evidence."
                    "_run_rclone",
                    side_effect=lambda arguments, **_: Path(
                        arguments[2]
                    ).touch(),
                ) as run_rclone,
            ):
                output, indices = _copy_annotation_render(
                    "remote:dataset", Path(directory), overwrite=True
                )

            self.assertEqual(indices, (22, 42))
            self.assertEqual(
                output,
                Path(directory) / "source-annotation-segment_max_22_42.tif",
            )
            self.assertEqual(
                run_rclone.call_args.kwargs["timeout_seconds"],
                ANNOTATION_COPY_TIMEOUT_SECONDS,
            )
            partial = Path(run_rclone.call_args.args[0][2])
            self.assertNotEqual(partial, output)
            self.assertTrue(partial.name.endswith(".partial"))
            self.assertTrue(output.is_file())
            self.assertFalse(partial.exists())

    def test_compressed_chunks_use_one_parallel_rclone_copy(self) -> None:
        info = ZarrLevel(
            remote="remote:bucket/render.zarr",
            level="2",
            attrs={},
            metadata={
                "shape": [5, 2, 4],
                "chunks": [5, 2, 2],
                "dtype": "|u1",
                "compressor": {"id": "zlib", "level": 1},
                "filters": None,
                "order": "C",
                "dimension_separator": "/",
            },
            level_zero_metadata={"shape": [5, 2, 4]},
            scale_zyx=(1.0, 4.0, 4.0),
        )
        calls: list[str] = []

        def fake_rclone(arguments: list[str], **_: object) -> bytes:
            calls.append(arguments[0])
            if arguments[0] == "lsf":
                return b"0/0/0\n0/0/1\n"
            destination = Path(arguments[2])
            files_path = Path(arguments[arguments.index("--files-from") + 1])
            for relative in files_path.read_text().splitlines():
                output = destination / relative
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(relative.encode())
            return b""

        with tempfile.TemporaryDirectory() as directory:
            with mock.patch(
                "tifxyz_label_transfer.prepare_canvas_offset_evidence."
                "_run_rclone",
                side_effect=fake_rclone,
            ):
                copied = _bulk_copy_compressed_chunks(
                    info, 0, [(0, 0), (0, 1)], Path(directory), 8
                )
            self.assertEqual([item[:2] for item in copied], [(0, 0), (0, 1)])

        self.assertEqual(calls, ["lsf", "copy"])

    def test_source_resolution_uses_source_zarr_provenance(self) -> None:
        source = ZarrLevel(
            remote="remote:source",
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
            remote="remote:physical",
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
            remote="remote:physical",
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

    def test_batched_ranges_preserve_remote_lexical_chunk_order(self) -> None:
        info = ZarrLevel(
            remote="remote:bucket/render.zarr",
            level="2",
            attrs={},
            metadata={
                "shape": [5, 4, 4],
                "chunks": [5, 2, 2],
                "dtype": "|u1",
                "compressor": None,
                "filters": None,
                "order": "C",
                "dimension_separator": "/",
            },
            level_zero_metadata={"shape": [5, 4, 4]},
            scale_zyx=(1.0, 1.0, 1.0),
        )

        def fake_rclone(arguments: list[str]) -> bytes:
            if arguments[0] == "lsf":
                return b"1/0\n0/1\n0/0\n"
            includes = [
                arguments[index + 1].lstrip("/")
                for index, value in enumerate(arguments)
                if value == "--include"
            ]
            values = {
                "0/0": b"aa",
                "0/1": b"bb",
                "1/0": b"cc",
            }
            return b"".join(values[name] for name in sorted(includes))

        with mock.patch(
            "tifxyz_label_transfer.prepare_canvas_offset_evidence."
            "_run_rclone",
            side_effect=fake_rclone,
        ):
            payloads = _batched_range_payloads(
                info, 0, byte_offset=3, byte_count=2, workers=2
            )

        self.assertEqual(
            payloads,
            [(0, 0, b"aa"), (0, 1, b"bb"), (1, 0, b"cc")],
        )

    def test_uncompressed_chunks_fetch_only_selected_z_bytes(self) -> None:
        volume = np.arange(5 * 3 * 4, dtype=np.uint8).reshape(5, 3, 4)
        info = ZarrLevel(
            remote="remote:bucket/render.zarr",
            level="2",
            attrs={},
            metadata={
                "shape": [5, 3, 4],
                "chunks": [5, 2, 2],
                "dtype": "|u1",
                "compressor": None,
                "filters": None,
                "order": "C",
                "dimension_separator": "/",
            },
            level_zero_metadata={
                "shape": [5, 3, 4],
                "dtype": "|u1",
            },
            scale_zyx=(1.0, 4.0, 4.0),
        )
        requested: list[tuple[int, int]] = []

        def fake_cat_range(
            remote: str, offset: int, count: int
        ) -> bytes:
            parts = remote.split("/")
            y_chunk, x_chunk = int(parts[-2]), int(parts[-1])
            chunk = np.zeros((5, 2, 2), dtype=np.uint8)
            source = volume[
                :,
                y_chunk * 2 : (y_chunk + 1) * 2,
                x_chunk * 2 : (x_chunk + 1) * 2,
            ]
            chunk[:, : source.shape[1], : source.shape[2]] = source
            requested.append((offset, count))
            return chunk.tobytes()[offset : offset + count]

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "center-max.tif"
            with (
                mock.patch(
                    "tifxyz_label_transfer.prepare_canvas_offset_evidence."
                    "_cat_range",
                    side_effect=fake_cat_range,
                ),
                mock.patch(
                    "tifxyz_label_transfer.prepare_canvas_offset_evidence."
                    "tifffile.memmap",
                    wraps=tifffile.memmap,
                ) as memmap,
            ):
                audit = extract_composite(
                    info,
                    [1, 2, 3],
                    output,
                    workers=1,
                    overwrite=False,
                )

            np.testing.assert_array_equal(
                tifffile.imread(output), volume[1:4].max(axis=0)
            )
            self.assertEqual(requested, [(4, 12)] * 4)
            self.assertEqual(audit["transferred_bytes"], 48)
            self.assertEqual(audit["transport"], "rclone")
            memmap.assert_called_once()
            self.assertEqual(list(Path(directory).glob("*.partial.tif")), [])
            self.assertEqual(
                list(Path(directory).glob(".tifxyz-compressed-chunks-*")),
                [],
            )


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
                    "tifxyz_label_transfer.estimate_canvas_offset_evidence."
                    "load_surface",
                    return_value=surface,
                ),
                mock.patch(
                    "tifxyz_label_transfer.estimate_canvas_offset_evidence."
                    "read_image",
                    return_value=np.zeros((25, 50), dtype=np.uint8),
                ),
                mock.patch(
                    "tifxyz_label_transfer.estimate_canvas_offset_evidence."
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
                    "tifxyz_label_transfer.prepare_canvas_offset_evidence."
                    "_load_aws_credentials",
                    return_value=None,
                ),
                mock.patch(
                    "tifxyz_label_transfer.prepare_canvas_offset_evidence."
                    "_source_dataset_remote",
                    return_value="remote:dataset",
                ),
                mock.patch(
                    "tifxyz_label_transfer.prepare_canvas_offset_evidence."
                    "_target_zarr_remote",
                    return_value="remote:target.zarr",
                ),
                mock.patch(
                    "tifxyz_label_transfer.prepare_canvas_offset_evidence."
                    "_surface_resolution_for_remote",
                    return_value=2.399,
                ),
                mock.patch(
                    "tifxyz_label_transfer.prepare_canvas_offset_evidence."
                    "inspect_zarr",
                    return_value=zarr,
                ),
                mock.patch(
                    "tifxyz_label_transfer.prepare_canvas_offset_evidence."
                    "extract_composite",
                    return_value={"cached": False},
                ) as extract,
                mock.patch(
                    "tifxyz_label_transfer.prepare_canvas_offset_evidence."
                    "_copy_annotation_render"
                ) as copy_annotation,
            ):
                result = prepare_evidence_main(
                    [
                        "--case-dir",
                        str(case_dir),
                        "--ink-rclone-root",
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
                    "tifxyz_label_transfer.prepare_canvas_offset_evidence."
                    "_load_aws_credentials",
                    return_value=None,
                ),
                mock.patch(
                    "tifxyz_label_transfer.prepare_canvas_offset_evidence."
                    "_source_dataset_remote",
                    return_value="remote:dataset",
                ),
                mock.patch(
                    "tifxyz_label_transfer.prepare_canvas_offset_evidence."
                    "_target_zarr_remote",
                    return_value="remote:target.zarr",
                ),
                mock.patch(
                    "tifxyz_label_transfer.prepare_canvas_offset_evidence."
                    "_surface_resolution_for_remote",
                    return_value=2.399,
                ),
                mock.patch(
                    "tifxyz_label_transfer.prepare_canvas_offset_evidence."
                    "inspect_zarr",
                    side_effect=[source, target],
                ),
                mock.patch(
                    "tifxyz_label_transfer.prepare_canvas_offset_evidence."
                    "extract_composite",
                    return_value={"cached": False},
                ) as extract,
            ):
                result = prepare_evidence_main(
                    [
                        "--case-dir",
                        str(case_dir),
                        "--ink-rclone-root",
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

    def test_open_data_default_is_anonymous_inline_remote(self) -> None:
        self.assertTrue(DEFAULT_OPEN_DATA_ROOT.startswith(":s3,"))
        self.assertIn("env_auth=false", DEFAULT_OPEN_DATA_ROOT)

    def test_private_ink_root_has_no_default_and_errors_actionably(
        self,
    ) -> None:
        self.assertIsNone(DEFAULT_INK_ROOT)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(SystemExit) as caught:
                prepare_evidence_main(["--case-dir", directory])
        self.assertIn("--ink-rclone-root", str(caught.exception))

    def test_source_dataset_remote_accepts_any_rclone_remote(self) -> None:
        selection = {
            "source_surface_zarrs": [
                {"path": "ink/scroll-x/source.zarr"}
            ]
        }
        self.assertEqual(
            _source_dataset_remote(
                selection, "my-mirror:bucket/datasets/ink"
            ),
            "my-mirror:bucket/datasets/ink/scroll-x",
        )


if __name__ == "__main__":
    unittest.main()
