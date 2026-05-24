import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


MODULE_DIR = Path(__file__).resolve().parents[1]
MODULE_PATH = MODULE_DIR / "build_deformation_field.py"


def _load_field_builder():
    module_name = "_test_build_deformation_field"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BuildDeformationFieldTest(unittest.TestCase):
    def test_load_control_points_reads_positions_and_displacements(self):
        builder = _load_field_builder()

        with tempfile.TemporaryDirectory() as temp_dir:
            controls_path = Path(temp_dir) / "controls.json"
            controls_path.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0.0",
                        "coordinate_order": "xyz",
                        "control_points": [
                            {
                                "source_xyz": [1.0, 2.0, 3.0],
                                "displacement_xyz": [0.5, 0.0, -1.0],
                            },
                            {
                                "source_xyz": [4.0, 5.0, 6.0],
                                "target_xyz": [5.0, 7.0, 9.0],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            positions, displacements = builder.load_control_points(controls_path)

        np.testing.assert_allclose(
            positions,
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64),
        )
        np.testing.assert_allclose(
            displacements,
            np.array([[0.5, 0.0, -1.0], [1.0, 2.0, 3.0]], dtype=np.float64),
        )

    def test_interpolate_idw_returns_exact_control_displacement_at_control_point(self):
        builder = _load_field_builder()

        controls_xyz = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float64)
        displacements_xyz = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]], dtype=np.float64)
        query_xyz = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=np.float64)

        interpolated = builder.interpolate_idw(
            query_xyz,
            controls_xyz,
            displacements_xyz,
            k=2,
            power=2.0,
        )

        np.testing.assert_allclose(interpolated[0], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(interpolated[1], [5.5, 11.0, 16.5])

    def test_interpolate_idw_can_chunk_query_points(self):
        builder = _load_field_builder()

        controls_xyz = np.array(
            [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [8.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        displacements_xyz = np.array(
            [[1.0, 0.0, 0.0], [5.0, 0.0, 0.0], [9.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        query_xyz = np.array(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
            dtype=np.float64,
        )

        full = builder.interpolate_idw(
            query_xyz,
            controls_xyz,
            displacements_xyz,
            k=2,
        )
        chunked = builder.interpolate_idw(
            query_xyz,
            controls_xyz,
            displacements_xyz,
            k=2,
            query_chunk_size=1,
        )

        np.testing.assert_allclose(chunked, full)

    def test_interpolate_idw_can_chunk_control_points(self):
        builder = _load_field_builder()

        controls_xyz = np.array(
            [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [8.0, 0.0, 0.0], [12.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        displacements_xyz = np.array(
            [[1.0, 0.0, 0.0], [5.0, 0.0, 0.0], [9.0, 0.0, 0.0], [13.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        query_xyz = np.array(
            [[1.0, 0.0, 0.0], [6.0, 0.0, 0.0], [11.0, 0.0, 0.0]],
            dtype=np.float64,
        )

        full = builder.interpolate_idw(
            query_xyz,
            controls_xyz,
            displacements_xyz,
            k=3,
        )
        chunked_controls = builder.interpolate_idw(
            query_xyz,
            controls_xyz,
            displacements_xyz,
            k=3,
            control_chunk_size=1,
        )

        np.testing.assert_allclose(chunked_controls, full)

    def test_interpolate_idw_control_chunks_match_full_batch_for_distance_ties(self):
        builder = _load_field_builder()

        query_xyz = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        controls_xyz = np.array(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=np.float64,
        )
        displacements_xyz = np.arange(6, dtype=np.float64)[:, None].repeat(3, axis=1)

        full = builder.interpolate_idw(
            query_xyz,
            controls_xyz,
            displacements_xyz,
            k=3,
        )

        np.testing.assert_allclose(full, [[1.0, 1.0, 1.0]])
        for control_chunk_size in (1, 2, 3, 4, 5):
            chunked = builder.interpolate_idw(
                query_xyz,
                controls_xyz,
                displacements_xyz,
                k=3,
                control_chunk_size=control_chunk_size,
            )
            np.testing.assert_allclose(chunked, full)

    def test_interpolate_idw_can_chunk_queries_and_controls_together(self):
        builder = _load_field_builder()

        controls_xyz = np.array(
            [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [8.0, 0.0, 0.0], [12.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        displacements_xyz = controls_xyz + np.array([1.0, 2.0, 3.0], dtype=np.float64)
        query_xyz = np.array(
            [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [6.0, 0.0, 0.0], [11.0, 0.0, 0.0]],
            dtype=np.float64,
        )

        full = builder.interpolate_idw(
            query_xyz,
            controls_xyz,
            displacements_xyz,
            k=2,
        )
        chunked = builder.interpolate_idw(
            query_xyz,
            controls_xyz,
            displacements_xyz,
            k=2,
            query_chunk_size=2,
            control_chunk_size=2,
        )

        np.testing.assert_allclose(chunked, full)

    def test_interpolate_idw_rejects_non_positive_k(self):
        builder = _load_field_builder()

        controls_xyz = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        displacements_xyz = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
        query_xyz = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)

        with self.assertRaisesRegex(ValueError, "k must be positive"):
            builder.interpolate_idw(
                query_xyz,
                controls_xyz,
                displacements_xyz,
                k=0,
            )

    def test_interpolate_idw_rejects_non_positive_control_chunk_size(self):
        builder = _load_field_builder()

        controls_xyz = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        displacements_xyz = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
        query_xyz = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)

        with self.assertRaisesRegex(ValueError, "control_chunk_size must be positive"):
            builder.interpolate_idw(
                query_xyz,
                controls_xyz,
                displacements_xyz,
                control_chunk_size=0,
            )

    def test_interpolate_idw_rejects_empty_controls(self):
        builder = _load_field_builder()

        controls_xyz = np.empty((0, 3), dtype=np.float64)
        displacements_xyz = np.empty((0, 3), dtype=np.float64)
        query_xyz = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)

        with self.assertRaisesRegex(ValueError, "at least one control point"):
            builder.interpolate_idw(
                query_xyz,
                controls_xyz,
                displacements_xyz,
            )

    def test_interpolate_idw_returns_empty_output_for_empty_query(self):
        builder = _load_field_builder()

        controls_xyz = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        displacements_xyz = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
        query_xyz = np.empty((0, 3), dtype=np.float64)

        interpolated = builder.interpolate_idw(
            query_xyz,
            controls_xyz,
            displacements_xyz,
        )

        self.assertEqual(interpolated.shape, (0, 3))

    def test_build_grid_field_uses_xyz_shape_and_spacing(self):
        builder = _load_field_builder()

        controls_xyz = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
        displacements_xyz = np.array([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float64)

        field, origin, spacing = builder.build_grid_field(
            controls_xyz,
            displacements_xyz,
            grid_shape_xyz=(3, 1, 1),
            spacing_xyz=(1.0, 1.0, 1.0),
            origin_xyz=(0.0, 0.0, 0.0),
            k=2,
        )

        self.assertEqual(field.shape, (1, 1, 3, 3))
        np.testing.assert_allclose(origin, [0.0, 0.0, 0.0])
        np.testing.assert_allclose(spacing, [1.0, 1.0, 1.0])
        np.testing.assert_allclose(field[0, 0, 0], [1.0, 0.0, 0.0])
        np.testing.assert_allclose(field[0, 0, 2], [3.0, 0.0, 0.0])

    def test_build_grid_field_accepts_query_chunk_size(self):
        builder = _load_field_builder()

        controls_xyz = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
        displacements_xyz = np.array([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float64)

        field, _, _ = builder.build_grid_field(
            controls_xyz,
            displacements_xyz,
            grid_shape_xyz=(3, 1, 1),
            spacing_xyz=(1.0, 1.0, 1.0),
            origin_xyz=(0.0, 0.0, 0.0),
            k=2,
            query_chunk_size=1,
        )

        self.assertEqual(field.shape, (1, 1, 3, 3))
        np.testing.assert_allclose(field[0, 0, 0], [1.0, 0.0, 0.0])
        np.testing.assert_allclose(field[0, 0, 2], [3.0, 0.0, 0.0])

    def test_build_grid_field_accepts_control_chunk_size(self):
        builder = _load_field_builder()

        controls_xyz = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
        displacements_xyz = np.array([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float64)

        field, _, _ = builder.build_grid_field(
            controls_xyz,
            displacements_xyz,
            grid_shape_xyz=(3, 1, 1),
            spacing_xyz=(1.0, 1.0, 1.0),
            origin_xyz=(0.0, 0.0, 0.0),
            k=2,
            control_chunk_size=1,
        )

        self.assertEqual(field.shape, (1, 1, 3, 3))
        np.testing.assert_allclose(field[0, 0, 0], [1.0, 0.0, 0.0])
        np.testing.assert_allclose(field[0, 0, 2], [3.0, 0.0, 0.0])

    def test_build_grid_field_rejects_non_positive_shape_or_spacing(self):
        builder = _load_field_builder()

        controls_xyz = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        displacements_xyz = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)

        with self.assertRaisesRegex(ValueError, "grid shape values must be positive"):
            builder.build_grid_field(
                controls_xyz,
                displacements_xyz,
                grid_shape_xyz=(0, 1, 1),
                spacing_xyz=(1.0, 1.0, 1.0),
                origin_xyz=(0.0, 0.0, 0.0),
            )
        with self.assertRaisesRegex(ValueError, "spacing values must be positive"):
            builder.build_grid_field(
                controls_xyz,
                displacements_xyz,
                grid_shape_xyz=(1, 1, 1),
                spacing_xyz=(1.0, 0.0, 1.0),
                origin_xyz=(0.0, 0.0, 0.0),
            )

    def test_parser_rejects_non_positive_k(self):
        builder = _load_field_builder()
        parser = builder.build_arg_parser()

        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "--controls",
                    "controls.json",
                    "--output",
                    "field.npz",
                    "--grid-shape",
                    "1,1,1",
                    "--spacing",
                    "1,1,1",
                    "--k",
                    "0",
                ]
            )

    def test_parser_accepts_query_chunk_size(self):
        builder = _load_field_builder()
        parser = builder.build_arg_parser()

        args = parser.parse_args(
            [
                "--controls",
                "controls.json",
                "--output",
                "field.npz",
                "--grid-shape",
                "1,1,1",
                "--spacing",
                "1,1,1",
                "--query-chunk-size",
                "2",
            ]
        )

        self.assertEqual(args.query_chunk_size, 2)

    def test_parser_accepts_control_chunk_size(self):
        builder = _load_field_builder()
        parser = builder.build_arg_parser()

        args = parser.parse_args(
            [
                "--controls",
                "controls.json",
                "--output",
                "field.npz",
                "--grid-shape",
                "1,1,1",
                "--spacing",
                "1,1,1",
                "--control-chunk-size",
                "2",
            ]
        )

        self.assertEqual(args.control_chunk_size, 2)

    def test_parser_rejects_non_positive_query_chunk_size(self):
        builder = _load_field_builder()
        parser = builder.build_arg_parser()

        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "--controls",
                    "controls.json",
                    "--output",
                    "field.npz",
                    "--grid-shape",
                    "1,1,1",
                    "--spacing",
                    "1,1,1",
                    "--query-chunk-size",
                    "0",
                ]
            )

    def test_parser_rejects_non_positive_control_chunk_size(self):
        builder = _load_field_builder()
        parser = builder.build_arg_parser()

        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "--controls",
                    "controls.json",
                    "--output",
                    "field.npz",
                    "--grid-shape",
                    "1,1,1",
                    "--spacing",
                    "1,1,1",
                    "--control-chunk-size",
                    "0",
                ]
            )

    def test_cli_writes_npz_with_displacement_field(self):
        builder = _load_field_builder()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            controls_path = temp / "controls.json"
            output_path = temp / "field.npz"
            controls_path.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0.0",
                        "coordinate_order": "xyz",
                        "control_points": [
                            {
                                "source_xyz": [0.0, 0.0, 0.0],
                                "displacement_xyz": [1.0, 2.0, 3.0],
                            },
                            {
                                "source_xyz": [2.0, 0.0, 0.0],
                                "displacement_xyz": [5.0, 6.0, 7.0],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            builder.main(
                [
                    "--controls",
                    str(controls_path),
                    "--output",
                    str(output_path),
                    "--grid-shape",
                    "3,1,1",
                    "--spacing",
                    "1,1,1",
                    "--origin",
                    "0,0,0",
                    "--k",
                    "2",
                ]
            )

            with np.load(output_path) as data:
                field = data["displacement_xyz"]
                origin = data["origin_xyz"]
                spacing = data["spacing_xyz"]
                coordinate_order = data["coordinate_order"].item()

        self.assertEqual(field.shape, (1, 1, 3, 3))
        np.testing.assert_allclose(origin, [0.0, 0.0, 0.0])
        np.testing.assert_allclose(spacing, [1.0, 1.0, 1.0])
        self.assertEqual(coordinate_order, "zyx_grid_xyz_vectors")


if __name__ == "__main__":
    unittest.main()
