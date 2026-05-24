import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


MODULE_DIR = Path(__file__).resolve().parents[1]
MODULE_PATH = MODULE_DIR / "export_deformation_control_points.py"


def _load_exporter():
    module_name = "_test_export_deformation_control_points"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ExportDeformationControlPointsTest(unittest.TestCase):
    def test_read_obj_vertices_ignores_non_vertex_records(self):
        exporter = _load_exporter()

        with tempfile.TemporaryDirectory() as temp_dir:
            obj_path = Path(temp_dir) / "mesh.obj"
            obj_path.write_text(
                "\n".join(
                    [
                        "# comment",
                        "v 1 2 3",
                        "vt 0.1 0.2",
                        "vn 0 0 1",
                        "f 1 2 3",
                        "v -1.5 0 4.25",
                    ]
                ),
                encoding="utf-8",
            )

            vertices = exporter.read_obj_vertices(obj_path)

        np.testing.assert_allclose(
            vertices,
            np.array([[1.0, 2.0, 3.0], [-1.5, 0.0, 4.25]], dtype=np.float64),
        )

    def test_build_control_points_computes_displacements_and_uniform_sample(self):
        exporter = _load_exporter()

        source = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        registered = source + np.array([0.5, -1.0, 2.0], dtype=np.float64)

        controls = exporter.build_control_points(source, registered, max_points=3)

        self.assertEqual(len(controls), 3)
        self.assertEqual([control["vertex_index"] for control in controls], [0, 2, 4])
        self.assertEqual(controls[1]["source_xyz"], [2.0, 0.0, 0.0])
        self.assertEqual(controls[1]["target_xyz"], [2.5, -1.0, 2.0])
        self.assertEqual(controls[1]["displacement_xyz"], [0.5, -1.0, 2.0])

    def test_build_control_points_rejects_mismatched_vertex_counts(self):
        exporter = _load_exporter()

        with self.assertRaisesRegex(ValueError, "same number of vertices"):
            exporter.build_control_points(
                np.zeros((2, 3), dtype=np.float64),
                np.zeros((3, 3), dtype=np.float64),
            )

    def test_parser_rejects_negative_max_points_per_mesh(self):
        exporter = _load_exporter()
        parser = exporter.build_arg_parser()

        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "--mesh-pair",
                    "source.obj",
                    "registered.obj",
                    "--output",
                    "controls.json",
                    "--max-points-per-mesh",
                    "-1",
                ]
            )

    def test_cli_writes_control_point_json_for_mesh_pair(self):
        exporter = _load_exporter()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            source_path = temp / "1_hz.obj"
            registered_path = temp / "1_hz_registered.obj"
            output_path = temp / "controls.json"
            source_path.write_text(
                "\n".join(["v 0 0 0", "v 10 0 0", "v 20 0 0"]),
                encoding="utf-8",
            )
            registered_path.write_text(
                "\n".join(["v 1 2 3", "v 11 2 3", "v 21 2 3"]),
                encoding="utf-8",
            )

            exporter.main(
                [
                    "--mesh-pair",
                    str(source_path),
                    str(registered_path),
                    "--output",
                    str(output_path),
                    "--max-points-per-mesh",
                    "2",
                ]
            )

            data = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(data["schema_version"], "1.0.0")
        self.assertEqual(data["coordinate_order"], "xyz")
        self.assertEqual(len(data["mesh_pairs"]), 1)
        self.assertEqual(data["mesh_pairs"][0]["source_mesh"], str(source_path))
        self.assertEqual(data["mesh_pairs"][0]["registered_mesh"], str(registered_path))
        self.assertEqual(len(data["control_points"]), 2)
        self.assertEqual(data["control_points"][0]["mesh_pair_index"], 0)
        self.assertEqual(data["control_points"][0]["source_xyz"], [0.0, 0.0, 0.0])
        self.assertEqual(data["control_points"][0]["target_xyz"], [1.0, 2.0, 3.0])
        self.assertEqual(data["control_points"][0]["displacement_xyz"], [1.0, 2.0, 3.0])


if __name__ == "__main__":
    unittest.main()
