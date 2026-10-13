import csv
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).parents[1] / "scripts" / "stl_generator.py"


def load_stl_generator():
    meshlib = types.ModuleType("meshlib")
    meshlib.mrmeshpy = types.ModuleType("meshlib.mrmeshpy")

    scrollcase = types.ModuleType("scrollcase")
    scrollcase.mesh = types.ModuleType("scrollcase.mesh")
    scrollcase.case = types.ModuleType("scrollcase.case")

    tqdm = types.ModuleType("tqdm")
    tqdm.tqdm = lambda iterable, **_kwargs: iterable

    modules = {
        "meshlib": meshlib,
        "meshlib.mrmeshpy": meshlib.mrmeshpy,
        "scrollcase": scrollcase,
        "scrollcase.mesh": scrollcase.mesh,
        "scrollcase.case": scrollcase.case,
        "tqdm": tqdm,
    }
    with mock.patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location("stl_generator", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class StlGeneratorTests(unittest.TestCase):
    def test_missing_meshes_write_header_only_summary(self):
        module = load_stl_generator()

        class ValidatingExecutor:
            def __init__(self, max_workers, **_kwargs):
                if max_workers <= 0:
                    raise ValueError("max_workers must be greater than 0")

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

        with tempfile.TemporaryDirectory() as temp_dir:
            input_root = Path(temp_dir) / "input"
            output_root = Path(temp_dir) / "output"
            (input_root / "4").mkdir(parents=True)
            output_root.mkdir()

            argv = [
                "stl_generator.py",
                "--input",
                str(input_root),
                "--output",
                str(output_root),
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(module, "ProcessPoolExecutor", ValidatingExecutor),
            ):
                module.main()

            with (output_root / "scroll_summary.csv").open(newline="") as csvfile:
                self.assertEqual(
                    list(csv.reader(csvfile)),
                    [["Scroll ID", "Height (mm)", "Diameter (mm)"]],
                )

    def test_read_axis_direction_and_two_points(self):
        module = load_stl_generator()
        with tempfile.TemporaryDirectory() as temp_dir:
            axis_file = Path(temp_dir) / "4-axis.txt"
            axis_file.write_text("# umbilicus direction\n0, 0.5, 1\n")
            self.assertEqual(module.read_axis(str(axis_file)), (0.0, 0.5, 1.0))
            axis_file.write_text("1 2 3\n1 2 13  # end point\n")
            self.assertEqual(module.read_axis(str(axis_file)), (0.0, 0.0, 10.0))
            for bad in ("1 2", "0 0 0", "1 2 3 1 2 3"):
                axis_file.write_text(bad)
                with self.assertRaises(ValueError):
                    module.read_axis(str(axis_file))


if __name__ == "__main__":
    unittest.main()
