import importlib.util
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class Vc3dFiberFormatAdapterTests(unittest.TestCase):
    def test_imports_service_local_package(self):
        source_root = Path(__file__).resolve().parents[1]
        adapter = source_root / "vc3d_fiber_format_adapter.py"
        package = source_root.parent / "vesuvius" / "src" / "vc3d_fiber_format"
        self.assertTrue(package.is_dir())
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            shutil.copy2(adapter, temporary / "vc3d_fiber_format_adapter.py")
            shutil.copytree(package, temporary / "vc3d_fiber_format")

            old_path = list(sys.path)
            sys.modules.pop("vc3d_fiber_format", None)
            try:
                sys.path = [path for path in sys.path if "vesuvius/src" not in path]
                spec = importlib.util.spec_from_file_location(
                    "temp_vc3d_fiber_format_adapter",
                    temporary / "vc3d_fiber_format_adapter.py")
                module = importlib.util.module_from_spec(spec)
                self.assertIsNotNone(spec.loader)
                spec.loader.exec_module(module)
            finally:
                sys.path = old_path
                sys.modules.pop("vc3d_fiber_format", None)

            self.assertTrue(callable(module.parse_vc3d_fiber_format))
            self.assertTrue(callable(module.validate_fiber_uuid))

    def test_present_fiber_uuid_must_be_canonical(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from vc3d_fiber_format_adapter import (parse_vc3d_fiber_format,
                                               validate_fiber_uuid)
        doc = {"type": "vc3d_fiber", "version": 1,
               "line_points": [[0, 0, 0], [1, 0, 0]],
               "control_points": [[0, 0, 0], [1, 0, 0]]}
        parse_vc3d_fiber_format(doc)  # legacy: uuid optional
        doc["uuid"] = "0f8fad5b-d9cb-469f-a165-70867728950e"
        parsed = parse_vc3d_fiber_format(doc)
        self.assertEqual(parsed.metadata["uuid"], doc["uuid"])
        for bad in ("fiber_name", "", None, 3, doc["uuid"].replace("-", "")):
            bad_doc = dict(doc, uuid=bad)
            with self.assertRaisesRegex(ValueError, "invalid uuid"):
                parse_vc3d_fiber_format(bad_doc)
            with self.assertRaisesRegex(ValueError, "invalid uuid"):
                validate_fiber_uuid(bad_doc)


if __name__ == "__main__":
    unittest.main()
