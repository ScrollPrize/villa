import importlib.util
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class Vc3dFiberFormatAdapterTests(unittest.TestCase):
    def test_imports_service_local_package(self):
        source_root = Path(__file__).resolve().parents[3]
        adapter = source_root / "scripts" / "spiral" / "vc3d_fiber_format_adapter.py"
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


if __name__ == "__main__":
    unittest.main()
