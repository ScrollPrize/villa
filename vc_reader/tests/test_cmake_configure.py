"""Fresh CMake configuration checks; run with Python containing nanobind.

Run: CUDACXX=/path/to/nvcc python -m unittest discover -s tests -p test_cmake_configure.py
CUDA checks are skipped when no compiler is available.
"""
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


@unittest.skipUnless(
    shutil.which("cmake") and importlib.util.find_spec("nanobind"),
    "requires CMake and nanobind",
)
class CMakeConfigureTests(unittest.TestCase):
    def configure(self, expected_cuda, *settings, success=True):
        with tempfile.TemporaryDirectory(prefix="vc-reader-cmake-") as build_dir:
            command = [
                "cmake",
                "-S", str(Path(__file__).resolve().parents[1]),
                "-B", build_dir,
                f"-DPython_EXECUTABLE={sys.executable}",
                "-DVC_READER_BUILD_READER=OFF",
                *settings,
            ]
            env = os.environ.copy()
            # Exercise compiler-derived defaults, without an architecture override.
            env.pop("CUDAARCHS", None)
            env.pop("CUDACXX", None)
            result = subprocess.run(
                command, env=env, capture_output=True, text=True, timeout=120,
            )
            if not success:
                self.assertNotEqual(result.returncode, 0)
                return
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            cache = (Path(build_dir) / "CMakeCache.txt").read_text()
            self.assertIn(
                f"VC_READER_ENABLE_CUDA:BOOL={'ON' if expected_cuda else 'OFF'}",
                cache,
            )

    def test_cpu(self):
        self.configure(False, "-DVC_READER_ENABLE_CUDA=OFF",
                       "-DCMAKE_CUDA_COMPILER=/nonexistent/nvcc")

    def test_auto_without_compiler(self):
        self.configure(False, "-DCMAKE_CUDA_COMPILER=NOTFOUND")

    def test_required_without_compiler(self):
        self.configure(True, "-DVC_READER_ENABLE_CUDA=ON",
                       "-DCMAKE_CUDA_COMPILER=NOTFOUND", success=False)

    def cuda_compiler(self):
        compiler = os.environ.get("CUDACXX") or shutil.which("nvcc")
        if not compiler:
            self.skipTest("set CUDACXX or put nvcc on PATH")
        return compiler

    def test_cuda_default_architectures(self):
        self.configure(True, "-DVC_READER_ENABLE_CUDA=ON",
                       f"-DCMAKE_CUDA_COMPILER={self.cuda_compiler()}")

    def test_auto_with_compiler(self):
        self.configure(True, f"-DCMAKE_CUDA_COMPILER={self.cuda_compiler()}")

    def test_auto_with_toolkit(self):
        root = Path(self.cuda_compiler()).resolve().parent.parent
        self.configure(True, f"-DCUDAToolkit_ROOT={root}")


if __name__ == "__main__":
    unittest.main()
