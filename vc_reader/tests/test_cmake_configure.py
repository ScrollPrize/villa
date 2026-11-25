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
                return result
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

    def test_hip_and_cuda_are_exclusive(self):
        result = self.configure(False, "-DVC_READER_ENABLE_HIP=ON",
                                "-DVC_READER_ENABLE_CUDA=ON", success=False)
        self.assertIn("mutually exclusive", result.stdout + result.stderr)

    def test_hip_rejects_rocm_6(self):
        with tempfile.TemporaryDirectory(prefix="vc-reader-hip-package-") as directory:
            package = Path(directory)
            (package / "hip-config.cmake").touch()
            (package / "hip-config-version.cmake").write_text('set(PACKAGE_VERSION "6.4.3")\n')
            result = self.configure(False, "-DVC_READER_ENABLE_HIP=ON",
                                    "-DCMAKE_HIP_ARCHITECTURES=gfx1100",
                                    f"-Dhip_DIR={package}", success=False)
        self.assertIn("requires ROCm 7.0 or newer (found HIP 6.4.3)", result.stdout + result.stderr)

    def cuda_compiler(self):
        compiler = os.environ.get("CUDACXX") or shutil.which("nvcc")
        if not compiler:
            self.skipTest("set CUDACXX or put nvcc on PATH")
        return compiler

    def hip_toolkit(self):
        root = Path(os.environ.get("ROCM_PATH", "/opt/rocm"))
        if not (root / "llvm/bin/clang++").exists():
            self.skipTest("set ROCM_PATH or install ROCm in /opt/rocm")
        return root

    def test_hip_accepts_wave32(self):
        self.hip_toolkit()
        self.configure(False, "-DVC_READER_ENABLE_HIP=ON",
                       "-DCMAKE_HIP_ARCHITECTURES=gfx1200")

    def test_hip_rejects_wave64(self):
        self.hip_toolkit()
        result = self.configure(False, "-DVC_READER_ENABLE_HIP=ON",
                                "-DCMAKE_HIP_ARCHITECTURES=gfx906", success=False)
        self.assertIn("wave32", result.stdout + result.stderr)

    def test_auto_with_compiler(self):
        compiler = self.cuda_compiler()
        # Automatic and required CUDA both use the compiler's default architectures.
        for settings in ((), ("-DVC_READER_ENABLE_CUDA=ON",)):
            with self.subTest(settings=settings):
                self.configure(True, *settings, f"-DCMAKE_CUDA_COMPILER={compiler}")

    def test_auto_with_toolkit(self):
        root = Path(self.cuda_compiler()).resolve().parent.parent
        self.configure(True, f"-DCUDAToolkit_ROOT={root}")


if __name__ == "__main__":
    unittest.main()
