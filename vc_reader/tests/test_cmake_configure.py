"""Fresh CMake configuration checks; run with Python containing nanobind.

Run: CUDACXX=/path/to/nvcc python -m unittest discover -s tests -p test_cmake_configure.py
CUDA checks are skipped when no compiler is available.
"""
import importlib.util
import os
from pathlib import Path
import re
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

    @unittest.skipIf(sys.platform == "win32", "Windows has no RPATH")
    def test_reader_module_rpath_follows_install_libdir(self):
        # GNUInstallDirs picks lib64 on Fedora, RHEL and openSUSE; the module must
        # look for the reader's shared library where it is installed.
        delta3d = Path(os.environ.get("VC_READER_BUILD_DIR", "")) / "_deps/vc_delta3d-src"
        if not (delta3d / "CMakeLists.txt").is_file():
            self.skipTest("set VC_READER_BUILD_DIR to a reader build (for its vc-delta3d sources)")
        for libdir in ("lib", "lib64"):
            with self.subTest(libdir=libdir), tempfile.TemporaryDirectory(prefix="vc-reader-cmake-") as build_dir:
                result = subprocess.run(
                    ["cmake", "-S", str(Path(__file__).resolve().parents[1]), "-B", build_dir,
                     f"-DPython_EXECUTABLE={sys.executable}", "-DVC_READER_ENABLE_CUDA=OFF",
                     f"-DCMAKE_INSTALL_LIBDIR={libdir}",
                     f"-DFETCHCONTENT_SOURCE_DIR_VC_DELTA3D={delta3d.resolve()}"],
                    capture_output=True, text=True, timeout=300,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                install = (Path(build_dir) / "cmake_install.cmake").read_text()
                rpaths = set(re.findall(r'RPATH "([^"]+)"', install))
                self.assertTrue(any(rpath.endswith(f"/../{libdir}") for rpath in rpaths), rpaths)

    def cuda_compiler(self):
        compiler = os.environ.get("CUDACXX") or shutil.which("nvcc")
        if not compiler:
            self.skipTest("set CUDACXX or put nvcc on PATH")
        return compiler

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
