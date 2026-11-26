"""Fresh CMake configuration checks; run with Python containing nanobind.

Run: CUDACXX=/path/to/nvcc python -m unittest discover -s tests -p test_cmake_configure.py
CUDA checks are skipped when no compiler is available. Checks that need CUDA to be on
are skipped when the compiler cannot build the CUDA backend; checks of that case are
skipped when it can.
"""
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

# What the CUDA backend needs from the toolkit's CUB (see cmake/cuda_backend_probe.cu).
BACKEND_PROBE = """
#include <cub/device/device_radix_sort.cuh>
#include <cuda/std/functional>
#include <cuda/std/tuple>
struct Key { unsigned long long value; unsigned sample; };
struct Decomposer { __host__ __device__ auto operator()(Key& k) const { return ::cuda::std::tie(k.sample, k.value); } };
cudaError_t probe(void* s, std::size_t& b, Key* i, Key* o, std::size_t n, cudaStream_t t) {
    return cub::DeviceRadixSort::SortKeys(s, b, i, o, n, Decomposer{}, 0, 96, t);
}
"""


def compiler_builds_backend(compiler):
    with tempfile.TemporaryDirectory(prefix="vc-reader-probe-") as work:
        source = Path(work) / "probe.cu"
        source.write_text(BACKEND_PROBE)
        result = subprocess.run(
            [compiler, "-std=c++20", "-c", str(source), "-o", str(Path(work) / "probe.o")],
            capture_output=True, text=True, timeout=300,
        )
        return result.returncode == 0


@unittest.skipUnless(
    shutil.which("cmake") and importlib.util.find_spec("nanobind"),
    "requires CMake and nanobind",
)
class CMakeConfigureTests(unittest.TestCase):
    def run_cmake(self, build_dir, *settings):
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
        return subprocess.run(
            command, env=env, capture_output=True, text=True, timeout=300,
        )

    def configure(self, expected_cuda, *settings, success=True):
        with tempfile.TemporaryDirectory(prefix="vc-reader-cmake-") as build_dir:
            result = self.run_cmake(build_dir, *settings)
            if not success:
                self.assertNotEqual(result.returncode, 0)
                return result
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(self.cached_cuda(build_dir), "ON" if expected_cuda else "OFF")
            return result

    def cached_cuda(self, build_dir):
        for line in (Path(build_dir) / "CMakeCache.txt").read_text().splitlines():
            if line.startswith("VC_READER_ENABLE_CUDA:BOOL="):
                return line.split("=", 1)[1]
        return None

    def test_cpu(self):
        self.configure(False, "-DVC_READER_ENABLE_CUDA=OFF",
                       "-DCMAKE_CUDA_COMPILER=/nonexistent/nvcc")

    def test_auto_without_compiler(self):
        self.configure(False, "-DCMAKE_CUDA_COMPILER=NOTFOUND")

    def test_required_without_compiler(self):
        self.configure(True, "-DVC_READER_ENABLE_CUDA=ON",
                       "-DCMAKE_CUDA_COMPILER=NOTFOUND", success=False)

    def any_cuda_compiler(self):
        compiler = os.environ.get("CUDACXX") or shutil.which("nvcc")
        if not compiler:
            self.skipTest("set CUDACXX or put nvcc on PATH")
        return compiler

    def cuda_compiler(self):
        compiler = self.any_cuda_compiler()
        if not compiler_builds_backend(compiler):
            self.skipTest(f"{compiler} cannot build the CUDA backend")
        return compiler

    def old_cuda_compiler(self):
        compiler = self.any_cuda_compiler()
        if compiler_builds_backend(compiler):
            self.skipTest(f"{compiler} can build the CUDA backend")
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

    def test_auto_with_old_toolkit_builds_without_cuda(self):
        compiler = self.old_cuda_compiler()
        result = self.configure(False, f"-DCMAKE_CUDA_COMPILER={compiler}")
        self.assertIn("Building vc_reader without CUDA", result.stderr)
        self.assertIn("-DVC_READER_ENABLE_CUDA=ON", result.stderr)
        self.assertIn(compiler, result.stderr)

    def test_required_with_old_toolkit_fails(self):
        compiler = self.old_cuda_compiler()
        result = self.configure(True, "-DVC_READER_ENABLE_CUDA=ON",
                                f"-DCMAKE_CUDA_COMPILER={compiler}", success=False)
        self.assertIn("VC_READER_ENABLE_CUDA=ON was requested", result.stderr)
        self.assertIn(compiler, result.stderr)

    def test_old_toolkit_decision_is_kept_by_a_second_configure(self):
        compiler = self.old_cuda_compiler()
        with tempfile.TemporaryDirectory(prefix="vc-reader-cmake-") as build_dir:
            result = self.run_cmake(build_dir, f"-DCMAKE_CUDA_COMPILER={compiler}")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(self.cached_cuda(build_dir), "OFF")
            self.assertIn("Building vc_reader without CUDA", result.stderr)
            # The second configure keeps the decision and does not fail.
            result = self.run_cmake(build_dir)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(self.cached_cuda(build_dir), "OFF")
            # An explicit request after the automatic decision is still refused.
            result = self.run_cmake(build_dir, "-DVC_READER_ENABLE_CUDA=ON")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("VC_READER_ENABLE_CUDA=ON was requested", result.stderr)

    def test_explicit_cpu_with_old_toolkit_is_silent(self):
        compiler = self.old_cuda_compiler()
        result = self.configure(False, "-DVC_READER_ENABLE_CUDA=OFF",
                                f"-DCMAKE_CUDA_COMPILER={compiler}")
        self.assertNotIn("without CUDA", result.stderr)

    def test_usable_toolkit_stays_on_in_a_second_configure(self):
        compiler = self.cuda_compiler()
        with tempfile.TemporaryDirectory(prefix="vc-reader-cmake-") as build_dir:
            for _ in range(2):
                result = self.run_cmake(build_dir, f"-DCMAKE_CUDA_COMPILER={compiler}")
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(self.cached_cuda(build_dir), "ON")
                self.assertNotIn("without CUDA", result.stderr)


if __name__ == "__main__":
    unittest.main()
