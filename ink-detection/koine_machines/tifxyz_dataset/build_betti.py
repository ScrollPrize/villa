#!/usr/bin/env python3
"""
Build Betti-Matching-3D for the local tifxyz_dataset training path.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def betti_dir() -> Path:
    return repo_root() / "external" / "Betti-Matching-3D"


def build_betti_matching() -> Path:
    """Clone and build Betti-Matching-3D under this repo's external/ directory."""
    root = repo_root()
    external_dir = root / "external"
    source_dir = betti_dir()
    build_dir = source_dir / "build"

    print(f"Repo root: {root}")
    print(f"Betti source dir: {source_dir}")

    try:
        external_dir.mkdir(parents=True, exist_ok=True)

        if not source_dir.exists():
            print("\nCloning Betti-Matching-3D...")
            subprocess.run(
                [
                    "git",
                    "clone",
                    "https://github.com/nstucki/Betti-Matching-3D.git",
                    str(source_dir),
                ],
                check=True,
            )
        else:
            print(f"\nBetti-Matching-3D already exists at {source_dir}")

        build_dir.mkdir(exist_ok=True)
        cmake_cmd = ["cmake"]

        try:
            import pybind11

            pybind11_dir = pybind11.get_cmake_dir()
            cmake_cmd.append(f"-Dpybind11_DIR={pybind11_dir}")
            print(f"Found pybind11 at: {pybind11_dir}")
        except ImportError:
            print("Warning: pybind11 not found in the current Python environment")
        except AttributeError:
            print("Warning: could not determine pybind11 CMake directory")

        if platform.system() == "Darwin" and platform.machine() == "arm64":
            cmake_cmd.append("-DCMAKE_OSX_ARCHITECTURES=arm64")

        cmake_cmd.append("..")

        print(f"\nRunning: {' '.join(cmake_cmd)}")
        subprocess.run(cmake_cmd, cwd=build_dir, check=True)

        print("\nRunning: make")
        subprocess.run(["make"], cwd=build_dir, check=True)

        candidates = []
        for pattern in ("betti_matching*.so", "betti_matching*.pyd", "betti_matching*.dll"):
            candidates.extend(build_dir.rglob(pattern))

        if not candidates:
            raise FileNotFoundError(
                f"Build completed but betti_matching extension was not found under {build_dir}"
            )

        module_path = candidates[0]
        print("\nBetti-Matching-3D built successfully")
        print(f"Extension: {module_path}")
        return module_path

    except subprocess.CalledProcessError as exc:
        print(f"\nFailed to build Betti-Matching-3D: {exc}")
        print("Manual build steps:")
        print(f"  cd {source_dir}")
        print("  mkdir -p build && cd build")
        print("  cmake ..")
        print("  make")
        raise SystemExit(1) from exc
    except Exception as exc:
        print(f"\nUnexpected error: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    try:
        build_betti_matching()
    except KeyboardInterrupt:
        sys.exit(130)
