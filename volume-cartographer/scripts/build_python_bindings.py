#!/usr/bin/env python3
"""
Helper to (re)build the vc.tracing.vc_tracing extension against the interpreter
running this script. The top-level CMake build now enables the bindings by
default; invoke this when you need a dedicated build directory per Python.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import sysconfig
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUILD_DIR = REPO_ROOT / f"build_py{sys.version_info.major}{sys.version_info.minor}"
DEFAULT_CMAKE = os.environ.get("CMAKE", "cmake")
PYTHON_TARGET = "vc_tracing"


def _run(command: list[str], *, cwd: Path) -> None:
    """Run a subprocess and echo the command."""
    print("+", " ".join(command))
    subprocess.run(command, check=True, cwd=cwd)


def _configure(build_dir: Path, *, generator: str | None, cmake_args: list[str]) -> None:
    command = [
        DEFAULT_CMAKE,
        "-S",
        str(REPO_ROOT),
        "-B",
        str(build_dir),
        "-DVC_BUILD_PYTHON=ON",
        f"-DPython3_EXECUTABLE={sys.executable}",
    ]
    if generator:
        command.extend(["-G", generator])
    command.extend(cmake_args)
    _run(command, cwd=REPO_ROOT)


def _build(build_dir: Path, *, config: str | None, target: str) -> None:
    command = [DEFAULT_CMAKE, "--build", str(build_dir), "--target", target]
    if config:
        command.extend(["--config", config])
    _run(command, cwd=REPO_ROOT)


def _discover_module(base_dir: Path) -> list[Path]:
    native_root = base_dir / "python" / "vc" / "tracing"
    if not native_root.is_dir():
        return []
    return sorted(native_root.glob("vc_tracing*.so"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild the vc.tracing.vc_tracing extension for the active Python interpreter.",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=DEFAULT_BUILD_DIR,
        help=f"Build directory to use (default: {DEFAULT_BUILD_DIR.name})",
    )
    parser.add_argument(
        "--generator",
        "-G",
        help="Optional CMake generator to use (defaults to CMake's native choice).",
    )
    parser.add_argument(
        "--build-type",
        help="CMake build type (Release/Debug/RelWithDebInfo/MinSizeRel). "
        "Added as -DCMAKE_BUILD_TYPE for single-config generators.",
    )
    parser.add_argument(
        "--config",
        help="Configuration to pass to `cmake --build` for multi-config generators "
        "(e.g. Release/Debug for Visual Studio or Ninja Multi-Config).",
    )
    parser.add_argument(
        "--target",
        default=PYTHON_TARGET,
        help=f"CMake target to build (default: {PYTHON_TARGET}).",
    )
    parser.add_argument(
        "--skip-configure",
        action="store_true",
        help="Skip the CMake configure step (only run the build).",
    )
    parser.add_argument(
        "--cmake-arg",
        dest="cmake_args",
        action="append",
        default=[],
        help="Additional -D or other arguments to forward to CMake (may be repeated).",
    )

    args = parser.parse_args(argv)
    build_dir = args.build_dir.resolve()
    build_dir.mkdir(parents=True, exist_ok=True)

    cmake_args = list(args.cmake_args)
    if args.build_type:
        cmake_args.append(f"-DCMAKE_BUILD_TYPE={args.build_type}")

    if not args.skip_configure:
        _configure(build_dir, generator=args.generator, cmake_args=cmake_args)

    _build(build_dir, config=args.config, target=args.target)

    built_modules = _discover_module(build_dir)
    if not built_modules:
        print(
            "warning: build completed but no vc.tracing.vc_tracing extension was found;"
            " check the build logs for errors.",
            file=sys.stderr,
        )
        return 1

    cache_tag = sys.implementation.cache_tag or "unknown"
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    print("Built vc.tracing.vc_tracing for:")
    print(f"  Python executable : {sys.executable}")
    print(f"  sys.implementation: {cache_tag}")
    if ext_suffix:
        print(f"  EXT_SUFFIX        : {ext_suffix}")
    print("  Output artifacts  :")
    for module_path in built_modules:
        print(f"    {module_path}")
    python_path_hint = build_dir / "python"
    print(f"Add `{python_path_hint}` to PYTHONPATH when importing vc.tracing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
