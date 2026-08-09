#!/usr/bin/env python3
"""Build the dependency-free optional native TIFXYZ rasterizer."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

from .native import (
    default_native_library_path,
    native_source_fingerprint,
    reset_native_library_cache,
)


def build_native(
    output: Path | str | None = None,
    compiler: str | None = None,
) -> Path:
    source = Path(__file__).with_name("native_rasterizer.cpp")
    source_fingerprint = native_source_fingerprint()
    output_path = (
        default_native_library_path() if output is None else Path(output)
    ).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    compiler_command = shlex.split(compiler or os.environ.get("CXX", "c++"))
    if not compiler_command:
        raise ValueError("CXX must name a C++ compiler")
    flags = [
        "-O3",
        "-std=c++17",
        "-fPIC",
        "-fno-fast-math",
        "-ffp-contract=off",
        "-fno-associative-math",
        "-fvisibility=hidden",
        "-Wall",
        "-Wextra",
        "-Wpedantic",
        f'-DVC_SOURCE_FINGERPRINT="{source_fingerprint}"',
    ]
    link_mode = "-dynamiclib" if sys.platform == "darwin" else "-shared"
    with tempfile.NamedTemporaryFile(
        prefix=f".{output_path.name}.build-",
        suffix=output_path.suffix,
        dir=output_path.parent,
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
    command = [
        *compiler_command,
        *flags,
        link_mode,
        str(source),
        "-o",
        str(temporary_path),
    ]
    try:
        subprocess.run(command, check=True)
        os.chmod(temporary_path, 0o755)
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    reset_native_library_cache()
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--compiler", default=None)
    args = parser.parse_args()
    output = build_native(args.output, args.compiler)
    print(output)


if __name__ == "__main__":
    main()
