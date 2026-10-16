"""Run a benchmark or tests against an explicit native build, without installing it.

    python benchmarks/run_native.py build/<tag>/_native*.so benchmarks/bench_sample.py --reps 7
    python benchmarks/run_native.py build/<tag>/_native*.so -m pytest -q
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import runpy
import sys


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    extension = Path(sys.argv[1]).resolve(strict=True)
    command = sys.argv[2:]
    spec = importlib.util.spec_from_file_location("vc_reader._native", extension)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load native extension: {extension}")
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    sys.modules[spec.name] = native
    print(f"native: {extension}", file=sys.stderr)
    if command[0] == "-m":
        if len(command) < 2:
            raise SystemExit("-m needs a module name")
        sys.argv = command[1:]
        runpy.run_module(command[1], run_name="__main__", alter_sys=True)
    else:
        sys.argv = command
        sys.path.insert(0, str(Path(command[0]).resolve().parent))
        runpy.run_path(command[0], run_name="__main__")


if __name__ == "__main__":
    main()
