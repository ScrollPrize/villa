#!/usr/bin/env python3
"""Check that every build-tree file an install rule copies is built by `all`.

Packaging (cpack / `cmake --install`) runs after the default build. An install
rule naming a target the default build skips (EXCLUDE_FROM_ALL, or a target
only some option combination builds) fails there with "file INSTALL cannot
find", and the packaging jobs only run after a merge. This check needs only a
configured Ninja build directory, so a PR job can run it before compiling.

For every quoted path under the build directory in the generated
cmake_install.cmake scripts, the file must either exist already (generated at
configure time) or be an output that `ninja all` builds.

Usage: check_install_rules.py <build-dir>
"""
import pathlib
import re
import subprocess
import sys


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    build = pathlib.Path(sys.argv[1]).resolve()
    if not (build / "build.ninja").is_file():
        print(f"{build}: not a configured Ninja build directory", file=sys.stderr)
        return 2

    # `ninja -t inputs` (ninja >= 1.11) lists every input of `all`,
    # recursively: the outputs the default build produces among them.
    inputs = subprocess.run(
        ["ninja", "-C", str(build), "-t", "inputs", "all"],
        check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    built = {(build / p).resolve() for p in inputs}

    prefix = str(build) + "/"
    quoted = re.compile(r'"([^"]*)"')
    missing: dict[pathlib.Path, set[str]] = {}
    for script in sorted(build.rglob("cmake_install.cmake")):
        if "_CPack_Packages" in script.parts:
            continue
        text = script.read_text(errors="replace")
        for line in text.splitlines():
            if "file(INSTALL" not in line:
                continue
            for raw in quoted.findall(line):
                if "$" in raw or not raw.startswith(prefix):
                    continue
                path = pathlib.Path(raw).resolve()
                if path.exists() or path in built:
                    continue
                missing.setdefault(path, set()).add(
                    str(script.relative_to(build)))

    if missing:
        print("Install rules copy files the default build does not produce;"
              " `cmake --install` / cpack would fail on them:", file=sys.stderr)
        for path, scripts in sorted(missing.items()):
            print(f"  {path.relative_to(build)}  (from {', '.join(sorted(scripts))})",
                  file=sys.stderr)
        return 1
    print(f"install rules OK: {build}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
