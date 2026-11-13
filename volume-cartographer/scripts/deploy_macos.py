#!/usr/bin/env python3
"""Deploy and sign a freshly staged VC3D.app (Python standard library only)."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
import subprocess
import time


@dataclass(frozen=True)
class LoadCommands:
    dependencies: tuple
    ids: tuple
    rpaths: tuple


def run(*args, stream=False):
    result = subprocess.run(args, capture_output=not stream, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"{args!r} failed ({result.returncode}):\n"
                           f"{result.stdout or ''}{result.stderr or ''}")
    return result.stdout or ""


@contextmanager
def timed(label):
    print(f"-- macOS packaging: {label}...", flush=True)
    start = time.monotonic()
    yield
    print(f"-- macOS packaging: {label}: {time.monotonic() - start:.2f}s", flush=True)


def parse_load_commands(output):
    """Read dependencies, dylib IDs and rpaths from a single otool -l call.

    Deduplicate commands repeated across slices of a universal binary.
    """
    dependencies, ids, rpaths = [], [], []
    command = None
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("Load command "):
            command = None
        elif line.startswith("cmd "):
            command = line[4:]
        else:
            match = re.fullmatch(r"(name|path) (.+) \(offset \d+\)", line)
            if not match:
                continue
            field, value = match.groups()
            if command == "LC_RPATH" and field == "path":
                rpaths.append(value)
            elif command == "LC_ID_DYLIB" and field == "name":
                ids.append(value)
            elif command in {
                "LC_LOAD_DYLIB", "LC_LOAD_WEAK_DYLIB", "LC_REEXPORT_DYLIB",
                "LC_LOAD_UPWARD_DYLIB", "LC_LAZY_LOAD_DYLIB",
            } and field == "name":
                dependencies.append(value)
    return LoadCommands(*(tuple(dict.fromkeys(values))
                          for values in (dependencies, ids, rpaths)))


def inspect(path):
    return parse_load_commands(run("otool", "-l", str(path)))


def macho_files(app):
    files = []
    # Include both byte orders and 32/64-bit fat headers. Do not follow framework
    # symlinks: rewriting/signing the same binary concurrently would race.
    magic = {bytes.fromhex(value) for value in (
        "feedface", "cefaedfe", "feedfacf", "cffaedfe",
        "cafebabe", "bebafeca", "cafebabf", "bfbafeca",
    )}
    for directory in ("Frameworks", "PlugIns", "MacOS"):
        for path in sorted((app / "Contents" / directory).rglob("*")):
            if path.is_symlink() or not path.is_file():
                continue
            with path.open("rb") as file:
                if file.read(4) in magic:
                    files.append(path)
    return files


def is_system(path):
    return path in ("/usr/lib", "/System") or path.startswith(("/usr/lib/", "/System/"))


def deployment_roots(app, commands):
    """Pass only CLIs that add dependencies to macdeployqt's existing roots.

    macdeployqt already scans VC3D and all staged .dylib/.so files. Its rewrite
    loop is dependencies x roots, even for roots without that dependency.
    Absolute names have the same meaning everywhere. For relative names, only
    share coverage when the loader directory AND ordered rpaths match; otherwise
    retain the CLI rather than assuming @rpath resolves to the same library.
    All CLIs are normalized and signed afterwards, including omitted roots.
    """
    main = app / "Contents/MacOS/VC3D"

    def keys(path):
        info = commands[path]
        return {
            (dep,) if dep.startswith("/") else (dep, path.parent, info.rpaths)
            for dep in info.dependencies if not is_system(dep)
        }

    covered = set()
    for path in commands:
        if path == main or path.suffix in (".dylib", ".so"):
            covered.update(keys(path))
    selected = []
    for path in sorted(commands):
        if path.parent != main.parent or path == main or path.suffix in (".dylib", ".so"):
            continue
        dependencies = keys(path)
        if not dependencies.issubset(covered):
            selected.append(path)
            covered.update(dependencies)
    return selected


def bundled_name(name):
    framework = re.search(r"([^/]+\.framework/.+)$", name)
    return framework.group(1) if framework else Path(name).name


def rewrite_args(app, path, info):
    args = []
    for dep in info.dependencies:
        if not dep.startswith("/") or is_system(dep):
            continue
        relative = bundled_name(dep)
        if not (app / "Contents/Frameworks" / relative).exists():
            raise RuntimeError(f"Bundled dependency for {dep} (needed by {path}) not found")
        args.extend(("-change", dep, f"@rpath/{relative}"))
    new_ids = {f"@rpath/{bundled_name(name)}" for name in info.ids
               if name.startswith("/") and not is_system(name)}
    if len(new_ids) > 1:
        raise RuntimeError(f"Conflicting dylib IDs in {path}: {info.ids}")
    if new_ids:
        args.extend(("-id", next(iter(new_ids))))
    for rpath in info.rpaths:
        if rpath.startswith("/") and not is_system(rpath):
            args.extend(("-delete_rpath", rpath))
    return args


def deploy(app, macdeployqt, plugin_dir, jobs):
    app = app.resolve()
    main = app / "Contents/MacOS/VC3D"
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        with timed("inspect deployment roots"):
            files = macho_files(app)
            commands = dict(zip(files, pool.map(inspect, files)))
            if main not in commands:
                raise RuntimeError(f"Missing Mach-O executable: {main}")
            roots = deployment_roots(app, commands)
            cli_count = sum(path.parent == main.parent and path != main
                            for path in commands)
            print(f"-- macOS packaging: {len(roots)}/{cli_count} CLI deployment roots", flush=True)

        with timed("macdeployqt"):
            # Keep Qt's copying, plugin selection and stripping behavior. Only
            # signing is deferred until all path changes have been made.
            run(str(macdeployqt), str(app), "-no-codesign",
                *(f"-executable={path}" for path in roots), stream=True)

        offscreen = plugin_dir / "platforms/libqoffscreen.dylib"
        if offscreen.exists():
            destination = app / "Contents/PlugIns/platforms"
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copy2(offscreen, destination)
        else:
            print(f"-- WARNING: offscreen Qt plugin not found at {offscreen}", flush=True)

        with timed("inspect and normalize bundle"):
            files = macho_files(app)
            # macdeployqt has changed existing binaries and added libraries.
            # Read each final binary once; do not reuse pre-deployment metadata.
            commands = dict(zip(files, pool.map(inspect, files)))
            # Validate the entire closure before starting any rewrites.
            edits = [(path, rewrite_args(app, path, info)) for path, info in commands.items()]

            def rewrite(edit):
                path, args = edit
                if args:
                    run("install_name_tool", *args, str(path))

            list(pool.map(rewrite, edits))
            print(f"-- macOS packaging: {len(files)} Mach-O files, "
                  f"{sum(bool(args) for _, args in edits)} rewrites, {jobs} workers", flush=True)

        with timed("sign nested binaries"):
            def sign(path):
                run("codesign", "--force", "--sign", "-", str(path))

            # Signing VC3D itself can cause codesign to seal the whole bundle.
            # Wait for all nested code to be signed before signing the app.
            list(pool.map(sign, (path for path in files if path != main)))

    with timed("sign and verify app"):
        run("codesign", "--force", "--sign", "-", str(app))
        run("codesign", "--verify", "--deep", "--strict", str(app))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", required=True, type=Path)
    parser.add_argument("--macdeployqt", required=True, type=Path)
    parser.add_argument("--plugin-dir", required=True, type=Path)
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    try:
        deploy(args.app, args.macdeployqt, args.plugin_dir, args.jobs)
    except (OSError, RuntimeError) as error:
        parser.exit(1, f"macOS deployment failed: {error}\n")


if __name__ == "__main__":
    main()
