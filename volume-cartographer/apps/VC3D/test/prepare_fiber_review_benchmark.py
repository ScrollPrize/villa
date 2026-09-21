"""Prepare an isolated Ninja benchmark using an already-built VC3D's objects.

Build VC3D normally first. This only writes a manifest; it does not build or
install anything, and never replaces application objects or executables.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    args = parser.parse_args()
    build = args.build_dir.resolve()
    output = build / "fiber-review-benchmark"
    output.mkdir(exist_ok=True)
    source = Path(__file__).with_name("benchmark_fiber_review_render.cpp").resolve()
    obj = output / "benchmark.o"
    depfile = output / "benchmark.d"
    commands = json.loads((build / "compile_commands.json").read_text())
    command = next(c for c in commands if c["file"].endswith("/apps/VC3D/VCAppMain.cpp"))
    compile_args = shlex.split(command["command"])
    compile_args[compile_args.index("-o") + 1] = str(obj)
    compile_args[compile_args.index("-c") + 1] = str(source)
    compile_args += ["-MMD", "-MF", str(depfile)]
    link_commands = subprocess.check_output(
        ["ninja", "-C", str(build), "-t", "commands", "bin/VC3D"], text=True)
    link = next(line for line in reversed(link_commands.splitlines()) if " -o bin/VC3D " in line)
    link = link.replace("apps/VC3D/CMakeFiles/VC3D.dir/VCAppMain.cpp.o", shlex.quote(str(obj)))
    # Keep the executable next to VC3D so its existing relative library paths work.
    binary = build / "bin" / "benchmark_fiber_review_render"
    link = link.replace(" -o bin/VC3D ", " -o " + shlex.quote(str(binary)) + " ")

    def command_escape(text):
        return text.replace("$", "$$")

    def path_escape(path):
        return str(path).replace("$", "$$").replace(" ", "$ ").replace(":", "$:")

    manifest = output / "build.ninja"
    manifest.write_text(
        "rule compile\n  command = " + command_escape(shlex.join(compile_args)) +
        "\n  depfile = " + path_escape(depfile) + "\n  deps = gcc\n" +
        "rule link\n  command = " + command_escape(link) + "\n" +
        f"build {path_escape(obj)}: compile {path_escape(source)}\n" +
        f"build {path_escape(binary)}: link {path_escape(obj)} | bin/VC3D\n")
    print(shlex.join(["ninja", "-C", str(build), "-f", str(manifest)]))


if __name__ == "__main__":
    main()
