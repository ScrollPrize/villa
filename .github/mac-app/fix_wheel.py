"""Make a delocated macOS wheel load torch's OpenMP runtime, and check it is self-contained.

  fix_wheel.py strip WHEEL OUTDIR [--keep-lib-dylibs]  drop the C++ install trees, repack into OUTDIR
  fix_wheel.py fix WHEEL OUTDIR TORCH_LIBOMP...   rewrite, re-sign, repack into OUTDIR
  fix_wheel.py check WHEEL                         print every Mach-O's links; fail on a leak
  fix_wheel.py check-locks FOLDER                  fail unless the locks pin each local wheel by its own hash only

strip: a non-editable scikit-build wheel holds everything `cmake --install` writes, so vc_reader's and
volume-cartographer's also put headers, static archives and command-line tools in site-packages/include, lib, bin and
share. Python never loads those; they are dropped before delocate (which would otherwise chase their dependencies).
--keep-lib-dylibs keeps each lib/*.dylib that a module outside lib/ links: vc_reader's _reader module loads lib/libvc_reader_native.dylib (its rpath is
@loader_path/../lib). volume-cartographer's modules load copies at the site-packages root instead, and its lib/ copies
would overwrite vc_reader's file of the same name.

fix: every Mach-O that links a libomp by absolute path (Homebrew's, which delocate is told to skip) is pointed at
@rpath/libomp.dylib with an rpath to site-packages/torch/lib, so the process holds the one OpenMP runtime torch already
loads (two copies stop at OMP Error #15). Absolute LC_RPATH entries (build and Homebrew directories) are removed. Each
changed file is ad-hoc re-signed, since an edited arm64 Mach-O no longer matches its signature. Then every libomp symbol
the wheel imports must be exported by each TORCH_LIBOMP given, else a symbol would be missing at load or first call.

check: fails if any load command, rpath or install id names /opt/homebrew, /usr/local, /Users, /private, /tmp or
/opt/llvm-openmp; prints each Mach-O's non-system dependencies.

check-locks: FOLDER holds the wheels and the *requirements-*.txt locks compiled against them and PyPI. Most of our
package names are unclaimed on PyPI, so a package published there under one of them at the same version would
otherwise get its hash into a lock, and the installer would accept that file
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

LEAK = re.compile(r"(/opt/homebrew|/usr/local|/Users/|/private/|/tmp/|/opt/llvm-openmp)")
MACHO_MAGIC = {b"\xcf\xfa\xed\xfe", b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca"}


def run(*args: str) -> str:
	return subprocess.run(args, check=True, capture_output=True, text=True).stdout


def macho_files(root: Path) -> list[Path]:
	found = []
	for path in sorted(root.rglob("*")):
		if path.is_file() and not path.is_symlink():
			with path.open("rb") as handle:
				if handle.read(4) in MACHO_MAGIC:
					found.append(path)
	return found


def linked(path: Path) -> list[str]:
	# otool -L prints the file name, then the install id (for a dylib) and each dependency
	return [line.split(" (compatibility")[0].strip() for line in run("otool", "-L", str(path)).splitlines()[1:]]


def rpaths(path: Path) -> list[str]:
	lines = run("otool", "-l", str(path)).splitlines()
	found = []
	for i, line in enumerate(lines):
		if line.strip() == "cmd LC_RPATH":
			found.append(lines[i + 2].split("path ", 1)[1].rsplit(" (offset", 1)[0].strip())
	return found


def install_id(path: Path) -> str | None:
	out = run("otool", "-D", str(path)).splitlines()
	return out[1].strip() if len(out) > 1 else None


def libomp_imports(path: Path) -> set[str]:
	# undefined symbols bound to the libomp dylib (two-level namespace: "(from libomp)")
	out = run("nm", "-u", "-m", str(path))
	return set(re.findall(r"external (\S+) \(from libomp\)", out))


def exports(path: Path) -> set[str]:
	return {line.split()[-1] for line in run("nm", "-gU", str(path)).splitlines() if line.strip()}


def fix(wheel: Path, outdir: Path, torch_libomps: list[Path]) -> None:
	with tempfile.TemporaryDirectory() as tmp:
		subprocess.run(["uvx", "--from", "wheel==0.45.1", "wheel", "unpack", "-d", tmp, str(wheel)], check=True)
		root = next(Path(tmp).iterdir())
		needed: set[str] = set()
		for path in macho_files(root):
			changed = False
			omp = [dep for dep in linked(path) if dep.endswith("/libomp.dylib") and not dep.startswith("@")]
			for dep in omp:
				run("install_name_tool", "-change", dep, "@rpath/libomp.dylib", str(path))
				changed = True
			if omp or "@rpath/libomp.dylib" in linked(path):
				torch_lib = "@loader_path/" + os.path.relpath(root / "torch" / "lib", path.parent)
				if torch_lib not in rpaths(path):
					run("install_name_tool", "-add_rpath", torch_lib, str(path))
					changed = True
				needed |= libomp_imports(path)
			for rpath in rpaths(path):
				if rpath.startswith("/"):
					run("install_name_tool", "-delete_rpath", rpath, str(path))
					changed = True
			if changed:
				run("codesign", "--force", "--sign", "-", str(path))
				print(f"fixed {path.relative_to(root)}")
		for libomp in torch_libomps:
			missing = needed - exports(libomp)
			if missing:
				raise SystemExit(f"{libomp} lacks {len(missing)} libomp symbols the wheel imports: {sorted(missing)[:20]}")
			print(f"all {len(needed)} libomp imports exported by {libomp}")
		outdir.mkdir(parents=True, exist_ok=True)
		subprocess.run(["uvx", "--from", "wheel==0.45.1", "wheel", "pack", "-d", str(outdir), str(root)], check=True)


INSTALL_TREES = ("include", "lib", "bin", "share")


def strip(wheel: Path, outdir: Path, keep_lib_dylibs: bool) -> None:
	with tempfile.TemporaryDirectory() as tmp:
		subprocess.run(["uvx", "--from", "wheel==0.45.1", "wheel", "unpack", "-d", tmp, str(wheel)], check=True)
		root = next(Path(tmp).iterdir())
		for name in INSTALL_TREES:
			tree = root / name
			if not tree.is_dir():
				continue
			if name == "lib" and keep_lib_dylibs:
				used = {Path(dep).name for path in macho_files(root) if tree not in path.parents
						for dep in linked(path) if dep.startswith("@rpath/")}
				for path in sorted(tree.iterdir()):
					if path.is_dir():
						shutil.rmtree(path)
					elif path.name not in used:
						path.unlink()
				print(f"kept {', '.join(sorted(p.name for p in tree.iterdir()))} in lib/ of {wheel.name}")
				continue
			shutil.rmtree(tree)
			print(f"dropped {name}/ from {wheel.name}")
		outdir.mkdir(parents=True, exist_ok=True)
		subprocess.run(["uvx", "--from", "wheel==0.45.1", "wheel", "pack", "-d", str(outdir), str(root)], check=True)


def check(wheel: Path) -> None:
	leaks = []
	with tempfile.TemporaryDirectory() as tmp:
		shutil.unpack_archive(str(wheel), tmp, format="zip")
		root = Path(tmp)
		files = macho_files(root)
		for path in files:
			rel = path.relative_to(root)
			deps = linked(path)
			names = [*deps, *rpaths(path), install_id(path) or ""]
			leaks += [f"{rel}: {name}" for name in names if LEAK.search(name)]
			ours = [dep for dep in deps if not dep.startswith(("/usr/lib/", "/System/"))]
			print(f"{rel}: {', '.join(ours) or 'system only'}; rpaths {rpaths(path)}")
	print(f"{wheel.name}: {len(files)} Mach-O files")
	if leaks:
		raise SystemExit("leaked paths:\n" + "\n".join(leaks))


def check_locks(folder: Path) -> None:
	wheels = {}
	for wheel in folder.glob("*.whl"):
		name, version = wheel.name.split("-")[:2]
		wheels[(normalize(name), version)] = "sha256:" + hashlib.sha256(wheel.read_bytes()).hexdigest()
	for lock in sorted(folder.glob("*requirements-*.txt")):
		# one block per requirement: the pin line, then its indented --hash and "# via" lines
		for block in re.split(r"\n(?=\S)", lock.read_text()):
			pin = re.match(r"([A-Za-z0-9_.-]+)==(\S+)", block)
			if not pin or (normalize(pin[1]), pin[2]) not in wheels:
				continue
			want = wheels[(normalize(pin[1]), pin[2])]
			hashes = re.findall(r"--hash=(sha256:[0-9a-f]+)", block)
			if hashes != [want]:
				raise SystemExit(f"{lock.name}: {pin[0]} pins {hashes}, the local wheel is {want}")
	print(f"{len(wheels)} local wheels; the locks pin only their hashes for them")


def normalize(name: str) -> str:
	return re.sub(r"[-_.]+", "-", name).lower()


if __name__ == "__main__":
	if sys.argv[1] == "strip":
		strip(Path(sys.argv[2]), Path(sys.argv[3]), "--keep-lib-dylibs" in sys.argv[4:])
	elif sys.argv[1] == "fix":
		fix(Path(sys.argv[2]), Path(sys.argv[3]), [Path(p) for p in sys.argv[4:]])
	elif sys.argv[1] == "check":
		for name in sys.argv[2:]:
			check(Path(name))
	elif sys.argv[1] == "check-locks":
		check_locks(Path(sys.argv[2]))
	else:
		raise SystemExit(__doc__)
