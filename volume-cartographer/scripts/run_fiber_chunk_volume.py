#!/usr/bin/env python3
"""Run the whole-volume command from this checkout's existing native build.

No installation or build side effects. VC_BUILD_DIR may select another build.
Explicit imports avoid an editable installation silently using another checkout.
"""
import importlib.util
import os
from pathlib import Path
import sys


def main():
    project = Path(__file__).resolve().parents[1]
    build = Path(os.environ.get("VC_BUILD_DIR", project / "build/fiber-evaluation"))
    library_variable = "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
    libraries = ":".join(str(build / p) for p in ("core", "utils", "libs/c3d", "libs/volcomp"))
    if not os.environ.get(library_variable, "").startswith(libraries):
        os.environ[library_variable] = libraries + ":" + os.environ.get(library_variable, "")
        os.execv(sys.executable, [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])
    native_root, source_root = build / "python/vc", project / "python/vc"
    spec = importlib.util.spec_from_file_location("vc", source_root / "__init__.py",
        submodule_search_locations=[str(native_root), str(source_root)])
    package = importlib.util.module_from_spec(spec)
    sys.modules["vc"] = package
    def load(name, path):
        spec = importlib.util.spec_from_file_location("vc." + name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        setattr(package, name, module)
        return module
    for name in ("volume", "fiber_trace"):
        modules = list(native_root.glob(name + ".*.so"))
        if len(modules) != 1:
            raise RuntimeError(f"expected one built {name} module below {native_root}; build vc_volume/vc_fiber_trace first")
        load(name, modules[0])
    spec.loader.exec_module(package)
    load("fiber_evaluation", source_root / "fiber_evaluation.py")
    workflow = load("fiber_chunk_volume", source_root / "fiber_chunk_volume.py")
    # The managed child must use this same checkout and build as its supervisor.
    os.environ["VC_CHUNK_VOLUME_LAUNCHER"] = str(Path(__file__).resolve())
    sys.path.insert(0, str(project.parent / "gpu_manager/src"))
    return workflow.main()


if __name__ == "__main__":
    raise SystemExit(main())
