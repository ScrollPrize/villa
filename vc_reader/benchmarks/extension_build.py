"""Select compiled extensions from an existing build for benchmarks and tests."""
import importlib.util
from pathlib import Path
import sys


def load_extensions(directory):
    import importlib.machinery

    for name in ("_reader", "_transforms"):
        candidates = [directory / (name + suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES]
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            raise ValueError(f"no {name} extension in {directory}")
        # Several test/benchmark modules can request the same build in one
        # process. Loading it twice re-registers nanobind types and breaks casts.
        loaded = sys.modules.get(f"vc_reader.{name}")
        if loaded is not None and Path(loaded.__file__).resolve() == path.resolve():
            continue
        spec = importlib.util.spec_from_file_location(f"vc_reader.{name}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[spec.name] = module
