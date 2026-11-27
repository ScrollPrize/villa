"""Check optional PyTorch ROCm runtime loading without requiring a GPU."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


class RocmInitTests(unittest.TestCase):
    def test_unloadable_torch_runtime_does_not_block_extension_loading(self):
        template = Path(__file__).resolve().parents[1] / "cmake" / "rocm_init.py.in"
        source = template.read_text().replace("@hip_VERSION_MAJOR@", "7").replace("@hip_VERSION_MINOR@", "2")
        namespace = {}
        exec(compile(source, str(template), "exec"), namespace)
        load = Mock(side_effect=OSError("missing runtime dependency"))

        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory) / "lib" / "libamdhip64.so"
            runtime.parent.mkdir()
            runtime.touch()
            namespace.update(
                sys=SimpleNamespace(modules={}),
                version=lambda _: "2.14.1+rocm7.2",
                find_spec=lambda _: SimpleNamespace(submodule_search_locations=[directory]),
                ctypes=SimpleNamespace(CDLL=load, RTLD_GLOBAL=1),
            )

            namespace["initialize"]()
            load.assert_called_once_with(str(runtime), mode=1)
