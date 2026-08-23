from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).parent / "vc" / "fiberlets_cli.py"
SPEC = importlib.util.spec_from_file_location("vc_fiberlets_cli_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_packaged_binary_is_private_to_vc_package():
    binary = MODULE.packaged_binary()
    assert binary.parent.name == "bin"
    assert binary.parent.parent.name == "vc"
    assert binary.name.startswith("vc_fiberlets")


def test_main_executes_native_binary_with_original_arguments(tmp_path, monkeypatch):
    binary = tmp_path / "vc_fiberlets"
    binary.touch()
    observed = {}

    def fake_execv(path, argv):
        observed["path"] = path
        observed["argv"] = argv
        raise RuntimeError("exec intercepted")

    monkeypatch.setattr(MODULE, "packaged_binary", lambda: binary)
    monkeypatch.setattr(MODULE.os, "execv", fake_execv)
    monkeypatch.setattr(sys, "argv", ["vc_fiberlets", "preprocess-volume", "input"])

    try:
        MODULE.main()
    except RuntimeError as exc:
        assert str(exc) == "exec intercepted"
    else:
        raise AssertionError("os.execv was not called")

    assert observed == {
        "path": str(binary),
        "argv": [str(binary), "preprocess-volume", "input"],
    }
