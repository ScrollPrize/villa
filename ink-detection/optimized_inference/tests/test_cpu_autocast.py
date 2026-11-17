"""CPU BF16 autocast is used only with native oneDNN BF16 support.

The tests do not need any particular Torch version: torch.cpu.get_capabilities()
(Torch 2.11+) and the older private probes are patched with create=True or removed.
"""

import ast
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import device_utils  # noqa: E402

CPU = torch.device("cpu")
PROBES = ("_is_avx512_bf16_supported", "_is_amx_tile_supported")


@contextmanager
def _backend(supported=True):
    """oneDNN is available, enabled, and its BF16 query returns *supported*."""
    with patch.object(torch.backends.mkldnn, "is_available", return_value=True), \
            patch.object(torch.ops.mkldnn, "_is_mkldnn_bf16_supported", return_value=supported), \
            torch.backends.mkldnn.flags(enabled=True):
        yield


@contextmanager
def _without_get_capabilities():
    """Emulate Torch < 2.11, which has no torch.cpu.get_capabilities()."""
    missing = object()
    saved = torch.cpu.__dict__.get("get_capabilities", missing)
    if saved is not missing:
        delattr(torch.cpu, "get_capabilities")
    try:
        yield
    finally:
        if saved is not missing:
            torch.cpu.get_capabilities = saved


@contextmanager
def _private_probes(avx512_bf16, amx_tile):
    with patch.object(torch.cpu, PROBES[0], return_value=avx512_bf16, create=True), \
            patch.object(torch.cpu, PROBES[1], return_value=amx_tile, create=True):
        yield


class TestCPUAutocast(unittest.TestCase):
    def test_cpu_policy_changes_actual_autocast(self):
        for supported in (False, True):
            with self.subTest(supported=supported), _backend(supported), \
                    patch.object(device_utils, "_native_cpu_bf16", return_value=True):
                with torch.autocast("cpu", enabled=device_utils.autocast_enabled(CPU)):
                    self.assertEqual(torch.is_autocast_enabled("cpu"), supported)
                    y = torch.ones(2, 2) @ torch.ones(2, 2)
                    self.assertEqual(y.dtype, torch.bfloat16 if supported else torch.float32)

    def test_unavailable_disabled_or_unknown_backend(self):
        with patch.object(torch.backends.mkldnn, "is_available", return_value=False):
            self.assertFalse(device_utils.autocast_enabled(CPU))
        with torch.backends.mkldnn.flags(enabled=False):
            self.assertFalse(device_utils.autocast_enabled(CPU))
        with patch.object(device_utils, "_native_cpu_bf16", return_value=True):
            for error in (AttributeError("missing"), RuntimeError("unsupported")):
                with patch.object(torch.ops.mkldnn, "_is_mkldnn_bf16_supported", side_effect=error):
                    self.assertFalse(device_utils.autocast_enabled(CPU))
        with _backend(True):
            for error in (AttributeError("missing"), RuntimeError("unsupported")):
                with patch.object(device_utils, "_native_cpu_bf16", side_effect=error):
                    self.assertFalse(device_utils.autocast_enabled(CPU))

    def test_native_hardware_is_required(self):
        # Torch 2.11+: get_capabilities() decides; private probes are ignored.
        cases = [
            ({}, False),
            ({"amx_tile": True}, False),
            ({"avx512_f": True}, False),
            ({"avx512_bf16": True}, True),
            ({"amx_bf16": True}, True),
        ]
        for capabilities, expected in cases:
            with self.subTest(caps=capabilities), _backend(True), _private_probes(True, True), \
                    patch.object(torch.cpu, "get_capabilities", return_value=capabilities, create=True):
                self.assertEqual(device_utils.autocast_enabled(CPU), expected)
        with _backend(True), patch.object(
            torch.cpu, "get_capabilities", side_effect=RuntimeError("cpuinfo"), create=True
        ):
            self.assertFalse(device_utils.autocast_enabled(CPU))

    def test_older_torch_uses_private_probes(self):
        # Torch 2.10 (the ink image) has no get_capabilities() and no AMX_BF16 probe.
        cases = [
            ((False, False), False),
            ((True, False), True),
            ((False, True), True),
        ]
        for probes, expected in cases:
            with self.subTest(probes=probes), _backend(True), _without_get_capabilities(), _private_probes(*probes):
                self.assertFalse(hasattr(torch.cpu, "get_capabilities"))
                self.assertEqual(device_utils.autocast_enabled(CPU), expected)
            with self.subTest(probes=probes, onednn=False), _backend(False), _without_get_capabilities(), \
                    _private_probes(*probes):
                self.assertFalse(device_utils.autocast_enabled(CPU))

    def test_no_probe_at_all_uses_fp32(self):
        missing = object()
        saved = {name: torch.cpu.__dict__.get(name, missing) for name in PROBES}
        with _backend(True), _without_get_capabilities():
            try:
                for name, value in saved.items():
                    if value is not missing:
                        delattr(torch.cpu, name)
                self.assertFalse(device_utils.autocast_enabled(CPU))
            finally:
                for name, value in saved.items():
                    if value is not missing:
                        setattr(torch.cpu, name, value)

    def test_installed_torch_paths_agree(self):
        # Unpatched: whichever probes this Torch has must run without error and agree.
        native = device_utils._native_cpu_bf16()
        self.assertIsInstance(native, bool)
        if hasattr(torch.cpu, "get_capabilities") and hasattr(torch.cpu, PROBES[0]):
            caps = torch.cpu.get_capabilities()
            self.assertEqual(bool(caps.get("avx512_bf16", False)), bool(torch.cpu._is_avx512_bf16_supported()))
            # Call the fallback probes directly: in Torch 2.11+ they read get_capabilities().
            fallback = any(bool(getattr(torch.cpu, name)()) for name in PROBES if hasattr(torch.cpu, name))
            # amx_tile may stand in for amx_bf16 in the fallback, never the reverse.
            self.assertTrue(fallback or not native)

    def test_accelerators_do_not_query_cpu(self):
        with patch.object(torch.backends.mkldnn, "is_available", side_effect=AssertionError("CPU queried")):
            for kind in ("cuda", "mps", "xpu"):
                self.assertTrue(device_utils.autocast_enabled(torch.device(kind)))

    def test_both_sites_use_policy(self):
        for name in ("inference.py", "entrypoint.py"):
            tree = ast.parse((ROOT / name).read_text())
            calls = [
                node for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "torch"
                and node.func.attr == "autocast"
            ]
            self.assertEqual(len(calls), 1)
            enabled = next(k.value for k in calls[0].keywords if k.arg == "enabled")
            self.assertEqual(ast.unparse(enabled), "autocast_enabled(device)")


if __name__ == "__main__":
    unittest.main()
