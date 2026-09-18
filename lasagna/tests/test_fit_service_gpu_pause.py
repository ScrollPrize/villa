"""A service started with --no-gpu-pause must not import gpu_pause.

gpu_pause coordinates GPU handover between processes through an advisory
file lock and a unix socket, so importing it pulls in fcntl and AF_UNIX.
Both call sites used to import it unconditionally and only then decide
whether to use it, which made the module a hard requirement even for a
service that was told there is nothing to coordinate with -- a private
service that owns its GPU for the run, or simply a host without fcntl.

These tests pin that the flag decides the import, in both directions.
"""
import sys
import unittest

import fit_service


class UnimportableGpuPause:
    """Make ``import gpu_pause`` fail, the way a host without fcntl would."""

    def __enter__(self):
        self._saved = sys.modules.get("gpu_pause", "absent")
        # A None entry in sys.modules is the documented way to make an
        # import raise ImportError without touching the import path.
        sys.modules["gpu_pause"] = None
        return self

    def __exit__(self, *exception):
        if self._saved == "absent":
            sys.modules.pop("gpu_pause", None)
        else:
            sys.modules["gpu_pause"] = self._saved
        return False


class GpuPauseOrNullTest(unittest.TestCase):
    def setUp(self):
        self._enabled = fit_service._gpu_pause_enabled
        self.addCleanup(
            setattr, fit_service, "_gpu_pause_enabled", self._enabled)

    def test_disabled_yields_a_no_op_without_importing_gpu_pause(self):
        fit_service._gpu_pause_enabled = False
        with UnimportableGpuPause():
            with fit_service._gpu_pause_or_null() as entered:
                self.assertIsNone(entered)

    def test_enabled_still_imports_gpu_pause(self):
        fit_service._gpu_pause_enabled = True
        with UnimportableGpuPause():
            with self.assertRaises(ImportError):
                fit_service._gpu_pause_or_null()

    def test_enabled_returns_the_real_context(self):
        sentinel = object()

        class StubModule:
            @staticmethod
            def gpu_pause_context():
                return sentinel

        saved = sys.modules.get("gpu_pause", "absent")
        sys.modules["gpu_pause"] = StubModule
        try:
            fit_service._gpu_pause_enabled = True
            self.assertIs(fit_service._gpu_pause_or_null(), sentinel)
        finally:
            if saved == "absent":
                sys.modules.pop("gpu_pause", None)
            else:
                sys.modules["gpu_pause"] = saved


if __name__ == "__main__":
    unittest.main()
