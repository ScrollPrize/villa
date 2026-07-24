"""Socket discovery, launch, console-tail, and process lifecycle tests."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import threading
import unittest
from unittest import mock

from vc3d_mcp import core, server as server_module
from vc3d_mcp.bridge_client import BridgeConnectionError, discover_registry_socket

class RegistryDiscoveryTest(unittest.TestCase):
    """discover_registry_socket: newest live entry wins, dead entries reaped.

    Mirrors AgentBridgeServer's registry-file convention (~/.vc3d/agent_bridge/
    <pid>.json with {pid, name, path, startedAt}) and the stale-PID cleanup of
    LasagnaServiceManager::discoverServices()."""

    def setUp(self) -> None:
        self.registry_dir = tempfile.mkdtemp(prefix="vc3d-registry-test-")

    def tearDown(self) -> None:
        shutil.rmtree(self.registry_dir, ignore_errors=True)

    def _write_entry(self, pid: int, path: str, started_at: float, name: str = "vc3d-agent") -> str:
        file_path = os.path.join(self.registry_dir, f"{pid}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(
                {"pid": pid, "name": f"{name}-{pid}", "path": path, "startedAt": started_at},
                f,
            )
        return file_path

    def test_missing_directory_returns_none(self) -> None:
        self.assertIsNone(
            discover_registry_socket(os.path.join(self.registry_dir, "does-not-exist"))
        )

    def test_empty_directory_returns_none(self) -> None:
        self.assertIsNone(discover_registry_socket(self.registry_dir))

    def test_dead_pid_entry_is_filtered_and_removed(self) -> None:
        # A PID that is extremely unlikely to be alive. os.kill(pid, 0) must
        # raise ProcessLookupError, so the entry is treated as stale.
        dead_pid = 2_000_000_000
        dead_file = self._write_entry(dead_pid, "/tmp/dead-bridge.sock", started_at=1000.0)

        result = discover_registry_socket(self.registry_dir)

        self.assertIsNone(result)
        self.assertFalse(
            os.path.exists(dead_file), "stale dead-PID registry file should be removed"
        )

    def test_live_entry_is_found(self) -> None:
        # os.getpid() is by definition alive right now.
        live_path = "/tmp/live-bridge.sock"
        live_file = self._write_entry(os.getpid(), live_path, started_at=1234.0)

        result = discover_registry_socket(self.registry_dir)

        self.assertEqual(result, live_path)
        self.assertTrue(os.path.exists(live_file), "live registry file must be kept")

    def test_newest_live_entry_wins_and_dead_reaped(self) -> None:
        # Two live entries (both this process's pid is only one file, so use the
        # real pid for the newest and the parent pid for an older-but-live one).
        older_path = "/tmp/older-bridge.sock"
        newer_path = "/tmp/newer-bridge.sock"
        dead_file = self._write_entry(2_000_000_001, "/tmp/dead2.sock", started_at=9999.0)
        # Newest startedAt should win even though the dead entry has a larger one.
        self._write_entry(os.getppid(), older_path, started_at=100.0)
        self._write_entry(os.getpid(), newer_path, started_at=500.0)

        result = discover_registry_socket(self.registry_dir)

        self.assertEqual(result, newer_path)
        self.assertFalse(os.path.exists(dead_file), "dead entry must be reaped")

    def test_malformed_file_is_reaped(self) -> None:
        bad_file = os.path.join(self.registry_dir, "99999.json")
        with open(bad_file, "w", encoding="utf-8") as f:
            f.write("{ not valid json")

        self.assertIsNone(discover_registry_socket(self.registry_dir))
        self.assertFalse(os.path.exists(bad_file), "malformed registry file should be removed")


class AutoLaunchTest(unittest.TestCase):
    def test_volume_package_uses_its_own_cli_option(self) -> None:
        self.assertEqual(
            server_module._launch_command("/tmp/VC3D", "/tmp/demo.volpkg.json"),
            ["/tmp/VC3D", "--agent-bridge", "--volpkg", "/tmp/demo.volpkg.json"],
        )

    def test_path_binary_is_used_before_repo_builds(self) -> None:
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(server_module.shutil, "which", return_value="/usr/bin/VC3D"),
            mock.patch.object(server_module, "default_vc3d_binary") as fallback,
        ):
            self.assertEqual(server_module.resolve_launch_binary(None), "/usr/bin/VC3D")
            fallback.assert_not_called()

    def test_invalid_explicit_binary_does_not_silently_fall_back(self) -> None:
        with (
            mock.patch.object(server_module, "_is_executable", return_value=False),
            mock.patch.object(server_module.shutil, "which") as path_lookup,
        ):
            self.assertIsNone(server_module.resolve_launch_binary("/missing/VC3D"))
            path_lookup.assert_not_called()

    def test_protocol_check_rejects_stale_bridge(self) -> None:
        with self.assertRaisesRegex(BridgeConnectionError, "expected 1, got None"):
            server_module._validate_protocol({"pong": True})


class NewTailLinesTest(unittest.TestCase):
    """_new_tail_lines: which consoleTail lines are new versus already seen."""

    def test_empty_prev_returns_all(self) -> None:
        self.assertEqual(core._new_tail_lines([], ["a", "b", "c"]), ["a", "b", "c"])

    def test_identical_lists_return_nothing(self) -> None:
        self.assertEqual(core._new_tail_lines(["a", "b"], ["a", "b"]), [])

    def test_rolling_window_slide(self) -> None:
        prev = [str(n) for n in range(2, 52)]   # 2..51
        cur = [str(n) for n in range(4, 54)]    # 4..53
        self.assertEqual(core._new_tail_lines(prev, cur), ["52", "53"])

    def test_disjoint_returns_all_of_cur(self) -> None:
        self.assertEqual(core._new_tail_lines(["a", "b"], ["x", "y"]), ["x", "y"])

    def test_cur_shorter_full_suffix_overlap(self) -> None:
        self.assertEqual(core._new_tail_lines(["a", "b", "c"], ["b", "c"]), [])


class TeardownReapTest(unittest.TestCase):
    """_terminate_launched_process must always reap the child, even when the
    post-kill() wait() times out."""

    def tearDown(self) -> None:
        server_module._launched_process = None

    def test_reap_process_waits(self) -> None:
        proc = subprocess.Popen(["true"])  # exits immediately
        server_module._reap_process(proc)
        self.assertIsNotNone(proc.returncode)

    def test_kill_timeout_schedules_background_reaper(self) -> None:
        # A child that survives terminate() and the timed wait()s after kill():
        # _terminate must spawn a background waiter that performs a final
        # unconditional (blocking) wait() so it can't linger as a zombie.
        class StubbornProc:
            def __init__(self) -> None:
                self.terminated = False
                self.killed = False
                self._reaped = threading.Event()

            def poll(self):
                return 0 if self._reaped.is_set() else None

            def terminate(self) -> None:
                self.terminated = True

            def kill(self) -> None:
                self.killed = True

            def wait(self, timeout=None):
                if timeout is not None:
                    # terminate()'s and kill()'s timed waits both "time out".
                    raise subprocess.TimeoutExpired(cmd="stubborn", timeout=timeout)
                # The background reaper's unconditional blocking wait reaps it.
                self._reaped.set()
                return 0

        proc = StubbornProc()
        server_module._launched_process = proc
        server_module._terminate_launched_process()
        self.assertTrue(proc.terminated)
        self.assertTrue(proc.killed)
        # Background reaper must eventually perform the blocking wait().
        self.assertTrue(proc._reaped.wait(timeout=3.0), "child was never reaped")
