"""Helper to launch the real VC3D binary and capture its agent-bridge handshake."""
from __future__ import annotations

import os
import subprocess
import threading
import time
from typing import Optional

from bridge_client import parse_handshake_line


def find_running_vc3d_pids() -> list[int]:
    """Best-effort scan for already-running VC3D processes (macOS/Linux)."""
    try:
        out = subprocess.check_output(["pgrep", "-x", "VC3D"], text=True)
    except subprocess.CalledProcessError:
        return []
    except FileNotFoundError:
        return []
    return [int(p) for p in out.split() if p.strip()]


class VC3DProcess:
    def __init__(self, binary: str, extra_args: list[str],
                 env_overrides: Optional[dict] = None, log_cap: int = 8000):
        env = dict(os.environ)
        if env_overrides:
            env.update(env_overrides)

        self.log_lines: list[str] = []
        self._log_lock = threading.Lock()
        self._log_cap = log_cap
        self._handshake_path: Optional[str] = None
        self._handshake_event = threading.Event()

        self.proc = subprocess.Popen(
            [binary] + extra_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        self._reader_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader_thread.start()

    def _read_stdout(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            stripped = line.rstrip("\n")
            with self._log_lock:
                self.log_lines.append(stripped)
                if len(self.log_lines) > self._log_cap:
                    del self.log_lines[0]
            if not self._handshake_event.is_set():
                path = parse_handshake_line(stripped)
                if path:
                    self._handshake_path = path
                    self._handshake_event.set()

    def wait_for_handshake(self, timeout: float = 60.0) -> str:
        if not self._handshake_event.wait(timeout):
            tail = self.tail_log(60)
            raise TimeoutError(
                "VC3D did not print the agent-bridge handshake line within "
                f"{timeout}s. Process alive={self.is_running()}. Log tail:\n" +
                "\n".join(tail)
            )
        assert self._handshake_path is not None
        return self._handshake_path

    def tail_log(self, n: int = 40) -> list[str]:
        with self._log_lock:
            return list(self.log_lines[-n:])

    def is_running(self) -> bool:
        return self.proc.poll() is None

    def exit_code(self) -> Optional[int]:
        return self.proc.poll()

    def terminate(self, timeout: float = 10.0) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                try:
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
