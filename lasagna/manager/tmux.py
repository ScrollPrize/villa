from __future__ import annotations

import os
import subprocess
from typing import Sequence


class Tmux:
    def __init__(self, executable: str = "tmux") -> None:
        self.executable = executable

    def _run(self, args: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [self.executable, *args], check=check, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )

    def has_session(self, session: str) -> bool:
        return self._run(("has-session", "-t", session), check=False).returncode == 0

    def create(self, session: str, window: str, argv: Sequence[str]) -> None:
        if self.has_session(session):
            raise ValueError(f"tmux session already exists: {session}")
        self._run(("new-session", "-d", "-s", session, "-n", window, *argv))

    def attach(self, session: str, *, environ: dict[str, str] | None = None) -> None:
        environment = os.environ if environ is None else environ
        if not environment.get("TMUX"):
            subprocess.run([self.executable, "attach-session", "-t", session], check=True)
            return
        current = self._run(("display-message", "-p", "#{window_index}")).stdout.strip()
        target = str(int(current) + 1)
        source = f"{session}:0"
        self._run(("link-window", "-a", "-s", source, "-t", current))
        self._run(("select-window", "-t", target))
