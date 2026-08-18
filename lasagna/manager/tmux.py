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

    def create(
        self, session: str, window: str, argv: Sequence[str], *, run_uuid: str,
    ) -> str:
        if self.has_session(session):
            raise ValueError(f"tmux session already exists: {session}")
        value = self._run((
            "new-session", "-d", "-P", "-F", "#{window_id}",
            "-s", session, "-n", window, *argv,
        )).stdout.strip()
        if not value.startswith("@"):
            raise RuntimeError(f"tmux returned invalid window id: {value!r}")
        self._run(("set-option", "-w", "-t", value, "@las_manager_run_uuid", run_uuid))
        return value

    def window_id(self, session: str) -> str:
        value = self._run(
            ("display-message", "-p", "-t", f"{session}:", "#{window_id}"),
        ).stdout.strip()
        if not value.startswith("@"):
            raise RuntimeError(f"tmux returned invalid window id for {session!r}: {value!r}")
        return value

    def has_window(self, window_id: str) -> bool:
        if not window_id.startswith("@"):
            return False
        return self._run(
            ("display-message", "-p", "-t", window_id, "#{window_id}"),
            check=False,
        ).returncode == 0

    def window_matches(self, window_id: str, run_uuid: str) -> bool:
        if not self.has_window(window_id):
            return False
        result = self._run((
            "show-options", "-w", "-v", "-t", window_id,
            "@las_manager_run_uuid",
        ), check=False)
        return result.returncode == 0 and result.stdout.strip() == run_uuid

    def tag_window(self, window_id: str, run_uuid: str) -> None:
        self._run((
            "set-option", "-w", "-t", window_id,
            "@las_manager_run_uuid", run_uuid,
        ))

    def _window_links(self, window_id: str) -> list[tuple[str, str]]:
        result = self._run((
            "list-windows", "-a", "-F",
            "#{session_name}\t#{window_index}\t#{window_id}",
        ))
        return [
            (parts[0], parts[1])
            for line in result.stdout.splitlines()
            if len(parts := line.split("\t")) == 3 and parts[2] == window_id
        ]

    def attach(
        self,
        session: str,
        *,
        window_id: str | None = None,
        run_uuid: str,
        environ: dict[str, str] | None = None,
    ) -> str:
        environment = os.environ if environ is None else environ
        if window_id and self.window_matches(window_id, run_uuid):
            source = window_id
        elif self.has_session(session):
            source = self.window_id(session)
            self.tag_window(source, run_uuid)
        else:
            raise ValueError(f"tmux window no longer exists for session {session!r}")
        if not environment.get("TMUX"):
            links = self._window_links(source)
            target_session = session if self.has_session(session) else (links[0][0] if links else "")
            target = next((f"{name}:{index}" for name, index in links if name == target_session), "")
            if not target_session or not target:
                raise ValueError(f"tmux window no longer exists for session {session!r}")
            self._run(("select-window", "-t", target))
            subprocess.run([self.executable, "attach-session", "-t", target_session], check=True)
            return source
        current_session = self._run(("display-message", "-p", "#{session_name}")).stdout.strip()
        current_index = self._run(("display-message", "-p", "#{window_index}")).stdout.strip()
        links = self._window_links(source)
        if not any(name == current_session for name, _index in links):
            self._run((
                "link-window", "-a", "-s", source,
                "-t", f"{current_session}:{current_index}",
            ))
            links = self._window_links(source)
        target = next(
            (f"{name}:{index}" for name, index in links if name == current_session),
            None,
        )
        if target is None:
            raise RuntimeError(f"linked tmux window {source} is not visible in {current_session}")
        self._run(("select-window", "-t", target))
        return source
