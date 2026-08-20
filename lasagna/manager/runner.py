from __future__ import annotations

import json
import os
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from .prefetch import execute_prefetch_request
from .runs import atomic_json, utc_now


class _RunInterrupted(BaseException):
    pass


class _TeeText:
    """Write authoritative output to a log and best-effort output to a pane."""

    def __init__(self, log, pane) -> None:
        self.log = log
        self.pane = pane
        self.encoding = getattr(pane, "encoding", None) or "utf-8"

    def write(self, value: str) -> int:
        written = self.log.write(value)
        self.log.flush()
        try:
            self.pane.write(value)
            self.pane.flush()
        except (BrokenPipeError, OSError):
            pass
        return written

    def flush(self) -> None:
        self.log.flush()
        try:
            self.pane.flush()
        except (BrokenPipeError, OSError):
            pass

    def isatty(self) -> bool:
        return bool(getattr(self.pane, "isatty", lambda: False)())

    def fileno(self) -> int:
        return self.pane.fileno()


def _load_completed_provenance(path: Path) -> tuple[list[object], dict | None, str | None]:
    if not path.is_file():
        return [], None, f"portable provenance was not created: {path.name}"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return [], None, f"portable provenance is unreadable: {error}"
    if not isinstance(value, dict):
        return [], None, "portable provenance must contain a JSON object"
    artifacts = value.get("artifacts")
    inventory = list(artifacts) if isinstance(artifacts, list) else []
    inference = value.get("inference")
    live_fetch = inference.get("live_fetch") if isinstance(inference, dict) else None
    live_fetch = dict(live_fetch) if isinstance(live_fetch, dict) else None
    if value.get("status") != "completed":
        return inventory, live_fetch, f"portable provenance status is {value.get('status')!r}, expected 'completed'"
    if not isinstance(artifacts, list):
        return [], live_fetch, "portable provenance has no artifact inventory"
    return inventory, live_fetch, None


def _process_start_time(pid: int) -> str | None:
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[21]
    except (OSError, IndexError):
        return None


def main(argv: list[str] | None = None) -> int:
    values = sys.argv[1:] if argv is None else argv
    if len(values) != 1:
        raise SystemExit("usage: python -m lasagna.manager.runner RUN_DIR")
    run_dir = Path(values[0]).resolve()
    metadata_path = run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    # New tmux launches reserve the window and then persist its atomically
    # returned ID.  Do not race that parent write with our first lifecycle
    # update.  Legacy command records omit the key and start immediately.
    if "tmux_window_id" in metadata and metadata["tmux_window_id"] is None:
        for _attempt in range(100):
            time.sleep(0.05)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("tmux_window_id") is not None:
                break
    command_record = json.loads((run_dir / "command.json").read_text(encoding="utf-8"))
    command = command_record["resolved_argv"]
    prefetch_request = command_record.get("prefetch")
    log = (run_dir / "run.log").open("a", encoding="utf-8", buffering=1)
    pane_stdout = _TeeText(log, sys.stdout)
    pane_stderr = _TeeText(log, sys.stderr)
    interrupted = False
    child: subprocess.Popen[bytes] | None = None

    def forward(signum, _frame):
        nonlocal interrupted
        interrupted = True
        if child is None:
            raise _RunInterrupted()
        try:
            os.killpg(child.pid, signum)
        except ProcessLookupError:
            pass

    signal.signal(signal.SIGINT, forward)
    signal.signal(signal.SIGTERM, forward)
    metadata.update(
        status="running", started_at=metadata.get("started_at") or utc_now(),
        pid=os.getpid(), runner_pid=os.getpid(),
        process_start_time=_process_start_time(os.getpid()),
    )
    lifecycle = metadata.setdefault("lifecycle", {})
    prefetch_details = metadata.setdefault(
        "prefetch", {"started_at": None, "ended_at": None, "error": None},
    )
    if prefetch_request is None:
        lifecycle["prefetch"] = "skipped"
    else:
        lifecycle["prefetch"] = "running"
        prefetch_details["started_at"] = utc_now()
    atomic_json(metadata_path, metadata)

    if prefetch_request is not None:
        try:
            with redirect_stdout(pane_stdout), redirect_stderr(pane_stderr):
                print("[las_manager] prefetch started", flush=True)
                prefetched = execute_prefetch_request(prefetch_request)
                print(f"[las_manager] prefetch completed: {prefetched}", flush=True)
        except _RunInterrupted:
            lifecycle["prefetch"] = "interrupted"
            prefetch_details["ended_at"] = utc_now()
            metadata.update(status="interrupted", ended_at=utc_now(), exit_code=128 + signal.SIGTERM)
            atomic_json(metadata_path, metadata)
            log.close()
            return 128 + signal.SIGTERM
        except BaseException as error:
            with redirect_stderr(pane_stderr):
                print("[las_manager] prefetch failed", file=sys.stderr, flush=True)
                traceback.print_exc()
            lifecycle["prefetch"] = "failed"
            prefetch_details.update(ended_at=utc_now(), error=f"{type(error).__name__}: {error}")
            metadata.update(status="failed", ended_at=utc_now(), exit_code=1)
            atomic_json(metadata_path, metadata)
            log.close()
            return 1
        lifecycle["prefetch"] = "completed"
        prefetch_details["ended_at"] = utc_now()
        atomic_json(metadata_path, metadata)
    try:
        child = subprocess.Popen(
            command, cwd=run_dir, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
    except Exception:
        metadata.update(status="failed", ended_at=utc_now(), exit_code=None)
        metadata["lifecycle"]["inference"] = "failed"
        atomic_json(metadata_path, metadata)
        log.close()
        raise
    metadata.update(status="running", pid=child.pid, process_start_time=_process_start_time(child.pid))
    metadata["lifecycle"]["inference"] = "running"
    atomic_json(metadata_path, metadata)
    assert child.stdout is not None
    log.flush()
    while True:
        chunk = child.stdout.read1(64 * 1024)
        if not chunk:
            break
        log.buffer.write(chunk)
        log.buffer.flush()
        try:
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
        except (BrokenPipeError, OSError):
            # Logging remains authoritative if the pane/client output closes.
            pass
    returncode = child.wait()
    status = "interrupted" if interrupted else "completed" if returncode == 0 else "failed"
    metadata.update(status=status, ended_at=utc_now(), exit_code=returncode)
    metadata["lifecycle"]["inference"] = status
    provenance_path = run_dir / metadata.get("artifacts", {}).get("provenance", "artifacts/inference.json")
    inventory, live_fetch, provenance_error = _load_completed_provenance(provenance_path)
    metadata.setdefault("artifacts", {})["inventory"] = inventory
    if live_fetch is not None:
        metadata["live_fetch"] = live_fetch
    if returncode == 0 and not interrupted and provenance_error:
        status = "failed"
        metadata.update(status=status)
        metadata["completion_error"] = provenance_error
    elif provenance_error:
        metadata["provenance_error"] = provenance_error
    metadata["lifecycle"]["inference"] = status
    atomic_json(metadata_path, metadata)
    log.close()
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
