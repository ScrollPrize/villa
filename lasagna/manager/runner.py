from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from .runs import atomic_json, utc_now


def _load_completed_provenance(path: Path) -> tuple[list[object], str | None]:
    if not path.is_file():
        return [], f"portable provenance was not created: {path.name}"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return [], f"portable provenance is unreadable: {error}"
    if not isinstance(value, dict):
        return [], "portable provenance must contain a JSON object"
    artifacts = value.get("artifacts")
    inventory = list(artifacts) if isinstance(artifacts, list) else []
    if value.get("status") != "completed":
        return inventory, f"portable provenance status is {value.get('status')!r}, expected 'completed'"
    if not isinstance(artifacts, list):
        return [], "portable provenance has no artifact inventory"
    return inventory, None


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
    command = json.loads((run_dir / "command.json").read_text(encoding="utf-8"))["resolved_argv"]
    log = (run_dir / "run.log").open("ab", buffering=0)
    try:
        child = subprocess.Popen(command, cwd=run_dir, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    except Exception:
        metadata.update(status="failed", ended_at=utc_now(), exit_code=None)
        metadata["lifecycle"]["inference"] = "failed"
        atomic_json(metadata_path, metadata)
        log.close()
        raise
    metadata.update(status="running", started_at=utc_now(), pid=child.pid, process_start_time=_process_start_time(child.pid))
    metadata["lifecycle"]["inference"] = "running"
    atomic_json(metadata_path, metadata)
    interrupted = False

    def forward(signum, _frame):
        nonlocal interrupted
        interrupted = True
        try:
            os.killpg(child.pid, signum)
        except ProcessLookupError:
            pass

    signal.signal(signal.SIGINT, forward)
    signal.signal(signal.SIGTERM, forward)
    returncode = child.wait()
    status = "interrupted" if interrupted else "completed" if returncode == 0 else "failed"
    metadata.update(status=status, ended_at=utc_now(), exit_code=returncode)
    metadata["lifecycle"]["inference"] = status
    provenance_path = run_dir / metadata.get("artifacts", {}).get("provenance", "artifacts/inference.json")
    inventory, provenance_error = _load_completed_provenance(provenance_path)
    metadata.setdefault("artifacts", {})["inventory"] = inventory
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
