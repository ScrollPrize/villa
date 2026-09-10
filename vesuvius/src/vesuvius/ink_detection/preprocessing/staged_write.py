"""Same-directory staging and atomic publication for preprocessing outputs."""

from __future__ import annotations

from pathlib import Path
import tempfile
import time

# Windows refuses to rename a path that anything still holds: ERROR_ACCESS_DENIED
# when a file inside a staged directory is open, ERROR_SHARING_VIOLATION when the
# path itself is in use. Both clear on their own; any other PermissionError means
# the rename is not going to start working, so it is raised at once.
_SHARING_VIOLATION_WINERRORS = frozenset({5, 32})


def create_staged_output(output_path: Path) -> Path:
    """Create an empty same-directory temporary path for one output file."""

    with tempfile.NamedTemporaryFile(
        dir=output_path.parent,
        prefix=f".{output_path.stem}.",
        suffix=output_path.suffix,
        delete=False,
    ) as stream:
        return Path(stream.name)


def publish_staged_output(
    staged_path: Path,
    output_path: Path,
    *,
    attempts: int = 6,
    retry_delay: float = 0.5,
) -> None:
    """Atomically replace an output with a completed same-filesystem stage.

    POSIX renames regardless of who holds the path and returns on the first pass.
    Windows raises PermissionError while anything holds the stage — one open file
    inside a staged directory is enough, and after a large write something holding
    a handle for a moment is ordinary — so a sharing violation is retried with
    backoff. If it never clears, the error carries the fact that matters: the
    staged output is finished, so nothing has to be computed again.
    """

    for attempt in range(1, attempts + 1):
        try:
            staged_path.replace(output_path)
            return
        except PermissionError as exc:
            if getattr(exc, "winerror", None) not in _SHARING_VIOLATION_WINERRORS:
                raise
            if attempt == attempts:
                exc.add_note(
                    "The staged output is complete and nothing needs recomputing; "
                    "it could not be published because something still holds it. "
                    "Renaming it finishes the job:\n"
                    f"  {staged_path}\n"
                    f"  -> {output_path}"
                )
                raise
            time.sleep(retry_delay * 2 ** (attempt - 1))


def discard_staged_output(staged_path: Path) -> None:
    """Remove an unpublished staged file if it still exists."""

    staged_path.unlink(missing_ok=True)
