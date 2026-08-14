"""Same-directory staging and atomic publication for preprocessing outputs."""

from __future__ import annotations

from pathlib import Path
import tempfile


def create_staged_output(output_path: Path) -> Path:
    """Create an empty same-directory temporary path for one output file."""

    with tempfile.NamedTemporaryFile(
        dir=output_path.parent,
        prefix=f".{output_path.stem}.",
        suffix=output_path.suffix,
        delete=False,
    ) as stream:
        return Path(stream.name)


def publish_staged_output(staged_path: Path, output_path: Path) -> None:
    """Atomically replace an output with a completed same-filesystem stage."""

    staged_path.replace(output_path)


def discard_staged_output(staged_path: Path) -> None:
    """Remove an unpublished staged file if it still exists."""

    staged_path.unlink(missing_ok=True)
