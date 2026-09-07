"""Guard against POSIX-only temporary paths baked into the package.

A literal like ``/tmp/vesuvius-models`` is not portable. On Windows it is a
*drive-relative* path, so ``os.path.abspath`` resolves it against whichever
drive happens to be current:

    cwd D:\\villa            ->  D:\\tmp\\vesuvius-models
    cwd C:\\Users\\someone   ->  C:\\tmp\\vesuvius-models

For a model or volume cache that means the cache silently moves when the
working drive changes -- multi-gigabyte checkpoints are re-downloaded, and
stray caches accumulate at the root of every drive the tool is ever run from.
Neither location is the platform temporary directory.

``tempfile.gettempdir()`` honours TMPDIR/TEMP/TMP and is correct on every
platform, so use it instead of a hardcoded literal.

This is a source-level check on purpose: the modules that held these literals
pull in torch and other heavy optional extras, so importing them here would
make the check unrunnable in a lightweight environment. Parsing keeps it cheap
and, unlike an import-time assertion, it fails on Linux too -- where ``/tmp``
happens to equal ``gettempdir()`` and a value-based assertion would pass while
the bug was still present.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

# POSIX temp roots that are wrong on Windows. Anchored so that unrelated paths
# such as "/tmpl" or "/tmp_data" are not flagged.
_POSIX_TEMP_ROOTS = ("/tmp", "/var/tmp")

_PACKAGE_ROOT = Path(__file__).parents[1] / "src" / "vesuvius"


def _is_posix_temp_literal(value: str) -> bool:
    return any(
        value == root or value.startswith(root + "/") for root in _POSIX_TEMP_ROOTS
    )


def _string_constants(tree: ast.AST) -> Iterator[tuple[int, str]]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.lineno, node.value


def test_no_hardcoded_posix_temp_paths_in_package():
    offenders: list[str] = []

    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError):  # pragma: no cover - defensive
            continue

        for lineno, value in _string_constants(tree):
            if _is_posix_temp_literal(value):
                rel = path.relative_to(_PACKAGE_ROOT.parents[1])
                offenders.append(f"{rel}:{lineno}: {value!r}")

    assert not offenders, (
        "Hardcoded POSIX temporary paths are not portable; on Windows they are "
        "drive-relative and the location silently follows the current drive. "
        "Use tempfile.gettempdir() instead.\n  " + "\n  ".join(offenders)
    )
