"""The fork guard in get_ink_metrics must survive a platform without fork.

Windows has no fork start method, and `multiprocessing.set_start_method('fork')`
raises ValueError there rather than RuntimeError. A guard that only catches
RuntimeError lets that through, and because the call lives in run_worker every
fold subprocess dies -- after the strip has already been built, with the
traceback redirected into logs/fold_N.log.

These tests emulate a fork-less platform by reducing the concrete-context table
to spawn, so the class of bug is caught without a Windows runner.
"""

import multiprocessing
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

GUARD_SOURCE = Path(__file__).resolve().parent.parent / "get_ink_metrics.py"


def _extract_guard() -> str:
    """The fork guard exactly as it stands in get_ink_metrics, de-indented.

    Read rather than copied so the test cannot drift away from the code it is
    protecting: if the guard is reworded, this picks up the new wording.

    Located by the call itself and then widened by indentation, so it does not
    depend on whether the guard is spelled as a `try` or as an `if`.
    """
    lines = GUARD_SOURCE.read_text(encoding="utf-8").splitlines()
    hits = [i for i, line in enumerate(lines) if "set_start_method('fork'" in line]
    assert len(hits) == 1, f"expected one fork call in get_ink_metrics.py, found {len(hits)}"
    call = hits[0]

    body = len(lines[call]) - len(lines[call].lstrip())
    start = call
    while start > 0:
        indent = len(lines[start - 1]) - len(lines[start - 1].lstrip())
        if lines[start - 1].strip() and indent >= body:
            start -= 1          # a comment or a `try:` at or inside the call's level
        elif lines[start - 1].strip() and indent == body - 4:
            start -= 1          # the `if`/`try` header the call hangs off
            break
        else:
            break

    end = call + 1
    while end < len(lines):
        stripped = lines[end].strip()
        indent = len(lines[end]) - len(lines[end].lstrip())
        if stripped and indent >= body:
            end += 1            # `except:` / `pass` still belong to the guard
        elif stripped.startswith(("except", "finally")):
            end += 1
        else:
            break

    return textwrap.dedent("\n".join(lines[start:end])) + "\n"


def _run_guard(fork_available: bool) -> subprocess.CompletedProcess:
    """Execute the guard in a fresh interpreter, optionally without fork.

    A subprocess, not monkeypatch: set_start_method mutates global interpreter
    state, and a test that leaves the start method changed would corrupt every
    later test in the session.
    """
    # Mutate the existing default context; do not replace it. multiprocessing
    # binds get_all_start_methods and get_start_method at import time to the
    # bound methods of the original DefaultContext instance, so rebinding
    # ctx._default_context leaves those module-level names answering from the
    # object that was replaced -- the guard is skipped correctly but the readout
    # still reports the real default. On Windows that is invisible, because
    # there genuinely is no fork and the emulation is a no-op.
    #
    # The get_all_start_methods override is only needed on CPython <= 3.13,
    # where it does not consult _concrete_contexts. requires-python is >=3.14,
    # so it is belt and braces for anyone running the suite on an older
    # interpreter.
    emulate = (
        "import multiprocessing, multiprocessing.context as ctx\n"
        "ctx._concrete_contexts = {'spawn': ctx._concrete_contexts['spawn']}\n"
        "ctx._default_context._default_context = ctx._concrete_contexts['spawn']\n"
        "ctx._default_context._actual_context = None\n"
        "multiprocessing.get_all_start_methods = lambda: ['spawn']\n"
        if not fork_available
        else ""
    )
    program = (
        "import multiprocessing\n"
        + emulate
        + _extract_guard()
        + "\nprint(multiprocessing.get_start_method())\n"
    )
    return subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
    )


def test_guard_does_not_raise_without_fork():
    """The guard must not propagate on a platform that has no fork."""
    result = _run_guard(fork_available=False)

    assert result.returncode == 0, (
        "the fork guard raised on a fork-less platform:\n" + result.stderr
    )
    assert result.stdout.strip() == "spawn"


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="no fork on this platform, so there is nothing to preserve",
)
def test_guard_still_selects_fork_where_fork_exists():
    """And it must keep doing what the comment above it is protecting."""
    result = _run_guard(fork_available=True)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "fork"
