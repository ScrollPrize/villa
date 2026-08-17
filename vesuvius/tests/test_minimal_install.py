from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest


# Blocking a module has to happen in a fresh interpreter: by the time a test runs, the
# extras are already in sys.modules, and deleting them there leaves partially
# initialised parents behind. `find_module` is gone from the meta-path protocol in
# 3.12+, so the blocker implements `find_spec`.
_BLOCKER = """
import sys

BLOCKED = {blocked!r}


class BlockExtras:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in BLOCKED:
            raise ImportError(f"No module named {{fullname!r}} (simulated)")
        return None


sys.meta_path.insert(0, BlockExtras())
"""


def _run_without(blocked: set[str], body: str) -> subprocess.CompletedProcess[str]:
    """Run `body` in a fresh interpreter where importing `blocked` raises ImportError."""

    script = _BLOCKER.format(blocked=sorted(blocked)) + textwrap.dedent(body)
    return subprocess.run(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


VOLUME_ONLY_MISSING = {"nest_asyncio", "aiohttp", "torch", "tifffile", "scipy"}


def test_catalog_reads_work_without_network_extras() -> None:
    """list_files and list_cubes only read packaged YAML, so a volume-only install
    must be able to call them. They used to fail because catalog.py imported
    aiohttp and nest_asyncio at module scope."""

    result = _run_without(
        VOLUME_ONLY_MISSING,
        """
        import vesuvius

        scrolls = vesuvius.list_files()
        cubes = vesuvius.list_cubes()
        assert isinstance(scrolls, dict) and scrolls, scrolls
        assert isinstance(cubes, dict), cubes
        print("OK")
        """,
    )
    assert result.returncode == 0, f"{result.stderr}\n{result.stdout}"
    assert "OK" in result.stdout


def test_missing_extra_reports_the_extra_not_a_none_typeerror() -> None:
    """A name guarded behind an uninstalled extra must say which extra is missing.

    The previous placeholder was None, so calling it raised
    `TypeError: 'NoneType' object is not callable` at the caller's own call site and
    threw away the ImportError that explained why.
    """

    result = _run_without(
        VOLUME_ONLY_MISSING,
        """
        import vesuvius

        # `except ... as exc` unbinds exc when the block exits, so read what we need
        # inside it rather than after.
        try:
            vesuvius.tifxyz.read_tifxyz
        except ImportError as exc:
            message = str(exc)
            cause = exc.__cause__
        else:
            raise AssertionError("expected ImportError")

        assert "render" in message, message
        assert "tifffile" in message, message
        assert isinstance(cause, ImportError), cause
        print("OK")
        """,
    )
    assert result.returncode == 0, f"{result.stderr}\n{result.stdout}"
    assert "OK" in result.stdout


def test_missing_extra_placeholder_stays_falsy_and_present() -> None:
    """Callers that probe with hasattr or truthiness must not change behaviour."""

    result = _run_without(
        VOLUME_ONLY_MISSING,
        """
        import vesuvius

        assert hasattr(vesuvius, "tifxyz")
        assert hasattr(vesuvius, "VCDataset")
        assert not vesuvius.tifxyz
        assert not vesuvius.VCDataset
        assert "unavailable" in repr(vesuvius.tifxyz), repr(vesuvius.tifxyz)
        print("OK")
        """,
    )
    assert result.returncode == 0, f"{result.stderr}\n{result.stdout}"
    assert "OK" in result.stdout


@pytest.mark.parametrize("name", ["list_files", "list_cubes", "update_list", "is_aws_ec2_instance"])
def test_catalog_names_are_callable_on_a_minimal_install(name: str) -> None:
    """Every catalog name stays a real callable, not a None placeholder."""

    result = _run_without(
        VOLUME_ONLY_MISSING,
        f"""
        import vesuvius

        target = getattr(vesuvius, {name!r})
        assert callable(target), (({name!r}, target))
        print("OK")
        """,
    )
    assert result.returncode == 0, f"{result.stderr}\n{result.stdout}"
    assert "OK" in result.stdout
