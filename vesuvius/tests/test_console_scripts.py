"""Every console_script target must resolve to a module that exists on disk.

Entry points are resolved only when the command is invoked, so a path left
stale by a file move survives the build, the install and the rest of the test
suite -- the command simply fails with ModuleNotFoundError for whoever runs
it. Two entry points were stale for about ten months for exactly this reason
and were fixed in #1447; this test is the regression guard for that class.

The check reads pyproject.toml and the source tree only. It deliberately
imports nothing from vesuvius, so it still runs when a package's optional
dependencies are absent.
"""

from __future__ import annotations

import tomllib
from pathlib import Path


def test_console_scripts_resolve_to_real_modules() -> None:
    root = Path(__file__).resolve().parents[1]
    with (root / "pyproject.toml").open("rb") as handle:
        scripts = tomllib.load(handle)["project"]["scripts"]

    src = root / "src"
    missing = []
    for name, target in sorted(scripts.items()):
        module = target.split(":", 1)[0]
        relative = Path(*module.split("."))
        if not (src / relative.with_suffix(".py")).is_file() and not (
            src / relative / "__init__.py"
        ).is_file():
            missing.append(f"{name} -> {module}")

    assert not missing, "console_scripts pointing at missing modules: " + ", ".join(
        missing
    )
