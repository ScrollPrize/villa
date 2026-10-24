"""uv.lock must satisfy the requirements of the monorepo path dependencies.

``volume-cartographer``, ``villa-gpu-manager`` and the vendored nnU-Net
packages are resolved from sibling folders via ``[tool.uv.sources]``. When one
of their ``pyproject.toml`` files raises a pin (e.g. ``vc-delta3d>=0.2.0``)
without a relock here, ``uv sync --frozen`` keeps installing the stale version
and ``uv lock --check`` fails. This reads the files directly, so it needs
neither uv nor network.
"""

from __future__ import annotations

from pathlib import Path
import tomllib

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

_PROJECT_ROOT = Path(__file__).parents[2]


def _load(path: Path) -> dict:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def _path_sources() -> list[tuple[str, Path]]:
    sources = _load(_PROJECT_ROOT / "pyproject.toml")["tool"]["uv"]["sources"]
    return [
        (name, (_PROJECT_ROOT / spec["path"]).resolve())
        for name, spec in sorted(sources.items())
        if isinstance(spec, dict) and "path" in spec
    ]


def _locked_versions() -> dict[str, list[Version]]:
    versions: dict[str, list[Version]] = {}
    for package in _load(_PROJECT_ROOT / "uv.lock")["package"]:
        if "version" not in package:  # dynamic-version local packages
            continue
        versions.setdefault(canonicalize_name(package["name"]), []).append(
            Version(package["version"])
        )
    return versions


@pytest.mark.parametrize("name,path", _path_sources(), ids=lambda v: str(v))
def test_locked_versions_satisfy_path_dependency_requirements(name, path):
    pyproject = path / "pyproject.toml"
    if not pyproject.exists():
        pytest.skip(f"{pyproject} not checked out")
    locked = _locked_versions()
    stale = []
    for text in _load(pyproject).get("project", {}).get("dependencies", []):
        requirement = Requirement(text)
        candidates = locked.get(canonicalize_name(requirement.name))
        if not candidates or not requirement.specifier:
            continue  # unlocked here (e.g. platform-gated) or unpinned
        if not any(requirement.specifier.contains(v, prereleases=True) for v in candidates):
            stale.append(f"{requirement} (locked: {', '.join(map(str, candidates))})")
    assert not stale, f"{name}: uv.lock is stale, run `uv lock`: {stale}"
