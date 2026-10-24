"""Windows installs of the ``models`` extra must get CUDA builds of torch.

PyPI only publishes CPU wheels of torch/torchvision for Windows. Without an
explicit source, ``uv sync --extra models`` on Windows installs
``torch 2.12.1+cpu`` and every ink/model command quietly runs on the CPU, while
Linux gets CUDA from the very same lock. These checks read the committed
``uv.lock`` so they run (and fail) on any OS, without installing anything.
"""

from __future__ import annotations

from pathlib import Path
import tomllib

import pytest

_PROJECT_ROOT = Path(__file__).parents[2]
_CUDA_INDEX_PREFIX = "https://download.pytorch.org/whl/cu"


def _load(name: str) -> dict:
    with (_PROJECT_ROOT / name).open("rb") as stream:
        return tomllib.load(stream)


def _locked(package: str) -> list[dict]:
    return [p for p in _load("uv.lock")["package"] if p["name"] == package]


@pytest.mark.parametrize("package", ["torch", "torchvision"])
def test_windows_lock_entry_comes_from_a_cuda_index(package):
    entries = _locked(package)
    cuda = [e for e in entries if e["source"].get("registry", "").startswith(_CUDA_INDEX_PREFIX)]
    assert cuda, (
        f"{package} is only locked from {[e['source'] for e in entries]}; "
        "PyPI has no CUDA wheels for Windows, so Windows would get a CPU-only build"
    )
    (entry,) = cuda
    wheel_names = [w["url"].rsplit("/", 1)[-1] for w in entry.get("wheels", [])]
    assert any(n.endswith("win_amd64.whl") for n in wheel_names), wheel_names
    assert "+cu" in entry["version"], entry["version"]


@pytest.mark.parametrize("package", ["torch", "torchvision"])
def test_cuda_index_is_limited_to_windows_and_versions_match_linux(package):
    sources = _load("pyproject.toml")["tool"]["uv"]["sources"][package]
    assert [s.get("marker") for s in sources] == ["sys_platform == 'win32'"]

    # Linux/macOS keep the PyPI wheels (Linux ones already bundle CUDA), and
    # Windows must not drift to a different release than the one Linux tests.
    versions = {e["source"].get("registry", ""): e["version"] for e in _locked(package)}
    pypi = versions.pop("https://pypi.org/simple")
    (windows,) = versions.values()
    assert windows.split("+", 1)[0] == pypi
