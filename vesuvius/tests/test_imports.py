from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

import vesuvius
from vesuvius.install.accept_terms import get_installation_path
from vesuvius.utils import catalog


def test_public_api_exports() -> None:
    """Package root should expose the documented public surface."""

    assert hasattr(vesuvius, "Volume")
    assert hasattr(vesuvius, "VCDataset")
    assert callable(vesuvius.list_files)
    assert callable(vesuvius.list_cubes)
    assert callable(vesuvius.update_list)
    assert hasattr(vesuvius, "utils")
    assert hasattr(vesuvius, "models")
    assert hasattr(vesuvius, "install")


def test_data_paths_lazy_exports() -> None:
    """vesuvius.data.paths should forward utilities without import-time recursion."""

    from vesuvius.data import paths

    assert paths.list_files is catalog.list_files
    assert paths.list_cubes is catalog.list_cubes
    assert paths.update_list is catalog.update_list
    assert paths.is_aws_ec2_instance is catalog.is_aws_ec2_instance


def test_config_directory_resolution() -> None:
    """Utility functions should resolve configuration files under vesuvius/install/configs."""

    install_root = Path(get_installation_path())
    config_dir = install_root / "vesuvius" / "install" / "configs"
    assert config_dir.is_dir()
    data = catalog.list_files()
    assert isinstance(data, dict)


@pytest.mark.parametrize(
    "module",
    [
        "vesuvius.models.training.train",
        "vesuvius.models.run.inference",
        "vesuvius.models.run.blending",
        "vesuvius.models.run.finalize_outputs",
        "vesuvius.structure_tensor.run_create_st",
    ],
)
def test_cli_entrypoints_show_help(module: str) -> None:
    """Console entrypoints should import cleanly and expose help output."""

    result = subprocess.run(
        [sys.executable, "-m", module, "-h"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert (  # noqa: PT017
        result.returncode == 0
    ), f"{module} -h failed: {result.stderr}\n{result.stdout}"


def _reimport_without(monkeypatch: pytest.MonkeyPatch, blocked: str, error=None):
    """Reimport vesuvius with one module import failing."""

    import builtins
    import importlib

    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        hit = name == blocked if "." in blocked else name.partition(".")[0] == blocked
        if hit:
            raise error or ModuleNotFoundError(f"No module named {blocked!r}", name=blocked)
        return real_import(name, *args, **kwargs)

    for module in [m for m in list(sys.modules) if m.startswith("vesuvius")]:
        monkeypatch.delitem(sys.modules, module, raising=False)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    return importlib.import_module("vesuvius")


CATALOG_HELPERS = ("list_files", "list_cubes", "update_list", "is_aws_ec2_instance")


@pytest.mark.parametrize(
    ("blocked", "distribution"),
    [("aiohttp", "aiohttp"), ("nest_asyncio", "nest-asyncio")],
)
def test_catalog_helpers_name_the_missing_package(
    monkeypatch: pytest.MonkeyPatch, blocked: str, distribution: str
) -> None:
    """Without a catalog dependency the helpers must name that exact package.

    test_public_api_exports already asserts vesuvius.list_files is callable. Before
    this change that held only when the optional dependencies happened to be
    installed: without them the helpers were bound to None, so the assertion failed
    and the first call raised ``TypeError: 'NoneType' object is not callable``,
    naming nothing.
    """

    reloaded = _reimport_without(monkeypatch, blocked)

    for name in CATALOG_HELPERS:
        helper = getattr(reloaded, name)
        assert callable(helper), f"{name} must stay callable so the error can name the cause"
        with pytest.raises(ImportError) as excinfo:
            helper()
        message = str(excinfo.value)
        assert blocked in message, f"the error should name {blocked}"
        assert distribution in message, "the error should give an installable package name"
        assert excinfo.value.__cause__ is not None, "the original ImportError should be chained"


def test_unrelated_missing_module_is_not_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing module that is not a catalog dependency must propagate untouched.

    ``vesuvius.utils.catalog`` also imports ``requests``. If that one is absent the
    package must not pretend the catalog extras are the problem - the real
    ModuleNotFoundError has to reach the caller, or genuine breakage inside utils
    would be reported as a missing optional dependency.
    """

    with pytest.raises(ModuleNotFoundError) as excinfo:
        _reimport_without(monkeypatch, "requests")
    assert excinfo.value.name == "requests"
