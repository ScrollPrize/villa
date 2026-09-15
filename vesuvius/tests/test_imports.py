from __future__ import annotations

from pathlib import Path
import asyncio
import importlib.util
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


class _MissingModules:
    """Make the named top-level modules look uninstalled, for import-time tests.

    ``evict`` names already-imported modules that import them; those are dropped for
    the duration (along with the parent-package attribute that ``from pkg import mod``
    would otherwise reuse) so the test sees the real first-import order.
    """

    def __init__(self, *names: str, evict: tuple = ()) -> None:
        self._names = frozenset(names)
        self._evict = tuple(evict)
        self._saved: dict = {}
        self._saved_attrs: list = []

    def find_spec(self, fullname, path=None, target=None):  # noqa: ARG002
        if fullname.partition(".")[0] in self._names:
            raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
        return None

    def __enter__(self) -> "_MissingModules":
        self._saved = {
            name: module
            for name, module in sys.modules.items()
            if name.partition(".")[0] in self._names or name in self._evict
        }
        for name in self._saved:
            del sys.modules[name]
        for name in self._evict:
            parent_name, _, child = name.rpartition(".")
            parent = sys.modules.get(parent_name)
            if parent is not None and child in vars(parent):
                self._saved_attrs.append((parent, child, vars(parent)[child]))
                delattr(parent, child)
        sys.meta_path.insert(0, self)
        return self

    def __exit__(self, *exc_info) -> bool:
        sys.meta_path.remove(self)
        for parent, child, value in self._saved_attrs:
            setattr(parent, child, value)
        self._saved_attrs.clear()
        sys.modules.update(self._saved)
        return False


def _load_catalog_isolated():
    """Execute catalog.py as a fresh module, leaving the imported one untouched."""

    spec = importlib.util.spec_from_file_location(
        "vesuvius_catalog_isolated", catalog.__file__
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_catalog_imports_without_refresh_only_dependencies() -> None:
    """Reading the packaged catalog must not require the catalog-refresh packages.

    Regression test for the failure where ``aiohttp`` and ``nest_asyncio`` -- which
    ship only in the heavier extras -- were imported at module scope. On a core
    install the import raised, the package root silently rebound ``list_files`` to
    ``None``, and calling it failed with ``TypeError: 'NoneType' object is not
    callable``, pointing at the call site rather than the missing package.
    """

    with _MissingModules("aiohttp", "nest_asyncio"):
        isolated = _load_catalog_isolated()
        assert callable(isolated.list_files)
        assert isinstance(isolated.list_files(), dict)


_PARSER_MODULES = ("vesuvius.data.paths.parser", "vesuvius.data.paths.fetcher")


@pytest.mark.parametrize("missing", ["aiohttp", "lxml"])
@pytest.mark.parametrize("entrypoint", ["scrape_website", "collect_subfolders"])
def test_refresh_path_names_the_missing_package(missing: str, entrypoint: str) -> None:
    """Refreshing without a refresh-only package should name it, not fail in the parser.

    Goes through the public entry points rather than the helper, because the parser
    imports ``aiohttp`` and ``lxml`` itself: the check only helps if it runs before
    that import. The check fires before any request, so no network is touched.
    """

    with _MissingModules(missing, evict=_PARSER_MODULES):
        with pytest.raises(ModuleNotFoundError) as excinfo:
            asyncio.run(getattr(catalog, entrypoint)("https://example.invalid/", []))

    message = str(excinfo.value)
    assert missing in message
    assert "list_files" in message


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
