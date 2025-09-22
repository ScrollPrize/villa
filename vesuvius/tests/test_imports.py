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
