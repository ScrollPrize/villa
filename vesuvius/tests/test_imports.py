from __future__ import annotations

import subprocess
import sys

import pytest

import vesuvius


def test_public_api_exports() -> None:
    """Package root should expose the documented public surface."""

    assert hasattr(vesuvius, "VCDataset")
    assert hasattr(vesuvius, "utils")
    assert hasattr(vesuvius, "models")
    assert not hasattr(vesuvius, "Volume")


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
