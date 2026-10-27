"""Optional dependency and package-data contracts for ink workflows."""

from __future__ import annotations

from pathlib import Path
import re
import tomllib


_INK_RUNTIME_DISTRIBUTIONS = {
    "accelerate",
    "connected-components-3d",
    "cucim-cu13",
    "diffusers",
    "imagecodecs",
    "opencv-python-headless",
    "tifffile",
}


def _distribution_names(requirements: list[str]) -> set[str]:
    return {
        re.split(r"[<>=!~;\[]", requirement, maxsplit=1)[0].strip().lower()
        for requirement in requirements
    }


def test_ink_runtime_dependencies_are_heavy_extras_and_configured_as_package_data():
    project_root = Path(__file__).parents[2]
    with (project_root / "pyproject.toml").open("rb") as stream:
        pyproject = tomllib.load(stream)

    project = pyproject["project"]
    extras = project["optional-dependencies"]
    model_core_names = _distribution_names(extras["models-core"])
    all_names = _distribution_names(extras["all"])
    core_names = _distribution_names(project["dependencies"])
    volume_names = _distribution_names(extras["volume-only"])

    # Ink workflows must not need the monorepo C++ packages, so they can be
    # installed where those cannot be built (e.g. macOS).
    assert _INK_RUNTIME_DISTRIBUTIONS <= model_core_names
    assert {"volume-cartographer", "vc-object-store"}.isdisjoint(model_core_names)
    # `models` stays the full stack: everything in models-core plus the C++ packages.
    assert set(extras["models"]) == {
        "vesuvius[models-core]", "volume-cartographer", "vc-object-store"}
    assert _INK_RUNTIME_DISTRIBUTIONS <= all_names
    assert _INK_RUNTIME_DISTRIBUTIONS.isdisjoint(core_names)
    assert _INK_RUNTIME_DISTRIBUTIONS.isdisjoint(volume_names)
    assert "ink_detection/configs/*.json" in pyproject["tool"]["setuptools"][
        "package-data"
    ]["vesuvius"]
