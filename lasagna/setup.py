from pathlib import Path
import tomllib

from setuptools import find_packages, setup
from pybind11.setup_helpers import Pybind11Extension, build_ext


ROOT = Path(__file__).resolve().parent


def sibling_dependencies(sibling: str, *, extra: str) -> list[str]:
    """Read a sibling's extra without installing its core distribution."""
    sibling_path = ROOT.parent / sibling
    pyproject = sibling_path / "pyproject.toml"
    if not pyproject.is_file():
        return [f"{sibling}[{extra}]"]
    with pyproject.open("rb") as handle:
        project = tomllib.load(handle)["project"]
    dependencies = list(project.get("dependencies", []))
    dependencies.extend(project.get("optional-dependencies", {}).get(extra, []))
    # Lasagna imports only vesuvius.models. The bootstrap owns the machine-
    # specific torch build; VC and cuCIM are unrelated to this model builder.
    excluded = ("volume-cartographer", "torch", "cucim-cu13")
    return [
        dependency
        for dependency in dependencies
        if not dependency.startswith(excluded)
    ]

# Lasagna predates its packaging metadata and its modules intentionally use
# top-level imports (for example ``import fit_data``).  Install those modules
# as-is so editable and regular installs behave like running from this folder.
py_modules = sorted(
    path.stem
    for path in ROOT.glob("*.py")
    if path.name != "setup.py"
)
packages = find_packages(
    include=("lasagna3d", "lasagna3d.*", "snap_surf", "snap_surf.*", "scripts")
)
# Preserve the programmatic import already used by callers and tests while the
# legacy internal imports continue to use the top-level ``scripts`` package.
packages.append("lasagna.scripts")

fit_service_requires = [
    "numpy>=1.24",
    "opencv-python-headless>=4.8",
    "scikit-image>=0.21",
    "scipy>=1.10",
    "tensorstore>=0.1.60",
    "tifffile>=2023.7.10",
    "torch>=2.1",
    "zarr>=3.0,<4",
]
downloader_requires = ["boto3>=1.28"]
preprocess_requires = [
    "numpy>=1.24",
    "opencv-python-headless>=4.8",
    "torch>=2.1",
    "zarr>=3.0,<4",
]
preprocess_3d_requires = [
    *preprocess_requires,
    *downloader_requires,
    "edt>=3.0",
    "numba>=0.58",
    "tensorstore>=0.1.60",
    "tensorboard>=2.14",
    "tifffile>=2023.7.10",
    *sibling_dependencies("vesuvius", extra="models"),
]
all_requires = sorted(set(
    fit_service_requires + downloader_requires + preprocess_3d_requires
))

ext_modules = [
    Pybind11Extension(
        "monotone_norm",
        ["monotone_norm.cpp"],
    ),
]

setup(
    name="vesuvius-lasagna",
    version="0.0.1",
    description="Surface fitting and OME-Zarr preprocessing tools",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    python_requires=">=3.14,<3.15",
    py_modules=py_modules,
    packages=packages,
    package_dir={"lasagna.scripts": "scripts"},
    install_requires=all_requires,
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    extras_require={
        "test": ["pytest>=7"],
    },
    entry_points={
        "console_scripts": [
            "lasagna-fit-service=fit_service:main",
            "lasagna-download=lasagna.scripts.download_omezarr:main",
            "lasagna-download-list=lasagna.scripts.download_volume_list:main",
            "lasagna-bootstrap=lasagna.scripts.bootstrap_venv:main",
            "lasagna-preprocess=preprocess_cos_omezarr:cli_main",
        ],
    },
    zip_safe=False,
)
