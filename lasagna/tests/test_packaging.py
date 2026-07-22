from pathlib import Path


LASAGNA_ROOT = Path(__file__).resolve().parents[1]


def test_packaging_declares_requested_console_commands():
    source = (LASAGNA_ROOT / "setup.py").read_text(encoding="utf-8")

    assert "install_requires=all_requires" in source
    assert '*sibling_dependencies("vesuvius", extra="models")' in source
    assert 'excluded = ("volume-cartographer", "torch", "cucim-cu13")' in source
    assert "lasagna-fit-service=fit_service:main" in source
    assert "lasagna-download=lasagna.scripts.download_omezarr:main" in source
    assert "lasagna-download-list=lasagna.scripts.download_volume_list:main" in source
    assert "lasagna-bootstrap=lasagna.scripts.bootstrap_venv:main" in source
    assert "lasagna-preprocess=preprocess_cos_omezarr:cli_main" in source


def test_pep517_build_dependencies_include_pybind11():
    source = (LASAGNA_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert '"pybind11>=2.10"' in source
