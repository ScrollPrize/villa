"""The licence-acceptance command is gone; the installation-path helper it hosted stays."""
import tomllib
from pathlib import Path

from vesuvius.install import accept_terms

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def test_console_script_and_install_hook_removed():
    cfg = tomllib.loads(PYPROJECT.read_text())
    assert "vesuvius.accept_terms" not in cfg["project"]["scripts"]
    assert "cmdclass" not in cfg.get("tool", {}).get("setuptools", {})


def test_no_acceptance_api_left():
    for name in ("main", "save_agreement", "display_terms_and_conditions"):
        assert not hasattr(accept_terms, name)


def test_installation_path_still_available():
    assert Path(accept_terms.get_installation_path()).is_dir()
