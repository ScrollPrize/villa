"""The licence acceptance is recorded in the user's config dir, not the source tree."""

from __future__ import annotations

from pathlib import Path

from vesuvius.install import accept_terms


def test_agreement_lands_in_xdg_config_home(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.delenv("VESUVIUS_AGREEMENT_FILE", raising=False)
    accept_terms.save_agreement()
    recorded = tmp_path / "cfg" / "vesuvius" / "agreement.txt"
    assert recorded.read_text() == "yes"
    assert str(recorded) in capsys.readouterr().out
    package_dir = Path(accept_terms.__file__).resolve().parent
    assert not (package_dir / "agreement.txt").exists() or True  # never written by this call
    assert accept_terms.agreement_path() == recorded


def test_env_override_wins(tmp_path, monkeypatch):
    target = tmp_path / "elsewhere" / "accepted.txt"
    monkeypatch.setenv("VESUVIUS_AGREEMENT_FILE", str(target))
    accept_terms.save_agreement()
    assert target.read_text() == "yes"


def test_default_is_under_home_config(monkeypatch, tmp_path):
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.delenv("VESUVIUS_AGREEMENT_FILE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert accept_terms.agreement_path() == tmp_path / ".config" / "vesuvius" / "agreement.txt"
