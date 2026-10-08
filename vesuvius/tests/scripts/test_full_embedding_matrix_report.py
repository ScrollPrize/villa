"""Small report-control regressions; no external data or GPU required."""
import importlib.util
import json
from pathlib import Path

import pytest


def module():
    path = Path(__file__).resolve().parents[2] / "scripts/report_full_embedding_matrix.py"
    spec = importlib.util.spec_from_file_location("full_embedding_report", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_pending_matrix_fails_closed(tmp_path, monkeypatch):
    m = module()
    monkeypatch.setattr(m, "ROOT", tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({"output_root": str(tmp_path / "pending")}))
    with pytest.raises(ValueError, match="still pending"):
        m.report({"historical": [], "configs": ["config.json"]})


def test_incomplete_marker_rejected(tmp_path):
    m = module()
    (tmp_path / "full").mkdir()
    (tmp_path / "full/incomplete.json").write_text("{}")
    with pytest.raises(ValueError, match="incomplete evaluation"):
        m.read_row("test", tmp_path)


def test_wrong_cohort_size_rejected(tmp_path):
    m = module()
    (tmp_path / "full").mkdir()
    (tmp_path / "full/summary.json").write_text(json.dumps({"completed": True, "cases": 2}))
    with pytest.raises(ValueError, match="100 completed"):
        m.read_row("pilot", tmp_path)
