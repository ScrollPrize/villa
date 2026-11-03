"""Offline geometry gallery: gaps, escaping, units and atomic output."""
import copy
import json

import pytest

from vc import fiber_evaluation_vis as vis
from vc import fiber_evaluation as evaluation
from test_fiber_chunk_trace import inputs
from test_fiber_evaluation import config


def test_gallery_uses_saved_geometry_and_preserves_metrics(inputs, tmp_path):
    report = evaluation.evaluate(config(inputs), base_directory=inputs)
    before = copy.deepcopy(report)
    assert len(report["references"]) == 2
    html = vis.render_html(report)
    assert html.count('<svg ') == 6
    assert html.count('class="trace"') == 3 * sum(len(c["segments"]) for c in report["cases"])
    assert "base voxels" in html and "Filtered stops are not failures" in html
    assert report == before
    path, output = tmp_path / "report.json", tmp_path / "gallery.html"
    path.write_text(json.dumps(report))
    assert vis.main([str(path), str(output)]) == 0
    assert output.read_text() == html
    assert not list(tmp_path.glob("gallery.html.*.tmp"))


def test_escaping_restart_gaps_and_failure_locations(inputs):
    report = evaluation.evaluate(config(inputs), base_directory=inputs)
    case = report["cases"][0]
    case["id"] = report["references"][0]["id"] = '<script>alert("x")</script>'
    case["segments"] = [dict(points_base_xyz=[[0, 0, 0], [1, 0, 0]]),
                        dict(points_base_xyz=[[3, 0, 0], [4, 0, 0]])]
    case["failures"] = [dict(index=0, reason="<unsafe>", reference_arc_base=2,
                             reference_point_base=[2, 0, 0], threshold_measurement=None)]
    html = vis.render_html(report)
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "&lt;unsafe&gt;" in html
    assert html.count('<circle class="failure"') == 3
    assert html.count('class="trace"') == 3 * sum(len(c["segments"]) for c in report["cases"])


def test_invalid_report_keeps_previous_gallery(inputs, tmp_path, monkeypatch):
    report = evaluation.evaluate(config(inputs), base_directory=inputs)
    output = tmp_path / "gallery.html"
    output.write_text("previous")
    report["references"][0]["points_base_xyz"][0][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        vis.write_gallery(report, output)
    assert output.read_text() == "previous"
    def fail(*args):
        raise OSError("replace failed")
    monkeypatch.setattr(vis.os, "replace", fail)
    with pytest.raises(OSError, match="replace failed"):
        vis.atomic_write_text(output, "next")
    assert output.read_text() == "previous"
    assert not list(tmp_path.glob("gallery.html.*.tmp"))


def test_empty_and_missing_reference_reports(inputs):
    report = evaluation.evaluate(config(inputs), base_directory=inputs)
    del report["references"]
    with pytest.raises(ValueError, match="missing saved reference"):
        vis.render_html(report)
    report.update(cases=[], references=[])
    assert "0 directed cases" in vis.render_html(report)


def test_cli_rejects_same_json_and_html_before_evaluation(tmp_path):
    with pytest.raises(SystemExit):
        evaluation.main([str(tmp_path / "absent.json"), str(tmp_path / "output"),
                         "--html", str(tmp_path / "output")])
