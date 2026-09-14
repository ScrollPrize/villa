import copy
import json

import pytest

from vesuvius.neural_tracing.fiber_trace_3d.visualization_report import fiber_rows, write_visualization_report


def summary():
    cases = [{"source_index": 0, "direction": direction,
              "reference_length_base_voxels": 10, "evaluated_through_base_voxels": 10,
              "evaluation_complete": True,
              "failures": [{"directional_reference_arc_base_voxels": x} for x in failures]}
             for direction, failures in (("forward", [4]), ("reverse", [2, 8]))]
    return {"completed": True, "cases": 2, "failures": 3, "directed_reference_length_mm": 20,
            "sources": [{"source_id": "collection", "summary": {"base_voxel_size_um": 1000},
                         "backend_config": {"coordinate_space": "scroll/scan@L0", "fibers": ["fiber.json"]},
                         "cases": cases}]}


def test_terminal_segments_and_both_directions():
    row, = fiber_rows(summary())
    assert row["restarts"] == 2
    assert row["mean_distance_mm"] == pytest.approx(20 / 3)
    assert row["directions"]["forward"]["mean_distance_mm"] == 5
    assert row["directions"]["reverse"]["mean_distance_mm"] == 5
    bad = copy.deepcopy(summary())
    bad["sources"][0]["cases"][1]["direction"] = "forward"
    with pytest.raises(ValueError, match="duplicate"):
        fiber_rows(bad)


def test_multiple_eligible_runs_do_not_turn_jumps_into_failures():
    s = summary()
    extra = copy.deepcopy(s["sources"][0]["cases"])
    for c in extra:
        c["run_index"] = 1
    s["sources"][0]["cases"] += extra
    s.update(cases=4, failures=6, directed_reference_length_mm=40)
    binding = s["sources"][0]["backend_config"]
    binding["fibers"].append("excluded.json")
    binding["reference_selection"] = [
        {"runs": [{}, {}], "original_length_base_voxels": 30},
        {"runs": [], "original_length_base_voxels": 5}]
    row, excluded = fiber_rows(s)
    assert row["eligible_runs"] == 2 and row["restarts"] == 4
    assert row["mean_distance_mm"] == 8  # 40/(4 episodes+1)
    assert row["directions"]["forward"]["mean_distance_mm"] == pytest.approx(20 / 3)
    assert row["excluded_length_mm"] == 10
    assert excluded["status"] == "excluded" and excluded["mean_distance_mm"] is None
    assert excluded["restarts"] == 0


def test_saved_trace_metrics_available_without_rendering_or_summary(tmp_path):
    from vesuvius.neural_tracing.fiber_trace_3d.visualization_report import write_trace_progress_report
    for index, arcs in enumerate(([4], [2, 8])):
        case = tmp_path / "full/cases" / str(index)
        case.mkdir(parents=True)
        (case / "trace.json").write_text(json.dumps({"provenance": {"source_id": "source",
            "fiber_path": "fiber.json", "reverse": bool(index), "case_index": index, "base_voxel_size_um": 1000},
            "replay": {"reference_begin_arc_base": 0, "reference_end_arc_base": 10,
                       "completed_reference_arc_base": 10, "failures": [{"reference_arc_base": a} for a in arcs]}}))
    report = write_trace_progress_report(tmp_path / "full", tmp_path)
    assert report["paired_fibers"] == 1
    assert report["fibers"][0]["mean_mm"] == pytest.approx(20 / 3)
    assert report["fibers"][0]["restarts"] == 2
    assert (tmp_path / "trace_progress.csv").exists()


@pytest.mark.parametrize("combined", [False, True])
def test_report_links_both_modalities_and_audits_missing_pages(tmp_path, combined):
    root = tmp_path / "full"
    root.mkdir()
    (root / "summary.json").write_text(json.dumps(summary()))
    for i, failures in enumerate((1, 2)):
        case = root / "cases" / str(i)
        case.mkdir(parents=True)
        (case / "trace.json").write_text(json.dumps({"provenance": {
            "source_id": "collection", "fiber_path": "fiber.json", "reverse": bool(i), "case_index": i},
            "replay": {"reference_begin_arc_base": 0, "reference_end_arc_base": 10,
                "completed_reference_arc_base": 10,
                "failures": [{"reference_arc_base": a} for a in ([4] if i == 0 else [2, 8])]}}))
        for name in (("rendered",) if combined else ("rendered", "rendered_presence")):
            folder = case / name
            folder.mkdir()
            (folder / "render.json").write_text(json.dumps({"pages": 1, "format": "vc_fiber_inspection" if combined else "legacy"}))
            (folder / "strips_0.jpg").write_bytes(b"test")
    report = write_visualization_report(root, tmp_path)
    key, directory = ("combined", "rendered") if combined else ("presence", "rendered_presence")
    assert report["fibers"][0]["directions"]["reverse"][key] == [f"full/cases/1/{directory}/strips_0.jpg"]
    assert "20/" not in (tmp_path / "README.md").read_text()
    assert "mean_distance_mm" in (tmp_path / "per_fiber.csv").read_text()
    (root / f"cases/1/{directory}/strips_0.jpg").unlink()
    with pytest.raises(ValueError, match="missing rendered"):
        write_visualization_report(root, tmp_path)
