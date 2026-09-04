import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import plot_fiber_benchmarks as plots


def _point(name, day, score, measured=True):
    return plots.PlotPoint(
        name, name, name, name, date.fromisoformat(day), score, measured
    )


def test_frontier_is_strict_measured_best_so_far():
    points = [
        _point("assumed", "2026-01-01", None, measured=False),
        _point("first", "2026-01-02", 10.0),
        _point("equal", "2026-01-02", 10.0),
        _point("regression", "2026-01-03", 8.0),
        _point("improvement", "2026-01-04", 12.0),
    ]

    assert plots._frontier_indices(points) == [1, 4]


def test_mean_segment_length_counts_failure_delimited_segments():
    assert plots._mean_segment_length_percent(0) == 100.0
    assert plots._mean_segment_length_percent(1) == 50.0
    assert plots._mean_segment_length_percent(4) == 20.0


def test_experiment_ticks_number_every_result_and_label_date_changes():
    points = [
        _point("first", "2026-01-01", 1.0),
        _point("same day", "2026-01-01", 2.0),
        _point("next day", "2026-01-02", 3.0),
    ]

    assert plots._experiment_tick_labels(points) == [
        "1\n2026-01-01",
        "2",
        "3\n2026-01-02",
    ]


def test_long_experiment_axis_uses_spaced_representative_dates():
    points = [
        _point("one", "2026-01-01", 1.0),
        _point("two", "2026-01-02", 2.0),
        _point("three", "2026-01-03", 3.0),
        _point("four", "2026-01-04", 4.0),
        _point("five", "2026-01-04", 5.0),
        _point("six", "2026-01-04", 6.0),
        _point("seven", "2026-01-04", 7.0),
    ]

    assert plots._experiment_tick_labels(points) == [
        "1\n2026-01-01",
        "2",
        "3\n2026-01-03",
        "4",
        "5",
        "6",
        "7\n2026-01-04",
    ]


def test_date_sort_preserves_recorded_order_within_a_day():
    points = [
        _point("same-day first", "2026-01-02", 1.0),
        _point("older", "2026-01-01", 2.0),
        _point("same-day second", "2026-01-02", 3.0),
    ]

    assert [point.algorithm for point in plots._sort_points_by_date(points)] == [
        "older",
        "same-day first",
        "same-day second",
    ]


def test_shared_method_requires_one_base_label():
    first = plots.PlotPoint(
        "first", "shared", "Shared", "Shared", date(2026, 1, 1), 1.0, True
    )
    second = plots.PlotPoint(
        "second",
        "shared",
        "Renamed",
        "Renamed + BP",
        date(2026, 1, 2),
        2.0,
        True,
    )

    try:
        plots._validate_method_labels({"a": [first], "b": [second]})
    except ValueError as error:
        assert "inconsistent method labels" in str(error)
    else:
        raise AssertionError("inconsistent labels for one method were accepted")


def test_render_labels_frontier_and_names_each_other_result_in_legend(tmp_path):
    points = [
        _point("assumed algorithm", "2026-01-01", None, measured=False),
        _point("frontier algorithm", "2026-01-02", 10.0),
        _point("equal algorithm", "2026-01-02", 10.0),
        _point("better algorithm", "2026-01-03", 12.0),
    ]
    output = tmp_path / "plot.svg"

    plots.render_plot(output, "Title", "Score", points, "Description")

    svg = output.read_text(encoding="utf-8")
    assert "<!-- Pareto frontier -->" in svg
    assert "<!-- equal algorithm -->" in svg
    assert "<!-- assumed algorithm (assumed floor) -->" in svg
    assert "<!-- frontier algorithm -->" in svg
    assert "<!-- better algorithm -->" in svg
    assert svg.count("<!-- frontier algorithm -->") == 1
    assert svg.count("<!-- better algorithm -->") == 1
    assert svg.count("<!-- equal algorithm -->") == 1
    assert "rotate(-330)" in svg
