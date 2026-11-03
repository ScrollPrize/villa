"""The record's files, scoring against a hold-out set, and the bar."""

import json

import numpy as np
import pytest

from volreg import record as rec
from volreg.scoring import BAR, bar_check, coarse_to_fine, score


def transform(matrix_xyz):
    return {"transformation_matrix": np.asarray(matrix_xyz, float)[:3].tolist()}


def holdout(units):
    return {"patches": [{"id": f"p{index:04d}", "centre_zyx": centre, "fine_zyx": fine,
                         "answer": {"answer": answer, "at": "t", "by": {}}}
                        for index, (centre, fine, answer) in enumerate(units)]}


def test_write_is_whole_and_stamped(tmp_path):
    record = rec.Record(tmp_path / "pair")
    record.write(rec.HOLDOUT, {"patches": []})
    assert record.read(rec.HOLDOUT) == {"schema": rec.SCHEMA, "patches": []}
    assert not list((tmp_path / "pair").glob("*.tmp"))


def test_transform_is_written_as_is(tmp_path):
    record = rec.Record(tmp_path)
    record.write(rec.TRANSFORM, {"transformation_matrix": [[1, 0, 0, 0]] * 3})
    assert "schema" not in record.read(rec.TRANSFORM)


def test_events_only_append(tmp_path):
    record = rec.Record(tmp_path)
    record.log("holdout-answer", unit="p0001", answer="lines-up", at="2026-11-03T00:00:00+00:00")
    record.log("blind-pick", item=0, choice=1)
    events = record.events()
    assert [event["kind"] for event in events] == ["holdout-answer", "blind-pick"]
    assert events[0]["at"] == "2026-11-03T00:00:00+00:00"
    with open(record.path(rec.EVENTS)) as handle:
        assert len(handle.readlines()) == 2


def test_reviewer_needs_handle_and_email():
    with pytest.raises(ValueError):
        rec.Reviewer("johannes", "")
    assert rec.Reviewer("j", "j@x").as_json() == {"handle": "j", "email": "j@x"}


@pytest.mark.parametrize("fixed_is_coarse", [True, False])
def test_affine_maps_coarse_to_fine_either_way(fixed_is_coarse):
    # moving -> fixed scales by 4 in XYZ and shifts x by 8; coarse -> fine is its inverse
    # when the fixed volume is the coarse one, and the matrix itself otherwise.
    matrix = np.diag([4.0, 4.0, 4.0, 1.0]); matrix[0, 3] = 8.0
    mapped = coarse_to_fine(transform(matrix), fixed_is_coarse)(np.array([[1.0, 2.0, 3.0]]))
    expected = [[0.25, 0.5, (3 - 8) / 4]] if fixed_is_coarse else [[4.0, 8.0, 20.0]]
    assert np.allclose(mapped, expected)


def test_only_lining_up_scores_and_distances_are_fine_voxels():
    units = [([0, 0, 0], [0, 0, 1], "lines-up"), ([0, 0, 0], [0, 3, 4], "lines-up"),
             ([0, 0, 0], [50, 0, 0], "partly")]
    result = score(transform(np.eye(4)), holdout(units), False, fine_um=2.0)
    assert result["lines_up"] == 2
    assert result["median_fine_voxels"] == pytest.approx(3.0)
    assert result["median_um"] == pytest.approx(6.0)
    assert result["worst"][0] == ["p0001", 5.0]


def test_octants_answered_alone_are_units():
    held = holdout([([0, 0, 0], [0, 0, 0], "partly")])
    held["patches"][0]["octants"] = [{"index": 3, "centre_zyx": [1, 1, 1], "fine_zyx": [1, 1, 3],
                                      "answer": {"answer": "lines-up"}}]
    result = score(transform(np.eye(4)), held, False, fine_um=1.0)
    assert result["lines_up"] == 1 and "p0000.3" in result["per_unit"]


def test_bar():
    good = {"lines_up": 40, "median_fine_voxels": 1.5, "p90_fine_voxels": 3.0, "far": 2}
    assert bar_check(good)["passes"]
    assert not bar_check(good | {"far": 3})["checks"]["far"]          # 3 > 40 / 20
    assert not bar_check(good | {"lines_up": 19})["checks"]["enough"]
    assert not bar_check(good | {"median_fine_voxels": 2.01})["passes"]
    assert bar_check(good)["bar"] == BAR["version"]
