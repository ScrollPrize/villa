"""A release takes each pair's newest decided record, copies its shared files, and says
what each record does not hold."""

import json

import pytest

from volreg import record as rec, release

BY = {"handle": "j", "email": "j@x"}


def make(root, name, pair, at, pipeline, publish="bend", decided=True):
    record = rec.Record(root / name)
    record.write(rec.PAIR, {"pair": pair, "coarse_side": "fixed"})
    record.write(rec.HOLDOUT, {"patches": [
        {"id": "p0001", "answer": {"answer": "lines-up", "at": at, "by": BY},
         "octants": [{"index": 2, "answer": {"answer": "no", "at": at, "by": BY}}]}]})
    decision = {"fit": "f002", "publish": publish, "reason": "", "override": None, "by": BY, "at": at}
    record.write(rec.FINAL_REVIEW, {"rounds": [{"id": "c001", "items": [{"patch": "p0001", "choice": 0}]}],
                                    "decision": decision if decided else None,
                                    "revisit": [] if decided else [{"text": "deferred", "done": None}]})
    score = {"lines_up": 25, "median_fine_voxels": 1.0, "p90_fine_voxels": 2.0, "far": 0,
             "median_um": 1.0, "p90_um": 2.0, "bar": {"passes": True}}
    fit = {"scores": {"bend": score, "affine": score}, "bar": {"version": "test"}}
    if pipeline:
        fit |= {"code": {"commit": "abc", "dirty": False}, "drift": {}}
        (record.directory / "work").mkdir()
    else:
        fit |= {"code": {"migrated_from_session_fit": "f002"}}
    record.write(rec.FIT, fit)
    record.write(rec.TRANSFORM, {"schema_version": "1.0.0"})
    record.log("holdout-answer", unit="p0001", answer="lines-up", by=BY)
    return record


def test_the_newest_decision_of_each_pair_is_released_whole(tmp_path):
    old, new = tmp_path / "record-v1", tmp_path / "pipeline-smoke"
    make(old, "a-auto", "A", None, pipeline=False)
    make(old, "b-auto", "B", None, pipeline=False)
    make(new, "a", "A", "2026-11-05T12:00:00+00:00", pipeline=True)
    make(new, "c", "C", "2026-11-05T12:00:00+00:00", pipeline=True, publish="nothing")
    index = release.release(tmp_path / "out", [old, new])
    pairs = {pair["pair"]: pair for pair in index["pairs"]}
    assert set(pairs) == {"A", "B"}
    assert pairs["A"]["source"] == "pipeline-smoke/a" and pairs["A"]["not_recorded"] == [
        "image hashes of blind items"]
    assert pairs["B"]["not_recorded"] == ["code commit", "decision time",
                                          "hold-out drift under the fit", "image hashes of blind items"]
    assert pairs["A"]["answers"] == {"lines-up": 1, "no": 1}
    assert {item["source"]: item["why"] for item in index["left_out"]} == {
        "record-v1/a-auto": "superseded by pipeline-smoke/a",
        "pipeline-smoke/c": "decided to publish nothing"}
    released = tmp_path / "out" / "A"
    assert sorted(path.relative_to(released).as_posix() for path in released.rglob("*.json")) == [
        "final-review.json", "fit/fit.json", "fit/transform.json", "holdout.json", "pair.json"]
    held = json.loads((released / "holdout.json").read_text())
    assert held["patches"][0]["answer"]["by"] == BY
    assert pairs["A"]["files"]["holdout.json"] == release.sha256(released / "holdout.json")
    assert json.loads((tmp_path / "out" / "index.json").read_text())["schema"] == release.SCHEMA
    with pytest.raises(FileExistsError):
        release.release(tmp_path / "out", [old, new])


def test_two_undated_decisions_of_one_pair_are_refused(tmp_path):
    old = tmp_path / "record-v1"
    make(old, "a-1", "A", None, pipeline=False)
    make(old, "a-2", "A", None, pipeline=False)
    with pytest.raises(ValueError, match="cannot tell"):
        release.release(tmp_path / "out", [old])


def test_a_pair_under_review_again_is_held_back_whole(tmp_path):
    old, new = tmp_path / "record-v1", tmp_path / "pipeline-smoke"
    make(old, "a-auto", "A", None, pipeline=False)
    make(new, "a", "A", None, pipeline=True, decided=False)
    index = release.release(tmp_path / "out", [old, new])
    assert index["pairs"] == []
    assert {item["source"]: item["why"] for item in index["left_out"]}["record-v1/a-auto"] == \
        "held back: pipeline-smoke/a is not decided (deferred)"
