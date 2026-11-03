"""The engine's state, read off the files: done, stale, waiting, failed, redo."""

import json

import pytest

from volreg.engine import (BLOCKED, DONE, FAILED, READY, STALE, WAITING, Pipeline, Stage,
                           digest_files)


def write(pipeline, name, value):
    path = pipeline.directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return name


def make(tmp_path, fail=False, params=None):
    def draw(pipeline, params):
        return [write(pipeline, "work/drawn.json", {"count": params["count"]})]

    def fit(pipeline, params):
        if fail:
            raise RuntimeError("no points")
        return [write(pipeline, "work/fit.json", {"fit": 1})]

    def score(pipeline, params):
        answers = json.loads((pipeline.directory / "answers.json").read_text())
        return [write(pipeline, "work/score.json", {"answers": len(answers)})]

    def answers_of(pipeline):
        path = pipeline.directory / "answers.json"
        return json.loads(path.read_text()) if path.exists() else None

    def judged(pipeline, params):
        path = pipeline.directory / "answers.json"
        return path.exists() and len(json.loads(path.read_text())) >= params["enough"]

    return Pipeline(tmp_path, [
        Stage("draw", "draw", run=draw, params={"count": 10}),
        Stage("judge", "judge", needs=("draw",), complete=judged, identity=answers_of,
              params={"enough": 2}),
        Stage("fit", "fit", needs=("draw",), run=fit),
        Stage("score", "score", needs=("fit", "judge"), run=score),
    ], params)


def test_fresh_pair(tmp_path):
    assert make(tmp_path).states() == {"draw": READY, "judge": BLOCKED, "fit": BLOCKED,
                                       "score": BLOCKED}


def test_fit_runs_while_a_person_judges(tmp_path):
    pipeline = make(tmp_path)
    pipeline.run_stage("draw")
    assert pipeline.states() == {"draw": DONE, "judge": WAITING, "fit": READY, "score": BLOCKED}
    _states, started = pipeline.advance(pipeline.run_stage)
    assert started == ["fit"]
    assert pipeline.states()["score"] == BLOCKED       # still waiting for the answers


def test_answers_complete_a_human_stage_and_new_ones_make_scoring_stale(tmp_path):
    pipeline = make(tmp_path)
    pipeline.run_stage("draw"); pipeline.run_stage("fit")
    write(pipeline, "answers.json", ["a", "b"])
    assert pipeline.states()["judge"] == DONE and pipeline.states()["score"] == READY
    pipeline.run_stage("score")
    assert pipeline.states()["score"] == DONE
    write(pipeline, "answers.json", ["a", "b", "c"])
    assert pipeline.states()["score"] == STALE


def test_changed_params_make_a_stage_stale(tmp_path):
    pipeline = make(tmp_path)
    pipeline.run_stage("draw")
    more = make(tmp_path, params={"draw": {"count": 20}})
    assert more.states()["draw"] == STALE
    more.run_stage("draw")
    assert json.loads((tmp_path / "work/drawn.json").read_text()) == {"count": 20}
    assert more.states()["fit"] == READY


def test_a_failure_is_stamped_with_its_error(tmp_path):
    pipeline = make(tmp_path, fail=True)
    pipeline.run_stage("draw")
    stamp = pipeline.run_stage("fit")
    assert stamp["status"] == "failed" and "no points" in stamp["error"]
    assert pipeline.states()["fit"] == FAILED
    assert [event["kind"] for event in pipeline.record.events()][-1] == "job-failed"


def test_redo_reruns_a_stage_and_what_depends_on_it(tmp_path):
    pipeline = make(tmp_path)
    write(pipeline, "answers.json", ["a", "b"])
    for name in ("draw", "fit", "score"):
        pipeline.run_stage(name)
    assert set(pipeline.redo("fit")) == {"fit", "score"}
    states = pipeline.states()
    assert states["draw"] == DONE and states["fit"] == READY and states["score"] == BLOCKED
    assert len(list((tmp_path / "work/stages/history").iterdir())) == 2


def test_stamp_names_the_code_and_inputs(tmp_path):
    pipeline = make(tmp_path)
    stamp = pipeline.run_stage("draw")
    assert set(stamp["code"]) == {"commit", "dirty"}
    assert stamp["output_digest"] == digest_files(tmp_path, ["work/drawn.json"])


def test_needs_must_come_first(tmp_path):
    with pytest.raises(ValueError):
        Pipeline(tmp_path, [Stage("b", "b", needs=("a",), run=print), Stage("a", "a", run=print)])
