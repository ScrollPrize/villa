"""Round 7 of the review: a saved decision from before decisions named their scores (a
legacy approval) taken for a score finished in the same second, or for one made after it
when the same stale page was sent twice; an answer that reached the event log but not
`holdout.json`; an answer changed and changed back while Score ran; and a job that failed
before it began, written after a redo of its stage.

Each test opens with the requirement it checks, quoted from the review. The clock is the
record's own (`record.now`, whole seconds), set by the test where the second matters. A
failed rewrite of `holdout.json` or `final-review.json` is made by refusing the replace, as
Windows refuses it while another process has the file open; wintest/r7_native.py does the
same with a real handle."""

import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from volreg import cli, engine, filelock
from volreg import record as rec
from volreg import stages as pipeline_stages
from volreg.engine import DONE, FAILED, READY, STALE

from test_approval_coherence import (BY, answer, assert_published_coherently, check, decide_on, decision,
                                     drain, page, scored, scored_pair, stored)  # noqa: F401 - a fixture
from test_redo_release import fake_stages, pick_all, released  # noqa: F401 - a fixture
from test_review_blockers import as_windows_locks
from test_sharing_conflicts import refuse_opens_of, scored_up_to_score

LEGACY = ("The upgrade path must establish which score the saved approval covers, while "
          "preserving coherent legacy records.")
LOGGED = ("Release must hold back the pair and its older candidates when a logged answer "
          "invalidates the score. Exported releases must remain checkable from their shipped files.")
UNDO = ("The successful stamp must describe the answers Score used so an inconsistent result can "
        "be rescored.")
REDO = "A failure from work invalidated by Redo must leave the stage eligible to run again."


class Clock:
    """The record's clock, in whole seconds as it is, standing still until `tick`."""

    def __init__(self):
        self.at = datetime(2026, 11, 24, 5, 0, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.at.isoformat(timespec="seconds")

    def tick(self):
        self.at += timedelta(seconds=1)


def legacy_decision(directory, publish, missed=None, reason="", round_=True):
    """A decision as the servers before this one wrote it: no basis; the round it was made
    on if `round_` (round 5 named it, main did not), and an override naming `missed`."""
    record = rec.Record(directory)
    made = {"fit": record.read(rec.FINAL_REVIEW)["rounds"][-1]["candidates"][0], "publish": publish,
            "reason": reason, "override": {"by": BY, "reason": reason, "bar_missed": missed} if missed else None}
    if round_:
        made["round"] = record.read(rec.FINAL_REVIEW)["rounds"][-1]["id"]
    return made


def refuse_replace_of(monkeypatch, name):
    """Replacing record file `name` is refused, as Windows refuses it past the wait while a
    reader has the file open. Undone by the returned function."""
    real = filelock.replace

    def refusing(source, target):
        if Path(target).name == Path(name).name:
            raise PermissionError(13, "The process cannot access the file because it is being used "
                                      "by another process", str(target))
        return real(source, target)
    monkeypatch.setattr(filelock, "replace", refusing)
    return lambda: monkeypatch.setattr(filelock, "replace", real)


def scored_on_nineteen(root, clock=None):
    """A pair scored on twenty, then on nineteen (p0001 answered "partly"), its round answered."""
    directory = scored_pair(root)
    answer(directory, "p0001", "partly")
    drain(directory)
    pick_all(directory)
    assert page(directory)["scores"]["scores"]["bend"]["lines_up"] == 19
    return directory


# ============================================================ 1. which score a legacy approval covers

def test_a_legacy_approval_does_not_cover_a_rescore_finished_in_the_same_second(tmp_path, scored, monkeypatch, capsys):
    """The upgrade path must establish which score the saved approval covers, while
    preserving coherent legacy records. (The review's first sequence: a legacy approval of a
    score on 19, then a rescore on 18 finished in the same second. Both miss only the count,
    so the override names the same failure; the clock cannot tell them apart.)"""
    monkeypatch.setattr(rec, "now", Clock())
    root = tmp_path / "records"
    directory = scored_on_nineteen(root)
    rec.Record(directory).decide(legacy_decision(directory, "bend", ["enough"], "19 will do"), BY)
    answer(directory, "p0002", "partly")                         # 18, scored in the same second
    pipeline = drain(directory)
    assert page(directory)["scores"]["scores"]["bend"]["lines_up"] == 18
    assert pipeline.states()["store"] == FAILED and "decide again" in pipeline.stamp("store")["error"]
    assert released(tmp_path, [root], "same-second")[0] == {}
    # Not stuck: a decision on what the page shows now is stored and released.
    decide_on(directory, "bend", page(directory), "18 will do")
    assert drain(directory).states()["store"] == DONE
    assert "A" in released(tmp_path, [root], "decided")[0]
    assert_published_coherently(tmp_path / "out-decided")
    assert check(root, capsys)[0] == 0


def test_two_identical_legacy_approvals_do_not_cover_the_score_between_them(tmp_path, scored, monkeypatch, capsys):
    """The upgrade path must establish which score the saved approval covers, while
    preserving coherent legacy records. (The review's second sequence: the same stale page
    on 19 sent twice, the second time after a rescore on 18, and that decision's rewrite of
    final-review.json failed. The two logged decisions are the same; once the view is
    rewritten from the log, the saved one is the later, logged after the score on 18.)"""
    clock = Clock()
    monkeypatch.setattr(rec, "now", clock)
    root = tmp_path / "records"
    directory = scored_on_nineteen(root)
    legacy = legacy_decision(directory, "bend", ["enough"], "19 will do")
    clock.tick()
    rec.Record(directory).decide(legacy, BY)
    clock.tick()
    answer(directory, "p0002", "partly")
    pipeline = drain(directory)                                  # 18; the decision is older
    assert pipeline.states()["store"] == FAILED
    clock.tick()
    restore = refuse_replace_of(monkeypatch, rec.FINAL_REVIEW)
    with pytest.raises(PermissionError):
        rec.Record(directory).decide(legacy, BY)                 # logged; the view not rewritten
    restore()
    logged = [event for event in rec.Record(directory).events() if event["kind"] == "decision"]
    assert len(logged) == 2 and {key: logged[0][key] for key in legacy} == {key: logged[1][key] for key in legacy}
    rec.Record(directory).refresh_final()                        # the view, rewritten from the log
    assert decision(directory)["at"] == logged[1]["at"]
    pipeline_stages.pipeline(directory).redo("store")            # the upgrade runs Store again
    pipeline = drain(directory)
    assert pipeline.states()["store"] == FAILED and "decide again" in pipeline.stamp("store")["error"]
    assert released(tmp_path, [root], "twice")[0] == {}
    assert check(root, capsys)[0] == 0                           # nothing stored, nothing to check


def test_a_record_stored_under_a_legacy_approval_of_other_scores_is_held_back(tmp_path, scored, monkeypatch, capsys):
    """The upgrade path must establish which score the saved approval covers, while
    preserving coherent legacy records. (A record the submitted head stored that way, 18
    under the approval for 19: Release holds it back, with the pair's older record, and
    `volreg check` fails it, from the record's own event log.)"""
    from test_release import make as make_record
    monkeypatch.setattr(rec, "now", Clock())
    old = tmp_path / "first-runs"
    make_record(old, "a-auto", "A", None, pipeline=False)
    root = tmp_path / "records"
    directory = scored_on_nineteen(root)
    rec.Record(directory).decide(legacy_decision(directory, "bend", ["enough"], "19 will do"), BY)
    answer(directory, "p0002", "partly")
    real = pipeline_stages.approval_problem
    monkeypatch.setattr(pipeline_stages, "approval_problem", lambda *args, **kwargs: None)  # as it did
    assert drain(directory).states()["store"] == DONE
    monkeypatch.setattr(pipeline_stages, "approval_problem", real)
    assert rec.Record(directory).read(rec.FIT)["scores"]["bend"]["lines_up"] == 18
    pairs, index = released(tmp_path, [old, root], "stored")
    assert pairs == {}
    why = {item["source"]: item["why"] for item in index["left_out"]}
    assert "names no scores" in why["records/a"] and "held back" in why["first-runs/a-auto"]
    code, out = check(root, capsys)
    assert code == 1 and "NOT WHAT IS STORED" in out


def test_a_legacy_approval_with_nothing_changed_since_its_round_stays_eligible(tmp_path, scored, monkeypatch, capsys):
    """The upgrade path must establish which score the saved approval covers, while
    preserving coherent legacy records. (Twenty lining up, decided by round 5's page, and by
    main's, which named no round; nothing changed after the round was drawn. Same second.)"""
    monkeypatch.setattr(rec, "now", Clock())
    for name, pair, round_ in (("a", "A", True), ("b", "B", False)):
        root = tmp_path / "records"
        directory = scored_pair(root, name, pair)
        pick_all(directory)
        rec.Record(directory).decide(legacy_decision(directory, "bend", round_=round_), BY)
        assert drain(directory).states()["store"] == DONE
    assert set(released(tmp_path, [root], "legacy")[0]) == {"A", "B"}
    assert_published_coherently(tmp_path / "out-legacy")
    assert check(root, capsys)[0] == 0


def test_a_legacy_approval_of_a_record_with_no_unit_lining_up_stays_eligible(tmp_path, scored, monkeypatch, capsys):
    """The upgrade path must establish which score the saved approval covers, while
    preserving coherent legacy records. (No unit lines up; the affine published on an
    override naming the count.)"""
    monkeypatch.setattr(rec, "now", Clock())
    root = tmp_path / "records"
    directory = scored_pair(root, zero=True)
    pick_all(directory)
    rec.Record(directory).decide(legacy_decision(directory, "affine", ["enough"], "nothing lines up yet"), BY)
    assert drain(directory).states()["store"] == DONE
    assert "A" in released(tmp_path, [root], "zero")[0]
    assert check(root, capsys)[0] == 0


def nineteen_when_drawn(root, name="a", pair="A"):
    """A pair whose blind round is drawn on nineteen: p0001 answered "partly" before
    anything ran (logged, as `volreg answer` logs it), so every score is on nineteen."""
    from test_redo_release import new_pair
    directory = new_pair(root, name, pair)
    rec.Record(directory).log("holdout-answer", unit="p0001", answer="partly", by=BY)
    drain(directory)
    pick_all(directory)
    assert page(directory)["scores"]["scores"]["bend"]["lines_up"] == 19
    return directory


def test_a_legacy_approval_of_the_score_its_round_was_drawn_on_stays_eligible(tmp_path, scored, monkeypatch, capsys):
    """The upgrade path must establish which score the saved approval covers, while
    preserving coherent legacy records. (The control for the review's first sequence: the
    approval for 19, on a round drawn on 19, and no rescore after it. Same second.)"""
    monkeypatch.setattr(rec, "now", Clock())
    root = tmp_path / "records"
    directory = nineteen_when_drawn(root)
    rec.Record(directory).decide(legacy_decision(directory, "bend", ["enough"], "19 will do"), BY)
    assert drain(directory).states()["store"] == DONE
    assert "A" in released(tmp_path, [root], "control")[0]
    assert_published_coherently(tmp_path / "out-control")
    assert check(root, capsys)[0] == 0


def test_the_same_approval_and_a_rescore_in_its_second_is_refused_for_the_rescore(tmp_path, scored, monkeypatch, capsys):
    """The upgrade path must establish which score the saved approval covers, while
    preserving coherent legacy records. (The review's first sequence, on a round drawn on
    19: only the answer that takes 19 to 18, in the approval's second, separates it from the
    control above, and that answer is what Store names.)"""
    monkeypatch.setattr(rec, "now", Clock())
    root = tmp_path / "records"
    directory = nineteen_when_drawn(root)
    rec.Record(directory).decide(legacy_decision(directory, "bend", ["enough"], "19 will do"), BY)
    answer(directory, "p0002", "partly")
    pipeline = drain(directory)
    assert page(directory)["scores"]["scores"]["bend"]["lines_up"] == 18
    error = pipeline.stamp("store")["error"]
    assert pipeline.states()["store"] == FAILED and "decide again" in error and "p0002" in error
    assert released(tmp_path, [root], "rescored")[0] == {}


def test_a_legacy_approval_after_a_rescore_on_a_round_drawn_before_it_is_held_back(tmp_path, scored, monkeypatch, capsys):
    """The upgrade path must establish which score the saved approval covers, while
    preserving coherent legacy records. (What this costs: blind keeps one round per fit, so
    a round drawn on 20 is still the round after the rescore on 19. An approval for 19 sent
    then may have come from a page loaded before the rescore -- the review's second
    sequence -- and the log cannot tell, so it is held back, and one decision on the page as
    it is stores it.)"""
    monkeypatch.setattr(rec, "now", Clock())
    root = tmp_path / "records"
    directory = scored_on_nineteen(root)
    rec.Record(directory).decide(legacy_decision(directory, "bend", ["enough"], "19 will do"), BY)
    pipeline = drain(directory)
    error = pipeline.stamp("store")["error"]
    assert pipeline.states()["store"] == FAILED and "decide again" in error and "p0001" in error
    decide_on(directory, "bend", page(directory), "19 will do")
    assert drain(directory).states()["store"] == DONE
    assert check(root, capsys)[0] == 0


# The event log's own sequences, for `legacy_approval_problem` alone (the fake blind stage
# logs no `blind-round`; the real one does, with the round's id).
def blind(round_id=None):
    started = [{"kind": "job-started", "stage": "blind"}]
    drawn = [{"kind": "blind-round", "round": round_id}] if round_id else []
    return started + drawn + [{"kind": "job-finished", "stage": "blind"}]


def job(stage):
    return [{"kind": "job-started", "stage": stage}, {"kind": "job-finished", "stage": stage}]


def said(unit, value):
    return [{"kind": "holdout-answer", "unit": unit, "answer": value}]


def decided(round_id="c001", fit="f001"):
    return [{"kind": "decision", "fit": fit, "publish": "bend", "override": None, "round": round_id}]


START = job("holdout") + said("p0001", "lines-up") + said("p0002", "lines-up") + job("fit")


@pytest.mark.parametrize("events, round_id, problem", [
    (START + job("score") + blind("c001") + decided(), "c001", None),
    (START + job("score") + blind() + decided(None), None, None),                     # main named no round
    (START + job("score") + blind("c001") + said("p0002", "no") + job("score") + decided(), "c001", "p0002 lines up"),
    (START + job("score") + blind("c001") + said("p0002", "no") + said("p0002", "lines-up") + decided(), "c001",
     "p0002 lines up"),                                                               # changed and back
    (START + said("p0002", "partly") + job("score") + blind("c001") + said("p0002", "no") + decided(), "c001",
     None),                                                                           # partly to no: not lining up either way
    (START + job("score") + blind("c001") + job("holdout") + job("score") + decided(), "c001", "drawn again"),
    (START + job("score") + decided() + blind("c001"), "c001", "does not show the blind round"),
    (START + blind("c001") + decided(), "c001", "does not show the scores"),
    (START + job("score") + blind("c001") + decided() + decided("c001", fit="f002"), "c001", "not the last decision"),
    (START + job("score") + blind("c001") + job("score") + blind("c002") + decided("c002"), "c002", None),
    (said("p0001", "lines-up"), "c001", "holds no decision"),
])
def test_which_scores_the_event_log_shows_a_legacy_approval_was_made_on(events, round_id, problem):
    """The upgrade path must establish which score the saved approval covers, while
    preserving coherent legacy records. (The rule on hand-written logs: the round named,
    or main's unnamed one; an answer that changes whether a unit lines up after the round's
    score began, even changed back; one that does not; the hold-out set drawn again; a
    decision logged before its round, or a round with no score before it; a decision that
    is not the last one logged; a log with no decision.)"""
    from volreg.scoring import legacy_approval_problem
    found = legacy_approval_problem(events, decided(round_id)[0])
    if problem is None:
        assert found is None
    else:
        assert found is not None and problem in found


# ======================================================= 2. the event log is the answers' source

def test_an_answer_logged_whose_rewrite_of_holdout_json_failed_holds_back_the_pair(tmp_path, scored, monkeypatch, capsys):
    """Release must hold back the pair and its older candidates when a logged answer
    invalidates the score. Exported releases must remain checkable from their shipped
    files. (`volreg answer` logged "no", and its replace of holdout.json was refused: Score
    and Store showed done, and Release shipped the old result.)"""
    from test_release import make as make_record
    old = tmp_path / "first-runs"
    make_record(old, "a-auto", "A", None, pipeline=False)
    root = tmp_path / "records"
    directory = stored(root)
    assert released(tmp_path, [old, root], "before")[0]["A"]["source"] == "records/a"
    restore = refuse_replace_of(monkeypatch, rec.HOLDOUT)
    with pytest.raises(PermissionError):
        cli.main(["answer", str(directory), "p0003", "no", "--handle", BY["handle"], "--email", BY["email"]])
    restore()
    record = rec.Record(directory)
    assert record.answers()["p0003"]["answer"] == "no"
    held = {patch["id"]: patch for patch in record.read(rec.HOLDOUT)["patches"]}
    assert held["p0003"]["answer"]["answer"] == rec.LINES_UP     # the file is behind the log
    # Before anything rewrites holdout.json from the log:
    states = pipeline_stages.pipeline(directory).states()
    assert states["score"] == STALE and states["store"] != DONE
    pairs, index = released(tmp_path, [old, root], "behind")
    assert pairs == {}
    why = {item["source"]: item["why"] for item in index["left_out"]}
    assert "event log" in why["records/a"] and "held back" in why["first-runs/a-auto"]
    code, out = check(root, capsys)
    assert code == 1 and "event log" in out
    held = {patch["id"]: patch for patch in record.read(rec.HOLDOUT)["patches"]}
    assert held["p0003"]["answer"]["answer"] == rec.LINES_UP     # status and check wrote nothing
    assert check(tmp_path / "out-before", capsys)[0] == 0        # a release is its shipped files
    # Scored again on the log's answers, decided again, stored and released.
    drain(directory)
    fresh = page(directory)
    assert fresh["scores"]["scores"]["bend"]["lines_up"] == 19
    decide_on(directory, "bend", fresh, "19 will do")
    assert drain(directory).states()["store"] == DONE
    assert released(tmp_path, [old, root], "after")[0]["A"]["source"] == "records/a"
    assert_published_coherently(tmp_path / "out-after")
    assert check(root, capsys)[0] == 0


def test_a_legacy_record_whose_log_is_ahead_of_holdout_json_is_held_back(tmp_path, scored, capsys):
    """Release must hold back the pair and its older candidates when a logged answer
    invalidates the score. Exported releases must remain checkable from their shipped
    files. (The same for a record stored before stores bound their answers: any logged
    answer its holdout.json does not hold is enough, whatever it does to the count.)"""
    root = tmp_path / "records"
    directory = stored(root)
    record = rec.Record(directory)
    fit = record.read(rec.FIT)
    fit.pop("schema", None)
    for key in ("answers_sha256", "basis"):
        fit["stored"].pop(key, None)
    record.write(rec.FIT, fit)
    final = record.read(rec.FINAL_REVIEW)
    final.pop("schema", None)
    final["decision"].pop("basis", None)
    record.write(rec.FINAL_REVIEW, final)
    pipeline = pipeline_stages.pipeline(directory)
    stamp = pipeline.stamp("store")
    pipeline._write_stamp("store", stamp | {"output_digest": engine.digest_files(directory, stamp["outputs"])})
    assert "A" in released(tmp_path, [root], "round5")[0]
    record.log("holdout-answer", unit="p0005", answer="no", by=BY)   # holdout.json never rewritten
    assert released(tmp_path, [root], "behind")[0] == {}
    assert check(root, capsys)[0] == 1


# ===================================================== 3. an answer changed and back while Score ran

def test_an_answer_changed_and_changed_back_while_score_ran_is_scored_again(tmp_path, scored, monkeypatch):
    """The successful stamp must describe the answers Score used so an inconsistent result
    can be rescored. (Changed after the job read its inputs, back after Score read
    holdout.json: the scores are of 19, the answers there are 20.)"""
    root = tmp_path / "records"
    directory = scored_pair(root)
    pipeline_stages.pipeline(directory).redo("score")
    assert pipeline_stages.pipeline(directory).states()["score"] == READY
    real_fit_unit, real_score, done = pipeline_stages.fit_unit, pipeline_stages.score, []

    def fit_unit_then_edit(pipeline):
        unit = real_fit_unit(pipeline)
        if "edit" not in done:
            done.append("edit")
            answer(directory, "p0007", "no")
        return unit

    def score_then_undo(*args, **kwargs):
        if "undo" not in done:
            done.append("undo")
            answer(directory, "p0007", rec.LINES_UP)
        return real_score(*args, **kwargs)
    monkeypatch.setattr(pipeline_stages, "fit_unit", fit_unit_then_edit)
    monkeypatch.setattr(pipeline_stages, "score", score_then_undo)
    assert pipeline_stages.pipeline(directory).run_stage("score")["status"] == "ok"
    monkeypatch.setattr(pipeline_stages, "fit_unit", real_fit_unit)
    monkeypatch.setattr(pipeline_stages, "score", real_score)
    assert done == ["edit", "undo"]
    scores = json.loads((directory / "work" / "scores.json").read_text())
    assert scores["scores"]["bend"]["lines_up"] == 19
    assert pipeline_stages.pipeline(directory).states()["score"] == STALE
    drain(directory)
    fresh = page(directory)
    assert fresh["scores"]["scores"]["bend"]["lines_up"] == 20
    decide_on(directory, "bend", fresh)
    assert drain(directory).states()["store"] == DONE


# ================================================= 4. a failure before the start, and a redo

def _redo_while_it_fails(monkeypatch, directory):
    """When the job comes to record its failure, `volreg redo <pair> score` runs in another
    thread, as another process would: it finishes before the stamp is written unless the
    job keeps it out. Returns the thread."""
    real, threads = engine.Pipeline._unstarted_failure, []

    def failing(self, name, error, *args, **kwargs):
        if not threads:
            thread = threading.Thread(target=lambda: pipeline_stages.pipeline(directory).redo("score"))
            threads.append(thread)
            thread.start()
            thread.join(timeout=1.0)
        return real(self, name, error, *args, **kwargs)
    monkeypatch.setattr(engine.Pipeline, "_unstarted_failure", failing)
    return threads


def test_a_redo_after_a_prerequisite_read_failed_leaves_the_stage_to_run_again(tmp_path, fake_stages, monkeypatch):
    """A failure from work invalidated by Redo must leave the stage eligible to run again.
    (A prerequisite read timed out; `redo score` came before the failure stamp.)"""
    as_windows_locks(monkeypatch)
    directory = scored_up_to_score(tmp_path)
    pipeline = pipeline_stages.pipeline(directory)

    def refused(name):
        raise PermissionError(13, "The process cannot access the file because it is being used by "
                                  "another process", str(directory / rec.HOLDOUT))
    pipeline._require_needs = refused
    threads = _redo_while_it_fails(monkeypatch, directory)
    assert pipeline.run_stage("score")["status"] == "failed"
    threads[0].join(timeout=30)
    assert not threads[0].is_alive()
    assert pipeline_stages.pipeline(directory).states()["score"] == READY
    assert drain(directory).states()["score"] == DONE


def test_a_redo_after_reading_the_redo_marks_failed_leaves_the_stage_to_run_again(tmp_path, fake_stages, monkeypatch):
    """A failure from work invalidated by Redo must leave the stage eligible to run again.
    (The same with redo.json the read that timed out.)"""
    as_windows_locks(monkeypatch)
    directory = scored_up_to_score(tmp_path)
    pipeline = pipeline_stages.pipeline(directory)
    refusals = [1]
    real_marks = pipeline.redo_marks

    def marks():
        if refusals:
            refusals.pop()
            raise PermissionError(13, "The process cannot access the file because it is being used by "
                                      "another process", str(directory / "work" / "stages" / "redo.json"))
        return real_marks()
    pipeline.redo_marks = marks
    threads = _redo_while_it_fails(monkeypatch, directory)
    assert pipeline.run_stage("score")["status"] == "failed"
    threads[0].join(timeout=30)
    assert not threads[0].is_alive()
    assert pipeline_stages.pipeline(directory).states()["score"] == READY


def test_a_redo_after_reading_the_settings_failed_leaves_the_stage_to_run_again(tmp_path, fake_stages, monkeypatch):
    """A failure from work invalidated by Redo must leave the stage eligible to run again.
    (`volreg job`: params.json timed out, `redo score` finished, then the job recorded its
    failure. It reads the settings again where no redo can come between, and runs.)"""
    as_windows_locks(monkeypatch)
    directory = scored_up_to_score(tmp_path)
    real, calls = pipeline_stages.pipeline, []

    def settings_timed_out(pair):
        if not calls:
            calls.append(1)
            real(pair).redo("score")                             # done while the read waited
            raise PermissionError(13, "The process cannot access the file because it is being used by "
                                      "another process", str(Path(pair) / pipeline_stages.PARAMS))
        return real(pair)
    monkeypatch.setattr(pipeline_stages, "pipeline", settings_timed_out)
    monkeypatch.setattr(cli, "advance", lambda pair: (None, []))
    assert cli.main(["job", str(directory), "score"]) == 0
    monkeypatch.setattr(pipeline_stages, "pipeline", real)
    assert pipeline_stages.pipeline(directory).states()["score"] == DONE


def test_without_a_redo_a_failure_before_the_start_still_stays_failed(tmp_path, fake_stages, monkeypatch):
    """A failure from work invalidated by Redo must leave the stage eligible to run again.
    (Only then: with no redo, a prerequisite that cannot be read fails the stage, as in
    round 6, so the error is seen.)"""
    as_windows_locks(monkeypatch)
    directory = scored_up_to_score(tmp_path)
    pipeline = pipeline_stages.pipeline(directory)

    def refused(name):
        raise PermissionError(13, "held", str(directory / rec.HOLDOUT))
    pipeline._require_needs = refused
    assert pipeline.run_stage("score")["status"] == "failed"
    assert pipeline_stages.pipeline(directory).states()["score"] == FAILED


# ============================================================================ 5. the viewer

def test_the_viewer_rides_out_a_replace_of_the_session(tmp_path, monkeypatch):
    """The Neuroglancer shortcut returned HTTP 500 during a brief Windows sharing conflict.
    (Not a blocker; the session is read as the record files are, asked again while Windows
    replaces it.)"""
    as_windows_locks(monkeypatch)
    session = tmp_path / "session"
    session.mkdir()
    (session / "session.json").write_text(json.dumps({"schema": pipeline_stages.fs.SCHEMA, "patches": []}))
    refused = refuse_opens_of(monkeypatch, session / "session.json", lambda: 3)
    loaded = pipeline_stages.fs.Session.load(session)
    assert refused[0] == 3 and loaded.directory == session
