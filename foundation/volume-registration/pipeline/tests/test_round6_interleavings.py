"""Round 6, step C: the interleavings of an answer edit, rescoring, a decision (fresh, from
a stale page, with an override), Store, Release and check, across records stored by this
code, by main, by round 5, and migrated ones, with units lining up and with none. Those
already in test_approval_coherence.py or test_sharing_conflicts.py are not repeated.

Each test opens with the requirement it checks, quoted from the review."""

import json
import time
from pathlib import Path

import pytest

from volreg import cli, engine, filelock
from volreg import record as rec
from volreg import stages as pipeline_stages
from volreg.engine import DONE, FAILED, STALE
from volreg.review.server import Conflict, Reviews

from test_approval_coherence import (BY, _answer_during_the_decision, _as_main_stored_it, _restamp_store,
                                     _without_basis, answer, assert_published_coherently, check, decide_on,
                                     decision, drain, install, page, scored, scored_pair, stored)  # noqa: F401
from test_redo_release import released
from test_review_blockers import CONTEXT, as_windows_locks, lock_kind  # noqa: F401 - a fixture
from test_sharing_conflicts import refuse_opens_of

COHERENCE = ("The saved approval and any override must cover the scores being published, and the "
             "release must contain the answers those scores used. Coherent legacy and migrated "
             "records must remain eligible.")


# ------------------------------------------------------------- decisions and what they saw

def test_a_note_added_while_the_decision_is_checked_does_not_stop_it(tmp_path, scored, monkeypatch):
    """The saved approval and any override must cover the scores being published, and the
    release must contain the answers those scores used. Coherent legacy and migrated records
    must remain eligible. (A note is not an answer: the scores are still the ones shown.)"""
    directory = scored_pair(tmp_path / "records")
    shown = page(directory)
    real = pipeline_stages.reviewed_unit

    def with_a_note(pipeline, *args, **kwargs):
        unit = real(pipeline, *args, **kwargs)
        rec.Record(directory).note("p0003", "a crack at the top", BY)
        return unit
    monkeypatch.setattr(pipeline_stages, "reviewed_unit", with_a_note)
    decide_on(directory, "bend", shown)
    assert decision(directory)["publish"] == "bend"


def test_an_answer_that_lines_nothing_up_changed_during_the_decision_does_not_stop_it(tmp_path, scored, monkeypatch):
    """The saved approval and any override must cover the scores being published, and the
    release must contain the answers those scores used. Coherent legacy and migrated records
    must remain eligible. ("partly" to "no" when nothing lines up: the units scored, none,
    are the same, and so is the score; the decision covers it, and it is stored.)"""
    root = tmp_path / "records"
    directory = scored_pair(root, zero=True)
    shown = page(directory)
    resume = _answer_during_the_decision(monkeypatch, directory, "p0004", "no")
    decide_on(directory, "affine", shown, "nothing lines up yet")
    resume()
    assert decision(directory)["override"]["bar_missed"] == ["enough"]
    pipeline = drain(directory)
    assert pipeline.states()["store"] == DONE
    assert released(tmp_path, [root], "zero")[0]["A"]["score"]["lines_up"] == 0


def test_a_page_whose_scores_are_being_made_again_cannot_decide(tmp_path, scored):
    """The saved approval and any override must cover the scores being published, and the
    release must contain the answers those scores used. Coherent legacy and migrated records
    must remain eligible. (An answer changed and not yet scored: the page shows the old
    scores, and the decision is refused until they are made again.)"""
    directory = scored_pair(tmp_path / "records")
    answer(directory, "p0005", "no")
    shown = page(directory)                     # the old scores, since nothing is scored yet
    with pytest.raises(Conflict):
        decide_on(directory, "bend", shown)
    assert decision(directory) is None


def test_a_page_with_no_scores_sends_no_basis_and_cannot_decide(tmp_path, scored):
    """The saved approval and any override must cover the scores being published, and the
    release must contain the answers those scores used. Coherent legacy and migrated records
    must remain eligible. (Before every item is answered the page shows no scores.)"""
    directory = scored_pair(tmp_path / "records")
    shown = Reviews(directory.parent, BY, auto_advance=False).blind(directory.name)
    assert shown["basis"] is None and shown["scores"] is None
    with pytest.raises(Conflict):
        Reviews(directory.parent, BY, auto_advance=False).decide(directory.name, "bend", "", "c001", "f001", None)


def test_an_override_decision_whose_scores_then_pass_is_decided_again(tmp_path, scored):
    """The saved approval and any override must cover the scores being published, and the
    release must contain the answers those scores used. Coherent legacy and migrated records
    must remain eligible. (An override for 19 units; the twentieth lines up again: the
    override does not carry over to scores it was not made on.)"""
    root = tmp_path / "records"
    directory = scored_pair(root)
    answer(directory, "p0001", "no")
    drain(directory)
    decide_on(directory, "bend", page(directory), "19 will do")
    drain(directory)
    answer(directory, "p0001", "partly")
    answer(directory, "p0002", "no")
    answer(directory, "p0001", rec.LINES_UP)
    answer(directory, "p0002", rec.LINES_UP)
    pipeline = drain(directory)                 # twenty again
    # Back to the twenty the scores before the override had: these are not the 19 it was
    # made on, so it is not taken for them.
    assert pipeline.states()["store"] == FAILED and "decide again" in pipeline.stamp("store")["error"]
    assert released(tmp_path, [root], "passing")[0] == {}
    decide_on(directory, "bend", page(directory))
    assert decision(directory)["override"] is None
    drain(directory)
    assert released(tmp_path, [root], "again")[0]["A"]["score"]["passes"]


# ---------------------------------------------------------- after Store, and Store again

def test_an_answer_edited_after_store_and_changed_back_is_released_again(tmp_path, scored, capsys):
    """The saved approval and any override must cover the scores being published, and the
    release must contain the answers those scores used. Coherent legacy and migrated records
    must remain eligible. (The answers the stored scores used are there again.)"""
    root = tmp_path / "records"
    directory = stored(root)
    answer(directory, "p0006", "no")
    assert released(tmp_path, [root], "edited")[0] == {}
    answer(directory, "p0006", rec.LINES_UP)
    assert "A" in released(tmp_path, [root], "back")[0]
    pipeline = drain(directory)                 # scored twice more, to the same scores
    assert pipeline.states()["store"] == DONE
    assert check(root, capsys)[0] == 0


def test_store_redone_after_an_edit_without_a_new_decision_fails_and_releases_nothing(tmp_path, scored, capsys):
    """The saved approval and any override must cover the scores being published, and the
    release must contain the answers those scores used. Coherent legacy and migrated records
    must remain eligible. (`volreg redo <pair> store` does not carry the old approval over.)"""
    root = tmp_path / "records"
    directory = stored(root)
    answer(directory, "p0006", "no")
    pipeline_stages.pipeline(directory).redo("store")
    pipeline = drain(directory)
    assert pipeline.states()["store"] == FAILED and "decide again" in pipeline.stamp("store")["error"]
    assert released(tmp_path, [root], "redone")[0] == {}
    assert check(root, capsys)[0] == 1


def test_a_record_main_stored_is_stored_again_on_upgrade_and_stays_eligible(tmp_path, scored, capsys):
    """The saved approval and any override must cover the scores being published, and the
    release must contain the answers those scores used. Coherent legacy and migrated records
    must remain eligible. (Store rests on the scores now, so a pair main stored shows it
    stale; running it again stores the same result, bound to its answers.)"""
    root = tmp_path / "records"
    directory = stored(root)
    _as_main_stored_it(directory)
    fit = rec.Record(directory).read(rec.FIT)
    fit.pop("schema", None)
    fit.pop("bend", None)
    rec.Record(directory).write(rec.FIT, fit)
    pipeline = pipeline_stages.pipeline(directory)
    stamp = pipeline.stamp("store")
    stamp["inputs"]["needs"].pop("score")                        # main's Store needed the decision only
    pipeline._write_stamp("store", stamp)
    _restamp_store(directory)
    assert pipeline_stages.pipeline(directory).states()["store"] == STALE
    assert "A" in released(tmp_path, [root], "main")[0]          # eligible as it is
    pipeline = drain(directory)
    assert pipeline.states()["store"] == DONE
    assert rec.Record(directory).read(rec.FIT)["stored"]["answers_sha256"]
    assert "A" in released(tmp_path, [root], "upgraded")[0]
    assert check(root, capsys)[0] == 0


def test_a_record_main_stored_can_keep_its_store_by_adopting(tmp_path, scored, capsys):
    """The saved approval and any override must cover the scores being published, and the
    release must contain the answers those scores used. Coherent legacy and migrated records
    must remain eligible. (`volreg adopt <pair> store`: kept as main stored it.)"""
    root = tmp_path / "records"
    directory = stored(root)
    pipeline = pipeline_stages.pipeline(directory)
    stamp = pipeline.stamp("store")
    stamp["inputs"]["needs"].pop("score")
    pipeline._write_stamp("store", stamp)
    assert cli.main(["adopt", str(directory), "store", "store rests on the scores too"]) == 0
    assert pipeline_stages.pipeline(directory).states()["store"] == DONE
    assert "A" in released(tmp_path, [root], "adopted")[0]


def test_check_on_a_zero_unit_record_without_an_override_answers_and_fails(tmp_path, scored, capsys):
    """The saved approval and any override must cover the scores being published, and the
    release must contain the answers those scores used. Coherent legacy and migrated records
    must remain eligible. (A record that publishes zero units with no override, as an older
    Store left it: `volreg check` says so, and does not raise.)"""
    root = tmp_path / "records"
    directory = stored(root, "affine", zero=True, reason="nothing lines up")
    final = rec.Record(directory).read(rec.FINAL_REVIEW)
    final.pop("schema", None)
    final["decision"]["override"] = None
    rec.Record(directory).write(rec.FINAL_REVIEW, final)
    code, out = check(root, capsys)
    assert code == 1 and "override" in out
    assert released(tmp_path, [root], "zero")[0] == {}


@pytest.mark.parametrize("kind", ["main", "round5"])
def test_a_zero_unit_legacy_record_is_checked_without_raising(tmp_path, scored, capsys, kind):
    """The saved approval and any override must cover the scores being published, and the
    release must contain the answers those scores used. Coherent legacy and migrated records
    must remain eligible. (Zero units, an override, stored before stores bound answers:
    eligible, checked; then an answer lines up: left out, and check answers 1.)"""
    root = tmp_path / "records"
    directory = stored(root, "affine", zero=True, reason="nothing lines up")
    record = rec.Record(directory)
    if kind == "main":
        _as_main_stored_it(directory)
        fit = record.read(rec.FIT)
        fit.pop("schema", None)
        fit.pop("bend", None)
        record.write(rec.FIT, fit)
    else:
        fit = record.read(rec.FIT)
        fit.pop("schema", None)
        for key in ("answers_sha256", "basis"):
            fit["stored"].pop(key, None)
            fit["scored"].pop(key, None)
        record.write(rec.FIT, fit)
        _without_basis(directory)
    _restamp_store(directory)
    assert "A" in released(tmp_path, [root], "legacy")[0]
    assert check(root, capsys)[0] == 0
    answer(directory, "p0003", rec.LINES_UP)
    assert released(tmp_path, [root], "edited")[0] == {}
    code, out = check(root, capsys)
    assert code == 1 and "NOT WHAT IS STORED" in out


# ------------------------------------------------- releases while another process works

def _edit_score_decide_store(directory, rounds, done):
    """In a process of its own: change an answer, score, decide on the page, store; again."""
    install()
    directory = Path(directory)
    for index in range(int(rounds)):
        answer(directory, "p0001", "no" if index % 2 == 0 else rec.LINES_UP)
        drain(directory)
        decide_on(directory, "bend", page(directory), "kept on purpose")
        drain(directory)
    Path(done).write_text("done")


def test_releases_while_another_process_edits_scores_decides_and_stores_are_coherent(tmp_path, scored, capsys, lock_kind):
    """The saved approval and any override must cover the scores being published, and the
    release must contain the answers those scores used. Coherent legacy and migrated records
    must remain eligible. (Every release made while answers, scores, decisions and stores
    change underneath holds scores of the answers it holds, covered by its decision.)"""
    root = tmp_path / "records"
    directory = stored(root)
    done = tmp_path / "done"
    worker = CONTEXT.Process(target=_edit_score_decide_store, args=(str(directory), 12, str(done)))
    worker.start()
    made, shipped = 0, 0
    try:
        while not done.exists() and made < 300 and worker.is_alive():
            label = f"busy{made}"
            try:
                index = released(tmp_path, [root], label)[1]
            except ValueError:
                index = None
            if index is not None:
                folder = tmp_path / f"out-{label}"
                assert_published_coherently(folder)
                assert check(folder, capsys)[0] == 0
                shipped += bool(index["pairs"])
            made += 1
    finally:
        worker.join(240)
    assert worker.exitcode == 0 and made > 0
    print("releases made:", made, "with the pair:", shipped)
    drain(directory)
    assert_published_coherently(released(tmp_path, [root], "end") and tmp_path / "out-end")


# ------------------------------------------------------- sharing conflicts on the server

def test_the_review_pages_ride_out_a_replace_of_holdout_json(tmp_path, scored, monkeypatch):
    """Routine record reads must recover from temporary sharing conflicts and still surface
    persistent errors. (The review server's pair list and blind page read the same files.)"""
    as_windows_locks(monkeypatch)
    root = tmp_path / "records"
    directory = scored_pair(root)
    pick = page(directory)
    refused = refuse_opens_of(monkeypatch, directory / rec.HOLDOUT, lambda: 3)
    reviews = Reviews(root, BY, auto_advance=False)
    assert reviews.pairs()["pairs"][0]["name"] == "a" and refused[0] == 3
    refused[0] = 0
    assert reviews.blind("a")["basis"] == pick["basis"]
    refused[0] = 0
    decide_on(directory, "bend", pick)
    assert refused[0] == 3 and decision(directory)["publish"] == "bend"


def test_release_rides_out_a_replace_and_still_checks_the_answers(tmp_path, scored, monkeypatch):
    """Routine record reads must recover from temporary sharing conflicts and still surface
    persistent errors. (Release reads holdout.json for the answers it checks.)"""
    as_windows_locks(monkeypatch)
    root = tmp_path / "records"
    directory = stored(root)
    refused = refuse_opens_of(monkeypatch, directory / rec.HOLDOUT, lambda: 3)
    assert "A" in released(tmp_path, [root], "busy")[0] and refused[0] == 3


# ------------------------------------------------- found by an independent review of the fix

def test_an_answer_logged_whose_rewrite_of_holdout_json_failed_is_scored_again(tmp_path, scored):
    """The saved approval and any override must cover the scores being published, and the
    release must contain the answers those scores used. Coherent legacy and migrated records
    must remain eligible. (`volreg answer` logged the answer, and its rewrite of holdout.json
    failed -- refused past the wait on Windows, or the process stopped. Since round 7 the
    log is what the stages read the answers from: the scores are stale at once, made again,
    and the pair is not stuck.)"""
    root = tmp_path / "records"
    directory = scored_pair(root)
    rec.Record(directory).log("holdout-answer", unit="p0004", answer="no", by=BY)
    assert pipeline_stages.pipeline(directory).states()["score"] == STALE
    drain(directory)                                             # scored again, on 19
    fresh = page(directory)
    assert fresh["scores"]["scores"]["bend"]["lines_up"] == 19
    decide_on(directory, "bend", fresh, "19 will do")
    assert drain(directory).states()["store"] == DONE
    assert_published_coherently(released(tmp_path, [root], "healed") and tmp_path / "out-healed")


def test_store_heals_holdout_json_when_its_rewrite_failed(tmp_path, scored):
    """The saved approval and any override must cover the scores being published, and the
    release must contain the answers those scores used. Coherent legacy and migrated records
    must remain eligible. (The same, with the decision made before: Store refuses, and the
    pair moves on to be scored and decided again.)"""
    root = tmp_path / "records"
    directory = scored_pair(root)
    decide_on(directory, "bend", page(directory))
    rec.Record(directory).log("holdout-answer", unit="p0004", answer="no", by=BY)
    pipeline = drain(directory)
    assert pipeline.states()["store"] == FAILED and "decide again" in pipeline.stamp("store")["error"]
    assert page(directory)["scores"]["scores"]["bend"]["lines_up"] == 19


def test_publishing_nothing_after_a_redo_survives_an_answer_edit(tmp_path, scored):
    """The saved approval and any override must cover the scores being published, and the
    release must contain the answers those scores used. Coherent legacy and migrated records
    must remain eligible. (Nothing is published, so no scores need covering: an answer edit
    does not fail Store, and the pair's older record stays released.)"""
    from test_approval_coherence import _migrated
    root, old = tmp_path / "records", tmp_path / "first-runs"
    _migrated(old, "a", "A", 20)
    directory = scored_pair(root)
    pipeline_stages.pipeline(directory).redo("recheck")
    drain(directory)
    decide_on(directory, "nothing", page(directory))
    assert drain(directory).states()["store"] == DONE
    assert "A" in released(tmp_path, [old, root], "before")[0]
    answer(directory, "p0003", "no")
    assert drain(directory).states()["store"] == DONE
    assert "A" in released(tmp_path, [old, root], "after")[0]


def test_the_fits_page_and_holdout_page_ride_out_a_replace(tmp_path, scored, monkeypatch):
    """Routine record reads must recover from temporary sharing conflicts and still surface
    persistent errors. (The fits page reads the stored fit.json; the hold-out page the scores.)"""
    as_windows_locks(monkeypatch)
    root = tmp_path / "records"
    directory = stored(root)
    refused = refuse_opens_of(monkeypatch, directory / rec.FIT, lambda: 2)
    reviews = Reviews(root, BY, auto_advance=False)
    assert reviews.fits()["fits"][0]["pair"] == "A" and refused[0] == 2
    refused = refuse_opens_of(monkeypatch, directory / "work" / "scores.json", lambda: 2)
    assert reviews.holdout("a")["scores"] and refused[0] == 2
