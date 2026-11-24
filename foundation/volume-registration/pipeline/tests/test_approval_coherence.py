"""Round 6 of the review, first blocker: the approval, the scores it covers and the hold-out
answers those scores used must stay one thing from the review page to the release.

Each test opens with the requirement it checks, quoted from the review. The pair runs on the
real Score stage (so `volreg check` can recompute what is stored), with stand-ins for the
stages that read volumes: twenty hold-out patches, each measured one fine voxel from where
an identity fit puts it, so twenty answered "lines up" clear the bar and nineteen do not."""

import json
import shutil

import pytest

from volreg import cli, engine, release
from volreg import record as rec
from volreg import stages as pipeline_stages
from volreg.engine import DONE, FAILED
from volreg.review.server import Conflict, Reviews
from volreg.scoring import bar_check, score

import test_redo_release as fakes
from test_fit_binding import Server, _as_main_stored_it, _without_basis
from test_redo_release import new_pair, pick_all, released
from test_review_blockers import REVIEWER

BY = REVIEWER
COHERENCE = ("The saved approval and any override must cover the scores being published, and the "
             "release must contain the answers those scores used. Coherent legacy and migrated "
             "records must remain eligible.")
PATCHES = 20
ANSWER = ["answer", "--handle", BY["handle"], "--email", BY["email"]]


# ------------------------------------------------------------------ the stand-in stages

def holdout_stage(pipeline, params):
    """Twenty measured hold-out patches; each answered "lines up" (or, with work/zero,
    "partly") when the stage first lists it."""
    patches = [{"id": f"p{index:04d}", "stretch": 0, "origin_zyx": [10 * index, 0, 0],
                "centre_zyx": [10.0 * index, 0.0, 0.0], "fine_zyx": [10.0 * index + 1.0, 0.0, 0.0]}
               for index in range(1, PATCHES + 1)]
    pipeline.record.refresh_holdout(patches, patch_coarse_voxels=128, stretches=5)
    answer = "partly" if pipeline.work("zero").exists() else rec.LINES_UP
    answered = pipeline.record.answers()
    for patch in patches:
        if patch["id"] not in answered:
            pipeline.record.answer(patch["id"], answer, BY)
    pipeline.work("holdout-patches.json").write_text(json.dumps([p["id"] for p in patches]) + "\n")
    return ["work/holdout-patches.json"]


def fit_stage(pipeline, params):
    """An identity fit, without a bend: coarse centres land where they are."""
    counter = pipeline.work("fits-made")
    made = int(counter.read_text()) + 1 if counter.exists() else 1
    counter.write_text(str(made))
    fit_id = f"f{made:03d}"
    out = pipeline.work("fit")
    out.mkdir(parents=True, exist_ok=True)
    (out / "transform.json").write_text(json.dumps({
        "schema_version": "1.1.0", "fixed_volume": "fixed.zarr", "fit": fit_id,
        "transformation_matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]}) + "\n")
    (out / "fit.json").write_text(json.dumps({"fit": fit_id, "fine_um": 2.0, "model": "affine",
                                              "points": {"ids": []}, "bend": None}) + "\n")
    return [pipeline_stages.FIT_TRANSFORM, pipeline_stages.FIT_META]


def recheck_stage(pipeline, params):
    pipeline.work("recheck.json").write_text(json.dumps({
        "under": fakes._fit_id(pipeline), "drift_fine_voxels": 0.5, "median": 0.0, "max": 0.0,
        "flagged": [], "patches": {}}) + "\n")
    return ["work/recheck.json"]


def install():
    for name, stand_in in (("run_capture", fakes.fake_capture), ("run_holdout", holdout_stage),
                           ("run_fit", fit_stage), ("run_recheck", recheck_stage),
                           ("run_blind", fakes.fake_blind)):
        setattr(pipeline_stages, name, stand_in)


@pytest.fixture
def scored(monkeypatch):
    """The stand-ins, with the real Score and Store."""
    for name, stand_in in (("run_capture", fakes.fake_capture), ("run_holdout", holdout_stage),
                           ("run_fit", fit_stage), ("run_recheck", recheck_stage),
                           ("run_blind", fakes.fake_blind)):
        monkeypatch.setattr(pipeline_stages, name, stand_in)


# ---------------------------------------------------------------------- the review page

def page(directory):
    """What the blind page holds once its round is answered: the round, the fit, the scores
    and whatever it sends back to say which scores it showed."""
    pick_all(directory)
    shown = Reviews(directory.parent, BY, auto_advance=False).blind(directory.name)
    return {"round": shown["round"]["id"], "fit": shown["fit"], "basis": shown.get("basis"),
            "scores": shown["scores"]}


def decide_on(directory, publish, shown, reason=""):
    """A decision sent as the page sends it, for what it showed."""
    reviews = Reviews(directory.parent, BY, auto_advance=False)
    try:
        return reviews.decide(directory.name, publish, reason, round_id=shown["round"],
                              fit_id=shown["fit"], basis=shown["basis"])
    except TypeError:                            # a server that takes no basis
        return reviews.decide(directory.name, publish, reason, round_id=shown["round"],
                              fit_id=shown["fit"])


def answer(directory, unit, value):
    """`volreg answer`, as a person types it."""
    assert cli.main(["answer", str(directory), unit, value, *ANSWER[1:]]) == 0


def drain(directory):
    return fakes.drain(directory)


def decision(directory):
    record = rec.Record(directory)
    return (record.read(rec.FINAL_REVIEW) if record.exists(rec.FINAL_REVIEW) else {}).get("decision")


def scored_pair(root, name="a", pair="A", zero=False):
    """A pair scored and waiting for its decision."""
    directory = new_pair(root, name, pair)
    if zero:
        (directory / "work" / "zero").write_text("")
    pipeline = drain(directory)
    assert pipeline.states()["final-review"] == engine.WAITING
    return directory


def stored(root, publish="bend", zero=False, reason="", **names):
    directory = scored_pair(root, zero=zero, **names)
    decide_on(directory, publish, page(directory), reason)
    pipeline = drain(directory)
    assert pipeline.states()["store"] == DONE
    return directory


def check(path, capsys):
    """`volreg check`, which must answer, never raise."""
    capsys.readouterr()
    code = cli.main(["check", str(path)])
    return code, capsys.readouterr().out


def recomputed(folder, pair="A"):
    """The published transform's score, recomputed from a released pair's own files."""
    held = json.loads((folder / pair / rec.HOLDOUT).read_text())
    transform = json.loads((folder / pair / rec.TRANSFORM).read_text())
    return score(transform, held, True, 2.0)


def assert_published_coherently(folder):
    """Every released pair: the stored score is the one its answers give, and the decision
    passes the bar or overrides exactly what it misses."""
    index = json.loads((folder / "index.json").read_text())
    for entry in index["pairs"]:
        fit = json.loads((folder / entry["folder"] / rec.FIT).read_text())
        final = json.loads((folder / entry["folder"] / rec.FINAL_REVIEW).read_text())
        publish = final["decision"]["publish"]
        result = recomputed(folder, entry["folder"])
        stored_score = fit["scores"][publish]
        assert result["lines_up"] == stored_score["lines_up"]
        for key in ("median_fine_voxels", "p90_fine_voxels"):
            assert result.get(key) == stored_score.get(key)
        missed = sorted(part for part, ok in bar_check(result)["checks"].items() if not ok)
        override = final["decision"].get("override") or {}
        assert missed == sorted(override.get("bar_missed") or [])


# ================================================== (a) an answer changed during a decision

def _answer_during_the_decision(monkeypatch, directory, unit, value, back=None):
    """Pause the decision handler after its readiness check (the scores read) and change an
    answer there with `volreg answer`, as the review did; `back` changes it back too."""
    real = pipeline_stages.reviewed_unit
    done = []

    def paused(pipeline, *args, **kwargs):
        unit_read = real(pipeline, *args, **kwargs)
        if not done:
            done.append(1)
            answer(directory, unit, value)
            if back:
                answer(directory, unit, back)
        return unit_read
    monkeypatch.setattr(pipeline_stages, "reviewed_unit", paused)
    return lambda: monkeypatch.setattr(pipeline_stages, "reviewed_unit", real)


def test_a_an_answer_changed_while_the_decision_is_checked_is_not_approved(tmp_path, scored, monkeypatch, capsys):
    """The saved approval and any override must cover the scores being published, and the
    release must contain the answers those scores used. Coherent legacy and migrated records
    must remain eligible. (a) The decision handler paused after its readiness check; one of
    20 lines-up answers changed with `volreg answer`; the approval was saved without an
    override, and after rescoring Store and Release published a result below the bar."""
    root = tmp_path / "records"
    directory = scored_pair(root)
    shown = page(directory)
    assert shown["scores"]["scores"]["bend"]["bar"]["passes"]
    resume = _answer_during_the_decision(monkeypatch, directory, "p0007", "no")
    with pytest.raises(Conflict):
        decide_on(directory, "bend", shown)
    resume()
    assert decision(directory) is None
    pipeline = drain(directory)                                 # rescored on 19
    assert pipeline.states()["store"] != DONE
    assert released(tmp_path, [root], "after")[0] == {}
    assert check(root, capsys)[0] == 0
    fresh = page(directory)                                     # the page as it is now
    assert not fresh["scores"]["scores"]["bend"]["bar"]["passes"]
    with pytest.raises(ValueError, match="reason"):
        decide_on(directory, "bend", fresh)
    decide_on(directory, "bend", fresh, "19 are enough here")
    assert decision(directory)["override"]["bar_missed"] == ["enough"]
    drain(directory)
    pairs, _index = released(tmp_path, [root], "decided")
    assert pairs["A"]["score"]["lines_up"] == 19
    assert_published_coherently(tmp_path / "out-decided")
    assert check(tmp_path / "out-decided", capsys)[0] == 0


def test_a_an_answer_changed_and_changed_back_during_the_decision_is_approved(tmp_path, scored, monkeypatch):
    """The saved approval and any override must cover the scores being published, and the
    release must contain the answers those scores used. Coherent legacy and migrated records
    must remain eligible. (The answers the scores used are the ones there again.)"""
    root = tmp_path / "records"
    directory = scored_pair(root)
    shown = page(directory)
    resume = _answer_during_the_decision(monkeypatch, directory, "p0007", "no", back=rec.LINES_UP)
    decide_on(directory, "bend", shown)
    resume()
    assert decision(directory)["publish"] == "bend" and decision(directory)["override"] is None
    drain(directory)
    assert_published_coherently(released(tmp_path, [root], "same") and tmp_path / "out-same")


# ================================================ (b) a page submitted after rescoring

def _rescored_behind_the_page(directory, unit="p0007", value="no"):
    shown = page(directory)
    answer(directory, unit, value)
    drain(directory)                                            # rescored; same fit, same round
    assert page(directory)["round"] == shown["round"] and page(directory)["fit"] == shown["fit"]
    return shown


def test_b_a_note_on_a_page_from_before_rescoring_is_not_turned_into_an_override(tmp_path, scored):
    """The saved approval and any override must cover the scores being published, and the
    release must contain the answers those scores used. Coherent legacy and migrated records
    must remain eligible. (b) A page submitted after rescoring finished, with the same fit and
    round: the reviewer's note must not become an override for a failure the page never
    showed."""
    directory = scored_pair(tmp_path / "records")
    shown = _rescored_behind_the_page(directory)
    assert shown["scores"]["scores"]["bend"]["bar"]["passes"]   # what the reviewer saw
    with pytest.raises(Conflict, match="reload"):
        decide_on(directory, "bend", shown, "looks good to me")
    assert decision(directory) is None


def test_b_a_page_from_before_rescoring_without_a_reason_hears_to_reload(tmp_path, scored):
    """The saved approval and any override must cover the scores being published, and the
    release must contain the answers those scores used. Coherent legacy and migrated records
    must remain eligible. (Without a reason the stale page is told to reload, not to give a
    reason for a failure it never showed.)"""
    directory = scored_pair(tmp_path / "records")
    shown = _rescored_behind_the_page(directory)
    with pytest.raises(Conflict, match="reload"):
        decide_on(directory, "bend", shown)
    assert decision(directory) is None


def test_b_a_page_from_before_rescoring_is_refused_while_the_new_scores_still_pass(tmp_path, scored):
    """The saved approval and any override must cover the scores being published, and the
    release must contain the answers those scores used. Coherent legacy and migrated records
    must remain eligible. (25 line up, one changes: still passing, but not the scores shown.)"""
    directory = new_pair(tmp_path / "records")
    drain(directory)
    record = rec.Record(directory)
    held = record.read(rec.HOLDOUT)
    for number in range(21, 26):                                # five more, answered
        held["patches"].append({"id": f"p{number:04d}", "stretch": 0, "origin_zyx": [10 * number, 0, 0],
                                "centre_zyx": [10.0 * number, 0, 0], "fine_zyx": [10.0 * number + 1, 0, 0]})
    record.refresh_holdout(held["patches"])
    for number in range(21, 26):
        answer(directory, f"p{number:04d}", rec.LINES_UP)
    drain(directory)
    shown = _rescored_behind_the_page(directory)
    assert page(directory)["scores"]["scores"]["bend"]["bar"]["passes"]
    with pytest.raises(Conflict, match="reload"):
        decide_on(directory, "bend", shown)


def test_b_a_page_from_before_rescoring_gets_a_409_over_http(tmp_path, scored):
    """The saved approval and any override must cover the scores being published, and the
    release must contain the answers those scores used. Coherent legacy and migrated records
    must remain eligible. (The same, through the server, as the page sends it.)"""
    root = tmp_path / "records"
    directory = scored_pair(root)
    shown = _rescored_behind_the_page(directory)
    server = Server(root)
    try:
        status, body = server.post("/api/pairs/a/decide", {"publish": "bend", "reason": "note",
                                                           "round": shown["round"], "fit": shown["fit"],
                                                           "basis": shown["basis"]})
        assert status == 409 and "reload" in body["error"]
        status, body = server.post("/api/pairs/a/decide", {"publish": "bend", "reason": "note",
                                                           "round": shown["round"], "fit": shown["fit"]})
        assert status == 409                                    # a page that says nothing of its scores
        fresh = page(directory)
        status, body = server.post("/api/pairs/a/decide", {"publish": "affine", "reason": "19 will do",
                                                           "round": fresh["round"], "fit": fresh["fit"],
                                                           "basis": fresh["basis"]})
        assert status == 200, body
    finally:
        server.close()
    assert decision(directory)["override"]["bar_missed"] == ["enough"]


def test_b_the_blind_page_sends_the_scores_it_showed():
    """The saved approval and any override must cover the scores being published, and the
    release must contain the answers those scores used. Coherent legacy and migrated records
    must remain eligible. (The page says which scores it showed.)"""
    from pathlib import Path
    import volreg.review.server as server_module
    page_source = (Path(server_module.__file__).parent / "blind.html").read_text(encoding="utf-8")
    assert "basis: state.basis" in page_source


# ======================================== (c) an answer edited after a successful Store

@pytest.mark.parametrize("publish", ["bend", "affine"])
def test_c_an_answer_edited_after_store_is_not_released_with_the_old_scores(tmp_path, scored, capsys, publish):
    """The saved approval and any override must cover the scores being published, and the
    release must contain the answers those scores used. Coherent legacy and migrated records
    must remain eligible. (c) A passing score on 20 units stored; then an answer edited:
    Release must not copy the old scores with the new answers."""
    root = tmp_path / "records"
    directory = stored(root, publish)
    released(tmp_path, [root], "before")
    assert_published_coherently(tmp_path / "out-before")
    answer(directory, "p0003", "no")
    pairs, index = released(tmp_path, [root], "edited")
    assert pairs == {}
    assert "answer" in {item["source"]: item["why"] for item in index["left_out"]}["records/a"]
    code, out = check(root, capsys)
    assert code == 1 and "NOT WHAT IS STORED" in out
    assert check(tmp_path / "out-before", capsys)[0] == 0       # what was released stays coherent
    pipeline = drain(directory)                                  # rescored: the decision is not for these
    assert pipeline.states()["store"] == FAILED and "decide again" in pipeline.stamp("store")["error"]
    assert released(tmp_path, [root], "rescored")[0] == {}
    decide_on(directory, publish, page(directory), "19 will do")
    drain(directory)
    pairs, _index = released(tmp_path, [root], "again")
    assert pairs["A"]["score"]["lines_up"] == 19
    assert_published_coherently(tmp_path / "out-again")
    assert check(root, capsys)[0] == 0


def test_c_a_zero_unit_override_is_released_and_checked_without_raising(tmp_path, scored, capsys):
    """The saved approval and any override must cover the scores being published, and the
    release must contain the answers those scores used. Coherent legacy and migrated records
    must remain eligible. (A permitted override with no unit lining up is coherent: released,
    and `volreg check` answers instead of raising.)"""
    root = tmp_path / "records"
    directory = stored(root, "affine", zero=True, reason="no patch lines up yet; publish the affine")
    assert decision(directory)["override"]["bar_missed"] == ["enough"]
    pairs, _index = released(tmp_path, [root], "zero")
    assert pairs["A"]["score"]["lines_up"] == 0
    code, out = check(tmp_path / "out-zero", capsys)
    assert code == 0, out
    assert check(root, capsys)[0] == 0


def test_c_a_zero_unit_score_is_not_released_after_an_answer_lines_up(tmp_path, scored, capsys):
    """The saved approval and any override must cover the scores being published, and the
    release must contain the answers those scores used. Coherent legacy and migrated records
    must remain eligible. (c) With a permitted override for zero units, an answer edited to
    "lines up" after Store: Release shipped one current lines-up answer and a stored count of
    zero, and `volreg check` raised KeyError."""
    root = tmp_path / "records"
    directory = stored(root, "affine", zero=True, reason="no patch lines up yet")
    answer(directory, "p0002", rec.LINES_UP)
    pairs, _index = released(tmp_path, [root], "edited")
    assert pairs == {}
    code, out = check(root, capsys)                              # never KeyError
    assert code == 1 and "NOT WHAT IS STORED" in out
    shipped = tmp_path / "shipped"                               # the record's shared files, copied with the edit
    shutil.copytree(directory, shipped / "A", ignore=shutil.ignore_patterns("work", "events.jsonl", "*.lock"))
    code, out = check(shipped, capsys)
    assert code == 1 and "NOT WHAT IS STORED" in out


def test_c_an_answer_edited_after_the_decision_before_store_is_not_stored(tmp_path, scored, capsys):
    """The saved approval and any override must cover the scores being published, and the
    release must contain the answers those scores used. Coherent legacy and migrated records
    must remain eligible. (The answer changes after the decision, before Store runs.)"""
    root = tmp_path / "records"
    directory = scored_pair(root)
    decide_on(directory, "bend", page(directory))
    answer(directory, "p0011", "no")
    pipeline = pipeline_stages.pipeline(directory)
    assert pipeline.run_stage("store")["status"] in ("failed", "superseded")   # chosen before the rescore
    assert not rec.Record(directory).exists(rec.FIT)
    pipeline = drain(directory)
    assert pipeline.states()["store"] == FAILED
    assert released(tmp_path, [root], "edited")[0] == {}
    code, out = check(root, capsys)
    assert code == 1 and "nothing is stored for its decision yet" in out


def test_c_an_answer_edited_while_store_writes_is_not_stored_under_the_old_scores(tmp_path, scored, monkeypatch, capsys):
    """The saved approval and any override must cover the scores being published, and the
    release must contain the answers those scores used. Coherent legacy and migrated records
    must remain eligible. (Store has read the decision and the scores; an answer changes
    before it writes.)"""
    root = tmp_path / "records"
    directory = scored_pair(root)
    decide_on(directory, "bend", page(directory))
    real = pipeline_stages.reviewed_unit

    def then_an_answer(pipeline, *args, **kwargs):
        unit = real(pipeline, *args, **kwargs)
        answer(directory, "p0011", "no")
        return unit
    monkeypatch.setattr(pipeline_stages, "reviewed_unit", then_an_answer)
    stamp = pipeline_stages.pipeline(directory).run_stage("store")
    monkeypatch.setattr(pipeline_stages, "reviewed_unit", real)
    assert stamp["status"] == "failed"
    assert not rec.Record(directory).exists(rec.FIT)
    assert released(tmp_path, [root], "edited")[0] == {}


def test_c_a_release_made_before_the_edit_holds_the_answers_its_scores_used(tmp_path, scored, capsys):
    """The saved approval and any override must cover the scores being published, and the
    release must contain the answers those scores used. Coherent legacy and migrated records
    must remain eligible. (A release is one moment's record; an edit after it is not in it.)"""
    root = tmp_path / "records"
    directory = stored(root)
    released(tmp_path, [root], "then")
    answer(directory, "p0001", "partly")
    assert_published_coherently(tmp_path / "out-then")
    assert check(tmp_path / "out-then", capsys)[0] == 0


def test_c_an_edit_holds_back_the_pairs_older_record_too(tmp_path, scored, capsys):
    """The saved approval and any override must cover the scores being published, and the
    release must contain the answers those scores used. Coherent legacy and migrated records
    must remain eligible. (The newest decision of the pair is out of date: what it replaced is
    not published in its place.)"""
    from test_release import make as make_record
    old = tmp_path / "first-runs"
    make_record(old, "a-auto", "A", None, pipeline=False)
    root = tmp_path / "records"
    directory = stored(root)
    assert released(tmp_path, [old, root], "both")[0]["A"]["source"] == "records/a"
    answer(directory, "p0004", "no")
    pairs, index = released(tmp_path, [old, root], "edited")
    assert pairs == {}
    assert "held back" in {item["source"]: item["why"] for item in index["left_out"]}["first-runs/a-auto"]


# ======================================================= the override names its failures

def test_o_an_override_is_not_reused_for_other_bar_failures(tmp_path, scored, capsys):
    """The saved approval and any override must cover the scores being published, and the
    release must contain the answers those scores used. Coherent legacy and migrated records
    must remain eligible. (An override given for "enough" is not taken for scores that miss
    the median: Store refuses it, and the decision is made again.)"""
    root = tmp_path / "records"
    directory = scored_pair(root)
    answer(directory, "p0001", "no")
    drain(directory)
    decide_on(directory, "bend", page(directory), "19 are enough")
    drain(directory)
    assert decision(directory)["override"]["bar_missed"] == ["enough"]
    record = rec.Record(directory)
    held = record.read(rec.HOLDOUT)
    for patch in held["patches"]:                               # remeasured far off: the median fails
        patch["fine_zyx"] = [patch["centre_zyx"][0] + 3.0, 0.0, 0.0]
    record.refresh_holdout(held["patches"])
    answer(directory, "p0001", rec.LINES_UP)                    # twenty again: "enough" passes
    pipeline = drain(directory)
    assert pipeline.states()["store"] == FAILED and "decide again" in pipeline.stamp("store")["error"]
    stored_fit = record.read(rec.FIT)
    assert stored_fit["scores"]["bend"]["lines_up"] == 19       # the old store, not the new scores
    assert released(tmp_path, [root], "reused")[0] == {}


def test_o_store_refuses_a_decision_whose_override_does_not_name_the_failures(tmp_path, scored):
    """The saved approval and any override must cover the scores being published, and the
    release must contain the answers those scores used. Coherent legacy and migrated records
    must remain eligible. (A decision written without the page, overriding something else.)"""
    root = tmp_path / "records"
    directory = scored_pair(root)
    answer(directory, "p0001", "no")
    drain(directory)
    pick_all(directory)
    rec.Record(directory).decide({"fit": "f001", "publish": "bend", "reason": "x",
                                  "override": {"by": BY, "reason": "x", "bar_missed": ["median"]}}, BY)
    pipeline = drain(directory)
    assert pipeline.states()["store"] == FAILED
    assert released(tmp_path, [root], "wrong")[0] == {}


# ================================================== legacy and migrated records remain eligible

def test_l_a_decision_from_before_decisions_named_their_scores_is_stored(tmp_path, scored, capsys):
    """The saved approval and any override must cover the scores being published, and the
    release must contain the answers those scores used. Coherent legacy and migrated records
    must remain eligible. (A decision written as the earlier server wrote it, on the scores
    there are.)"""
    root = tmp_path / "records"
    directory = scored_pair(root)
    pick_all(directory)
    rec.Record(directory).decide({"fit": "f001", "publish": "bend", "reason": "", "override": None,
                                  "round": "c001"}, BY)
    pipeline = drain(directory)
    assert pipeline.states()["store"] == DONE
    assert "A" in released(tmp_path, [root], "legacy")[0]
    assert check(root, capsys)[0] == 0


def test_l_a_decision_from_before_decisions_named_their_scores_is_not_stored_on_newer_scores(tmp_path, scored):
    """The saved approval and any override must cover the scores being published, and the
    release must contain the answers those scores used. Coherent legacy and migrated records
    must remain eligible. (Such a decision, then an answer changed and scored again: the
    decision did not see the scores there are now.)"""
    root = tmp_path / "records"
    directory = scored_pair(root)
    pick_all(directory)
    record = rec.Record(directory)
    record.log("decision", fit="f001", publish="bend", reason="", override=None, round="c001", by=BY,
               at="2026-01-01T00:00:00+00:00")
    record.refresh_final()
    answer(directory, "p0001", "partly")
    answer(directory, "p0002", rec.LINES_UP)
    answer(directory, "p0001", rec.LINES_UP)                     # 20 again, rescored after the decision
    pipeline = drain(directory)
    assert pipeline.states()["store"] == FAILED and "decide again" in pipeline.stamp("store")["error"]


def test_l_a_record_stored_by_main_stays_eligible_until_an_answer_changes(tmp_path, scored, capsys):
    """The saved approval and any override must cover the scores being published, and the
    release must contain the answers those scores used. Coherent legacy and migrated records
    must remain eligible. (A record main stored: no binding, its scores name their units.)"""
    root = tmp_path / "records"
    directory = stored(root)
    _as_main_stored_it(directory)
    fit = rec.Record(directory).read(rec.FIT)
    fit.pop("schema", None)
    fit.pop("bend", None)
    rec.Record(directory).write(rec.FIT, fit)
    _without_basis(directory)                                   # main's decisions name no scores
    _restamp_store(directory)
    assert "A" in released(tmp_path, [root], "main")[0]
    assert check(root, capsys)[0] == 0
    answer(directory, "p0009", "cant-tell")
    assert released(tmp_path, [root], "edited")[0] == {}
    code, out = check(root, capsys)
    assert code == 1


def test_l_a_record_stored_by_round_five_stays_eligible_until_an_answer_changes(tmp_path, scored, capsys):
    """The saved approval and any override must cover the scores being published, and the
    release must contain the answers those scores used. Coherent legacy and migrated records
    must remain eligible. (A record stored with the fit, variant and transform bound, but not
    yet the answers.)"""
    root = tmp_path / "records"
    directory = stored(root)
    record = rec.Record(directory)
    fit = record.read(rec.FIT)
    fit.pop("schema", None)
    for key in ("answers_sha256", "basis"):
        fit["stored"].pop(key, None)
        fit["scored"].pop(key, None)
    record.write(rec.FIT, fit)
    _without_basis(directory)
    _restamp_store(directory)
    assert "A" in released(tmp_path, [root], "round5")[0]
    assert check(root, capsys)[0] == 0
    answer(directory, "p0009", "no")
    assert released(tmp_path, [root], "edited")[0] == {}
    assert check(root, capsys)[0] == 1


def _restamp_store(directory):
    pipeline = pipeline_stages.pipeline(directory)
    stamp = pipeline.stamp("store")
    pipeline._write_stamp("store", stamp | {"output_digest": engine.digest_files(directory, stamp["outputs"])})


def _migrated(root, name, pair, lines_up, override=None, answers=None):
    """A record as `volreg migrate` writes it: scores with their units, no store binding."""
    record = rec.Record(root / name)
    patches = [{"id": f"p{index:04d}", "centre_zyx": [10.0 * index, 0, 0],
                "fine_zyx": [10.0 * index + 1.0, 0, 0],
                "answer": {"answer": (answers or {}).get(index, rec.LINES_UP if index <= lines_up else "partly"),
                           "at": None, "by": BY}} for index in range(1, 21)]
    record.write(rec.PAIR, {"pair": pair, "coarse_side": "fixed"})
    record.write(rec.HOLDOUT, {"patches": patches})
    transform = {"schema_version": "1.1.0", "fixed_volume": "fixed.zarr",
                 "transformation_matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]}
    held = {"patches": [patch | {"answer": {"answer": rec.LINES_UP}} if index <= lines_up else patch
                        for index, patch in enumerate(patches, 1)]}
    scores = {variant: score(transform, held, True, 2.0, affine_only=variant == "affine")
              for variant in ("bend", "affine")}
    for value in scores.values():
        value["bar"] = bar_check(value)
    record.write(rec.TRANSFORM, transform)
    record.write(rec.FIT, {"pair": pair, "fit": "f002", "fine_um": 2.0, "bar": {"version": "test"},
                           "scores": scores, "code": {"migrated_from_session_fit": "f002"}})
    record.write(rec.FINAL_REVIEW, {"rounds": [], "decision": {
        "fit": "f002", "publish": "affine", "reason": "", "by": BY, "at": None, "override": override}})
    return record


def test_m_migrated_records_stay_eligible_and_checked(tmp_path, capsys):
    """The saved approval and any override must cover the scores being published, and the
    release must contain the answers those scores used. Coherent legacy and migrated records
    must remain eligible. (Migrated: 20 units passing; and none, on an override.)"""
    old = tmp_path / "first-runs"
    _migrated(old, "a", "A", 20)
    _migrated(old, "b", "B", 0, override={"by": BY, "reason": "first runs", "bar_missed": ["enough"]})
    index = release.release(tmp_path / "out", [old])
    assert sorted(entry["pair"] for entry in index["pairs"]) == ["A", "B"]
    code, out = check(old, capsys)
    assert code == 0, out
    assert check(tmp_path / "out", capsys)[0] == 0


def test_m_a_migrated_record_whose_answers_do_not_give_its_scores_is_not_released(tmp_path, capsys):
    """The saved approval and any override must cover the scores being published, and the
    release must contain the answers those scores used. Coherent legacy and migrated records
    must remain eligible. (A migrated record edited by hand after it was scored.)"""
    old = tmp_path / "first-runs"
    record = _migrated(old, "a", "A", 20)
    held = record.read(rec.HOLDOUT)
    held.pop("schema", None)
    held["patches"][4]["answer"]["answer"] = "no"
    record.write(rec.HOLDOUT, held)
    index = release.release(tmp_path / "out", [old])
    assert index["pairs"] == []
    code, out = check(old, capsys)
    assert code == 1 and "NOT WHAT IS STORED" in out


def test_m_a_migrated_override_that_does_not_cover_the_failure_is_not_released(tmp_path, capsys):
    """The saved approval and any override must cover the scores being published, and the
    release must contain the answers those scores used. Coherent legacy and migrated records
    must remain eligible. (A migrated score below the bar with no override for it.)"""
    old = tmp_path / "first-runs"
    _migrated(old, "a", "A", 5)
    index = release.release(tmp_path / "out", [old])
    assert index["pairs"] == []
    assert check(old, capsys)[0] == 1


def test_n_publishing_nothing_is_untouched_by_answer_edits(tmp_path, scored, capsys):
    """The saved approval and any override must cover the scores being published, and the
    release must contain the answers those scores used. Coherent legacy and migrated records
    must remain eligible. (Nothing published: nothing to cover.)"""
    root = tmp_path / "records"
    directory = stored(root, "nothing")
    answer(directory, "p0002", "no")
    pairs, index = released(tmp_path, [root], "nothing")
    assert pairs == {}
    assert check(root, capsys)[0] == 0
