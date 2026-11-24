"""Round 5 of the review: Store and Release coherence, and decisions bound to the fit and
the completed round the reviewer saw.

Each test opens with the requirement it checks, quoted from the review. Pauses are made
with hooks put into the stage's own process; where the review ran Store in a process of its
own, so do these tests (forked where the system can fork, spawned on Windows, where every
worker sets up the stand-in stages itself). Those that take `lock_kind` run twice off
Windows: with flock, and with the Windows byte lock put in its place (see
test_review_blockers.py); on Windows the real LockFileEx is used."""

import builtins
import json
import os
import shutil
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from volreg import cli, engine, filelock, release
from volreg import record as rec
from volreg import stages as pipeline_stages
from volreg.engine import DONE, FAILED
from volreg.review.server import Reviews, handler_for

from test_review_blockers import CONTEXT, REVIEWER, as_windows_locks, lock_kind  # noqa: F401 - a fixture
import test_redo_release as fakes
from test_redo_release import _fit_id, drain, new_pair, pick_all, released, released_transform

BY = REVIEWER
STORE_COHERENCE = ("Store must use the transform, metadata, and scores for one reviewed fit. Release "
                   "must require a complete matching result and preserve eligibility for coherent "
                   "legacy records.")
DECISION_BINDING = ("A decision must apply to the fit and completed round shown to the reviewer, "
                    "including when a page stays open across Redo.")
WAIT = 30


def install_fakes():
    """The stand-in stages of test_redo_release, set without pytest's monkeypatch: what a
    spawned worker does for itself."""
    for name, fake in (("run_capture", fakes.fake_capture), ("run_holdout", fakes.fake_holdout),
                       ("run_fit", fakes.fake_fit), ("run_recheck", fakes.fake_recheck),
                       ("run_score", fakes.fake_score), ("run_blind", fakes.fake_blind)):
        setattr(pipeline_stages, name, fake)


@pytest.fixture
def stand_ins(monkeypatch):
    for name, fake in (("run_capture", fakes.fake_capture), ("run_holdout", fakes.fake_holdout),
                       ("run_fit", fakes.fake_fit), ("run_recheck", fakes.fake_recheck),
                       ("run_score", fakes.fake_score), ("run_blind", fakes.fake_blind)):
        monkeypatch.setattr(pipeline_stages, name, fake)


# ----------------------------------------------------------------------- the review page

class Shown(tuple):
    """The round and fit a page shows, as a pair, with the `basis` of the scores it shows."""

    def __new__(cls, round_id, fit_id, basis=None):
        shown = super().__new__(cls, (round_id, fit_id))
        shown.basis = basis
        return shown


def page(directory):
    """What the blind page shows: the round's id and, once every item is answered, its fit,
    and the scores it shows (by their basis)."""
    shown = Reviews(directory.parent, BY, auto_advance=False).blind(directory.name)
    return Shown(shown["round"]["id"], shown.get("fit") or (shown["round"].get("candidates") or [None])[0],
                 shown.get("basis"))


def decide_shown(directory, publish, shown, reason=""):
    """A decision sent the way the page sends it: for the round, fit and scores it showed."""
    round_id, fit_id = shown
    reviews = Reviews(directory.parent, BY, auto_advance=False)
    try:
        return reviews.decide(directory.name, publish, reason, round_id=round_id, fit_id=fit_id,
                              basis=getattr(shown, "basis", None))
    except TypeError:                     # a server that takes no round or fit
        return reviews.decide(directory.name, publish, reason)


def decide(directory, publish):
    pick_all(directory)
    return decide_shown(directory, publish, page(directory))


def stored_pair(root, name="a", pair="A", publish="bend"):
    directory = new_pair(root, name, pair)
    drain(directory)
    decide(directory, publish)
    pipeline = drain(directory)
    assert pipeline.states()["store"] == DONE
    return directory


class Server:
    """The review server on a free port, for what goes through HTTP."""

    def __init__(self, root):
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(Reviews(root, BY, auto_advance=False)))
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def post(self, path, body):
        request = urllib.request.Request(f"http://127.0.0.1:{self.httpd.server_port}{path}",
                                         data=json.dumps(body).encode(), method="POST",
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=WAIT) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server(tmp_path):
    root = tmp_path / "records"
    root.mkdir()
    running = Server(root)
    yield running
    running.close()


# ------------------------------------------------------------------- pausing a worker

def _wait_for(path, timeout=WAIT):
    deadline = time.time() + timeout
    while not Path(path).exists():
        if time.time() > deadline:
            raise AssertionError(f"{path} never appeared")
        time.sleep(0.02)


def _hold(held, go):
    Path(held).write_text("held")
    deadline = time.time() + WAIT
    while not Path(go).exists() and time.time() < deadline:
        time.sleep(0.02)


def _store_paused_after_record_write(directory, count, held, go, out):
    """Run Store with a pause after its `count`-th write of a file under the record's fit/."""
    install_fakes()
    directory = Path(directory)
    real, seen = filelock.replace, [0]

    def replace(source, target):
        result = real(source, target)
        if Path(target).parent == directory / "fit":
            seen[0] += 1
            if seen[0] == int(count):
                _hold(held, go)
        return result
    filelock.replace = replace
    stamp = pipeline_stages.pipeline(directory).run_stage("store")
    Path(out).write_text(json.dumps({"status": stamp["status"], "error": stamp.get("error")}))


def _store_paused_after_reading_fit(directory, held, go, out):
    """Run Store with a pause right after it has read the fit's metadata (work/fit/fit.json)."""
    install_fakes()
    target = (Path(directory) / "work" / "fit" / "fit.json").resolve()
    real_text, real_bytes, paused = Path.read_text, Path.read_bytes, [False]

    def after(path):
        if not paused[0] and Path(path).resolve() == target:
            paused[0] = True
            _hold(held, go)

    def read_text(self, *args, **kwargs):
        result = real_text(self, *args, **kwargs)
        after(self)
        return result

    def read_bytes(self):
        result = real_bytes(self)
        after(self)
        return result
    Path.read_text, Path.read_bytes = read_text, read_bytes
    stamp = pipeline_stages.pipeline(directory).run_stage("store")
    Path(out).write_text(json.dumps({"status": stamp["status"], "error": stamp.get("error")}))


def start_worker(target, *args):
    worker = CONTEXT.Process(target=target, args=tuple(str(arg) for arg in args))
    worker.start()
    return worker


def finish(worker, go, out):
    Path(go).write_text("go")
    worker.join(WAIT)
    assert worker.exitcode == 0
    return json.loads(Path(out).read_text())


def assert_release_matches_its_decision(out, pair="A"):
    """Whatever a release holds for a pair is one fit and one variant: the decision's."""
    index = json.loads((out / "index.json").read_text())
    for entry in index["pairs"]:
        if entry["pair"] != pair:
            continue
        decision = entry["decision"]
        fit = json.loads((out / pair / rec.FIT).read_text())
        transform = json.loads((out / pair / rec.TRANSFORM).read_text())
        assert fit["fit"] == decision["fit"]
        assert transform["fit"] == decision["fit"]                 # the stand-in fit names itself
        assert fit.get("under", decision["fit"]) == decision["fit"]   # so do its scores
        assert ("deformation" in transform) == (decision["publish"] == "bend")


# ===================================================== blocker 1: Store and Release coherence

def test_a_new_fit_decided_but_not_yet_stored_is_not_released_with_the_old_fits_files(tmp_path, stand_ins):
    """Store must use the transform, metadata, and scores for one reviewed fit. Release must
    require a complete matching result and preserve eligibility for coherent legacy records.
    (a) `more --fit`, approve f002, release before Store finishes."""
    root = tmp_path / "records"
    directory = stored_pair(root)                               # f001 stored, no redo
    pipeline_stages.more(rec.Record(directory), 0, 5, BY)
    drain(directory)                                            # f002 and its round
    pick_all(directory)
    decide_shown(directory, "bend", page(directory))            # f002 approved; store not run
    assert _fit_id(pipeline_stages.pipeline(directory)) == "f002"
    pairs, index = released(tmp_path, [root], "early")
    assert_release_matches_its_decision(tmp_path / "out-early")
    assert pairs == {} and [item for item in index["left_out"] if item["source"] == "records/a"]
    drain(directory)
    pairs, _index = released(tmp_path, [root], "stored")
    assert pairs["A"]["decision"]["fit"] == "f002"
    assert_release_matches_its_decision(tmp_path / "out-stored")


def test_a_store_paused_between_its_writes_is_not_released_half_done(tmp_path, stand_ins, lock_kind):
    """Store must use the transform, metadata, and scores for one reviewed fit. Release must
    require a complete matching result and preserve eligibility for coherent legacy records.
    (b) a bend-to-affine Store paused between its writes, released meanwhile."""
    root = tmp_path / "records"
    directory = stored_pair(root, publish="bend")
    pick_all(directory)
    decide_shown(directory, "affine", page(directory))
    held, go, out = tmp_path / "held", tmp_path / "go", tmp_path / "out.json"
    worker = start_worker(_store_paused_after_record_write, directory, 1, held, go, out)
    try:
        _wait_for(held)
        pairs, index = released(tmp_path, [root], "paused")
        assert_release_matches_its_decision(tmp_path / "out-paused")
        assert pairs == {}
    finally:
        result = finish(worker, go, out)
    assert result["status"] == "ok"
    pairs, _index = released(tmp_path, [root], "after")
    assert pairs["A"]["decision"]["publish"] == "affine"
    assert "deformation" not in released_transform(tmp_path, "after")


def test_a_store_paused_while_a_new_fit_is_scored_does_not_mix_the_two(tmp_path, stand_ins, lock_kind):
    """Store must use the transform, metadata, and scores for one reviewed fit. Release must
    require a complete matching result and preserve eligibility for coherent legacy records.
    (c) Store paused after reading f001's metadata while a refit from `more --fit` scores f002."""
    root = tmp_path / "records"
    directory = stored_pair(root)
    pipeline_stages.pipeline(directory).redo("store")           # store runs again under f001's decision
    held, go, out = tmp_path / "held", tmp_path / "go", tmp_path / "out.json"
    worker = start_worker(_store_paused_after_reading_fit, directory, held, go, out)
    try:
        _wait_for(held)
        pipeline_stages.more(rec.Record(directory), 0, 5, BY)
        drain(directory)                                         # f002 fitted, rechecked, scored
        assert _fit_id(pipeline_stages.pipeline(directory)) == "f002"
    finally:
        result = finish(worker, go, out)
    assert result["status"] != "ok"                              # nothing stamped as stored
    stored = rec.Record(directory).read(rec.FIT)
    transform = json.loads((directory / rec.TRANSFORM).read_text())
    assert stored["fit"] == transform["fit"] == stored.get("under", stored["fit"])
    released(tmp_path, [root], "mixed")
    assert_release_matches_its_decision(tmp_path / "out-mixed")


def test_a_metadata_write_that_fails_after_the_transform_is_not_released(tmp_path, stand_ins, monkeypatch):
    """Store must use the transform, metadata, and scores for one reviewed fit. Release must
    require a complete matching result and preserve eligibility for coherent legacy records.
    (d) no Redo: the second of Store's writes fails, after the first replaced its file."""
    root = tmp_path / "records"
    directory = stored_pair(root, publish="bend")
    pick_all(directory)
    decide_shown(directory, "affine", page(directory))
    real, seen = filelock.replace, [0]

    def failing(source, target):
        if Path(target).parent == directory / "fit":
            seen[0] += 1
            if seen[0] == 2:
                raise PermissionError(13, "a reader has it open", str(target))
        return real(source, target)
    monkeypatch.setattr(filelock, "replace", failing)
    stamp = pipeline_stages.pipeline(directory).run_stage("store")
    monkeypatch.setattr(filelock, "replace", real)
    assert stamp["status"] == "failed"
    assert not pipeline_stages.pipeline(directory).redo_marks()["stages"]     # no redo anywhere
    pairs, _index = released(tmp_path, [root], "torn")
    assert_release_matches_its_decision(tmp_path / "out-torn")
    assert pairs == {}


def edit_decisions(directory, edit):
    """Rewrite the decisions in a record's event log with `edit(event)`, and final-review.json
    from them, as if they had been made so."""
    record = rec.Record(directory)
    events = [edit(event) if event["kind"] == "decision" else event for event in record.events()]
    record.path(rec.EVENTS).write_text("".join(json.dumps(event) + "\n" for event in events))
    record.refresh_final()


def _without_basis(directory):
    """The decision as a server that named no scores made it."""
    edit_decisions(directory, lambda event: {key: value for key, value in event.items() if key != "basis"})


def _as_main_stored_it(directory):
    """Rewrite a stored pair as main's Store leaves it: fit.json is the fit, its scores and
    the code, nothing else; the store stamp has no redo mark; there is no redo file."""
    record = rec.Record(directory)
    stored = record.read(rec.FIT)
    work_fit = json.loads((directory / "work" / "fit" / "fit.json").read_text())
    scores = json.loads((directory / "work" / "scores.json").read_text())
    scores.pop("scored", None)
    main = work_fit | scores | {"code": stored.get("code")}
    main.pop("schema", None)
    main["bend"] = {"regularisation": 1.0, "effective_parameters": 3.0}   # run_fit's, for a fit with a bend
    record.write(rec.FIT, main)
    _without_basis(directory)                                   # main's decisions name no scores
    pipeline = pipeline_stages.pipeline(directory)
    stamp = pipeline.stamp("store")
    outputs = [rec.TRANSFORM, rec.FIT]
    pipeline._write_stamp("store", {key: value for key, value in stamp.items() if key != "redo_mark"}
                          | {"outputs": outputs, "output_digest": engine.digest_files(directory, outputs)})
    redo = directory / "work" / "stages" / "redo.json"
    if redo.exists():
        redo.unlink()


def test_a_legacy_record_whose_decision_changed_before_store_is_not_released_as_the_old_variant(tmp_path, stand_ins):
    """Store must use the transform, metadata, and scores for one reviewed fit. Release must
    require a complete matching result and preserve eligibility for coherent legacy records.
    (e) a main-created record releases its old bend after the decision changed to affine."""
    root = tmp_path / "records"
    directory = stored_pair(root, publish="bend")
    _as_main_stored_it(directory)
    pairs, _index = released(tmp_path, [root], "legacy")
    assert pairs["A"]["decision"]["publish"] == "bend"           # coherent legacy: still released
    pick_all(directory)
    decide_shown(directory, "affine", page(directory))          # store not run yet
    pairs, index = released(tmp_path, [root], "changed")
    assert_release_matches_its_decision(tmp_path / "out-changed")
    assert pairs == {}
    drain(directory)
    pairs, _index = released(tmp_path, [root], "restored")
    assert pairs["A"]["decision"]["publish"] == "affine"
    assert "deformation" not in released_transform(tmp_path, "restored")


def test_check_verifies_the_binding_not_just_the_score(tmp_path, stand_ins, capsys):
    """Store must use the transform, metadata, and scores for one reviewed fit. Release must
    require a complete matching result and preserve eligibility for coherent legacy records.
    (`volreg check` passed in the one-voxel case because it recomputed the copied transform's
    score: it must check that the parts belong together.)"""
    root = tmp_path / "records"
    directory = stored_pair(root, publish="affine")
    _scorable(directory)
    released(tmp_path, [root], "good")
    out = tmp_path / "out-good"
    assert cli.main(["check", str(out)]) == 0
    tampered = tmp_path / "tampered"
    shutil.copytree(out, tampered)
    transform = json.loads((tampered / "A" / rec.TRANSFORM).read_text())
    transform["fit"] = "f002"                                    # another fit, the same matrix
    (tampered / "A" / rec.TRANSFORM).write_text(json.dumps(transform, indent=1) + "\n")
    capsys.readouterr()
    assert cli.main(["check", str(tampered)]) == 1
    assert "NOT WHAT IS STORED" in capsys.readouterr().out


def _scorable(directory):
    """Give a stored stand-in pair a hold-out patch the scorer can place and stored scores it
    agrees with, so `volreg check` can recompute them (the fake fit's affine is the identity)."""
    from volreg.scoring import bar_check, score
    record = rec.Record(directory)
    held = record.read(rec.HOLDOUT)
    for patch in held["patches"]:
        patch |= {"centre_zyx": [0, 0, 0], "fine_zyx": [0.5, 0, 0]}
    held.pop("schema", None)
    record.write(rec.HOLDOUT, held)
    stored = record.read(rec.FIT)
    stored.pop("schema", None)
    transform = record.read(rec.TRANSFORM)
    for variant in ("bend", "affine"):
        value = score(transform, held, True, 2.0, affine_only=True)
        stored["scores"][variant] = value | {"bar": bar_check(value)}
    stored_override = record.read(rec.FINAL_REVIEW)
    assert stored_override["decision"]["publish"] == "affine"
    # As a store of these scores, on these answers, binds them.
    from volreg.scoring import answers_digest, basis
    stored["stored"]["answers_sha256"] = answers_digest(held)
    stored["stored"]["basis"] = basis(stored["stored"]["fit_digest"], stored["scores"], stored.get("bar"),
                                      stored["stored"]["answers_sha256"])
    record.write(rec.FIT, stored)
    # The store's own record of what it wrote, so the redo check still sees its writes.
    pipeline = pipeline_stages.pipeline(directory)
    stamp = pipeline.stamp("store")
    pipeline._write_stamp("store", stamp | {"output_digest": engine.digest_files(directory, stamp["outputs"])})
    edit_decisions(directory, lambda event: event | {
        "override": {"by": BY, "reason": "too few lines up", "bar_missed": ["enough"]},
        "basis": stored["stored"]["basis"]})


# ============================================================ blocker 2: decision binding

def _decided_then_redone(root):
    directory = stored_pair(root)
    pick_all(directory)
    shown = page(directory)                                     # the page as it was loaded
    assert shown == ("c001", "f001")
    pipeline_stages.pipeline(directory).redo("fit")
    return directory, shown


def test_a_page_left_open_across_redo_cannot_decide_while_the_new_fit_is_made(tmp_path, stand_ins):
    """A decision must apply to the fit and completed round shown to the reviewer, including
    when a page stays open across Redo. (The fit is being made again.)"""
    directory, shown = _decided_then_redone(tmp_path / "records")
    with pytest.raises(ValueError):
        decide_shown(directory, "affine", shown)
    assert rec.Record(directory).read(rec.FINAL_REVIEW)["decision"]["publish"] == "bend"


def test_a_page_left_open_across_redo_cannot_approve_the_new_fit(tmp_path, stand_ins):
    """A decision must apply to the fit and completed round shown to the reviewer, including
    when a page stays open across Redo. (f002 written while f001's round is the latest
    answered one; then f002's round drawn and left unanswered.)"""
    root = tmp_path / "records"
    directory, shown = _decided_then_redone(root)
    pipeline = pipeline_stages.pipeline(directory)
    pipeline.run_stage("fit")                                   # f002 is written; no new round yet
    assert _fit_id(pipeline) == "f002"
    with pytest.raises(ValueError):
        decide_shown(directory, "bend", shown)
    drain(directory)                                            # f002's round, unanswered
    with pytest.raises(ValueError):
        decide_shown(directory, "bend", shown)
    decision = rec.Record(directory).read(rec.FINAL_REVIEW)["decision"]
    assert decision["fit"] == "f001"
    pipeline = drain(directory)
    assert pipeline.states()["store"] != DONE or rec.Record(directory).read(rec.FIT)["fit"] == "f001"
    pairs, _index = released(tmp_path, [root], "unanswered")
    assert "A" not in pairs or pairs["A"]["decision"]["fit"] == "f001"


def test_an_old_round_sent_over_http_is_refused_with_a_conflict(tmp_path, stand_ins, server):
    """A decision must apply to the fit and completed round shown to the reviewer, including
    when a page stays open across Redo. (The old round's id sent through the endpoint after
    the new round is complete.)"""
    root = tmp_path / "records"
    directory = stored_pair(root)
    pipeline_stages.pipeline(directory).redo("fit")
    drain(directory)
    pick_all(directory)                                          # c002, f002's round, complete
    assert page(directory) == ("c002", "f002")
    for body in ({"round": "c001", "fit": "f001"}, {"round": "c001", "fit": "f002"},
                 {"round": "c002", "fit": "f001"}, {}):
        status, reply = server.post("/api/pairs/a/decide", {"publish": "affine", "reason": ""} | body)
        assert status == 409, (body, status, reply)
        assert "reload" in reply["error"]
    assert rec.Record(directory).read(rec.FINAL_REVIEW)["decision"]["fit"] == "f001"
    status, reply = server.post("/api/pairs/a/decide", {"publish": "affine", "reason": "",
                                                        "round": "c002", "fit": "f002",
                                                        "basis": page(directory).basis})
    assert status == 200, reply
    decision = rec.Record(directory).read(rec.FINAL_REVIEW)["decision"]
    assert (decision["fit"], decision["publish"], decision.get("round")) == ("f002", "affine", "c002")


def test_an_unanswered_round_is_refused_over_http(tmp_path, stand_ins, server):
    """A decision must apply to the fit and completed round shown to the reviewer, including
    when a page stays open across Redo. (A round with an unanswered item is not complete.)"""
    directory = new_pair(tmp_path / "records")
    drain(directory)
    scores = pipeline_stages.reviewed_unit(pipeline_stages.pipeline(directory)).basis   # current
    status, reply = server.post("/api/pairs/a/decide", {"publish": "bend", "reason": "",
                                                        "round": "c001", "fit": "f001", "basis": scores})
    assert status == 409 and "round" in reply["error"]
    assert not rec.Record(directory).read(rec.FINAL_REVIEW).get("decision")


def test_the_blind_page_sends_the_round_and_fit_it_shows():
    """A decision must apply to the fit and completed round shown to the reviewer, including
    when a page stays open across Redo. (What the page sends.)"""
    page_source = (Path(pipeline_stages.__file__).parent / "review" / "blind.html").read_text(encoding="utf-8")
    call = page_source[page_source.index("/decide`"):]
    call = call[:call.index("})")]
    assert "round: state.round.id" in call and "fit: state.fit" in call


def test_the_blind_api_names_the_fit_of_a_completed_round(tmp_path, stand_ins):
    """A decision must apply to the fit and completed round shown to the reviewer, including
    when a page stays open across Redo. (The page has the fit to send.)"""
    directory = new_pair(tmp_path / "records")
    drain(directory)
    shown = Reviews(directory.parent, BY, auto_advance=False).blind(directory.name)
    assert shown.get("fit") is None                              # not before the round is done
    pick_all(directory)
    shown = Reviews(directory.parent, BY, auto_advance=False).blind(directory.name)
    assert shown["fit"] == "f001" and shown["round"]["id"] == "c001"


def _redo_in_another_process(directory, done):
    install_fakes()
    pipeline_stages.pipeline(directory).redo("fit")
    Path(done).write_text(str(time.time()))


def test_no_redo_can_come_between_a_decisions_check_and_its_write(tmp_path, stand_ins, monkeypatch, lock_kind):
    """A decision must apply to the fit and completed round shown to the reviewer, including
    when a page stays open across Redo. (The check is made under the lock the write is made
    under: a Redo waits for both.)"""
    root = tmp_path / "records"
    directory = new_pair(root)
    drain(directory)
    pick_all(directory)
    shown = page(directory)
    held, go, done = tmp_path / "held", tmp_path / "go", tmp_path / "done"
    real = rec.Record.log

    def paused(self, kind, **fields):
        if kind == "decision":
            _hold(held, go)
        return real(self, kind, **fields)
    monkeypatch.setattr(rec.Record, "log", paused)
    deciding = threading.Thread(target=decide_shown, args=(directory, "bend", shown))
    deciding.start()
    _wait_for(held)
    redo = start_worker(_redo_in_another_process, directory, done)
    time.sleep(1.0)
    redone_while_deciding = done.exists()
    Path(go).write_text("go")
    deciding.join(WAIT)
    redo.join(WAIT)
    assert not redone_while_deciding
    kinds = [event["kind"] for event in rec.Record(directory).events() if event["kind"] in ("decision", "redo")]
    assert kinds == ["decision", "redo"]


# ============================================== non-blocking: status of a read-only copy

def test_status_reads_a_pipeline_whose_stage_lock_cannot_be_opened_for_writing(tmp_path, stand_ins, monkeypatch, capsys):
    """`volreg status` on a readable pipeline copy with a read-only stage lock lists its
    status, as main does, instead of raising PermissionError."""
    root = tmp_path / "records"
    directory = stored_pair(root)
    real = builtins.open

    def read_only(file, mode="r", *args, **kwargs):
        if str(file).endswith(".lock") and any(flag in mode for flag in "wa+"):
            raise PermissionError(13, "read-only", str(file))
        return real(file, mode, *args, **kwargs)
    monkeypatch.setattr(builtins, "open", read_only)
    cli.main(["status", str(root)])
    assert "finished" in capsys.readouterr().out


# ================================ step C: the interleavings, one test for each reachable one
#
# Refit / Redo / Store (success, a failure at each write, paused) / decision change / stale
# decision / release / check, across records stored by this code and by main. Interleavings
# already covered by a test above or in test_redo_release.py are named there, not repeated.

def _legacy_score(pipeline, params):
    """Scores as score wrote them before it named the fit it scored."""
    pipeline.work("scores.json").write_text(json.dumps({
        "bar": {"version": "test"}, "scores": {"bend": fakes._score(), "affine": fakes._score()},
        "drift": {}, "under": _fit_id(pipeline)}) + "\n")
    return ["work/scores.json"]


def _fail_record_write(monkeypatch, directory, which):
    """Make the `which`-th replace of a file under the record's fit/ fail."""
    real, seen = filelock.replace, [0]

    def failing(source, target):
        if Path(target).parent == directory / "fit":
            seen[0] += 1
            if seen[0] == which:
                raise PermissionError(13, "a reader has it open", str(target))
        return real(source, target)
    monkeypatch.setattr(filelock, "replace", failing)
    return lambda: monkeypatch.setattr(filelock, "replace", real)


def test_c_store_whose_first_write_fails_leaves_the_record_as_it_was(tmp_path, stand_ins, monkeypatch):
    """Store must use the transform, metadata, and scores for one reviewed fit. Release must
    require a complete matching result and preserve eligibility for coherent legacy records.
    (The decision changes; Store's first write fails: nothing it names is released.)"""
    root = tmp_path / "records"
    directory = stored_pair(root, publish="bend")
    before = {name: (directory / name).read_bytes() for name in (rec.FIT, rec.TRANSFORM)}
    pick_all(directory)
    decide_shown(directory, "affine", page(directory))
    put_back = _fail_record_write(monkeypatch, directory, 1)
    assert pipeline_stages.pipeline(directory).run_stage("store")["status"] == "failed"
    put_back()
    assert {name: (directory / name).read_bytes() for name in before} == before
    assert released(tmp_path, [root], "failed")[0] == {}          # bend stored, affine decided
    pipeline_stages.pipeline(directory)._write_stamp("store", {})  # what a redo of store clears
    drain(directory)
    pairs, _index = released(tmp_path, [root], "again")
    assert pairs["A"]["decision"]["publish"] == "affine"
    assert_release_matches_its_decision(tmp_path / "out-again")


def test_c_store_whose_second_write_fails_is_released_once_it_runs_again(tmp_path, stand_ins, monkeypatch):
    """Store must use the transform, metadata, and scores for one reviewed fit. Release must
    require a complete matching result and preserve eligibility for coherent legacy records.
    (The torn record of (d), then Store again.)"""
    root = tmp_path / "records"
    directory = stored_pair(root, publish="bend")
    pick_all(directory)
    decide_shown(directory, "affine", page(directory))
    put_back = _fail_record_write(monkeypatch, directory, 2)
    assert pipeline_stages.pipeline(directory).run_stage("store")["status"] == "failed"
    put_back()
    pairs, index = released(tmp_path, [root], "torn")
    assert pairs == {} and "store it again" in index["left_out"][0]["why"]
    assert cli.main(["check", str(root)]) == 1
    pipeline_stages.pipeline(directory)._write_stamp("store", {})
    drain(directory)
    assert released(tmp_path, [root], "again")[0]["A"]["decision"]["publish"] == "affine"


def test_c_store_paused_after_reading_everything_stores_the_decided_fit_whole(tmp_path, stand_ins):
    """Store must use the transform, metadata, and scores for one reviewed fit. Release must
    require a complete matching result and preserve eligibility for coherent legacy records.
    (Store has read f001 whole -- fit, transform, scores -- when the refit writes f002: it
    stores f001, the decided fit, as one unit.)"""
    root = tmp_path / "records"
    directory = stored_pair(root)
    pipeline_stages.pipeline(directory).redo("store")
    held, go, out = tmp_path / "held", tmp_path / "go", tmp_path / "out.json"
    worker = start_worker(_store_paused_after_reading_scores, directory, held, go, out)
    try:
        _wait_for(held)
        pipeline_stages.more(rec.Record(directory), 0, 5, BY)
        drain(directory)
        assert _fit_id(pipeline_stages.pipeline(directory)) == "f002"
    finally:
        result = finish(worker, go, out)
    assert result["status"] == "ok"
    pairs, _index = released(tmp_path, [root], "whole")
    assert pairs["A"]["decision"]["fit"] == "f001"
    assert_release_matches_its_decision(tmp_path / "out-whole")


def _store_paused_after_reading_scores(directory, held, go, out):
    install_fakes()
    target = (Path(directory) / "work" / "scores.json").resolve()
    real_bytes, real_text, paused = Path.read_bytes, Path.read_text, [False]

    def after(path):
        if not paused[0] and Path(path).resolve() == target:
            paused[0] = True
            _hold(held, go)

    def read_bytes(self):
        result = real_bytes(self)
        after(self)
        return result

    def read_text(self, *args, **kwargs):
        result = real_text(self, *args, **kwargs)
        after(self)
        return result
    Path.read_bytes, Path.read_text = read_bytes, read_text
    stamp = pipeline_stages.pipeline(directory).run_stage("store")
    Path(out).write_text(json.dumps({"status": stamp["status"], "error": stamp.get("error")}))


def test_c_legacy_record_switched_from_affine_to_bend_is_not_released_as_the_affine(tmp_path, stand_ins):
    """Store must use the transform, metadata, and scores for one reviewed fit. Release must
    require a complete matching result and preserve eligibility for coherent legacy records.
    (Main stored the affine of a fit with a bend; the decision is now the bend.)"""
    root = tmp_path / "records"
    directory = stored_pair(root, publish="affine")
    _as_main_stored_it(directory)
    assert released(tmp_path, [root], "legacy")[0]["A"]["decision"]["publish"] == "affine"
    pick_all(directory)
    decide_shown(directory, "bend", page(directory))
    assert released(tmp_path, [root], "switched")[0] == {}
    drain(directory)
    pairs, _index = released(tmp_path, [root], "stored")
    assert pairs["A"]["decision"]["publish"] == "bend"
    assert_release_matches_its_decision(tmp_path / "out-stored")


def test_c_legacy_record_restored_by_a_store_that_fails_half_way_is_not_released(tmp_path, stand_ins, monkeypatch):
    """Store must use the transform, metadata, and scores for one reviewed fit. Release must
    require a complete matching result and preserve eligibility for coherent legacy records.
    (A record main stored; the decision changes; this code's Store fails after its first write.)"""
    root = tmp_path / "records"
    directory = stored_pair(root, publish="bend")
    _as_main_stored_it(directory)
    pick_all(directory)
    decide_shown(directory, "affine", page(directory))
    put_back = _fail_record_write(monkeypatch, directory, 2)
    assert pipeline_stages.pipeline(directory).run_stage("store")["status"] == "failed"
    put_back()
    assert released(tmp_path, [root], "torn")[0] == {}


def test_c_legacy_record_with_another_fits_id_is_not_released(tmp_path, stand_ins):
    """Store must use the transform, metadata, and scores for one reviewed fit. Release must
    require a complete matching result and preserve eligibility for coherent legacy records.
    (Main stored f001; f002 is decided; Store has not run.)"""
    root = tmp_path / "records"
    directory = stored_pair(root)
    _as_main_stored_it(directory)
    pipeline_stages.more(rec.Record(directory), 0, 5, BY)
    drain(directory)
    decide(directory, "bend")
    assert released(tmp_path, [root], "other")[0] == {}
    drain(directory)
    assert released(tmp_path, [root], "stored")[0]["A"]["decision"]["fit"] == "f002"


def test_c_a_decision_from_before_decisions_named_their_round_is_still_stored(tmp_path, stand_ins):
    """Store must use the transform, metadata, and scores for one reviewed fit. Release must
    require a complete matching result and preserve eligibility for coherent legacy records.
    (A decision logged without a round, as earlier versions logged it.)"""
    root = tmp_path / "records"
    directory = new_pair(root)
    drain(directory)
    pick_all(directory)
    rec.Record(directory).decide({"fit": "f001", "publish": "bend", "reason": "", "override": None}, BY)
    pipeline = drain(directory)
    assert pipeline.states()["store"] == DONE
    assert released(tmp_path, [root], "old")[0]["A"]["decision"]["fit"] == "f001"


def test_c_scores_from_before_they_named_their_fit_are_followed_through_their_stamps(tmp_path, stand_ins, monkeypatch):
    """Store must use the transform, metadata, and scores for one reviewed fit. Release must
    require a complete matching result and preserve eligibility for coherent legacy records.
    (Scores without the fit they scored: taken when their stamps rest on the fit there is;
    refused when they rest on another.)"""
    monkeypatch.setattr(pipeline_stages, "run_score", _legacy_score)
    root = tmp_path / "records"
    directory = stored_pair(root)
    assert "scored" not in rec.Record(directory).read(rec.FIT)
    assert released(tmp_path, [root], "chained")[0]["A"]["decision"]["fit"] == "f001"
    pipeline = pipeline_stages.pipeline(directory)
    pipeline.redo("store")
    fakes.fake_fit(pipeline, {})                # f002's files, written over f001's under its stamp
    pipeline._write_stamp("fit", pipeline.stamp_path("fit").exists() and pipeline.stamp("fit") | {
        "output_digest": engine.digest_files(directory, [pipeline_stages.FIT_TRANSFORM, pipeline_stages.FIT_META])})
    with pytest.raises(pipeline_stages.Incoherent, match="not those of f002"):
        pipeline_stages.run_store(pipeline, {})      # as Store reads it, mid-job


def test_c_a_decision_on_scores_made_before_an_answer_changed_is_refused(tmp_path, stand_ins):
    """A decision must apply to the fit and completed round shown to the reviewer, including
    when a page stays open across Redo. (An answer changed since the round was scored: what
    the page shows is not current until it is scored again.)"""
    directory = new_pair(tmp_path / "records")
    drain(directory)
    pick_all(directory)
    shown = page(directory)
    rec.Record(directory).answer("p0001", "partly", BY)
    with pytest.raises(ValueError, match="reload"):
        decide_shown(directory, "bend", shown)
    drain(directory)
    pick_all(directory)
    decide_shown(directory, "affine", page(directory))
    assert rec.Record(directory).read(rec.FINAL_REVIEW)["decision"]["publish"] == "affine"


def test_c_a_round_being_regrown_cannot_be_decided(tmp_path, stand_ins):
    """A decision must apply to the fit and completed round shown to the reviewer, including
    when a page stays open across Redo. (The blind round's size raised: it is redrawn.)"""
    root = tmp_path / "records"
    directory = new_pair(root)
    drain(directory)
    pick_all(directory)
    shown = page(directory)
    Reviews(root, BY, auto_advance=False).set_params("a", "blind", {"count": 30})
    with pytest.raises(ValueError, match="not ready"):
        decide_shown(directory, "bend", shown)


def test_c_a_decided_round_that_grows_waits_for_its_new_items(tmp_path, stand_ins):
    """A decision must apply to the fit and completed round shown to the reviewer, including
    when a page stays open across Redo. (A round grown after the decision: Store does not run
    again until it is complete; the stored result, still the decision's, stays released.)"""
    root = tmp_path / "records"
    directory = stored_pair(root)
    record = rec.Record(directory)
    rounds = record.read(rec.FINAL_REVIEW)["rounds"]
    rounds[-1]["items"].append({"patch": "p0002", "options": rounds[-1]["items"][0]["options"], "choice": None})
    record.refresh_final(rounds)
    pipeline = pipeline_stages.pipeline(directory)
    assert pipeline.states()["final-review"] == engine.WAITING and pipeline.states()["store"] == engine.BLOCKED
    assert released(tmp_path, [root], "grown")[0]["A"]["decision"]["fit"] == "f001"
    pick_all(directory)
    pipeline = drain(directory)
    assert pipeline.states()["store"] == DONE


def _store_paused_after_unlink_or_write(directory, held, go, out):
    """Store with a pause after its first write under the record's fit/ (the bound fit.json)."""
    _store_paused_after_record_write(directory, 1, held, go, out)


@pytest.mark.parametrize("first, then", [("bend", "nothing"), ("nothing", "bend")])
def test_c_a_store_to_or_from_nothing_paused_half_way_is_not_released(tmp_path, stand_ins, first, then):
    """Store must use the transform, metadata, and scores for one reviewed fit. Release must
    require a complete matching result and preserve eligibility for coherent legacy records.
    (Publishing nothing removes the transform after fit.json; publishing again writes it after.)"""
    root = tmp_path / "records"
    directory = stored_pair(root, publish=first)
    pick_all(directory)
    decide_shown(directory, then, page(directory))
    held, go, out = tmp_path / "held", tmp_path / "go", tmp_path / "out.json"
    worker = start_worker(_store_paused_after_unlink_or_write, directory, held, go, out)
    try:
        _wait_for(held)
        pairs, index = released(tmp_path, [root], "half")
        assert pairs == {}
        assert cli.main(["check", str(root)]) == 1
    finally:
        result = finish(worker, go, out)
    assert result["status"] == "ok"
    pairs, _index = released(tmp_path, [root], "done")
    assert ("A" in pairs) == (then == "bend")
    assert_release_matches_its_decision(tmp_path / "out-done")


@pytest.mark.parametrize("part", ["transform", "binding", "scores"])
def test_c_check_refuses_each_part_from_another_result(tmp_path, stand_ins, part, capsys):
    """Store must use the transform, metadata, and scores for one reviewed fit. Release must
    require a complete matching result and preserve eligibility for coherent legacy records.
    (`volreg check` on a release whose transform, binding or scores come from another fit.)"""
    root = tmp_path / "records"
    stored_pair(root, publish="affine")
    released(tmp_path, [root], "good")
    out = tmp_path / "out-good" / "A"
    if part == "transform":
        transform = json.loads((out / rec.TRANSFORM).read_text())
        (out / rec.TRANSFORM).write_text(json.dumps(transform | {"fit": "f002"}, indent=1) + "\n")
    else:
        fit = json.loads((out / rec.FIT).read_text())
        if part == "binding":
            fit["stored"]["fit"] = "f002"
        else:
            fit["scored"]["fit"] = "f002"
        (out / rec.FIT).write_text(json.dumps(fit, indent=1) + "\n")
    capsys.readouterr()
    assert cli.main(["check", str(tmp_path / "out-good")]) == 1
    assert "NOT WHAT IS STORED" in capsys.readouterr().out


def test_c_check_leaves_migrated_records_as_they_were(tmp_path, capsys):
    """Store must use the transform, metadata, and scores for one reviewed fit. Release must
    require a complete matching result and preserve eligibility for coherent legacy records.
    (A migrated record, which no store bound, is checked by its score alone, as before.)"""
    from test_release import make as make_record
    record = make_record(tmp_path / "old", "a", "A", None, pipeline=False)
    assert release.stored_mismatch(record, record.read(rec.FINAL_REVIEW)["decision"], legacy_too=False) is None
    index = release.release(tmp_path / "out", [tmp_path / "old"])
    assert [entry["pair"] for entry in index["pairs"]] == ["A"]


def _flip_and_store(directory, rounds, done):
    """Decide bend, then affine, then bend ..., storing after each, `rounds` times."""
    install_fakes()
    directory = Path(directory)
    for index in range(int(rounds)):
        shown = page(directory)
        decide_shown(directory, ("affine", "bend")[index % 2], shown)
        pipeline_stages.pipeline(directory).run_stage("store")
    Path(done).write_text("done")


def test_c_releases_while_another_process_keeps_restoring_each_match_their_decision(tmp_path, stand_ins, lock_kind):
    """Store must use the transform, metadata, and scores for one reviewed fit. Release must
    require a complete matching result and preserve eligibility for coherent legacy records.
    (Many releases while a process keeps changing the decision and storing it.)"""
    root = tmp_path / "records"
    directory = stored_pair(root, publish="bend")
    pick_all(directory)
    done = tmp_path / "done"
    worker = start_worker(_flip_and_store, directory, 40, done)
    made, shipped = 0, {True: 0, False: 0}
    try:
        while not done.exists() and made < 200:
            label = f"busy{made}"
            try:
                released(tmp_path, [root], label)
            except ValueError:
                pass
            if (tmp_path / f"out-{label}").exists():
                assert_release_matches_its_decision(tmp_path / f"out-{label}")
                index = json.loads((tmp_path / f"out-{label}" / "index.json").read_text())
                shipped[bool(index["pairs"])] += 1
            made += 1
    finally:
        worker.join(120)
    assert worker.exitcode == 0 and made > 0
    print("releases with the pair, without:", shipped[True], shipped[False])
    drain(directory)
    assert_release_matches_its_decision(released(tmp_path, [root], "end") and tmp_path / "out-end")


# ============================= Windows reads and keys: a stamp being replaced, a path's spelling

def _refuse_reads_of(monkeypatch, target, times):
    """Reads of `target` refused `times` times, as Windows refuses a read that lands while
    the file is being replaced."""
    real, refused = Path.read_text, [0]

    def read_text(self, *args, **kwargs):
        if self == target and refused[0] < times():
            refused[0] += 1
            raise PermissionError(13, "the file is being replaced", str(self))
        return real(self, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", read_text)
    return refused


def test_a_stamp_read_while_the_stamp_is_replaced_is_asked_again(tmp_path, stand_ins, monkeypatch):
    """Windows refuses a read of a stamp that lands while its replace is under way. Store and
    the decision's reads ask again, as replace and unlink wait for readers, instead of failing
    or answering 500."""
    as_windows_locks(monkeypatch)
    root = tmp_path / "records"
    directory = stored_pair(root)
    pipeline = pipeline_stages.pipeline(directory)
    limit = [3]
    refused = _refuse_reads_of(monkeypatch, pipeline.stamp_path("fit"), lambda: limit[0])
    assert pipeline_stages._stamp(pipeline, "fit")["status"] == "ok" and refused[0] == 3
    refused[0] = 0
    assert pipeline_stages.fit_unit(pipeline).fit_id == "f001" and refused[0] == 3
    refused[0], limit[0] = 0, 2
    assert pipeline_stages.pipeline(directory).run_stage("store")["status"] == "ok" and refused[0] == 2
    shown = page(directory)               # the page reads no stamp; the decision it sends does
    refused[0], limit[0] = 0, 2
    decide_shown(directory, "affine", shown)
    assert refused[0] == 2 and rec.Record(directory).read(rec.FINAL_REVIEW)["decision"]["publish"] == "affine"


def test_a_stamp_that_stays_unreadable_is_not_hidden(tmp_path, stand_ins, monkeypatch):
    """The wait is for a replace in progress. A permission problem that does not go away
    surfaces after it, as it does for replace and unlink."""
    as_windows_locks(monkeypatch)
    directory = stored_pair(tmp_path / "records")
    pipeline = pipeline_stages.pipeline(directory)
    _refuse_reads_of(monkeypatch, pipeline.stamp_path("fit"), lambda: 10 ** 9)
    monkeypatch.setattr(filelock, "RETRY_SECONDS", 0.2)
    with pytest.raises(PermissionError):
        pipeline_stages._stamp(pipeline, "fit")


@pytest.mark.parametrize("spelling", ["symlink", "case"])
def test_the_record_lock_is_one_lock_however_the_records_path_is_spelled(tmp_path, monkeypatch, spelling):
    """The thread that holds a record's lock holds it for every spelling of the record's
    path (a link to it, the other case on Windows): taking it again through another one is
    the same hold, not a second lock on the same file, which Windows would refuse for ever."""
    directory = tmp_path / "Record"
    directory.mkdir()
    if spelling == "symlink":
        link = tmp_path / "link"
        try:
            os.symlink(directory, link, target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("this system cannot make a symbolic link")
        other = link
    else:
        if os.name != "nt":
            pytest.skip("paths differ by case on Windows only")
        other = Path(str(directory).swapcase())
    taken, real = [], filelock.lock_exclusive
    monkeypatch.setattr(filelock, "lock_exclusive", lambda handle: (taken.append(1), real(handle))[1])
    first, second = rec.Record(directory), rec.Record(other)
    with first.lock():
        with second.lock():
            with first.lock():
                pass
    assert len(taken) == 1
    with second.lock():
        pass
    assert len(taken) == 2
