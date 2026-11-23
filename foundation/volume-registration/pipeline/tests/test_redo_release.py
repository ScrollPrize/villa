"""Failed redos, release after a redo, status readers against advancement, jobs from a
temporary uv environment, settings saves from the command line, and record file modes.

Each test opens with the requirement it checks. Windows-only branches run on any system with
the Windows primitive put in their place, as in test_review_blockers.py; on Windows (or under
Wine) the real primitives are used, and a reader that keeps a file open is a real open handle."""

import builtins
import io
import json
import os
import stat
import sys
import time
from pathlib import Path

import pytest

from volreg import cli, engine, filelock, release
from volreg import record as rec
from volreg import stages as pipeline_stages
from volreg.engine import DONE, READY, STALE
from volreg.review.server import Reviews

from test_review_blockers import CONTEXT, REVIEWER, WINDOWS, as_windows_locks, lock_kind, make  # noqa: F401

BY = REVIEWER


# --------------------------------------------------------------- a pair on the real stages

def _fit_id(pipeline):
    path = pipeline.work("fit", "fit.json")
    return json.loads(path.read_text())["fit"] if path.exists() else None


def fake_capture(pipeline, params):
    pipeline.work("capture.json").write_text("{}\n")
    return ["work/capture.json"]


def fake_holdout(pipeline, params):
    pipeline.record.refresh_holdout([{"id": "p0001", "stretch": 0, "origin_zyx": [0, 0, 0]}],
                                    patch_coarse_voxels=128, stretches=5)
    if not pipeline.record.answers().get("p0001"):
        pipeline.record.answer("p0001", rec.LINES_UP, BY)
    pipeline.work("holdout-patches.json").write_text('[["p0001", [0, 0, 0]]]\n')
    return ["work/holdout-patches.json"]


def fake_fit(pipeline, params):
    counter = pipeline.work("fits-made")
    made = int(counter.read_text()) + 1 if counter.exists() else 1
    counter.write_text(str(made))
    fit_id = f"f{made:03d}"
    out = pipeline.work("fit")
    out.mkdir(parents=True, exist_ok=True)
    (out / "transform.json").write_text(json.dumps({
        "schema_version": "1.2.0", "fixed_volume": "fixed.zarr", "fit": fit_id,
        "transformation_matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
        "deformation": {"of": fit_id}}) + "\n")
    (out / "fit.json").write_text(json.dumps({"fit": fit_id, "fine_um": 2.0}) + "\n")
    return ["work/fit/transform.json", "work/fit/fit.json"]


def fake_recheck(pipeline, params):
    pipeline.work("recheck.json").write_text(json.dumps({"under": _fit_id(pipeline)}) + "\n")
    return ["work/recheck.json"]


def _score():
    return {"lines_up": 1, "median_fine_voxels": 1.0, "p90_fine_voxels": 2.0, "far": 0,
            "median_um": 2.0, "p90_um": 4.0, "bar": {"passes": True, "checks": {}}}


def fake_score(pipeline, params):
    pipeline.work("scores.json").write_text(json.dumps({
        "bar": {"version": "test"}, "scores": {"bend": _score(), "affine": _score()},
        "drift": {}, "under": _fit_id(pipeline)}) + "\n")
    return ["work/scores.json"]


def fake_blind(pipeline, params):
    """A one-item round per fit, as run_blind keeps one per fit."""
    fit_id = _fit_id(pipeline)
    candidates = [fit_id, fit_id + "-affine"]
    final = pipeline.record.read(rec.FINAL_REVIEW) if pipeline.record.exists(rec.FINAL_REVIEW) else {}
    rounds = final.get("rounds") or []
    if not any(round_["candidates"] == candidates for round_ in rounds):
        rounds = rounds + [{"id": f"c{len(rounds) + 1:03d}", "candidates": candidates,
                            "items": [{"patch": "p0001", "options": [[candidates[0]], [candidates[1]]],
                                       "choice": None}]}]
        pipeline.record.refresh_final(rounds)
    return [rec.FINAL_REVIEW]


@pytest.fixture
def fake_stages(monkeypatch):
    for name, fake in (("run_capture", fake_capture), ("run_holdout", fake_holdout),
                       ("run_fit", fake_fit), ("run_recheck", fake_recheck),
                       ("run_score", fake_score), ("run_blind", fake_blind)):
        monkeypatch.setattr(pipeline_stages, name, fake)


def drain(directory):
    """Run every ready or stale stage here, in order, until a person is next."""
    for _ in range(50):
        pipeline = pipeline_stages.pipeline(directory)
        states = pipeline.states()
        ready = [name for name in pipeline.order
                 if states[name] in (READY, STALE) and not pipeline.stages[name].human]
        if not ready:
            return pipeline
        pipeline.run_stage(ready[0])
    raise AssertionError("the pipeline does not settle")


def pick_all(directory):
    record = rec.Record(directory)
    final = record.read(rec.FINAL_REVIEW)
    for round_ in final["rounds"]:
        for index, item in enumerate(round_["items"]):
            if item["choice"] is None:
                record.blind_pick(round_["id"], index, 0, BY)


def decide(directory, publish):
    """The decision as the review page makes it, for the fit there is now."""
    pick_all(directory)
    Reviews(directory.parent, BY, auto_advance=False).decide(directory.name, publish, "")


def new_pair(root, name="a", pair="A"):
    directory = root / name
    record = rec.Record(directory)
    record.write(rec.PAIR, {"pair": pair, "coarse_side": "fixed"})
    (directory / "work").mkdir(parents=True, exist_ok=True)
    return directory


def stored_pair(root, name="a", pair="A", publish="bend"):
    """A pair taken through every stage to a stored decision."""
    directory = new_pair(root, name, pair)
    drain(directory)
    decide(directory, publish)
    pipeline = drain(directory)
    assert pipeline.states()["store"] == DONE
    return directory


def released(tmp_path, roots, label):
    index = release.release(tmp_path / f"out-{label}", roots)
    return {entry["pair"]: entry for entry in index["pairs"]}, index


def released_transform(tmp_path, label, pair="A"):
    return json.loads((tmp_path / f"out-{label}" / pair / rec.TRANSFORM).read_text())


# ---------------------------------------------------------------------- 1. a failed redo

def _fail_on_move_of(stage, monkeypatch, directory):
    """Make the move of `stage`'s stamp to the history fail, as a Windows reader holding it
    open past the replace's wait does; with a real reader on Windows."""
    real = filelock.replace
    if WINDOWS:
        handle = open(Path(directory) / "work" / "stages" / f"{stage}.json")
        monkeypatch.setattr(filelock, "RETRY_SECONDS", 0.3)
        return handle.close

    def flaky(source, target):
        if Path(source).name == f"{stage}.json" and Path(target).parent.name == "history":
            raise PermissionError(13, "a reader has it open", str(source))
        return real(source, target)
    monkeypatch.setattr(engine.filelock, "replace", flaky)
    return lambda: monkeypatch.setattr(engine.filelock, "replace", real)


def test_a_redo_that_fails_after_a_stamp_moved_leaves_the_pair_releasable(tmp_path, fake_stages, monkeypatch):
    """An aborted Redo must leave stage state and release eligibility consistent."""
    root = tmp_path / "records"
    directory = stored_pair(root)
    marks = pipeline_stages.pipeline(directory).redo_marks()
    assert "A" in released(tmp_path, [root], "before")[0]
    let_go = _fail_on_move_of("recheck", monkeypatch, directory)   # fit's stamp moves first
    try:
        with pytest.raises(PermissionError):
            pipeline_stages.pipeline(directory).redo("fit")
    finally:
        let_go()
    pipeline = pipeline_stages.pipeline(directory)
    assert all(state == DONE for state in pipeline.states().values())
    assert pipeline.redo_marks() == marks                          # the marker is rolled back too
    assert pipeline.advance(lambda name: None)[1] == []            # nothing to continue
    assert "A" in released(tmp_path, [root], "after")[0]           # and still released
    failed = [event for event in pipeline.record.events() if event["kind"] == "redo-failed"]
    assert len(failed) == 1 and failed[0]["stage"] == "fit" and "error" in failed[0]
    assert not [event for event in pipeline.record.events() if event["kind"] == "redo"]


def test_a_redo_that_fails_on_its_marker_after_every_stamp_moved_is_undone(tmp_path, fake_stages, monkeypatch):
    """An aborted Redo must leave stage state and release eligibility consistent."""
    root = tmp_path / "records"
    directory = stored_pair(root)
    real = filelock.replace

    def flaky(source, target):
        if Path(target).name == "redo.json":
            raise PermissionError(13, "a reader has it open", str(target))
        return real(source, target)
    monkeypatch.setattr(engine.filelock, "replace", flaky)
    with pytest.raises(PermissionError):
        pipeline_stages.pipeline(directory).redo("fit")
    monkeypatch.setattr(engine.filelock, "replace", real)
    pipeline = pipeline_stages.pipeline(directory)
    assert all(state == DONE for state in pipeline.states().values())
    assert pipeline.redo_mark("store") == 0
    assert "A" in released(tmp_path, [root], "after")[0]


def test_after_a_failed_redo_the_same_redo_can_be_made_and_finished(tmp_path, fake_stages, monkeypatch):
    """An aborted Redo must leave stage state and release eligibility consistent."""
    root = tmp_path / "records"
    directory = stored_pair(root)
    let_go = _fail_on_move_of("score", monkeypatch, directory)
    try:
        with pytest.raises(PermissionError):
            pipeline_stages.pipeline(directory).redo("fit")
    finally:
        let_go()
    affected = pipeline_stages.pipeline(directory).redo("fit")
    assert affected == ["fit", "recheck", "score", "blind", "store"]
    assert released(tmp_path, [root], "redone")[0] == {}
    drain(directory)
    decide(directory, "bend")
    drain(directory)
    assert "A" in released(tmp_path, [root], "again")[0]
    assert released_transform(tmp_path, "again")["fit"] == "f002"


# --------------------------------------------------------------- 2. release after a redo

def test_a_store_that_fails_after_a_redo_does_not_release_the_old_transform(tmp_path, fake_stages, monkeypatch):
    """A failed store must not write a stamp under the current Redo marker; the post-Redo
    release check must reject a transform whose fit or publication variant differs from what
    was reviewed."""
    root = tmp_path / "records"
    directory = stored_pair(root)
    pipeline_stages.pipeline(directory).redo("score")      # same fit, scored again, store again
    drain(directory)                                        # store waits on nothing new: the decision stands
    pipeline = pipeline_stages.pipeline(directory)
    assert pipeline.states()["store"] == DONE and "A" in released(tmp_path, [root], "stored")[0]
    pipeline.redo("store")
    real = rec.filelock.replace
    held = None
    if WINDOWS:
        held = open(directory / rec.TRANSFORM)              # the old transform, held open
        monkeypatch.setattr(filelock, "RETRY_SECONDS", 0.3)
    else:
        def flaky(source, target):
            if Path(target) == directory / rec.TRANSFORM:
                raise PermissionError(13, "a reader has it open", str(target))
            return real(source, target)
        monkeypatch.setattr(rec.filelock, "replace", flaky)
    try:
        stamp = pipeline_stages.pipeline(directory).run_stage("store")
    finally:
        if held:
            held.close()
        monkeypatch.setattr(rec.filelock, "replace", real)
    assert stamp["status"] == "failed"
    pipeline = pipeline_stages.pipeline(directory)
    assert pipeline.stamp("store").get("redo_mark") != pipeline.redo_mark("store")
    pairs, index = released(tmp_path, [root], "failed")
    assert pairs == {} and any("store" in item["why"] for item in index["left_out"])


def test_a_decision_changed_after_a_post_redo_store_is_not_released_as_the_old_variant(tmp_path, fake_stages):
    """The post-Redo release check must reject a transform whose fit or publication variant
    differs from what was reviewed."""
    root = tmp_path / "records"
    directory = stored_pair(root)
    pipeline_stages.pipeline(directory).redo("fit")
    drain(directory)
    decide(directory, "bend")
    drain(directory)
    assert "A" in released(tmp_path, [root], "bend")[0]
    assert "deformation" in released_transform(tmp_path, "bend")
    decide(directory, "affine")                             # store is stale now; not run yet
    assert pipeline_stages.pipeline(directory).states()["store"] == STALE
    pairs, _index = released(tmp_path, [root], "stale")
    assert pairs == {}
    drain(directory)
    pairs, _index = released(tmp_path, [root], "affine")
    assert pairs["A"]["decision"]["publish"] == "affine"
    assert "deformation" not in released_transform(tmp_path, "affine")


def test_a_new_fit_is_not_stored_or_released_under_an_older_fits_decision(tmp_path, fake_stages):
    """The post-Redo release check must reject a transform whose fit or publication variant
    differs from what was reviewed. (a) a new fit shipped under an older fit's decision."""
    root = tmp_path / "records"
    directory = stored_pair(root)                           # f001, decided bend
    pipeline_stages.pipeline(directory).redo("fit")
    pipeline = drain(directory)                             # f002 and its blind round, unanswered
    assert _fit_id(pipeline) == "f002"
    stored = pipeline.record.read(rec.FIT) if pipeline.record.exists(rec.FIT) else {}
    assert stored.get("fit") != "f002"
    assert json.loads((directory / rec.TRANSFORM).read_text()).get("fit") != "f002"
    pairs, _index = released(tmp_path, [root], "unreviewed")
    assert pairs == {}
    decide(directory, "bend")                               # the new round answered, f002 decided
    drain(directory)
    pairs, _index = released(tmp_path, [root], "reviewed")
    assert pairs["A"]["decision"]["fit"] == "f002" and released_transform(tmp_path, "reviewed")["fit"] == "f002"


def _two_records(tmp_path):
    root = tmp_path / "records"
    older = stored_pair(root, "a1", "A")
    time.sleep(1.1)                                          # decisions are timed to the second
    newer = stored_pair(root, "a2", "A")
    return root, older, newer


def test_an_unresolved_newest_record_holds_back_the_older_ones(tmp_path, fake_stages):
    """Hold back older outputs while the newest record is unresolved."""
    root, older, newer = _two_records(tmp_path)
    assert released(tmp_path, [root], "both")[0]["A"]["source"] == "records/a2"
    pipeline_stages.pipeline(newer).redo("fit")
    pairs, index = released(tmp_path, [root], "newer-redone")
    assert pairs == {}
    assert {item["source"] for item in index["left_out"]} == {"records/a1", "records/a2"}


def test_redoing_an_older_record_does_not_suppress_a_newer_decision(tmp_path, fake_stages):
    """Hold back older outputs while the newest record is unresolved (and not otherwise:
    main chose the newer record)."""
    root, older, newer = _two_records(tmp_path)
    pipeline_stages.pipeline(older).redo("fit")
    pairs, _index = released(tmp_path, [root], "older-redone")
    assert pairs["A"]["source"] == "records/a2"


def test_a_newest_record_redone_and_decided_but_not_stored_holds_back_the_older(tmp_path, fake_stages):
    """Hold back older outputs while the newest record is unresolved."""
    root, older, newer = _two_records(tmp_path)
    pipeline_stages.pipeline(newer).redo("store")
    (newer / rec.TRANSFORM).unlink()                        # as if store had never run on it
    pairs, _index = released(tmp_path, [root], "no-transform")
    assert pairs == {}


def test_a_record_with_no_redo_history_is_released_as_before(tmp_path, fake_stages):
    """Preserve eligibility for legacy records with no Redo history."""
    from test_release import make as make_record
    root = tmp_path / "records"
    make_record(root, "a", "A", "2026-11-05T12:00:00+00:00", pipeline=True)    # no stamps at all
    directory = stored_pair(root, "b", "B")
    assert not pipeline_stages.pipeline(directory).redo_marks()["stages"]
    pipeline_stages.pipeline(directory)._write_stamp("store", {"status": "failed"})   # main ignores stamps
    pairs, _index = released(tmp_path, [root], "legacy")
    assert set(pairs) == {"A", "B"}


def _fs_calls_under(directory, at_call, action):
    """Run `action` once, at the `at_call`-th file open or stat under `directory`."""
    real_open, real_io_open, real_stat = builtins.open, io.open, os.stat
    state = {"calls": 0, "done": False}
    directory = str(directory)

    def counting(real):
        def wrapped(path, *args, **kwargs):
            if not state["done"] and isinstance(path, (str, os.PathLike)) and \
                    os.fspath(path).startswith(directory):
                state["calls"] += 1
                if state["calls"] == at_call:
                    state["done"] = True
                    restore()
                    try:
                        action()
                    finally:
                        install()
            return real(path, *args, **kwargs)
        return wrapped

    def install():
        builtins.open, io.open, os.stat = counting(real_open), counting(real_io_open), counting(real_stat)

    def restore():
        builtins.open, io.open, os.stat = real_open, real_io_open, real_stat

    return install, restore, state


@pytest.mark.parametrize("redone_first", [False, True])
def test_a_publish_nothing_store_during_a_release_leaves_it_whole_and_consistent(tmp_path, fake_stages, redone_first):
    """Release must be all-or-nothing. (b) a partial release directory left when a `publish
    nothing` store completes during selection. Swept over every file access of the release."""
    for at_call in range(1, 80):
        case = tmp_path / f"case{at_call}"
        root = case / "records"
        directory = stored_pair(root)
        if redone_first:
            pipeline_stages.pipeline(directory).redo("store")
            drain(directory)

        def store_nothing(directory=directory):
            decide(directory, "nothing")
            drain(directory)
        install, restore, state = _fs_calls_under(directory, at_call, store_nothing)
        out = case / "out"
        install()
        try:
            index = release.release(out, [root])
        except Exception:                                    # noqa: BLE001 - allowed, if nothing is left
            index = None
        finally:
            restore()
        if index is None:
            assert not out.exists() and not [p for p in case.iterdir() if p.name != "records"]
        else:
            assert [p.name for p in case.iterdir() if p.name not in ("records", "out")] == []
            for entry in index["pairs"]:
                folder = out / entry["folder"]
                assert {name: release.sha256(folder / name) for name in entry["files"]} == entry["files"]
                transform = json.loads((folder / rec.TRANSFORM).read_text())
                assert entry["decision"]["publish"] == "bend" and "deformation" in transform
                assert json.loads((folder / rec.FINAL_REVIEW).read_text())["decision"]["publish"] == "bend"
        if not state["done"]:
            break                                            # every access has been tried
    assert at_call > 5


def test_a_release_that_fails_part_way_leaves_no_folder(tmp_path, fake_stages, monkeypatch):
    """Release must be all-or-nothing."""
    root = tmp_path / "records"
    stored_pair(root, "a", "A")
    stored_pair(root, "b", "B")
    written = []
    real_copy, real_write = release.shutil.copyfile, Path.write_bytes

    def failing(real):
        def wrapped(*args, **kwargs):
            written.append(args)
            if len(written) == 4:
                raise OSError(28, "No space left on device")
            return real(*args, **kwargs)
        return wrapped
    monkeypatch.setattr(release.shutil, "copyfile", failing(real_copy))
    monkeypatch.setattr(Path, "write_bytes", failing(real_write))
    with pytest.raises(OSError):
        release.release(tmp_path / "out", [root])
    assert sorted(p.name for p in tmp_path.iterdir()) == ["records"]


# ------------------------------------------------------- 3. status readers and advancement

def _paused_reader(directory, held, go, kind):
    """A status read that stops just after it takes the stage's probe, until told to go."""
    if kind == "windows" and not WINDOWS:
        import _pytest.monkeypatch
        as_windows_locks(_pytest.monkeypatch.MonkeyPatch())
    real = filelock.try_lock

    def pausing(handle, *args, **kwargs):
        taken = real(handle, *args, **kwargs)
        if taken and str(getattr(handle, "name", "")).endswith("fit.lock"):
            Path(held).write_text("up")
            deadline = time.time() + 30
            while not Path(go).exists() and time.time() < deadline:
                time.sleep(0.02)
        return taken
    filelock.try_lock = pausing
    make(Path(directory)).states()


def _worker(directory, out, kind):
    if kind == "windows" and not WINDOWS:
        import _pytest.monkeypatch
        as_windows_locks(_pytest.monkeypatch.MonkeyPatch())
    try:
        status = make(Path(directory)).run_stage("fit")["status"]
    except RuntimeError as error:
        status = f"error: {error}"
    Path(out).write_text(status)


def test_a_paused_status_reader_does_not_stop_advance_launching_a_ready_stage(tmp_path, lock_kind):
    """A status read must distinguish a live worker from another reader while preserving
    single ownership and crash recovery."""
    pipeline = make(tmp_path)
    for name in ("draw", "fit"):
        pipeline.run_stage(name)
    pipeline.redo("fit")                                    # ready again; its lock file is there
    held, go, out = tmp_path / "held", tmp_path / "go", tmp_path / "out"
    reader = CONTEXT.Process(target=_paused_reader, args=(str(tmp_path), str(held), str(go), lock_kind))
    reader.start()
    worker = None
    try:
        deadline = time.time() + 30
        while not held.exists() and time.time() < deadline:
            time.sleep(0.02)
        assert held.exists()
        other = make(tmp_path)
        assert other.running("fit") is False and other.state("fit") == READY
        launched = []
        _states, started = other.advance(launched.append)
        assert started == ["fit"] and launched == ["fit"]
        # The job advance launched gets the stage, though the reader has not let go yet.
        worker = CONTEXT.Process(target=_worker, args=(str(tmp_path), str(out), lock_kind))
        worker.start()
        time.sleep(1.0)
        go.write_text("go")
        worker.join(60)
        assert out.read_text() == "ok"
    finally:
        go.write_text("go")
        reader.join(30)
        if worker is not None:
            worker.join(30)
    assert make(tmp_path).state("fit") == DONE


def _hold_stage(directory, held, finish, kind):
    if kind == "windows" and not WINDOWS:
        import _pytest.monkeypatch
        as_windows_locks(_pytest.monkeypatch.MonkeyPatch())
    pipeline = make(Path(directory))
    handle = pipeline._take_lock("fit")
    Path(held).write_text("up")
    deadline = time.time() + 30
    while not Path(finish).exists() and time.time() < deadline:
        time.sleep(0.02)
    pipeline._release_lock(handle)


def test_readers_at_once_see_a_live_worker_and_do_not_take_its_place(tmp_path, lock_kind):
    """A status read must distinguish a live worker from another reader while preserving
    single ownership and crash recovery."""
    pipeline = make(tmp_path)
    pipeline.run_stage("draw")
    held, finish = tmp_path / "held", tmp_path / "finish"
    owner = CONTEXT.Process(target=_hold_stage, args=(str(tmp_path), str(held), str(finish), lock_kind))
    owner.start()
    try:
        while not held.exists():
            time.sleep(0.02)
        readers = [make(tmp_path) for _ in range(3)]
        assert all(reader.running("fit") for reader in readers)
        assert make(tmp_path).advance(lambda name: None)[1] == []
        with pytest.raises(RuntimeError, match="already running"):
            make(tmp_path).run_stage("fit")
    finally:
        finish.write_text("go")
        owner.join(30)
    assert make(tmp_path).running("fit") is False


# ------------------------------------------------------------ 4. jobs from a uv environment

def _uv_build_env(monkeypatch, tmp_path):
    executable = ("C:\\Users\\u\\AppData\\Local\\uv\\cache\\builds-v0\\.tmp1\\Scripts\\python.exe"
                  if WINDOWS else "/home/u/.cache/uv/builds-v0/.tmp1/bin/python")
    monkeypatch.setattr(sys, "executable", executable)
    empty = tmp_path / "no-uv-here"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))                  # uv is not on PATH
    monkeypatch.delenv("UV", raising=False)


def test_a_job_from_a_uv_run_with_uv_off_path_goes_back_through_that_uv(tmp_path, monkeypatch):
    """A launch from a temporary environment must preserve access to its requirements or give
    an actionable error before reporting success."""
    _uv_build_env(monkeypatch, tmp_path)
    uv = tmp_path / "tools" / ("uv.exe" if WINDOWS else "uv")
    uv.parent.mkdir()
    uv.write_text("")
    monkeypatch.setenv("UV", str(uv))                       # what `uv run` tells what it runs
    command = cli.python()
    assert command[0] == str(uv) and command[1] == "run" and command[-1] == "python"
    requirements = command[command.index("--with-requirements") + 1]
    assert Path(requirements).is_file() and Path(requirements).name == "requirements.txt"


def test_a_job_from_a_uv_run_with_no_uv_to_be_found_is_refused_before_anything_starts(tmp_path, monkeypatch, capsys):
    """A launch from a temporary environment must preserve access to its requirements or give
    an actionable error before reporting success."""
    _uv_build_env(monkeypatch, tmp_path)
    directory = new_pair(tmp_path / "records")
    started = []
    monkeypatch.setattr(engine.subprocess, "Popen", lambda command, **kwargs: started.append(command))
    with pytest.raises(SystemExit) as stopped:
        cli.main(["run", str(directory)])
    assert started == []
    message = str(stopped.value.code) + capsys.readouterr().err
    assert "uv" in message and "PATH" in message and "started" not in capsys.readouterr().out
    assert not pipeline_stages.pipeline(directory).log_path("capture").exists()


# ------------------------------------------------------------- 5. settings from the CLI

def _paused_writer(root, pause_file, go_file, action):
    """Run `action` with every write of params.json held back until told to go."""
    real = rec.Record.write

    def pausing(self, name, data):
        if name == pipeline_stages.PARAMS:
            Path(pause_file).write_text("up")
            deadline = time.time() + 30
            while not Path(go_file).exists() and time.time() < deadline:
                time.sleep(0.02)
        return real(self, name, data)
    rec.Record.write = pausing
    if action == "ui-passes":
        Reviews(Path(root), BY, auto_advance=False).set_params("pair-a", "fit", {"passes": 3})
    else:
        cli.main(["more", str(Path(root) / "pair-a"), "--holdout", "20",
                  "--handle", BY["handle"], "--email", BY["email"]])


def _plain(root, action, done):
    if action == "ui-passes":
        Reviews(Path(root), BY, auto_advance=False).set_params("pair-a", "fit", {"passes": 3})
    else:
        cli.main(["more", str(Path(root) / "pair-a"), "--holdout", "20",
                  "--handle", BY["handle"], "--email", BY["email"]])
    Path(done).write_text("done")


@pytest.mark.parametrize("paused, other", [("ui-passes", "cli-more"), ("cli-more", "ui-passes")])
def test_a_cli_more_and_a_ui_save_at_once_both_survive(tmp_path, paused, other):
    """Both successful edits must survive, and the log must describe what was saved."""
    record = rec.Record(tmp_path / "pair-a")
    record.write(rec.PAIR, {"pair": "A-to-B", "coarse_side": "fixed"})
    pause, go, done = tmp_path / "paused", tmp_path / "go", tmp_path / "done"
    first = CONTEXT.Process(target=_paused_writer, args=(str(tmp_path), str(pause), str(go), paused))
    first.start()
    second = None
    try:
        deadline = time.time() + 30
        while not pause.exists() and time.time() < deadline:
            time.sleep(0.02)
        assert pause.exists()
        second = CONTEXT.Process(target=_plain, args=(str(tmp_path), other, str(done)))
        second.start()
        time.sleep(1.5)
        assert not done.exists()                             # it waits for the first save
    finally:
        go.write_text("go")
        first.join(60)
        if second is not None:
            second.join(60)
    assert first.exitcode == 0 and second.exitcode == 0
    saved = record.read(pipeline_stages.PARAMS)
    assert saved["holdout"]["patches"] == 80 and saved["fit"]["passes"] == 3
    more = [event for event in record.events() if event["kind"] == "more"]
    params = [event for event in record.events() if event["kind"] == "params"]
    assert len(more) == 1 and more[0]["saved"] == {"holdout": {"patches": 80}}
    assert len(params) == 1 and params[0]["values"] == {"passes": 3}


def test_cli_more_refuses_a_negative_count(tmp_path):
    """Both successful edits must survive, and the log must describe what was saved."""
    record = rec.Record(tmp_path / "pair-a")
    record.write(rec.PAIR, {"pair": "A-to-B", "coarse_side": "fixed"})
    with pytest.raises(SystemExit):
        cli.main(["more", str(tmp_path / "pair-a"), "--holdout", "-20",
                  "--handle", BY["handle"], "--email", BY["email"]])
    assert not record.exists(pipeline_stages.PARAMS)
    assert not [event for event in record.events() if event["kind"] == "more"]


def test_cli_revisit_keeps_a_round_written_beside_it(tmp_path, monkeypatch):
    """Both successful edits must survive (the other CLI writer of a shared view)."""
    record = rec.Record(tmp_path / "pair-a")
    record.write(rec.PAIR, {"pair": "A-to-B", "coarse_side": "fixed"})
    record.refresh_final([{"id": "c001", "candidates": ["f001", "f001-affine"], "items": []}])
    real_read = rec.Record.read
    raced = {"done": False}

    def read_then_round(self, name):
        data = real_read(self, name)
        if name == rec.FINAL_REVIEW and not raced["done"]:
            raced["done"] = True                             # a blind round is written meanwhile
            import threading
            thread = threading.Thread(target=lambda: rec.Record(self.directory).refresh_final(
                data.get("rounds", []) + [{"id": "c002", "candidates": ["f002", "f002-affine"],
                                           "items": []}]))
            thread.start()
            thread.join(1.0)
        return data
    monkeypatch.setattr(rec.Record, "read", read_then_round)
    cli.main(["revisit", str(tmp_path / "pair-a"), "look again", "--handle", BY["handle"],
              "--email", BY["email"]])
    monkeypatch.setattr(rec.Record, "read", real_read)
    time.sleep(0.2)
    final = record.read(rec.FINAL_REVIEW)
    assert [round_["id"] for round_ in final["rounds"]] == ["c001", "c002"]
    assert [item["text"] for item in final["revisit"]] == ["look again"]


# ------------------------------------------------------------------- 6. record file modes

posix_only = pytest.mark.skipif(WINDOWS, reason="POSIX file modes")


@pytest.fixture
def umask():
    before = os.umask(0o022)
    os.umask(before)

    def set_to(value):
        os.umask(value)
    yield set_to
    os.umask(before)


@posix_only
@pytest.mark.parametrize("mask, mode", [(0o077, 0o600), (0o027, 0o640), (0o022, 0o644), (0o002, 0o664)])
def test_a_new_record_file_honours_the_umask(tmp_path, umask, mask, mode):
    """New records must honor the umask."""
    umask(mask)
    rec.Record(tmp_path).write(rec.HOLDOUT, {"patches": []})
    assert stat.S_IMODE((tmp_path / rec.HOLDOUT).stat().st_mode) == mode


@posix_only
@pytest.mark.parametrize("existing", [0o600, 0o640, 0o644])
def test_a_rewrite_keeps_the_existing_mode_throughout_the_save(tmp_path, umask, monkeypatch, existing):
    """Rewrites must preserve the existing record's privacy throughout the save."""
    umask(0o022)
    record = rec.Record(tmp_path)
    record.write(rec.HOLDOUT, {"patches": []})
    os.chmod(tmp_path / rec.HOLDOUT, existing)
    seen = []
    real_dump = rec.json.dump

    def watching(data, handle, *args, **kwargs):
        seen.append(stat.S_IMODE(os.fstat(handle.fileno()).st_mode))   # before any data
        return real_dump(data, handle, *args, **kwargs)
    monkeypatch.setattr(rec.json, "dump", watching)
    real_replace = rec.filelock.replace
    monkeypatch.setattr(rec.filelock, "replace", lambda source, target: (
        seen.append(stat.S_IMODE(os.stat(source).st_mode)), real_replace(source, target)))
    record.write(rec.HOLDOUT, {"patches": [{"id": "p1"}]})
    assert seen == [existing, existing]
    assert stat.S_IMODE((tmp_path / rec.HOLDOUT).stat().st_mode) == existing
