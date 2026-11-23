"""Redo against queued jobs, background launch, stale-lock recovery, settings saves, volume
names. Windows-only branches are run on any system by putting the Windows primitive in their
place (LockFileEx over POSIX byte locks, `windows()` answering yes)."""

import errno
import json
import multiprocessing
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from volreg import cli, engine, filelock, release
from volreg import record as rec
from volreg import stages as pipeline_stages
from volreg.cores.transform_io import volume_name
from volreg.engine import DONE, Pipeline, Stage, Superseded
from volreg.review.server import Reviews

# Fork where there is one; Windows spawns, which the workers below allow for (module-level
# targets, picklable arguments, everything they need passed in or built there).
CONTEXT = multiprocessing.get_context(
    "fork" if "fork" in multiprocessing.get_all_start_methods() else "spawn")
WINDOWS = os.name == "nt"
REVIEWER = {"handle": "test", "email": "test@example.org"}


# ------------------------------------------------------------------- a Windows byte lock

def fake_windows_lock(handle, offset, shared):
    """LockFileEx on one byte, refused at once, over a POSIX record lock on that byte.
    Process-wide, as Windows byte locks are per handle but contend the same way between
    processes."""
    import fcntl
    try:
        fcntl.lockf(handle.fileno(), (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB,
                    1, offset)
        return True
    except OSError as exc:
        if exc.errno in (errno.EAGAIN, errno.EACCES):
            return False
        raise


def fake_windows_unlock(handle, offset):
    import fcntl
    fcntl.lockf(handle.fileno(), fcntl.LOCK_UN, 1, offset)


def as_windows_locks(monkeypatch):
    if WINDOWS:
        return      # the real LockFileEx, process_token and pid_alive
    monkeypatch.setattr(filelock, "fcntl", None)
    monkeypatch.setattr(filelock, "_windows_lock", fake_windows_lock)
    monkeypatch.setattr(filelock, "_windows_unlock", fake_windows_unlock)
    monkeypatch.setattr(filelock, "process_token", lambda pid=None: 4242 if pid is None else None)
    monkeypatch.setattr(filelock, "pid_alive", lambda pid, token=None: pid == os.getpid())


# ------------------------------------------------------------------------------ the pipeline

def write(pipeline, name, value):
    path = pipeline.directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return name


def make(tmp_path, hooks=None):
    """draw -> fit -> score, with `hooks[stage](pipeline)` called as a stage's job runs."""
    hooks = hooks or {}

    def job(name):
        def run(pipeline, params):
            hooks.get(name, lambda pipeline: None)(pipeline)
            return [write(pipeline, f"work/{name}.json", {"ran": time.time()})]
        return run

    return Pipeline(tmp_path, [
        Stage("draw", "draw", run=job("draw")),
        Stage("fit", "fit", needs=("draw",), run=job("fit")),
        Stage("score", "score", needs=("fit",), run=job("score")),
    ])


# ---------------------------------------------------------------- 1. redo against queued jobs

def test_a_job_chosen_before_a_redo_does_not_run_after_it(tmp_path):
    ran = []
    pipeline = make(tmp_path, {"score": lambda p: ran.append("score")})
    for name in ("draw", "fit"):
        pipeline.run_stage(name)
    assert pipeline.states()["score"] == engine.READY      # chosen here ...
    make(tmp_path).redo("fit")                              # ... and a redo comes first
    stamp = pipeline.run_stage("score")
    assert stamp["status"] == "superseded" and ran == []
    assert not (tmp_path / "work/score.json").exists() and pipeline.stamp("score") is None
    assert not pipeline.lock_path("score").read_text()     # the lock is let go of


def test_a_redo_during_a_job_stops_its_result_being_stamped(tmp_path):
    pipeline = make(tmp_path, {"score": lambda p: make(tmp_path).redo("fit")})
    for name in ("draw", "fit"):
        pipeline.run_stage(name)
    assert pipeline.run_stage("score")["status"] == "superseded"
    assert pipeline.stamp("score") is None
    assert pipeline.states()["fit"] == engine.READY


def test_a_job_that_publishes_does_so_only_if_no_redo_came_between(tmp_path):
    def publish(pipeline):
        make(tmp_path).redo("draw")
        with pipeline.publishing("score"):
            write(pipeline, "record/transform.json", {"old": True})
    pipeline = make(tmp_path, {"score": publish})
    for name in ("draw", "fit"):
        pipeline.run_stage(name)
    assert pipeline.run_stage("score")["status"] == "superseded"
    assert not (tmp_path / "record/transform.json").exists()


def test_a_redo_elsewhere_in_the_pair_does_not_stop_a_job(tmp_path):
    pipeline = make(tmp_path)
    pipeline.run_stage("draw")
    assert pipeline.run_stage("fit")["status"] == "ok"
    pipeline.redo("score")                                  # nothing fit rests on
    assert pipeline.run_stage("score")["status"] == "ok"
    assert pipeline.stamp("score")["redo_mark"] == 1


def _slow_job(directory, started, go):
    def wait(pipeline):
        Path(started).write_text("up")
        deadline = time.time() + 20
        while not Path(go).exists() and time.time() < deadline:
            time.sleep(0.02)

    pipeline = make(Path(directory), {"score": wait})
    sys.exit(0 if pipeline.run_stage("score")["status"] == "superseded" else 1)


def test_a_worker_process_gives_up_when_a_redo_ran_while_it_was_busy(tmp_path):
    pair = tmp_path / "pair"
    pipeline = make(pair)
    for name in ("draw", "fit"):
        pipeline.run_stage(name)
    started, go = tmp_path / "started", tmp_path / "go"
    worker = CONTEXT.Process(target=_slow_job, args=(str(pair), str(started), str(go)))
    worker.start()
    while not started.exists():
        time.sleep(0.02)
    pipeline.redo("fit")
    go.write_text("go")
    worker.join(30)
    assert worker.exitcode == 0
    assert pipeline.stamp("score") is None


def test_the_rollback_of_a_failed_redo_keeps_a_newer_stamp(tmp_path, monkeypatch):
    pipeline = make(tmp_path)
    for name in ("draw", "fit", "score"):
        pipeline.run_stage(name)
    real, moves = filelock.replace, []

    def flaky(source, target):
        moves.append(target)
        if len(moves) == 1:
            return real(source, target)         # draw's stamp goes to history
        # fit's move times out; meanwhile draw finished again, with a newer stamp
        pipeline.stamp_path("draw").write_text(json.dumps({"status": "ok", "newer": True}))
        raise PermissionError("a reader has it open")

    monkeypatch.setattr(engine.filelock, "replace", flaky)
    with pytest.raises(PermissionError):
        pipeline.redo("draw")
    monkeypatch.setattr(engine.filelock, "replace", real)
    assert pipeline.stamp("draw") == {"status": "ok", "newer": True}
    assert pipeline.stamp("fit")["status"] == "ok"


def test_the_rollback_still_restores_what_nothing_replaced(tmp_path, monkeypatch):
    pipeline = make(tmp_path)
    for name in ("draw", "fit"):
        pipeline.run_stage(name)
    before = pipeline.stamp("draw")
    real, moves = filelock.replace, []

    def flaky(source, target):
        moves.append(target)
        if len(moves) == 2:
            raise PermissionError("a reader has it open")
        return real(source, target)

    monkeypatch.setattr(engine.filelock, "replace", flaky)
    with pytest.raises(PermissionError):
        pipeline.redo("draw")
    assert pipeline.stamp("draw") == before


def test_release_leaves_out_a_transform_that_predates_a_redo(tmp_path):
    from test_release import make as make_record
    root = tmp_path / "records"
    record = make_record(root, "a", "A", "2026-11-05T12:00:00+00:00", pipeline=True)
    assert release.release(tmp_path / "out1", [root])["pairs"][0]["pair"] == "A"
    pipeline = pipeline_stages.pipeline(record.directory)
    pipeline.redo("fit")
    index = release.release(tmp_path / "out2", [root])
    assert index["pairs"] == []
    assert "a redo came after" in index["left_out"][0]["why"]
    # Stored again after the redo, as run_store writes it (under its decision), and stamped
    # as run_stage stamps it: released again.
    record.write(rec.FIT, record.read(rec.FIT) | {"fit": "f002", "decision": {"fit": "f002", "publish": "bend"}})
    outputs = [rec.TRANSFORM, rec.FIT]
    pipeline._write_stamp("store", {"status": "ok", "redo_mark": pipeline.redo_mark("store"),
                                    "outputs": outputs,
                                    "output_digest": engine.digest_files(record.directory, outputs)})
    assert release.release(tmp_path / "out3", [root])["pairs"][0]["pair"] == "A"
    # A stamp from a failed store, or one that does not match the files, is not enough.
    pipeline._write_stamp("store", {"status": "failed", "failed_under_redo_mark": pipeline.redo_mark("store")})
    assert release.release(tmp_path / "out4", [root])["pairs"] == []


# ------------------------------------------------------------------------ 2. background launch

@pytest.mark.parametrize("path, temporary", [
    ("/home/u/.cache/uv/builds-v0/.tmpAbC/bin/python", True),
    ("C:\\Users\\u\\AppData\\Local\\uv\\cache\\builds-v0\\.tmpAbC\\Scripts\\python.exe", True),
    ("C:/Users/u/AppData/Local/uv/cache/builds-v1/.tmpAbC/Scripts/python.exe", True),
    ("C:\\Python312\\python.exe", False),
    ("/usr/bin/python3", False),
    ("/home/u/builds-v0-notes/python", False)])
def test_a_uv_build_environment_is_found_with_either_separator(path, temporary):
    assert cli.in_uv_build_env(path) is temporary


def test_a_job_from_a_temporary_uv_environment_relaunches_through_uv(monkeypatch):
    monkeypatch.delenv("UV", raising=False)
    monkeypatch.setattr(sys, "executable", "C:\\x\\uv\\cache\\builds-v0\\.tmp1\\Scripts\\python.exe")
    monkeypatch.setattr(cli.shutil, "which", lambda name: "C:\\tools\\uv.exe")
    command = cli.python()
    assert command[:2] == ["C:\\tools\\uv.exe", "run"] and command[-1] == "python"
    assert "--with-requirements" in command and "builds-v" not in " ".join(command)


def test_without_uv_a_temporary_environment_is_refused_not_run_without_its_requirements(monkeypatch):
    """The base interpreter has none of the requirements: no fallback to it."""
    monkeypatch.delenv("UV", raising=False)
    monkeypatch.setattr(sys, "executable", "/home/u/.cache/uv/builds-v0/.t/bin/python")
    monkeypatch.setattr(sys, "_base_executable", "/usr/bin/python3.12", raising=False)
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    with pytest.raises(cli.CannotLaunch, match="uv"):
        cli.python()


def test_an_ordinary_interpreter_is_used_as_it_is(monkeypatch):
    monkeypatch.setattr(sys, "executable", "/usr/bin/python3")
    assert cli.python() == ["/usr/bin/python3"]


def test_posix_launch_keeps_its_own_session(monkeypatch):
    monkeypatch.setattr(engine, "windows", lambda: False)
    assert engine.detach_options() == {"start_new_session": True}


def test_windows_launch_has_a_group_and_a_console_of_its_own_not_a_session(monkeypatch):
    monkeypatch.setattr(engine, "windows", lambda: True)
    options = engine.detach_options()
    assert "start_new_session" not in options
    assert options["creationflags"] & 0x200 and options["creationflags"] & 0x08000000


def test_start_passes_the_options_to_popen(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(engine, "windows", lambda: True)
    monkeypatch.setattr(engine.subprocess, "Popen", lambda command, **kwargs: seen.update(kwargs))
    make(tmp_path).start("draw", ["python", "-c", "pass"])
    assert "creationflags" in seen and "start_new_session" not in seen


# ----------------------------------------------------------------------- 3. stale-lock recovery

def _hold_lock(directory, held, release_file, die):
    pipeline = make(Path(directory))
    handle = pipeline._take_lock("fit")
    Path(held).write_text("up")
    if die:
        os._exit(1)            # a crash: nothing is cleaned up
    while not Path(release_file).exists():
        time.sleep(0.02)
    pipeline._release_lock(handle)


@pytest.fixture(params=["posix", "windows"])
def lock_kind(request, monkeypatch):
    if request.param == "posix" and WINDOWS:
        pytest.skip("flock is POSIX only; the Windows byte lock is the real lock here")
    if request.param == "windows":
        as_windows_locks(monkeypatch)
    return request.param


def _hold_empty(path, held, finish):
    """A worker caught between taking its lock and writing its pid, for as long as asked."""
    with open(path, "w+b") as owner:
        assert filelock.try_lock(owner, filelock.OWNER_OFFSET)
        Path(held).write_text("up")
        while not Path(finish).exists():
            time.sleep(0.02)


def test_an_empty_lock_file_is_not_stale_however_old(tmp_path, lock_kind):
    pipeline = make(tmp_path)
    lock = pipeline.lock_path("fit")
    lock.parent.mkdir(parents=True)
    held, finish = tmp_path / "held", tmp_path / "finish"
    worker = CONTEXT.Process(target=_hold_empty, args=(str(lock), str(held), str(finish)))
    worker.start()
    while not held.exists():
        time.sleep(0.02)
    try:
        os.utime(lock, (time.time() - 3600, time.time() - 3600))
        assert lock.read_text() == "" and pipeline.running("fit") is True
        with pytest.raises(RuntimeError, match="already running"):
            pipeline.run_stage("fit")
    finally:
        finish.write_text("done")
        worker.join(20)
    assert pipeline.running("fit") is False         # let go: nobody owns it


def test_a_worker_that_died_leaves_a_lock_that_is_taken_over(tmp_path, lock_kind):
    pipeline = make(tmp_path)
    pipeline.run_stage("draw")
    held = tmp_path / "held"
    worker = CONTEXT.Process(target=_hold_lock, args=(str(tmp_path), str(held), "", True))
    worker.start(); worker.join(20)
    assert held.exists() and pipeline.lock_path("fit").read_text().strip()
    assert pipeline.running("fit") is False and pipeline.state("fit") == engine.READY
    assert pipeline.run_stage("fit")["status"] == "ok"


def test_a_live_owner_keeps_its_lock_against_other_processes(tmp_path, lock_kind):
    pipeline = make(tmp_path)
    pipeline.run_stage("draw")
    held, finish = tmp_path / "held", tmp_path / "finish"
    worker = CONTEXT.Process(target=_hold_lock, args=(str(tmp_path), str(held), str(finish), False))
    worker.start()
    while not held.exists():
        time.sleep(0.02)
    try:
        assert pipeline.running("fit") is True and pipeline.state("fit") == engine.RUNNING
        with pytest.raises(RuntimeError, match="already running"):
            pipeline.run_stage("fit")
        assert pipeline.lock_path("fit").read_text().split()[0] == str(worker.pid)
    finally:
        finish.write_text("done")
        worker.join(20)
    assert pipeline.running("fit") is False


def _contend(directory, counter, barrier_file):
    pipeline = make(Path(directory))
    while not Path(barrier_file).exists():
        time.sleep(0.005)
    try:
        handle = pipeline._take_lock("fit")
    except RuntimeError:
        os._exit(0)
    # Owner for a while; the count of owners at once is the file's length.
    with open(counter, "ab") as out:
        out.write(b"x")
    time.sleep(2)               # longer than a contender waits for the lock
    pipeline._release_lock(handle)
    os._exit(0)


def test_only_one_of_many_contenders_takes_a_lock_file(tmp_path, lock_kind):
    make(tmp_path)
    counter, barrier = tmp_path / "owners", tmp_path / "go"
    counter.write_bytes(b"")
    workers = [CONTEXT.Process(target=_contend, args=(str(tmp_path), str(counter), str(barrier)))
               for _ in range(6)]
    for worker in workers:
        worker.start()
    barrier.write_text("go")
    for worker in workers:
        worker.join(30)
    assert counter.read_bytes() == b"x"


def test_a_legacy_pid_only_lock_is_read_as_before(tmp_path):
    pipeline = make(tmp_path)
    lock = pipeline.lock_path("fit")
    lock.parent.mkdir(parents=True)
    lock.write_text(f"{os.getpid()}\n")                 # an older job, alive
    assert pipeline.running("fit") is True
    with pytest.raises(RuntimeError, match="already running"):
        pipeline.run_stage("fit")
    lock.write_text("999999999\n")                      # an older job, dead
    assert pipeline.running("fit") is False
    pipeline.run_stage("draw")
    assert pipeline.run_stage("fit")["status"] == "ok"
    lock.write_text("not a pid\n")
    assert pipeline.running("fit") is False


def test_the_windows_lock_sits_clear_of_the_pid_so_it_can_be_read(tmp_path, monkeypatch):
    as_windows_locks(monkeypatch)
    pipeline = make(tmp_path)
    handle = pipeline._take_lock("fit")
    try:
        assert pipeline.lock_path("fit").read_text().split()[0] == str(os.getpid())
        assert filelock.OWNER_OFFSET > 1024
    finally:
        pipeline._release_lock(handle)


# --------------------------------------------------------------------------- 4. settings saves

@pytest.fixture
def reviews(tmp_path):
    record = rec.Record(tmp_path / "pair-a")
    record.write(rec.PAIR, {"pair": "A-to-B", "coarse_side": "fixed"})
    return Reviews(tmp_path, REVIEWER, auto_advance=False), record


def _save(root, stage, values, out):
    reviews = Reviews(Path(root), REVIEWER, auto_advance=False)
    try:
        reviews.set_params("pair-a", stage, values)
        outcome = "ok"
    except ValueError as error:
        outcome = f"refused: {error}"
    Path(out).write_text(outcome)


def test_overlapping_saves_of_different_settings_both_stay(tmp_path, reviews):
    _reviews, record = reviews
    jobs = []
    for index in range(8):
        stage, values = (("holdout", {"patches": 61 + index}) if index % 2 == 0
                         else ("fit", {"patches": 201 + index}))
        jobs.append(CONTEXT.Process(target=_save, args=(str(tmp_path), stage, values,
                                                     str(tmp_path / f"out{index}"))))
    for job in jobs:
        job.start()
    for job in jobs:
        job.join(60)
    saved = record.read("params.json")
    outcomes = [(tmp_path / f"out{index}").read_text() for index in range(8)]
    events = [event for event in record.events() if event["kind"] == "params"]
    # A smaller count that comes after a larger one is refused; every other save is kept,
    # logged once, and the disk holds the largest of each stage's.
    assert all(outcome == "ok" or "can only grow" in outcome for outcome in outcomes)
    assert len(events) == outcomes.count("ok")
    for stage, values in (("holdout", range(61, 69, 2)), ("fit", range(202, 209, 2))):
        logged = [event["values"]["patches"] for event in events if event["stage"] == stage]
        assert logged == sorted(logged) and saved[stage]["patches"] == logged[-1] == max(values)
    assert not list((tmp_path / "pair-a").glob("params.json.*.tmp"))


def test_the_last_value_saved_is_the_one_logged_and_a_smaller_one_is_refused(tmp_path, reviews):
    """100 then 80: grow-only is judged against what is on disk when the second one gets its
    turn, whichever turn that is."""
    _reviews, record = reviews
    jobs = [CONTEXT.Process(target=_save, args=(str(tmp_path), "holdout", {"patches": count},
                                             str(tmp_path / f"out{count}"))) for count in (100, 80)]
    for job in jobs:
        job.start()
    for job in jobs:
        job.join(60)
    saved = record.read("params.json")["holdout"]["patches"]
    logged = [event["values"]["patches"] for event in record.events() if event["kind"] == "params"]
    outcomes = {count: (tmp_path / f"out{count}").read_text() for count in (100, 80)}
    assert outcomes[100] == "ok" and saved == 100
    if outcomes[80] == "ok":                  # it came first, when 80 still grew from 60
        assert logged == [80, 100]
    else:
        assert "can only grow" in outcomes[80] and logged == [100]
    assert logged[-1] == saved


def test_a_save_below_the_value_on_disk_is_refused(reviews):
    handle, record = reviews
    handle.set_params("pair-a", "holdout", {"patches": 100})
    with pytest.raises(ValueError, match="can only grow"):
        handle.set_params("pair-a", "holdout", {"patches": 80})
    assert record.read("params.json")["holdout"]["patches"] == 100


def test_writers_use_scratch_files_of_their_own(tmp_path, monkeypatch):
    seen = []
    real = filelock.replace
    monkeypatch.setattr(rec.filelock, "replace", lambda source, target: (seen.append(Path(source).name),
                                                                         real(source, target)))
    record = rec.Record(tmp_path)
    record.write("a.json", {"n": 1})
    record.write("a.json", {"n": 2})
    assert len(set(seen)) == 2 and all(name.startswith("a.json.") for name in seen)
    assert record.read("a.json")["n"] == 2 and not list(tmp_path.glob("*.tmp"))


def test_a_failed_write_leaves_no_scratch_file(tmp_path, monkeypatch):
    monkeypatch.setattr(rec.filelock, "replace", lambda source, target: (_ for _ in ()).throw(PermissionError()))
    with pytest.raises(PermissionError):
        rec.Record(tmp_path).write("a.json", {"n": 1})
    assert not list(tmp_path.glob("*.tmp"))


def test_a_save_under_the_windows_locks_and_a_reader_retry(tmp_path, reviews, monkeypatch):
    handle, record = reviews
    as_windows_locks(monkeypatch)
    handle.set_params("pair-a", "holdout", {"patches": 70})
    assert record.read("params.json")["holdout"]["patches"] == 70


# ------------------------------------------------------------------------- 5. the volume name

@pytest.mark.parametrize("location, name", [
    ("https://dl.example.org/data/coarse.zarr", "coarse.zarr"),
    ("https://dl.example.org/data/coarse.zarr/", "coarse.zarr"),
    ("s3://bucket/some/path/coarse.zarr", "coarse.zarr"),
    ("/data/volumes/coarse.zarr", "coarse.zarr"),
    ("/data/volumes/coarse.zarr/", "coarse.zarr"),
    ("relative/coarse.zarr", "coarse.zarr"),
    ("coarse.zarr", "coarse.zarr"),
    ("file:///data/coarse.zarr", "coarse.zarr"),
    ("C:\\data\\volumes\\coarse.zarr", "coarse.zarr"),
    ("C:\\data\\volumes\\coarse.zarr\\", "coarse.zarr"),
    ("C:/data/volumes/coarse.zarr", "coarse.zarr"),
    ("c:\\data/volumes\\coarse.zarr", "coarse.zarr"),
    ("\\\\server\\share\\coarse.zarr", "coarse.zarr"),
    ("data\\coarse.zarr", "coarse.zarr")])
def test_volume_name(location, name):
    assert volume_name(location) == name


@pytest.mark.parametrize("location", [
    "https://dl.example.org/data/coarse.zarr", "https://x.org/a/b/", "/data/volumes/coarse.zarr/",
    "s3://b/a\\b/c.zarr", "/odd/dir\\name/coarse.zarr", "file:///C:/data/coarse.zarr"])
def test_urls_and_posix_paths_come_out_as_they_did(location):
    assert volume_name(location) == location.rstrip("/").rsplit("/", 1)[-1]


def test_store_names_a_windows_volume_by_its_zarr(tmp_path):
    work = tmp_path / "pair" / "work"
    (work / "fit").mkdir(parents=True)
    record = rec.Record(tmp_path / "pair")
    record.write(rec.FINAL_REVIEW, {"rounds": [], "decision": {
        "fit": "f001", "publish": "bend", "reason": "", "override": None, "by": {}, "at": "t"}})
    (work / "fit" / "fit.json").write_text(json.dumps({"fit": "f001"}))
    (work / "scores.json").write_text(json.dumps({"scores": {}}))
    (work / "fit" / "transform.json").write_text(json.dumps({
        "fixed_volume": "C:\\scans\\coarse.zarr", "transformation_matrix": [[1, 0, 0, 0]] * 3}))
    pipeline = pipeline_stages.pipeline(tmp_path / "pair")
    outputs = [pipeline_stages.FIT_TRANSFORM, pipeline_stages.FIT_META]     # as the fit's job stamps them
    pipeline._write_stamp("fit", {"status": "ok", "outputs": outputs,
                                  "output_digest": engine.digest_files(pipeline.directory, outputs)})
    unit = pipeline_stages.fit_unit(pipeline)
    (work / "scores.json").write_text(json.dumps({"scores": {}, "scored": {"fit": "f001", "fit_digest": unit.digest}}))
    pipeline_stages.run_store(pipeline, {})
    assert record.read(rec.TRANSFORM)["fixed_volume"] == "coarse.zarr"
