"""Round 6 of the review, second blocker: Windows refuses to open a file while another
process holds it with delete access (a replace in progress), and record reads gave up at
once, so `volreg status` and the score job raised PermissionError, and the job did so while
reading what it needs, before its failure handler: Score was left ready with no worker and
no stamp.

Each test opens with the requirement it checks, quoted from the review. Off Windows the
refusal is made where the file is opened, as Windows makes it, and the Windows primitives are
put in place (see test_review_blockers.py); on Windows, or under Wine, the last tests hold a
real delete-access handle."""

import builtins
import ctypes
import io
import json
import os
import threading
import time
from pathlib import Path

import pytest

from volreg import cli, engine, filelock
from volreg import record as rec
from volreg import stages as pipeline_stages
from volreg.engine import DONE, FAILED, READY

from test_redo_release import drain, fake_stages, new_pair  # noqa: F401 - a fixture
from test_review_blockers import REVIEWER, WINDOWS, as_windows_locks

BY = REVIEWER
READS = ("Routine record reads must recover from temporary sharing conflicts and still surface "
         "persistent errors.")


def refuse_opens_of(monkeypatch, path, times):
    """Opening `path` for reading is refused `times()` times, as Windows refuses it while
    another process holds the file with delete access. Counts the refusals."""
    real_open, refused, target = builtins.open, [0], os.path.realpath(path)

    def guarded(file, mode="r", *args, **kwargs):
        if (isinstance(file, (str, os.PathLike)) and os.path.realpath(file) == target
                and not any(flag in mode for flag in "wax+") and refused[0] < times()):
            refused[0] += 1
            raise PermissionError(13, "The process cannot access the file because it is being "
                                      "used by another process", str(file))
        return real_open(file, mode, *args, **kwargs)
    monkeypatch.setattr(builtins, "open", guarded)
    monkeypatch.setattr(io, "open", guarded)
    return refused


def scored_up_to_score(tmp_path):
    """A pair whose Score is ready: everything before it done, the hold-out set answered."""
    directory = new_pair(tmp_path / "records")
    pipeline = pipeline_stages.pipeline(directory)
    for name in ("capture", "holdout", "fit", "recheck"):
        assert pipeline.run_stage(name)["status"] == "ok"
    assert pipeline_stages.pipeline(directory).states()["score"] == READY
    return directory


# ------------------------------------------------------------------ temporary conflicts

def test_a_record_read_rides_out_a_replace_in_progress(tmp_path, monkeypatch):
    """Routine record reads must recover from temporary sharing conflicts and still surface
    persistent errors. (`Record.read`, `read_bytes` and the event log.)"""
    as_windows_locks(monkeypatch)
    record = rec.Record(tmp_path / "a")
    record.write(rec.HOLDOUT, {"patches": []})
    record.log("note", patch="p0001", text="x", by=BY)
    limit = [3]
    refused = refuse_opens_of(monkeypatch, record.path(rec.HOLDOUT), lambda: limit[0])
    assert record.read(rec.HOLDOUT)["patches"] == [] and refused[0] == 3
    refused[0] = 0
    assert record.read_bytes(rec.HOLDOUT).startswith(b"{") and refused[0] == 3
    refused = refuse_opens_of(monkeypatch, record.path(rec.EVENTS), lambda: 2)
    assert record.events()[0]["kind"] == "note" and refused[0] == 2


def test_status_rides_out_a_replace_of_holdout_json(tmp_path, fake_stages, monkeypatch, capsys):
    """Routine record reads must recover from temporary sharing conflicts and still surface
    persistent errors. (`volreg status` raised PermissionError.)"""
    as_windows_locks(monkeypatch)
    directory = scored_up_to_score(tmp_path)
    refused = refuse_opens_of(monkeypatch, directory / rec.HOLDOUT, lambda: 4)
    cli.main(["status", str(directory.parent)])
    assert refused[0] == 4 and "score ready" in capsys.readouterr().out


def test_the_score_job_rides_out_a_replace_of_holdout_json(tmp_path, fake_stages, monkeypatch):
    """Routine record reads must recover from temporary sharing conflicts and still surface
    persistent errors. (The score job reads the hold-out answers for what it needs, then
    scores them.)"""
    as_windows_locks(monkeypatch)
    directory = scored_up_to_score(tmp_path)
    refused = refuse_opens_of(monkeypatch, directory / rec.HOLDOUT, lambda: 4)
    assert engine.main_job(pipeline_stages.pipeline(directory), "score") == 0
    assert refused[0] == 4
    assert pipeline_stages.pipeline(directory).states()["score"] == DONE


def test_the_job_rides_out_a_replace_of_its_settings(tmp_path, fake_stages, monkeypatch):
    """Routine record reads must recover from temporary sharing conflicts and still surface
    persistent errors. (`volreg job` reads the pair's settings before anything else.)"""
    as_windows_locks(monkeypatch)
    directory = scored_up_to_score(tmp_path)
    rec.Record(directory).write(pipeline_stages.PARAMS, {"blind": {"count": 20}})
    refused = refuse_opens_of(monkeypatch, directory / pipeline_stages.PARAMS, lambda: 3)
    monkeypatch.setattr(cli, "advance", lambda pair: (None, []))
    assert cli.main(["job", str(directory), "score"]) == 0 and refused[0] == 3


def test_the_redo_marks_ride_out_a_replace(tmp_path, fake_stages, monkeypatch):
    """Routine record reads must recover from temporary sharing conflicts and still surface
    persistent errors. (The redo marks are replaced whole, and read by every job.)"""
    as_windows_locks(monkeypatch)
    directory = scored_up_to_score(tmp_path)
    pipeline_stages.pipeline(directory).redo("score")
    refused = refuse_opens_of(monkeypatch, directory / "work" / "stages" / "redo.json", lambda: 3)
    assert pipeline_stages.pipeline(directory).run_stage("score")["status"] == "ok"
    assert refused[0] >= 3


# ------------------------------------------------------------------ persistent errors

def test_a_record_that_stays_unreadable_is_not_hidden(tmp_path, monkeypatch):
    """Routine record reads must recover from temporary sharing conflicts and still surface
    persistent errors."""
    as_windows_locks(monkeypatch)
    monkeypatch.setattr(filelock, "RETRY_SECONDS", 0.2)
    record = rec.Record(tmp_path / "a")
    record.write(rec.HOLDOUT, {"patches": []})
    refuse_opens_of(monkeypatch, record.path(rec.HOLDOUT), lambda: 10 ** 9)
    with pytest.raises(PermissionError):
        record.read(rec.HOLDOUT)


def test_status_surfaces_a_persistent_error(tmp_path, fake_stages, monkeypatch):
    """Routine record reads must recover from temporary sharing conflicts and still surface
    persistent errors. (`volreg status`.)"""
    as_windows_locks(monkeypatch)
    directory = scored_up_to_score(tmp_path)
    monkeypatch.setattr(filelock, "RETRY_SECONDS", 0.2)
    refuse_opens_of(monkeypatch, directory / rec.HOLDOUT, lambda: 10 ** 9)
    with pytest.raises(PermissionError):
        cli.main(["status", str(directory.parent)])


def test_a_job_that_cannot_read_what_it_needs_fails_through_its_failure_handler(tmp_path, fake_stages, monkeypatch):
    """Routine record reads must recover from temporary sharing conflicts and still surface
    persistent errors. (The job read its prerequisites before its failure handler: Score was
    left ready with no worker or stamp once the handle closed.)"""
    as_windows_locks(monkeypatch)
    directory = scored_up_to_score(tmp_path)
    monkeypatch.setattr(filelock, "RETRY_SECONDS", 0.2)
    held = [True]
    refuse_opens_of(monkeypatch, directory / rec.HOLDOUT, lambda: 10 ** 9 if held[0] else 0)
    code = engine.main_job(pipeline_stages.pipeline(directory), "score")
    held[0] = False                                             # the handle closes
    assert code == 1
    pipeline = pipeline_stages.pipeline(directory)
    stamp = pipeline.stamp("score")
    assert stamp["status"] == "failed" and "PermissionError" in stamp["error"]
    assert pipeline.states()["score"] == FAILED                 # not ready with nobody on it
    assert not pipeline.running("score")
    failed = [event for event in pipeline.record.events() if event["kind"] == "job-failed"]
    assert failed and failed[-1]["stage"] == "score"
    pipeline.redo("score")                                      # what a failed stage takes
    assert drain(directory).states()["score"] == DONE


def test_a_job_that_cannot_read_its_settings_fails_through_its_failure_handler(tmp_path, fake_stages, monkeypatch, capsys):
    """Routine record reads must recover from temporary sharing conflicts and still surface
    persistent errors. (`volreg job`, before the pipeline is even built.)"""
    as_windows_locks(monkeypatch)
    directory = scored_up_to_score(tmp_path)
    # A setting of the stage itself: a stamp that took the defaults for it would not match.
    rec.Record(directory).write(pipeline_stages.PARAMS, {"score": {"note": "a setting"}})
    monkeypatch.setattr(filelock, "RETRY_SECONDS", 0.2)
    held = [True]
    refuse_opens_of(monkeypatch, directory / pipeline_stages.PARAMS, lambda: 10 ** 9 if held[0] else 0)
    monkeypatch.setattr(cli, "advance", lambda pair: (None, []))
    assert cli.main(["job", str(directory), "score"]) == 1
    held[0] = False
    pipeline = pipeline_stages.pipeline(directory)
    assert pipeline.stamp("score")["status"] == "failed"
    assert pipeline.states()["score"] == FAILED


def test_a_failure_after_the_stage_began_is_still_recorded_as_before(tmp_path, fake_stages, monkeypatch):
    """Routine record reads must recover from temporary sharing conflicts and still surface
    persistent errors. (A read that fails inside the stage's own work: as any failure.)"""
    as_windows_locks(monkeypatch)
    directory = scored_up_to_score(tmp_path)
    monkeypatch.setattr(filelock, "RETRY_SECONDS", 0.2)

    def score_reading_pair(pipeline, params):
        pipeline.record.read(rec.PAIR)
        return []
    monkeypatch.setattr(pipeline_stages, "run_score", score_reading_pair)
    refuse_opens_of(monkeypatch, directory / rec.PAIR, lambda: 10 ** 9)
    stamp = pipeline_stages.pipeline(directory).run_stage("score")
    assert stamp["status"] == "failed" and stamp["inputs"]


# ------------------------------------------------------- a real delete-access handle (Windows)

_DELETE, _SHARE_READ_WRITE, _OPEN_EXISTING = 0x00010000, 0x1 | 0x2, 3


def _hold_with_delete_access(path, seconds):
    """Open `path` as a replace does (delete access, no delete sharing), and close it after
    `seconds`, in a thread. Windows only."""
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.restype = ctypes.c_void_p
    kernel.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p,
                                   ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.CreateFileW(str(path), _DELETE, _SHARE_READ_WRITE, None, _OPEN_EXISTING, 0, None)
    assert handle not in (None, ctypes.c_void_p(-1).value), ctypes.get_last_error()
    closer = threading.Timer(seconds, kernel.CloseHandle, (handle,))
    closer.start()
    return closer


@pytest.mark.skipif(not WINDOWS, reason="a delete-access handle is a Windows sharing conflict")
def test_windows_status_and_score_job_ride_out_a_delete_access_handle(tmp_path, fake_stages, capsys):
    """Routine record reads must recover from temporary sharing conflicts and still surface
    persistent errors. (As the review did: a delete-access handle on holdout.json.)"""
    directory = scored_up_to_score(tmp_path)
    holdout = directory / rec.HOLDOUT
    with pytest.raises(PermissionError):                         # the conflict is real
        _hold_with_delete_access(holdout, 1.0)
        open(holdout).close()
    time.sleep(1.2)
    closer = _hold_with_delete_access(holdout, 1.0)
    cli.main(["status", str(directory.parent)])
    closer.join()
    assert "score ready" in capsys.readouterr().out
    closer = _hold_with_delete_access(holdout, 1.0)
    assert engine.main_job(pipeline_stages.pipeline(directory), "score") == 0
    closer.join()
    assert pipeline_stages.pipeline(directory).states()["score"] == DONE


@pytest.mark.skipif(not WINDOWS, reason="a delete-access handle is a Windows sharing conflict")
def test_windows_a_delete_access_handle_held_too_long_fails_the_job_visibly(tmp_path, fake_stages, monkeypatch):
    """Routine record reads must recover from temporary sharing conflicts and still surface
    persistent errors. (Held past the wait: the job fails through its handler.)"""
    directory = scored_up_to_score(tmp_path)
    monkeypatch.setattr(filelock, "RETRY_SECONDS", 0.3)
    closer = _hold_with_delete_access(directory / rec.HOLDOUT, 2.0)
    code = engine.main_job(pipeline_stages.pipeline(directory), "score")
    closer.join()
    assert code == 1
    assert pipeline_stages.pipeline(directory).states()["score"] == FAILED
