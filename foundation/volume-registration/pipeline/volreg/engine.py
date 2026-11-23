"""Where a pair stands, and moving it on: the stages, their state, and their jobs.

State is never stored as such; it is read off the files, as make does. Every automatic stage
leaves a stamp, `work/stages/<stage>.json`, naming the digests of what it ran on -- the
outputs of the stages it needs, and its parameters -- and the digests of what it wrote. A
stage is

    done      its stamp says it finished, and what it ran on has not changed since
    stale     it finished, but on inputs that have changed since (new answers, more patches)
    running   a job holds its lock and is alive
    failed    its last job failed; the stamp keeps the error and the log
    ready     everything it needs is done, and it is automatic
    waiting   everything it needs is done, and a person is next
    blocked   something it needs is not done

A human stage is done when its answers say so (see `Stage.complete`); its digest is the
digest of its `identity` -- what the person decided, and nothing else: the answers, not the
notes beside them, the pictures they were shown, or when. So a new answer makes everything
after it stale, and a rerun of scoring picks it up, while a note does not. The event log
keeps all of it, for provenance; identity is only what the stages after it rest on.

    <pair>/                  the record (see record.py), shared
    <pair>/work/stages/      stamps and logs
    <pair>/work/...          everything a stage writes that can be recomputed
"""

import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from . import filelock
from . import record as rec

DONE, STALE, RUNNING, FAILED, READY, WAITING, BLOCKED = (
    "done", "stale", "running", "failed", "ready", "waiting", "blocked")

# What a job writes after its pid in the lock file once it holds the file's OS lock; a lock
# file without it is an older one, whose pid alone says whether the job lives.
OS_LOCK = "os-lock"


class Superseded(Exception):
    """A redo ran after the job was chosen: what it ran on is no longer current."""


def windows():
    return os.name == "nt"


def detach_options():
    """Popen options that let a job outlive its launcher and the launcher's console."""
    if windows():
        # Windows ignores start_new_session. A hidden console of its own (not DETACHED_PROCESS,
        # whose children each open a console window) outlives the caller's console.
        return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200)
                | getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)}
    return {"start_new_session": True}


@dataclass
class Stage:
    name: str
    title: str
    needs: tuple = ()
    # Automatic: `run(pair, params)` does the work and returns the paths it wrote, relative
    # to the pair. Human: None, and `complete(pair, params)` says whether the answers suffice.
    run: Optional[Callable] = None
    complete: Optional[Callable] = None
    # A human stage's output for the stages after it: `identity(pipeline)`, JSON-able.
    identity: Optional[Callable] = None
    params: dict = field(default_factory=dict)

    @property
    def human(self):
        return self.run is None


def digest_files(directory, paths):
    """One digest over files, in order; a missing file counts as empty, by name."""
    hasher = hashlib.sha256()
    for path in paths:
        hasher.update(str(path).encode() + b"\0")
        target = Path(directory) / path
        if target.is_file():
            hasher.update(target.read_bytes())
        hasher.update(b"\1")
    return hasher.hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def code_version():
    """The commit the code runs from, and whether the tree differs from it."""
    here = Path(__file__).resolve().parent
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "."], cwd=here,
                                    capture_output=True, text=True).stdout.strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


class Pipeline:
    """The stages of one pair, over its folder."""

    def __init__(self, directory, stages, params=None):
        self.directory = Path(directory)
        self.record = rec.Record(self.directory)
        self.stages = {stage.name: stage for stage in stages}
        self.order = [stage.name for stage in stages]
        for stage in stages:
            for need in stage.needs:
                if self.order.index(need) >= self.order.index(stage.name):
                    raise ValueError(f"{stage.name} needs {need}, which must come before it")
        self.params = params or {}
        self._started = {}      # stage -> its redo mark when this process's job began
        self._guard_depth = 0

    # ---------------------------------------------------------------------- the files

    def work(self, *parts):
        return self.directory.joinpath("work", *parts)

    def stamp_path(self, name):
        return self.work("stages", f"{name}.json")

    def stamp(self, name):
        path = self.stamp_path(name)
        return json.loads(path.read_text()) if path.exists() else None

    def _write_stamp(self, name, stamp):
        path = self.stamp_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        scratch = path.with_name(path.name + ".tmp")
        scratch.write_text(json.dumps(stamp, indent=1) + "\n")
        filelock.replace(scratch, path)

    def lock_path(self, name):
        return self.work("stages", f"{name}.lock")

    def log_path(self, name):
        return self.work("stages", f"{name}.log")

    def stage_params(self, name):
        return self.stages[name].params | self.params.get(name, {})

    # ---------------------------------------------------------------------- the state

    def output_digest(self, name):
        """What a stage's dependents ran on: its outputs, or a human stage's answers."""
        stage = self.stages[name]
        if stage.human:
            return digest_json(stage.identity(self))
        stamp = self.stamp(name)
        return None if stamp is None else stamp.get("output_digest")

    def inputs(self, name):
        return {"needs": {need: self.output_digest(need) for need in self.stages[name].needs},
                "params": digest_json(self.stage_params(name))}

    def running(self, name):
        """Whether a job holds the stage's lock and is alive. A job holds the OS lock on its
        lock file for its whole life, so a lock nobody can take belongs to no one; only a lock
        file from before that (no `os-lock` in it) is judged by its pid (on Windows also what
        tells it from a later process with the same pid); nothing is signalled."""
        try:
            with open(self.lock_path(name), "r+b") as handle:
                if not filelock.try_lock(handle, filelock.OWNER_OFFSET):
                    return True
                try:
                    handle.seek(0)      # try_lock left it at the locked byte on Windows
                    parts = handle.read().decode(errors="replace").split()
                finally:
                    filelock.unlock(handle, filelock.OWNER_OFFSET)
        except FileNotFoundError:
            return False
        if not parts or OS_LOCK in parts:
            return False
        try:
            pid = int(parts[0])
            token = int(parts[1]) if len(parts) > 1 else None
        except (ValueError, OverflowError):
            return False
        return filelock.pid_alive(pid, token)

    def state(self, name, _memo=None):
        memo = {} if _memo is None else _memo
        if name in memo:
            return memo[name]
        stage = self.stages[name]
        needs_done = all(self.state(need, memo) == DONE for need in stage.needs)
        if stage.human:
            result = (BLOCKED if not needs_done else
                      DONE if stage.complete(self, self.stage_params(name)) else WAITING)
        elif self.running(name):
            result = RUNNING
        else:
            stamp = self.stamp(name)
            if stamp and stamp.get("status") == "failed" and stamp.get("inputs") == self.inputs(name):
                result = FAILED
            elif stamp and stamp.get("status") == "ok":
                # A result whose own inputs still match waits, too, while a stage before it
                # runs again: what it rests on is about to change.
                result = (BLOCKED if not needs_done else
                          DONE if stamp.get("inputs") == self.inputs(name) else STALE)
            else:
                result = READY if needs_done else BLOCKED
        memo[name] = result
        return result

    def states(self):
        memo = {}
        return {name: self.state(name, memo) for name in self.order}

    # ----------------------------------------------------------------------- the jobs

    def run_stage(self, name):
        """Run one automatic stage here and now, under its lock; stamp the result. If a redo
        ran after the job was chosen, nothing is run or published: the stamp returned says
        `superseded`."""
        stage = self.stages[name]
        if stage.human:
            raise ValueError(f"{name} is a person's stage")
        handle = self._take_lock(name)
        try:
            try:
                with self._launch_guard():
                    # Under the stage's lock, and with no redo half done: what it needs must
                    # still be done, as it was when the job was chosen.
                    self._require_needs(name)
                    self._started[name] = self.redo_mark(name)
                    inputs, params = self.inputs(name), self.stage_params(name)
                stamp = {"stage": name, "inputs": inputs, "params": params, "code": code_version(),
                         "started": rec.now(), "redo_mark": self._started[name]}
                self.record.log("job-started", stage=name, params=params)
                try:
                    written = list(stage.run(self, params) or [])
                    stamp |= {"status": "ok", "finished": rec.now(), "outputs": written,
                              "output_digest": digest_files(self.directory, written)}
                    self.record.log("job-finished", stage=name, outputs=written)
                except Superseded:
                    raise
                except Exception as error:   # noqa: BLE001 - a stage's failure is recorded, not raised
                    stamp |= {"status": "failed", "finished": rec.now(),
                              "error": f"{type(error).__name__}: {error}",
                              "traceback": traceback.format_exc()}
                    self.record.log("job-failed", stage=name, error=stamp["error"])
                with self._launch_guard():
                    self._require_current(name)
                    self._write_stamp(name, stamp)
            except Superseded as why:
                self.record.log("job-superseded", stage=name, why=str(why))
                return {"stage": name, "status": "superseded", "error": str(why)}
            finally:
                self._started.pop(name, None)
        finally:
            self._release_lock(handle)
        return stamp

    def _take_lock(self, name):
        """The stage's lock file, open and OS-locked, with this process's pid in it; or
        RuntimeError if a job holds it. The lock is the owner's: one that cannot be taken
        has a live owner, and one that can has none, however the last owner ended."""
        lock = self.lock_path(name)
        lock.parent.mkdir(parents=True, exist_ok=True)
        with open(lock, "ab"):
            pass
        handle = open(lock, "r+b")
        try:
            # Someone asking whether the job runs holds the lock for an instant: try again.
            for attempt in range(5):
                if filelock.try_lock(handle, filelock.OWNER_OFFSET):
                    break
                time.sleep(0.05 * (attempt + 1))
            else:
                raise RuntimeError(f"{name} is already running")
            handle.seek(0)      # try_lock left it at the locked byte on Windows
            parts = handle.read().decode(errors="replace").split()
            if parts and OS_LOCK not in parts and self._legacy_alive(parts):
                raise RuntimeError(f"{name} is already running")
            token = filelock.process_token()
            handle.seek(0)
            handle.truncate()
            handle.write(f"{os.getpid()} {'-' if token is None else token} {OS_LOCK}\n".encode())
            handle.flush()
        except BaseException:
            handle.close()
            raise
        return handle

    @staticmethod
    def _legacy_alive(parts):
        try:
            return filelock.pid_alive(int(parts[0]), int(parts[1]) if len(parts) > 1 else None)
        except (ValueError, OverflowError):
            return False

    @staticmethod
    def _release_lock(handle):
        try:
            handle.seek(0)
            handle.truncate()
            handle.flush()
        finally:
            try:
                filelock.unlock(handle, filelock.OWNER_OFFSET)
            finally:
                handle.close()

    # ------------------------------------------------------------- redo and queued jobs

    def _redo_path(self):
        return self.work("stages", "redo.json")

    def redo_marks(self):
        path = self._redo_path()
        return json.loads(path.read_text()) if path.exists() else {"seq": 0, "stages": {}}

    def ancestors(self, name):
        found = {name}
        for other in reversed(self.order[:self.order.index(name) + 1]):
            if other in found:
                found.update(self.stages[other].needs)
        return found

    def redo_mark(self, name):
        """How many redos have touched the stage or anything it rests on, so far."""
        stages = self.redo_marks()["stages"]
        return max((stages.get(other, 0) for other in self.ancestors(name)), default=0)

    def _require_needs(self, name):
        for need in self.stages[name].needs:
            if self.state(need) != DONE:
                raise Superseded(f"{name} needs {need}, which is {self.state(need)} now")

    def _require_current(self, name):
        if name in self._started and self.redo_mark(name) != self._started[name]:
            raise Superseded(f"{name} was redone, or something it rests on was, while it ran")

    @contextmanager
    def publishing(self, name):
        """Held by a stage while it writes what others read (the record): the writes are made
        only if no redo has touched the stage or what it rests on since its job began, and
        no redo can begin in the middle of them."""
        with self._launch_guard():
            self._require_current(name)
            yield

    @contextmanager
    def _launch_guard(self):
        """Held while jobs are chosen and started, while a redo clears stamps, and while a job
        checks what it rests on and writes its result, so a job is never started on the half
        of a redo that is done, and never publishes across one. Re-entrant in a process."""
        if self._guard_depth:
            self._guard_depth += 1
            try:
                yield
            finally:
                self._guard_depth -= 1
            return
        path = self.work("stages", "launch.guard")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as handle:
            filelock.lock_exclusive(handle)
            self._guard_depth = 1
            try:
                yield
            finally:
                self._guard_depth = 0
                filelock.unlock(handle)

    def start(self, name, command):
        """Start a stage as a background job: `command` (a list) is run with its output going
        to the stage's log, detached from the caller."""
        log = self.log_path(name)
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a") as handle:
            handle.write(f"\n--- {rec.now()} {' '.join(command)}\n")
            handle.flush()
            return subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, **detach_options(),
                                    cwd=Path(__file__).resolve().parent.parent)

    def advance(self, launch):
        """Start every ready or stale automatic stage; `launch(name)` starts one. Returns the
        states before, and the stages started."""
        with self._launch_guard():
            states = self.states()
            started = [name for name in self.order
                       if states[name] in (READY, STALE) and not self.stages[name].human]
            for name in started:
                launch(name)
        return states, started

    def adopt_needs(self, name, reason):
        """After a change to what a stage needs (not to what it does), take its finished
        result as resting on the new needs, instead of running it again. Only the needs
        change: its parameters must match, and a stage whose new needs are not all done is
        left alone. The old stamp is kept beside, and the change is logged with `reason`."""
        stamp = self.stamp(name)
        if not stamp or stamp.get("status") != "ok":
            raise ValueError(f"{name} has no finished result to keep")
        inputs = self.inputs(name)
        if stamp["inputs"]["params"] != inputs["params"]:
            raise ValueError(f"{name} ran with other parameters; run it again instead")
        if not all(self.state(need) == DONE for need in self.stages[name].needs):
            raise ValueError(f"what {name} needs is not done yet")
        if stamp["inputs"] == inputs:
            return False
        history = self.work("stages", "history")
        history.mkdir(parents=True, exist_ok=True)
        (history / f"{name}.{rec.now().replace(':', '')}.json").write_text(json.dumps(stamp, indent=1) + "\n")
        self._write_stamp(name, stamp | {"inputs": inputs, "adopted": {"at": rec.now(), "reason": reason,
                                                                        "needs_were": stamp["inputs"]["needs"]}})
        self.record.log("needs-adopted", stage=name, reason=reason, were=stamp["inputs"]["needs"],
                        now=inputs["needs"])
        return True

    def redo(self, name):
        """Make a stage and everything after it that depends on it run again. Its stamp is
        kept beside, so what it was is not lost. Jobs already chosen see the redo (the mark
        below) and give up instead of publishing. All or nothing: if a stamp cannot be moved
        (a reader has it open for longer than a replace waits), those already moved go back
        -- unless a newer stamp has taken the place, which is left alone -- and the error is
        raised."""
        affected = self.downstream(name)
        history = self.work("stages", "history")
        moved = []
        with self._launch_guard():
            marks = self.redo_marks()
            marks["seq"] += 1
            marks["stages"].update({stage: marks["seq"] for stage in affected})
            scratch = self._redo_path().with_name("redo.json.tmp")
            scratch.parent.mkdir(parents=True, exist_ok=True)
            scratch.write_text(json.dumps(marks) + "\n")
            filelock.replace(scratch, self._redo_path())
            try:
                for stage in affected:
                    stamp = self.stamp_path(stage)
                    if stamp.exists():
                        history.mkdir(parents=True, exist_ok=True)
                        kept = history / f"{stage}.{rec.now().replace(':', '')}.json"
                        filelock.replace(stamp, kept)
                        moved.append((stamp, kept))
            except BaseException:
                for stamp, kept in reversed(moved):
                    if stamp.exists():
                        continue   # finished again since: that one is newer than what was kept
                    try:
                        filelock.replace(kept, stamp)
                    except OSError:
                        pass   # put back what can be; the error that matters is the first
                raise
        self.record.log("redo", stage=name, affected=affected)
        return affected

    def downstream(self, name):
        found = [name]
        for other in self.order:
            if any(need in found for need in self.stages[other].needs) and other not in found:
                found.append(other)
        return [stage for stage in found if not self.stages[stage].human]


def main_job(pipeline, name):
    """Entry point of a background job: run the stage, exit 1 if it failed."""
    stamp = pipeline.run_stage(name)
    if stamp["status"] == "superseded":
        print(f"{name}: not run: {stamp['error']}", file=sys.stderr)
        return 0
    if stamp["status"] != "ok":
        print(stamp.get("traceback", stamp.get("error")), file=sys.stderr)
        return 1
    return 0
