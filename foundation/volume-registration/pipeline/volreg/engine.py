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

# A lock file with no pid in it yet is a job caught between creating it and writing to it.
LOCK_WRITE_SECONDS = 5


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
        """Whether a job holds the stage's lock and is alive. The lock names the job's pid
        (on Windows also what tells it from a later process with the same pid); nothing is
        signalled."""
        lock = self.lock_path(name)
        try:
            text = lock.read_text()
            age = time.time() - lock.stat().st_mtime
        except FileNotFoundError:
            return False
        parts = text.split()
        if not parts:
            # Just created, its owner still writing its pid in; a crash there goes stale.
            return age < LOCK_WRITE_SECONDS
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
        """Run one automatic stage here and now, under its lock; stamp the result."""
        stage = self.stages[name]
        if stage.human:
            raise ValueError(f"{name} is a person's stage")
        lock = self.lock_path(name)
        lock.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._create_lock(lock)
        except FileExistsError:
            # A job's lock, live or left behind by one that died. Only one process at a time
            # may judge it, and it is removed only if still dead once it is that one's turn;
            # a worker that took the stage in the meantime is left alone.
            with self._stage_guard(name):
                if self.running(name):
                    raise RuntimeError(f"{name} is already running")
                filelock.unlink(lock, missing_ok=True)
                try:
                    self._create_lock(lock)
                except FileExistsError:
                    raise RuntimeError(f"{name} is already running") from None
        inputs, params = self.inputs(name), self.stage_params(name)
        stamp = {"stage": name, "inputs": inputs, "params": params, "code": code_version(),
                 "started": rec.now()}
        self.record.log("job-started", stage=name, params=params)
        try:
            written = list(stage.run(self, params) or [])
            stamp |= {"status": "ok", "finished": rec.now(), "outputs": written,
                      "output_digest": digest_files(self.directory, written)}
            self.record.log("job-finished", stage=name, outputs=written)
        except Exception as error:   # noqa: BLE001 - a stage's failure is recorded, not raised
            stamp |= {"status": "failed", "finished": rec.now(),
                      "error": f"{type(error).__name__}: {error}",
                      "traceback": traceback.format_exc()}
            self.record.log("job-failed", stage=name, error=stamp["error"])
        finally:
            try:
                self._write_stamp(name, stamp)
            finally:
                filelock.unlink(lock, missing_ok=True)
        return stamp

    @staticmethod
    def _create_lock(lock):
        """Create the lock file, or raise FileExistsError, with this process's pid in it."""
        token = filelock.process_token()
        owner = f"{os.getpid()}\n" if token is None else f"{os.getpid()} {token}\n"
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(descriptor, owner.encode())
        os.close(descriptor)

    @contextmanager
    def _stage_guard(self, name):
        """Held while a stage's left-over lock is judged and replaced."""
        with open(self.work("stages", f"{name}.recover"), "w") as handle:
            filelock.lock_exclusive(handle)
            try:
                yield
            finally:
                filelock.unlock(handle)

    @contextmanager
    def _launch_guard(self):
        """Held while jobs are chosen and started, and while a redo clears stamps, so a job is
        never started on the half of a redo that is done."""
        path = self.work("stages", "launch.guard")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as handle:
            filelock.lock_exclusive(handle)
            try:
                yield
            finally:
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
                                    stdin=subprocess.DEVNULL, start_new_session=True,
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
        kept beside, so what it was is not lost. All or nothing: if a stamp cannot be moved
        (a reader has it open for longer than a replace waits), those already moved go back,
        and the error is raised."""
        affected = self.downstream(name)
        history = self.work("stages", "history")
        moved = []
        with self._launch_guard():
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
    if stamp["status"] != "ok":
        print(stamp.get("traceback", stamp.get("error")), file=sys.stderr)
        return 1
    return 0
