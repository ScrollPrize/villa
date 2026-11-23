"""The per-pair record: what a person decided, kept apart from everything recomputable.

    <record>/pair.json            the two volumes and the start transform
    <record>/holdout.json         the hold-out patches, their own measurements, the answers
    <record>/final-review.json    the blind rounds and the decision
    <record>/patch-reviews.json   verdicts and notes on fit patches (the first reviews had them)
    <record>/fit/transform.json   the final transform, in the atlas's format
    <record>/fit/fit.json         how it was fitted, the points it rests on, its scores, and
                                  (`stored`) the fit, variant and transform digest it was
                                  stored with: what binds it to the transform beside it
    <record>/events.jsonl         every answer and job, append-only

Files other than `events.jsonl` are replaced whole, in one step, so a reader never sees half
a file. Coordinates are ZYX level-0 voxels of their own volume, as in the cores.
"""

import json
import os
import secrets
import stat
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from . import filelock

SCHEMA = "volreg-record/1"

PAIR = "pair.json"
HOLDOUT = "holdout.json"
FINAL_REVIEW = "final-review.json"
PATCH_REVIEWS = "patch-reviews.json"
TRANSFORM = "fit/transform.json"
FIT = "fit/fit.json"
EVENTS = "events.jsonl"

# What a person answers about a hold-out patch, shown it at its own measurement and nothing
# about any fit. Only "lines-up" scores; see `scoring.py`.
ANSWERS = ("lines-up", "partly", "no", "cant-tell", "not-scroll")
LINES_UP = "lines-up"
# What a person can pick in a blind round instead of an option.
BLIND_ANSWERS = ("unsure", "none")
# What a decision can publish.
PUBLISH = ("bend", "affine", "nothing")


# The record locks this thread holds, and how deep: a lock taken again by its holder (a
# decision checked and written under one hold) is not waited for.
_held = threading.local()


def dumps(data):
    """A record file's bytes, as `Record.write` writes them."""
    return (json.dumps(data, indent=1) + "\n").encode()


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Reviewer:
    """Who gave an answer: a handle for what is shared, and an email to reach them."""

    def __init__(self, handle, email):
        if not handle or not email:
            raise ValueError("a reviewer needs a handle and an email")
        self.handle, self.email = handle, email

    def as_json(self):
        return {"handle": self.handle, "email": self.email}


class Record:
    """One pair's record, in a local folder."""

    def __init__(self, directory):
        self.directory = Path(directory)

    def path(self, name):
        return self.directory / name

    def exists(self, name):
        return self.path(name).exists()

    def read(self, name):
        with open(self.path(name)) as handle:
            return json.load(handle)

    def read_bytes(self, name):
        return self.path(name).read_bytes()

    def write(self, name, data):
        """Replace a file in one step, with the schema stamped in."""
        if isinstance(data, dict) and name.endswith(".json") and name != TRANSFORM:
            data = {"schema": SCHEMA} | data
        self.write_bytes(name, dumps(data))

    def write_bytes(self, name, content):
        """Replace a file in one step with `content`, exactly."""
        target = self.path(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, scratch = _scratch(target)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            filelock.replace(scratch, target)
        except BaseException:
            filelock.unlink(scratch, missing_ok=True)
            raise

    @contextmanager
    def lock(self, name=".lock"):
        """Held while a view (`holdout.json`, `final-review.json`) is rewritten from the
        log, so a page and a job doing it at once cannot lose each other's answers. Another
        `name` is another lock, for what is read, changed and written back (the settings).
        Taken again by the thread that holds it, it is already held: a check and the writes
        it allows are made under one hold. Other threads and processes wait."""
        key = (os.path.normcase(os.path.realpath(self.directory)), name)    # one file, one key, however it is spelled
        held = _held.__dict__.setdefault("depth", {})
        if held.get(key):
            held[key] += 1
            try:
                yield
            finally:
                held[key] -= 1
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        with open(self.directory / name, "w") as handle:
            filelock.lock_exclusive(handle)
            held[key] = 1
            try:
                yield
            finally:
                held.pop(key, None)
                filelock.unlock(handle)

    def log(self, kind, **fields):
        """Append one event; `at` defaults to now. Never rewrites what is there. Under the
        record lock: a Windows append seeks to the end and then writes, so two writers at
        once could pick the same spot and one event would overwrite the other."""
        event = {"at": fields.pop("at", None) or now(), "kind": kind} | fields
        with self.lock(), open(self.path(EVENTS), "a") as handle:
            handle.write(json.dumps(event, sort_keys=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def events(self):
        if not self.exists(EVENTS):
            return []
        with open(self.path(EVENTS)) as handle:
            return [json.loads(line) for line in handle if line.strip()]

    # ------------------------------------------------------------------ the answers

    def answer(self, unit, answer, reviewer):
        """A person's answer about a hold-out unit (a patch, or `<patch>.<octant>`), or None
        to take it back. The event log is the answers' only source; `holdout.json` is
        rewritten from it."""
        if answer is not None and answer not in ANSWERS:
            raise ValueError(f"unknown answer {answer!r}; one of {', '.join(ANSWERS)}")
        units = holdout_units(self.read(HOLDOUT)) if self.exists(HOLDOUT) else set()
        if unit not in units:
            raise ValueError(f"{unit} is not a hold-out unit")
        self.log("holdout-answer", unit=unit, answer=answer, by=reviewer)
        self.refresh_holdout()

    def answers(self):
        """The latest answer per hold-out unit, from the event log."""
        latest = {}
        for event in self.events():
            if event["kind"] == "holdout-answer":
                latest[event["unit"]] = (None if event["answer"] is None else
                                         {"answer": event["answer"], "at": event["at"],
                                          "by": event.get("by")})
        return latest

    def refresh_holdout(self, patches=None, **fields):
        """Rewrite `holdout.json` with `patches` (or the ones it holds), `fields`, and the
        latest answers and notes."""
        with self.lock():
            return self._refresh_holdout(patches, fields)

    def _refresh_holdout(self, patches, fields):
        held = self.read(HOLDOUT) if self.exists(HOLDOUT) else {"patches": []}
        held |= fields
        if patches is not None:
            held = held | {"patches": patches}
        latest = self.answers()
        notes = self.notes()
        for patch in held["patches"]:
            patch["answer"] = latest.get(patch["id"])
            if patch["id"] in notes:
                patch["note"] = notes[patch["id"]]
            for octant in patch.get("octants") or []:
                octant["answer"] = latest.get(f"{patch['id']}.{octant['index']}")
        held.pop("schema", None)
        self.write(HOLDOUT, held)
        return held


    # ------------------------------------------------------- notes and the final review

    def note(self, patch_id, text, reviewer):
        """What a person saw in a patch, in their words; empty to remove it."""
        text = (text or "").strip()
        self.log("note", patch=patch_id, text=text, by=reviewer)
        self.refresh_holdout()


    def notes(self):
        latest = {}
        for event in self.events():
            if event["kind"] == "note":
                latest[event["patch"]] = ({"text": event["text"], "at": event["at"], "by": event.get("by")}
                                          if event["text"] else None)
        return latest


    def blind_pick(self, round_id, index, choice, reviewer):
        """The option a person picked for item `index` of a round, or one of `BLIND_ANSWERS`."""
        final = self.read(FINAL_REVIEW)
        round_ = next(r for r in final["rounds"] if r["id"] == round_id)
        item = round_["items"][index]
        if choice not in BLIND_ANSWERS and not (isinstance(choice, int)
                                                and 0 <= choice < len(item["options"])):
            raise ValueError(f"no option {choice!r}")
        self.log("blind-pick", round=round_id, item=index, patch=item["patch"], choice=choice,
                 by=reviewer)
        self.refresh_final()


    def decide(self, decision, reviewer, check=None):
        """The final decision: `{"fit", "publish", "reason", "override"}` and, from the review
        page, the `round` it was made on; logged, then shown. `check(final)`, given
        `final-review.json` as it stands, raises if the decision does not apply to it; it is
        called under the record lock the decision is then written under, so no round can be
        added or answered between the check and the write."""
        if decision["publish"] not in PUBLISH:
            raise ValueError(f"publish one of {', '.join(PUBLISH)}")
        with self.lock():
            if check is not None:
                check(self.read(FINAL_REVIEW) if self.exists(FINAL_REVIEW) else {})
            self.log("decision", **decision, by=reviewer)
            self.refresh_final()


    def refresh_final(self, rounds=None):
        """Rewrite `final-review.json` with the latest picks and decision from the log. Rounds
        and anything else it holds (a migrated decision, open questions) are kept."""
        with self.lock():
            final = self.read(FINAL_REVIEW) if self.exists(FINAL_REVIEW) else {"rounds": [], "decision": None}
            final.pop("schema", None)
            if rounds is not None:
                final["rounds"] = rounds
            picks, decision = {}, None
            for event in self.events():
                if event["kind"] == "blind-pick" and not event.get("migrated"):
                    picks[(event["round"], event["item"])] = event["choice"]
                elif event["kind"] == "decision":
                    decision = {key: event[key] for key in ("fit", "publish", "reason", "override")
                                } | {"by": event.get("by"), "at": event["at"]} | (
                                    {"round": event["round"]} if event.get("round") else {})
            for round_ in final["rounds"]:
                for index, item in enumerate(round_["items"]):
                    if (round_["id"], index) in picks:
                        item["choice"] = picks[(round_["id"], index)]
            if decision:
                final["decision"] = decision
                final["status"] = "decided"
            self.write(FINAL_REVIEW, final)
            return final


def _scratch(target):
    """A new scratch file beside `target`, of its own so two writers never share one, open
    for writing and already with the mode the record will have: the file's own mode on a
    rewrite (set before anything is written, so a private record is never readable by
    others, even half written), or for a new file the usual mode under the umask, as `open`
    would give it."""
    try:
        mode = stat.S_IMODE(os.stat(target).st_mode)
    except FileNotFoundError:
        mode = None
    while True:
        scratch = target.with_name(f"{target.name}.{secrets.token_hex(6)}.tmp")
        try:
            # A rewrite starts private and then takes the file's mode; a new file gets 0o666
            # less the umask from the system.
            descriptor = os.open(scratch, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
                                 0o600 if mode is not None else 0o666)
            break
        except FileExistsError:
            continue
    try:
        if mode is not None and hasattr(os, "fchmod") and os.name != "nt":
            os.fchmod(descriptor, mode)
    except BaseException:
        os.close(descriptor)
        filelock.unlink(scratch, missing_ok=True)
        raise
    return descriptor, scratch


def holdout_units(held):
    units = set()
    for patch in held["patches"]:
        units.add(patch["id"])
        units.update(f"{patch['id']}.{octant['index']}" for octant in patch.get("octants") or [])
    return units
