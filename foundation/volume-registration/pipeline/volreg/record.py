"""The per-pair record: what a person decided, kept apart from everything recomputable.

    <record>/pair.json            the two volumes and the start transform
    <record>/holdout.json         the hold-out patches, their own measurements, the answers
    <record>/final-review.json    the blind rounds and the decision
    <record>/patch-reviews.json   verdicts and notes on fit patches (the first reviews had them)
    <record>/fit/transform.json   the final transform, in the atlas's format
    <record>/fit/fit.json         how it was fitted, the points it rests on, its scores
    <record>/events.jsonl         every answer and job, append-only

Files other than `events.jsonl` are replaced whole, in one step, so a reader never sees half
a file. Coordinates are ZYX level-0 voxels of their own volume, as in the cores.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

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

    def write(self, name, data):
        """Replace a file in one step, with the schema stamped in."""
        target = self.path(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, dict) and name.endswith(".json") and name != TRANSFORM:
            data = {"schema": SCHEMA} | data
        scratch = target.with_name(target.name + ".tmp")
        with open(scratch, "w") as handle:
            json.dump(data, handle, indent=1)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(scratch, target)

    def log(self, kind, **fields):
        """Append one event; `at` defaults to now. Never rewrites what is there."""
        self.directory.mkdir(parents=True, exist_ok=True)
        event = {"at": fields.pop("at", None) or now(), "kind": kind} | fields
        with open(self.path(EVENTS), "a") as handle:
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

    def refresh_holdout(self, patches=None):
        """Rewrite `holdout.json` with `patches` (or the ones it holds) and the latest answers."""
        held = self.read(HOLDOUT) if self.exists(HOLDOUT) else {"patches": []}
        if patches is not None:
            held = held | {"patches": patches}
        latest = self.answers()
        for patch in held["patches"]:
            patch["answer"] = latest.get(patch["id"])
            for octant in patch.get("octants") or []:
                octant["answer"] = latest.get(f"{patch['id']}.{octant['index']}")
        held.pop("schema", None)
        self.write(HOLDOUT, held)
        return held


def holdout_units(held):
    units = set()
    for patch in held["patches"]:
        units.add(patch["id"])
        units.update(f"{patch['id']}.{octant['index']}" for octant in patch.get("octants") or [])
    return units

