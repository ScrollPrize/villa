"""Gather the decided records of every pair into one release set.

For each pair, the record with the newest decision is taken, whether the pipeline made it
or it was migrated from the first runs; a pipeline record wins over a migrated one of the
same pair, since a migrated decision has no time of its own. A pair whose pipeline record
is not decided yet (deferred, or still under review) is held back whole: an older decision
of it is not released in the meantime. A pipeline record that has been redone is released
only once its store has run since the last redo, wrote the files it holds, and was run
under the decision it holds now (the same fit, the same variant); until then it is left
out, and if it is the newest decision of its pair, the pair is held back whole too. A
record with no redo in its history is taken as it is. Of a record, the files a person
decided and what they rest on are copied whole, as they were read at one moment; the event
log, the stage settings and the derived state under `work/` stay behind. `index.json` lists
each pair with where it came from, its decision and score, the hash of every file, and what
its record does not hold (a migrated record has no code commit, for example). Nothing is
filled in. The set is written beside `out` and moved there whole: a release that fails
leaves nothing.

    python -m volreg.cli release OUT RECORDS_DIR [RECORDS_DIR ...]
"""

import hashlib
import json
import os
import secrets
import shutil
import time
from pathlib import Path

from . import record as rec
from .engine import digest_contents

SCHEMA = "volreg-release/1"
# What a released pair holds: the shared part of its record.
FILES = (rec.PAIR, rec.HOLDOUT, rec.FINAL_REVIEW, rec.PATCH_REVIEWS, rec.TRANSFORM, rec.FIT)
# What says whether a pipeline record's files are what its store wrote since the last redo.
STORE_STAMP, REDO_MARKS = "work/stages/store.json", "work/stages/redo.json"
# How often a record that changes while it is read is read again.
READS = 20


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def is_pipeline(record):
    return record.path("work").exists()


class Snapshot:
    """A record's shared files (and, for a pipeline record, its store stamp and redo marks)
    as read at one moment: read until two reads in a row agree, so a store writing meanwhile
    is seen before or after, never half way. Reads like a Record."""

    def __init__(self, record, contents):
        self.directory, self.contents = record.directory, contents

    @classmethod
    def take(cls, record):
        """The snapshot, or None if the record never stood still long enough to read."""
        names = FILES + ((STORE_STAMP, REDO_MARKS) if is_pipeline(record) else ())
        before = None
        for attempt in range(READS):
            now = {}
            for name in names:
                try:
                    now[name] = record.path(name).read_bytes()
                except FileNotFoundError:
                    pass
            if now == before:
                return cls(record, now)
            before = now
            if attempt:
                time.sleep(0.05)
        return None

    def path(self, name):
        return self.directory / name

    def exists(self, name):
        return name in self.contents

    def read(self, name):
        if name not in self.contents:
            raise FileNotFoundError(self.path(name))
        return json.loads(self.contents[name])

    def sha256(self, name):
        return hashlib.sha256(self.contents[name]).hexdigest()


def not_stored_since_redo(record):
    """Why a pipeline record redone since its decision was stored cannot be released as it
    stands -- its store has not finished since the last redo of it or of anything it rests on,
    the files are not the ones that store wrote, or they were stored under another decision
    than the one it holds now (another fit, another variant) -- or None. None also for a
    record with no redo in its history, or one that is not the pipeline's: it is as it was."""
    if not is_pipeline(record):
        return None
    from . import stages   # not at the top: stages reads what this module reads
    marks = record.read(REDO_MARKS) if record.exists(REDO_MARKS) else {"stages": {}}
    ancestors = stages.pipeline(record.directory).ancestors("store")
    mark = max((marks.get("stages", {}).get(stage, 0) for stage in ancestors), default=0)
    if not mark:
        return None
    stamp = record.read(STORE_STAMP) if record.exists(STORE_STAMP) else {}
    if stamp.get("status") != "ok" or stamp.get("redo_mark") != mark:
        return "a redo came after its decision was stored; store it again"
    outputs = stamp.get("outputs") or []
    if (not set(outputs) <= set(FILES) or (rec.TRANSFORM in outputs) != record.exists(rec.TRANSFORM)
            or digest_contents((name, record.contents.get(name)) for name in outputs)
            != stamp.get("output_digest")):
        return "its record is not what its store wrote after the redo; store it again"
    decision = record.read(rec.FINAL_REVIEW)["decision"]
    stored = record.read(rec.FIT) if record.exists(rec.FIT) else {}
    if stored.get("fit") != decision.get("fit") or stored.get("decision") != {
            "fit": decision.get("fit"), "publish": decision.get("publish")}:
        return "its decision is not the one its fit was stored under; store it again"
    return None


def newness(record):
    """Order of the decisions of one pair: a pipeline record after a migrated one, then by
    the decision's time."""
    decision = record.read(rec.FINAL_REVIEW)["decision"]
    return is_pipeline(record), decision.get("at") or ""


def not_recorded(record):
    """What the record leaves out that a pipeline record has."""
    fit, final = record.read(rec.FIT), record.read(rec.FINAL_REVIEW)
    missing = []
    if not (fit.get("code") or {}).get("commit"):
        missing.append("code commit")
    if not final["decision"].get("at"):
        missing.append("decision time")
    if "drift" not in fit:
        missing.append("hold-out drift under the fit")
    items = [item for round_ in final.get("rounds") or [] for item in round_["items"]]
    if any("image_sha256" not in item for item in items):
        missing.append("image hashes of blind items")
    return missing


def answers(record):
    held = record.read(rec.HOLDOUT)["patches"]
    found = [unit["answer"]["answer"] for patch in held
             for unit in [patch, *(patch.get("octants") or [])] if unit.get("answer")]
    return {answer: found.count(answer) for answer in rec.ANSWERS if answer in found}


def entry(record, source, folder):
    pair, fit = record.read(rec.PAIR), record.read(rec.FIT)
    decision = record.read(rec.FINAL_REVIEW)["decision"]
    scores = fit["scores"][decision["publish"]]
    return {"pair": pair["pair"], "folder": folder, "source": source,
            "made_by": "pipeline" if is_pipeline(record) else "first runs, migrated",
            "decision": decision,
            "score": {key: scores.get(key) for key in
                      ("lines_up", "median_fine_voxels", "p90_fine_voxels", "far",
                       "median_um", "p90_um")} | {"passes": scores["bar"]["passes"]},
            "bar": fit.get("bar"), "answers": answers(record),
            "files": {name: record.sha256(name) for name in FILES if record.exists(name)},
            "not_recorded": not_recorded(record)}


def release(out, roots):
    """Write the release set to `out`, which must not exist yet."""
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"{out} exists; a release is written whole, into a new folder")
    by_pair, pending, unresolved, left_out = {}, {}, {}, []
    for root in map(Path, roots):
        for directory in sorted(root.iterdir()):
            if not rec.Record(directory).exists(rec.PAIR):
                continue
            source = f"{root.name}/{directory.name}"
            record = Snapshot.take(rec.Record(directory))
            if record is None or not record.exists(rec.PAIR):
                name = rec.Record(directory).read(rec.PAIR)["pair"]
                why = "its record kept changing while it was read; release again"
                left_out.append({"source": source, "pair": name, "why": why})
                unresolved.setdefault(name, []).append(((True, "\uffff"), source, why))
                continue
            name = record.read(rec.PAIR)["pair"]
            final = record.read(rec.FINAL_REVIEW) if record.exists(rec.FINAL_REVIEW) else {}
            decision = final.get("decision")
            why = decision and not_stored_since_redo(record)
            if not decision:
                left_out.append({"source": source, "pair": name,
                                 "why": final.get("status") or "not decided"})
                if is_pipeline(record):
                    open_items = [item["text"] for item in final.get("revisit") or []
                                  if not item.get("done")]
                    pending[name] = (source, open_items)
            elif why:
                left_out.append({"source": source, "pair": name, "why": why})
                unresolved.setdefault(name, []).append((newness(record), source, why))
            elif decision["publish"] == "nothing" or not record.exists(rec.TRANSFORM):
                left_out.append({"source": source, "pair": name, "why": "decided to publish nothing"})
            else:
                by_pair.setdefault(name, []).append((record, source))
    pairs, written = [], []
    for name, found in sorted(by_pair.items()):
        if name in pending:
            # The pair is being looked at again and nobody has decided yet: an older decision
            # is not published in the meantime.
            source, open_items = pending[name]
            why = f"held back: {source} is not decided" + (f" ({open_items[0]})" if open_items else "")
            left_out.extend({"source": older, "pair": name, "why": why} for _record, older in found)
            continue
        found.sort(key=lambda item: newness(item[0]))
        newer = [item for item in unresolved.get(name, []) if item[0] >= newness(found[-1][0])]
        if newer:
            # Its newest decision was redone and is not stored again yet: what it replaces
            # is not published in the meantime.
            _newness, source, why = max(newer)
            left_out.extend({"source": older, "pair": name, "why": f"held back: {source} is newer: {why}"}
                            for _record, older in found)
            continue
        if len(found) > 1 and newness(found[-1][0]) == newness(found[-2][0]):
            raise ValueError(f"{name}: cannot tell which decision is newest: "
                             f"{', '.join(source for _record, source in found)}")
        record, source = found[-1]
        for _older, older_source in found[:-1]:
            left_out.append({"source": older_source, "pair": name, "why": f"superseded by {source}"})
        written.extend((Path(name) / file, record.contents[file]) for file in FILES if record.exists(file))
        pairs.append(entry(record, source, name))
    index = {"schema": SCHEMA, "created": rec.now(), "pairs": pairs,
             "left_out": sorted(left_out, key=lambda item: item["source"])}
    written.append((Path("index.json"), (json.dumps(index, indent=1) + "\n").encode()))
    # Written beside, then moved into place in one step: a release is there whole or not at all.
    partial = out.with_name(f".{out.name}.{secrets.token_hex(4)}.partial")
    try:
        for path, content in written:
            target = partial / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        if out.exists():
            raise FileExistsError(f"{out} exists; a release is written whole, into a new folder")
        os.rename(partial, out)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return index
