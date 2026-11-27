"""Gather the decided records of every pair into one release set.

For each pair, the record with the newest decision is taken, whether the pipeline made it
or it was migrated from the first runs; a pipeline record wins over a migrated one of the
same pair, since a migrated decision has no time of its own. A pair whose pipeline record
is not decided yet (deferred, or still under review) is held back whole: an older decision
of it is not released in the meantime. A pipeline record that has been redone is released
only once its store has run since the last redo, wrote the files it holds, and was run
under the decision it holds now (the same fit, the same variant); until then it is left
out, and if it is the newest decision of its pair, the pair is held back whole too.

Every pipeline record is also checked for what its store bound together (see
`stored_mismatch`): the fit, the variant and the transform stored must be the ones its
decision names, whether or not it was ever redone. A record whose store has not run since
its decision changed, or stopped between its writes, is left out the same way. A record
stored before stores bound their files (by main, say) is checked on what it holds: the fit
it names and whether its transform has a bend; one that agrees with its decision is taken
as it is. Of a record, the files a person
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

from . import filelock
from . import record as rec
from . import scoring
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
        # The event log too, for a pipeline record: the answers' and decisions' only source,
        # which `scores_problem` checks the files against. Read, never released.
        names = FILES + ((STORE_STAMP, REDO_MARKS, rec.EVENTS) if is_pipeline(record) else ())
        before = None
        for attempt in range(READS):
            now = {}
            for name in names:
                try:
                    # a read that lands while Windows replaces the file is refused; ask again
                    now[name] = filelock.retry(record.path(name).read_bytes)
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

    def read_bytes(self, name):
        if name not in self.contents:
            raise FileNotFoundError(self.path(name))
        return self.contents[name]

    def sha256(self, name):
        return hashlib.sha256(self.contents[name]).hexdigest()

    def events(self):
        content = self.contents.get(rec.EVENTS, b"").decode()
        return [json.loads(line) for line in content.splitlines() if line.strip()]


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
        return "a redo came after its decision was stored; store it again with `volreg redo <pair> store`"
    outputs = stamp.get("outputs") or []
    if (not set(outputs) <= set(FILES) or (rec.TRANSFORM in outputs) != record.exists(rec.TRANSFORM)
            or digest_contents((name, record.contents.get(name)) for name in outputs)
            != stamp.get("output_digest")):
        return "its record is not what its store wrote after the redo; store it again with `volreg redo <pair> store`"
    decision = record.read(rec.FINAL_REVIEW)["decision"]
    stored = record.read(rec.FIT) if record.exists(rec.FIT) else {}
    if stored.get("fit") != decision.get("fit") or stored.get("decision") != {
            "fit": decision.get("fit"), "publish": decision.get("publish")}:
        return "its decision is not the one its fit was stored under; store it again with `volreg redo <pair> store`"
    return None


def stored_mismatch(record, decision, legacy_too=True):
    """Why what a record stores is not the one result its decision names, or None.

    A store binds what it writes in `fit.json` (`stored`): the fit, the variant, and the
    digest of the transform written with it (none for publishing nothing). All of it must be
    the decision's, the scores it holds must be that fit's, and the transform there must be
    the one named: a store that stopped between its writes, or has not run since the decision
    changed, fails here. A record stored before stores bound their files is checked, if
    `legacy_too`, on what it holds: the fit it names, that an affine decision has a
    transform without a bend, and that a bend decision has one with a bend if the fit it
    names had one. `record` is a Record or a Snapshot."""
    if not record.exists(rec.FIT):
        return "nothing is stored for its decision yet; store it" if legacy_too else None
    stored = record.read(rec.FIT)
    content = record.read_bytes(rec.TRANSFORM) if record.exists(rec.TRANSFORM) else None
    fit, publish = decision.get("fit"), decision.get("publish")
    binding = stored.get("stored")
    if binding is not None:
        if (binding.get("fit"), binding.get("publish"), stored.get("fit")) != (fit, publish, fit):
            return (f"it stores {binding.get('publish')} of {binding.get('fit')}, but its decision is "
                    f"{publish} of {fit}; store it again with `volreg redo <pair> store`")
        scored = (stored.get("scored") or {}).get("fit", fit)
        if scored != fit:
            return f"its scores are of {scored}, not of {fit}; store it again with `volreg redo <pair> store`"
        if (content and hashlib.sha256(content).hexdigest()) != binding.get("transform_sha256"):
            return ("its transform is not the one its fit.json was stored with (a store that did not finish, "
                    "or a file changed since); store it again with `volreg redo <pair> store`")
        return None
    if not legacy_too:
        return None
    if stored.get("decision") not in (None, {"fit": fit, "publish": publish}):
        old = stored["decision"]
        return f"it stores {old.get('publish')} of {old.get('fit')}, but its decision is {publish} of {fit}; store it again with `volreg redo <pair> store`"
    if stored.get("fit") not in (None, fit):
        return f"it stores {stored['fit']}, but its decision is about {fit}; store it again with `volreg redo <pair> store`"
    if publish in ("bend", "affine") and content is None:
        return "it stores no transform; store it again with `volreg redo <pair> store`"
    bent = content is not None and json.loads(content).get("deformation") is not None
    if publish == "affine" and bent:
        return "it stores the bend, but its decision is the affine; store it again with `volreg redo <pair> store`"
    if publish == "bend" and stored.get("bend") and not bent:
        # The fit has a bend (its metadata says so), and what is stored is without it.
        return "it stores the affine, but its decision is the bend; store it again with `volreg redo <pair> store`"
    return None


def scores_problem(record, decision):
    """Why the scores a record publishes are not those of the hold-out answers it holds, or
    not what its decision and any override cover, or None. Recomputed from the files as they
    are (a Record or a Snapshot), not taken from what Store said.

    A store binds the answers its scores used (`answers_sha256`, see `scoring.answers_digest`)
    and the basis the decision was checked against (`scoring.basis`): both must still be what
    the record's own `holdout.json` and `fit.json` give, and a decision that names its basis
    must name that one. A record stored before stores bound them (by main, an earlier version,
    or migrated from the first runs) is checked on what it holds: the published variant's
    score must name the units the answers line up (`per_unit`), or count as many. Either way
    the published variant must pass the bar, or the decision's override must name every part
    it misses.

    A pipeline record is checked against its event log as well, the answers' and decisions'
    only source: an answer logged that its `holdout.json` does not hold (the rewrite after
    it failed) means the scores are not of the answers there are, and a decision that names
    no scores must be one the log shows was made on the scores stored
    (`scoring.legacy_approval_problem`). A released pair has no event log, and is checked on
    its shipped files alone."""
    publish = decision.get("publish")
    if publish not in ("bend", "affine") or not record.exists(rec.FIT):
        return None
    stored = record.read(rec.FIT)
    held = record.read(rec.HOLDOUT) if record.exists(rec.HOLDOUT) else {"patches": []}
    used = scoring.answers_digest(held)
    binding = stored.get("stored") or {}
    result = (stored.get("scores") or {}).get(publish)
    if result is None:
        return f"it stores no score for the {publish}; store it again with `volreg redo <pair> store`"
    again = "the decision is made again on the new scores, and stored"
    events = record.events() if is_pipeline(record) and record.exists(rec.EVENTS) else None
    if events is not None:
        behind = rec.answers_behind(held, events)
        if behind:
            return (f"its event log holds hold-out answers its holdout.json does not ({', '.join(behind)}: "
                    f"the rewrite after them failed), so its scores are not of the answers there are; it is "
                    f"released once {again}")
        if decision.get("basis") is None and any(event["kind"] == "decision" for event in events):
            problem = scoring.legacy_approval_problem(events, decision)
            if problem:
                return f"its decision names no scores, and {problem}; it is released once {again}"
    if binding.get("answers_sha256") is not None:
        if binding["answers_sha256"] != used:
            return (f"a hold-out answer changed since it was stored, so its scores are not those of "
                    f"the answers it holds; it is released once {again}")
        basis = scoring.basis(binding.get("fit_digest"), stored.get("scores"), stored.get("bar"), used)
        if binding.get("basis") != basis:
            return "its scores are not the ones it was stored with; store it again with `volreg redo <pair> store`"
        if decision.get("basis") is not None and decision["basis"] != basis:
            return f"its decision was made on other scores than it stores; it is released once {again}"
    elif decision.get("basis") is not None:
        return ("its decision names the scores it was made on, and its store does not; store it again "
                "with `volreg redo <pair> store`")
    units = {unit[0] for unit in scoring.answers_used(held)}
    named = result.get("per_unit")
    if (named is not None and set(named) != units) or result.get("lines_up", 0) != len(units):
        return (f"its {publish} score is on {result.get('lines_up', 0)} units lining up, but its hold-out "
                f"answers line up {len(units)} (an answer changed since it was scored); it is released "
                f"once {again}")
    missed = scoring.bar_failures(result)
    covered = set((decision.get("override") or {}).get("bar_missed") or [])
    if not set(missed) <= covered:
        return (f"its {publish} misses the bar ({', '.join(missed)}) and its decision's override does not "
                f"name {', '.join(sorted(set(missed) - covered))}; decide again")
    return None


def decision_of(record):
    """A record's decision: a pipeline record's is the latest its event log holds, the
    decisions' only source; any other's is the one in its final-review.json."""
    final = record.read(rec.FINAL_REVIEW) if record.exists(rec.FINAL_REVIEW) else {}
    if is_pipeline(record):
        final = rec.with_logged_final(final, record.events())
    return final.get("decision")


def decision_behind(record, decision):
    """Why a record's final-review.json does not hold `decision`, its `decision_of`, or None."""
    final = record.read(rec.FINAL_REVIEW) if record.exists(rec.FINAL_REVIEW) else {}
    if final.get("decision") != decision:
        return ("its event log holds a newer decision than its final-review.json (the rewrite after it "
                "failed); it is released once its store runs again")
    return None


def unreleasable(record, decision):
    """Why a decided record cannot be released as it stands, or None: a final-review.json
    behind its event log (`decision_behind`), a redo since its store
    (`not_stored_since_redo`), a stored result that is not the one its decision names
    (`stored_mismatch`; a migrated record has nothing to check but what a store bound), or
    scores that are not those of its answers, or not covered by its decision
    (`scores_problem`)."""
    return decision_behind(record, decision) or not_stored_since_redo(record) or (
        decision["publish"] != "nothing" and (stored_mismatch(record, decision, legacy_too=is_pipeline(record))
                                              or scores_problem(record, decision)))


def newness(record):
    """Order of the decisions of one pair: a pipeline record after a migrated one, then by
    the decision's time."""
    return is_pipeline(record), decision_of(record).get("at") or ""


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
            decision = decision_of(record)
            why = decision and unreleasable(record, decision)
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
