"""Gather the decided records of every pair into one release set.

For each pair, the record with the newest decision is taken, whether the pipeline made it
or it was migrated from the first runs; a pipeline record wins over a migrated one of the
same pair, since a migrated decision has no time of its own. A pair whose pipeline record
is not decided yet (deferred, or still under review) is held back whole: an older decision
of it is not released in the meantime. Of a record, the files a person decided and what
they rest on are copied whole; the event log, the stage settings and the derived state under
`work/` stay behind. `index.json` lists each pair with where it came
from, its decision and score, the hash of every file, and what its record does not hold
(a migrated record has no code commit, for example). Nothing is filled in.

    python -m volreg.cli release OUT RECORDS_DIR [RECORDS_DIR ...]
"""

import hashlib
import json
import shutil
from pathlib import Path

from . import record as rec

SCHEMA = "volreg-release/1"
# What a released pair holds: the shared part of its record.
FILES = (rec.PAIR, rec.HOLDOUT, rec.FINAL_REVIEW, rec.PATCH_REVIEWS, rec.TRANSFORM, rec.FIT)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def is_pipeline(record):
    return record.path("work").exists()


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
            "files": {name: sha256(record.path(name)) for name in FILES if record.exists(name)},
            "not_recorded": not_recorded(record)}


def release(out, roots):
    """Write the release set to `out`, which must not exist yet."""
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"{out} exists; a release is written whole, into a new folder")
    by_pair, pending, left_out = {}, {}, []
    for root in map(Path, roots):
        for directory in sorted(root.iterdir()):
            record = rec.Record(directory)
            if not record.exists(rec.PAIR):
                continue
            source = f"{root.name}/{directory.name}"
            final = record.read(rec.FINAL_REVIEW) if record.exists(rec.FINAL_REVIEW) else {}
            decision = final.get("decision")
            if not decision:
                left_out.append({"source": source, "pair": record.read(rec.PAIR)["pair"],
                                 "why": final.get("status") or "not decided"})
                if is_pipeline(record):
                    open_items = [item["text"] for item in final.get("revisit") or []
                                  if not item.get("done")]
                    pending[record.read(rec.PAIR)["pair"]] = (source, open_items)
            elif decision["publish"] == "nothing" or not record.exists(rec.TRANSFORM):
                left_out.append({"source": source, "pair": record.read(rec.PAIR)["pair"],
                                 "why": "decided to publish nothing"})
            else:
                by_pair.setdefault(record.read(rec.PAIR)["pair"], []).append((record, source))
    pairs = []
    for name, found in sorted(by_pair.items()):
        if name in pending:
            # The pair is being looked at again and nobody has decided yet: an older decision
            # is not published in the meantime.
            source, open_items = pending[name]
            why = f"held back: {source} is not decided" + (f" ({open_items[0]})" if open_items else "")
            left_out.extend({"source": older, "pair": name, "why": why} for _record, older in found)
            continue
        found.sort(key=lambda item: newness(item[0]))
        if len(found) > 1 and newness(found[-1][0]) == newness(found[-2][0]):
            raise ValueError(f"{name}: cannot tell which decision is newest: "
                             f"{', '.join(source for _record, source in found)}")
        record, source = found[-1]
        for _older, older_source in found[:-1]:
            left_out.append({"source": older_source, "pair": name, "why": f"superseded by {source}"})
        for file in FILES:
            if record.exists(file):
                target = out / name / file
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(record.path(file), target)
        pairs.append(entry(record, source, name))
    out.mkdir(parents=True, exist_ok=True)
    index = {"schema": SCHEMA, "created": rec.now(), "pairs": pairs,
             "left_out": sorted(left_out, key=lambda item: item["source"])}
    (out / "index.json").write_text(json.dumps(index, indent=1) + "\n")
    return index
