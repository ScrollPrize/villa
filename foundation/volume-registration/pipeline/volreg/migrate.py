"""Convert a session of the first runs (`fit_session.py`, schema volreg-session/0.1) into a record.

Only what a person decided and what it rests on is carried over: the hold-out patches with
their own measurements and answers, the blind rounds, the verdicts and notes on fit patches,
and the final fit with the points it used. Blocks and fits nobody chose stay behind; they
can be recomputed. The decision comes from the staged transform's provenance, when there is
one, or from `UNDECIDED`.

    python -m volreg.migrate SESSION OUT --handle H --email E [--staged DIR]
"""

import argparse
import json
from pathlib import Path

import numpy as np

from . import record as rec
from .scoring import BAR, bar_check, score

SESSION_SCHEMA = "volreg-session/0.1"
OCTANT_MIN_MATCH = 0.85          # as in fit_session.py: an octant that measured well
TEST_STRETCHES = 5

# Sessions with answers but nothing staged, and why (from the staged README, 2026-10-26).
UNDECIDED = {
    "paris4-auto": "held: the DLS scan's appearance limits the local fit to about 2-2.5 fine voxels",
    "paris4-1um-auto": "fails the bar on p90 and on patches over 5, also after a refit from 150 more patches",
    "0500p2-055-auto": "open: the start does not capture; no answers yet",
}


def zyx4(matrix_xyz):
    """A 3x4 XYZ matrix as the 4x4 ZYX one."""
    matrix = np.vstack([np.asarray(matrix_xyz, float)[:3], [0, 0, 0, 1]])
    swap = np.eye(4)[[2, 1, 0, 3]]
    return swap @ matrix @ swap


def apply(matrix, points):
    points = np.asarray(points, float)
    return points @ matrix[:3, :3].T + matrix[:3, 3]


class Session:
    def __init__(self, directory):
        self.directory = Path(directory)
        with open(self.directory / "session.json") as handle:
            self.state = json.load(handle)
        if self.state.get("schema") != SESSION_SCHEMA:
            raise ValueError(f"{directory}: not a {SESSION_SCHEMA} session")
        self.fixed_is_coarse = self.state["coarse_side"] == "fixed"
        coarse = self.state[self.state["coarse_side"]]
        fine = self.state["moving" if self.fixed_is_coarse else "fixed"]
        self.coarse_um, self.fine_um = coarse["voxel_um"], fine["voxel_um"]

    def fit(self, fit_id):
        return next(fit for fit in self.state["fits"] if fit["id"] == fit_id)

    def fine_to_coarse(self, fit_id):
        matrix_xyz = (self.state["start"]["matrix_xyz"] if fit_id == "start"
                      else self.fit(fit_id)["matrix_xyz"])
        matrix = zyx4(matrix_xyz)
        return matrix if self.fixed_is_coarse else np.linalg.inv(matrix)

    def stretch(self, patch):
        along = self.state["sampling"]["rounds"][patch["round"]].get("along") or [0.0, 1.0]
        return min(int(along[0] * TEST_STRETCHES + 0.5), TEST_STRETCHES - 1)

    def octants(self, patch):
        """The octants that measured well, at their NCC optimum, with any answers."""
        measurement = patch["measurement"]
        sign = self.state["sampling"]["sign"]
        to_fine = np.linalg.inv(self.fine_to_coarse(measurement["under"]))
        answers = patch.get("test_octants") or {}
        found = []
        for index, octant in enumerate(measurement.get("octants") or []):
            if "rejected" in octant or octant["check"]["peak_match"] < OCTANT_MIN_MATCH:
                continue
            centre = np.asarray(octant["centre_zyx"], float)
            fine = apply(to_fine, centre + sign * np.asarray(octant["check"]["peak_zyx"], float))
            found.append({"index": index, "corner_zyx": octant["corner_zyx"],
                          "centre_zyx": centre.tolist(), "fine_zyx": fine.tolist(),
                          "match": octant["check"]["peak_match"],
                          "answer": answers.get(str(index))})
        return found


def answer(given, reviewer):
    return None if not given else {"answer": given["answer"], "at": given["at"], "by": reviewer}


def holdout(session, reviewer, every_octant=False):
    """The hold-out patches; with `every_octant`, every octant that measured well is listed so
    it can be answered on its own, else only the ones that were."""
    patches = []
    for patch in session.state["patches"]:
        if not patch.get("test"):
            continue
        measurement = patch["measurement"]
        entry = {"id": patch["id"], "origin_zyx": patch["origin_zyx"],
                 "stretch": session.stretch(patch), "auto": patch["auto"],
                 "answer": answer(patch.get("test_answer"), reviewer),
                 "image_sha256": None}     # the first runs rendered images on request
        if measurement:
            entry |= {"centre_zyx": measurement["centre_zyx"], "fine_zyx": measurement["fine_zyx"],
                      "match": measurement["match"], "measured_under": measurement["under"]}
            octants = session.octants(patch)
            for octant in octants:
                octant["answer"] = answer(octant["answer"], reviewer)
            if every_octant or any(octant["answer"] for octant in octants):
                entry["octants"] = octants
        if patch.get("human"):      # a verdict from the review loop, given before test answers were
            entry["verdict"] = patch["human"] | {"by": reviewer}
        if patch.get("note"):
            entry["note"] = patch["note"] | {"by": reviewer}
        patches.append(entry)
    return {"patch_coarse_voxels": session.state["sampling"]["block"],
            "stretches": TEST_STRETCHES, "patches": patches}


def patch_reviews(session, reviewer):
    reviews = []
    for patch in session.state["patches"]:
        if patch.get("test") or not (patch.get("human") or patch.get("note")):
            continue
        entry = {"id": patch["id"], "origin_zyx": patch["origin_zyx"]}
        if patch.get("human"):
            entry["verdict"] = patch["human"] | {"by": reviewer}
        if patch.get("note"):
            entry["note"] = patch["note"] | {"by": reviewer}
        if patch["measurement"]:
            entry |= {"centre_zyx": patch["measurement"]["centre_zyx"],
                      "fine_zyx": patch["measurement"]["fine_zyx"]}
        reviews.append(entry)
    return reviews


def blind_rounds(session, reviewer):
    rounds = []
    for round_ in session.state.get("comparisons") or []:
        rounds.append(round_ | {"by": reviewer,
                                "items": [item | {"spread": round(item["spread"], 4)}
                                          for item in round_["items"]]})
    return rounds


def staged_for(session_dir, staged):
    """The provenance of the transform staged from this session, if any."""
    if staged is None:
        return None
    for path in sorted(Path(staged).glob("*.provenance.json")):
        with open(path) as handle:
            provenance = json.load(handle)
        if Path(provenance["session"]).name == Path(session_dir).name:
            return path, provenance
    return None


def migrate(session_dir, out, reviewer, staged=None):
    session = Session(session_dir)
    state = session.state
    record = rec.Record(out)
    found = staged_for(session_dir, staged)
    pair_name = (found[1]["pair"].removesuffix(".json") if found
                 else Path(state["start"]["source"]).name.removesuffix(".json"))

    record.write(rec.PAIR, {
        "pair": pair_name,
        "fixed": state["fixed"], "moving": state["moving"], "coarse_side": state["coarse_side"],
        "start": state["start"],
        "sampling": {key: state["sampling"].get(key)
                     for key in ("block", "sign", "fine_scale", "coarse_shape", "fine_shape")},
        "created": state["created"],
        "migrated_from": {"session": Path(session_dir).name, "schema": SESSION_SCHEMA,
                          "at": rec.now()},
    })
    held = holdout(session, reviewer)
    record.write(rec.HOLDOUT, {"pair": pair_name} | held)
    reviews = patch_reviews(session, reviewer)
    if reviews:
        record.write(rec.PATCH_REVIEWS, {"pair": pair_name, "patches": reviews})

    decision = None
    if found:
        path, provenance = found
        fit_id, published = provenance["fit"], provenance["published"]
        decision = {"fit": fit_id, "publish": "bend" if published == "affine + bend" else "affine",
                    "reason": provenance["why"], "by": reviewer, "at": None, "override": None}
        transform = json.loads(Path(path.with_name(provenance["pair"])).read_text())
    else:
        fit_id = state["current"] if state["fits"] else None
        transform = None
        if fit_id:
            with open(session.directory / session.fit(fit_id)["transform"]) as handle:
                transform = json.load(handle)
    record.write(rec.FINAL_REVIEW, {
        "pair": pair_name, "rounds": blind_rounds(session, reviewer), "decision": decision,
        "status": "decided" if decision else UNDECIDED.get(Path(session_dir).name, "no decision"),
    })

    if fit_id:
        fit = session.fit(fit_id)
        record.write(rec.TRANSFORM, transform)
        with open(session.directory / fit["transform"]) as handle:
            full = json.load(handle)
        scores = {
            "bend": score(full, held, session.fixed_is_coarse, session.fine_um),
            "affine": score(full, held, session.fixed_is_coarse, session.fine_um, affine_only=True),
        }
        for value in scores.values():
            value["bar"] = bar_check(value)
        if decision and not scores[decision["publish"]]["bar"]["passes"]:
            missed = [name for name, ok in scores[decision["publish"]]["bar"]["checks"].items() if not ok]
            decision["override"] = {"by": reviewer, "reason": decision["reason"],
                                    "bar_missed": missed}
            record.write(rec.FINAL_REVIEW, record.read(rec.FINAL_REVIEW) | {"decision": decision})
        record.write(rec.FIT, {
            "pair": pair_name, "fit": fit_id,
            "points": {"ids": fit["used"],
                       "fixed_xyz": full["fixed_landmarks"], "moving_xyz": full["moving_landmarks"],
                       "weighted_down": (fit["bend"] or {}).get("weighted_down", {})},
            # Fits from before these choices existed used what are now the defaults.
            "model": fit["model"], "hold_out": fit.get("hold_out", "regions"),
            "robust": fit.get("robust", "ransac"), "unit": fit.get("unit", "patches"), "bend": fit["bend"] and {key: fit["bend"][key] for key in
                                                        ("regularisation", "effective_parameters")},
            "fine_um": session.fine_um, "coarse_um": session.coarse_um, "bar": BAR,
            "scores": scores,
            "code": {"migrated_from_session_fit": fit_id},
        })

    for event in sorted(human_events(held, reviews, state), key=lambda event: event["at"] or ""):
        record.log(event.pop("kind"), **event, by=reviewer, migrated=True)
    record.log("migrated", session=Path(session_dir).name, by=reviewer)
    return record


def human_events(held, reviews, state):
    """Every answer, verdict and note the session holds, as events. Blind picks carry no
    time of their own in the first runs; they take their round's."""
    for patch in held["patches"]:
        if patch["answer"]:
            yield {"kind": "holdout-answer", "at": patch["answer"]["at"], "unit": patch["id"],
                   "answer": patch["answer"]["answer"]}
        for octant in patch.get("octants") or []:
            if octant["answer"]:
                yield {"kind": "holdout-answer", "at": octant["answer"]["at"],
                       "unit": f"{patch['id']}.{octant['index']}", "answer": octant["answer"]["answer"]}
        if patch.get("verdict"):
            yield {"kind": "patch-verdict", "at": patch["verdict"]["at"], "patch": patch["id"],
                   "verdict": patch["verdict"]["verdict"]}
        if patch.get("note"):
            yield {"kind": "note", "at": patch["note"]["at"], "patch": patch["id"],
                   "text": patch["note"]["text"]}
    for patch in reviews:
        if patch.get("verdict"):
            yield {"kind": "patch-verdict", "at": patch["verdict"]["at"], "patch": patch["id"],
                   "verdict": patch["verdict"]["verdict"]}
        if patch.get("note"):
            yield {"kind": "note", "at": patch["note"]["at"], "patch": patch["id"],
                   "text": patch["note"]["text"]}
    for round_ in state.get("comparisons") or []:
        for index, item in enumerate(round_["items"]):
            if item["choice"] is not None:
                yield {"kind": "blind-pick", "at": round_["at"], "round": round_["id"],
                       "item": index, "patch": item["patch"], "choice": item["choice"],
                       "time_is_the_rounds": True}


def import_session(session_dir, record_dir, pair_dir):
    """Take a first run's session into a pipeline pair, so nothing it measured or a person
    answered has to be done again: every patch with its measurement and cached blocks
    (the hold-out ones frozen as they were judged), every fit (as `prior-<id>`, the last
    one current, so the next fit goes on from it), and the answers, notes and verdicts
    from the migrated record's log, under the patches' new ids. Blind rounds and the
    decision stay behind: they judged a fit the pipeline has not made. Only into a pair of
    the same two volumes and sign convention that has no hold-out set yet."""
    import shutil
    old = Session(session_dir)
    pair = Path(pair_dir)
    target = pair / "work" / "session"
    with open(target / "session.json") as handle:
        state = json.load(handle)
    for side in ("fixed", "moving"):
        if state[side]["url"] != old.state[side]["url"]:
            raise ValueError(f"the {side} volume differs: {state[side]['url']} vs {old.state[side]['url']}")
    if state["sampling"]["sign"] != old.state["sampling"]["sign"]:
        raise ValueError("the two sessions disagree on the sign convention")
    if any(patch.get("test") for patch in state["patches"]):
        raise ValueError("the pair already has a hold-out set")
    source = {"session": str(old.directory)}

    rounds, ids = {}, {}
    for patch in old.state["patches"]:
        if patch["round"] not in rounds:
            rounds[patch["round"]] = len(state["sampling"]["rounds"])
            state["sampling"]["rounds"].append(old.state["sampling"]["rounds"][patch["round"]]
                                               | {"imported_from": source | {"round": patch["round"]}})
        ids[patch["id"]] = f"p{len(state['patches']) + len(ids) + 1:04d}"
    fit_ids = {"start": "start"} | {fit["id"]: f"prior-{fit['id']}" for fit in old.state["fits"]}

    def unit(name):
        patch_id, dot, octant = name.partition(".")
        return ids.get(patch_id, patch_id) + dot + octant

    for patch in old.state["patches"]:
        moved = json.loads(json.dumps(patch)) | {"id": ids[patch["id"]], "round": rounds[patch["round"]],
                                                 "imported_from": patch["id"]}
        if moved["measurement"]:
            moved["measurement"]["under"] = fit_ids[moved["measurement"]["under"]]
        state["patches"].append(moved)
        block = old.directory / "blocks" / f"{patch['id']}.npz"
        if block.exists():
            shutil.copyfile(block, target / "blocks" / f"{ids[patch['id']]}.npz")
    for fit in old.state["fits"]:
        moved = json.loads(json.dumps(fit)) | {
            "id": fit_ids[fit["id"]], "transform": f"fits/{fit_ids[fit['id']]}.json",
            "parent": fit_ids.get(fit.get("parent"), fit.get("parent")),
            "measured_under": [fit_ids.get(f, f) for f in fit.get("measured_under") or []],
            "imported_from": source | {"fit": fit["id"]}}
        for key in ("candidates", "ransac_dropped", "used", "guard_failures"):
            if isinstance(moved.get(key), list):
                moved[key] = [unit(name) if isinstance(name, str) else name for name in moved[key]]
        shutil.copyfile(old.directory / fit["transform"], target / moved["transform"])
        state["fits"].append(moved)
    if old.state["fits"]:
        state["current"] = fit_ids[old.state["current"]]
    scratch = target / "session.json.tmp"
    scratch.write_text(json.dumps(state, indent=1))
    scratch.replace(target / "session.json")

    record = rec.Record(pair)
    carried = 0
    for event in rec.Record(record_dir).events():
        if event["kind"] not in ("holdout-answer", "note", "patch-verdict"):
            continue
        key = "unit" if event["kind"] == "holdout-answer" else "patch"
        if event.get(key, "").partition(".")[0] not in ids:
            continue
        fields = {k: v for k, v in event.items() if k not in ("kind", key)}
        fields |= {key: unit(event[key]),
                   "imported_from": {"record": str(Path(record_dir)), key: event[key]}}
        record.log(event["kind"], **fields)
        carried += 1
    record.log("session-imported", **source, record=str(Path(record_dir)), patches=ids,
               fits=fit_ids, current=state["current"])
    return {"patches": len(ids), "events": carried, "current": state["current"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("session")
    parser.add_argument("out")
    parser.add_argument("--handle", required=True)
    parser.add_argument("--email", required=True)
    parser.add_argument("--staged")
    args = parser.parse_args(argv)
    migrate(args.session, args.out, rec.Reviewer(args.handle, args.email).as_json(), args.staged)


if __name__ == "__main__":
    main()
