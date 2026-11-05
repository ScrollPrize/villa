"""The stages of a pair, in order (see DESIGN.md), and what each one does.

The stages that read the volumes work on one session of `cores/fit_session.py` in
`work/session/`: its patches, cached blocks and fits. Only one job holds it at a time
(`session_lock`); the hold-out set's measurements are never re-measured once drawn, so what
a person judged stays what is scored.
"""

import fcntl
import json
import shutil
from contextlib import contextmanager

import numpy as np

from . import migrate
from . import record as rec
from . import render
from .cores import fit_session as fs
from .cores import session_overview
from .engine import Pipeline, Stage
from .scoring import BAR, bar_check, score

PARAMS = "params.json"
# Draws of the hold-out stage, at most, to make up for patches that cannot be measured.
HOLDOUT_DRAWS = 4
SESSION = "session"


@contextmanager
def session_lock(pipeline):
    """The pair's session, held by one job at a time (others wait)."""
    path = pipeline.work("session.lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


# The cache directory tag (bford.ca/cachedir): backups run with --exclude-caches skip a
# directory holding it.
CACHE_TAG = ("Signature: 8a477f597d28d172789f06886806bc55\n"
             "# A cache of volume reads by the volume-registration pipeline; it can be read again"
             " from the volumes.\n")


def tag_caches(directory):
    """Mark a session's block cache and overview as caches, so backups leave them out."""
    for name in ("blocks", "overview"):
        tag = directory / name / "CACHEDIR.TAG"
        if tag.parent.is_dir() and not tag.exists():
            tag.write_text(CACHE_TAG)


def open_session(pipeline):
    """The pair's session, created from `pair.json` the first time."""
    directory = pipeline.work(SESSION)
    if (directory / "session.json").exists():
        session = fs.Session.load(directory)
    else:
        pair = pipeline.record.read(rec.PAIR)
        session = fs.Session.create(directory, pair["fixed"]["url"], pair["moving"]["url"],
                                    pair["fixed"]["voxel_um"], pair["moving"]["voxel_um"],
                                    pair["start"]["matrix_xyz"], pair["start"]["source"])
    tag_caches(directory)
    return session


def run_capture(pipeline, params):
    """Does the start transform land patches close enough to measure? A few probe patches
    are drawn and measured under it; they stay as fit patches afterwards. Fails, saying how
    well they matched, when too few match -- before anyone is asked to judge anything."""
    with session_lock(pipeline):
        session = open_session(pipeline)
        # The overview images are for the review page; a store without pyramid levels has none.
        if params["overview"] and not (session.directory / "overview").exists():
            session_overview.build(session)
        probes = [patch for patch in session.state["patches"] if patch["round"] == 0]
        if not probes:
            session.sample(params["probes"], seed=params["seed"])
            try:
                session.correlate(only_new=True)
            except RuntimeError as error:   # e.g. the sign calibration finds nothing to read
                raise RuntimeError(f"the start does not capture: {error}") from error
            probes = [patch for patch in session.state["patches"] if patch["round"] == 0]
        matches = [patch["measurement"]["match"] for patch in probes if patch["measurement"]]
        share = float(np.mean([match >= params["min_match"] for match in matches])) if matches else 0.0
        result = {"probes": len(probes), "measured": len(matches), "matches": matches,
                  "share_matching": share, "captured": share >= params["min_share"]}
        pipeline.work("capture.json").write_text(json.dumps(result, indent=1) + "\n")
        if not result["captured"]:
            raise RuntimeError(
                f"the start does not capture: {share:.0%} of {len(matches)} probes match "
                f">= {params['min_match']} (median match {np.median(matches) if matches else 0:.2f})")
    return ["work/capture.json"]


def run_holdout(pipeline, params):
    """Draw the hold-out set -- or, after `volreg more`, the patches it is short of --
    measure the new ones under the start, render what a person will judge, and list them in
    the record. Only measured patches are listed: an unmeasured one has nothing to judge."""
    with session_lock(pipeline):
        session = open_session(pipeline)
        for _ in range(HOLDOUT_DRAWS):
            measure_new_holdout(session)
            have = sum(1 for patch in session.state["patches"] if patch.get("test"))
            missing = params["patches"] - have
            if missing <= 0:
                break
            added = session.sample_test(missing, seed=params["seed"] + len(session.state["patches"]))
            if not added:
                break
            keep_only_asked(session, added, missing)
        if not any(patch.get("test") for patch in session.state["patches"]):
            raise RuntimeError(f"no hold-out patch could be drawn and measured "
                               f"({params['patches']} asked for)")
        held = migrate.holdout(migrate.Session(session.directory), None, every_octant=True)
        patches = [patch for patch in held["patches"] if "centre_zyx" in patch]
        render_holdout(pipeline, session, patches)
    pipeline.record.refresh_holdout(patches, patch_coarse_voxels=held["patch_coarse_voxels"],
                                    stretches=held["stretches"])
    # What the stages after this one rest on: which patches are held out, and where. Not
    # `holdout.json`, which also changes with every answer and every new rendering -- a fit
    # need not run again because a picture was redrawn.
    listed = pipeline.work("holdout-patches.json")
    listed.write_text(json.dumps([[patch["id"], patch["origin_zyx"]] for patch in patches]) + "\n")
    return ["work/holdout-patches.json"]


def measure_new_holdout(session):
    """Measure the hold-out patches drawn but not measured yet. One that cannot be measured
    -- where the fine scan does not reach, as for much of a 45 um scan against a DLS one --
    has nothing to judge: it leaves the hold-out set, and another is drawn in its place."""
    if any(patch.get("test") and patch["measurement"] is None for patch in session.state["patches"]):
        session.correlate(only_new=True)
    for patch in session.state["patches"]:
        if patch.get("test") and patch["measurement"] is None:
            patch["test"] = False
    session.save()


def render_holdout(pipeline, session, patches):
    """Pictures of every hold-out unit not rendered yet -- the measurements are frozen, so a
    picture once made stays right -- with their digests on the record's entries."""
    out = pipeline.work("images", "holdout")
    overview = (session.directory / "overview").exists()
    grey_anchors = display_anchors(pipeline, session)
    stored = pipeline.work("images", "anchors.json")
    if not stored.exists() or json.loads(stored.read_text()) != grey_anchors:
        # New grey levels: every picture is drawn again, so all of a pair's pictures share them.
        shutil.rmtree(out, ignore_errors=True)
        stored.parent.mkdir(parents=True, exist_ok=True)
        stored.write_text(json.dumps(grey_anchors, indent=1) + "\n")
    for patch in patches:
        source = session.patch(patch["id"])
        units = [(patch, patch["id"], (0, 0, 0), fs.dense.BLOCK)] + [
            (octant, f"{patch['id']}.{octant['index']}", tuple(octant["corner_zyx"]),
             fs.dense.BLOCK // 2) for octant in patch.get("octants") or []]
        for entry, unit_id, corner, size in units:
            digests = out / unit_id / "digests.json"
            if not digests.exists():
                found = render.holdout_unit(session, out, source, unit_id, entry["centre_zyx"],
                                            entry["fine_zyx"], grey_anchors, corner, size)
                digests.write_text(json.dumps(found))
            entry["image_sha256"] = json.loads(digests.read_text())
        if overview and not (out / patch["id"] / "section.png").exists():
            render.section(session, out, source)


def keep_only_asked(session, added, asked):
    """The sampler keeps every textured block of a read batch, so it can draw several times
    the patches asked for (30 for 10 on PHerc0814) -- each one a person would then judge.
    Per stretch, the first textured ones up to its share stay in the hold-out set; the rest,
    and blocks without texture, become fit patches, so nothing read is wasted."""
    by_stretch = {}
    for patch_id in added:
        patch = session.patch(patch_id)
        by_stretch.setdefault(session.stretch(patch), []).append(patch)
    for index in range(fs.TEST_STRETCHES):
        share = (asked * (index + 1)) // fs.TEST_STRETCHES - (asked * index) // fs.TEST_STRETCHES
        kept = 0
        for patch in by_stretch.get(index, []):
            if patch["auto"] == fs.OK and kept < share:
                kept += 1
            else:
                patch["test"] = False
    session.save()


def display_anchors(pipeline, session):
    """The pair's grey-level anchors (see render.py), measured once when the hold-out set is
    first rendered and kept after, so pictures drawn later -- more hold-out patches, a blind
    round -- share them."""
    stored = pipeline.work("images", "anchors.json")
    if stored.exists():
        return json.loads(stored.read_text())
    return render.pair_anchors(session)


def run_fit(pipeline, params):
    """Sample fit patches up to the number asked for, then correlate and fit, `passes` times,
    as the first runs did (patch hold-out, gate). Hold-out patches are never used or
    re-measured.

    Two settings a pair may carry in `params.json`, off unless set: `at_most`, a seeded
    random choice of that many fit patches when there are more (a session imported from
    the first runs can hold 500, and the bend's cost grows with the cube of its points);
    `octants` 0, one point per patch and never its octants; `unit` "octants", every octant
    that measured well as a point (the fit otherwise keeps its last fit's unit)."""
    with session_lock(pipeline):
        session = open_session(pipeline)
        fit_patches = sorted(patch["id"] for patch in session.state["patches"]
                             if not patch.get("test") and patch["auto"] == fs.OK)
        have = len(fit_patches)
        if params["patches"] > have:
            session.sample(params["patches"] - have, seed=params["seed"] + have)
            fit_patches = sorted(patch["id"] for patch in session.state["patches"]
                                 if not patch.get("test") and patch["auto"] == fs.OK)
        among = None
        if params.get("at_most") and len(fit_patches) > params["at_most"]:
            rng = np.random.default_rng(params["seed"])
            among = set(rng.choice(fit_patches, params["at_most"], replace=False).tolist())
        octants = bool(params.get("octants", 1))
        for _ in range(params["passes"]):
            session.correlate(skip_test=True, among=among)
            session.fit("patches", "gate", unit=params.get("unit") if octants else "patches",
                        among=among, octants=octants)
        record = session.fit_record()
        transform = json.loads((session.directory / record["transform"]).read_text())
    out = pipeline.work("fit")
    out.mkdir(parents=True, exist_ok=True)
    (out / "transform.json").write_text(json.dumps(transform, indent=1) + "\n")
    (out / "fit.json").write_text(json.dumps({
        "fit": record["id"],
        "points": {"ids": record["used"], "fixed_xyz": transform["fixed_landmarks"],
                   "moving_xyz": transform["moving_landmarks"],
                   "weighted_down": (record["bend"] or {}).get("weighted_down", {})},
        "model": record["model"], "hold_out": record["hold_out"], "robust": record["robust"],
        "unit": record["unit"],
        "bend": record["bend"] and {key: record["bend"][key]
                                    for key in ("regularisation", "effective_parameters")},
        "fine_um": session.fine_um, "coarse_um": session.coarse_um,
    }, indent=1) + "\n")
    return ["work/fit/transform.json", "work/fit/fit.json"]


def run_recheck(pipeline, params):
    """Measure every hold-out patch again under the fit, the way the first runs did after
    each pass, without changing the frozen measurement a person judged; record how far
    each would move. A patch measured under a rough start can move once the fit is better;
    one that moves more than `drift_fine_voxels` is flagged to be judged again."""
    with session_lock(pipeline):
        session = open_session(pipeline)
        _levels, fine_array, fine_scale = session.open()
        fine_to_coarse = session.fine_to_coarse()
        to_fine = np.linalg.inv(fine_to_coarse)
        sign = session.state["sampling"]["sign"]
        moved = {}
        for patch in session.state["patches"]:
            if not patch.get("test") or not patch["measurement"]:
                continue
            measured = fs.dense.measure_block(
                np.asarray(patch["origin_zyx"]), session.coarse_block(patch["id"]), fine_array,
                fine_scale, fine_to_coarse, sign, session._fine, patch["id"], with_check=True)
            if isinstance(measured, str):
                moved[patch["id"]] = {"refused": measured}
                continue
            centre, _fine, _shift, _error, _match, check, _octants = measured
            optimum = fs.dense.apply(to_fine, centre + sign * np.asarray(check["peak_zyx"]))
            frozen = np.asarray(patch["measurement"]["fine_zyx"], float)
            moved[patch["id"]] = {"fine_voxels": round(float(np.linalg.norm(optimum - frozen)), 4),
                                  "match": check["peak_match"]}
    distances = np.array([entry["fine_voxels"] for entry in moved.values() if "fine_voxels" in entry])
    flagged = sorted(patch_id for patch_id, entry in moved.items()
                     if entry.get("fine_voxels", 0) > params["drift_fine_voxels"])
    result = {"under": session.state["current"], "drift_fine_voxels": params["drift_fine_voxels"],
              "median": float(np.median(distances)) if len(distances) else None,
              "max": float(distances.max()) if len(distances) else None,
              "flagged": flagged, "patches": moved}
    pipeline.work("recheck.json").write_text(json.dumps(result, indent=1) + "\n")
    return ["work/recheck.json"]


def holdout_identity(pipeline):
    """What judging the hold-out set decided: the answer per unit, nothing else."""
    if not pipeline.record.exists(rec.HOLDOUT):
        return {}
    found = {}
    for patch in pipeline.record.read(rec.HOLDOUT)["patches"]:
        found[patch["id"]] = (patch.get("answer") or {}).get("answer")
        for octant in patch.get("octants") or []:
            found[f"{patch['id']}.{octant['index']}"] = (octant.get("answer") or {}).get("answer")
    return found


def final_identity(pipeline):
    """What the final review decided: the picks of each round and the decision."""
    if not pipeline.record.exists(rec.FINAL_REVIEW):
        return {}
    final = pipeline.record.read(rec.FINAL_REVIEW)
    decision = final.get("decision") or {}
    return {"picks": {round_["id"]: [item["choice"] for item in round_["items"]]
                      for round_ in final.get("rounds") or []},
            "decision": {key: decision.get(key) for key in ("fit", "publish", "override")}}


def holdout_judged(pipeline, params):
    """Enough hold-out patches line up, or every one is answered."""
    if not pipeline.record.exists(rec.HOLDOUT):
        return False
    patches = pipeline.record.read(rec.HOLDOUT)["patches"]
    answered = [patch for patch in patches if patch.get("answer")]
    lines_up = sum(patch["answer"]["answer"] == rec.LINES_UP for patch in answered) + sum(
        (octant.get("answer") or {}).get("answer") == rec.LINES_UP
        for patch in patches for octant in patch.get("octants") or [])
    return lines_up >= params["min_lines_up"] or (patches and len(answered) == len(patches))


def run_score(pipeline, params):
    """Score the fit's bend and its affine part against the answered hold-out set."""
    pair = pipeline.record.read(rec.PAIR)
    transform = json.loads(pipeline.work("fit", "transform.json").read_text())
    fit = json.loads(pipeline.work("fit", "fit.json").read_text())
    held = pipeline.record.read(rec.HOLDOUT)
    fixed_is_coarse = pair["coarse_side"] == "fixed"
    scores = {name: score(transform, held, fixed_is_coarse, fit["fine_um"], affine_only=name == "affine")
              for name in ("bend", "affine")}
    for value in scores.values():
        value["bar"] = bar_check(value)
    recheck = json.loads(pipeline.work("recheck.json").read_text())
    lining_up = {patch["id"] for patch in held["patches"]
                 if (patch.get("answer") or {}).get("answer") == rec.LINES_UP}
    drift = {"median": recheck["median"], "max": recheck["max"],
             "limit": recheck["drift_fine_voxels"],
             # Lining-up patches whose measurement would move under the fit: their answer
             # was given to a picture that may be off, so they are worth judging again.
             "judge_again": [patch_id for patch_id in recheck["flagged"] if patch_id in lining_up]}
    out = pipeline.work("scores.json")
    out.write_text(json.dumps({"bar": BAR, "scores": scores, "drift": drift}, indent=1) + "\n")
    return ["work/scores.json"]


def run_blind(pipeline, params):
    """A blind round for the fit: the hold-out patches answered "lines up" or "partly" where
    its bend and its affine part place the patch furthest apart, the two options shuffled.

    A round for this fit as long as asked is kept. One shorter than asked -- drawn from too
    few answers, or before the count was raised -- grows: if someone has picked in it, the
    next patches by spread are added after its items, which keep their places and picks;
    if nobody has, it is drawn again under its id. A new fit gets a new round."""
    fit = json.loads(pipeline.work("fit", "fit.json").read_text())
    candidates = [fit["fit"], fit["fit"] + fs.AFFINE]
    final = pipeline.record.read(rec.FINAL_REVIEW) if pipeline.record.exists(rec.FINAL_REVIEW) else {}
    rounds = final.get("rounds") or []
    events = pipeline.record.events()
    picked = {event["round"] for event in events if event["kind"] == "blind-pick"}
    this_fit = [round_ for round_ in rounds if round_["candidates"] == candidates]
    if any(len(round_["items"]) >= params["count"] for round_ in this_fit):
        return [rec.FINAL_REVIEW]
    grown = next((round_ for round_ in this_fit if round_["id"] in picked), None)
    if grown is None:
        rounds = [round_ for round_ in rounds if round_ not in this_fit]
        taken = {round_["id"] for round_ in final.get("rounds") or []} | {
            event["round"] for event in events if event["kind"] == "blind-round"}
        # Nobody has picked in it, so no answer is tied to its id: it keeps the id.
        round_id = this_fit[0]["id"] if this_fit else next(
            f"c{number:03d}" for number in range(len(taken) + 1, 1000) if f"c{number:03d}" not in taken)
        grown = {"id": round_id, "at": rec.now(), "candidates": candidates, "test_only": True,
                 "items": []}
        rounds.append(grown)
    held = {patch["id"]: patch for patch in pipeline.record.read(rec.HOLDOUT)["patches"]}
    shown = {item["patch"] for item in grown["items"]}
    eligible = {patch_id for patch_id, patch in held.items()
                if (patch.get("answer") or {}).get("answer") in params["answers"]}
    with session_lock(pipeline):
        session = open_session(pipeline)
        to_coarse = session.fine_um / session.coarse_um
        ranked = []
        for patch_id in sorted(eligible - shown):
            patch = session.patch(patch_id)
            placed = [session.placement(candidate, patch) for candidate in candidates]
            spread = float(np.linalg.norm(placed[0] - placed[1]) * to_coarse)
            if spread >= fs.SAME_PLACE_VOXELS:
                ranked.append((spread, patch_id))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        before = len(grown["items"])
        for spread, patch_id in ranked[:params["count"] - before]:
            index = len(grown["items"])
            # Each item's order from the seed, the round and its place, so an item added
            # later is shuffled as it would have been in the first draw.
            rng = np.random.default_rng([params["seed"], int(grown["id"][1:]), index])
            options = [[candidates[i]] for i in rng.permutation(len(candidates))]
            digests = render.blind_item(session, pipeline.work("images", "blind", grown["id"], str(index)),
                                        session.patch(patch_id), options,
                                        display_anchors(pipeline, session))
            grown["items"].append({"patch": patch_id, "spread": round(spread, 4), "options": options,
                                   "choice": None, "image_sha256": digests})
    pipeline.record.refresh_final(rounds)
    pipeline.record.log("blind-round", round=grown["id"], candidates=candidates,
                        items=len(grown["items"]), added=len(grown["items"]) - before,
                        eligible=len(eligible),
                        replaces=[round_["id"] for round_ in this_fit if round_ is not grown])
    return [rec.FINAL_REVIEW]


def final_decided(pipeline, params):
    return pipeline.record.exists(rec.FINAL_REVIEW) and \
        bool(pipeline.record.read(rec.FINAL_REVIEW).get("decision"))


def run_store(pipeline, params):
    """Put the decided fit into the record: the transform as it will be published (without
    its bend when the decision says affine), how it was fitted, and its scores."""
    decision = pipeline.record.read(rec.FINAL_REVIEW)["decision"]
    fit = json.loads(pipeline.work("fit", "fit.json").read_text())
    scores = json.loads(pipeline.work("scores.json").read_text())
    stamp = pipeline.stamp("fit") or {}
    if decision["publish"] == "nothing":
        pipeline.record.path(rec.TRANSFORM).unlink(missing_ok=True)
        pipeline.record.write(rec.FIT, fit | scores | {"code": stamp.get("code")})
        return [rec.FIT]
    transform = json.loads(pipeline.work("fit", "transform.json").read_text())
    if decision["publish"] == "affine":
        transform = {key: value for key, value in transform.items() if key != "deformation"}
        if transform.get("schema_version") in ("1.2.0", "1.3.0"):
            transform["schema_version"] = "1.1.0" if transform.get("fit") else "1.0.0"
    transform["fixed_volume"] = transform["fixed_volume"].rstrip("/").rsplit("/", 1)[-1]
    pipeline.record.write(rec.TRANSFORM, transform)
    pipeline.record.write(rec.FIT, fit | scores | {"code": stamp.get("code")})
    return [rec.TRANSFORM, rec.FIT]


def stages():
    return [
        Stage("capture", "capture check", run=run_capture,
              params={"probes": 12, "min_match": 0.85, "min_share": 0.5, "seed": 0,
                      "overview": True}),
        Stage("holdout", "draw and measure the hold-out set", needs=("capture",),
              run=run_holdout, params={"patches": 60, "seed": 1000}),
        Stage("judge-holdout", "judge the hold-out set", needs=("holdout",),
              complete=holdout_judged, identity=holdout_identity,
              params={"min_lines_up": BAR["min_lines_up"]}),
        # Starts right after the capture check, beside the hold-out set: neither draws on the
        # other's patches (the sampler skips taken origins, a fit never uses a hold-out
        # patch), so more hold-out patches do not fit again, and more fit patches do not
        # touch what a person judged.
        Stage("fit", "sample, measure and fit", needs=("capture",),
              run=run_fit, params={"patches": 200, "passes": 2, "seed": 2000}),
        Stage("recheck", "measure the hold-out set again under the fit", needs=("fit", "holdout"),
              run=run_recheck, params={"drift_fine_voxels": 0.5}),
        Stage("score", "score against the hold-out set", needs=("recheck", "judge-holdout"),
              run=run_score),
        Stage("blind", "draw the blind round of bend against affine", needs=("score",),
              run=run_blind, params={"count": 20, "seed": 3000,
                                     "answers": [rec.LINES_UP, "partly"]}),
        Stage("final-review", "judge the final fit", needs=("blind",),
              complete=final_decided, identity=final_identity),
        Stage("store", "store the decided fit", needs=("final-review",), run=run_store),
    ]


def pipeline(directory):
    """A pair's pipeline, with the parameters its `params.json` sets (`volreg more` and
    friends write there; it is part of the record, since it says what was asked for)."""
    record = rec.Record(directory)
    params = record.read(PARAMS) if record.exists(PARAMS) else {}
    params.pop("schema", None)
    return Pipeline(directory, stages(), params)
