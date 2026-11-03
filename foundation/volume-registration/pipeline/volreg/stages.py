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
from .cores import fit_session as fs
from .cores import session_overview
from .engine import Pipeline, Stage
from .scoring import BAR, bar_check, score

PARAMS = "params.json"
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


def open_session(pipeline):
    """The pair's session, created from `pair.json` the first time."""
    directory = pipeline.work(SESSION)
    if (directory / "session.json").exists():
        return fs.Session.load(directory)
    pair = pipeline.record.read(rec.PAIR)
    return fs.Session.create(directory, pair["fixed"]["url"], pair["moving"]["url"],
                             pair["fixed"]["voxel_um"], pair["moving"]["voxel_um"],
                             pair["start"]["matrix_xyz"], pair["start"]["source"])


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
    """Draw the hold-out set -- or, after `volreg more`, the patches it is short of -- and
    measure the new ones under the start, then list them in the record for judging."""
    with session_lock(pipeline):
        session = open_session(pipeline)
        have = sum(1 for patch in session.state["patches"] if patch.get("test"))
        missing = params["patches"] - have
        if missing > 0:
            added = session.sample_test(missing, seed=params["seed"] + have)
            if not added:
                raise RuntimeError(f"no hold-out patch could be drawn ({missing} asked for)")
            keep_only_asked(session, added, missing)
            session.correlate(only_new=True)
        held = migrate.holdout(migrate.Session(session.directory), None, every_octant=True)
    pipeline.record.refresh_holdout(held["patches"])
    held_now = pipeline.record.read(rec.HOLDOUT)
    pipeline.record.write(rec.HOLDOUT, {key: value for key, value in held_now.items()
                                        if key != "schema"}
                          | {"patch_coarse_voxels": held["patch_coarse_voxels"],
                             "stretches": held["stretches"]})
    return [rec.HOLDOUT]


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


def run_fit(pipeline, params):
    """Sample fit patches up to the number asked for, then correlate and fit, `passes` times,
    as the first runs did (patch hold-out, gate). Hold-out patches are never used or
    re-measured."""
    with session_lock(pipeline):
        session = open_session(pipeline)
        have = sum(1 for patch in session.state["patches"]
                   if not patch.get("test") and patch["auto"] == fs.OK)
        if params["patches"] > have:
            session.sample(params["patches"] - have, seed=params["seed"] + have)
        for _ in range(params["passes"]):
            session.correlate(skip_test=True)
            session.fit("patches", "gate")
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
    out = pipeline.work("scores.json")
    out.write_text(json.dumps({"bar": BAR, "scores": scores}, indent=1) + "\n")
    return ["work/scores.json"]


def final_decided(pipeline, params):
    return pipeline.record.exists(rec.FINAL_REVIEW) and \
        bool(pipeline.record.read(rec.FINAL_REVIEW).get("decision"))


def run_store(pipeline, params):
    """Put the decided fit into the record: the transform as it will be published (without
    its bend when the decision says affine), how it was fitted, and its scores."""
    decision = pipeline.record.read(rec.FINAL_REVIEW)["decision"]
    transform = json.loads(pipeline.work("fit", "transform.json").read_text())
    if decision["publish"] == "affine":
        transform = {key: value for key, value in transform.items() if key != "deformation"}
        if transform.get("schema_version") in ("1.2.0", "1.3.0"):
            transform["schema_version"] = "1.1.0" if transform.get("fit") else "1.0.0"
    transform["fixed_volume"] = transform["fixed_volume"].rstrip("/").rsplit("/", 1)[-1]
    fit = json.loads(pipeline.work("fit", "fit.json").read_text())
    scores = json.loads(pipeline.work("scores.json").read_text())
    stamp = pipeline.stamp("fit") or {}
    pipeline.record.write(rec.TRANSFORM, transform)
    pipeline.record.write(rec.FIT, fit | scores | {"code": stamp.get("code")})
    if pipeline.work("fit", "points.json").exists():
        shutil.copy(pipeline.work("fit", "points.json"), pipeline.record.path("fit/points.json"))
    return [rec.TRANSFORM, rec.FIT]


def stages():
    return [
        Stage("capture", "capture check", run=run_capture,
              params={"probes": 12, "min_match": 0.85, "min_share": 0.5, "seed": 0,
                      "overview": True}),
        Stage("holdout", "draw and measure the hold-out set", needs=("capture",),
              run=run_holdout, params={"patches": 60, "seed": 1000}),
        Stage("judge-holdout", "judge the hold-out set", needs=("holdout",),
              complete=holdout_judged, answers=(rec.HOLDOUT,),
              params={"min_lines_up": BAR["min_lines_up"]}),
        # Starts right after the hold-out set is drawn, while a person judges it.
        Stage("fit", "sample, measure and fit", needs=("holdout",),
              run=run_fit, params={"patches": 200, "passes": 2, "seed": 2000}),
        Stage("score", "score against the hold-out set", needs=("fit", "judge-holdout"),
              run=run_score),
        Stage("final-review", "judge the final fit", needs=("score",),
              complete=final_decided, answers=(rec.FINAL_REVIEW,)),
        Stage("store", "store the decided fit", needs=("final-review",), run=run_store),
    ]


def pipeline(directory):
    """A pair's pipeline, with the parameters its `params.json` sets (`volreg more` and
    friends write there; it is part of the record, since it says what was asked for)."""
    record = rec.Record(directory)
    params = record.read(PARAMS) if record.exists(PARAMS) else {}
    params.pop("schema", None)
    return Pipeline(directory, stages(), params)
