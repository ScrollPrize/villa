"""Score a transform against a hold-out set, and check the publishing bar.

Needs nothing but the record: each hold-out patch answered "lines up" carries its coarse
centre and the fine point its own measurement put it at, and a transform maps one to the
other. How far it lands from that point, in fine level-0 voxels, is the patch's score.

A score is only as good as the answers it was made of, and a decision only covers the scores
it was made on. `answers_digest` names the answers a score used (the units answered "lines
up", where they sit), and `basis` names what a reviewer decides on: the scores of a fit, with
the bar's verdict on each variant, and those answers. The review page, the decision, Store
and Release all carry the basis, so none of them can pair an approval with other scores, or
scores with other answers.
"""

import hashlib
import json

import numpy as np

from .cores.deformable import correction_from_json
from .record import LINES_UP

# The bar agreed on 2026-10-22, on at least `min_lines_up` hold-out patches answered
# "lines up", in fine voxels. A bend is kept only where it also wins its blind round, and the
# reviewer may keep a barely visible bend out; that part is the final review's, not this.
BAR = {"version": "2026-10-22", "min_lines_up": 20, "median_max": 2.0, "p90_max": 4.0,
       "far_voxels": 5.0, "far_share_max": 1 / 20}


def coarse_to_fine(transform, fixed_is_coarse, affine_only=False):
    """A function taking coarse ZYX points to fine ZYX points through a transform file.

    The file maps moving to fixed, in XYZ, with an optional thin-plate-spline bend.
    """
    matrix = np.vstack([np.asarray(transform["transformation_matrix"], float), [0, 0, 0, 1]])
    deformation = None if affine_only else transform.get("deformation")
    if deformation is None:
        affine = np.linalg.inv(matrix) if fixed_is_coarse else matrix
        return lambda zyx: (np.asarray(zyx, float)[:, ::-1] @ affine[:3, :3].T
                            + affine[:3, 3])[:, ::-1]
    correction = correction_from_json(matrix[:3], deformation)
    if fixed_is_coarse:
        correction = correction.inverted()
    return lambda zyx: correction.apply(np.asarray(zyx, float)[:, ::-1])[:, ::-1]


def lining_up(holdout):
    """`(id, coarse centre, fine point)` of every hold-out unit answered "lines up": a patch,
    or `<patch>.<octant>` for an octant judged on its own."""
    units = []
    for patch in holdout["patches"]:
        answer = (patch.get("answer") or {}).get("answer")
        if answer == LINES_UP:
            units.append((patch["id"], patch["centre_zyx"], patch["fine_zyx"]))
        for octant in patch.get("octants") or []:
            if (octant.get("answer") or {}).get("answer") == LINES_UP:
                units.append((f"{patch['id']}.{octant['index']}", octant["centre_zyx"],
                              octant["fine_zyx"]))
    return units


def answers_used(holdout):
    """What a score rests on: `[id, coarse centre, fine point]` of every unit answered "lines
    up", by id. Like `lining_up`, but for any record (a unit's points are None where it has
    none), and in an order of its own."""
    found = []
    for patch in holdout.get("patches") or []:
        units = [(patch["id"], patch)] + [(f"{patch['id']}.{octant['index']}", octant)
                                          for octant in patch.get("octants") or []]
        for unit_id, unit in units:
            if (unit.get("answer") or {}).get("answer") == LINES_UP:
                found.append([unit_id, unit.get("centre_zyx"), unit.get("fine_zyx")])
    return sorted(found, key=lambda unit: unit[0])


def answers_digest(holdout):
    """The digest of `answers_used`: which answers, and so which units, a score was made of.
    A note, a picture, or another answer than "lines up" changed into another one that is not
    either, leaves it as it was, since the score does not change."""
    return hashlib.sha256(json.dumps(answers_used(holdout), sort_keys=True).encode()).hexdigest()


def basis(fit_digest, scores, bar, answers):
    """What a decision is made on, as one digest: the fit (by its stamp's digest), its scores
    for every variant with the bar's verdict on each (`scores`, as `scores.json` holds them),
    the bar, and the answers they used (`answers_digest`)."""
    return hashlib.sha256(json.dumps({"fit_digest": fit_digest, "scores": scores, "bar": bar,
                                      "answers": answers}, sort_keys=True).encode()).hexdigest()


def bar_failures(result):
    """The parts of the bar a stored score misses, as its own check recorded them, sorted. A
    score said to miss the bar without saying where misses "bar"."""
    bar = (result or {}).get("bar") or {}
    missed = sorted(part for part, ok in (bar.get("checks") or {}).items() if not ok)
    return missed or (["bar"] if bar.get("passes") is False else [])


def score(transform, holdout, fixed_is_coarse, fine_um, affine_only=False, units=None):
    """Distances in fine voxels at every unit lining up, and their summary."""
    units = lining_up(holdout) if units is None else units
    if not units:
        return {"lines_up": 0}
    mapped = coarse_to_fine(transform, fixed_is_coarse, affine_only)(
        np.array([unit[1] for unit in units], float))
    distances = np.linalg.norm(mapped - np.array([unit[2] for unit in units], float), axis=1)
    order = np.argsort(-distances)
    return {
        "lines_up": len(units),
        "median_fine_voxels": float(np.median(distances)),
        "p90_fine_voxels": float(np.percentile(distances, 90)),
        "worst_fine_voxels": float(distances.max()),
        "far": int((distances > BAR["far_voxels"]).sum()),
        "median_um": float(np.median(distances) * fine_um),
        "p90_um": float(np.percentile(distances, 90) * fine_um),
        "worst": [[units[index][0], round(float(distances[index]), 3)] for index in order[:5]],
        "per_unit": {unit[0]: round(float(value), 4) for unit, value in zip(units, distances)},
    }


def bar_check(result, bar=BAR):
    """Which parts of the bar a score clears; `passes` only if all do."""
    count = result.get("lines_up", 0)
    checks = {"enough": count >= bar["min_lines_up"]}
    if count:
        checks |= {"median": result["median_fine_voxels"] <= bar["median_max"],
                   "p90": result["p90_fine_voxels"] <= bar["p90_max"],
                   "far": result["far"] <= bar["far_share_max"] * count}
    return {"bar": bar["version"], "checks": checks, "passes": all(checks.values())}
