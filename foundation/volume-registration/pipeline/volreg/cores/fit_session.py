"""A registration session: the state the review tool works on, and what can be done to it.

`research/dense_blocks.py` measures a pair in one pass and writes a transform. A person
reviewing that transform needs more than its result: which patches it rests on, which
were rejected and why, and what happens to the fit when one of them is taken out. So a
session keeps all of that in one directory, and every operation reads and updates it.

    <session>/session.json     the state: volumes, start transform, patches, fits
    <session>/blocks/<id>.npz  each patch's coarse block, and the fine patch behind it
    <session>/fits/<id>.json   each fit, as a transform file in the project's schema

A correspondence is a coarse point and the fine point it matches. It does not depend on the
transform the fine volume was resampled under, so setting a person's verdict on a patch
needs a refit and nothing else; only `correlate` touches the volumes again, and only the
blocks already cached. See `review-tool.md` for the tool this is the engine of.

Coordinates are ZYX level-0 voxels of their own volume throughout, as in
`dense_correlation.py`. Matrices are stored the way the schema stores them -- 3x4, XYZ,
moving -> fixed -- so a session file reads like a transform file.

    uv run --python 3.12 --with-requirements requirements.txt --with scikit-image \\
        python fit_session.py init SESSION --pair PAIR --manifest MANIFEST --transform FILE
    ... python fit_session.py sample SESSION --blocks 80
    ... python fit_session.py overview SESSION
    ... python fit_session.py refine SESSION --passes 3
    ... python fit_session.py verdict SESSION p0007 debris
    ... python fit_session.py fit SESSION
    ... python fit_session.py export SESSION OUT.json
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import numpy as np

from . import dense_correlation as dense
from . import session_overview
from .deformable import (
    correction_from_json,
    fit_chosen_correction,
    fit_correction,
    smoother,
)
from .transform_fit import select_transform_from_points, selection_to_json
from .transform_io import (
    matrix_swap_xyz_zyx,
    points_swap_xyz_zyx,
    read_transform_json,
    write_transform_json,
)
from .volumes import open_volume_levels

SCHEMA = "volreg-session/0.1"

# What a person can say about a patch. The reasons are kept apart rather than collapsed
# into good/bad because they are the labels the automatic rules will be measured against:
# which kinds of bad patch a weak correlation peak catches, and which it does not.
VERDICTS = (
    "good",
    # Pinned across the sheet, noise along it. Used at full weight until the fit can
    # weight a patch per direction; recorded now so those patches can be found again.
    "good-across-sheet",
    "debris",
    "broken-off",
    "case-or-wrapping",
    "no-papyrus",
    "noise-or-flat",
    "other",
    # The patch did not move as a whole -- the scroll deformed inside it -- but its
    # octants agree among themselves: each octant that measured well is used as a point
    # of its own instead of the patch.
    "use-octants",
)
USABLE = {"good", "good-across-sheet"}
OCTANTS = "use-octants"
# Verdicts about what a patch holds, not about how it measured: a blind comparison of
# fits leaves these patches out, since none of the fits should be judged on them.
NOT_SCROLL = {"debris", "broken-off", "case-or-wrapping", "no-papyrus", "noise-or-flat"}
# The comparison candidates that are not fits: each patch drawn at its own measurement,
# from the phase correlation, or at the NCC optimum its peak check found next to it.
OWN = "own"
PEAK = "peak"
# A fit's id with this appended stands for its affine part alone, without the bend: the
# candidate a blind round needs to judge whether a bend helps.
AFFINE = "/affine"
# How a fit chooses how stiff its bend is: by predicting each of eight regions from the
# rest (the research's rule, which punishes a deformation only one region shows), or
# each patch from the rest (which lets several neighbouring patches carry a local
# deformation, while one patch alone still cannot). A patch's octants are held out
# together, so they cannot vouch for one another.
HOLD_OUTS = ("regions", "patches")
# What a point of the fit is: a patch's measurement, or each of its octants that measured
# well. Octants are for a pair whose patches mostly do not move as one piece -- on
# PHercParis4 a person answered "partly" for 83% of the test patches.
UNITS = ("patches", "octants")
# With patches held out, no patch -- its octants together -- may pull the fit more than
# this share of the way to itself; a patch that would is weighted down until it does not.
# Holding out one patch at a time on its own let a lone patch bend the fit most of the way
# to it (about 60% on a synthetic spike).
MAX_PATCH_LEVERAGE = 0.5

# Two candidates that place a patch closer than this, in coarse voxels, are shown as one
# option: no overlay tells them apart, and asking to would only collect noise.
SAME_PLACE_VOXELS = 0.3
# What a person can answer instead of an option: that they cannot tell the options
# apart, or that none of them lines the patch up -- which says the patch needs more
# than one shift (its octants, or patches around it), not that the person hesitated.
ANSWERS = ("unsure", "none")
# An octant of a "use-octants" patch is used if its NCC optimum matches at least this.
OCTANT_MIN_MATCH = 0.85
# When no person has judged a patch, the rule that stands in for "use-octants": enough
# octants measured well, agreeing about a gradient larger than a patch that moved as a
# whole shows (90% of those reviewed good on PHerc0139 stayed under 0.32 voxels).
AUTO_OCTANTS = {"octants": 6, "gradient": 0.4, "scatter": 0.2}

# How a fit decides which patches no person judged it may use: RANSAC against one
# affine (the research's rule), or a gate on how well the patch matches at its NCC
# optimum. On PHerc0139 RANSAC dropped 14 good patches to catch 3 bad ones, mostly at
# the ends, which bend away from one affine; a gate at 0.87 kept 117 of the 120 patches
# a person included.
ROBUST = ("ransac", "gate")
GATE_MIN_MATCH = 0.87

# A test set: patches drawn apart from the ones fits are made of and never used by any,
# each judged by a person on its own measurement alone -- is the content scroll, and does
# the patch line up at its own shift -- so a fit can be scored by how far it lands from
# them. It is drawn in equal shares along the scroll, so no stretch goes untested (the
# ends, where fits part most, least of all); a test patch the fit places further than
# `TEST_MISS_VOXELS` from its own measurement counts as missed.
TEST_STRETCHES = 5
TEST_MISS_VOXELS = 0.5
# What a person answers about a test patch, shown it at its own measurement and nothing
# about any fit: it lines up at that one shift; part of it does, so no one shift is an
# answer for it; it does not, so the measurement itself is wrong; there is too little to
# tell; or it is not scroll. Only the first scores. The share answered "partly" says how
# often a patch is too large to move as one piece, and "no" how often a measurement fails.
TEST_ANSWERS = ("lines-up", "partly", "no", "cant-tell", "not-scroll")
LINES_UP = "lines-up"
# A test set is reviewed until this many patches line up. A fit then passes if it misses
# at most `TEST_ALLOWED_MISSES` of them -- none of 30 bounds the share of places it misses
# below 10% at 95% confidence -- and fails as soon as it misses more, however few are in.
TEST_TARGET = 30
TEST_ALLOWED_MISSES = 1

# The automatic verdict of a patch whose block reads, holds texture and correlates.
OK = "ok"

START = "start"


# What a person writes on a patch, through the review or the test page. A background job
# holding an older copy of the session must not save over these -- see `Session.save`.
PERSON_KEYS = ("human", "note", "test_answer", "test_octants")


def person_fields(state):
    """`{patch id: {key: value}}` for `PERSON_KEYS`, plus the blind rounds."""
    fields = {patch["id"]: {key: patch.get(key) for key in PERSON_KEYS}
              for patch in state["patches"]}
    return fields | {None: {"comparisons": state.get("comparisons")}}


class TestUnit(NamedTuple):
    """What a test set is judged and scored on: a test patch, or one of its octants."""

    id: str           # the patch's id, or `<patch>.<octant>`
    patch: dict
    corner_zyx: tuple  # where in the patch the unit starts, coarse voxels
    size: int          # its edge, coarse voxels
    centre_zyx: list   # coarse point
    fine_zyx: list     # where its own measurement puts it
    answer: object     # a person's answer, one of TEST_ANSWERS, or None
    at: object         # when it was given


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def zyx4(matrix_xyz):
    """A stored 3x4 XYZ matrix as the 4x4 ZYX one the fits and the resampling work in."""
    return dense.as_zyx(matrix_xyz)


def spread(values):
    """`(id, distance)` pairs summed up: how many, median, p90, how many are missed, worst."""
    if not values:
        return {"patches": 0}
    distances = np.array([value for _id, value in values])
    worst = sorted(values, key=lambda item: -item[1])[:5]
    return {"patches": len(values), "median": float(np.median(distances)),
            "p90": float(np.percentile(distances, 90)),
            "missed": int((distances > TEST_MISS_VOXELS).sum()),
            "worst": [[patch_id, round(value, 3)] for patch_id, value in worst]}


def wants_octants(measurement):
    """Whether `AUTO_OCTANTS` would have a patch lend the fit its octants."""
    summary = octant_summary(measurement)
    return bool(summary) and summary["octants"] >= AUTO_OCTANTS["octants"] \
        and summary["gradient"] >= AUTO_OCTANTS["gradient"] \
        and summary["scatter"] <= AUTO_OCTANTS["scatter"]


def octant_summary(measurement):
    """How a patch's octants that measured well vary inside it, or None with under five.

    `octants` is how many measured well (see `OCTANT_MIN_MATCH`); `gradient` the change
    in shift across half a patch of the plane fitted through their NCC optima, in coarse
    voxels; `scatter` the RMS of the octants about that plane. A patch that moved as a
    whole has both small; one that deformed inside it has a gradient and little scatter;
    one with nothing to trust scatters.
    """
    octants = [octant for octant in (measurement or {}).get("octants") or []
               if "rejected" not in octant and octant["check"]["peak_match"] >= OCTANT_MIN_MATCH]
    if len(octants) < 5:
        return None
    centres = np.array([octant["centre_zyx"] for octant in octants], float)
    shifts = np.array([octant["check"]["peak_zyx"] for octant in octants], float)
    design = np.hstack([np.ones((len(octants), 1)), centres - centres.mean(0)])
    coefficients = np.linalg.lstsq(design, shifts, rcond=None)[0]
    scatter = np.sqrt(((shifts - design @ coefficients) ** 2).sum(1).mean())
    return {"octants": len(octants),
            "gradient": float(np.linalg.norm(coefficients[1:] * dense.BLOCK / 2, ord=2)),
            "scatter": float(scatter)}


class Session:
    """One registration attempt, held in a directory."""

    def __init__(self, directory, state):
        self.directory = Path(directory)
        self.state = state
        self._volumes = None
        self._coarse = {}
        self._fine = {}
        self._maps = {}
        self._as_read = (self._stamp(), json.loads(json.dumps(person_fields(state))))

    # ------------------------------------------------------------------ the directory

    @classmethod
    def create(cls, directory, fixed_url, moving_url, fixed_um, moving_um, start_matrix,
               start_source, landmarks=None):
        """A new session starting from `start_matrix` (3x4, XYZ, moving -> fixed).

        `landmarks` are the start transform's own (fixed, moving) points in XYZ, kept only
        so a run can be sampled inside the box they span for comparison with earlier work.
        """
        directory = Path(directory)
        if (directory / "session.json").exists():
            raise FileExistsError(f"{directory} already holds a session")
        fixed_is_coarse = float(fixed_um) >= float(moving_um)
        state = {
            "schema": SCHEMA,
            "created": now(),
            "fixed": {"url": fixed_url, "voxel_um": float(fixed_um)},
            "moving": {"url": moving_url, "voxel_um": float(moving_um)},
            "coarse_side": "fixed" if fixed_is_coarse else "moving",
            "start": {"source": start_source,
                      "matrix_xyz": np.asarray(start_matrix, float)[:3].tolist()},
            "sampling": {"block": dense.BLOCK, "sign": None, "rounds": []},
            "patches": [],
            "fits": [],
            "current": START,
        }
        if landmarks is not None:
            fixed, moving = (np.asarray(points, float) for points in landmarks)
            coarse = fixed if fixed_is_coarse else moving
            if len(coarse):
                coarse_zyx = np.asarray(points_swap_xyz_zyx(coarse), float)
                state["start"]["landmark_box_zyx"] = [coarse_zyx.min(0).tolist(),
                                                      coarse_zyx.max(0).tolist()]
        session = cls(directory, state)
        (directory / "blocks").mkdir(parents=True, exist_ok=True)
        (directory / "fits").mkdir(exist_ok=True)
        session.open()
        session.save()
        return session

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        with open(directory / "session.json") as handle:
            state = json.load(handle)
        if state.get("schema") != SCHEMA:
            raise ValueError(f"unknown session schema {state.get('schema')!r}")
        return cls(directory, state)

    def _stamp(self):
        target = self.directory / "session.json"
        return target.stat().st_mtime_ns if target.exists() else None

    def save(self):
        """Write the state. Replaced in one step, so an interrupted write loses nothing.

        A long job -- `sample`, `correlate`, a refit -- holds the session for minutes while
        a person keeps answering on the review or test page. If the file changed since
        this copy read it, what a person wrote there (`PERSON_KEYS`, the blind rounds) is
        taken over wherever this copy left it as it found it, so the job saves its own
        work without undoing theirs.
        """
        target = self.directory / "session.json"
        stamp, as_read = self._as_read
        if target.exists() and self._stamp() != stamp:
            with open(target) as handle:
                on_disk = person_fields(json.load(handle))
            ours = person_fields(self.state)
            for patch_id, fields in on_disk.items():
                if patch_id not in ours:
                    continue
                into = self.state if patch_id is None else self.patch(patch_id)
                for key, value in fields.items():
                    if value is None and key not in into:
                        continue
                    if ours[patch_id][key] == as_read.get(patch_id, {}).get(key):
                        into[key] = value
        scratch = target.with_suffix(".json.tmp")
        with open(scratch, "w") as handle:
            json.dump(self.state, handle, indent=1)
        os.replace(scratch, target)
        self._as_read = (self._stamp(), json.loads(json.dumps(person_fields(self.state))))

    # -------------------------------------------------------------------- the volumes

    @property
    def fixed_is_coarse(self):
        return self.state["coarse_side"] == "fixed"

    @property
    def coarse_um(self):
        return self.state[self.state["coarse_side"]]["voxel_um"]

    @property
    def fine_um(self):
        return self.state["moving" if self.fixed_is_coarse else "fixed"]["voxel_um"]

    @property
    def fixed_um(self):
        return self.state["fixed"]["voxel_um"]

    def open(self):
        """Open both stores and pick the fine level matched to the coarse voxel.

        The level is recorded the first time and reused after, so a session cannot change
        which pyramid level it correlates against between two of its own fits.
        """
        if self._volumes is not None:
            return self._volumes
        coarse = self.state[self.state["coarse_side"]]["url"]
        fine = self.state["moving" if self.fixed_is_coarse else "fixed"]["url"]
        coarse_levels = open_volume_levels(coarse)
        fine_levels = open_volume_levels(fine)
        recorded = self.state["sampling"].get("fine_scale")
        if recorded is None:
            _array, recorded = dense.pick_fine_level(fine_levels,
                                                     self.coarse_um / self.fine_um)
            self.state["sampling"]["fine_scale"] = recorded
            self.state["sampling"]["coarse_shape"] = list(coarse_levels[0][0].shape)
            self.state["sampling"]["fine_shape"] = list(fine_levels[0][0].shape)
        fine_array = next(array for array, scale in fine_levels if scale == recorded)
        self._volumes = (coarse_levels, fine_array, recorded)
        return self._volumes

    # ----------------------------------------------------------------- the transforms

    def fit_record(self, fit_id=None):
        fit_id = fit_id or self.state["current"]
        if fit_id == START:
            return None
        return next(fit for fit in self.state["fits"] if fit["id"] == fit_id)

    def matrix(self, fit_id=None):
        """The affine part of a fit (or of the start), 4x4 ZYX, moving -> fixed."""
        record = self.fit_record(fit_id)
        return zyx4(record["matrix_xyz"] if record else self.state["start"]["matrix_xyz"])

    def fine_to_coarse(self, fit_id=None):
        matrix = self.matrix(fit_id)
        return matrix if self.fixed_is_coarse else np.linalg.inv(matrix)

    def coarse_to_fine(self, fit_id=None):
        """Coarse level-0 ZYX points to fine level-0 ZYX points, through the whole fit.

        Returns `(map, bent)`. With a bend this is the spline -- forward or its fitted
        inverse, whichever runs from the coarse side -- so a patch is drawn the way the
        transform that would ship places it, not the way its affine part alone does.
        """
        fit_id = fit_id or self.state["current"]
        if fit_id not in self._maps:
            record = self.fit_record(fit_id)
            affine = np.linalg.inv(self.fine_to_coarse(fit_id))
            if record is None or record["bend"] is None:
                self._maps[fit_id] = ((lambda points: dense.apply(affine, points)), False)
            else:
                with open(self.directory / record["transform"]) as handle:
                    data = json.load(handle)
                correction = correction_from_json(
                    np.asarray(data["transformation_matrix"], float), data["deformation"])
                if self.fixed_is_coarse:
                    correction = correction.inverted()
                self._maps[fit_id] = (
                    (lambda points: correction.apply(
                        np.asarray(points, float)[:, ::-1])[:, ::-1]), True)
        return self._maps[fit_id]

    # -------------------------------------------------------------------- the patches

    def patch(self, patch_id):
        for patch in self.state["patches"]:
            if patch["id"] == patch_id:
                return patch
        raise KeyError(f"no patch {patch_id}")

    def coarse_block(self, patch_id):
        if patch_id not in self._coarse:
            self._load_block(patch_id)
        return self._coarse[patch_id]

    def _block_path(self, patch_id):
        return self.directory / "blocks" / f"{patch_id}.npz"

    def _load_block(self, patch_id):
        with np.load(self._block_path(patch_id)) as data:
            self._coarse[patch_id] = data["coarse"]
            if "fine" in data:
                self._fine[patch_id] = (data["fine_lo"], data["fine_hi"], data["fine"])

    def _save_block(self, patch_id):
        arrays = {"coarse": self._coarse[patch_id]}
        if patch_id in self._fine:
            lo, hi, patch = self._fine[patch_id]
            arrays.update(fine_lo=lo, fine_hi=hi, fine=patch)
        np.savez(self._block_path(patch_id), **arrays)

    def region(self, kind="overlap"):
        """The box blocks are drawn from, in coarse level-0 ZYX voxels.

        `overlap` is the fine volume's whole extent carried into the coarse one -- every
        place a block could hold both volumes. `landmarks` is the box the start
        transform's own points span, which is where every block in the research was drawn
        and is kept only so a session can be compared with that work.
        """
        self.open()
        if kind == "landmarks":
            box = self.state["start"].get("landmark_box_zyx")
            if box is None:
                raise ValueError("the start transform carried no landmarks")
            return np.asarray(box[0], float), np.asarray(box[1], float)
        if kind != "overlap":
            raise ValueError(f"unknown region {kind!r}")
        shape = np.asarray(self.state["sampling"]["fine_shape"], float)
        corners = np.array([[a, b, c] for a in (0, shape[0] - 1)
                            for b in (0, shape[1] - 1) for c in (0, shape[2] - 1)])
        mapped = dense.apply(self.fine_to_coarse(START), corners)
        coarse = np.asarray(self.state["sampling"]["coarse_shape"], float)
        return np.maximum(mapped.min(0), 0), np.minimum(mapped.max(0), coarse - 1)

    def sample(self, wanted, region="overlap", seed=None, along=None, test=False):
        """Draw blocks until `wanted` more of them hold texture.

        With `test`, the blocks join the test set (see `TEST_STRETCHES`) instead.

        `along`, a pair of fractions, keeps the draw to that stretch of the region along
        z, the scroll's axis -- `(0.8, 1)` is its last fifth -- so a stretch a fit cannot
        follow for want of patches can be given more.

        Rejected blocks are kept as patches too, with the reason, because the first
        sessions are meant to show a person everything the automatic rules decided. A
        block already in the session is never drawn again, since a duplicate would count
        one patch twice in every fit.
        """
        coarse_levels, _fine_array, _scale = self.open()
        coarse_array = coarse_levels[0][0]
        low, high = self.region(region)
        if along is not None:
            start, end = (float(value) for value in along)
            if not 0 <= start < end <= 1:
                raise ValueError(f"along must be two fractions 0 <= start < end <= 1, not {along}")
            length = high[0] - low[0]
            low, high = low.copy(), high.copy()
            low[0], high[0] = low[0] + start * length, low[0] + end * length
        rounds = self.state["sampling"]["rounds"]
        seed = len(rounds) if seed is None else int(seed)
        rng = np.random.default_rng(seed)

        where = dense.tissue_map(coarse_levels, low, high)
        taken = {tuple(patch["origin_zyx"]) for patch in self.state["patches"]}
        origins = [origin for origin in dense.chunk_grid_origins(
                       coarse_array.shape, (low, high), rng,
                       wanted * dense.MAX_ATTEMPTS_PER_BLOCK + len(taken))
                   if tuple(int(value) for value in origin) not in taken
                   and dense.worth_reading(where, origin)]

        def fetch(origin):
            block = dense.try_read(coarse_array, origin, origin + dense.BLOCK)
            return None if block is None else (origin, block)

        added, usable, tried = [], 0, 0
        with ThreadPoolExecutor(max_workers=dense.THREADS) as pool:
            while tried < len(origins) and usable < wanted:
                # Each round is sized by what is still missing, so a first round that
                # falls one short does not read a whole second round.
                step = max(dense.THREADS, int((wanted - usable) * dense.SAMPLE_ROUND))
                batch = origins[tried:tried + step]
                tried += len(batch)
                for result in pool.map(fetch, batch):
                    if result is None:
                        continue
                    origin, block = result
                    patch_id = f"p{len(self.state['patches']) + 1:04d}"
                    verdict = dense.block_rejection(block) or OK
                    self.state["patches"].append({
                        "id": patch_id,
                        "origin_zyx": [int(value) for value in origin],
                        "round": len(rounds),
                        "auto": verdict,
                        "human": None,
                        "measurement": None,
                        "test": bool(test),
                    })
                    self._coarse[patch_id] = block
                    self._save_block(patch_id)
                    added.append(patch_id)
                    usable += verdict == OK
        rounds.append({"at": now(), "region": region, "seed": seed,
                       "along": None if along is None else [float(value) for value in along],
                       "low_zyx": low.tolist(), "high_zyx": high.tolist(),
                       "wanted": int(wanted), "tried": tried, "added": len(added),
                       "usable": usable, "test": bool(test)})
        self.save()
        return added

    def sample_test(self, wanted, seed=None):
        """Draw a test set of `wanted` textured blocks, in equal shares along the scroll."""
        base = len(self.state["sampling"]["rounds"]) if seed is None else int(seed)
        added = []
        for index in range(TEST_STRETCHES):
            share = (wanted * (index + 1)) // TEST_STRETCHES - (wanted * index) // TEST_STRETCHES
            if share:
                added += self.sample(share, seed=base + index, test=True,
                                     along=(index / TEST_STRETCHES,
                                            (index + 1) / TEST_STRETCHES))
        return added

    def _correlatable(self):
        """Patches whose coarse block passed the texture rule."""
        return [patch for patch in self.state["patches"]
                if patch["auto"] in (OK, "not-covered", "shift-out-of-range")]

    def calibrate(self):
        """Fix the sign convention against a planted shift, under the current fit."""
        _coarse_levels, fine_array, fine_scale = self.open()
        patches = self._correlatable()
        blocks = [(np.asarray(patch["origin_zyx"]), self.coarse_block(patch["id"]))
                  for patch in patches]
        if not blocks:
            raise ValueError("no textured patch to calibrate on; sample first")
        sign = dense.calibrate_sign(blocks, fine_array, fine_scale, self.fine_to_coarse(),
                                    cache=self._fine, keys=[patch["id"] for patch in patches])
        self.state["sampling"]["sign"] = float(sign)
        self.save()
        return sign

    def correlate(self, only_new=False, skip_test=False):
        """Measure every textured patch under the current fit's affine part, or with
        `only_new` those not measured yet -- a test set drawn after the fits, say. With
        `skip_test`, test patches are left as they are: the pipeline keeps a hold-out
        measurement exactly as the person who judged it saw it.

        A patch is measured at the NCC optimum its peak check finds next to the phase
        correlation's answer: on PHerc0139 a blind review preferred the optimum eleven
        times to one where the two told apart, among them patches the phase correlation
        had put two to three voxels off. The phase correlation's point and match are kept
        beside it.

        Resampled under the affine and never the bend, as every pass in the research was:
        drawing a block through the spline is a dense kernel per lattice point, and after
        the first pass the remaining shift is a fraction of a voxel either way.
        """
        if self.state["sampling"]["sign"] is None:
            self.calibrate()
        _coarse_levels, fine_array, fine_scale = self.open()
        sign = self.state["sampling"]["sign"]
        fine_to_coarse = self.fine_to_coarse()
        under = self.state["current"]
        patches = [patch for patch in self._correlatable()
                   if (not only_new or patch["measurement"] is None)
                   and not (skip_test and patch.get("test"))]
        for patch in patches:
            self.coarse_block(patch["id"])
        held = {patch["id"]: self._fine.get(patch["id"]) for patch in patches}

        def one(patch):
            origin = np.asarray(patch["origin_zyx"])
            return dense.measure_block(origin, self._coarse[patch["id"]], fine_array,
                                       fine_scale, fine_to_coarse, sign, self._fine,
                                       patch["id"], with_check=True, with_octants=True)

        with ThreadPoolExecutor(max_workers=dense.THREADS) as pool:
            results = list(pool.map(one, patches))
        for patch, measured in zip(patches, results):
            if isinstance(measured, str):
                patch["auto"] = measured
                patch["measurement"] = None
            else:
                centre, fine_point, shift, error, match, check, octants = measured
                optimum = dense.apply(np.linalg.inv(fine_to_coarse),
                                      centre + sign * np.asarray(check["peak_zyx"]))
                patch["auto"] = OK
                patch["measurement"] = {
                    "under": under,
                    "centre_zyx": np.asarray(centre, float).tolist(),
                    # At the NCC optimum; the phase correlation's own point is kept below.
                    "fine_zyx": np.asarray(optimum, float).tolist(),
                    "match": check["peak_match"],
                    "phase_fine_zyx": np.asarray(fine_point, float).tolist(),
                    "phase_match": match,
                    "shift_zyx": np.asarray(shift, float).tolist(),
                    "error": float(error),
                    # Whether the shift is where the blocks match best; see peak_check.
                    "check": check,
                    # The eight half-size sub-blocks on their own; see measure_octants.
                    "octants": octants,
                }
            if self._fine.get(patch["id"]) is not held[patch["id"]]:
                self._save_block(patch["id"])
        self.save()

    def set_verdict(self, patch_id, verdict, note=None):
        """A person's verdict on a patch, or None to clear it; `note`, if given, replaces
        the patch's note."""
        if verdict is not None and verdict not in VERDICTS:
            raise ValueError(f"unknown verdict {verdict!r}; one of {', '.join(VERDICTS)}")
        patch = self.patch(patch_id)
        patch["human"] = None if verdict is None else {"verdict": verdict, "at": now()}
        if note is not None:
            self._note(patch, note)
        self.save()

    def set_note(self, patch_id, text):
        """What a person saw in a patch, in their words; empty to remove it.

        Kept apart from the verdict, so a patch can carry an observation before anyone
        decides on it -- "own fit somewhat off too" -- and changing the verdict keeps it.
        """
        self._note(self.patch(patch_id), text)
        self.save()

    def set_test_answer(self, unit_id, answer):
        """A person's answer about a test patch, or `<patch>.<octant>` for one of its
        octants (one of `TEST_ANSWERS`), or None to clear it.

        Kept apart from the verdict, which is about whether a fit should use a patch:
        a test patch is never used, and what is asked of it is whether its own
        measurement is an answer a fit can be scored against.
        """
        if answer is not None and answer not in TEST_ANSWERS:
            raise ValueError(f"unknown answer {answer!r}; one of {', '.join(TEST_ANSWERS)}")
        patch_id, _dot, octant = unit_id.partition(".")
        patch = self.patch(patch_id)
        if not patch.get("test"):
            raise ValueError(f"{patch_id} is not in the test set")
        value = None if answer is None else {"answer": answer, "at": now()}
        if not octant:
            patch["test_answer"] = value
        else:
            if int(octant) not in {index for index, _c, _f in self.octant_points(patch["measurement"])}:
                raise ValueError(f"{unit_id} is not an octant that measured well")
            answers = patch.setdefault("test_octants", {})
            if value is None:
                answers.pop(octant, None)
            else:
                answers[octant] = value
        self.save()

    def octant_points(self, measurement):
        """`(index, coarse_zyx, fine_zyx)` for each octant of a measurement that measured
        well, at its NCC optimum -- see `points`."""
        if not measurement:
            return []
        sign = self.state["sampling"]["sign"]
        to_fine = np.linalg.inv(self.fine_to_coarse(measurement["under"]))
        found = []
        for index, octant in enumerate(measurement.get("octants") or []):
            if "rejected" in octant or octant["check"]["peak_match"] < OCTANT_MIN_MATCH:
                continue
            centre = np.asarray(octant["centre_zyx"], float)
            fine = dense.apply(to_fine, centre + sign * np.asarray(octant["check"]["peak_zyx"]))
            found.append((index, centre.tolist(), np.asarray(fine, float).tolist()))
        return found

    def test_units(self, octants=False):
        """The test set as it is judged: one unit per measured test patch, or with
        `octants` one per octant of one that measured well, each at its own measurement."""
        units = []
        for patch in self.state["patches"]:
            measurement = patch["measurement"]
            if not patch.get("test") or measurement is None:
                continue
            if not octants:
                given = patch.get("test_answer") or {}
                units.append(TestUnit(patch["id"], patch, (0, 0, 0), dense.BLOCK,
                                      measurement["centre_zyx"], measurement["fine_zyx"],
                                      given.get("answer"), given.get("at")))
                continue
            answers = patch.get("test_octants") or {}
            for index, centre, fine in self.octant_points(measurement):
                given = answers.get(str(index)) or {}
                units.append(TestUnit(f"{patch['id']}.{index}", patch,
                                      tuple(measurement["octants"][index]["corner_zyx"]),
                                      dense.BLOCK // 2, centre, fine,
                                      given.get("answer"), given.get("at")))
        return units

    def stretch(self, patch):
        """Which of the `TEST_STRETCHES` stretches along the scroll a test patch was drawn in."""
        along = self.state["sampling"]["rounds"][patch["round"]].get("along") or [0.0, 1.0]
        return min(int(along[0] * TEST_STRETCHES + 0.5), TEST_STRETCHES - 1)

    def test_progress(self, octants=False):
        """Per stretch: how many test units line up, are answered, and are left to look
        at. Says nothing about any fit, so a reviewer can watch it."""
        stretches = [{"lines_up": 0, "answered": 0, "left": 0} for _ in range(TEST_STRETCHES)]
        answers = dict.fromkeys(TEST_ANSWERS, 0)
        for unit in self.test_units(octants):
            into = stretches[self.stretch(unit.patch)]
            answer = unit.answer
            if answer is None:
                into["left"] += 1
                continue
            into["answered"] += 1
            into["lines_up"] += answer == LINES_UP
            answers[answer] += 1
        return {"stretches": stretches, "answers": answers, "target": TEST_TARGET,
                "lines_up": answers[LINES_UP]}

    def next_test(self, octants=False):
        """The test unit to look at next: an unanswered one from the stretch that has the
        fewest lining up so far, so coverage never depends on how far a review got."""
        progress = self.test_progress(octants)["stretches"]
        waiting = [unit for unit in self.test_units(octants) if unit.answer is None]
        if not waiting:
            return None
        return min(waiting, key=lambda unit: (progress[self.stretch(unit.patch)]["lines_up"],
                                              self.stretch(unit.patch), unit.id)).id

    @staticmethod
    def _note(patch, text):
        text = (text or "").strip()
        patch["note"] = {"text": text, "at": now()} if text else None

    # ------------------------------------------------------------------------ fitting

    def candidates(self):
        """Patches a fit may use: measured, not marked bad by a person, not in the test set."""
        chosen = []
        for patch in self.state["patches"]:
            if patch["auto"] != OK or patch["measurement"] is None or patch.get("test"):
                continue
            human = patch["human"]
            if human is not None and human["verdict"] not in USABLE | {OCTANTS}:
                continue
            chosen.append(patch)
        return chosen

    def points(self, patches, auto_octants=False, every_octant=False):
        """The points a fit is made from: `(id, coarse_zyx, fine_zyx, trusted)`.

        One per patch, at its measurement, except for a patch marked "use-octants" -- or,
        with `auto_octants`, one no person judged that `wants_octants`, or with
        `every_octant` any no person judged -- which gives one per octant that measured
        well, with ids `<patch>.<octant>`. An octant is
        placed at its NCC optimum rather than its phase-correlation shift, which on a
        half-size block is biased towards zero. A point is trusted -- kept whatever RANSAC
        says -- when a person judged its patch.
        """
        points = []
        for patch in patches:
            measurement, human = patch["measurement"], patch["human"]
            by_octants = (human["verdict"] == OCTANTS if human is not None
                          else every_octant or (auto_octants and wants_octants(measurement)))
            if not by_octants:
                points.append((patch["id"], measurement["centre_zyx"],
                               measurement["fine_zyx"], human is not None))
                continue
            for index, centre, fine in self.octant_points(measurement):
                points.append((f"{patch['id']}.{index}", centre, fine, human is not None))
        return points

    def fit(self, hold_out=None, robust=None, unit=None):
        """Fit the current measurements and make the result the current transform.

        `hold_out` is one of `HOLD_OUTS`, `robust` one of `ROBUST`, `unit` one of `UNITS`;
        None keeps the current fit's, "regions", "ransac" and "patches" for a session's
        first. With "gate", a patch no person judged is also split into octants when
        `wants_octants` says so; with "octants", always.

        The robust step is the one the research used: RANSAC against an affine with the
        threshold in coarse voxels. A patch a person marked good is used even when RANSAC
        would drop it -- the person looked, and RANSAC cannot tell a block that bent from
        one that is wrong. Then the model ladder, then a bend if a region hold-out says it
        beats the affine.
        """
        if hold_out is None:
            hold_out = (self.fit_record() or {}).get("hold_out", "regions")
        if hold_out not in HOLD_OUTS:
            raise ValueError(f"unknown hold-out {hold_out!r}; one of {', '.join(HOLD_OUTS)}")
        if robust is None:
            robust = (self.fit_record() or {}).get("robust", "ransac")
        if robust not in ROBUST:
            raise ValueError(f"unknown robust rule {robust!r}; one of {', '.join(ROBUST)}")
        if unit is None:
            unit = (self.fit_record() or {}).get("unit", "patches")
        if unit not in UNITS:
            raise ValueError(f"unknown unit {unit!r}; one of {', '.join(UNITS)}")
        patches = self.candidates()
        points = self.points(patches, auto_octants=robust == "gate",
                             every_octant=unit == "octants")
        if len(points) < 6:
            raise ValueError(f"only {len(points)} usable points; a fit needs at least 6")
        ids = [point[0] for point in points]
        coarse = np.array([point[1] for point in points], float)
        fine = np.array([point[2] for point in points], float)
        fixed, moving = (coarse, fine) if self.fixed_is_coarse else (fine, coarse)

        threshold = dense.RANSAC_THRESHOLD_COARSE_VOXELS * self.coarse_um / self.fixed_um
        if robust == "ransac":
            inliers = set(int(index) for index in dense.ransac(fixed, moving, threshold))
        else:
            match = {patch["id"]: patch["measurement"]["match"] for patch in patches}
            # An octant point has already passed OCTANT_MIN_MATCH on its own.
            inliers = {index for index, point in enumerate(points)
                       if "." in point[0] or (match[point[0]] or 0) >= GATE_MIN_MATCH}
        trusted = {index for index, point in enumerate(points) if point[3]}
        used = sorted(inliers | trusted)
        selection = select_transform_from_points(fixed[used], moving[used])
        if selection is None:
            raise ValueError("no model could be fitted to the usable patches")

        matrix_xyz = matrix_swap_xyz_zyx(selection.matrix)
        fixed_xyz = np.array(points_swap_xyz_zyx(fixed[used]), float)
        moving_xyz = np.array(points_swap_xyz_zyx(moving[used]), float)
        labels, cap = None, None
        if hold_out == "patches":
            owners = [ids[index].split(".")[0] for index in used]
            labels, cap = np.unique(owners, return_inverse=True)[1], MAX_PATCH_LEVERAGE
        correction = fit_chosen_correction(fixed_xyz, moving_xyz, matrix_xyz[:3], labels,
                                           max_leverage=cap, groups=labels)

        fit_id = f"f{len(self.state['fits']) + 1:03d}"
        record = selection_to_json(selection)
        record.update(method="dense-correlation", block_voxels=dense.BLOCK,
                      point_estimator="windowed-centroid", session_fit=fit_id)
        path = self.directory / "fits" / f"{fit_id}.json"
        write_transform_json(str(path), self.state["fixed"]["url"], matrix_xyz,
                             fixed_xyz.tolist(), moving_xyz.tolist(), fit=record,
                             deformation=correction.to_json() if correction else None)

        self.state["fits"].append({
            "id": fit_id,
            "parent": self.state["current"],
            "at": now(),
            "measured_under": sorted({patch["measurement"]["under"] for patch in patches}),
            "candidates": [patch["id"] for patch in patches],
            # Point ids: a patch's own, or `<patch>.<octant>` for an octant of one.
            "ransac_dropped": [ids[index] for index in range(len(ids))
                               if index not in inliers],
            "used": [ids[index] for index in used],
            "model": selection.model,
            "held_out_rms": selection.held_out_rms,
            "hold_out": hold_out,
            "robust": robust,
            "unit": unit,
            "guard_failures": list(selection.guard_failures),
            "bend": (None if correction is None else
                     {"regularisation": correction.spline.lam,
                      "effective_parameters": 3 * correction.spline.edf,
                      # Points weighted down so no one patch holds up the bend alone.
                      "weighted_down": {} if correction.spline.point_weights is None else
                      {ids[index]: float(weight) for index, weight
                       in zip(used, correction.spline.point_weights) if weight < 0.999}}),
            "matrix_xyz": matrix_xyz[:3].tolist(),
            "transform": str(path.relative_to(self.directory)),
        })
        self.state["current"] = fit_id
        self.state["fits"][-1]["diagnostics"] = self._diagnose(
            fit_id, [ids[index] for index in used], fixed_xyz, moving_xyz,
            matrix_xyz[:3], correction)
        self.save()
        return fit_id

    def _diagnose(self, fit_id, used, fixed_xyz, moving_xyz, matrix_xyz, correction):
        """Per point: how much the fit rests on it; per patch: how far its image moved.

        * `leverage` -- the smoother's diagonal: how much of a patch's own displacement the
          fit follows. For a fit without a bend, the same for the least-squares affine.
        * `influence` -- refit the bend without the patch, at the same smoothing, and take
          the largest change that makes anywhere among the used patches, in coarse voxels.
          A large influence from one patch is a bend that patch alone is holding up.
        * `moved` -- how far this fit places the fine image from where the fit before it
          did, at every measured patch, in coarse voxels. Under about half a voxel no
          overlay shows the difference, so this is where a review is worth doing.
        """
        to_coarse = self.fixed_um / self.coarse_um
        result = {patch["id"]: {} for patch in self.state["patches"]
                  if patch["measurement"] is not None}
        if correction is None:
            centre = moving_xyz.mean(0)
            design = np.hstack([np.ones((len(moving_xyz), 1)), moving_xyz - centre])
            leverage = np.einsum("ij,ji->i", design,
                                 np.linalg.pinv(design.T @ design) @ design.T)
        else:
            spline = correction.spline
            leverage = np.diag(smoother(spline.control, spline.lam, spline.point_weights))
            full = correction.apply(moving_xyz)
            keep = np.ones(len(used), bool)
            for index, point_id in enumerate(used):
                keep[index] = False
                without = fit_correction(
                    fixed_xyz[keep], moving_xyz[keep], matrix_xyz, spline.lam,
                    spline.centre, spline.scale, with_edf=False,
                    point_weights=None if spline.point_weights is None
                    else spline.point_weights[keep])
                keep[index] = True
                change = np.linalg.norm(without.apply(moving_xyz) - full, axis=1)
                result.setdefault(point_id, {})["influence"] = float(change.max() * to_coarse)
        for point_id, value in zip(used, leverage):
            result.setdefault(point_id, {})["leverage"] = float(value)

        parent = self.fit_record(fit_id)["parent"]
        to_fine_now, _bent = self.coarse_to_fine(fit_id)
        to_fine_before, _bent = self.coarse_to_fine(parent)
        measured = [patch for patch in self.state["patches"] if patch["measurement"] is not None]
        centres = np.array([patch["measurement"]["centre_zyx"] for patch in measured])
        if len(centres):
            moved = np.linalg.norm(to_fine_now(centres) - to_fine_before(centres), axis=1)
            for patch, value in zip(measured, moved * self.fine_um / self.coarse_um):
                result[patch["id"]]["moved"] = float(value)
        return result

    def refine(self, passes=3, hold_out=None, robust=None, unit=None):
        """Correlate and fit, `passes` times, as `research/dense_blocks.py` does."""
        fit_id = None
        for _ in range(passes):
            self.correlate()
            fit_id = self.fit(hold_out, robust, unit)
        return fit_id

    # --------------------------------------------------------------- blind comparison

    def candidate_map(self, candidate):
        """Coarse to fine ZYX through a fit, or through its affine part alone for
        `<fit>` + `AFFINE`."""
        if candidate.endswith(AFFINE):
            affine = np.linalg.inv(self.fine_to_coarse(candidate.removesuffix(AFFINE)))
            return lambda points: dense.apply(affine, np.asarray(points, float))
        return self.coarse_to_fine(candidate)[0]

    def placement(self, candidate, patch):
        """Where `candidate` -- a fit id, a fit's affine part (`AFFINE`), "own" or "peak" --
        puts a patch's centre, fine ZYX."""
        measurement = patch["measurement"]
        if candidate == OWN:
            return np.asarray(measurement["fine_zyx"], float)
        if candidate == PEAK:
            to_fine = np.linalg.inv(self.fine_to_coarse(measurement["under"]))
            shift = np.asarray(measurement["check"]["peak_zyx"], float)
            return np.asarray(dense.apply(to_fine, np.asarray(measurement["centre_zyx"], float)
                                          + self.state["sampling"]["sign"] * shift), float)
        return self.candidate_map(candidate)(np.asarray([measurement["centre_zyx"]], float))[0]

    def start_comparison(self, candidates, count=20, seed=None, test_only=False):
        """A blind round: the patches where `candidates` disagree most, options shuffled.

        With `test_only`, only test-set patches are asked about: no fit used them, so a
        round there judges the fits on ground none of them was made from.

        Each patch shows the candidates that place it apart from one another as options;
        candidates closer than `SAME_PLACE_VOXELS` share one. Patches judged to hold
        something other than scroll are left out, and so is any patch where every
        candidate agrees. Which option is which candidate stays in the session and is
        not shown until the round is done.
        """
        candidates = list(dict.fromkeys(candidates))
        for candidate in candidates:
            if candidate not in (OWN, PEAK):
                self.fit_record(candidate.removesuffix(AFFINE))
        if len(candidates) < 2:
            raise ValueError("a comparison needs at least two candidates")
        to_coarse = self.fine_um / self.coarse_um
        rounds = self.state.setdefault("comparisons", [])
        rng = np.random.default_rng(len(rounds) if seed is None else int(seed))
        ranked = []
        for patch in self.state["patches"]:
            human = patch["human"]
            if patch["measurement"] is None or (human and human["verdict"] in NOT_SCROLL):
                continue
            if test_only and not patch.get("test"):
                continue
            if PEAK in candidates and not patch["measurement"].get("check"):
                continue
            placed = {candidate: self.placement(candidate, patch) for candidate in candidates}
            options = []
            for candidate in candidates:
                for option in options:
                    if np.linalg.norm(placed[candidate] - placed[option[0]]) * to_coarse \
                            < SAME_PLACE_VOXELS:
                        option.append(candidate)
                        break
                else:
                    options.append([candidate])
            if len(options) < 2:
                continue
            spread = max(np.linalg.norm(placed[a] - placed[b]) * to_coarse
                         for a in candidates for b in candidates)
            ranked.append((spread, patch["id"], options))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        items = []
        for spread, patch_id, options in ranked[:count]:
            order = rng.permutation(len(options))
            items.append({"patch": patch_id, "spread": float(spread),
                          "options": [options[index] for index in order], "choice": None})
        record = {"id": f"c{len(rounds) + 1:03d}", "at": now(), "candidates": candidates,
                  "test_only": bool(test_only), "items": items}
        rounds.append(record)
        self.save()
        return record["id"]

    def comparison(self, round_id):
        return next(record for record in self.state.get("comparisons", [])
                    if record["id"] == round_id)

    def choose(self, round_id, index, choice):
        """The option a person picked for item `index`, or one of `ANSWERS`."""
        item = self.comparison(round_id)["items"][index]
        if choice not in ANSWERS and not (isinstance(choice, int)
                                       and 0 <= choice < len(item["options"])):
            raise ValueError(f"no option {choice!r}")
        item["choice"] = choice
        self.save()

    def tally(self, round_id):
        """Per candidate: the items it won, and of those the ones its option held no other
        candidate; and how many items were left unsure, fitted by none, or unanswered."""
        record = self.comparison(round_id)
        result = {candidate: {"won": 0, "won_alone": 0} for candidate in record["candidates"]}
        unsure = none = unanswered = 0
        for item in record["items"]:
            if item["choice"] is None:
                unanswered += 1
            elif item["choice"] == "unsure":
                unsure += 1
            elif item["choice"] == "none":
                none += 1
            else:
                option = item["options"][item["choice"]]
                for candidate in option:
                    result[candidate]["won"] += 1
                    result[candidate]["won_alone"] += len(option) == 1
        return {"candidates": result, "items": len(record["items"]),
                "unsure": unsure, "none": none, "unanswered": unanswered}

    def score(self, fit_id=None, octants=False):
        """How far a fit lands from the test set's own measurements, in coarse voxels --
        at each test patch, or with `octants` at each of their octants that measured well.

        `checked` holds the test patches a person said line up at their own measurement,
        the score that counts; `unchecked` those nobody has answered yet that pass the
        gate, a first look that flatters a pair whose good patches match poorly. A test
        patch answered anything else is left out either way: its own measurement is no
        answer. `decision` applies `TEST_TARGET` and `TEST_ALLOWED_MISSES` to `checked`;
        `partly` is the share of patches seen whole that were answered "partly".
        """
        fit_id = fit_id or self.state["current"]
        to_coarse = self.fine_um / self.coarse_um
        to_fine = self.candidate_map(fit_id)
        groups = {"checked": [], "unchecked": []}
        for unit in self.test_units(octants):
            if unit.answer is not None:
                if unit.answer != LINES_UP:
                    continue
                group = "checked"
            elif octants or (unit.patch["measurement"].get("match") or 0) >= GATE_MIN_MATCH:
                # An octant is a test unit only once it measured well (OCTANT_MIN_MATCH).
                group = "unchecked"
            else:
                continue
            off = np.linalg.norm(to_fine(np.asarray([unit.centre_zyx], float))[0]
                                 - np.asarray(unit.fine_zyx, float))
            groups[group].append((unit.id, float(off * to_coarse)))
        result = {"fit": fit_id} | {name: spread(values) for name, values in groups.items()}
        checked = result["checked"]
        if checked["patches"] and checked["missed"] > TEST_ALLOWED_MISSES:
            result["decision"] = "fail"
        elif checked["patches"] >= TEST_TARGET:
            result["decision"] = "pass"
        else:
            result["decision"] = "continue"
        answers = self.test_progress(octants)["answers"]
        whole = answers[LINES_UP] + answers["partly"]
        result["partly"] = answers["partly"] / whole if whole else None
        result["miss_voxels"], result["coarse_um"] = TEST_MISS_VOXELS, self.coarse_um
        return result

    def export(self, out, fit_id=None, affine_only=False):
        """Write a fit as a transform file, naming the fixed volume as the atlas does -- by
        its zarr, not the URL the session reads it from; with `affine_only`, without its
        bend, for a pair whose test set shows no credible gain from the bend."""
        record = self.fit_record(fit_id)
        if record is None:
            raise ValueError("nothing has been fitted yet")
        with open(self.directory / record["transform"]) as handle:
            data = json.load(handle)
        data["fixed_volume"] = data["fixed_volume"].rstrip("/").rsplit("/", 1)[-1]
        if not affine_only:
            with open(out, "w") as handle:
                json.dump(data, handle, indent=2)
            return
        matrix = np.vstack([np.asarray(data["transformation_matrix"], float), [0, 0, 0, 1]])
        write_transform_json(str(out), data["fixed_volume"], matrix, data["fixed_landmarks"],
                             data["moving_landmarks"], fit=data.get("fit"))

    def summary(self):
        patches = self.state["patches"]
        count = {}
        for patch in patches:
            count[patch["auto"]] = count.get(patch["auto"], 0) + 1
        human = {}
        for patch in patches:
            if patch["human"]:
                verdict = patch["human"]["verdict"]
                human[verdict] = human.get(verdict, 0) + 1
        tests = sum(bool(patch.get("test")) for patch in patches)
        return {"patches": len(patches), "auto": count, "human": human,
                "current": self.state["current"], "test_patches": tests,
                "test_score": self.score() if tests and self.state["fits"] else None,
                "fits": [{key: fit[key] for key in
                          ("id", "parent", "model", "held_out_rms", "bend",
                           "guard_failures")}
                         | {"used": len(fit["used"]),
                            "ransac_dropped": len(fit["ransac_dropped"])}
                         for fit in self.state["fits"]]}


# ------------------------------------------------------------------------------ CLI

def _init(args):
    if args.pair:
        with open(args.manifest) as handle:
            entry = json.load(handle)[args.pair]
        fixed = entry.get("fixed_direct") or entry["fixed"]
        moving = entry.get("moving_direct") or entry["moving"]
        fixed_um, moving_um = entry["fixed_voxel_size_um"], entry["moving_voxel_size_um"]
    else:
        fixed, moving = args.fixed, args.moving
        fixed_um, moving_um = args.fixed_voxel_size, args.moving_voxel_size
    if None in (fixed, moving, fixed_um, moving_um):
        raise SystemExit("give --pair and --manifest, or both volumes and voxel sizes")
    # The start's affine part only: a deformation it carries is measured again by the
    # first correlation pass, and refitted from those measurements.
    matrix, fixed_points, moving_points = read_transform_json(args.transform)
    Session.create(args.session, fixed, moving, fixed_um, moving_um, matrix,
                   str(args.transform), (fixed_points, moving_points))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    init = commands.add_parser("init", help="start a session from a transform file")
    init.add_argument("session")
    init.add_argument("--transform", required=True)
    init.add_argument("--pair")
    init.add_argument("--manifest")
    init.add_argument("--fixed")
    init.add_argument("--moving")
    init.add_argument("--fixed-voxel-size", type=float)
    init.add_argument("--moving-voxel-size", type=float)

    sample = commands.add_parser("sample", help="draw more blocks")
    sample.add_argument("session")
    sample.add_argument("--blocks", type=int, default=80)
    sample.add_argument("--region", choices=("overlap", "landmarks"), default="overlap")
    sample.add_argument("--seed", type=int)
    sample.add_argument("--along", type=float, nargs=2, metavar=("START", "END"),
                        help="only this stretch along the scroll, as fractions, e.g. 0.8 1")
    sample.add_argument("--test", action="store_true",
                        help="draw a test set, in equal shares along the scroll, that no fit uses")

    for name, text in (("calibrate", "fix the sign convention"),
                       ("overview", "build the overview images of the coarse volume"),
                       ("status", "summarise the session")):
        commands.add_parser(name, help=text).add_argument("session")

    correlate = commands.add_parser("correlate", help="measure every patch under the current fit")
    correlate.add_argument("session")
    correlate.add_argument("--only-new", action="store_true",
                           help="only the patches not measured yet")

    score = commands.add_parser("score", help="score a fit on the test set")
    score.add_argument("session")
    score.add_argument("--fit")

    fit = commands.add_parser("fit", help="fit the current measurements")
    fit.add_argument("session")
    fit.add_argument("--hold-out", choices=HOLD_OUTS,
                     help="how the bend's stiffness is chosen; default: the current fit's")
    fit.add_argument("--robust", choices=ROBUST,
                     help="which unjudged patches a fit uses; default: the current fit's")
    fit.add_argument("--unit", choices=UNITS,
                     help="whole patches or their octants as points; default: the current fit's")

    refine = commands.add_parser("refine", help="correlate and fit, repeatedly")
    refine.add_argument("session")
    refine.add_argument("--passes", type=int, default=3)
    refine.add_argument("--hold-out", choices=HOLD_OUTS)
    refine.add_argument("--robust", choices=ROBUST)
    refine.add_argument("--unit", choices=UNITS)

    verdict = commands.add_parser("verdict", help="set a person's verdict on a patch")
    verdict.add_argument("session")
    verdict.add_argument("patch")
    verdict.add_argument("verdict", choices=VERDICTS + ("clear",))
    verdict.add_argument("--note")

    export = commands.add_parser("export", help="write a fit as a transform file")
    export.add_argument("session")
    export.add_argument("out")
    export.add_argument("--fit")
    export.add_argument("--affine-only", action="store_true",
                        help="leave the bend out: the fit's affine part alone")

    args = parser.parse_args(argv)
    if args.command == "init":
        _init(args)
        return
    session = Session.load(args.session)
    if args.command == "sample":
        added = (session.sample_test(args.blocks, args.seed) if args.test else
                 session.sample(args.blocks, args.region, args.seed, args.along))
        print(f"added {len(added)} patches")
    elif args.command == "calibrate":
        print(f"sign {session.calibrate():+.0f}")
    elif args.command == "correlate":
        session.correlate(args.only_new)
    elif args.command == "score":
        json.dump(session.score(args.fit), sys.stdout, indent=1)
        print()
    elif args.command == "overview":
        print(json.dumps(session_overview.build(session), indent=1))
    elif args.command == "fit":
        print(session.fit(args.hold_out, args.robust, args.unit))
    elif args.command == "refine":
        print(session.refine(args.passes, args.hold_out, args.robust, args.unit))
    elif args.command == "verdict":
        session.set_verdict(args.patch, None if args.verdict == "clear" else args.verdict,
                            args.note)
    elif args.command == "export":
        session.export(args.out, args.fit, args.affine_only)
    if args.command in ("status", "fit", "refine", "correlate", "sample"):
        json.dump(session.summary(), sys.stdout, indent=1)
        print()


if __name__ == "__main__":
    main()
