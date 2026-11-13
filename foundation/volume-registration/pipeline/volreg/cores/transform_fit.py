"""Fit the transform model the landmarks can actually support, and prove it.

A 12-parameter affine fitted to 4 landmark pairs reproduces those 4 pairs exactly and
predicts nothing: `example_transform.json` in this directory records an RMS of 0.00 for
precisely that reason. The usual fix is to pick a smaller model by hand and remember to
switch back later. This module picks it from the data instead.

Two rules do the work:

**A model must survive losing a pair.** With `n` pairs and `p` parameters, a fit is only
offered when `3(n - 1) > p` *and* the held-out fit keeps at least 4 points. That makes
leave-one-out error a real number rather than an artefact, and it is what stops a model
from being chosen on the strength of a fit it could not have got wrong. The 4-point floor
is not redundant with the equation count: three points are always coplanar, so a held-out
fit on 3 of them is ambiguous under reflection and its error is noise -- measurably so,
disagreeing with itself by up to 3000x depending on coordinate order.

**Among models that predict equally well, the simpler one wins.** The model with the
lowest held-out error is only preferred over a simpler one when it beats it by more than
one standard error, which keeps the choice from flapping between models on noise.

Measured on 16 real landmark sets (120 random subsets each), the crossovers come out at
similarity7 for 5-8 pairs and scaled-rigid9 from about 9, with affine12 taking over where
the data supports it -- so in practice this reproduces what a careful person would have
chosen, without being told.

The guards below then refuse a fit that is geometrically impossible for two scans of one
object, whatever its residual says. Their thresholds come from measuring all 30 committed
transform files: see `auto-refinement-research.md`.
"""

from typing import Callable, NamedTuple, Optional

import numpy as np

from .point_fits import fit_constrained_transform_from_points


class ModelSpec(NamedTuple):
    """One transform model on the ladder."""

    name: str
    parameters: int
    fit: Callable
    auto_select: bool
    minimum_pairs: int
    description: str

    def verifiable_pairs(self) -> int:
        """Fewest pairs that leave the fit *determined* after holding one out.

        Each pair supplies 3 equations, so counting gives the smallest `n` with
        `3(n - 1) > p`. Counting is not enough: three points always lie in a plane, and
        matching two triangles leaves a reflection through that plane undetermined no
        matter how many equations are nominally available. Measured on the real landmark
        sets, a held-out similarity7 fit on 3 points disagrees with itself by up to 3000x
        depending on which order its coordinates arrive in, in a third of all subsets,
        while every fit on 4 or more points is exactly order-invariant.
        """
        return max(self.parameters // 3 + 2, MIN_VERIFIABLE_PAIRS)


# A held-out fit must keep at least 4 points, or its points are forced to be coplanar and
# the linear part is not determined by them. That makes 5 the fewest pairs any model here
# can be chosen on.
MIN_VERIFIABLE_PAIRS = 5


class ModelScore(NamedTuple):
    """How one model did on one landmark set."""

    name: str
    parameters: int
    in_sample_rms: float
    held_out_rms: float
    held_out_se: float


class Geometry(NamedTuple):
    """What the fitted linear part does, in terms that can be sanity-checked."""

    scale: float
    anisotropy: float
    shear_deg: float
    reflected: bool


class FitSelection(NamedTuple):
    """The chosen transform, why it was chosen, and whether it is believable."""

    matrix: np.ndarray
    model: str
    parameters: int
    landmark_pairs: int
    in_sample_rms: float
    held_out_rms: Optional[float]
    selection: str
    scores: list
    geometry: Geometry
    guard_failures: list

    @property
    def ok(self) -> bool:
        return not self.guard_failures


# --------------------------------------------------------------------------- guards
#
# Every one of the 23 plausible committed transforms has anisotropy <= 1.027 and shear
# <= 0.81 deg; the file with a visibly broken fit has 4.05 and 60.5 deg. Two scans of one
# object cannot differ by a genuine shear, so these bounds sit well clear of real data
# without needing to be tight.
MAX_ANISOTROPY = 1.05
MAX_SHEAR_DEG = 2.0

# The fitted scale is known in advance from the two voxel sizes, so it is a free check.
# A quarter is far more than calibration error and far less than a wrong flip or a
# mismatched resolution level.
MAX_SCALE_RATIO = 1.25

# Residual measured against the spread of the landmark cloud, so the threshold does not
# depend on resolution or on how far apart the landmarks are. Across the committed files
# the plausible ones reach 0.030 and the broken ones start at 0.085.
MAX_RESIDUAL_FRACTION = 0.05

# Whether a volume's slices were written in a left-handed order is a property of the
# reconstruction, which the atlas records per volume. A transform between two volumes must
# therefore reverse orientation exactly when their flags differ -- true for all 26
# committed transforms, and three volumes really are left-handed (two DLS 7.91 um scans of
# PHercParis4 and the 0.55 um ESRF scan of PHerc0500P2). No amount of bad clicking changes
# the sign of a determinant, so a disagreement means one side is mirrored with respect to
# the volume it names, or a volume's flag is wrong.
#
# What it does not catch: swapping a matrix between XYZ and ZYX order is a similarity
# transform and exchanging the fixed and moving sides is an inversion, and both preserve
# the sign. Only an actual mirroring shows up here.


# --------------------------------------------------------------------------- models

def fit_affine12(fixed, moving):
    """Unconstrained affine: 9 linear parameters and 3 translations."""
    moving_h = np.column_stack([moving, np.ones(len(moving))])
    solution = fixed.T @ np.linalg.pinv(moving_h.T)
    return solution[:, :3], solution[:, 3]


def fit_similarity7(fixed, moving):
    """Rotation, one isotropic scale, translation (Umeyama).

    Both handednesses are tried and the better kept: several of these registrations
    genuinely include an axis flip, and forcing a proper rotation turns those into
    nonsense -- it cost a factor of a thousand in residual when it was left in by mistake.
    """
    mu_f, mu_m = fixed.mean(0), moving.mean(0)
    centred_f, centred_m = fixed - mu_f, moving - mu_m
    covariance = centred_f.T @ centred_m / len(fixed)
    u, singulars, vt = np.linalg.svd(covariance)
    moving_variance = (centred_m ** 2).sum() / len(moving)

    best = None
    for handedness in (1, -1):
        correction = np.eye(3)
        if np.linalg.det(u) * np.linalg.det(vt) * handedness < 0:
            correction[2, 2] = -1
        rotation = u @ correction @ vt
        scale = np.trace(np.diag(singulars) @ correction) / moving_variance
        linear = scale * rotation
        offset = mu_f - linear @ mu_m
        cost = ((fixed - (moving @ linear.T + offset)) ** 2).sum()
        if best is None or cost < best[0]:
            best = (cost, linear, offset)
    return best[1], best[2]


def fit_scaled_rigid9(fixed, moving):
    """Rotation, an independent scale per axis, translation.

    Solved by alternating between the rotation (an SVD given the scales) and the scales
    (a least-squares solve given the rotation), which converges in a handful of rounds.
    """
    mu_f, mu_m = fixed.mean(0), moving.mean(0)
    centred_f, centred_m = fixed - mu_f, moving - mu_m
    scale = np.ones(3)
    linear = np.eye(3)
    for _ in range(50):
        u, _s, vt = np.linalg.svd(centred_f.T @ (centred_m * scale))
        correction = np.eye(3)
        if np.linalg.det(u) * np.linalg.det(vt) < 0:
            correction[2, 2] = -1
        rotation = u @ correction @ vt
        denominator = (centred_m ** 2).sum(0)
        scale = np.where(denominator > 0, (centred_f @ rotation * centred_m).sum(0) / denominator, 1.0)
        linear = rotation @ np.diag(scale)
    return linear, mu_f - linear @ mu_m


def fit_constrained5(fixed, moving):
    """The legacy constrained mode: isotropic scale, z-rotation, translation, 8 flips.

    Kept selectable so existing sessions can reproduce their old results. It is not
    offered to automatic selection: on all 17 landmark sets it could be tested against it
    lost to scaled-rigid9, by between 2x and 25x, including where its assumption that the
    two volumes share a z axis holds to within a tenth of a degree.
    """
    matrix = fit_constrained_transform_from_points(fixed.tolist(), moving.tolist())
    if matrix is None:
        raise ValueError("constrained5 needs at least 3 landmark pairs")
    return np.asarray(matrix)[:3, :3], np.asarray(matrix)[:3, 3]


# `minimum_pairs` is what the model needs to be solvable at all, used when a model is
# forced; `verifiable_pairs()` is what it needs to be chosen automatically.
MODELS = (
    ModelSpec("affine12", 12, fit_affine12, True, 4, "affine, 9 linear + 3 translation"),
    ModelSpec("scaled-rigid9", 9, fit_scaled_rigid9, True, 3, "rotation + per-axis scale + translation"),
    ModelSpec("similarity7", 7, fit_similarity7, True, 3, "rotation + one scale + translation"),
    ModelSpec("constrained5", 5, fit_constrained5, False, 3, "isotropic scale + z-rotation + translation"),
)

MODELS_BY_NAME = {spec.name: spec for spec in MODELS}
AUTO = "auto"
FIT_MODE_CHOICES = (AUTO,) + tuple(spec.name for spec in MODELS)


# --------------------------------------------------------------------------- scoring

def to_matrix(linear, offset) -> np.ndarray:
    """Pack a linear part and a translation into a 4x4 homogeneous matrix."""
    matrix = np.eye(4)
    matrix[:3, :3] = linear
    matrix[:3, 3] = offset
    return matrix


def residuals(linear, offset, fixed, moving) -> np.ndarray:
    """Per-pair distance between a fixed landmark and its transformed partner."""
    return np.linalg.norm((moving @ linear.T + offset) - fixed, axis=1)


def in_sample_rms(linear, offset, fixed, moving) -> float:
    """RMS over the pairs the fit was given -- optimistic, and the number the tool used to report."""
    return float(np.sqrt(np.mean(residuals(linear, offset, fixed, moving) ** 2)))


def held_out_errors(spec: ModelSpec, fixed, moving) -> np.ndarray:
    """Error on each pair, from a fit that never saw that pair."""
    errors = np.empty(len(fixed))
    for i in range(len(fixed)):
        keep = np.arange(len(fixed)) != i
        linear, offset = spec.fit(fixed[keep], moving[keep])
        errors[i] = np.linalg.norm(moving[i] @ linear.T + offset - fixed[i])
    return errors


def available_models(pair_count: int, include_manual_only: bool = False) -> list:
    """Models that stay over-determined with one pair held out."""
    return [
        spec for spec in MODELS
        if (spec.auto_select or include_manual_only) and pair_count >= spec.verifiable_pairs()
    ]


def score_models(fixed, moving, specs) -> list:
    """Score each model by held-out error, with the spread needed to compare them."""
    scores = []
    for spec in specs:
        try:
            errors = held_out_errors(spec, fixed, moving)
            linear, offset = spec.fit(fixed, moving)
        except Exception:  # noqa: BLE001 - a model that cannot fit is simply not a candidate
            continue
        scores.append(
            ModelScore(
                name=spec.name,
                parameters=spec.parameters,
                in_sample_rms=in_sample_rms(linear, offset, fixed, moving),
                held_out_rms=float(np.sqrt(np.mean(errors ** 2))),
                held_out_se=float(np.std(errors) / np.sqrt(len(errors))),
            )
        )
    return scores


def choose(scores: list) -> Optional[ModelScore]:
    """The simplest model that is not clearly beaten by the best one.

    "Clearly" means by more than one standard error of the held-out error; inside that
    band the models are indistinguishable on this much data, and the fewer parameters the
    better it will extrapolate away from the landmarks.
    """
    if not scores:
        return None
    best = min(scores, key=lambda s: s.held_out_rms)
    tolerance = best.held_out_rms + best.held_out_se
    within = [s for s in scores if s.held_out_rms <= tolerance]
    return min(within, key=lambda s: s.parameters)


# --------------------------------------------------------------------------- geometry

def describe_geometry(linear) -> Geometry:
    """Decompose the linear part into the quantities worth sanity-checking."""
    singulars = np.linalg.svd(linear, compute_uv=False)
    unit = linear / np.linalg.norm(linear, axis=0)
    overlaps = unit.T @ unit
    worst_overlap = max(abs(overlaps[0, 1]), abs(overlaps[0, 2]), abs(overlaps[1, 2]))
    return Geometry(
        scale=float(np.cbrt(np.prod(singulars))),
        anisotropy=float(singulars.max() / singulars.min()),
        shear_deg=float(np.degrees(np.arcsin(np.clip(worst_overlap, 0.0, 1.0)))),
        reflected=bool(np.linalg.det(linear) < 0),
    )


def check_guards(
    geometry: Geometry,
    in_sample: float,
    held_out: Optional[float],
    fixed,
    nominal_scale: Optional[float],
    expect_reflected: Optional[bool] = None,
) -> list:
    """Reasons to refuse this transform, in plain words. Empty means it is believable."""
    failures = []

    if geometry.anisotropy > MAX_ANISOTROPY:
        failures.append(
            f"axis scales differ by {geometry.anisotropy:.2f}x (limit {MAX_ANISOTROPY:.2f}x) -- "
            "two scans of one object cannot stretch unevenly by this much"
        )

    if geometry.shear_deg > MAX_SHEAR_DEG:
        failures.append(
            f"axes sheared by {geometry.shear_deg:.1f} deg (limit {MAX_SHEAR_DEG:.1f} deg) -- "
            "a rigid object has no shear"
        )

    if expect_reflected is not None and geometry.reflected != expect_reflected:
        sides = "differ" if expect_reflected else "agree"
        failures.append(
            f"the fit is {'' if geometry.reflected else 'not '}orientation-reversing, but the "
            f"two volumes {sides} on handedness -- one landmark set is mirrored with respect "
            "to the volume it names, or a volume's handedness is recorded wrongly"
        )

    if nominal_scale is not None and nominal_scale > 0 and geometry.scale > 0:
        ratio = geometry.scale / nominal_scale
        if ratio > MAX_SCALE_RATIO or ratio < 1.0 / MAX_SCALE_RATIO:
            failures.append(
                f"scale {geometry.scale:.4f} is {ratio:.2f}x the {nominal_scale:.4f} the voxel "
                "sizes imply -- check the voxel sizes, or whether a landmark pair is mismatched"
            )

    spread = float(np.sqrt(np.mean(np.sum((fixed - fixed.mean(0)) ** 2, axis=1))))
    if spread > 0 and in_sample / spread > MAX_RESIDUAL_FRACTION:
        failures.append(
            f"residual is {in_sample / spread:.1%} of the landmark spread "
            f"(limit {MAX_RESIDUAL_FRACTION:.0%}) -- the landmarks do not agree on any transform"
        )

    # The in-sample residual cannot see a fit that merely memorised its landmarks. One
    # committed file fits 4 pairs to 1.5 voxels and misses a fifth by 386, and only the
    # held-out number says so.
    if spread > 0 and held_out is not None and held_out / spread > MAX_RESIDUAL_FRACTION:
        failures.append(
            f"a landmark held out of the fit lands {held_out / spread:.1%} of the landmark "
            f"spread away (limit {MAX_RESIDUAL_FRACTION:.0%}) -- this transform fits the "
            "landmarks it was given and predicts nothing"
        )

    return failures


# --------------------------------------------------------------------------- entry point

def select_transform_from_points(
    fixed_points,
    moving_points,
    *,
    mode: str = AUTO,
    nominal_scale: Optional[float] = None,
    expect_reflected: Optional[bool] = None,
) -> Optional[FitSelection]:
    """Fit the landmarks and report which model was used and whether to trust it.

    `mode` is `auto` or a model name. Returns None when there are too few pairs for the
    requested mode; a returned selection with `guard_failures` is a fit the caller should
    refuse to apply, not one it should quietly use.
    """
    fixed = np.asarray(fixed_points, dtype=float)
    moving = np.asarray(moving_points, dtype=float)
    if len(fixed) != len(moving) or len(fixed) < min(spec.minimum_pairs for spec in MODELS):
        return None

    if mode == AUTO:
        specs = available_models(len(fixed))
        scores = score_models(fixed, moving, specs)
        chosen = choose(scores)
        if chosen is None:
            return None
        spec = MODELS_BY_NAME[chosen.name]
        selection = "auto"
        held_out = chosen.held_out_rms
    else:
        spec = MODELS_BY_NAME.get(mode)
        if spec is None:
            raise ValueError(f"unknown fit model: {mode}")
        if len(fixed) < spec.minimum_pairs:
            return None
        # Below the verifiable count a held-out fit is itself under-determined, so its
        # error would be an artefact. Report no held-out number rather than a made-up one.
        scores = score_models(fixed, moving, [spec]) if len(fixed) >= spec.verifiable_pairs() else []
        selection = "forced"
        held_out = scores[0].held_out_rms if scores else None

    linear, offset = spec.fit(fixed, moving)
    in_sample = in_sample_rms(linear, offset, fixed, moving)
    geometry = describe_geometry(linear)

    return FitSelection(
        matrix=to_matrix(linear, offset),
        model=spec.name,
        parameters=spec.parameters,
        landmark_pairs=len(fixed),
        in_sample_rms=in_sample,
        held_out_rms=held_out,
        selection=selection,
        scores=scores,
        geometry=geometry,
        guard_failures=check_guards(geometry, in_sample, held_out, fixed, nominal_scale,
                                    expect_reflected),
    )


def format_selection(selection: FitSelection) -> str:
    """A compact account of the fit, for the terminal."""
    lines = []
    width = 64
    lines.append("-" * width)
    held_out = (
        f"{selection.held_out_rms:.2f}" if selection.held_out_rms is not None else "not verifiable"
    )
    lines.append(
        f"  Fit: {selection.model} ({selection.parameters} params, {selection.selection}) "
        f"from {selection.landmark_pairs} pairs"
    )
    lines.append(f"  in-sample RMS {selection.in_sample_rms:.2f}    held-out RMS {held_out}")
    lines.append("-" * width)

    if len(selection.scores) > 1:
        lines.append(f"  {'model':<16}{'params':>8}{'in-sample':>12}{'held-out':>12}")
        for score in sorted(selection.scores, key=lambda s: s.parameters):
            marker = "*" if score.name == selection.model else " "
            lines.append(
                f"{marker} {score.name:<15}{score.parameters:>8}"
                f"{score.in_sample_rms:>12.2f}{score.held_out_rms:>12.2f}"
            )
        lines.append("-" * width)

    geometry = selection.geometry
    lines.append(
        f"  scale {geometry.scale:.4f}   anisotropy {geometry.anisotropy:.3f}   "
        f"shear {geometry.shear_deg:.2f} deg" + ("   reflected" if geometry.reflected else "")
    )
    for failure in selection.guard_failures:
        lines.append(f"  ! REFUSED: {failure}")
    lines.append("-" * width)
    return "\n".join(lines)


def selection_to_json(selection: FitSelection) -> dict:
    """The provenance record written into the transform file."""
    return {
        "method": "landmark-fit",
        "model": selection.model,
        "parameters": selection.parameters,
        "selection": selection.selection,
        "landmark_pairs": selection.landmark_pairs,
        "in_sample_rms": round(selection.in_sample_rms, 4),
        "held_out_rms": (
            round(selection.held_out_rms, 4) if selection.held_out_rms is not None else None
        ),
        "candidates": [
            {
                "model": score.name,
                "parameters": score.parameters,
                "in_sample_rms": round(score.in_sample_rms, 4),
                "held_out_rms": round(score.held_out_rms, 4),
            }
            for score in sorted(selection.scores, key=lambda s: s.parameters)
        ],
        "geometry": {
            "scale": round(selection.geometry.scale, 6),
            "anisotropy": round(selection.geometry.anisotropy, 4),
            "shear_deg": round(selection.geometry.shear_deg, 4),
            "reflected": selection.geometry.reflected,
        },
    }
