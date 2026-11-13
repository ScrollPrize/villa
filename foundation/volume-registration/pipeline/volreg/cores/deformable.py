"""A transform allowed to bend: regularised thin-plate splines in three dimensions.

Every model on the `transform_fit` ladder is one linear map for the whole volume. Dense
correlation measured what that costs: after the best global affine, 3 to 8 um of
displacement is left at each block, five to twenty-five times the measurement's own
precision, and `research/dense_variogram.py` showed it is a *field* -- two blocks close
together disagree by 2.75 um where two far apart disagree by 20. A model that may vary
across the volume should absorb most of that.

A thin-plate spline is the smoothest such model in a precise sense: among all maps that
hit the correspondences, it minimises the bending energy. In three dimensions its Green's
function is U(r) = -r -- the sign matters and is not cosmetic. The bending energy the
regularisation trades against is W^T K W, and that is a penalty only where K is positive
semi-definite on the subspace the constraint picks out. Measured on the null space of P^T,
+r gives eigenvalues from -1.99 to -0.18 and -r gives +0.18 to +1.99, so with +r the ridge
term pushes the solution the wrong way: the degrees of freedom came out at 50 on 30
correspondences, which is not a smoother at all. The map is

    y(x) = [1 x] A - sum_i w_i * |x - c_i|        subject to  P^T W = 0

solved from

    [ K + lam I   P ] [W]   [Y]
    [ P^T         0 ] [A] = [0]

and the two ends of `lam` are what make this a ladder rather than a leap:

  * lam = 0 interpolates every correspondence exactly, which on measurements means
    interpolating their noise as well
  * lam -> infinity drives W to zero and A to the least-squares affine, so the model
    degenerates to precisely the affine12 that is shipped today

So the deformable model is a strict superset of the current one, and choosing `lam`
chooses how far along that line to go. It is chosen the same way the ladder chooses a
model -- the simplest within one standard error of the best -- with effective degrees of
freedom standing in for the parameter count, since a smoother has no integer one.
`effective_dof` is the trace of the matrix taking measured displacements to fitted ones;
it runs from 4 at the affine limit to the number of correspondences at interpolation.

What `lam` must be chosen against is not negotiable. Leave-one-out surrounds the held-out
point with its own neighbours, all of them in the training set, and a model free to bend
towards them is rewarded for doing so -- `research/blocked_cv.py` bounds that concern for
7-to-12-parameter models and shows it does not apply to them, and the same argument says
it applies with full force here. Hold out a region.

Coordinates are normalised to the unit ball before the kernel sees them, so `lam` is
dimensionless and means the same thing on a 2.4 um volume as on a 9.4 um one.

Points are XYZ throughout and the map runs moving -> fixed, matching
`transform_schema.json`. Nothing here reads or writes files.
"""

from typing import NamedTuple, Optional

import numpy as np
from scipy.spatial.distance import cdist

from .transform_fit import ModelScore, choose

# Below this the kernel matrix is solved as if it were unregularised, which is the
# interpolating spline. Above zero but tiny, `solve` would be reporting conditioning, not
# a fit.
INTERPOLATION_LAM = 1e-12


def normalisation(points):
    """Origin and radius that put a point set inside the unit ball.

    The kernel is a length, so without this `lam` would mean one thing on a volume
    measured in 2.4 um voxels and something else on the same volume measured in 9.4 um
    ones, and no single value could be quoted for a pair.
    """
    points = np.asarray(points, float)
    centre = points.mean(0)
    scale = float(np.linalg.norm(points - centre, axis=1).max())
    return centre, (scale or 1.0)


def kernel(left, right):
    """U(r) = -r between two sets of points, the 3-D thin-plate Green's function.

    Negative because that is the sign under which the quadratic form is positive
    semi-definite on the subspace `P^T W = 0`, which is what makes `lam` a smoothing
    parameter rather than a roughening one.

    Through `cdist` rather than broadcasting, which would build an (m, n, 3) intermediate:
    resampling a 128^3 block against eighty control points is two million rows, and that
    array alone would be gigabytes before the distance is taken.
    """
    return -cdist(np.asarray(left, float), np.asarray(right, float))


def _system(control, lam, point_weights=None):
    """The (n+4) square matrix of the regularised spline, and the count it was built from.

    `point_weights`, if given, weigh the points' misfit against the smoothing: a point of
    weight w is fitted as if its noise were 1/sqrt(w) of the others'. None is all ones.
    """
    count = len(control)
    design = np.hstack([np.ones((count, 1)), control])
    matrix = np.zeros((count + 4, count + 4))
    matrix[:count, :count] = kernel(control, control)
    if lam > INTERPOLATION_LAM:
        ridge = (np.ones(count) if point_weights is None
                 else 1.0 / np.asarray(point_weights, float))
        matrix[:count, :count] += lam * np.diag(ridge)
    matrix[:count, count:] = design
    matrix[count:, :count] = design.T
    return matrix, design


class Deformable(NamedTuple):
    """A fitted bending transform, moving -> fixed, in the moving volume's own XYZ.

    `points` are the control points as they were given; `control` is the same points
    normalised, which is what the kernel is evaluated on. Both are kept because
    un-normalising is not exact in floating point, and what gets written to a file has to
    be the points the weights were actually solved against.
    """

    centre: np.ndarray
    scale: float
    points: np.ndarray
    control: np.ndarray
    weights: np.ndarray
    affine: np.ndarray
    lam: float
    edf: float
    # How much each point counted against the smoothing; None when all counted alike.
    point_weights: Optional[np.ndarray] = None

    def apply(self, points, batch: int = 100_000):
        """Map moving points to fixed ones.

        Batched because the kernel is dense: resampling a 128^3 block is two million
        points, and against eighty control points that single matrix would be 1.3 GB in
        float64. The batch size bounds it without changing the result, and is small enough
        that several workers can each hold one.
        """
        points = np.asarray(points, float)
        out = np.empty((len(points), 3))
        for start in range(0, len(points), batch):
            chunk = (points[start:start + batch] - self.centre) / self.scale
            design = np.hstack([np.ones((len(chunk), 1)), chunk])
            out[start:start + batch] = (kernel(chunk, self.control) @ self.weights
                                        + design @ self.affine)
        return out

    def matrix(self):
        """The affine part alone, as the 3x4 the schema stores, in unnormalised XYZ.

        This is what the map becomes far from every control point -- the kernel's weights
        sum to zero and are orthogonal to the coordinates, so its contribution decays as
        the point recedes. It is therefore the honest thing for a reader that ignores the
        deformation to use.
        """
        linear = (self.affine[1:] / self.scale).T
        offset = self.affine[0] - (self.centre / self.scale) @ self.affine[1:]
        return np.hstack([linear, offset[:, None]])

    def bend(self, points):
        """How far the deformation moves a point away from the affine part alone."""
        points = np.asarray(points, float)
        matrix = self.matrix()
        return self.apply(points) - (points @ matrix[:, :3].T + matrix[:, 3])


def fit_deformable(fixed, moving, lam, centre=None, scale=None,
                   with_edf: bool = True, point_weights=None) -> Deformable:
    """Fit a regularised spline from `moving` to `fixed`.

    `centre` and `scale` may be supplied so a cross-validation's folds all normalise the
    same way; leaving them out takes them from the points given, which is what a final fit
    wants. `with_edf` off skips the inverse the degrees of freedom need, which a
    cross-validation fitting thousands of folds does not want per fold.
    """
    fixed = np.asarray(fixed, float)
    moving = np.asarray(moving, float)
    if len(fixed) != len(moving):
        raise ValueError("fixed and moving must have the same number of points")
    if len(fixed) < 5:
        raise ValueError("a spline needs at least 5 correspondences to be determined")
    if centre is None or scale is None:
        centre, scale = normalisation(moving)
    control = (moving - centre) / scale

    matrix, _design = _system(control, lam, point_weights)
    target = np.vstack([fixed, np.zeros((4, 3))])
    solution = np.linalg.solve(matrix, target)
    return Deformable(
        centre=np.asarray(centre, float),
        scale=float(scale),
        points=moving,
        control=control,
        weights=solution[:len(control)],
        affine=solution[len(control):],
        lam=float(lam),
        edf=effective_dof(control, lam, point_weights) if with_edf else float("nan"),
        point_weights=None if point_weights is None else np.asarray(point_weights, float),
    )


def smoother(control, lam, point_weights=None) -> np.ndarray:
    """The matrix `S` with `fitted = S @ measured` at the control points.

    Its diagonal is each point's leverage: how much of its own measured displacement the
    fit follows. A point the fit follows almost entirely is one the bend is resting on.
    """
    control = np.asarray(control, float)
    count = len(control)
    matrix, design = _system(control, lam, point_weights)
    # The fitted value at a control point is the spline evaluated there, which uses the
    # bare kernel -- the ridge term belongs to the system being solved, not to the map.
    evaluate = np.hstack([kernel(control, control), design])
    return evaluate @ np.linalg.solve(matrix, np.eye(count + 4))[:, :count]


def effective_dof(control, lam, point_weights=None) -> float:
    """Trace of the smoother: 4 at the affine limit, one per point at interpolation.

    The trace of `S` is how many independent quantities the fit is really estimating. It
    is what stands in for a parameter count when the ladder is asked to prefer the simpler
    of two models that score alike.
    """
    return float(np.trace(smoother(control, lam, point_weights)))


def fit_affine(fixed, moving):
    """Least-squares affine, which is what `fit_deformable` becomes as `lam` grows.

    Kept here so the limit can be asserted against something rather than assumed, and so
    a caller comparing the two is comparing fits made the same way.
    """
    fixed = np.asarray(fixed, float)
    moving = np.asarray(moving, float)
    design = np.hstack([moving, np.ones((len(moving), 1))])
    coefficients = np.linalg.lstsq(design, fixed, rcond=None)[0]
    return coefficients[:3].T, coefficients[3]


def predict_deformable(train_fixed, train_moving, test_moving, lam,
                       centre=None, scale=None) -> Optional[np.ndarray]:
    """Where a spline fitted to the training pairs puts the test points.

    Returns None when the training set is too small to determine one, which a
    cross-validation that holds out whole regions runs into and must survive.
    """
    if len(train_fixed) < 5:
        return None
    try:
        return fit_deformable(train_fixed, train_moving, lam, centre, scale,
                              with_edf=False).apply(test_moving)
    except np.linalg.LinAlgError:
        return None


# --------------------------------------------------------------- the stored form

class Correction(NamedTuple):
    """An affine plus a bend, which is how a deformable transform is written to a file.

    The obvious way to store a spline would be to put its own affine part in the schema's
    `transformation_matrix`, so the whole map is one object. Measured, that is a trap: the
    kernel absorbs part of the global trend, and what is left over is a poor transform on
    its own. On the five pairs measured, the spline's own affine part predicts a block to
    13.7, 18.2 and 1.7 um where the ladder's affine predicts it to 6.6, 5.4 and 0.8 -- so
    a reader that ignored the deformation would be handed something twice as bad as what
    it gets today, with no way to know.

    So the split is the other way round. `matrix` is exactly the transform that would be
    written without any of this, and `spline` maps a moving point to the displacement to
    *add*, in fixed coordinates. Dropping the deformation therefore leaves today's file,
    unchanged to the last digit, which is the only backwards compatibility worth claiming.
    """

    matrix: np.ndarray
    spline: Deformable
    inverse: Optional[Deformable] = None

    def apply(self, points):
        points = np.asarray(points, float)
        return (points @ self.matrix[:, :3].T + self.matrix[:, 3]
                + self.spline.apply(points))

    def inverted(self) -> "Correction":
        """The fixed -> moving map, as a Correction in its own right.

        Raises if the inverse was never fitted, rather than quietly handing back the
        affine inverse alone: a caller asking to go backwards through a bent transform
        and silently getting the unbent one is the failure this is meant to prevent.
        """
        if self.inverse is None:
            raise ValueError("this correction carries no inverse")
        return Correction(invert_matrix(self.matrix), self.inverse, self.spline)

    def to_json(self) -> dict:
        """The `deformation` object of schema 1.2.0, with 1.3.0's `inverse` when fitted.

        Control points are written in the moving volume's own coordinates rather than
        normalised ones, so they can be read against `moving_landmarks` -- which is what
        they are -- instead of taken on trust. A reader normalises them itself with
        `centre` and `scale`.
        """
        # Nothing here is rounded, and the control points are the ones the fit was given
        # rather than the normalised ones un-normalised again. They have to be *exactly*
        # the points the weights were solved against, and the schema says they are the
        # moving landmarks, so anything else would both perturb the map and make that
        # claim false.
        record = {
            "model": "thin-plate-spline",
            "kernel": "-r",
            "centre": self.spline.centre.tolist(),
            "scale": self.spline.scale,
            "regularisation": self.spline.lam,
            "effective_parameters": 3 * self.spline.edf,
            "control_points": self.spline.points.tolist(),
            "weights": self.spline.weights.tolist(),
            "affine": self.spline.affine.tolist(),
        }
        if self.inverse is not None:
            record["inverse"] = {
                "centre": self.inverse.centre.tolist(),
                "scale": self.inverse.scale,
                "control_points": self.inverse.points.tolist(),
                "weights": self.inverse.weights.tolist(),
                "affine": self.inverse.affine.tolist(),
                "round_trip_max": self.round_trip_error(),
            }
        return record

    def round_trip_error(self) -> float:
        """Worst |F(F-1(p)) - p| over the region the correspondences cover, fixed voxels.

        Written into the file because the inverse is a separate fit rather than an
        algebraic inverse, so how nearly the two compose to the identity is a property of
        this particular pair and not something a reader can assume. Measured over a grid
        reaching half again beyond the control points, because a segment being transformed
        is not confined to the hull the blocks were sampled in.
        """
        if self.inverse is None:
            raise ValueError("this correction carries no inverse")
        forward = self.apply(self.spline.points)
        low, high = forward.min(0), forward.max(0)
        middle, half = (low + high) / 2.0, (high - low) / 2.0
        axis = np.linspace(-1.5, 1.5, 9)
        grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)
        probe = middle + grid * half
        back = self.inverted().apply(probe)
        return float(np.abs(self.apply(back) - probe).max())


def invert_matrix(matrix):
    """The 3x4 affine that undoes `matrix`."""
    matrix = np.asarray(matrix, float)
    linear = np.linalg.inv(matrix[:, :3])
    return np.hstack([linear, (-linear @ matrix[:, 3])[:, None]])


def fit_correction(fixed, moving, matrix, lam, centre=None, scale=None,
                   with_edf: bool = True, with_inverse: bool = False,
                   point_weights=None) -> Correction:
    """Fit the bend that `matrix` leaves behind.

    `matrix` is the 3x4 the file already stores, and the spline is fitted to what it gets
    wrong. As `lam` grows the spline collapses to the least-squares affine of that
    residual, which is bounded by the gap between `matrix` and a full affine12 -- so the
    smoothed-out end of the ladder is never worse than the affine, however `matrix` was
    chosen.
    """
    fixed = np.asarray(fixed, float)
    moving = np.asarray(moving, float)
    matrix = np.asarray(matrix, float)
    residual = fixed - (moving @ matrix[:, :3].T + matrix[:, 3])
    spline = fit_deformable(residual, moving, lam, centre, scale, with_edf, point_weights)
    if not with_inverse:
        return Correction(matrix, spline)
    # The inverse is the same fit run backwards -- which is what the schema used to tell a
    # reader to do for itself, now done once and written down. It is normalised on the
    # fixed points, because those are the coordinates it will be evaluated at. `lam` is
    # deliberately the *same*: both directions are then the same smoothing of the same
    # correspondences, which is why the two compose to the identity so closely. Fitting
    # the backward map to the forward one's exact numerical inverse instead -- a noiseless
    # function, which this `lam` is far too strong for -- was measured at 0.06 to 3.0 um
    # of round-trip error against 0.001 to 0.026 um for this.
    backward = invert_matrix(matrix)
    inverse_residual = moving - (fixed @ backward[:, :3].T + backward[:, 3])
    inverse_centre, inverse_scale = normalisation(fixed)
    return Correction(matrix, spline,
                      fit_deformable(inverse_residual, fixed, lam,
                                     inverse_centre, inverse_scale, with_edf=False,
                                     point_weights=point_weights))


def _deformable_from_json(record, lam, edf) -> Deformable:
    centre = np.asarray(record["centre"], float)
    scale = float(record["scale"])
    points = np.asarray(record["control_points"], float)
    return Deformable(centre=centre, scale=scale, points=points,
                      control=(points - centre) / scale,
                      weights=np.asarray(record["weights"], float),
                      affine=np.asarray(record["affine"], float),
                      lam=lam, edf=edf)


def correction_from_json(matrix, record) -> Correction:
    """Rebuild a stored correction, so a reader and the writer cannot drift apart."""
    if record.get("model") != "thin-plate-spline":
        raise ValueError(f"unknown deformation model {record.get('model')!r}")
    if record.get("kernel", "-r") != "-r":
        raise ValueError(f"unknown deformation kernel {record.get('kernel')!r}")
    lam = float(record["regularisation"])
    inverse = record.get("inverse")
    return Correction(
        np.asarray(matrix, float),
        _deformable_from_json(record, lam, float(record["effective_parameters"]) / 3.0),
        _deformable_from_json(inverse, lam, float("nan")) if inverse else None,
    )


# -------------------------------------------------------------- choosing the bend

# Smoothing levels the chooser scores, spanning interpolation to the affine limit.
LAMBDAS = tuple(10.0 ** power for power in np.arange(-6.0, 3.01, 0.5))
FOLDS = 8
SEED = 0
# Below this a fold is measuring conditioning rather than prediction.
MIN_TRAIN = 10
# Above this condition number of I - S_gg a held-out group is refitted rather than read
# off the full fit: the spline follows it so closely that the identity loses digits.
LEAVE_OUT_MAX_CONDITION = 1e8
# With the leverage capped, smoothing levels whose held-out error is within this share of
# the lowest count as tied, and the stiffest of them is taken.
TIED_SHARE = 0.005


def spatial_folds(points, folds=FOLDS, seed=SEED):
    """Partition points into contiguous groups, by Lloyd's algorithm on their positions.

    Contiguity is the whole point: a fold has to be a region the fit can be made blind
    to, not a scatter of points that each still have neighbours in the training set. A
    model free to bend would otherwise be rewarded for bending towards neighbours that
    are in its own training data, which is the one way of scoring it that cannot work.
    """
    points = np.asarray(points, float)
    rng = np.random.default_rng(seed)
    centres = points[rng.choice(len(points), folds, replace=False)]
    for _ in range(50):
        distances = np.linalg.norm(points[:, None, :] - centres[None, :, :], axis=2)
        labels = distances.argmin(1)
        moved = np.array([points[labels == k].mean(0) if (labels == k).any() else centres[k]
                          for k in range(folds)])
        if np.allclose(moved, centres):
            break
        centres = moved
    return labels


def fold_errors(fixed, moving, matrix, labels, lam, centre=None, scale=None,
                point_weights=None, smoothing=None):
    """Error at every point, from a model that never saw that point's region.

    `lam` of None is the affine, refitted on the training set exactly as every bend is.
    That fairness is not a detail and I had it wrong: the null used to be the incumbent
    `matrix` scored without refitting, on the reasoning that a deformation should have to
    beat something that had seen all the data before being written. Measured at blocks no
    fit ever saw, that reasoning cost PHerc0814 and PHerc0500P2 a bend worth 64% and 65%
    of their remaining error. Comparing a fitted model against a cross-validated one is
    the mistake cross-validation exists to prevent, whichever direction it errs in.

    `matrix` still sets what the bend is a correction *to*, which is what the schema
    stores, and makes no difference to the total map: an affine base lies inside the span
    the spline's own polynomial part already covers.

    Normalisation is taken from the whole set so that a fold cannot change what `lam`
    means by changing the coordinates it is measured in.

    With `smoothing`, the bend's `S` on all the points (`smoother` with the same `lam` and
    weights), no fold is refitted: the spline is a penalised least-squares fit, so what
    a fit without group g predicts at g follows from the fit on everything,

        f_(-g)(x_g) = (I - S_gg)^-1 (fitted_g - S_gg y_g),

    the leave-out identity of smoothing splines. One solve per smoothing level instead
    of one per group: refitting cost the patch hold-out groups x points^3, which on 560
    patches and 3400 octant points was hours a pass. A group where `I - S_gg` is near
    singular (the spline all but interpolates it) is refitted as before.
    """
    fixed = np.asarray(fixed, float)
    moving = np.asarray(moving, float)
    matrix = np.asarray(matrix, float)
    if centre is None or scale is None:
        centre, scale = normalisation(moving)
    errors = np.full(len(fixed), np.nan)
    fitted = None
    if smoothing is not None and lam is not None:
        residual = fixed - (moving @ matrix[:, :3].T + matrix[:, 3])
        fitted = smoothing @ residual
    for group in np.unique(labels):
        train, held = labels != group, labels == group
        if train.sum() < MIN_TRAIN:
            continue
        if fitted is not None:
            inside = np.flatnonzero(held)
            block = np.eye(len(inside)) - smoothing[np.ix_(inside, inside)]
            if np.linalg.cond(block) < LEAVE_OUT_MAX_CONDITION:
                left_out = np.linalg.solve(block, fitted[inside]
                                           - smoothing[np.ix_(inside, inside)] @ residual[inside])
                errors[held] = np.linalg.norm(left_out - residual[inside], axis=1)
                continue
        try:
            if lam is None:
                linear, offset = fit_affine(fixed[train], moving[train])
                predicted = moving[held] @ linear.T + offset
            else:
                predicted = fit_correction(
                    fixed[train], moving[train], matrix, lam, centre, scale, with_edf=False,
                    point_weights=None if point_weights is None
                    else np.asarray(point_weights)[train]).apply(moving[held])
        except np.linalg.LinAlgError:
            continue
        errors[held] = np.linalg.norm(predicted - fixed[held], axis=1)
    return errors


def fold_score(name, parameters, errors, labels) -> ModelScore:
    """Score a model with the spread taken across folds rather than across points.

    The ladder's standard error is the spread over held-out points, which assumes they
    are independent. Under a region hold-out they are not: every point in a fold is
    predicted by the same fit and sits in the same piece of the deformation field, so
    their errors move together. Counting them as independent understates the spread,
    narrows the one-standard-error band, and lets `choose` buy flexibility it cannot
    justify -- on PHerc0139 it bought 111 degrees of freedom where the fold-wise band
    stops at 67, for a tenth of a micron. Eight folds give eight independent numbers.
    """
    good = errors[np.isfinite(errors)]
    rms = float(np.sqrt(np.mean(good ** 2)))
    per_fold = []
    for group in np.unique(labels):
        inside = errors[(labels == group) & np.isfinite(errors)]
        if len(inside):
            per_fold.append(np.sqrt(np.mean(inside ** 2)))
    spread = (float(np.std(per_fold, ddof=1) / np.sqrt(len(per_fold)))
              if len(per_fold) > 1 else float("inf"))
    return ModelScore(name, parameters, rms, rms, spread)


def capped_weights(control, lam, groups, max_leverage, rounds=20):
    """Point weights under which no group pulls the fit more than `max_leverage` of the
    way to itself, lowering only the groups that would; None if that cannot be reached.

    A group's leverage is lowered by weighting it down -- treating it as noisier -- by the
    factor that would bring a lone point from its leverage to the cap; since lowering one
    group raises its neighbours' a little, this is repeated until every group is within
    a hair of the cap. Deterministic, so the weights a chosen smoothing level was scored
    with can be recomputed for the final fit.
    """
    groups = np.asarray(groups)
    weights = np.ones(len(control))
    members = [np.flatnonzero(groups == group) for group in np.unique(groups)]
    for _ in range(rounds):
        smoothing = smoother(control, lam, weights)
        worst = 0.0
        for inside in members:
            block = smoothing[np.ix_(inside, inside)]
            leverage = float(np.linalg.eigvalsh((block + block.T) / 2).max())
            worst = max(worst, leverage)
            if leverage > max_leverage:
                factor = max_leverage * (1 - leverage) / (leverage * (1 - max_leverage))
                weights[inside] *= max(factor, 1e-6)
        if worst <= max_leverage * 1.01:
            return weights
    return None


def choose_regularisation(fixed, moving, matrix, labels=None, lambdas=LAMBDAS,
                          max_leverage=None, groups=None):
    """How much to smooth, and whether to bend at all.

    Returns `(lam, edf, scores)`, with `lam` None when the affine alone is not beaten --
    which is the answer whenever the correspondences are too few or too scattered for a
    field to be visible, and has to be sayable.

    Both arms are refitted blind to each region in turn; see `fold_errors` for why that
    had to be corrected. With `max_leverage`, each smoothing level is scored with the
    points weighted by `capped_weights`, so that no one of `groups` (each point its own
    by default) pulls the fit more than that share of the way to itself: a bend that one
    group alone holds up is not one to write down, but a bend several groups agree on
    can be as sharp as they show it. With the cap the level is the one `lowest` picks;
    without it, the one-standard-error rule of `choose`. A level whose degrees of freedom have collapsed to an affine's
    is reported as no bend rather than as a bend of twelve parameters, because writing a
    correction that only refits the affine would change the transform under the guise of
    a deformation.
    """
    fixed = np.asarray(fixed, float)
    moving = np.asarray(moving, float)
    if labels is None:
        labels = spatial_folds(fixed)
    centre, scale = normalisation(moving)
    control = (moving - centre) / scale

    rows = [(fold_score("affine", 12,
                        fold_errors(fixed, moving, matrix, labels, None), labels),
             None, 4.0)]
    if groups is None:
        groups = np.arange(len(fixed))
    for lam in lambdas:  # noqa: B007 - `rows` is built in order, affine first
        weights = None
        if max_leverage is not None:
            weights = capped_weights(control, lam, groups, max_leverage)
            if weights is None:
                continue
        smoothing = smoother(control, lam, weights)
        errors = fold_errors(fixed, moving, matrix, labels, lam, centre, scale, weights,
                             smoothing)
        if np.isfinite(errors).sum() < MIN_TRAIN:
            continue
        edf = float(np.trace(smoothing))
        rows.append((fold_score(f"lam={lam:.0e}", int(round(3 * edf)), errors, labels),
                     lam, edf))

    # Too few points to hold any out leaves every score undefined: then there is no
    # evidence for a bend, rather than an error.
    scored = [row[0] for row in rows if np.isfinite(row[0].held_out_rms)]
    if max_leverage is None:
        best = choose(scored)
    else:
        best = lowest(scored)
    if best is None:
        return None, 4.0, rows
    for entry, lam, edf in rows:
        if entry.name != best.name:
            continue
        # A correction with an affine's degrees of freedom is not a bend.
        return (None, 4.0, rows) if round(3 * edf) <= 12 else (lam, edf, rows)
    return None, 4.0, rows


def lowest(scores: list) -> Optional[ModelScore]:
    """The stiffest score tied with the lowest held-out error (see `TIED_SHARE`).

    For a bend whose leverage is capped. The one-standard-error rule `choose` applies
    buys caution the cap already provides -- no group can hold up a bend alone -- and it
    stops on the slope of the error curve, where a change in the score of a few
    thousandths moves it a whole step of the grid: 82 to 150 degrees of freedom on
    PHerc1667, 14 to 86 on PHerc0343P, between passes that barely moved a patch. On the
    test sets of five pairs, patches no fit used, the most flexible capped bend predicted
    as well as or better than that choice (PHerc0139 1 -> 2 um: 2.15 against 2.31 um
    median). The error curve is flat where the cap holds, so the lowest is stable.
    """
    if not scores:
        return None
    floor = min(score.held_out_rms for score in scores)
    tied = [score for score in scores if score.held_out_rms <= floor * (1 + TIED_SHARE)]
    return min(tied, key=lambda score: score.parameters)


def fit_chosen_correction(fixed, moving, matrix, labels=None, max_leverage=None,
                          groups=None) -> Optional[Correction]:
    """The bend `matrix` leaves behind, smoothed as `choose_regularisation` decides.

    None when the affine is not beaten under the hold-out -- by region unless `labels`
    groups the points otherwise. The inverse is always fitted, because a correction that
    is written down will be resampled with.
    """
    lam, _edf, _scores = choose_regularisation(fixed, moving, matrix, labels,
                                               max_leverage=max_leverage, groups=groups)
    if lam is None:
        return None
    weights = None
    if max_leverage is not None:
        centre, scale = normalisation(moving)
        weights = capped_weights((np.asarray(moving, float) - centre) / scale, lam,
                                 np.arange(len(moving)) if groups is None else groups,
                                 max_leverage)
    return fit_correction(fixed, moving, matrix, lam, with_inverse=True,
                          point_weights=weights)
