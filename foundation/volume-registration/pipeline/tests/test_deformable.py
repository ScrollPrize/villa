"""What a bending transform has to be true to.

The two ends of the regularisation are the whole argument for the model: at one end it
reproduces the correspondences, at the other it *is* the affine that ships today. If
either end drifts, the claim that the deformable model is a superset of the current one
stops being true, and there is no way to notice that by looking at a residual.

The rest is the trap this project keeps stepping in: a direction or an axis order silently
reversed, which leaves every summary statistic looking healthy. So the map is checked
against a bend planted on real landmark geometry, where the right answer is known.
"""

import json
import os
from pathlib import Path

import jsonschema
import numpy as np
import pytest

from volreg.cores.deformable import (
    Deformable,
    capped_weights,
    TIED_SHARE,
    choose_regularisation,
    correction_from_json,
    effective_dof,
    fit_affine,
    fit_correction,
    fit_deformable,
    normalisation,
    predict_deformable,
    smoother,
)
from volreg.cores.transform_io import points_swap_xyz_zyx, write_transform_json

LANDMARK_DIR = Path(os.environ.get("VOLREG_LANDMARKS", Path(__file__).resolve().parent))


def a_landmark_set():
    """The largest committed landmark set, for its real geometry rather than its values."""
    best = None
    for path in sorted(LANDMARK_DIR.glob("*.json")):
        try:
            data = json.load(open(path))
        except Exception:
            continue
        if not (isinstance(data, dict) and data.get("fixed_landmarks")):
            continue
        fixed = np.array(points_swap_xyz_zyx(data["fixed_landmarks"]), float)
        moving = np.array(points_swap_xyz_zyx(data["moving_landmarks"]), float)
        if len(fixed) == len(moving) and (best is None or len(fixed) > len(best[0])):
            best = (fixed, moving)
    return best


def synthetic_pairs(count=40, seed=0):
    """Points spread through a box, mapped by a known affine. No file needed."""
    rng = np.random.default_rng(seed)
    moving = rng.uniform(-500, 500, size=(count, 3))
    linear = np.array([[0.26, 0.01, -0.004],
                       [-0.012, 0.257, 0.006],
                       [0.003, -0.007, 0.259]])
    offset = np.array([120.0, -40.0, 33.0])
    return moving @ linear.T + offset, moving, linear, offset


def test_no_regularisation_reproduces_every_correspondence():
    fixed, moving, _linear, _offset = synthetic_pairs()
    rng = np.random.default_rng(1)
    bent = fixed + rng.normal(0, 3.0, fixed.shape)

    spline = fit_deformable(bent, moving, lam=0.0)

    assert np.allclose(spline.apply(moving), bent, atol=1e-6)


def test_heavy_regularisation_is_the_least_squares_affine():
    """The limit that makes this a superset of the shipped model rather than a rival."""
    fixed, moving, _linear, _offset = synthetic_pairs()
    rng = np.random.default_rng(2)
    bent = fixed + rng.normal(0, 3.0, fixed.shape)
    linear, offset = fit_affine(bent, moving)

    spline = fit_deformable(bent, moving, lam=1e9)

    assert np.allclose(spline.apply(moving), moving @ linear.T + offset, atol=1e-3)
    assert np.abs(spline.weights).max() < 1e-6


def test_the_stored_matrix_is_the_affine_the_spline_decays_to():
    """A reader that ignores the deformation must still get a usable transform."""
    fixed, moving, _linear, _offset = synthetic_pairs()
    rng = np.random.default_rng(3)
    spline = fit_deformable(fixed + rng.normal(0, 3.0, fixed.shape), moving, lam=1e-2)

    matrix = spline.matrix()
    # Far from every control point the kernel's contribution vanishes, because its
    # weights sum to zero and are orthogonal to the coordinates.
    far = moving.mean(0) + np.array([[1e7, 0, 0], [0, 1e7, 0], [0, 0, -1e7]])
    affine_only = far @ matrix[:, :3].T + matrix[:, 3]

    assert np.allclose(spline.apply(far), affine_only, rtol=0, atol=1e-3)
    assert np.abs(spline.bend(far)).max() < 1e-3


def test_degrees_of_freedom_run_from_the_affine_to_one_per_point():
    _fixed, moving, _linear, _offset = synthetic_pairs(count=30)
    centre, scale = normalisation(moving)
    control = (moving - centre) / scale

    assert effective_dof(control, lam=0.0) == pytest.approx(30, abs=1e-6)
    assert effective_dof(control, lam=1e9) == pytest.approx(4, abs=1e-3)
    ladder = [effective_dof(control, lam) for lam in (1e-4, 1e-3, 1e-2, 1e-1, 1.0)]
    assert ladder == sorted(ladder, reverse=True), "more smoothing must cost fewer degrees"


def test_a_planted_bend_is_recovered_where_an_affine_cannot_follow_it():
    """The claim under test, on the geometry the volumes actually have."""
    pairs = a_landmark_set()
    if pairs is None or len(pairs[0]) < 20:
        pytest.skip("needs a committed landmark set of 20+ pairs; set VOLREG_LANDMARKS")
    fixed, moving = pairs

    # A smooth second-order displacement, the shape a scroll relaxing between scans makes.
    from research.nonrigid import bend  # noqa: PLC0415 - research code, imported for reuse
    displaced = fixed + bend(fixed, amplitude=8.0)

    linear, offset = fit_affine(displaced, moving)
    affine_error = np.linalg.norm(moving @ linear.T + offset - displaced, axis=1)
    spline = fit_deformable(displaced, moving, lam=1e-3)
    spline_error = np.linalg.norm(spline.apply(moving) - displaced, axis=1)

    # The bend has to be one the affine genuinely cannot follow, or the comparison is
    # between two models that both fit and says nothing.
    assert np.median(affine_error) > 1.0
    assert np.median(spline_error) < 0.3 * np.median(affine_error)


def test_a_fit_that_cannot_be_determined_says_so_instead_of_guessing():
    fixed, moving, _linear, _offset = synthetic_pairs(count=40)

    assert predict_deformable(fixed[:3], moving[:3], moving[3:5], lam=1e-3) is None
    assert predict_deformable(fixed, moving, moving[:2], lam=1e-3) is not None


def test_too_few_points_to_hold_out_is_no_bend_rather_than_an_error():
    fixed, moving, linear, offset = synthetic_pairs(count=9)
    matrix = np.hstack([linear, offset[:, None]])
    lam, _edf, rows = choose_regularisation(fixed, moving, matrix)
    assert lam is None
    assert not any(np.isfinite(row[0].held_out_rms) for row in rows)


def test_the_map_runs_moving_to_fixed_and_not_the_other_way():
    """The direction error that leaves every statistic looking healthy."""
    fixed, moving, _linear, _offset = synthetic_pairs()
    spline = fit_deformable(fixed, moving, lam=1e-4)

    assert np.linalg.norm(spline.apply(moving) - fixed, axis=1).max() < 1e-3
    # The reverse is not merely worse, it is a different volume's coordinates entirely.
    assert np.linalg.norm(spline.apply(fixed) - moving, axis=1).min() > 1.0


def test_normalisation_does_not_change_what_is_fitted():
    """`lam` is comparable across pairs only if the coordinates are scaled away first."""
    fixed, moving, _linear, _offset = synthetic_pairs()
    rng = np.random.default_rng(5)
    bent = fixed + rng.normal(0, 3.0, fixed.shape)

    here = fit_deformable(bent, moving, lam=1e-3)
    # The same geometry expressed in voxels a tenth the size, with the fixed side
    # unchanged: a pure change of the moving volume's units.
    there = fit_deformable(bent, moving * 10.0, lam=1e-3)

    assert np.allclose(here.apply(moving), there.apply(moving * 10.0), atol=1e-6)
    assert here.edf == pytest.approx(there.edf, rel=1e-9)


def test_it_is_a_named_tuple_so_a_fit_can_be_stored_and_compared():
    fixed, moving, _linear, _offset = synthetic_pairs()
    spline = fit_deformable(fixed, moving, lam=1e-3)
    assert isinstance(spline, Deformable)
    assert spline.control.shape == (len(moving), 3)
    assert spline.weights.shape == (len(moving), 3)
    assert spline.affine.shape == (4, 3)


# ---------------------------------------------------------------- the stored form

def test_dropping_the_deformation_leaves_exactly_the_transform_that_ships_today():
    """The only backwards-compatibility claim worth making, so it is the one asserted."""
    fixed, moving, linear, offset = synthetic_pairs()
    rng = np.random.default_rng(6)
    bent = fixed + rng.normal(0, 4.0, fixed.shape)
    matrix = np.hstack([linear, offset[:, None]])

    correction = fit_correction(bent, moving, matrix, lam=1e-2)

    assert np.array_equal(correction.matrix, matrix)
    assert np.allclose(moving @ correction.matrix[:, :3].T + correction.matrix[:, 3],
                       moving @ linear.T + offset)


def test_the_correction_is_a_displacement_added_to_that_transform():
    fixed, moving, linear, offset = synthetic_pairs()
    rng = np.random.default_rng(7)
    bent = fixed + rng.normal(0, 4.0, fixed.shape)
    matrix = np.hstack([linear, offset[:, None]])

    correction = fit_correction(bent, moving, matrix, lam=1e-4)

    affine_error = np.linalg.norm(moving @ linear.T + offset - bent, axis=1)
    total_error = np.linalg.norm(correction.apply(moving) - bent, axis=1)
    assert np.median(total_error) < 0.1 * np.median(affine_error)


def test_smoothing_the_correction_away_never_leaves_it_worse_than_the_affine():
    """The far end of the ladder has to be safe whatever `matrix` was chosen by."""
    fixed, moving, linear, offset = synthetic_pairs()
    rng = np.random.default_rng(8)
    bent = fixed + rng.normal(0, 4.0, fixed.shape)
    # A deliberately mediocre starting transform, as a 7-parameter pick can be.
    matrix = np.hstack([linear * 1.002, (offset + 3.0)[:, None]])

    affine_error = np.linalg.norm(moving @ matrix[:, :3].T + matrix[:, 3] - bent, axis=1)
    smoothed = fit_correction(bent, moving, matrix, lam=1e9)
    smoothed_error = np.linalg.norm(smoothed.apply(moving) - bent, axis=1)

    assert np.sqrt(np.mean(smoothed_error ** 2)) <= np.sqrt(np.mean(affine_error ** 2))


def test_a_stored_correction_reads_back_as_the_same_map():
    """A reader and the writer drifting apart is the failure this file cannot see."""
    fixed, moving, linear, offset = synthetic_pairs()
    rng = np.random.default_rng(9)
    bent = fixed + rng.normal(0, 4.0, fixed.shape)
    matrix = np.hstack([linear, offset[:, None]])
    correction = fit_correction(bent, moving, matrix, lam=1e-3)

    record = json.loads(json.dumps(correction.to_json()))
    restored = correction_from_json(matrix, record)

    assert np.allclose(restored.apply(moving), correction.apply(moving), atol=1e-4)
    assert record["control_points"] == moving.tolist(), \
        "the control points are the moving landmarks exactly, not a rounded copy"


def test_an_unknown_deformation_model_is_refused_rather_than_ignored():
    fixed, moving, linear, offset = synthetic_pairs()
    matrix = np.hstack([linear, offset[:, None]])
    record = fit_correction(fixed, moving, matrix, lam=1e-3).to_json()
    record["model"] = "b-spline-ffd"

    with pytest.raises(ValueError, match="unknown deformation model"):
        correction_from_json(matrix, record)


def test_a_written_file_with_a_deformation_still_validates_against_the_schema(tmp_path):
    """`additionalProperties: false` makes a new field a breaking change until declared."""
    fixed, moving, linear, offset = synthetic_pairs()
    rng = np.random.default_rng(10)
    bent = fixed + rng.normal(0, 4.0, fixed.shape)
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = linear, offset
    correction = fit_correction(bent, moving, matrix[:3], lam=1e-3)

    path = tmp_path / "with_deformation.json"
    write_transform_json(str(path), "a.zarr", matrix, bent.tolist(), moving.tolist(),
                         fit={"method": "dense-correlation"},
                         deformation=correction.to_json())
    written = json.loads(path.read_text())
    schema = json.loads(
        (Path(__file__).resolve().parent.parent / "volreg" / "cores" / "transform_schema.json").read_text())

    jsonschema.validate(written, schema)
    assert written["schema_version"] == "1.2.0"

    # The same call without one is unchanged, down to the version it claims.
    plain = tmp_path / "without.json"
    write_transform_json(str(plain), "a.zarr", matrix, bent.tolist(), moving.tolist(),
                         fit={"method": "dense-correlation"})
    without = json.loads(plain.read_text())
    jsonschema.validate(without, schema)
    assert without["schema_version"] == "1.1.0"
    assert written["transformation_matrix"] == without["transformation_matrix"]


def test_a_file_round_trips_through_the_schema_to_the_same_map(tmp_path):
    fixed, moving, linear, offset = synthetic_pairs()
    rng = np.random.default_rng(11)
    bent = fixed + rng.normal(0, 4.0, fixed.shape)
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = linear, offset
    correction = fit_correction(bent, moving, matrix[:3], lam=1e-3)

    path = tmp_path / "round_trip.json"
    write_transform_json(str(path), "a.zarr", matrix, bent.tolist(), moving.tolist(),
                         deformation=correction.to_json())
    written = json.loads(path.read_text())
    restored = correction_from_json(np.array(written["transformation_matrix"]),
                                    written["deformation"])

    assert np.allclose(restored.apply(moving), correction.apply(moving), atol=1e-4)


def test_batching_the_kernel_does_not_change_where_a_point_lands():
    """Resampling feeds two million points through this; the batch size must not show."""
    fixed, moving, _linear, _offset = synthetic_pairs()
    spline = fit_deformable(fixed, moving, lam=1e-3)
    rng = np.random.default_rng(12)
    lattice = rng.uniform(-600, 600, size=(5000, 3))

    whole = spline.apply(lattice, batch=len(lattice) + 1)
    pieces = spline.apply(lattice, batch=97)

    assert np.array_equal(whole, pieces)


# ------------------------------------------------------------------ the stored inverse

def test_fitting_an_inverse_leaves_the_forward_map_untouched():
    """1.3.0 must only ever *add* to a 1.2.0 file, or the backfill rewrites transforms."""
    fixed, moving, linear, offset = synthetic_pairs()
    rng = np.random.default_rng(20)
    bent = fixed + rng.normal(0, 4.0, fixed.shape)
    matrix = np.hstack([linear, offset[:, None]])

    plain = fit_correction(bent, moving, matrix, lam=1e-2)
    with_inverse = fit_correction(bent, moving, matrix, lam=1e-2, with_inverse=True)

    assert np.array_equal(plain.spline.weights, with_inverse.spline.weights)
    assert np.array_equal(plain.spline.affine, with_inverse.spline.affine)
    assert plain.inverse is None and with_inverse.inverse is not None
    assert plain.to_json() == {k: v for k, v in with_inverse.to_json().items()
                               if k != "inverse"}


def test_the_stored_inverse_undoes_the_forward_map():
    """The claim the file makes by carrying an inverse at all, on real geometry."""
    pairs = a_landmark_set()
    if pairs is None or len(pairs[0]) < 20:
        pytest.skip("needs a committed landmark set of 20+ pairs; set VOLREG_LANDMARKS")
    fixed, moving = pairs
    from research.nonrigid import bend  # noqa: PLC0415 - research code, imported for reuse
    displaced = fixed + bend(fixed, amplitude=8.0)
    linear, offset = fit_affine(displaced, moving)
    matrix = np.hstack([linear, offset[:, None]])

    correction = fit_correction(displaced, moving, matrix, lam=1e-3, with_inverse=True)

    # Probe beyond the hull the correspondences cover: a segment being transformed is not
    # confined to it, and an inverse that only held inside would be a trap.
    forward = correction.apply(moving)
    centre, half = (forward.max(0) + forward.min(0)) / 2, (forward.max(0) - forward.min(0)) / 2
    rng = np.random.default_rng(21)
    probe = centre + (rng.random((400, 3)) * 2 - 1) * half * 1.5
    recovered = correction.apply(correction.inverted().apply(probe))

    moved = np.linalg.norm(correction.spline.apply(moving), axis=1).max()
    assert np.abs(recovered - probe).max() < 1e-2 * moved, "the inverse must undo the bend"
    assert correction.round_trip_error() < 1e-2 * moved
    # And it is a real inverse, not the affine one wearing its name.
    affine_only = probe @ np.linalg.inv(matrix[:, :3]).T - offset @ np.linalg.inv(matrix[:, :3]).T
    assert np.abs(correction.inverted().apply(probe) - affine_only).max() > 1e-6


def test_the_inverse_survives_the_file(tmp_path):
    fixed, moving, linear, offset = synthetic_pairs()
    rng = np.random.default_rng(22)
    bent = fixed + rng.normal(0, 4.0, fixed.shape)
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = linear, offset
    correction = fit_correction(bent, moving, matrix[:3], lam=1e-3, with_inverse=True)

    path = tmp_path / "inverse.json"
    write_transform_json(str(path), "a.zarr", matrix, bent.tolist(), moving.tolist(),
                         deformation=correction.to_json())
    written = json.loads(path.read_text())
    schema = json.loads(
        (Path(__file__).resolve().parent.parent / "volreg" / "cores" / "transform_schema.json").read_text())
    jsonschema.validate(written, schema)
    assert written["schema_version"] == "1.3.0"
    assert written["deformation"]["inverse"]["control_points"] == written["fixed_landmarks"]

    restored = correction_from_json(np.array(written["transformation_matrix"]),
                                    written["deformation"])
    assert np.allclose(restored.inverted().apply(bent),
                       correction.inverted().apply(bent), atol=1e-9)


def test_a_correction_without_an_inverse_says_so_rather_than_guessing():
    fixed, moving, linear, offset = synthetic_pairs()
    correction = fit_correction(fixed, moving, np.hstack([linear, offset[:, None]]),
                                lam=1e-3)
    with pytest.raises(ValueError, match="no inverse"):
        correction.inverted()
    with pytest.raises(ValueError, match="no inverse"):
        correction.round_trip_error()


def test_capping_weighs_down_only_the_points_that_pull_too_hard():
    rng = np.random.default_rng(11)
    # A dense cloud and two lone points far out, which an uncapped bend would follow.
    control = np.vstack([rng.uniform(-0.5, 0.5, size=(60, 3)),
                         [[1.0, 1.0, 1.0], [-1.0, 1.0, -1.0]]])
    lam = 1e-2
    before = np.diag(smoother(control, lam))
    assert before[-2:].min() > 0.5
    weights = capped_weights(control, lam, np.arange(len(control)), 0.5)
    after = np.diag(smoother(control, lam, weights))
    assert after.max() <= 0.5 * 1.01
    over = before > 0.5
    assert (weights[~over] == 1.0).all() and (weights[over] < 1.0).all()


def test_a_capped_bend_takes_the_stiffest_of_the_levels_tied_lowest():
    """With leverage capped, the stiffness is the lowest held-out error, not the edge of a
    one-standard-error band, so a tiny change in the data cannot move it a grid step."""
    rng = np.random.default_rng(4)
    fixed, moving, linear, offset = synthetic_pairs(count=120, seed=4)
    fixed = fixed + 2.0 * np.sin(moving / 300.0) + rng.normal(scale=0.3, size=fixed.shape)
    matrix = np.hstack([linear, offset[:, None]])
    groups = np.arange(len(fixed))
    lam, _edf, rows = choose_regularisation(fixed, moving, matrix, groups, max_leverage=0.5,
                                            groups=groups)
    assert lam is not None, "the planted field needs a bend"
    scores = [row[0] for row in rows]
    floor = min(score.held_out_rms for score in scores)
    chosen = next(row[0] for row in rows if row[1] == lam)
    assert chosen.held_out_rms <= floor * (1 + TIED_SHARE)
    assert not any(score.parameters < chosen.parameters
                   and score.held_out_rms <= floor * (1 + TIED_SHARE) for score in scores)
    # Nudging every point by a hundredth of the noise leaves the choice where it was.
    nudged = fixed + rng.normal(scale=0.003, size=fixed.shape)
    again, _edf, _rows = choose_regularisation(nudged, moving, matrix, groups,
                                               max_leverage=0.5, groups=groups)
    assert again == lam
