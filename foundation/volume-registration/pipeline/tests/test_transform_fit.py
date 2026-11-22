"""Regression tests for model selection, the sanity guards, and fit provenance.

These run against the transform files sitting in this directory (or wherever
VOLREG_LANDMARKS points), because the interesting cases are the real ones: a file whose
fit memorised its landmarks, a file with 60 degrees of shear, and twenty that are simply
fine. Nothing here needs network access.
"""

import glob
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from volreg.cores.transform_fit import (
    MODELS,
    MODELS_BY_NAME,
    available_models,
    describe_geometry,
    fit_affine12,
    fit_scaled_rigid9,
    fit_similarity7,
    held_out_errors,
    in_sample_rms,
    select_transform_from_points,
    selection_to_json,
)
from volreg.cores.transform_io import read_transform_json, write_transform_json

LANDMARK_DIR = Path(os.environ.get("VOLREG_LANDMARKS", Path(__file__).parent))

# Files whose stored fit is visibly broken: either the landmarks disagree with each other
# or the fit memorised them. The guards exist to catch exactly these.
KNOWN_BAD = {
    "PHercParis4-1um-to-2.4um-v1",
    "PHercParis4-2.4um137keVPOI-20230205180739-v3",
}


def landmark_sets():
    """Every usable landmark set on disk, as (name, fixed, moving)."""
    for path in sorted(glob.glob(str(LANDMARK_DIR / "*.json"))):
        try:
            data = json.load(open(path))
        except Exception:
            continue
        if not (isinstance(data, dict) and data.get("fixed_landmarks")):
            continue
        fixed = np.array(data["fixed_landmarks"], dtype=float)
        moving = np.array(data["moving_landmarks"], dtype=float)
        if len(fixed) != len(moving) or len(fixed) < 3:
            continue
        yield Path(path).stem, fixed, moving


@pytest.fixture(scope="module")
def sets():
    found = list(landmark_sets())
    if len(found) < 3:
        pytest.skip(f"need landmark sets in {LANDMARK_DIR}; set VOLREG_LANDMARKS")
    return found


def test_a_model_is_only_offered_once_holding_a_pair_out_still_determines_it():
    """The whole point: never choose a model on the strength of a fit it could not fail."""
    for spec in MODELS:
        n = spec.verifiable_pairs()
        assert 3 * (n - 1) > spec.parameters, f"{spec.name} is not over-determined at {n}"
        # The held-out fit must keep 4 points; 3 are always coplanar and leave the
        # linear part ambiguous under reflection however many equations they supply.
        assert n - 1 >= 4, f"{spec.name} would be chosen on a coplanar held-out fit"

    assert [s.name for s in available_models(4)] == []
    assert [s.name for s in available_models(5)] == ["scaled-rigid9", "similarity7"]
    assert [s.name for s in available_models(6)] == ["affine12", "scaled-rigid9", "similarity7"]


def test_a_held_out_fit_on_three_points_is_not_treated_as_verification(sets):
    """A fit that changes when its coordinates are reordered has not verified anything."""
    from volreg.cores.transform_fit import MODELS_BY_NAME, held_out_errors

    spec = MODELS_BY_NAME["similarity7"]
    rng = np.random.default_rng(0)
    worst = 1.0
    for _name, fixed, moving in sets:
        if len(fixed) < 6:
            continue
        for _ in range(20):
            index = rng.permutation(len(fixed))[:4]
            f, m = fixed[index], moving[index]
            a = float(np.sqrt(np.mean(held_out_errors(spec, f, m) ** 2)))
            b = float(np.sqrt(np.mean(held_out_errors(spec, f[:, ::-1], m[:, ::-1]) ** 2)))
            if min(a, b) > 0:
                worst = max(worst, max(a, b) / min(a, b))
    print(f"similarity7 on 3 held-out-fit points disagrees with itself by up to {worst:.0f}x")
    assert worst > 10, "expected the degeneracy this floor exists to avoid"
    assert spec.verifiable_pairs() > 4, "so it must not be offered at 4 pairs"


def test_a_four_pair_affine_reports_no_held_out_error_instead_of_a_flattering_zero(sets):
    """12 parameters through 4 pairs fits exactly; the tool used to report that as RMS 0.00."""
    _name, fixed, moving = max(sets, key=lambda s: len(s[1]))
    four_fixed, four_moving = fixed[:4], moving[:4]

    forced = select_transform_from_points(four_fixed, four_moving, mode="affine12")
    assert forced.model == "affine12"
    assert forced.in_sample_rms < 1e-6, "expected an exact fit, which is the problem"
    assert forced.held_out_rms is None, "an unverifiable fit must not report a held-out error"

    # Nothing is verifiable at 4 pairs, so automatic selection declines rather than guess.
    assert select_transform_from_points(four_fixed, four_moving) is None

    five_fixed, five_moving = fixed[:5], moving[:5]
    automatic = select_transform_from_points(five_fixed, five_moving)
    assert automatic.model in ("similarity7", "scaled-rigid9")
    assert automatic.held_out_rms is not None


def test_too_few_pairs_produce_no_automatic_fit(sets):
    _name, fixed, moving = max(sets, key=lambda s: len(s[1]))
    for n in (3, 4):
        assert select_transform_from_points(fixed[:n], moving[:n]) is None
    forced = select_transform_from_points(fixed[:3], moving[:3], mode="constrained5")
    assert forced is not None and forced.held_out_rms is None


def test_guards_refuse_the_broken_files_and_pass_the_rest(sets):
    refused = set()
    passed = set()
    for name, fixed, moving in sets:
        selection = select_transform_from_points(fixed, moving)
        if selection is None:
            continue
        (refused if selection.guard_failures else passed).add(name)

    print(f"{len(passed)} passed, {len(refused)} refused: {sorted(refused)}")
    available_bad = KNOWN_BAD & (refused | passed)
    assert available_bad <= refused, f"guards missed {available_bad - refused}"
    assert not (passed & KNOWN_BAD)
    assert len(passed) >= 15, "guards must not be refusing ordinary work"


def test_automatic_selection_is_never_much_worse_than_the_best_fixed_model(sets):
    """Auto should track whichever model happens to be right, not average them."""
    losses = []
    for _name, fixed, moving in sets:
        selection = select_transform_from_points(fixed, moving)
        if selection is None or selection.guard_failures:
            continue
        best = min(
            score.held_out_rms for score in selection.scores
        )
        losses.append(selection.held_out_rms / best)

    losses = np.array(losses)
    print(f"held-out error relative to the best available model: "
          f"median {np.median(losses):.3f}, max {losses.max():.3f}")
    assert np.median(losses) < 1.05
    assert losses.max() < 1.6


def test_auto_beats_always_forcing_the_legacy_constrained_model(sets):
    """The measurement that motivated replacing the constrained mode."""
    ratios = []
    for _name, fixed, moving in sets:
        selection = select_transform_from_points(fixed, moving)
        if selection is None or selection.guard_failures:
            continue
        spec = MODELS_BY_NAME["constrained5"]
        if len(fixed) < spec.verifiable_pairs():
            continue
        constrained = float(np.sqrt(np.mean(held_out_errors(spec, fixed, moving) ** 2)))
        ratios.append(constrained / selection.held_out_rms)

    ratios = np.array(ratios)
    print(f"constrained5 held-out error relative to auto, over {len(ratios)} sets: "
          f"median {np.median(ratios):.1f}x, best case {ratios.min():.1f}x")
    assert len(ratios) >= 10
    assert np.median(ratios) > 2.0


def test_similarity_fit_handles_a_reflected_transform(sets):
    """Several of these registrations include an axis flip; forcing a proper rotation broke them."""
    reflected = []
    for _name, fixed, moving in sets:
        if len(fixed) < 6:
            continue
        linear, _offset = fit_affine12(fixed, moving)
        if np.linalg.det(linear) < 0:
            reflected.append((fixed, moving))
    if not reflected:
        pytest.skip("no reflected transform among the available landmark sets")

    print(f"{len(reflected)} reflected landmark sets")
    for fixed, moving in reflected:
        linear, offset = fit_similarity7(fixed, moving)
        similarity = in_sample_rms(linear, offset, fixed, moving)
        linear, offset = fit_affine12(fixed, moving)
        affine = in_sample_rms(linear, offset, fixed, moving)
        assert similarity < 20 * affine + 10, "similarity7 collapsed on a reflected transform"


def test_geometry_description_matches_a_known_transform(sets):
    """Scale, anisotropy and shear must describe what the matrix actually does."""
    _name, fixed, moving = max(sets, key=lambda s: len(s[1]))
    linear, _offset = fit_scaled_rigid9(fixed, moving)
    geometry = describe_geometry(linear)
    singulars = np.linalg.svd(linear, compute_uv=False)
    assert geometry.scale == pytest.approx(float(np.cbrt(np.prod(singulars))))
    assert geometry.anisotropy >= 1.0
    assert 0.0 <= geometry.shear_deg < 90.0

    # A deliberate shear must be reported as one.
    sheared = linear @ np.array([[1.0, 0.3, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert describe_geometry(sheared).shear_deg > 15.0


def test_scale_guard_uses_the_voxel_sizes(sets):
    """The two voxel sizes predict the fitted scale, so a wrong one is caught for free."""
    _name, fixed, moving = max(sets, key=lambda s: len(s[1]))
    honest = select_transform_from_points(fixed, moving)
    nominal = honest.geometry.scale

    assert not select_transform_from_points(fixed, moving, nominal_scale=nominal).guard_failures
    wrong = select_transform_from_points(fixed, moving, nominal_scale=nominal * 2.0)
    assert any("scale" in failure for failure in wrong.guard_failures)


def test_handedness_guard_catches_a_mirrored_landmark_set(sets):
    """The atlas records handedness per volume, so the sign of the fit is predictable.

    Mirroring one side of a real landmark set is what a flipped axis in a viewer would do:
    the landmarks still fit a transform, the residual stays small, and every other guard
    passes. Only the sign of the determinant says anything is wrong.
    """
    _name, fixed, moving = max(sets, key=lambda s: len(s[1]))
    honest = select_transform_from_points(fixed, moving)
    reflected = honest.geometry.reflected

    assert not select_transform_from_points(
        fixed, moving, expect_reflected=reflected).guard_failures
    disagrees = select_transform_from_points(fixed, moving, expect_reflected=not reflected)
    assert any("handedness" in failure for failure in disagrees.guard_failures)

    # Mirroring one side flips the determinant, so the expectation that held before fails.
    mirrored = moving * np.array([1.0, 1.0, -1.0])
    flipped = select_transform_from_points(fixed, mirrored, expect_reflected=reflected)
    assert flipped.geometry.reflected != reflected
    assert any("handedness" in failure for failure in flipped.guard_failures)


def test_fit_record_round_trips_through_the_schema(sets):
    """Provenance has to survive being written and read back, or it is not provenance."""
    _name, fixed, moving = max(sets, key=lambda s: len(s[1]))
    selection = select_transform_from_points(fixed, moving)
    record = selection_to_json(selection)

    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "out.json")
        write_transform_json(path, "fixed_volume", selection.matrix, fixed, moving, fit=record)
        written = json.load(open(path))
        assert written["schema_version"] == "1.1.0"
        assert written["fit"]["model"] == selection.model
        assert written["fit"]["selection"] == "auto"
        assert written["fit"]["held_out_rms"] == pytest.approx(selection.held_out_rms, abs=1e-3)
        assert {c["model"] for c in written["fit"]["candidates"]} == {s.name for s in selection.scores}
        # read_transform_json validates against transform_schema.json.
        read_transform_json(path)

        without = os.path.join(directory, "plain.json")
        write_transform_json(without, "fixed_volume", selection.matrix, fixed, moving)
        assert json.load(open(without))["schema_version"] == "1.0.0"
        assert "fit" not in json.load(open(without))
        read_transform_json(without)


def test_existing_transform_files_still_load(sets):
    """Every file that loaded before the schema change must still load."""
    loaded = 0
    for path in sorted(glob.glob(str(LANDMARK_DIR / "*.json"))):
        try:
            data = json.load(open(path))
        except Exception:
            continue
        # The directory also holds the schema itself and batch manifests, which are not
        # transforms and are not expected to validate as one.
        if not (isinstance(data, dict) and "transformation_matrix" in data):
            continue
        try:
            read_transform_json(path)
            loaded += 1
        except ValueError as e:
            if "does not match schema" in str(e):
                raise AssertionError(f"{Path(path).name} no longer validates: {e}")
        except Exception:
            continue
    print(f"{loaded} transform files still validate")
    assert loaded >= 15


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
