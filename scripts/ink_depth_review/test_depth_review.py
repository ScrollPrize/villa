import os
from pathlib import Path

import numpy as np
import pytest
import tifffile
import zarr
from depth_review import export_review, fingerprint, load_volume, suggest

SAMPLE = Path(
    os.environ.get(
        "INK_REVIEW_SAMPLE", Path(__file__).parent / "sample"
    )
)


@pytest.fixture
def real():
    if not (SAMPLE / "image.tif").exists():
        pytest.skip("Run fetch_sample.py into work/sample first")
    return load_volume(SAMPLE / "image.tif"), load_volume(SAMPLE / "prior.tif")


def test_real_proposals_are_bounded_and_deterministic(real):
    image, prior = real
    a, r = suggest(image, prior)
    b, _ = suggest(image, prior)
    assert np.array_equal(a, b)
    assert r["prior_active_layers"] == [32]
    assert r["confirmed_ink_voxels"] == 0
    assert not a[:24].any() and not a[41:].any()
    assert not a[:, ~prior.any(axis=0)].any()
    assert r["candidate_voxels"] == 42662
    assert r["search_support_voxels"] == 558688


def review_for(image, voxels):
    return {
        "schema": "ink-depth-review-v1",
        "image_sha256": fingerprint(image),
        "shape": list(image.shape),
        "coordinate_system": "surface_dyx",
        "reviewer": "AUTOMATED TEST: not scientific ground truth",
        "voxels": voxels,
    }


def test_real_export_keeps_unknown_and_preserves_intensities(real, tmp_path):
    image, _ = real
    # Exercise edits on two depths of a real CT. These are format tests, not ink assertions.
    i = int(np.ravel_multi_index((25, 79, 137), image.shape))
    j = int(np.ravel_multi_index((34, 100, 80), image.shape))
    out = tmp_path / "reviewed"
    counts = export_review(image, review_for(image, [[i, 1], [j, 2]]), out)
    assert counts == {"ink": 1, "background": 1, "unknown": image.size - 2}
    assert np.array_equal(load_volume(out / "image.tif"), image)
    labels = load_volume(out / "inklabels.tif").ravel()
    supervision = load_volume(out / "supervision_mask.tif").ravel()
    assert labels[i] == 1 and labels[j] == 0 and supervision[i] == supervision[j] == 1
    assert supervision.sum() == 2
    with pytest.raises(FileExistsError):
        export_review(image, review_for(image, [[i, 1]]), out)


@pytest.mark.parametrize(
    "voxels", [[], [[0, 1], [0, 2]], [[-1, 1]], [[0, 255]], [[0.1, 1]], [[True, 1]]]
)
def test_reject_invalid_reviews(real, tmp_path, voxels):
    image, _ = real
    with pytest.raises(ValueError):
        export_review(image, review_for(image, voxels), tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_reject_wrong_image(real, tmp_path):
    image, _ = real
    review = review_for(image, [[0, 1]])
    review["image_sha256"] = "wrong"
    with pytest.raises(ValueError):
        export_review(image, review, tmp_path / "bad")


def test_reject_bad_inputs(real):
    image, prior = real
    with pytest.raises(ValueError):
        suggest(image, prior[:2])
    with pytest.raises(ValueError):
        suggest(image, prior, threshold=float("nan"))
    with pytest.raises(ValueError):
        suggest(image, prior, center=1000)
    bad = prior.copy()
    bad[0, 0, 0] = 4
    with pytest.raises(ValueError):
        suggest(image, bad)


def test_zarr_input_matches_tiff(real, tmp_path):
    image, _ = real
    zarr.save(str(tmp_path / "image.zarr"), image)
    assert np.array_equal(load_volume(tmp_path / "image.zarr"), image)


def test_rgb_rejected(tmp_path):
    tifffile.imwrite(
        tmp_path / "rgb.tif", np.zeros((16, 16, 3), np.uint8), photometric="rgb"
    )
    with pytest.raises(ValueError):
        load_volume(tmp_path / "rgb.tif")
