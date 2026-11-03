"""SheetCompressionTransform must compress the gaps between sheets, by the exact
voxel amount, and move image, labels and padding mask together (villa #1487).

The phantom is a stack of 16 parallel sheets (thickness 3, period 10) along
axis 0, so the sheet spacing before and after is easy to measure.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from vesuvius.models.augmentation.transforms.spatial.sheet_compression import (
    SheetCompressionTransform,
)

N_SHEETS, THICKNESS, PERIOD, FIRST = 16, 3, 10, 3
LENGTH = N_SHEETS * PERIOD
LATERAL = 8
STRENGTHS = (0.1, 0.2, 0.35)


def _phantom(lateral=LATERAL):
    """Image (1, D, L, L) with bright sheets, matching int16 sheet ids, all-valid mask."""
    image = torch.zeros(1, LENGTH, lateral, lateral)
    ids = torch.zeros(1, LENGTH, lateral, lateral, dtype=torch.int16)
    for k in range(N_SHEETS):
        start = FIRST + PERIOD * k
        image[:, start:start + THICKNESS] = 1.0
        ids[:, start:start + THICKNESS] = k + 1
    return image, ids, torch.ones_like(image)


def _transform(strength, axes=(0,), intensity_sigma=1.0):
    return SheetCompressionTransform(
        compression_strength=strength, intensity_sigma=intensity_sigma, spatial_smoothing=0.0,
        compression_axes=axes,
    )


def _runs(profile, threshold):
    """(start, stop, weighted centroid) of every run where profile > threshold."""
    profile = np.asarray(profile, dtype=np.float64)
    on = profile > threshold
    runs, i = [], 0
    while i < len(on):
        if not on[i]:
            i += 1
            continue
        j = i
        while j < len(on) and on[j]:
            j += 1
        weights = profile[i:j]
        runs.append((i, j, float((np.arange(i, j) * weights).sum() / weights.sum())))
        i = j
    return runs


def _spacing_ratio(image_profile):
    centroids = [c for _, _, c in _runs(image_profile, 0.5)]
    return float(np.diff(centroids).mean()) / PERIOD


def _run(strength, **extra):
    image, ids, mask = _phantom()
    out = _transform(strength)(image=image, segmentation=ids, padding_mask=mask, **extra)
    return image, out


def test_is_spatial_like_the_other_spatial_transforms():
    assert SheetCompressionTransform._is_spatial is True


@pytest.mark.parametrize("strength", STRENGTHS)
def test_compression_shrinks_the_spacing_between_sheets(strength):
    _, out = _run(strength)
    ratio = _spacing_ratio(out["image"][0, :, 3, 3])
    assert ratio < 1.0
    # the expected spacing assumes the gaps (7 voxels of 10) shrink by about 1 / (1 + strength)
    expected = (THICKNESS + (PERIOD - THICKNESS) / (1 + strength)) / PERIOD
    assert ratio == pytest.approx(expected, abs=0.015)


def test_stronger_compression_gives_closer_sheets():
    ratios = [_spacing_ratio(_run(s)[1]["image"][0, :, 3, 3]) for s in STRENGTHS]
    assert ratios[0] > ratios[1] > ratios[2]



def test_spatial_smoothing_leaves_a_uniform_stack_unchanged():
    # The default spatial_smoothing is (3, 8): it blurs the displacement across the
    # sheets. On a laterally uniform stack the result must not change. The blur pads by
    # reflection, so the stack is wider than its kernel.
    image, ids, mask = _phantom(lateral=48)
    outs = {}
    for smoothing in (0.0, 5.0):
        transform = SheetCompressionTransform(
            compression_strength=0.35, intensity_sigma=1.0, spatial_smoothing=smoothing,
            compression_axes=(0,),
        )
        outs[smoothing] = transform(image=image.clone(), segmentation=ids.clone(), padding_mask=mask.clone())
    smoothed, plain = outs[5.0], outs[0.0]
    torch.testing.assert_close(smoothed["image"][0, :, 24, 24], plain["image"][0, :, 24, 24], atol=1e-4, rtol=0)
    seg = smoothed["segmentation"][0, :, 24, 24].numpy()
    assert sorted(set(seg[seg > 0].tolist())) == list(range(1, N_SHEETS + 1))
    assert _spacing_ratio(smoothed["image"][0, :, 24, 24]) < 1.0
    assert torch.equal(smoothed["padding_mask"][0, :, 24, 24], plain["padding_mask"][0, :, 24, 24])


@pytest.mark.parametrize("strength", STRENGTHS)
def test_all_sheets_survive_and_none_merge(strength):
    _, out = _run(strength)
    seg_profile = out["segmentation"][0, :, 3, 3].numpy()
    assert sorted(set(seg_profile[seg_profile > 0].tolist())) == list(range(1, N_SHEETS + 1))
    runs = _runs(seg_profile > 0, 0.5)
    assert len(runs) == N_SHEETS                              # one run per sheet: nothing merged
    assert all(b[0] - a[1] >= 1 for a, b in zip(runs, runs[1:]))  # every gap still has a voxel
    assert len(_runs(out["image"][0, :, 3, 3], 0.5)) == N_SHEETS


@pytest.mark.parametrize("padded", [False, True], ids=["no-padding", "padding-slab"])
@pytest.mark.parametrize("strength", STRENGTHS)
def test_labels_stay_aligned_with_the_image(strength, padded):
    image, ids, mask = _phantom()
    if padded:
        mask[:, 60:80] = 0.0  # padding over two sheets: must not blank them in the image only
    out = _transform(strength)(image=image, segmentation=ids, padding_mask=mask)
    image_centroids = [c for _, _, c in _runs(out["image"][0, :, 3, 3], 0.5)]
    seg_profile = out["segmentation"][0, :, 3, 3].numpy()
    assert len(image_centroids) == N_SHEETS
    for k, image_centroid in enumerate(image_centroids, start=1):
        voxels = np.nonzero(seg_profile == k)[0]
        assert voxels.size >= THICKNESS - 1
        assert voxels.mean() == pytest.approx(image_centroid, abs=0.5)


def _point_image(shape, axis):
    """All zeros, one bright voxel at the end of `axis`: normalised intensity is 0
    (gap weight 1) everywhere else, so the voxel displacement is strength * (x + 1)."""
    image = torch.zeros(1, *shape)
    at = [0, 0, 0]
    at[axis] = shape[axis] - 1
    image[(0, *at)] = 1.0
    return image


SHAPE = (24, 20, 28)  # every axis different: a wrong (dim - 1) / dim factor cannot cancel


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_displacement_is_converted_from_voxels_to_grid_units_exactly(axis):
    strength, n = 0.2, SHAPE[axis]
    transform = _transform(strength, axes=(axis,), intensity_sigma=0.0)
    params = transform.get_parameters(image=_point_image(SHAPE, axis))
    assert params["chosen_axis"] == axis
    # align_corners=True: one voxel is 2 / (n - 1) grid units
    voxels = params["displacement"] * (n - 1) / 2
    line = torch.movedim(voxels, axis, 0)[:, 1, 1]
    expected = strength * (torch.arange(n, dtype=torch.float32) + 1)
    torch.testing.assert_close(line, expected, atol=1e-4, rtol=1e-5)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_ramp_is_shifted_by_the_voxel_displacement_towards_lower_indices(axis):
    strength, n = 0.2, SHAPE[axis]
    transform = _transform(strength, axes=(axis,), intensity_sigma=0.0)
    params = transform.get_parameters(image=_point_image(SHAPE, axis))
    view = [1, 1, 1]
    view[axis] = n
    ramp = torch.arange(n, dtype=torch.float32).view(view).expand(1, *SHAPE).contiguous()
    out = transform._apply_compression(ramp, **params)
    line = torch.movedim(out[0], axis, 0)[:, 1, 1]
    x = torch.arange(n, dtype=torch.float32)
    # output[x] = input[x + d(x)], border-clamped: content moves to lower indices
    expected = torch.clamp(x + strength * (x + 1), max=n - 1)
    torch.testing.assert_close(line, expected, atol=1e-3, rtol=0)
    assert (line >= x).all()
    # nothing moves along the other axes: every other line shows the same ramp
    # (line [0, 0] holds the bright voxel and so has its own displacement)
    lines = torch.movedim(out[0], axis, 0)
    torch.testing.assert_close(lines[:, 1, 1], lines[:, 2, 3], atol=1e-4, rtol=0)


def _mask_with_invalid_from(axis, start):
    mask = torch.ones(1, *SHAPE)
    index = [slice(None)] * 4
    index[axis + 1] = slice(start, None)
    mask[tuple(index)] = 0.0
    return mask


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_padding_mask_is_warped_with_the_same_grid(axis):
    strength, n, start = 0.2, SHAPE[axis], 15
    transform = _transform(strength, axes=(axis,), intensity_sigma=0.0)
    image = _point_image(SHAPE, axis)
    out = transform(image=image, padding_mask=_mask_with_invalid_from(axis, start))["padding_mask"]
    assert out.shape == (1, *SHAPE)
    assert set(out.unique().tolist()) <= {0.0, 1.0}
    x = torch.arange(n, dtype=torch.float32)
    source = x + strength * (x + 1)
    near_tie = (source - source.floor() - 0.5).abs() < 1e-3
    rounded = torch.round(source)
    expected = ((rounded <= n - 1) & (rounded < start)).float()
    line = torch.movedim(out[0], axis, 0)[:, 1, 1]
    assert torch.equal(line[~near_tie], expected[~near_tie])
    assert line[-1] == 0  # the tail pulled in from outside the volume is padding


def test_mask_labels_and_image_agree_after_the_full_transform():
    image, _, _ = _phantom()
    mask = torch.ones_like(image)
    mask[:, 60:80] = 0.0                                  # padding slab over two sheets
    invalid = (1 - mask).to(torch.int16)                  # a label that is 1 exactly on the padding
    out = _transform(0.35)(image=image, segmentation=invalid, padding_mask=mask)
    mask_out = out["padding_mask"][0, :, 3, 3]
    label_out = out["segmentation"][0, :, 3, 3]
    assert ((label_out != 1) | (mask_out == 0)).all()     # label and mask moved together
    leftover = torch.nonzero((mask_out == 0) & (label_out == 0)).flatten()
    assert leftover.numel() > 0 and leftover.min() > 80
    assert (mask_out[leftover.min():] == 0).all()         # only the exposed tail, in one block
    assert (label_out == 1).sum() < 20                    # the slab itself got compressed


def test_stale_padding_does_not_overwrite_warped_voxels():
    image, _, _ = _phantom()
    mask = torch.ones_like(image)
    mask[:, 60:80] = 0.0
    transform = _transform(0.35)
    params = transform.get_parameters(image=image)
    out = transform.apply({"image": image.clone(), "padding_mask": mask.clone()}, **params)
    expected = transform._apply_compression(image, **params)
    torch.testing.assert_close(out["image"], expected)


def test_boolean_padding_mask_stays_boolean():
    image, _, mask = _phantom()
    out = _transform(0.2)(image=image, padding_mask=mask.bool())["padding_mask"]
    assert out.dtype == torch.bool
    assert out[0, 0, 0, 0] and not out[0, -1, 0, 0]


def test_uniform_image_leaves_the_mask_alone():
    image = torch.full((1, 20, 6, 6), 0.5)
    mask = torch.ones_like(image)
    mask[:, 10:] = 0.0
    out = _transform(0.2, intensity_sigma=0.0)(image=image.clone(), padding_mask=mask.clone())
    assert torch.equal(out["padding_mask"], mask)
