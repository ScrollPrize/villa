"""The dinovol copy of SheetCompressionTransform must behave like the vesuvius one
(villa #1487): compress the gaps between sheets by the exact voxel amount and move
image, labels, padding mask and the paired second-resolution view together.

The phantom is a stack of 16 parallel sheets (thickness 3, period 10) along axis 0.
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from dinovol_2.augmentation.transforms.spatial.sheet_compression import SheetCompressionTransform

N_SHEETS, THICKNESS, PERIOD, FIRST = 16, 3, 10, 3
LENGTH = N_SHEETS * PERIOD
LATERAL = 8
STRENGTHS = (0.1, 0.2, 0.35)
SHAPE = (24, 20, 28)  # every axis different: a wrong (dim - 1) / dim factor cannot cancel


def phantom(scale=1, lateral=LATERAL):
    """Image with bright sheets, matching int16 sheet ids, all-valid mask.
    scale=2 is the same field of view at twice the resolution."""
    image = torch.zeros(1, LENGTH * scale, lateral * scale, lateral * scale)
    ids = torch.zeros(1, LENGTH * scale, lateral * scale, lateral * scale, dtype=torch.int16)
    for k in range(N_SHEETS):
        start = (FIRST + PERIOD * k) * scale
        image[:, start:start + THICKNESS * scale] = 1.0
        ids[:, start:start + THICKNESS * scale] = k + 1
    return image, ids, torch.ones_like(image)


def make(strength, axes=(0,), intensity_sigma=1.0):
    return SheetCompressionTransform(
        compression_strength=strength, intensity_sigma=intensity_sigma, spatial_smoothing=0.0,
        compression_axes=axes,
    )


def runs(profile, threshold):
    """(start, stop, weighted centroid) of every run where profile > threshold."""
    profile = np.asarray(profile, dtype=np.float64)
    on = profile > threshold
    found, i = [], 0
    while i < len(on):
        if not on[i]:
            i += 1
            continue
        j = i
        while j < len(on) and on[j]:
            j += 1
        weights = profile[i:j]
        found.append((i, j, float((np.arange(i, j) * weights).sum() / weights.sum())))
        i = j
    return found


def spacing_ratio(image_profile):
    centroids = [c for _, _, c in runs(image_profile, 0.5)]
    return float(np.diff(centroids).mean()) / PERIOD


def point_image(axis):
    """All zeros, one bright voxel at the end of `axis`: normalised intensity is 0
    (gap weight 1) everywhere else, so the voxel displacement is strength * (x + 1)."""
    image = torch.zeros(1, *SHAPE)
    at = [0, 0, 0]
    at[axis] = SHAPE[axis] - 1
    image[(0, *at)] = 1.0
    return image


class SheetCompressionTests(unittest.TestCase):
    def test_is_spatial_like_the_other_spatial_transforms(self):
        self.assertTrue(SheetCompressionTransform._is_spatial)

    def test_compression_shrinks_the_spacing_between_sheets(self):
        previous = None
        for strength in STRENGTHS:
            with self.subTest(strength=strength):
                image, ids, mask = phantom()
                out = make(strength)(image=image, segmentation=ids, padding_mask=mask)
                ratio = spacing_ratio(out["image"][0, :, 3, 3])
                self.assertLess(ratio, 1.0)
                # the expected spacing assumes the gaps (7 voxels of 10) shrink by about 1 / (1 + strength)
                expected = (THICKNESS + (PERIOD - THICKNESS) / (1 + strength)) / PERIOD
                self.assertAlmostEqual(ratio, expected, delta=0.015)
                if previous is not None:
                    self.assertLess(ratio, previous)
                previous = ratio

    def test_spatial_smoothing_leaves_a_uniform_stack_unchanged(self):
        # The default spatial_smoothing is (3, 8): it blurs the displacement across the
        # sheets. On a laterally uniform stack the result must not change. The blur pads by
        # reflection, so the stack is wider than its kernel.
        image, ids, mask = phantom(lateral=48)
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
        self.assertEqual(sorted(set(seg[seg > 0].tolist())), list(range(1, N_SHEETS + 1)))
        self.assertLess(spacing_ratio(smoothed["image"][0, :, 24, 24]), 1.0)
        self.assertTrue(torch.equal(smoothed["padding_mask"][0, :, 24, 24], plain["padding_mask"][0, :, 24, 24]))

    def test_all_sheets_survive_and_none_merge(self):
        for strength in STRENGTHS:
            with self.subTest(strength=strength):
                image, ids, mask = phantom()
                out = make(strength)(image=image, segmentation=ids, padding_mask=mask)
                seg_profile = out["segmentation"][0, :, 3, 3].numpy()
                self.assertEqual(sorted(set(seg_profile[seg_profile > 0].tolist())), list(range(1, N_SHEETS + 1)))
                found = runs(seg_profile > 0, 0.5)
                self.assertEqual(len(found), N_SHEETS)  # one run per sheet: nothing merged
                self.assertTrue(all(b[0] - a[1] >= 1 for a, b in zip(found, found[1:])))  # every gap keeps a voxel
                self.assertEqual(len(runs(out["image"][0, :, 3, 3], 0.5)), N_SHEETS)

    def test_labels_stay_aligned_with_the_image(self):
        for strength in STRENGTHS:
            for padded in (False, True):
                with self.subTest(strength=strength, padded=padded):
                    image, ids, mask = phantom()
                    if padded:
                        mask[:, 60:80] = 0.0  # padding over two sheets: must not blank them in the image only
                    out = make(strength)(image=image, segmentation=ids, padding_mask=mask)
                    image_centroids = [c for _, _, c in runs(out["image"][0, :, 3, 3], 0.5)]
                    seg_profile = out["segmentation"][0, :, 3, 3].numpy()
                    self.assertEqual(len(image_centroids), N_SHEETS)
                    for k, image_centroid in enumerate(image_centroids, start=1):
                        voxels = np.nonzero(seg_profile == k)[0]
                        self.assertGreaterEqual(voxels.size, THICKNESS - 1)
                        self.assertAlmostEqual(float(voxels.mean()), image_centroid, delta=0.5)

    def test_displacement_is_converted_from_voxels_to_grid_units_exactly(self):
        strength = 0.2
        for axis in range(3):
            with self.subTest(axis=axis):
                n = SHAPE[axis]
                params = make(strength, axes=(axis,), intensity_sigma=0.0).get_parameters(image=point_image(axis))
                self.assertEqual(params["chosen_axis"], axis)
                # align_corners=True: one voxel is 2 / (n - 1) grid units
                voxels = params["displacement"] * (n - 1) / 2
                line = torch.movedim(voxels, axis, 0)[:, 1, 1]
                expected = strength * (torch.arange(n, dtype=torch.float32) + 1)
                torch.testing.assert_close(line, expected, atol=1e-4, rtol=1e-5)

    def test_ramp_is_shifted_by_the_voxel_displacement_towards_lower_indices(self):
        strength = 0.2
        for axis in range(3):
            with self.subTest(axis=axis):
                n = SHAPE[axis]
                transform = make(strength, axes=(axis,), intensity_sigma=0.0)
                params = transform.get_parameters(image=point_image(axis))
                view = [1, 1, 1]
                view[axis] = n
                ramp = torch.arange(n, dtype=torch.float32).view(view).expand(1, *SHAPE).contiguous()
                out = transform._apply_compression(ramp, **params)
                lines = torch.movedim(out[0], axis, 0)
                x = torch.arange(n, dtype=torch.float32)
                # output[x] = input[x + d(x)], border-clamped: content moves to lower indices
                expected = torch.clamp(x + strength * (x + 1), max=n - 1)
                torch.testing.assert_close(lines[:, 1, 1], expected, atol=1e-3, rtol=0)
                self.assertTrue((lines[:, 1, 1] >= x).all())
                # nothing moves along the other axes (line [0, 0] holds the bright voxel)
                torch.testing.assert_close(lines[:, 1, 1], lines[:, 2, 3], atol=1e-4, rtol=0)

    def test_padding_mask_is_warped_with_the_same_grid(self):
        strength, start = 0.2, 15
        for axis in range(3):
            with self.subTest(axis=axis):
                n = SHAPE[axis]
                mask = torch.ones(1, *SHAPE)
                index = [slice(None)] * 4
                index[axis + 1] = slice(start, None)
                mask[tuple(index)] = 0.0
                out = make(strength, axes=(axis,), intensity_sigma=0.0)(
                    image=point_image(axis), padding_mask=mask
                )["padding_mask"]
                self.assertEqual(tuple(out.shape), (1, *SHAPE))
                self.assertTrue(set(out.unique().tolist()) <= {0.0, 1.0})
                x = torch.arange(n, dtype=torch.float32)
                source = x + strength * (x + 1)
                near_tie = (source - source.floor() - 0.5).abs() < 1e-3
                rounded = torch.round(source)
                expected = ((rounded <= n - 1) & (rounded < start)).float()
                line = torch.movedim(out[0], axis, 0)[:, 1, 1]
                self.assertTrue(torch.equal(line[~near_tie], expected[~near_tie]))
                self.assertEqual(float(line[-1]), 0.0)  # the tail pulled in from outside the volume is padding

    def test_mask_labels_and_image_agree_after_the_full_transform(self):
        image, _, _ = phantom()
        mask = torch.ones_like(image)
        mask[:, 60:80] = 0.0                                  # padding slab over two sheets
        invalid = (1 - mask).to(torch.int16)                  # a label that is 1 exactly on the padding
        out = make(0.35)(image=image, segmentation=invalid, padding_mask=mask)
        mask_out = out["padding_mask"][0, :, 3, 3]
        label_out = out["segmentation"][0, :, 3, 3]
        self.assertTrue(((label_out != 1) | (mask_out == 0)).all())  # label and mask moved together
        leftover = torch.nonzero((mask_out == 0) & (label_out == 0)).flatten()
        self.assertGreater(leftover.numel(), 0)
        self.assertGreater(int(leftover.min()), 80)
        self.assertTrue((mask_out[int(leftover.min()):] == 0).all())  # only the exposed tail, in one block
        self.assertLess(int((label_out == 1).sum()), 20)              # the slab itself got compressed

    def test_stale_padding_does_not_overwrite_warped_voxels(self):
        image, _, _ = phantom()
        mask = torch.ones_like(image)
        mask[:, 60:80] = 0.0
        transform = make(0.35)
        params = transform.get_parameters(image=image)
        out = transform.apply({"image": image.clone(), "padding_mask": mask.clone()}, **params)
        torch.testing.assert_close(out["image"], transform._apply_compression(image, **params))

    def test_boolean_padding_mask_stays_boolean(self):
        image, _, mask = phantom()
        out = make(0.2)(image=image, padding_mask=mask.bool())["padding_mask"]
        self.assertEqual(out.dtype, torch.bool)
        self.assertTrue(bool(out[0, 0, 0, 0]))
        self.assertFalse(bool(out[0, -1, 0, 0]))

    def test_uniform_image_leaves_the_mask_alone(self):
        image = torch.full((1, 20, 6, 6), 0.5)
        mask = torch.ones_like(image)
        mask[:, 10:] = 0.0
        out = make(0.2, intensity_sigma=0.0)(image=image.clone(), padding_mask=mask.clone())
        self.assertTrue(torch.equal(out["padding_mask"], mask))

    def _coarse_displacement_voxels(self, fine_voxels, axis, fine_shape, coarse_shape):
        """Displacement that _grid_for gives a coarse tensor, read back in coarse voxels."""
        n_fine, n_coarse = fine_shape[axis], coarse_shape[axis]
        params = {
            'displacement': fine_voxels * (2.0 / (n_fine - 1)),
            'spatial_shape': fine_shape,
            'chosen_axis': axis,
        }
        grid = make(0.35)._grid_for(torch.zeros((1, *coarse_shape)), params)[0]
        grid_idx = {0: 2, 1: 1, 2: 0}[axis]
        index = torch.arange(n_coarse, dtype=torch.float32)
        base = (-1.0 + 2.0 * index / (n_coarse - 1)).view([-1 if d == axis else 1 for d in range(3)])
        return (grid[..., grid_idx] - base) * ((n_coarse - 1) / 2.0)

    def test_paired_grid_scales_a_uniform_displacement_to_coarse_voxels(self):
        fine_shape, coarse_shape = (64, 40, 48), (32, 20, 24)
        for axis in range(3):
            with self.subTest(axis=axis):
                coarse = self._coarse_displacement_voxels(
                    torch.full(fine_shape, 3.0), axis, fine_shape, coarse_shape
                )
                expected = 3.0 * coarse_shape[axis] / fine_shape[axis]
                torch.testing.assert_close(coarse, torch.full(coarse_shape, expected), atol=1e-5, rtol=0)

    def test_paired_grid_places_a_ramp_on_edge_aligned_voxels(self):
        # The paired view is resized from the same crop with align_corners=False, so
        # coarse voxel j sits at fine coordinate (j + 0.5) * r - 0.5, with r = Nf / Nc.
        fine_shape, coarse_shape, alpha = (64, 40, 48), (32, 20, 24), 0.25
        for axis in range(3):
            with self.subTest(axis=axis):
                view = [-1 if d == axis else 1 for d in range(3)]
                x = torch.arange(fine_shape[axis], dtype=torch.float32).view(view)
                coarse = self._coarse_displacement_voxels(
                    (alpha * x).expand(fine_shape).contiguous(), axis, fine_shape, coarse_shape
                )
                r = fine_shape[axis] / coarse_shape[axis]
                j = torch.arange(coarse_shape[axis], dtype=torch.float32).view(view)
                expected = (alpha * ((j + 0.5) * r - 0.5) / r).expand(coarse_shape)
                torch.testing.assert_close(coarse, expected, atol=1e-5, rtol=0)

    def test_paired_view_is_warped_with_the_same_displacement(self):
        # dinovol's BasicTransform also warps `paired_spatial_image` (a second-resolution
        # view of the same field of view) for spatial transforms.
        image, ids, mask = phantom()
        fine, _, _ = phantom(scale=2)
        out = make(0.35)(image=image, segmentation=ids, padding_mask=mask, paired_spatial_image=fine)
        self.assertEqual(tuple(out["paired_spatial_image"].shape), tuple(fine.shape))
        coarse = [c for _, _, c in runs(out["image"][0, :, 3, 3], 0.5)]
        paired = [c / 2 for _, _, c in runs(out["paired_spatial_image"][0, :, 6, 6], 0.5)]
        self.assertEqual(len(paired), N_SHEETS)
        # same sheets at the same place (centroid conventions differ by half a voxel at 2x)
        for a, b in zip(coarse, paired):
            self.assertAlmostEqual(a, b - 0.25, delta=0.5)


if __name__ == "__main__":
    unittest.main()
