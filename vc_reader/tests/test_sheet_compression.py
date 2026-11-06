"""SheetCompression pulls parallel sheets together along an axis, in voxels, with all targets aligned.

The phantom is binary (sheets 1, gaps 0), so with intensity_sigma=0 the compression weight of a voxel is 1 in a
gap and 0 in a sheet and the sampling coordinate has a closed form: x + strength * (number of gap voxels in 0..x).
"""
import unittest

import numpy as np

from build_support import load_build
load_build()

from vc_reader import transforms as A

SHEETS, THICKNESS, PERIOD, FIRST = 16, 3, 10, 3
LENGTH = SHEETS * PERIOD


def phantom(lateral=8):
    image = np.zeros((1, LENGTH, lateral, lateral), np.float32)
    ids = np.zeros(image.shape, np.int16)
    for k in range(SHEETS):
        start = FIRST + PERIOD * k
        image[:, start:start + THICKNESS] = 1
        ids[:, start:start + THICKNESS] = k + 1
    return image, ids


def source_coordinate(image, strength):
    """Closed-form sampling coordinate along the axis: x + strength * (gap voxels in 0..x)."""
    gaps = np.cumsum(1 - image[0, :, 0, 0].astype(np.float64))
    return np.arange(len(gaps)) + strength * gaps


def centroids(profile, threshold=.5):
    """Intensity-weighted centroids of the runs above the threshold."""
    on, found, i = profile > threshold, [], 0
    while i < len(on):
        if not on[i]:
            i += 1
            continue
        j = i
        while j < len(on) and on[j]:
            j += 1
        found.append(float((np.arange(i, j) * profile[i:j]).sum() / profile[i:j].sum()))
        i = j
    return found


class SheetCompressionTests(unittest.TestCase):
    def backends(self):
        return ('cpu', 'cuda') if A._transforms.cuda_available() else ('cpu',)

    def run_transform(self, strength, backend, smoothing=0, padding=None, lateral=8, threshold=None, **extra):
        image, ids = phantom(lateral)
        ramp = np.broadcast_to(np.arange(LENGTH, dtype=np.float32).reshape(1, LENGTH, 1, 1), image.shape).copy()
        valid = np.ones_like(image)
        if padding is not None:
            valid[:, padding[0]:padding[1]] = 0
        options = {} if threshold is None else {'threshold': threshold}
        pipeline = A.Pipeline([A.SheetCompression(strength=strength, intensity_sigma=0, smoothing=smoothing, axes='z', **options)],
                              targets={'image': A.Image(), 'seg': A.Label(), 'ramp': A.ScalarField(),
                                       'padding_mask': A.ValidMask()}, backend=backend)
        sample = {'image': image, 'seg': ids, 'ramp': ramp, 'padding_mask': valid, **extra}
        return image, ids, pipeline(sample, seed=0)

    def test_sampling_coordinate_is_in_voxels_and_moves_forward(self):
        # The output at x samples the input beyond x: sheets move toward lower indices.
        for backend in self.backends():
            for strength in (.1, .2, .35):
                with self.subTest(backend=backend, strength=strength):
                    image, _, out = self.run_transform(strength, backend)
                    expected = np.minimum(source_coordinate(image, strength), LENGTH - 1)
                    ramp = out['ramp'][0, :, 4, 4]
                    np.testing.assert_allclose(ramp, expected, atol=1e-3)
                    self.assertTrue((ramp >= np.arange(LENGTH) - 1e-3).all())
                    self.assertGreater(ramp[LENGTH // 2], LENGTH // 2 + 1)

    def test_every_axis_gives_the_same_closed_form_in_3d_and_2d(self):
        # Sheets are stacked along `axes`; the compression must act on that axis only, whatever its position.
        profile = phantom()[0][0, :, 0, 0]
        for backend in self.backends():
            for dims, axes, position in ((3, 'z', 0), (3, 'y', 1), (3, 'x', 2), (2, 'y', 0), (2, 'x', 1)):
                with self.subTest(backend=backend, dims=dims, axes=axes):
                    shape = [1] + [LENGTH if a == position else 8 for a in range(dims)]
                    along = lambda values: np.broadcast_to(values.reshape(
                        [1] + [LENGTH if a == position else 1 for a in range(dims)]), shape).copy()
                    image, ramp = along(profile), along(np.arange(LENGTH, dtype=np.float32))
                    pipeline = A.Pipeline([A.SheetCompression(strength=.35, intensity_sigma=0, smoothing=0, axes=axes)],
                                          targets={'image': A.Image(), 'ramp': A.ScalarField()}, backend=backend)
                    out = pipeline({'image': image, 'ramp': ramp}, seed=0)
                    expected = np.minimum(source_coordinate(phantom()[0], .35), LENGTH - 1)
                    columns = np.moveaxis(out['ramp'], 1 + position, 1).reshape(LENGTH, -1)
                    np.testing.assert_allclose(columns, np.repeat(expected[:, None], columns.shape[1], 1), atol=1e-3)

    def test_threshold_counts_the_gaps_below_it(self):
        # With a threshold the weight is 1 below it and 0 above it, which on the binary phantom equals the gap weight.
        for backend in self.backends():
            for strength in (.1, .35):
                with self.subTest(backend=backend, strength=strength):
                    image, _, out = self.run_transform(strength, backend, threshold=.4)
                    expected = np.minimum(source_coordinate(image, strength), LENGTH - 1)
                    np.testing.assert_allclose(out['ramp'][0, :, 4, 4], expected, atol=1e-3)

    def test_sheets_move_together_by_the_expected_ratio_and_all_are_kept(self):
        for backend in self.backends():
            for strength in (.1, .2, .35):
                with self.subTest(backend=backend, strength=strength):
                    _, _, out = self.run_transform(strength, backend)
                    image, labels = out['image'][0, :, 4, 4], out['seg'][0, :, 4, 4]
                    spacing = np.diff(centroids(image)).mean() / PERIOD
                    # The gap (7 of every 10 voxels) shrinks by 1 / (1 + strength); sheets keep their thickness.
                    # Interpolated sheet edges move the centroids slightly, hence the tolerance.
                    self.assertAlmostEqual(spacing, (THICKNESS + (PERIOD - THICKNESS) / (1 + strength)) / PERIOD, delta=.02)
                    self.assertLess(spacing, 1)
                    self.assertEqual(len(centroids(image)), SHEETS)
                    self.assertEqual(sorted(set(labels[labels > 0].tolist())), list(range(1, SHEETS + 1)))
                    self.assertEqual(int(((labels[:-1] > 0) & (labels[1:] > 0) & (labels[:-1] != labels[1:])).sum()), 0)

    def test_image_and_labels_stay_aligned_with_padding(self):
        for backend in self.backends():
            for strength in (.1, .35):
                with self.subTest(backend=backend, strength=strength):
                    _, _, out = self.run_transform(strength, backend, padding=(60, 80))
                    image, labels, mask = out['image'][0, :, 4, 4], out['seg'][0, :, 4, 4], out['padding_mask'][0, :, 4, 4]
                    label_centers = [float(np.nonzero(labels == k)[0].mean()) for k in range(1, SHEETS + 1)]
                    image_centers = centroids(image)
                    # Padding blanks the image sheets that were inside it (and only those); the others sit on
                    # their labelled sheet to within half a voxel plus rounding (labels are nearest-neighbour, the image linear). A sheet cut by
                    # the edge of the padding has a shifted centroid and is not compared.
                    self.assertEqual(len(image_centers), SHEETS - 2)
                    for center in image_centers:
                        if (mask[int(center) - 3:int(center) + 4] == 0).any():
                            continue
                        self.assertLess(min(abs(center - other) for other in label_centers), .6)

    def test_tail_pulled_in_from_outside_the_volume_is_marked_invalid(self):
        for backend in self.backends():
            for strength in (.1, .2, .35):
                with self.subTest(backend=backend, strength=strength):
                    image, _, out = self.run_transform(strength, backend)
                    mask, picture = out['padding_mask'][0, :, 4, 4], out['image'][0, :, 4, 4]
                    outside = source_coordinate(image, strength) > LENGTH - 1
                    self.assertGreater(outside.sum(), 5)
                    self.assertLessEqual(abs(int(mask.size - mask.sum()) - int(outside.sum())), 1)
                    self.assertTrue((mask[:10] == 1).all() and (mask[-3:] == 0).all())
                    np.testing.assert_array_equal(picture[mask == 0], 0)

    def test_padding_moves_with_the_image(self):
        for backend in self.backends():
            with self.subTest(backend=backend):
                image, _, out = self.run_transform(.35, backend, padding=(60, 80))
                mask = out['padding_mask'][0, :, 4, 4]
                coordinate = source_coordinate(image, .35)
                inside = (coordinate > 60) & (coordinate < 79)
                self.assertTrue((mask[inside] == 0).all())
                self.assertTrue((mask[coordinate < 59] == 1).all())

    def test_smoothing_leaves_a_laterally_uniform_volume_unchanged(self):
        for backend in self.backends():
            with self.subTest(backend=backend):
                _, _, plain = self.run_transform(.35, backend, smoothing=0, lateral=16)
                _, _, smooth = self.run_transform(.35, backend, smoothing=2, lateral=16)
                for key in ('image', 'ramp', 'padding_mask'):
                    np.testing.assert_allclose(smooth[key], plain[key], atol=1e-3)
                # Labels may differ where a rounding error moves a coordinate across a nearest-neighbour tie.
                self.assertLess((smooth['seg'] != plain['seg']).mean(), .01)

    def test_smoothing_blurs_the_displacement_across_the_axis_only(self):
        scipy = __import__('pytest').importorskip('scipy.ndimage')
        lateral = 16
        for backend in self.backends():
            with self.subTest(backend=backend):
                image, ids = phantom(lateral)
                image[:, :, :, lateral // 2:] = 1  # Right half is solid: no gaps, no displacement.
                ramp = np.broadcast_to(np.arange(LENGTH, dtype=np.float32).reshape(1, LENGTH, 1, 1), image.shape).copy()
                gaps = np.cumsum(1 - image.astype(np.float64), axis=1)
                field = scipy.gaussian_filter(.35 * gaps, (0, 0, 2, 2), mode='mirror', truncate=6)
                expected = np.clip(np.arange(LENGTH).reshape(1, LENGTH, 1, 1) + field, 0, LENGTH - 1)
                pipeline = A.Pipeline([A.SheetCompression(strength=.35, intensity_sigma=0, smoothing=2, axes='z')],
                                      targets={'image': A.Image(), 'ramp': A.ScalarField()}, backend=backend)
                out = pipeline({'image': image, 'ramp': ramp}, seed=0)
                np.testing.assert_allclose(out['ramp'], expected, atol=.05)  # Kernel truncation differs by ~.02
                self.assertGreater(abs(expected[0, 80, :, 7] - expected[0, 80, :, 8]).max(), 0)

    def test_points_follow_their_sheet(self):
        for backend in self.backends():
            for strength in (.1, .35):
                with self.subTest(backend=backend, strength=strength):
                    image, ids = phantom()
                    centers = np.array([FIRST + PERIOD * k + 1 for k in range(SHEETS)], np.float32)
                    points = np.stack([centers, np.full(SHEETS, 4, np.float32), np.full(SHEETS, 4, np.float32)], 1)
                    pipeline = A.Pipeline([A.SheetCompression(strength=strength, intensity_sigma=0, smoothing=0, axes='z')],
                                          targets={'image': A.Image(), 'seg': A.Label(), 'points': A.Points(order='zyx', outside='keep')},
                                          backend=backend)
                    out = pipeline({'image': image, 'seg': ids, 'points': points}, seed=0)
                    # The point at the middle of sheet k lands where the sampling coordinate equals its input index.
                    coordinate = source_coordinate(image, strength)
                    expected = np.interp(centers, coordinate, np.arange(LENGTH))
                    np.testing.assert_allclose(out['points'][:, 0], expected, atol=1e-3)
                    labels = out['seg'][0, :, 4, 4]
                    for k, z in enumerate(out['points'][:, 0], start=1):
                        self.assertEqual(int(labels[int(round(float(z)))]), k)


if __name__ == '__main__':
    unittest.main()
