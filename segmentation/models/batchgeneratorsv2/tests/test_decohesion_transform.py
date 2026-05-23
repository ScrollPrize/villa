import unittest

import torch

from batchgeneratorsv2.transforms.noise.extranoisetransforms import (
    DecohesionTransform,
    SmearTransform,
)


class DecohesionTransformTests(unittest.TestCase):
    def test_smears_previous_slice_without_changing_shape_dtype_or_device(self):
        image = torch.zeros((1, 4, 3, 3), dtype=torch.float32)
        image[0, 0, 1, 1] = 1

        transform = DecohesionTransform(
            shift=(0, 1),
            alpha=1.0,
            num_prev_slices=1,
            smear_axis=1,
            p_per_channel=1.0,
        )

        out = transform(image=image.clone())["image"]

        expected_slice = torch.roll(image[0, 0], shifts=(0, 1), dims=(0, 1))
        self.assertEqual(out.shape, image.shape)
        self.assertEqual(out.dtype, image.dtype)
        self.assertEqual(out.device, image.device)
        torch.testing.assert_close(out[0, 0], image[0, 0])
        torch.testing.assert_close(out[0, 1], expected_slice)

    def test_image_only_transform_leaves_segmentation_unchanged(self):
        image = torch.zeros((1, 4, 3, 3), dtype=torch.float32)
        image[0, 0, 1, 1] = 1
        segmentation = torch.arange(4 * 3 * 3, dtype=torch.int64).reshape(1, 4, 3, 3)

        transform = DecohesionTransform(
            shift=(1, 0),
            alpha=0.5,
            num_prev_slices=2,
            smear_axis=1,
            p_per_channel=1.0,
        )

        out = transform(image=image.clone(), segmentation=segmentation.clone())

        torch.testing.assert_close(out["segmentation"], segmentation)

    def test_randomized_scroll_prize_config_path_runs(self):
        image = torch.zeros((1, 5, 12, 12), dtype=torch.float32)
        image[0, 0, 5, 5] = 1

        out = DecohesionTransform(
            shift=((-6, 6), (-6, 6)),
            alpha=(0.15, 0.45),
            num_prev_slices=(1, 3),
            smear_axis=1,
            p_per_channel=1.0,
        )(image=image.clone())["image"]

        self.assertEqual(out.shape, image.shape)
        self.assertEqual(out.dtype, image.dtype)

    def test_zero_channel_probability_is_noop(self):
        image = torch.rand((2, 5, 6, 6), dtype=torch.float32)

        out = DecohesionTransform(
            shift=(1, 1),
            alpha=1.0,
            num_prev_slices=2,
            smear_axis=1,
            p_per_channel=0.0,
        )(image=image.clone())["image"]

        torch.testing.assert_close(out, image)

    def test_2d_image_supports_integer_shift(self):
        image = torch.zeros((1, 4, 5), dtype=torch.float32)
        image[0, 0, 2] = 1

        out = DecohesionTransform(
            shift=1,
            alpha=1.0,
            num_prev_slices=1,
            smear_axis=1,
            p_per_channel=1.0,
        )(image=image.clone())["image"]

        expected_slice = torch.roll(image[0, 0], shifts=1, dims=0)
        torch.testing.assert_close(out[0, 1], expected_slice)

    def test_invalid_smear_axis_raises_clear_error(self):
        image = torch.zeros((1, 4, 3, 3), dtype=torch.float32)

        transform = DecohesionTransform(
            shift=(1, 1),
            alpha=0.5,
            num_prev_slices=1,
            smear_axis=4,
            p_per_channel=1.0,
        )

        with self.assertRaisesRegex(ValueError, "smear_axis"):
            transform(image=image)

        with self.assertRaisesRegex(ValueError, "smear_axis"):
            DecohesionTransform(smear_axis=1.2)

        with self.assertRaisesRegex(ValueError, "smear_axis"):
            DecohesionTransform(smear_axis=True)

        with self.assertRaisesRegex(ValueError, "smear_axis"):
            DecohesionTransform(smear_axis="1")

    def test_shift_dimension_must_match_non_smear_plane(self):
        image = torch.zeros((1, 4, 3), dtype=torch.float32)

        transform = DecohesionTransform(
            shift=(1, 1),
            alpha=0.5,
            num_prev_slices=1,
            smear_axis=1,
            p_per_channel=1.0,
        )

        with self.assertRaisesRegex(ValueError, "shift"):
            transform(image=image)

    def test_invalid_ranges_raise_clear_errors(self):
        with self.assertRaisesRegex(ValueError, "alpha"):
            DecohesionTransform(alpha=(0.8, 0.2))

        with self.assertRaisesRegex(ValueError, "num_prev_slices"):
            DecohesionTransform(num_prev_slices=(3, 1))

        with self.assertRaisesRegex(ValueError, "p_per_channel"):
            DecohesionTransform(p_per_channel=1.5)

        with self.assertRaisesRegex(ValueError, "shift"):
            DecohesionTransform(shift=((2, -2), (0, 1)))

    def test_smear_transform_keeps_backward_compatible_name(self):
        image = torch.zeros((1, 3, 2, 2), dtype=torch.float32)
        image[0, 0, 0, 0] = 2

        out = SmearTransform(
            shift=(1, 0),
            alpha=1.0,
            num_prev_slices=1,
            smear_axis=1,
        )(image=image.clone())["image"]

        expected_slice = torch.roll(image[0, 0], shifts=(1, 0), dims=(0, 1))
        torch.testing.assert_close(out[0, 1], expected_slice)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_tensor_stays_on_cuda(self):
        image = torch.zeros((1, 4, 8, 8), dtype=torch.float32, device="cuda")
        image[0, 0, 3, 3] = 1

        out = DecohesionTransform(
            shift=(1, 1),
            alpha=0.5,
            num_prev_slices=1,
            smear_axis=1,
            p_per_channel=1.0,
        )(image=image)["image"]

        self.assertEqual(out.device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
