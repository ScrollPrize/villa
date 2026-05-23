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
