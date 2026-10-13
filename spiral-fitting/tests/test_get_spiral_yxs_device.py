"""get_spiral_yxs allocates on the caller's device, not an implicit CUDA one.

A CPU export/preview run passes a CPU ``dr_per_winding`` tensor and no
explicit device: the spiral grid must then be built on CPU. Numeric
``dr_per_winding`` keeps the historical CUDA default so GPU callers that pass
only numbers are unchanged, and an explicit ``device`` argument always wins.
"""
import unittest

import torch

from sample_spiral import get_spiral_yxs


class GetSpiralYxsDeviceTests(unittest.TestCase):
    def test_cpu_tensor_dr_infers_cpu(self):
        yxs = get_spiral_yxs(13, torch.tensor(10.0), 2,
                             group_by_winding=True)
        self.assertTrue(all(part.device.type == 'cpu' for part in yxs))
        flat = torch.cat(yxs, dim=0)
        self.assertEqual(flat.device.type, 'cpu')

    @unittest.skipIf(not torch.cuda.is_available(), 'CUDA default needs a GPU')
    def test_numeric_dr_defaults_to_cuda(self):
        # Historical CUDA default for numeric dr: unchanged GPU route.
        yxs = get_spiral_yxs(13, 10.0, 2)
        self.assertTrue(yxs.is_cuda)

    def test_numeric_dr_without_cuda_stays_buildable_on_cpu(self):
        # A machine without CUDA (the CPU-only export path) must be able to
        # ask for CPU explicitly; it did before the fix too, this pins it.
        yxs = get_spiral_yxs(13, 10.0, 2, device='cpu')
        self.assertEqual(yxs.device.type, 'cpu')

    @unittest.skipIf(not torch.cuda.is_available(), 'Device override needs a GPU')
    def test_explicit_device_beats_tensor_inference(self):
        yxs = get_spiral_yxs(4, torch.tensor(10.0), 2, device='cuda')
        self.assertTrue(yxs.is_cuda)
        expected = get_spiral_yxs(4, 10.0, 2, device='cuda')
        torch.testing.assert_close(yxs, expected)


if __name__ == '__main__':
    unittest.main()
