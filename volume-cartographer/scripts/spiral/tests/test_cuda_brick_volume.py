import unittest

import numpy as np
import torch

from cuda_brick_volume import CudaBrickVolume
from sdt_losses import sample_sdt_trilinear


class ArrayFixture:
    def __init__(self, data, chunks=(2, 2, 2)):
        self.data = np.asarray(data, dtype=np.uint8)
        self.shape = self.data.shape
        self.chunks = chunks
        self.dtype = self.data.dtype

    def __getitem__(self, selection):
        return self.data[selection]


class CudaBrickVolumeTests(unittest.TestCase):
    def test_multichannel_gather_preserves_order_and_roi_origin(self):
        base = np.arange(6 * 4 * 4, dtype=np.uint8).reshape(6, 4, 4)
        store = CudaBrickVolume(
            [ArrayFixture(base), ArrayFixture(base + 100)],
            z_origin=1, roi_shape=(4, 4, 4),
            capacity_bytes=8 * 2 * 8, brick_size=2, device='cpu',
            workers=2)
        try:
            indices = torch.tensor(
                [[[0, 0, 0], [3, 3, 3]], [[1, 2, 1], [0, 0, 0]]])
            actual = store.gather_channels(indices, channels=(1, 0))
            source = indices.clone()
            source[..., 0] += 1
            expected_base = torch.from_numpy(
                base[source[..., 0], source[..., 1], source[..., 2]])
            expected = torch.stack([expected_base + 100, expected_base], dim=-1)
            torch.testing.assert_close(actual, expected)
            self.assertGreater(store.resident_bricks, 0)
            self.assertEqual(store.last_timings['loaded_bricks'],
                             store.resident_bricks)

            # A fully resident repeat takes the no-miss fast path.
            repeated = store.gather_channels(indices, channels=(1, 0))
            torch.testing.assert_close(repeated, expected)
            self.assertEqual(store.last_timings['loaded_bricks'], 0)
            self.assertEqual(store.last_timings['requested_unique_bricks'], 0)
        finally:
            store.close()

    def test_zero_bricks_share_no_data_slot(self):
        data = np.zeros((2, 2, 4), dtype=np.uint8)
        data[:, :, 2:] = 7
        store = CudaBrickVolume(
            [ArrayFixture(data)], z_origin=0, roi_shape=data.shape,
            capacity_bytes=3 * 8, brick_size=2, device='cpu')
        try:
            actual = store.gather(torch.tensor([[0, 0, 0], [1, 1, 3]]))
            self.assertEqual(actual.tolist(), [0, 7])
            self.assertEqual(store.resident_bricks, 1)
            self.assertEqual(store.last_timings['zero_bricks'], 1)
        finally:
            store.close()

    def test_brick_can_span_multiple_source_chunks(self):
        data = np.arange(8 * 8 * 8, dtype=np.uint16).reshape(8, 8, 8)
        data = (data % 251).astype(np.uint8)
        store = CudaBrickVolume(
            [ArrayFixture(data, chunks=(2, 2, 2))],
            z_origin=0, roi_shape=data.shape,
            capacity_bytes=3 * 4 ** 3, brick_size=4, device='cpu')
        try:
            indices = torch.tensor([
                [0, 0, 0], [1, 3, 2], [3, 3, 3],
                [4, 4, 4], [7, 6, 5],
            ])
            actual = store.gather(indices)
            expected = torch.from_numpy(
                data[indices[:, 0], indices[:, 1], indices[:, 2]])
            torch.testing.assert_close(actual, expected)
            self.assertEqual(store.read_shape, (4, 4, 4))
        finally:
            store.close()

    def test_lru_eviction_reloads_exact_values(self):
        data = np.empty((2, 2, 6), dtype=np.uint8)
        data[:, :, :2] = 11
        data[:, :, 2:4] = 22
        data[:, :, 4:] = 33
        # Two total slots: permanent no-data plus one resident data brick.
        store = CudaBrickVolume(
            [ArrayFixture(data)], z_origin=0, roi_shape=data.shape,
            capacity_bytes=2 * 8, brick_size=2, device='cpu')
        try:
            self.assertEqual(int(store.gather(torch.tensor([[0, 0, 0]]))), 11)
            self.assertEqual(int(store.gather(torch.tensor([[0, 0, 3]]))), 22)
            self.assertEqual(int(store.gather(torch.tensor([[0, 0, 0]]))), 11)
            self.assertEqual(store.resident_bricks, 1)
            self.assertEqual(store.total_evicted_bricks, 2)
        finally:
            store.close()

    def test_request_larger_than_capacity_is_rejected(self):
        data = np.ones((2, 2, 4), dtype=np.uint8)
        store = CudaBrickVolume(
            [ArrayFixture(data)], z_origin=0, roi_shape=data.shape,
            capacity_bytes=2 * 8, brick_size=2, device='cpu')
        try:
            with self.assertRaisesRegex(RuntimeError, 'one gather requires'):
                store.gather(torch.tensor([[0, 0, 0], [0, 0, 3]]))
        finally:
            store.close()

    def test_sdt_trilinear_matches_dense_backend(self):
        x = np.arange(8, dtype=np.float32)
        encoded = (np.clip(np.rint(np.abs(x - 3.0) - 1.0), -127, 127)
                   + 128).astype(np.uint8)
        data = np.broadcast_to(encoded, (4, 4, 8)).copy()
        store = CudaBrickVolume(
            [ArrayFixture(data)], z_origin=0, roi_shape=data.shape,
            capacity_bytes=16 * 8, brick_size=2, device='cpu')
        common = {
            'kind': 'sdt', 'z_origin': 0, 'scale_zyx': (1.0, 1.0, 1.0),
            'unit': 1.0, 'offset': 128, 'cap': 127.0, 'shape': data.shape,
        }
        dense = {'backend': 'dense_cuda',
                 'volume': torch.from_numpy(data), **common}
        bricks = {'backend': 'cuda_bricks', 'store': store, **common}
        points = torch.tensor([
            [0.5, 0.5, 0.5], [1.25, 2.5, 3.75], [2.0, 1.0, 6.5],
        ])
        try:
            dense_value, dense_valid, dense_corners = sample_sdt_trilinear(
                dense, points)
            brick_value, brick_valid, brick_corners = sample_sdt_trilinear(
                bricks, points)
            torch.testing.assert_close(brick_corners, dense_corners)
            torch.testing.assert_close(brick_value, dense_value)
            torch.testing.assert_close(brick_valid, dense_valid)
        finally:
            store.close()


if __name__ == '__main__':
    unittest.main()
