import unittest

import torch

from dinovol_2.ops.point_embeddings import grid_sample_geometry, sample_grid_points, sample_selected_grid_points
from dinovol_2.eval.embed_volume import DENSE_POINT_CHUNK_SIZE, validate_resume


class SamplingGeometryTests(unittest.TestCase):
    def check_device(self, device):
        torch.manual_seed(42)
        for dtype in (torch.float32, torch.bfloat16):
            grid = torch.randn(2, 5, 7, 9, 11, device=device, dtype=dtype, requires_grad=True)
            coordinates = (torch.rand(2, 31, 3, device=device) * 35 - 5).requires_grad_()
            geometry = grid_sample_geometry(grid.shape[1:4], (24, 28, 32), coordinates)
            eager = sample_grid_points(grid, coordinates, (24, 28, 32))
            cached = sample_grid_points(grid, coordinates, (24, 28, 32), geometry=geometry)
            torch.testing.assert_close(cached, eager, rtol=0, atol=0)
            old_grads = torch.autograd.grad(eager.sum(), (grid, coordinates), retain_graph=True)
            new_grads = torch.autograd.grad(cached.sum(), (grid, coordinates))
            # CUDA scatter-add ordering may vary; forward results must be bit-identical.
            for old, new in zip(old_grads, new_grads):
                torch.testing.assert_close(new, old, rtol=0, atol=0 if device == 'cpu' else 1e-4)

    def test_cached_geometry_cpu(self):
        self.check_device('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cached_geometry_cuda(self):
        self.check_device('cuda')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_fused_sampler_layouts_edges_and_padding(self):
        from dinovol_2.ops.selected_interpolation import available
        if not available():
            self.skipTest('Triton required')
        torch.manual_seed(123)
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for shape in ((2, 7, 9, 11, 24), (1, 12, 12, 12, 480), (1, 1, 3, 5, 1)):
                for channels_last in (True, False):
                    with self.subTest(dtype=dtype, shape=shape, channels_last=channels_last):
                        grid = torch.randn(shape, dtype=dtype, device='cuda')
                        if not channels_last:
                            grid = grid.permute(0, 4, 1, 2, 3).contiguous().permute(0, 2, 3, 4, 1)
                        # Noncontiguous coordinates, partial GPU blocks, and out-of-bounds clamping.
                        points = (torch.rand(shape[0], 266, 3, device='cuda') * 240 - 24)[:, ::2]
                        points[:, :2] = torch.tensor([[-100, -100, -100], [300, 300, 300]], device='cuda')
                        expected = sample_grid_points(grid, points, (192, 160, 128))
                        actual = sample_grid_points(grid, points, (192, 160, 128), use_triton=True)
                        self.assertEqual(actual.dtype, torch.float32)
                        torch.testing.assert_close(actual, expected, rtol=3e-6, atol=4e-6)
                        empty = sample_grid_points(grid, points[:, :0], (192, 160, 128), use_triton=True)
                        self.assertEqual(empty.shape, (shape[0], 0, shape[-1]))

    def test_fused_sampler_keeps_autograd_and_cpu_fallback(self):
        for device in ('cpu', 'cuda') if torch.cuda.is_available() else ('cpu',):
            grid = torch.randn(2, 3, 5, 7, 8, device=device, requires_grad=True)
            points = torch.rand(2, 19, 3, device=device, requires_grad=True)
            expected = sample_grid_points(grid, points, (4, 6, 8))
            actual = sample_grid_points(grid, points, (4, 6, 8), use_triton=True)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            actual.sum().backward()
            self.assertIsNotNone(grid.grad)
            self.assertIsNotNone(points.grad)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_shared_selected_kernel(self):
        torch.manual_seed(42)
        grid = torch.randn(2, 5, 7, 9, 17, device='cuda')
        points = torch.rand(3, 23, 3, device='cuda') * 40 - 4
        indices = torch.tensor([0, 22, 26, 68, 11], device='cuda')
        rows = torch.tensor([1, 0, 1], device='cuda')
        expected = sample_selected_grid_points(grid, points, (24, 28, 32), indices, rows, use_triton=False)
        actual = sample_selected_grid_points(grid, points, (24, 28, 32), indices, rows)
        torch.testing.assert_close(actual, expected, rtol=3e-6, atol=4e-6)


class AsyncWriterTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_flush_waits_for_nondefault_stream_and_preserves_chunks(self):
        import tempfile
        from pathlib import Path
        from dinovol_2.eval.embed_volume import ZarrGroupWriter, read_embeddings
        with tempfile.TemporaryDirectory() as tmp:
            writer = ZarrGroupWriter(Path(tmp) / 'output', attrs={}, workers=2)
            writer.create('dense', (24, 8, 8, 5), (4, 4, 4, 5), 'int8', codec='vc-delta3d')
            expected = torch.arange(24 * 8 * 8 * 5).reshape(24, 8, 8, 5).remainder(127).to(torch.int8)
            producer = torch.cuda.Stream()
            try:
                with torch.cuda.stream(producer):
                    for z in range(0, 24, 4):
                        # A strided block exercises the contiguous copy before staging.
                        block = expected[z:z+4].cuda().transpose(1, 2)
                        writer.write_tensor('dense', (z, 0, 0), block)
                        del block
                writer.flush()
                import numpy as np
                np.testing.assert_array_equal(read_embeddings(writer.path, 'dense'),
                                              expected.transpose(1, 2).numpy())
                self.assertEqual(writer._transfers, [])
            finally:
                writer.close()

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_transfer_submission_failure_reaches_close(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from dinovol_2.eval.embed_volume import ZarrGroupWriter
        with tempfile.TemporaryDirectory() as tmp:
            writer = ZarrGroupWriter(Path(tmp) / 'output', attrs={})
            with patch.object(writer, '_write', side_effect=RuntimeError('staging failed')):
                writer.write_tensor('dense', (0, 0, 0), torch.ones(4, 4, 4, 2, device='cuda', dtype=torch.int8))
                with self.assertRaisesRegex(RuntimeError, 'staging failed'):
                    writer.close()


class ResumePerformanceTests(unittest.TestCase):

    def test_old_manifest_implies_original_point_batch(self):
        # Deliberately incomplete schema: matching defaults gets past config validation.
        saved = {'outputs': {}, 'slabs': [], 'completed_slabs': {}}
        expected = dict(saved, point_batch_size=DENSE_POINT_CHUNK_SIZE)
        from pathlib import Path
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / '.zgroup').write_text('{"zarr_format":2}')
            validate_resume(saved, expected, path)
            with self.assertRaisesRegex(ValueError, 'point_batch_size'):
                validate_resume(saved, dict(expected, point_batch_size=65536), path)


if __name__ == '__main__':
    unittest.main()
