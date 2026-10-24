"""Lattice blending, resume and 3D vc-delta3d output of full-volume surface inference."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from vc_reader.reader import open_zarr

from dinovol_2.eval.embed_volume import ZarrGroupWriter
from .infer_volume import axis_starts, ramp, stream


def run(ct, *, tile, stride, bbox, start_chunk=0, end_chunk=None, calls=None):
    """Stream with a pointwise predictor (p = ct / 255), returning the written chunks."""
    (y0, y1), (x0, x1) = bbox
    volume = torch.from_numpy(ct)

    def read(z0, z1):
        out = torch.zeros((z1 - z0, y1 - y0, x1 - x0), dtype=torch.uint8)
        a, b = max(0, z0), min(ct.shape[0], z1)
        if a < b:
            out[a - z0:b - z0] = volume[a:b, y0:y1, x0:x1]
        return out

    written = {}
    def predict(t):
        if calls is not None:
            calls.append(1)
        return t.float() / 255

    stream(read, predict, shape=ct.shape, bbox=bbox, tile=tile, stride=stride,
           write=lambda c, slab: written.__setitem__(c, slab), start_chunk=start_chunk, end_chunk=end_chunk,
           device="cpu", log=lambda m: None)
    return written


class InferVolumeTests(unittest.TestCase):
    def test_ramp_weights_sum_to_one_inside(self):
        for tile, stride in ((12, 8), (192, 128), (192, 96), (10, 10)):
            w = ramp(tile, stride)
            starts = axis_starts(0, 100, tile, stride)
            total = np.zeros(100)
            for s in starts:
                a, b = max(0, s), min(100, s + tile)
                total[a:b] += w[a - s:b - s]
            np.testing.assert_allclose(total, 1.0, atol=1e-12)
            # Every tile intersects the range and the lattice keeps chunk k centred on tile k.
            self.assertTrue(all(s + tile > 0 and s < 100 for s in starts))
            self.assertEqual(starts[1] - starts[0], stride)
            self.assertEqual(starts[0] % stride, (-((tile - stride) // 2)) % stride)

    def test_pointwise_prediction_is_reproduced(self):
        rng = np.random.default_rng(3)
        ct = rng.integers(1, 256, size=(37, 29, 30), dtype=np.uint8)
        ct[:, :5] = 0          # empty margin
        ct[20:, :, 18:] = 0    # empty tiles are skipped
        ct[30:, 20:, :] = 0
        ct[29, 21, 3] = 7      # isolated voxel inside an otherwise empty core
        bbox = ((0, 29), (0, 30))
        tiles = []
        out = run(ct, tile=12, stride=8, bbox=bbox, calls=tiles)
        # One tile per chunk-sized core that holds nonzero CT.
        cores = sum(bool(ct[z:z + 8, y:y + 8, x:x + 8].any())
                    for z in range(0, 37, 8) for y in range(0, 29, 8) for x in range(0, 30, 8))
        self.assertEqual(len(tiles), cores)
        self.assertEqual(sorted(out), list(range(5)))
        got = np.concatenate([out[c] for c in range(5)])
        np.testing.assert_array_equal(got, ct)

    def test_bbox_offset_and_resume_match_full_run(self):
        rng = np.random.default_rng(4)
        ct = np.zeros((40, 32, 40), np.uint8)
        ct[:, 8:24, 8:32] = rng.integers(1, 256, size=(40, 16, 24), dtype=np.uint8)
        bbox = ((8, 24), (8, 32))
        full = run(ct, tile=12, stride=8, bbox=bbox)
        np.testing.assert_array_equal(np.concatenate([full[c] for c in range(5)]), ct[:, 8:24, 8:32])
        resumed = run(ct, tile=12, stride=8, bbox=bbox, start_chunk=2, end_chunk=4)
        self.assertEqual(sorted(resumed), [2, 3])
        for c in (2, 3):
            np.testing.assert_array_equal(resumed[c], full[c])

    def test_delta3d_3d_writer_roundtrip(self):
        rng = np.random.default_rng(5)
        data = rng.integers(0, 256, size=(20, 17, 19), dtype=np.uint8)
        data[:8] = 0  # an all-fill chunk is omitted
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.zarr"
            writer = ZarrGroupWriter(path, attrs={"format": "test"}, workers=2)
            writer.create("0", data.shape, (8, 8, 8), "uint8", codec="vc-delta3d", channel_axis=None)
            writer.write("0", (0, 0, 0), data)
            writer.close()
            self.assertFalse((path / "0" / "0" / "0" / "0").exists())
            np.testing.assert_array_equal(np.asarray(open_zarr(str(path / "0"))[:]), data)
            # Reopening an existing group keeps the 3D layout.
            reopened = ZarrGroupWriter(path, attrs=None, workers=1)
            self.assertIsNone(reopened.channel_axes["0"])
            reopened.close()


if __name__ == "__main__":
    unittest.main()
