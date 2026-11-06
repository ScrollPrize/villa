import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from vc_reader.reader import open_zarr

from dinovol_2.eval.embed_volume import ZarrGroupWriter
from dinovol_2.eval.embeddings_to_rgb import ColorProjector, convert, downsample_colors


class EmbeddingRgbTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_gpu_pyramid_matches_cpu_exactly(self):
        rng = np.random.default_rng(1)
        blocks = rng.integers(0, 256, (4, 3, 48, 16, 32), dtype=np.uint8)
        blocks[0] = 255
        blocks[1, :, :8] = 0
        np.testing.assert_array_equal(downsample_colors(blocks, "cuda"), downsample_colors(blocks, "cpu"))

    def test_rgb_pyramid_resume_and_source_preservation(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "source.zarr", Path(tmp) / "rgb.zarr"
            rng = np.random.default_rng(123)
            values = rng.integers(-127, 128, (17, 16, 19, 5), dtype=np.int8)
            values[:8, :8, :8] = 0
            writer = ZarrGroupWriter(source, attrs={"complete": True})
            writer.create("dense", values.shape, (8, 8, 8, 5), "int8",
                          attrs={"value_scale": 1 / 127}, codec="vc-delta3d")
            writer.write("dense", (0, 0, 0, 0), values)
            writer.close()
            def hashes():
                return {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in source.rglob("*") if p.is_file()}
            original = hashes()
            options = dict(levels=3, threads=2, writer_workers=2, batch_chunks=3,
                           sample_chunks=16, sample_voxels=4096)
            # Interrupt after some writes; uncommitted chunks are deterministically replayed.
            real = ColorProjector.__call__
            calls = 0
            def interrupted(projector, blocks):
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise RuntimeError("test interruption")
                return real(projector, blocks)
            with patch.object(ColorProjector, "__call__", interrupted):
                with self.assertRaisesRegex(RuntimeError, "test interruption"):
                    convert(source, output, **options)
            attrs = convert(source, output, resume=True, **options)
            self.assertTrue(attrs["embedding_rgb"]["complete"])
            colors = ColorProjector(attrs["embedding_rgb"]["pca"], "cpu")
            expected = np.moveaxis(colors([values])[0], -1, 0)
            actual = open_zarr(str(output / "0")).read(...)
            np.testing.assert_array_equal(actual, expected)
            self.assertTrue(np.all(actual[:, :8, :8, :8] == 0))
            for level in (1, 2):
                padded = np.pad(expected, [(0, 0), *[(0, n % 2) for n in expected.shape[1:]]])
                c, z, y, x = padded.shape
                expected = np.rint(padded.reshape(c, z // 2, 2, y // 2, 2, x // 2, 2)
                                   .mean(axis=(2, 4, 6), dtype=np.float32)).astype(np.uint8)
                np.testing.assert_array_equal(open_zarr(str(output / str(level))).read(...), expected)
            meta = json.loads((output / "0" / ".zarray").read_text())
            self.assertEqual(meta["compressor"]["channel_axis"], "first")
            self.assertEqual([a["name"] for a in attrs["multiscales"][0]["axes"]], list("czyx"))
            self.assertEqual(hashes(), original)
            self.assertEqual(convert(source, output, resume=True, **options), attrs)
            with self.assertRaises(ValueError):
                convert(source, source / "rgb", **options)


if __name__ == "__main__":
    unittest.main()
