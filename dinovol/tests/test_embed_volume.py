import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from vc_reader import transforms as A

from dinovol_2.eval.embed_volume import (
    AxisLattice, RegionEmbedder, ZarrGroupWriter, add_unit_tokens, global_normalization, read_embeddings,
    statistics_from_histogram,
)
from dinovol_2.eval.napari_visualizer import LoadedBackbone
from dinovol_2.model import build_backbone
from dinovol_2.model.model import PointDenseHead


class ArrayReader:
    """In-memory stand-in for a vc_reader ZarrReader (zero outside the array)."""

    def __init__(self, array, chunks=(16, 16, 16)):
        self.array, self.chunks = array, chunks
        self.shape, self.dtype = array.shape, array.dtype

    def read_region(self, start, size):
        out = np.zeros(size, dtype=self.array.dtype)
        src = tuple(slice(max(s, 0), min(s + n, d)) for s, n, d in zip(start, size, self.shape))
        dst = tuple(slice(a.start - s, a.stop - s) for a, s in zip(src, start))
        if all(a.stop > a.start for a in src):
            out[dst] = self.array[src]
        return out

    def read_regions(self, regions):
        return [self.read_region(start, size) for start, size in regions]


class MemorySink:
    def __init__(self, shapes):
        self.arrays = {name: np.zeros(shape, dtype) for name, (shape, dtype) in shapes.items()}

    def write(self, name, start, block):
        self.arrays[name][tuple(slice(s, s + n) for s, n in zip(start, block.shape))] = block


class LatticeTests(unittest.TestCase):
    def test_weights_partition_unity_on_patch_aligned_supports(self):
        lattice = AxisLattice(192, 24, 24)
        self.assertEqual(lattice.stride, 120)
        u = torch.arange(-500, 1500, dtype=torch.float64) + 0.5
        total = torch.zeros_like(u)
        for k in lattice.tiles(-500, 1500):
            w = lattice.weights(u - lattice.origin(k))
            inside = (u >= k * lattice.stride) & (u < k * lattice.stride + lattice.support)
            self.assertTrue(torch.all(w[~inside] == 0))
            total += w
        torch.testing.assert_close(total, torch.ones_like(total), rtol=0, atol=1e-12)
        self.assertEqual(list(lattice.tiles(14000, 14192)), [116, 117, 118])

    def test_invalid_lattice_is_rejected(self):
        with self.assertRaises(ValueError):
            AxisLattice(192, 60, 40).validate(step=8, name="z")
        with self.assertRaises(ValueError):
            AxisLattice(192, 20, 24).validate(step=8, name="z")


@unittest.skipUnless(torch.cuda.is_available(), "vc_reader batches run on CUDA")
class NormalizationTests(unittest.TestCase):
    @staticmethod
    def apply(transforms, image):
        pipeline = A.Pipeline(transforms, targets={"image": A.Image(valid=None)}, backend="cuda", compact=False)
        return pipeline.batch({"image": image.cuda()}, seeds=[0], device="cuda")["image"].cpu()

    def test_histogram_statistics_reproduce_vc_reader_robust(self):
        rng = np.random.default_rng(3)
        block = np.clip(rng.gamma(3, 18, size=(24, 20, 22)), 1, 255).astype(np.uint8)
        stats = statistics_from_histogram(np.bincount(block.ravel(), minlength=256))
        image = torch.from_numpy(block.astype(np.float32))[None, None]
        torch.testing.assert_close(self.apply(global_normalization("robust", stats, clip_lower=True), image),
                                   self.apply([A.Normalize("robust")], image), rtol=0, atol=1e-5)

    def test_robust_keeps_masked_zeros_below_the_lower_percentile(self):
        stats = statistics_from_histogram(np.bincount(np.arange(20, 220), minlength=256))
        image = torch.tensor([0.0, stats["percentile_01"], 250.0]).reshape(1, 1, 1, 1, 3)
        out = self.apply(global_normalization("robust", stats), image).flatten()
        center, scale = stats["median_clipped"], stats["robust_scale"]
        torch.testing.assert_close(out, torch.tensor([-center / scale, (stats["percentile_01"] - center) / scale,
                                                      (stats["percentile_99"] - center) / scale]))


class WriterTests(unittest.TestCase):
    def test_chunk_writes_round_trip_and_skip_empty_chunks(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            writer = ZarrGroupWriter(Path(tmp) / "out.zarr", attrs={"x": 1})
            writer.create("dense", (4, 10, 10, 3), (4, 5, 5, 3), np.int8)
            writer.create("dense_scale", (4, 10, 10), (4, 5, 5), np.float16)
            values = np.zeros((4, 10, 10, 3), np.int8)
            values[:, 5:, 5:] = np.arange(4 * 25 * 3).reshape(4, 5, 5, 3) % 127
            scale = (values.max(-1) > 0).astype(np.float16) * np.float16(0.5)
            writer.write("dense", (0, 5, 5), values[:, 5:, 5:])
            writer.write("dense_scale", (0, 5, 5), scale[:, 5:, 5:])
            writer.write("dense", (0, 0, 0), values[:, :5, :5])
            writer.close()
            chunks = sorted(p.relative_to(Path(tmp) / "out.zarr/dense").as_posix()
                            for p in (Path(tmp) / "out.zarr/dense").rglob("*") if p.is_file() and p.name != ".zarray")
            self.assertEqual(chunks, ["0/1/1/0"])
            np.testing.assert_array_equal(read_embeddings(Path(tmp) / "out.zarr", "dense"),
                                          values.astype(np.float32) * scale[..., None].astype(np.float32))


@unittest.skipUnless(torch.cuda.is_available(), "tiled embedding runs on CUDA")
class RegionEmbedderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        config = dict(model_type="v3", embedding_type="deeper", embed_dim=48, depth=2, num_heads=2,
                      patch_size=[8, 8, 8], global_crops_size=[48] * 3, local_crops_size=[16] * 3,
                      deeper_embed_stage_channels=[4, 8, 8, 16], deeper_embed_stage_strides=[1, 2, 2, 2],
                      deeper_embed_depth_per_level=[1, 1, 1, 1], deeper_embed_stem_channels=4,
                      drop_path_rate=0.0, num_reg_tokens=1)
        backbone = build_backbone(config).cuda().eval()
        head = PointDenseHead(token_dim=48, skip_channels=tuple(backbone.down_projection.skip_channels),
                              hidden_dim=16, out_dim=8, ink_dim=4).cuda().eval()
        cls.loaded = LoadedBackbone(checkpoint_path=Path("."), backbone=backbone, source_branch="teacher",
                                    device=torch.device("cuda"), patch_size=(8, 8, 8), input_channels=1,
                                    normalization_scheme="robust", intensity_properties=None,
                                    embedding_type="deeper", global_input_size=(48, 48, 48), dense_head=head)
        rng = np.random.default_rng(1)
        volume = rng.integers(20, 200, size=(120, 110, 100), dtype=np.uint8)
        volume[:, :, :30] = 0                   # masked-out slab
        volume[60:, 70:, 60:] = 0               # masked-out corner
        cls.volume = volume
        cls.reader = ArrayReader(volume)
        stats = statistics_from_histogram(np.bincount(volume[volume > 0], minlength=256))
        cls.transforms = global_normalization("robust", stats)

    def embedder(self, *, dense_stride=2, dtype="float16", outputs=("tokens", "tokens_unit", "dense", "ink")):
        return RegionEmbedder(self.reader, self.loaded, transforms=self.transforms, outputs=outputs,
                              dense_stride=dense_stride, lattice=[AxisLattice(48, 8, 8)] * 3, dtype=dtype,
                              point_chunk_size=4096)

    def embed(self, z_range, *, y_range=None, x_range=None, dense_stride=2, dtype="float16"):
        embedder = self.embedder(dense_stride=dense_stride, dtype=dtype)
        shapes = {}
        for name, _, shape, _ in embedder.arrays(z_range):
            shapes[name] = (shape, np.dtype(dtype))
            if dtype == "int8":
                shapes[f"{name}_scale"] = (shape[:3], np.float16)
        sink = MemorySink(shapes)
        stats = embedder.run(sink, z_range=z_range, y_range=y_range, x_range=x_range, progress=False)
        return sink.arrays, stats

    def test_unit_tokens_are_normalized_tokens(self):
        arrays, _ = self.embed((32, 64))
        tokens, unit = arrays["tokens"].astype(np.float32), arrays["tokens_unit"].astype(np.float32)
        valid = np.abs(tokens).sum(-1) > 0
        np.testing.assert_array_equal(np.abs(unit).sum(-1) > 0, valid)
        np.testing.assert_allclose(np.linalg.norm(unit[valid], axis=-1), 1, atol=2e-3)
        expected = tokens[valid] / np.linalg.norm(tokens[valid], axis=-1, keepdims=True)
        np.testing.assert_allclose(unit[valid], expected, rtol=0, atol=2e-3)

    def test_unit_tokens_derived_from_stored_tokens_match_the_run(self):
        z_range = (32, 64)
        embedder = self.embedder(dtype="int8", outputs=("tokens", "tokens_unit"))
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            path = Path(tmp) / "out.zarr"
            sink = ZarrGroupWriter(path, attrs={"outputs": {}})
            attrs = {"outputs": {}}
            for name, spec, shape, chunks in embedder.arrays(z_range):
                sink.create(name, shape, chunks, np.int8)
                sink.create(f"{name}_scale", shape[:3], chunks[:3], np.float16)
                attrs["outputs"][name] = {"step_zyx": list(spec.step)}
            (path / ".zattrs").write_text(json.dumps(attrs))
            embedder.run(sink, z_range=z_range, progress=False)
            sink.close()
            self.assertGreater(add_unit_tokens(path, name="tokens_unit_derived"), 0)
            run = read_embeddings(path, "tokens_unit")
            derived = read_embeddings(path, "tokens_unit_derived")
            valid = np.abs(run).sum(-1) > 0
            np.testing.assert_array_equal(np.abs(derived).sum(-1) > 0, valid)
            a, b = run[valid], derived[valid]
            cos = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
            self.assertGreater(cos.min(), 0.9999)
            meta = json.loads((path / ".zattrs").read_text())["outputs"]["tokens_unit_derived"]
            self.assertEqual(meta["derived_from"], "tokens")

    def test_outputs_do_not_depend_on_requested_region(self):
        a, stats = self.embed((32, 64))
        self.assertGreater(stats["skipped_columns"], 0)  # x < 30 is masked out
        b, _ = self.embed((16, 96), y_range=(24, 72), x_range=(48, 100))
        for name, step in (("tokens", 8), ("dense", 2), ("ink", 2)):
            za, zb = slice(0, 32 // step), slice(16 // step, 48 // step)
            # Lattice stride 24: cells overlapping y [24, 72) and x [48, 100) are exactly those ranges.
            ys, xs = slice(24 // step, 72 // step), slice(48 // step, 100 // step)
            got, want = b[name][zb, ys, xs].astype(np.float32), a[name][za, ys, xs].astype(np.float32)
            self.assertTrue(np.any(want != 0))
            np.testing.assert_allclose(got, want, rtol=0, atol=2e-3, err_msg=name)
            self.assertFalse(np.any(b[name][:, :24 // step]), name)
            self.assertFalse(np.any(b[name][:, :, :48 // step]), name)

    def reference(self, z_range, step):
        """Brute force: every lattice tile, full outputs, blended on the global grid."""
        lattice = AxisLattice(48, 8, 8)
        z0, z1 = z_range
        shape = self.volume.shape
        pipeline = A.Pipeline(self.transforms, targets={"image": A.Image(valid=None)}, backend="cuda",
                              compact=False)
        token_shape = ((z1 - z0) // 8, -(-shape[1] // 8), -(-shape[2] // 8))
        point_shape = ((z1 - z0) // step, -(-shape[1] // step), -(-shape[2] // step))
        tokens = torch.zeros((*token_shape, 48), device="cuda")
        points = torch.zeros((*point_shape, 12), device="cuda")
        weight_of = lambda c, a: lattice.weights(c - a)
        backbone, head = self.loaded.backbone, self.loaded.dense_head
        backbone.return_embed_skips = True
        for kz in lattice.tiles(z0, z1):
            for ky in lattice.tiles(0, shape[1]):
                for kx in lattice.tiles(0, shape[2]):
                    origin = [lattice.origin(k) for k in (kz, ky, kx)]
                    raw = torch.from_numpy(self.reader.read_region(origin, (48,) * 3)).cuda()[None, None]
                    image = pipeline.batch({"image": raw}, seeds=[0], device="cuda")["image"]
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                        out = backbone.forward_features(image, masks=None, view_kind="global")
                        grid = out["x_norm_patchtokens"].reshape(1, 6, 6, 6, -1)
                        idx = torch.stack(torch.meshgrid(*[torch.arange(48 // step, device="cuda")] * 3,
                                                         indexing="ij"), -1).reshape(-1, 3)
                        coords = (idx * step + (step - 1) / 2).float()
                        heads = head.forward_points(grid, out["embed_skips"], coords[None], (48,) * 3)
                    for values, grid_step, acc in ((grid[0].reshape(-1, 48), 8, tokens),
                                                   (torch.cat([heads["dense"][0], heads["ink"][0]], -1), step, points)):
                        n = 48 // grid_step
                        local = torch.stack(torch.meshgrid(*[torch.arange(n, device="cuda")] * 3, indexing="ij"),
                                            -1).reshape(-1, 3)
                        centers = local * grid_step + grid_step / 2
                        w = torch.ones(len(local), device="cuda")
                        for axis in range(3):
                            w = w * weight_of(centers[:, axis].float(), 0)
                        glob = local + torch.tensor(origin, device="cuda") // grid_step
                        glob[:, 0] -= z0 // grid_step
                        keep = (w > 0) & (glob >= 0).all(1) & (glob < torch.tensor(acc.shape[:3], device="cuda")).all(1)
                        acc.index_put_(tuple(glob[keep].T), values[keep].float() * w[keep, None], accumulate=True)
        backbone.return_embed_skips = False
        dense = torch.nn.functional.normalize(points[..., :8], dim=-1)
        ink = torch.nn.functional.normalize(points[..., 8:], dim=-1)
        return {"tokens": tokens.cpu().numpy(), "dense": dense.cpu().numpy(), "ink": ink.cpu().numpy()}

    def test_blend_matches_brute_force_reference(self):
        got, _ = self.embed((32, 64))
        want = self.reference((32, 64), 2)
        for name in ("tokens", "dense", "ink"):
            valid = np.abs(got[name].astype(np.float32)).sum(-1) > 0
            self.assertGreater(valid.mean(), 0.3)
            np.testing.assert_allclose(got[name].astype(np.float32)[valid], want[name][valid],
                                       rtol=0, atol=3e-2, err_msg=name)
            err = np.abs(got[name].astype(np.float32)[valid] - want[name][valid])
            self.assertLess(err.mean(), 2e-3, name)

    def test_masked_voxels_are_not_embedded(self):
        arrays, _ = self.embed((32, 64), dense_stride=1)
        nonzero = self.volume[32:64] != 0
        dense = arrays["dense"].astype(np.float32)
        norms = np.linalg.norm(dense, axis=-1)
        self.assertTrue(np.all(norms[~nonzero] == 0))
        np.testing.assert_allclose(norms[nonzero], 1, atol=2e-3)
        tokens = np.abs(arrays["tokens"]).sum(-1) > 0
        patches = np.zeros((32, 112, 104), bool)
        patches[:, :110, :100] = nonzero
        token_valid = patches.reshape(4, 8, 14, 8, 13, 8).any(axis=(1, 3, 5))
        np.testing.assert_array_equal(tokens, token_valid)

    def test_int8_matches_float16_closely(self):
        f16, _ = self.embed((32, 64))
        i8, _ = self.embed((32, 64), dtype="int8")
        for name in ("tokens", "dense", "ink"):
            ref = f16[name].astype(np.float32)
            got = i8[name].astype(np.float32) * i8[f"{name}_scale"].astype(np.float32)[..., None]
            valid = np.abs(ref).sum(-1) > 0
            np.testing.assert_array_equal(np.abs(got).sum(-1) > 0, valid)
            cos = (ref * got).sum(-1)[valid] / (np.linalg.norm(ref, axis=-1) * np.linalg.norm(got, axis=-1))[valid]
            self.assertGreater(cos.min(), 0.999, name)


if __name__ == "__main__":
    unittest.main()
