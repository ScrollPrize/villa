"""Small regressions for conversion, ignore handling, freezing and CUDA parity."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import tifffile
import torch
from vc_reader import cuda, transforms as A
from vc_reader.reader import open_zarr

from dinovol_2.model.backbone import build_backbone
from .data import SurfaceData
from .model import SurfaceModel
from .prepare import pairs, prepare
from .train import compute_loss
from .vesuvius import components, medial_loss


def source_checkout():
    # Works from either the final dinovol/evals location or an installed test tree.
    for ancestor in Path(__file__).resolve().parents:
        source = ancestor / "vesuvius/src"
        if (source / "vesuvius/models/build/decoder.py").exists():
            return source
    return None


class SurfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.source = source_checkout()
        components(cls.source)

    def fixture(self, root):
        image = np.arange(19 * 21 * 23, dtype=np.uint8).reshape(19, 21, 23)
        label = np.zeros_like(image, dtype=np.uint8)
        label[:, 8:12, 4:20] = 1
        label[:3] = 2
        for split, name in (("", "train_case"), ("val", "val_case")):
            for kind, values, suffix in (("images", image, ""), ("labels", label, "_surface")):
                directory = root / split / kind
                directory.mkdir(parents=True)
                tifffile.imwrite(directory / f"{name}{suffix}.tif", values, photometric="minisblack")
        return image, label

    def test_conversion_padding_replay_and_source_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image, label = self.fixture(root / "source")
            manifest = prepare(root / "source", root / "data", chunk=8, workers=2)
            self.assertEqual(len(manifest["cases"]), 2)
            for key, expected in (("image", image), ("label", label)):
                with open_zarr(root / f"data/train/train_case.zarr/{key}") as reader:
                    np.testing.assert_array_equal(reader[:], expected)
                    padded = reader.read_region((-2, -2, -2), (4, 4, 4))
                    self.assertTrue(np.all(padded[0] == (2 if key == "label" else 0)))
            self.assertEqual(manifest, prepare(root / "source", root / "data", chunk=8, workers=1))
            data = SurfaceData(root / "data", (16, 16, 16), device="cpu", normalization="zscore")
            try:
                first = list(data.validation(seed=7, crops_per_case=1))
                second = list(data.validation(seed=7, crops_per_case=1))
                self.assertEqual(first[0][1], second[0][1])
                for key in first[0][0]:
                    torch.testing.assert_close(first[0][0][key], second[0][0][key], rtol=0, atol=0)
                batch = data.training_batch(np.random.default_rng(4), 2)
                self.assertEqual(tuple(batch["image"].shape), (2, 1, 16, 16, 16))
                self.assertTrue(torch.isfinite(batch["image"]).all())
                self.assertTrue((batch["label_skel"][batch["label"] == 2] == 0).all())
            finally:
                data.close()
            (root / "source/images/train_case.tif").touch()
            with self.assertRaisesRegex(ValueError, "changed"):
                prepare(root / "source", root / "data", chunk=8, workers=1)

    def test_split_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            (root / "val/images/val_case.tif").rename(root / "val/images/train_case.tif")
            (root / "val/labels/val_case_surface.tif").rename(root / "val/labels/train_case_surface.tif")
            with self.assertRaisesRegex(ValueError, "overlap"):
                pairs(root)

    def test_medial_target_semantics_and_ignore_gradients(self):
        labels = np.zeros((1, 16, 16, 16), np.uint8)
        labels[:, :, 6:10, 2:14] = 1
        labels[:, :2] = 2
        transforms = [A.MedialSurface(on=["label"], ignore={"label": 2})]
        native = A.Pipeline(transforms, targets={"label": A.Label(ignore_value=2)}, backend="cpu")
        actual = native({"label": labels}, seed=0)["label_skel"]
        # vc_reader's native thinning is not bit-identical to skimage's. Check
        # the supervision contract: binary medial tube restricted to foreground.
        self.assertEqual(actual.shape, labels.shape)
        self.assertTrue(np.isin(actual, [0, 1]).all())
        self.assertGreater(actual.sum(), 0)
        self.assertTrue((actual[labels != 1] == 0).all())
        logits = torch.randn(1, 2, 16, 16, 16, requires_grad=True)
        sample = {"label": torch.from_numpy(labels)[None], "label_skel": torch.from_numpy(actual)[None]}
        criterion = medial_loss(self.source)
        compute_loss(criterion, logits, sample).backward()
        ignored = (sample["label"] == 2).expand_as(logits)
        self.assertTrue((logits.grad[ignored] == 0).all())
        logits.grad = None
        sample["label"].fill_(2)
        empty_loss = compute_loss(criterion, logits, sample)
        empty_loss.backward()
        self.assertEqual(float(empty_loss.detach()), 0.0)
        self.assertTrue((logits.grad == 0).all())

    def test_compiled_decoder_matches_eager_and_keeps_encoder_frozen(self):
        encoder = build_backbone(dict(
            model_type="v3", input_channels=1, global_crops_size=(16, 16, 16),
            local_crops_size=(16, 16, 16), embedding_type="deeper", patch_size=(8, 8, 8),
            embed_dim=36, depth=1, num_heads=3, mlp_ratio=2, num_reg_tokens=0,
            deeper_embed_stage_channels=[4, 8, 12, 16], deeper_embed_stage_strides=[1, 2, 2, 2],
            deeper_embed_depth_per_level=[1, 1, 1, 1], deeper_embed_stem_channels=4,
            attention_backend="math", drop_path_rate=0.5))
        model = SurfaceModel(encoder, blocks=1, vesuvius_source=self.source).train()
        before = {k: v.clone() for k, v in encoder.state_dict().items()}
        opt = torch.optim.AdamW(model.decoder.parameters(), lr=1e-3)
        image = torch.randn(1, 1, 16, 16, 16)
        eager_logits = model(image)
        eager_logits.square().mean().backward()
        eager_grads = {k: v.grad.clone() for k, v in model.decoder.named_parameters() if v.grad is not None}
        opt.zero_grad(set_to_none=True)
        parameter_ids = [id(v) for v in model.parameters()]
        state_keys = list(model.state_dict())
        # AOTAutograd checks forward/backward parity on CPU; the real 192-cubed
        # CUDA smoke run separately exercises the production Inductor backend.
        model.compile(backend="aot_eager", dynamic=False)
        self.assertEqual(parameter_ids, [id(v) for v in model.parameters()])
        self.assertEqual(state_keys, list(model.state_dict()))
        logits = model(image)
        self.assertEqual(tuple(logits.shape), (1, 2, 16, 16, 16))
        torch.testing.assert_close(logits, eager_logits)
        logits.square().mean().backward()
        for k, v in model.decoder.named_parameters():
            if k in eager_grads:
                torch.testing.assert_close(v.grad, eager_grads[k])
        self.assertTrue(any(v.grad is not None for v in model.decoder.parameters()))
        self.assertFalse(encoder.training)
        self.assertTrue(all(v.grad is None and not v.requires_grad for v in encoder.parameters()))
        opt.step()
        for k, v in encoder.state_dict().items():
            torch.testing.assert_close(v, before[k], rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available() and cuda.is_available(), "CUDA device/build required")
    def test_cuda_reader_and_transform_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root / "source")
            prepare(root / "source", root / "data", chunk=8, workers=1)
            cpu = SurfaceData(root / "data", (16, 16, 16), device="cpu", normalization="zscore")
            gpu = SurfaceData(root / "data", (16, 16, 16), device="cuda:0", normalization="zscore")
            try:
                a, info_a = next(cpu.validation(seed=3, crops_per_case=1))
                b, info_b = next(gpu.validation(seed=3, crops_per_case=1))
                self.assertEqual(info_a, info_b)
                for key in a:
                    self.assertEqual(b[key].device.type, "cuda")
                    torch.testing.assert_close(a[key], b[key].cpu(), atol=1e-5, rtol=1e-5)
                batch = gpu.training_batch(np.random.default_rng(5), 1)
                self.assertTrue(torch.isfinite(batch["image"]).all())
                self.assertTrue((batch["label_skel"][batch["label"] == 2] == 0).all())
            finally:
                cpu.close()
                gpu.close()


if __name__ == "__main__":
    unittest.main()
