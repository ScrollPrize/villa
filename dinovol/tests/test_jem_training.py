"""JEM views, processing, model, objective integration, EMA, config and checkpoint contracts."""
from contextlib import nullcontext
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from test_cuda_pipeline import write_array
from dinovol_2.dataset.cuda_pipeline import BatchProcessor
from dinovol_2.dataset.jem_pipeline import JEM_EXCLUDED_OPERATIONS, patch_masks
from dinovol_2.dataset.ssl_zarr_dataset import SSLZarrDataset
from dinovol_2.loss.jem_losses import entropy, fp32_reductions
from dinovol_2.methods.jem import JemMethod
from dinovol_2.model.model import DINOHead
from dinovol_2.training.config import normalize_config

CUDA = torch.cuda.is_available()


def tiny_config(root, *, chunk=None, drop_path=0.0, batch=3, masking=None, scope="crop_type",
                weighting="uniform", backend="contraction", device="cpu", normalization="robust"):
    return normalize_config({
        "backbone": dict(embed_dim=64, depth=3, num_heads=4, patch_size=[4] * 3, embedding_type="deeper",
                         deeper_embed_stem_type="residual", deeper_embed_stem_channels=8,
                         deeper_embed_stage_channels=[8, 16], deeper_embed_stage_strides=[[2, 2, 2]] * 2,
                         deeper_embed_depth_per_level=[1, 1], global_crops_size=[16] * 3,
                         local_crops_size=[12] * 3, qk_norm=True, mlp_ratio=2.0, scale_mlp=False,
                         drop_path_rate=drop_path, class_token=False, untie_global_and_local_cls_norm=False,
                         rope_type="compass", attention_backend=None),
        "method": {"name": "jem", "global_batch_size": batch, "source_chunk_size": chunk, "categories": 16,
                   "head_hidden_dim": 64, "head_bottleneck_dim": 32, "structural_dim": 64,
                   "alignment_depths": [1, 2, 3], "structural_depths": [0, 1, 2],
                   "depth_weights": {"1": .25, "2": .25, "3": 1}, "regularizer_warmup_steps": 2,
                   "global_collapse_schedule": {"start": 1.0, "end": 0.1},
                   "structural_schedule": {"warm": 0.5, "final": 1.0},
                   "entropy_view_scope": scope, "structural_pair_weighting": weighting,
                   "structural_backend": backend, "structural_tile": 16, "masking": masking or {},
                   "views": {"source_size": 32, "teacher_size": 16, "global_size": 16, "local_size": 12}},
        "data": {"batch_size": batch, "num_workers": 0, "dataset": {
            "datasets": [{"volume_path": str(root / "volume"), "volume_scale": 0, "pixel_size_um": 2.0}],
            "vol_trim_pct": 1, "epoch_length": 8, "augmentation": "full", "normalization_scheme": normalization}},
        "runtime": {"max_iterations": 10, "use_amp": device == "cuda", "device": device,
                    "output_dir": str(root / "output"), "warmup_steps": 0},
        "optimizer": {"lr": 1e-3, "warmup_steps": 0},
    })


def build(root, *, seed=0, augment=True, device="cpu", **kwargs):
    config = tiny_config(root, device=device, **kwargs)
    torch.manual_seed(seed)
    method = JemMethod(config, device=device)
    dataset = SSLZarrDataset(method.dataset_config(config["data"]["dataset"], training=True),
                             do_augmentations=augment)
    return method, dataset, config


def batch_for(method, dataset, device, count=3, seed=0):
    np.random.seed(seed)
    samples = [dataset[i] for i in range(count)]
    return method.build_processor(dataset, device)(samples), samples


def run(method, batch, device="cpu"):
    autocast = lambda: torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda")
    method.model_module.student.zero_grad(set_to_none=True)
    output = method.forward_backward(batch, 0, autocast=autocast, sync_context=lambda sync=True: nullcontext(),
                                     backward=lambda loss: loss.backward())
    grads = {name: p.grad.detach().clone() for name, p in method.model_module.student.named_parameters()}
    return output, grads


class JemTestCase(unittest.TestCase):
    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.root = Path(self._directory.name)
        write_array(self.root / "volume", np.random.default_rng(0).integers(1, 255, (48,) * 3, dtype=np.uint8))

    def tearDown(self):
        self._directory.cleanup()


class ViewTests(JemTestCase):
    def _check_ids(self, device):
        ids = np.arange(48 ** 3, dtype=np.float32).reshape((48,) * 3)
        write_array(self.root / "ids", ids)
        config = tiny_config(self.root, device=device, normalization="none",
                             masking={"global": {"probability": 0.0}, "local": {"probability": 0.0}})
        config["data"]["dataset"]["datasets"][0]["volume_path"] = str(self.root / "ids")
        config["runtime"]["use_amp"] = False  # FP32 crop storage keeps voxel IDs exact
        method = JemMethod(config, device=device)
        dataset = SSLZarrDataset(method.dataset_config(config["data"]["dataset"], training=True),
                                 do_augmentations=False)
        batch, _ = batch_for(method, dataset, device)
        patch = 4

        def cells(image, grid):
            n = grid[0]
            return image.reshape(n, patch, n, patch, n, patch).permute(0, 2, 4, 1, 3, 5).reshape(n ** 3, -1)

        teacher = [cells(crop[0].float().cpu(), batch["teacher_grid"]) for crop in batch["teacher_crops"]]
        for kind in ("global", "local"):
            targets = batch[f"{kind}_targets"].cpu()
            counts = (targets >= 0).sum(-1)
            # Valid overlap after orientation equals the overlap accepted before orientation.
            self.assertTrue(torch.equal(counts, batch[f"{kind}_overlap"].cpu()))
            for row, crop in enumerate(batch[f"{kind}_crops"]):
                student = cells(crop[0].float().cpu(), batch[f"{kind}_grid"])
                matched = targets[row] >= 0
                t = teacher[int(batch[f"{kind}_teacher_rows"][row])][targets[row][matched]]
                s = student[matched]
                # Same source voxels in each matched cell (orders differ between orientations).
                self.assertTrue(torch.equal(s.sort(-1).values, t.sort(-1).values))
                # Unmatched student cells share no voxel with the teacher crop.
                teacher_ids = set(teacher[int(batch[f"{kind}_teacher_rows"][row])].flatten().tolist())
                outside = set(student[~matched].flatten().tolist())
                self.assertFalse(outside & teacher_ids)

    def test_exact_correspondence_cpu(self):
        self._check_ids("cpu")

    @unittest.skipUnless(CUDA, "CUDA")
    def test_exact_correspondence_cuda(self):
        self._check_ids("cuda")

    def test_orientations_are_independent_and_crops_native(self):
        method, dataset, _ = build(self.root)
        np.random.seed(1)
        orientations = []
        for _ in range(20):
            sample = dataset[0]
            plan = sample["jem_plan"]
            orientations += plan["teacher_orientations"].tolist() + plan["local_orientations"].tolist()
            for view in sample["teacher_views"] + sample["global_views"] + sample["local_views"]:
                self.assertEqual(tuple(view.raw.shape), tuple(view.target))  # no resampling
            self.assertEqual(int(plan["source_resamples"]), 0)
        self.assertGreater(len(set(orientations)), 30)

    def test_masks_and_withheld_values(self):
        method, dataset, _ = build(self.root)
        batch, _ = batch_for(method, dataset, "cpu")
        for kind, fraction, n in (("global", 0.55, 64), ("local", 0.40, 27)):
            masks = batch[f"{kind}_masks"]
            applied = batch[f"{kind}_mask_applied"]
            expected = math.floor(fraction * n + 0.5)
            self.assertTrue(torch.equal(masks.sum(-1)[applied], torch.full((int(applied.sum()),), expected)))
            self.assertEqual(int(masks[~applied].sum()), 0)
            grid = batch[f"{kind}_grid"]
            crops = batch[f"{kind}_crops"][:, 0]
            cells = crops.reshape(-1, grid[0], 4, grid[1], 4, grid[2], 4).permute(0, 1, 3, 5, 2, 4, 6)
            hidden = cells.reshape(len(crops), -1, 64)[masks.view(len(crops), -1)]
            self.assertEqual(float(hidden.abs().max()) if hidden.numel() else 0.0, 0.0)
        # Probability per view approaches 0.7, independent of the source.
        draws = patch_masks((8, 8, 8), torch.arange(4000), {"mode": "random", "probability": 0.7, "fraction": 0.4})
        self.assertAlmostEqual(float(draws["applied"].float().mean()), 0.7, delta=0.03)
        # Changing withheld voxel values before the same cell masking cannot change outputs.
        from vc_reader import views
        entry = method._chunk(batch, torch.arange(1))["global"]
        masks = entry["masks"]
        if not bool(masks.any()):
            return
        cells = masks.view(-1, 4, 4, 4).to(torch.uint8)
        voxel = masks.view(-1, 1, 4, 4, 4)
        for axis in (2, 3, 4):
            voxel = voxel.repeat_interleave(4, axis)
        withheld = entry["crops"].clone()
        withheld[voxel] = 123.0
        method.model.eval()
        with torch.no_grad():
            a = method.model_module.student_forward(views.mask_patch_cells(entry["crops"], cells, (4, 4, 4)),
                                                    masks, view_kind="global")
            b = method.model_module.student_forward(views.mask_patch_cells(withheld, cells, (4, 4, 4)),
                                                    masks, view_kind="global")
        for depth in a["logits"]:
            torch.testing.assert_close(a["logits"][depth], b["logits"][depth], rtol=0, atol=0)

    def test_jem_recipe_excludes_degradations_only_for_jem(self):
        method, dataset, _ = build(self.root)
        processor = method.build_processor(dataset, "cpu")
        recipe = processor.recipe((16,) * 3)
        for name in JEM_EXCLUDED_OPERATIONS:
            self.assertNotIn(name, recipe["intensity"])
        self.assertIn("MedianFilter", recipe["intensity"])
        self.assertIn("Sharpen", recipe["intensity"])
        dino = [name for name, _ in BatchProcessor({}, device="cpu").intensity_recipe((16,) * 3)]
        for name in JEM_EXCLUDED_OPERATIONS:
            self.assertIn(name, dino)

    def test_views_replay_alone_and_reordered(self):
        method, dataset, _ = build(self.root)
        np.random.seed(4)
        samples = [dataset[i] for i in range(3)]
        processor = method.build_processor(dataset, "cpu")
        together = processor(samples)
        reordered = processor([samples[2], samples[0], samples[1]])
        alone = processor([samples[1]])
        for kind in ("teacher", "global", "local"):
            crops = together[f"{kind}_crops"]
            n = len(crops) // 3
            for slot in range(n):
                torch.testing.assert_close(crops[slot * 3 + 1], alone[f"{kind}_crops"][slot], rtol=0, atol=0)
                torch.testing.assert_close(crops[slot * 3 + 1], reordered[f"{kind}_crops"][slot * 3 + 2], rtol=0, atol=0)


class ObjectiveTests(JemTestCase):
    def dense_reference(self, method, batch):
        """Full-batch objective with autograd through the marginals (no surrogate)."""
        recipe = method.recipe
        chunk = method._chunk(batch, torch.arange(batch["batch_size"]))
        teacher = method._teacher(chunk, nullcontext)
        outputs = method._student(chunk, structural=True)
        w_global, w_struct = method.regularizer_weights(method.successful_updates)
        n_views = method.views_per_source * method.local_batch_size
        loss = 0
        with fp32_reductions("cpu"):
            for kind in ("global", "local"):
                entry = chunk[kind]
                for depth in method.alignment_depths:
                    weight = method.depth_weights[depth]
                    alignment, *_ = method._categorical_terms(outputs[kind]["logits"][depth],
                                                              teacher["logits"][depth].float(), entry, [])
                    loss = loss + weight * alignment.sum() / n_views
                    log_q = method._log_q(outputs[kind]["logits"][depth])
                    for group in method.groups:
                        if group.view_type != kind:
                            continue
                        rows = torch.isin(entry["slots"], torch.tensor(group.slots))
                        a = entry["valid"][rows].float()
                        q = log_q[rows].exp()
                        M = a.sum(0)
                        mu = (q * a[..., None]).sum(0) / M.clamp_min(1)[:, None]
                        values = (entropy(log_q[rows]) * a).sum(0) / M.clamp_min(1) + torch.special.xlogy(mu, mu).sum(-1)
                        present = M > 0
                        loss = loss + w_global * weight * group.weight * values[present].mean()
                from dinovol_2.loss.jem_losses import nonaffine_target
                for depth, head in outputs[kind]["structural"].items():
                    targets = nonaffine_target(teacher["structural"][depth])
                    structural = method._structural(head, targets, entry, kind, 16)
                    loss = loss + w_struct * structural.sum() / (len(method.structural_depths) * n_views)
        method.model_module.student.zero_grad(set_to_none=True)
        loss.backward()
        return loss.detach(), {n: p.grad.clone() for n, p in method.model_module.student.named_parameters()}

    def test_single_and_chunked_match_dense_full_batch_gradient(self):
        for scope in ("crop_type", "slot"):
            reference_method, dataset, _ = build(self.root, seed=3, scope=scope)
            reference_method.successful_updates = 1  # nonzero structural weight
            batch, _ = batch_for(reference_method, dataset, "cpu")
            reference_loss, reference_grads = self.dense_reference(reference_method, batch)
            for chunk in (None, 1, 2):
                with self.subTest(scope=scope, chunk=chunk):
                    method, _, _ = build(self.root, seed=3, scope=scope, chunk=chunk)
                    method.successful_updates = 1
                    output, grads = run(method, batch)
                    torch.testing.assert_close(output.loss, reference_loss, rtol=1e-5, atol=1e-6)
                    for name, grad in reference_grads.items():
                        torch.testing.assert_close(grads[name], grad, rtol=2e-4, atol=1e-6, msg=name)

    def test_drop_path_replay_in_chunks(self):
        method, dataset, _ = build(self.root, drop_path=0.5, chunk=1)
        method.verify_replay = True
        batch, _ = batch_for(method, dataset, "cpu")
        method.train()
        output, grads = run(method, batch)
        self.assertEqual(float(output.metrics["replay_max_abs_diff"]), 0.0)
        self.assertTrue(all(torch.isfinite(g).all() for g in grads.values()))

    def test_distance_stratified_and_tiled_backends_run(self):
        for weighting, backend in (("distance_stratified", "contraction"), ("uniform", "tiled")):
            method, dataset, _ = build(self.root, weighting=weighting, backend=backend)
            method.successful_updates = 1
            batch, _ = batch_for(method, dataset, "cpu")
            if weighting == "distance_stratified":
                self.assertIn("global_lattice", batch)
            output, _ = run(method, batch)
            self.assertTrue(math.isfinite(float(output.loss)))
            self.assertGreater(float(output.metrics["structural_loss"]), 0.0)

    def test_mask_status_contributions_add_up(self):
        method, dataset, _ = build(self.root)
        method.successful_updates = 1
        batch, _ = batch_for(method, dataset, "cpu")
        output, _ = run(method, batch)
        m = output.metrics
        w_global, w_struct = method.regularizer_weights(1)
        for name, total in (("align", m["alignment_loss"]), ("global", w_global * m["global_collapse_loss"]),
                            ("struct", w_struct * m["structural_loss"])):
            torch.testing.assert_close(m[f"contrib_{name}_masked"] + m[f"contrib_{name}_unmasked"], total,
                                       rtol=1e-4, atol=1e-6)

    def test_linear_tokenizer_comparison(self):
        config = tiny_config(self.root)
        backbone = dict(config["backbone"], embedding_type="default")
        config = normalize_config({**config, "backbone": backbone})
        torch.manual_seed(0)
        method = JemMethod(config, device="cpu")
        embed = method.model_module.student.backbone.down_projection.proj
        self.assertEqual((tuple(embed.kernel_size), tuple(embed.stride), embed.bias is not None), ((4,) * 3, (4,) * 3, True))
        dataset = SSLZarrDataset(method.dataset_config(config["data"]["dataset"], training=True), do_augmentations=True)
        batch, _ = batch_for(method, dataset, "cpu")
        output, grads = run(method, batch)
        self.assertTrue(math.isfinite(float(output.loss)))
        self.assertTrue(all(torch.isfinite(g).all() for g in grads.values()))

    @unittest.skipUnless(CUDA, "CUDA")
    def test_cuda_bf16_step(self):
        method, dataset, _ = build(self.root, device="cuda", chunk=2)
        batch, _ = batch_for(method, dataset, "cuda")
        self.assertTrue(batch["global_crops"].is_cuda)
        output, grads = run(method, batch, device="cuda")
        self.assertTrue(math.isfinite(float(output.loss)))
        self.assertTrue(all(torch.isfinite(g).all() for g in grads.values()))


class ModelTests(JemTestCase):
    def test_heads_and_parameter_groups(self):
        method, _, _ = build(self.root)
        student = method.model_module.student
        self.assertEqual(set(student.keys()), {"backbone", "categorical_heads", "structural_heads"})
        self.assertEqual(set(method.model_module.teacher.keys()), {"backbone", "categorical_heads"})
        head = student.categorical_heads[0]
        self.assertEqual([n for n, _ in head.last_layer.named_parameters()], ["weight"])
        self.assertFalse(hasattr(head.last_layer, "parametrizations"))
        groups = method.parameter_groups()
        last = [g for g in groups if any(p is head.last_layer.weight for p in g["params"])]
        self.assertEqual(last[0]["wd_multiplier"], 1.0)
        big = DINOHead(256, 512, hidden_dim=1024, bottleneck_dim=256, last_layer_type="linear")
        for module in big.modules():
            if isinstance(module, torch.nn.Linear):
                self.assertAlmostEqual(float(module.weight.std()), 0.02, delta=0.002)
                if module.bias is not None:
                    self.assertEqual(float(module.bias.abs().max()), 0.0)
        # DINO keeps weight normalization; norm_last_layer only freezes its scale.
        for frozen in (False, True):
            dino = DINOHead(16, 8, hidden_dim=16, bottleneck_dim=4, norm_last_layer=frozen)
            scale = dino.last_layer.parametrizations.weight.original0
            self.assertEqual(scale.requires_grad, not frozen)

    def test_direct_weight_ema(self):
        method, _, _ = build(self.root)
        module = method.model_module
        student = module.student.categorical_heads[0].last_layer.weight
        teacher = module.teacher.categorical_heads[0].last_layer.weight
        with torch.no_grad():
            student.zero_(); student[0, :2] = torch.tensor([1.0, 0.0])
            teacher.zero_(); teacher[0, :2] = torch.tensor([0.0, 1.0])
        method.recipe["momentum_teacher"] = 0.5
        method.after_optimizer_step(0)
        torch.testing.assert_close(teacher[0, :2], torch.tensor([0.5, 0.5]))  # no row renormalization
        self.assertEqual(method.successful_updates, 1)
        for momentum, expected in ((1.0, [0.5, 0.5]), (0.0, [1.0, 0.0])):
            method.recipe["momentum_teacher"] = momentum
            method.after_optimizer_step(0)
            torch.testing.assert_close(teacher[0, :2], torch.tensor(expected))
        before = teacher.detach().clone()
        method.optimizer_step_skipped(0)
        torch.testing.assert_close(teacher, before)
        self.assertEqual(method.skipped_updates, 1)

    def test_scratch_initialization_copies_student(self):
        method, _, _ = build(self.root)
        module = method.model_module
        student = dict(module.student.named_parameters())
        for name, parameter in module.teacher.named_parameters():
            self.assertTrue(torch.equal(parameter, student[name]))
            self.assertFalse(parameter.requires_grad)

    def test_config_validation(self):
        with self.assertRaisesRegex(ValueError, "DINO settings"):
            normalize_config({"method": {"name": "jem", "teacher_temp": 0.07},
                              "backbone": {"class_token": False, "untie_global_and_local_cls_norm": False}})
        with self.assertRaisesRegex(ValueError, "class_token"):
            normalize_config({"method": {"name": "jem"}, "backbone": {"untie_global_and_local_cls_norm": False}})
        with self.assertRaisesRegex(ValueError, "axial RoPE"):
            normalize_config({"method": {"name": "jem"}, "backbone": {
                "class_token": False, "untie_global_and_local_cls_norm": False, "embed_dim": 512,
                "num_heads": 8, "rope_type": "axial"}})
        with self.assertRaisesRegex(ValueError, "RoPE coordinate augmentation"):
            normalize_config({"method": {"name": "jem"}, "backbone": {
                "class_token": False, "untie_global_and_local_cls_norm": False,
                "rope_kwargs": {"rescale_coords": 2.0}}})
        resolved = normalize_config({"method": {"name": "jem"}, "backbone": {
            "class_token": False, "untie_global_and_local_cls_norm": False}})["backbone"]["rope_kwargs"]
        self.assertEqual([resolved[k] for k in ("shift_coords", "jitter_coords", "rescale_coords")], [None] * 3)
        with self.assertRaises(NotImplementedError):
            normalize_config({"method": {"name": "jem", "auxiliaries": {"ink_weight": 0.1}},
                              "backbone": {"class_token": False, "untie_global_and_local_cls_norm": False}})
        config = tiny_config(self.root)
        config["data"]["batch_size"] = 2
        with self.assertRaisesRegex(ValueError, "global_batch_size"):
            JemMethod(config, device="cpu")


class CheckpointTests(JemTestCase):
    def test_fit_save_and_resume(self):
        from dinovol_2.training.trainer import Trainer
        config = tiny_config(self.root)
        config["runtime"].update(max_iterations=3, save_every_n=1, train_monitor_every_n=1, log_every=1)
        trainer = Trainer(config)
        trainer.fit()
        checkpoints = sorted((self.root / "output").glob("checkpoint_step_*.pt"))
        self.assertTrue(checkpoints)
        self.assertTrue(list((self.root / "output" / "monitor").glob("jem_monitor_step_*.png")))
        saved = torch.load(checkpoints[-1], map_location="cpu", weights_only=False)
        self.assertEqual(saved["method"]["name"], "jem")
        state = saved["method"]["state"]
        self.assertEqual(state["recipe"]["categories"], 16)
        resumed = Trainer({**config, "runtime": {**config["runtime"], "resume_from": str(checkpoints[-1])}})
        step = resumed.load_checkpoint(checkpoints[-1])
        self.assertEqual(resumed.method.successful_updates, state["successful_updates"])
        for name, value in resumed.method.model_module.teacher.state_dict().items():
            self.assertTrue(torch.equal(value, state["teacher"][name]), name)
        self.assertEqual(step, int(saved["step"]))
        # The exported evaluation backbone is the CLS-free teacher backbone.
        from dinovol_2.model.backbone import build_backbone
        from dinovol_2.training.checkpoint import backbone_state
        from dinovol_2.training.config import as_legacy_config
        backbone = build_backbone(as_legacy_config(saved["config"])["model"])
        weights, branch = backbone_state(saved)
        backbone.load_state_dict(weights, strict=True)
        self.assertEqual(branch, "teacher")
        self.assertEqual(backbone.num_class_tokens, 0)


if __name__ == "__main__":
    unittest.main()
