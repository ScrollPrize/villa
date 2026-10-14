from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import torch.nn.functional as F
import zarr
from scipy import ndimage as ndi

from dinovol_2.augmentation.transforms.spatial.mirroring import MirrorTransform
from dinovol_2.augmentation.transforms.spatial.rot90 import Rot90Transform
from dinovol_2.augmentation.transforms.spatial.transpose import TransposeAxesTransform
from dinovol_2.augmentation.transforms.utils.compose import ComposeTransforms
from dinovol_2.dataset.object_labels import (
     build_mesh_segment_from_arrays,
    crop_mesh_objects,
    crossing_fiber_pairs,
    decode_fiber_directions,
    distinct_fiber_pairs,
    fiber_objects,
    load_mesh_segments_cache,
    save_mesh_segments_cache,
    sheet_objects,
    stacked_sheet_pairs,
)
from dinovol_2.dataset.ssl_zarr_dataset import SSLZarrDataset
from dinovol_2.ops.point_embeddings import sample_grid_points
from dinovol_2.pretrain import DinoIBOTPretrainer

SIZE = 96


def _sheets(shape, bridge: bool = False) -> np.ndarray:
    """Vertical sheets (normal along x) at x = 30, 40 and 60, three voxels thick;
    optionally the first two are joined by a local bridge, as neighbouring wraps often are."""
    surface = np.zeros(shape, np.uint8)
    for x in (30, 40, 60):
        surface[:, :, x:x + 3] = 255
    if bridge:
        surface[2:6, 20:24, 30:43] = 255
    return surface


def _encode(direction_zyx) -> tuple[int, int]:
    z, y, x = direction_zyx
    sign = -1.0 if z < 0 else 1.0
    return int(round(x * sign * 127 + 128)), int(round(y * sign * 127 + 128))


def _make_volume(root: Path) -> dict:
    coords = np.arange(SIZE, dtype=np.float32)
    ramp = (1.0 + 0.5 * coords[:, None, None] + 0.3 * coords[None, :, None] + 0.2 * coords[None, None, :])
    image = zarr.open_group(str(root / "image.zarr"), mode="w", zarr_format=2)
    image.create_array("0", shape=ramp.shape, chunks=(32, 32, 32), dtype="float32")[:] = ramp
    surface = zarr.open_group(str(root / "surface.zarr"), mode="w", zarr_format=2)
    surface.create_array("0", shape=ramp.shape, chunks=(32, 32, 32), dtype="uint8")[:] = _sheets(ramp.shape)
    # Fibers at half resolution: walls of fibers running along y, every 6 voxels in x.
    half = SIZE // 2
    presence = np.zeros((half, half, half), np.uint8)
    presence[:, :, 4::6] = 255
    nx_value, ny_value = _encode((0.0, 1.0, 0.0))
    paths = {}
    for name, value in (("presence", presence), ("nx", np.full_like(presence, nx_value)),
                        ("ny", np.full_like(presence, ny_value))):
        group = zarr.open_group(str(root / f"{name}.zarr"), mode="w", zarr_format=2)
        group.create_array("1", shape=value.shape, chunks=(16, 16, 16), dtype="uint8")[:] = value
        paths[name] = str(root / f"{name}.zarr")
    # Cached mesh segments at x=30, 40, 60 with inklabels and supervision masks.
    u = np.arange(48, dtype=np.float32)[:, None] * 2.0
    v = np.arange(48, dtype=np.float32)[None, :] * 2.0
    z_grid = np.broadcast_to(u, (48, 48)).copy()
    y_grid = np.broadcast_to(v, (48, 48)).copy()
    inklabels = np.zeros((48, 48), dtype=np.uint8)
    supervision = np.full((48, 48), 255, dtype=np.uint8)
    inklabels[:24, :] = 255
    meshes = [
        build_mesh_segment_from_arrays(
            f"sheet_{x}",
            np.full((48, 48), float(x), dtype=np.float32),
            y_grid,
            z_grid,
            (SIZE, SIZE, SIZE),
            (SIZE, SIZE, SIZE),
            inklabels=inklabels,
            supervision=supervision,
        )
        for x in (30, 40, 60)
    ]
    mesh_cache_path = root / "meshes_cache.npz"
    save_mesh_segments_cache(mesh_cache_path, meshes)
    return {
        "volume_path": str(root / "image.zarr"),
        "volume_scale": 0,
        "surface_path": str(root / "surface.zarr"),
        "surface_level": 0,
        "mesh_cache_path": str(mesh_cache_path),
        "fiber_presence_path": paths["presence"],
        "fiber_nx_path": paths["nx"],
        "fiber_ny_path": paths["ny"],
        "fiber_level": 1,
        "fiber_factor": 2,
    }


class ObjectLabelTests(unittest.TestCase):
    def test_bridged_sheets_stay_separate_and_are_marked_stacked(self):
        labels = sheet_objects(_sheets((16, 96, 96), bridge=True))
        # The block holding the bridge is rejected; the other y block keeps every sheet.
        ids = {x: set(np.unique(labels[:, 70, x:x + 3])) - {0} for x in (30, 40, 60)}
        for x, found in ids.items():
            self.assertTrue(found, f"sheet at x={x} was not labeled")
        # No object spans two sheets.
        self.assertFalse(ids[30] & ids[40])
        self.assertFalse(ids[40] & ids[60])
        pairs = set(stacked_sheet_pairs(labels, rng=np.random.default_rng(0)))
        self.assertTrue(any((min(a, b), max(a, b)) in pairs for a in ids[30] for b in ids[40]))

    def test_crossing_fibers_are_split_and_marked_different(self):
        present = np.zeros((8, 40, 40), bool)
        direction = np.zeros((8, 40, 40, 3), np.float32)
        present[4, :, 20] = True
        direction[4, :, 20] = (0.0, 1.0, 0.0)
        present[4, 20, :] = True
        direction[4, 20, :] = (0.0, 0.0, 1.0)
        labels, directions = fiber_objects(present, direction, min_voxels=5)
        self.assertGreaterEqual(labels.max(), 2)
        along_y = set(np.unique(labels[4, :15, 20])) - {0}
        along_x = set(np.unique(labels[4, 20, :15])) - {0}
        self.assertFalse(along_y & along_x)
        pairs = set(crossing_fiber_pairs(directions))
        self.assertTrue(any((min(a, b), max(a, b)) in pairs for a in along_y for b in along_x))

    def test_fiber_direction_decoding(self):
        direction = np.array([0.6, 0.0, 0.8])
        nx, ny = _encode(direction)
        decoded = decode_fiber_directions(np.array([nx], np.uint8), np.array([ny], np.uint8))[0]
        np.testing.assert_allclose(np.abs(decoded @ direction), 1.0, atol=2e-2)

    def test_object_loss_is_float32_under_half_precision_autocast(self):
        from dinovol_2.loss import ObjectLoss

        torch.manual_seed(0)
        views, batch, points, objects_count = 2, 2, 12, 4
        dense = F.normalize(torch.randn(views * batch, points, 8), dim=-1)
        tokens = torch.randn(views * batch, points, 6)
        objects = torch.randint(0, objects_count + 1, (batch, points))
        negatives = torch.zeros(batch, objects_count + 1, objects_count + 1, dtype=torch.bool)
        negatives[:, 1, 2] = negatives[:, 2, 1] = True
        head = torch.nn.Linear(6, 5)
        inputs = dict(
            student_dense=dense, teacher_dense=dense.flip(0), student_tokens=tokens, teacher_tokens=tokens,
            student_valid=torch.ones(views * batch, points, dtype=torch.bool),
            teacher_valid=torch.ones(views * batch, points, dtype=torch.bool),
            objects=objects, negatives=negatives,
            student_region_head=head, teacher_region_head=head,
        )
        reference = ObjectLoss()(**inputs)
        with torch.autocast("cpu", dtype=torch.float16):
            result = ObjectLoss()(**inputs)
        for name in ("point", "object", "region"):
            self.assertEqual(getattr(result, name).dtype, torch.float32)
            torch.testing.assert_close(getattr(result, name), getattr(reference, name))

    def test_point_term_only_pulls_corresponding_points(self):
        from dinovol_2.loss import ObjectLoss

        views, batch, points = 2, 1, 10
        # Every point has the same embedding: nothing separates neighbours, and the
        # pull-only term must not penalize that.
        dense = F.normalize(torch.ones(views * batch, points, 8), dim=-1)
        inputs = dict(
            student_dense=dense, teacher_dense=dense, student_tokens=torch.zeros(views * batch, points, 6),
            teacher_tokens=torch.zeros(views * batch, points, 6),
            student_valid=torch.ones(views * batch, points, dtype=torch.bool),
            teacher_valid=torch.ones(views * batch, points, dtype=torch.bool),
            objects=torch.zeros(batch, points, dtype=torch.long),
            negatives=torch.zeros(batch, 3, 3, dtype=torch.bool),
            student_region_head=torch.nn.Linear(6, 4), teacher_region_head=torch.nn.Linear(6, 4),
        )
        result = ObjectLoss()(**inputs)
        torch.testing.assert_close(result.point, torch.tensor(0.0))
        self.assertEqual(int(result.point_anchors), views * batch * points)
        # Opposite embeddings at corresponding points give the maximum, 2.
        result = ObjectLoss()(**{**inputs, "teacher_dense": -dense})
        torch.testing.assert_close(result.point, torch.tensor(2.0))

    def test_grid_sampling_matches_grid_sample(self):
        torch.manual_seed(0)
        grid = torch.randn(2, 5, 6, 7, 3)
        coordinates = torch.rand(2, 9, 3) * torch.tensor([19.0, 23.0, 27.0])
        input_shape = (20, 24, 28)
        result = sample_grid_points(grid, coordinates, input_shape)
        normalized = (coordinates + 0.5) / torch.tensor(input_shape, dtype=torch.float32) * 2 - 1
        expected = F.grid_sample(
            grid.permute(0, 4, 1, 2, 3), normalized.flip(-1)[:, :, None, None], align_corners=False,
            padding_mode="border",
        )[..., 0, 0].transpose(1, 2)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_fiber_dedup_within_1_5_voxels_and_orientation_agnostic_negatives(self):
        present = np.zeros((16, 40, 40), bool)
        direction = np.zeros((16, 40, 40, 3), np.float32)
        # Fiber A at z=4, x=10 along y (36 voxels).
        present[4, 2:38, 10] = True
        direction[4, 2:38, 10] = (0.0, 1.0, 0.0)
        # Duplicate fiber at z=5, x=11 (diagonal in 3D so not 6-connected; distance = sqrt(2) ~= 1.414 <= 1.5 voxels).
        present[5, 5:25, 11] = True
        direction[5, 5:25, 11] = (0.0, 1.0, 0.0)
        # Distinct parallel Fiber B at z=4, x=25 along y (36 voxels).
        present[4, 2:38, 25] = True
        direction[4, 2:38, 25] = (0.0, 1.0, 0.0)
        labels, directions = fiber_objects(present, direction, min_voxels=5, dedup_distance=1.5)
        id_a = set(np.unique(labels[4, 2:38, 10])) - {0}
        id_dup = set(np.unique(labels[5, 5:25, 11])) - {0}
        id_b = set(np.unique(labels[4, 2:38, 25])) - {0}
        self.assertEqual(len(id_a), 1)
        self.assertEqual(len(id_b), 1)
        # The duplicate fiber within 1.5 voxels average distance is deduplicated (cleared to 0).
        self.assertEqual(len(id_dup), 0)
        # Parallel distinct fibers A and B are marked as negative pairs regardless of orientation.
        pairs = set(distinct_fiber_pairs(directions))
        a, b = next(iter(id_a)), next(iter(id_b))
        self.assertIn((min(a, b), max(a, b)), pairs)

    def test_mesh_uv_components_dedup_and_ink_supervision(self):
        # Create a 2-wrap spiral mesh: two bands in v separated by a middle band that exits the crop.
        u = np.arange(20, dtype=np.float32)[:, None]
        v = np.arange(30, dtype=np.float32)[None, :]
        z = np.broadcast_to(10.0 + u * 1.5, (20, 30)).copy()
        y = np.broadcast_to(10.0 + (v % 10) * 2.0, (20, 30)).copy()
        x = np.zeros((20, 30), dtype=np.float32)
        # Wrap 1 at x=15 (v in 0..9), middle band outside crop at x=200 (v in 10..19), Wrap 2 at x=35 (v in 20..29).
        x[:, :10] = 15.0
        x[:, 10:20] = 200.0
        x[:, 20:] = 35.0
        inklabels = np.zeros((20, 30), dtype=np.uint8)
        supervision = np.zeros((20, 30), dtype=np.uint8)
        supervision[:, :10] = 255
        inklabels[:10, :10] = 255  # Top half of Wrap 1 is confirmed ink (+1), bottom half is confirmed non-ink (-1).
        seg_main = build_mesh_segment_from_arrays(
            "spiral_seg", x, y, z, inklabels=inklabels, supervision=supervision
        )
        # Duplicate mesh shifted by 0.8 voxels in x on Wrap 1 (fewer points so seg_main wins).
        seg_dup = build_mesh_segment_from_arrays(
            "dup_seg", x[:15, :10] + 0.8, y[:15, :10], z[:15, :10]
        )
        sheets, neg_pairs = crop_mesh_objects(
            [seg_main, seg_dup],
            np.array([0.0, 0.0, 0.0]),
            np.array([64.0, 64.0, 64.0]),
            min_points=10,
            dedup_distance=1.5,
            min_separation=3.0,
            rng=np.random.default_rng(0),
        )
        # Wrap 1 and Wrap 2 are split into 2 distinct sheets; seg_dup is deduplicated.
        self.assertEqual(len(sheets), 2)
        self.assertEqual(neg_pairs, [(1, 2)])
        ink_states_0 = set(np.unique(sheets[0][1]).tolist())
        ink_states_1 = set(np.unique(sheets[1][1]).tolist())
        self.assertEqual({frozenset(ink_states_0), frozenset(ink_states_1)}, {frozenset({-1, 1}), frozenset({0})})

    def test_supcon_l_out_margin_and_cross_batch_ink_loss(self):
        from dinovol_2.loss import ObjectLoss

        torch.manual_seed(0)
        views, batch, points = 2, 2, 8
        dense = F.normalize(torch.randn(views * batch, points, 8), dim=-1)
        ink_emb = F.normalize(torch.randn(views * batch, points, 6), dim=-1)
        tokens = torch.randn(views * batch, points, 6)
        objects = torch.tensor([
            [1, 1, 1, 1, 2, 2, 2, 2],
            [1, 1, 1, 1, 2, 2, 2, 2],
        ], dtype=torch.long)
        negatives = torch.zeros(batch, 3, 3, dtype=torch.bool)
        negatives[:, 1, 2] = negatives[:, 2, 1] = True
        ink = torch.tensor([
            [1, 1, -1, -1, 0, 0, 0, 0],
            [1, 1, -1, -1, 0, 0, 0, 0],
        ], dtype=torch.long)
        head = torch.nn.Linear(6, 5)
        loss_fn = ObjectLoss(pos_margin=0.75, ink_queue_size=16, ink_dim=6)
        result = loss_fn(
            student_dense=dense,
            teacher_dense=dense.flip(0),
            student_tokens=tokens,
            teacher_tokens=tokens,
            student_valid=torch.ones(views * batch, points, dtype=torch.bool),
            teacher_valid=torch.ones(views * batch, points, dtype=torch.bool),
            objects=objects,
            negatives=negatives,
            student_region_head=head,
            teacher_region_head=head,
            student_ink=ink_emb,
            teacher_ink=ink_emb.flip(0),
            ink=ink,
            n_views=views,
        )
        self.assertGreater(float(result.object), 0.0)
        self.assertGreater(float(result.ink), 0.0)
        # 2 positive (+1) and 2 negative (-1) supervised ink points per sample * 2 samples * 2 views = 16 anchors.
        self.assertEqual(int(result.ink_anchors), views * batch * 4)
        self.assertGreater(int(loss_fn.ink_pos_count.item()), 0)
        self.assertGreater(int(loss_fn.ink_neg_count.item()), 0)


class ObjectPointDatasetTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.volume = _make_volume(Path(self.tempdir.name))

    def tearDown(self):
        self.tempdir.cleanup()

    def dataset_config(self, **overrides):
        return {
            "epoch_length": 4, "vol_trim_pct": 1.0, "nonzero_threshold": 0.3,
            "global_crop_size": [32, 32, 32], "local_crop_size": [16, 16, 16],
            "global_crop_scale": [0.5, 1.0], "local_crop_scale": [0.3, 0.6],
            "num_local_crops": 1, "source_sampling_size": [64, 64, 64],
            "datasets": [self.volume],
            "object_points": {
                "enabled": True, "points_per_sample": 96, "max_labeled_points": 48,
                "points_per_object": 6, "max_objects": 12, "slab_thickness": 8,
            },
            **overrides,
        }

    def test_points_follow_crops_flips_rotations_and_transposes(self):
        # No intensity normalization: the robust scheme clips, which bends the ramp.
        dataset = SSLZarrDataset(self.dataset_config(normalization_scheme="none"), do_augmentations=True)
        spatial = ComposeTransforms([
            Rot90Transform(allowed_axes={0, 1, 2}),
            MirrorTransform(allowed_axes=(0, 1, 2)),
            TransposeAxesTransform(allowed_axes={0, 1, 2}),
        ])
        dataset.global_transforms = [spatial, spatial]
        for seed in range(4):
            np.random.seed(seed)
            torch.manual_seed(seed)
            self._check_points_follow_views(dataset[0])
        dataset.close()

    def _check_points_follow_views(self, sample):
        source = sample["object_point_source"].numpy()
        self.assertEqual(source.shape, (96, 3))
        self.assertGreater(int((sample["object_point_objects"] > 0).sum()), 0)
        negatives = sample["object_negatives"].numpy()
        np.testing.assert_array_equal(negatives, negatives.T)
        self.assertTrue(negatives.any(), "stacked sheets were not marked as different")
        # The ramp image (an affine function of position) must be perfectly correlated
        # with the source ramp at the mapped coordinates.
        window_ramp = 0.5 * source[:, 0] + 0.3 * source[:, 1] + 0.2 * source[:, 2]
        for view, coordinates, valid in zip(
            sample["global_views"], sample["object_point_coordinates"], sample["object_point_valid"]
        ):
            self.assertTrue(bool(valid.all()))
            coordinates = coordinates.numpy()
            interior = np.all((coordinates > 1) & (coordinates < 30), axis=1)
            values = ndi.map_coordinates(view[0].numpy(), coordinates[interior].T, order=1)
            self.assertGreater(np.corrcoef(values, window_ramp[interior])[0, 1], 0.999)

    def test_dino_augmentation_preset_keeps_spatial_and_intensity_jitter(self):
        dataset = SSLZarrDataset(self.dataset_config(augmentation="dino"), do_augmentations=True)
        transforms = [getattr(t, "transform", t) for t in dataset.global_transforms[0].transforms]
        names = {type(t).__name__ for t in transforms}
        self.assertEqual(names, {
            "Rot90Transform", "MirrorTransform", "TransposeAxesTransform", "OneOfTransform",
            "BrightnessAdditiveTransform",
        })
        for transform in transforms:
            if isinstance(transform, (Rot90Transform, TransposeAxesTransform)):
                self.assertEqual(set(transform.allowed_axes), {0, 1, 2})
        dataset.close()


class ObjectLossTrainingTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.volume = _make_volume(self.root)

    def tearDown(self):
        self.tempdir.cleanup()

    def config(self, *, objects: bool, output: str):
        config = {
            "device": "cpu", "use_amp": False, "warmup_steps": 0, "max_iterations": 4, "batch_size": 2,
            "lr": 1e-3, "output_dir": str(self.root / output), "ibot_masked_loss_chunk_size": 256,
            "koleo_loss_group_size": None,
            "model": {
                "model_type": "v2", "input_channels": 1, "embedding_type": "deeper",
                "global_crops_size": [32, 32, 32], "local_crops_size": [16, 16, 16], "patch_size": [8, 8, 8],
                "embed_dim": 72, "depth": 2, "num_heads": 6, "num_reg_tokens": 2,
                "dino_out_dim": 128, "ibot_out_dim": 128, "dino_head_hidden_dim": 128,
                "dino_head_bottleneck_dim": 72, "ibot_head_hidden_dim": 128, "ibot_head_bottleneck_dim": 72,
                "deeper_embed_stage_channels": [8, 16, 32, 48], "deeper_embed_stage_strides": [1, 2, 2, 2],
                "deeper_embed_depth_per_level": [1, 1, 1, 1], "deeper_embed_final_projection": True,
            },
            "dataset": {
                "epoch_length": 8, "vol_trim_pct": 1.0, "global_crop_size": [32, 32, 32],
                "local_crop_size": [16, 16, 16], "global_crop_scale": [0.5, 1.0], "local_crop_scale": [0.3, 0.6],
                "num_local_crops": 1, "source_sampling_size": [64, 64, 64], "datasets": [self.volume],
                "augmentation": "dino",
                "object_points": {
                    "enabled": True, "points_per_sample": 96, "max_labeled_points": 48,
                    "points_per_object": 6, "max_objects": 12, "slab_thickness": 8,
                },
            },
        }
        if objects:
            config["object_losses"] = {"enabled": True, "head": {"hidden_dim": 32, "out_dim": 16, "region_dim": 16}}
            config["grad_accumulation_steps"] = 2
        return config

    def test_accumulated_step_trains_dense_heads_and_cnn(self):
        torch.manual_seed(0)
        np.random.seed(0)
        trainer = DinoIBOTPretrainer(self.config(objects=True, output="objects"))
        loader = trainer.build_dataloader()
        try:
            iterator = iter(loader)
            batches = [next(iterator), next(iterator)]
            self.assertEqual(tuple(batches[0]["object_point_coordinates"].shape), (4, 96, 3))
            student = trainer.model_module.student
            before = {name: p.detach().clone() for name, p in student.named_parameters()}
            teacher_head = trainer.model_module.teacher.dense_head.mlp[0].weight.detach().clone()
            metrics = trainer.train_step(batches, 0)
            for key in ("object_point_loss", "object_loss", "object_region_loss", "object_ink_loss", "loss"):
                self.assertTrue(np.isfinite(metrics[key]), key)
            self.assertGreater(metrics["object_point_anchors"], 0)
            self.assertGreater(metrics["object_point_loss"], 0)
            self.assertGreater(metrics["object_ink_anchors"], 0)
            self.assertGreater(metrics["object_ink_loss"], 0)
            changed = {name for name, p in student.named_parameters() if not torch.equal(p, before[name])}
            self.assertTrue(any(name.startswith("dense_head.fpn_refines.") for name in changed))
            self.assertTrue(any(name.startswith("dense_head.ink_proj.") for name in changed))
            self.assertTrue(any(name.startswith("region_head.") for name in changed))
            self.assertTrue(any("down_projection.stages.0" in name for name in changed))
            self.assertFalse(torch.equal(trainer.model_module.teacher.dense_head.mlp[0].weight, teacher_head))
            # Monitor rows: input, then 4 token PCA sets, 4 dense-head PCA sets, and 1 ink PCA set, per view.
            from PIL import Image
            with Image.open(trainer.save_monitor_image(trainer._monitor_subset(batches[-1]), 1, metrics)) as image:
                self.assertEqual(image.size, (2 * (1 + 2 * 4 + 1) * 32, 2 * 32))
        finally:
            trainer._close_dataloader(loader)
            trainer._close_auxiliary_datasets()

    def test_resume_adds_heads_and_keeps_optimizer_state(self):
        torch.manual_seed(0)
        np.random.seed(0)
        base = DinoIBOTPretrainer(self.config(objects=False, output="base"))
        loader = base.build_dataloader()
        try:
            base.train_step(next(iter(loader)), 0)
        finally:
            base._close_dataloader(loader)
        checkpoint = base.save_checkpoint(0)
        reference = {
            name: base.optimizer.state[parameter]["exp_avg"].clone()
            for name, parameter in base.model_module.student.named_parameters()
            if parameter in base.optimizer.state
        }

        resumed = DinoIBOTPretrainer({**self.config(objects=True, output="resumed"), "resume_from": str(checkpoint)})
        self.assertEqual(resumed.load_checkpoint(checkpoint), 0)
        # The object losses ramp up from the step after the checkpoint.
        self.assertEqual(resumed.object_heads_start_step, 1)
        resumed.object_ramp_steps = 4
        self.assertEqual([resumed._object_loss_ramp(step) for step in (1, 2, 4, 9)], [0.25, 0.5, 1.0, 1.0])
        student = resumed.model_module.student
        teacher = resumed.model_module.teacher
        for name, parameter in student.named_parameters():
            if name in reference:
                torch.testing.assert_close(resumed.optimizer.state[parameter]["exp_avg"], reference[name])
            elif name.startswith(("dense_head.", "region_head.")):
                self.assertNotIn(parameter, resumed.optimizer.state)
        for key, value in base.model_module.student.backbone.state_dict().items():
            torch.testing.assert_close(student.backbone.state_dict()[key], value, msg=key)
        for (name, s_param), t_param in zip(student.dense_head.named_parameters(), teacher.dense_head.parameters()):
            torch.testing.assert_close(s_param, t_param, msg=name)
        resumed._close_auxiliary_datasets()
        base._close_auxiliary_datasets()

    def test_resume_axial_checkpoint_with_mixed_rope(self):
        torch.manual_seed(0)
        np.random.seed(0)
        axial = self.config(objects=True, output="axial")
        axial["model"]["rope_type"] = "axial"
        base = DinoIBOTPretrainer(axial)
        loader = base.build_dataloader()
        try:
            iterator = iter(loader)
            base.train_step([next(iterator), next(iterator)], 0)
        finally:
            base._close_dataloader(loader)
        checkpoint = base.save_checkpoint(0)
        reference = {
            name: base.optimizer.state[parameter]["exp_avg"].clone()
            for name, parameter in base.model_module.student.named_parameters()
            if parameter in base.optimizer.state
        }

        mixed = self.config(objects=True, output="mixed")
        mixed["model"].update(rope_type="mixed", rope_shift_coords=1.0, rope_jitter_coords=1.2, rope_rescale_coords=2.0)
        resumed = DinoIBOTPretrainer(mixed)
        self.assertEqual(resumed.load_checkpoint(checkpoint), 0)
        # Not an added head: the object-loss ramp keeps the checkpoint's start step.
        self.assertEqual(resumed.object_heads_start_step, 0)
        student = resumed.model_module.student
        mix_names = [name for name, _ in student.named_parameters() if name.endswith("rope_embed.mix_frequencies")]
        self.assertEqual(len(mix_names), 2)
        for name, parameter in student.named_parameters():
            if name in reference:
                torch.testing.assert_close(resumed.optimizer.state[parameter]["exp_avg"], reference[name], msg=name)
            else:
                self.assertIn(name, mix_names)
                self.assertNotIn(parameter, resumed.optimizer.state)
        # Warm start reproduces axial RoPE exactly, in both branches (no coordinate
        # augmentation outside training).
        x = torch.randn(2, 1, 32, 32, 32)
        for branch in ("student", "teacher"):
            old, new = getattr(base.model_module, branch).backbone, getattr(resumed.model_module, branch).backbone
            old.eval(), new.eval()
            with torch.no_grad():
                torch.testing.assert_close(
                    new(x)["x_norm_patchtokens"], old(x)["x_norm_patchtokens"], rtol=1e-4, atol=1e-5, msg=branch
                )
        resumed.model.train()
        before = {name: p.detach().clone() for name, p in student.named_parameters() if name in mix_names}
        loader = resumed.build_dataloader()
        try:
            iterator = iter(loader)
            metrics = resumed.train_step([next(iterator), next(iterator)], 1)
        finally:
            resumed._close_dataloader(loader)
        self.assertTrue(np.isfinite(metrics["loss"]))
        for name, parameter in student.named_parameters():
            if name in before:
                self.assertFalse(torch.equal(parameter, before[name]), name)
        resumed._close_auxiliary_datasets()
        base._close_auxiliary_datasets()


if __name__ == "__main__":
    unittest.main()
