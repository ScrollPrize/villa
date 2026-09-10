from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import zarr

from dinovol_2.dataset.point_annotations import (
    load_point_collection,
    map_scale0_voxel_centers,
    xyz_to_zyx,
)
from dinovol_2.dataset.ssl_zarr_dataset import SSLZarrDataset, SampledCropRegion
from dinovol_2.augmentation.transforms.spatial.mirroring import MirrorTransform
from dinovol_2.augmentation.transforms.spatial.rot90 import Rot90Transform
from dinovol_2.loss import PointCosineLoss
from dinovol_2.ops.collate import build_dino_ibot_collate_fn
from dinovol_2.ops.point_embeddings import gather_variable_points, sample_normalized_patch_embeddings
from dinovol_2.pretrain import DinoIBOTPretrainer


def _write_points(path: Path, collections: list[list[list[float]]], *, version=1) -> Path:
    document = {
        "version": version,
        "collections": [
            {"points": [{"p": point, "wind_a": 99} for point in points]}
            for points in collections
        ],
    }
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _make_pyramid(root: Path) -> Path:
    path = root / "points.zarr"
    group = zarr.open_group(str(path), mode="w", zarr_format=2)
    group.create_array(
        "0", shape=(96, 96, 96), chunks=(24, 24, 24), dtype="float32", fill_value=1.0
    )
    group.create_array(
        "1", shape=(48, 48, 48), chunks=(24, 24, 24), dtype="float32", fill_value=1.0
    )
    return path


def _ddp_point_worker(rank: int, init_path: str, result_path: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=2,
    )
    try:
        if rank == 0:
            embeddings = torch.tensor([[1.0, 0.0]], requires_grad=True)
            labels = torch.tensor([0])
        else:
            embeddings = torch.tensor([[0.8, 0.2], [0.8, 0.6]], requires_grad=True)
            labels = torch.tensor([0, 1])
        gathered_embeddings, gathered_labels = gather_variable_points(embeddings, labels)
        result = PointCosineLoss(0.0)(gathered_embeddings, gathered_labels)
        result.loss.backward()
        if rank == 0:
            empty_embeddings = torch.empty((0, 2), requires_grad=True)
            empty_labels = torch.empty((0,), dtype=torch.long)
        else:
            empty_embeddings = torch.tensor([[1.0, 0.0], [0.8, 0.6]], requires_grad=True)
            empty_labels = torch.tensor([0, 1])
        zero_rank_embeddings, zero_rank_labels = gather_variable_points(empty_embeddings, empty_labels)
        zero_rank_result = PointCosineLoss(0.0)(zero_rank_embeddings, zero_rank_labels)
        zero_rank_result.loss.backward()
        if rank == 0:
            torch.save(
                {
                    "point_count": result.point_count,
                    "same_pairs": result.same_pair_count,
                    "different_pairs": result.different_pair_count,
                    "loss": float(result.loss.detach()),
                    "grad_finite": bool(torch.isfinite(embeddings.grad).all()),
                    "zero_rank_point_count": zero_rank_result.point_count,
                    "zero_rank_grad_finite": bool(torch.isfinite(empty_embeddings.grad).all()),
                },
                result_path,
            )
    finally:
        dist.destroy_process_group()


class PointAnnotationTests(unittest.TestCase):
    def test_version_one_parsing_xyz_conversion_and_pyramid_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            path = _write_points(
                Path(tempdir) / "points.json",
                [[[1, 2, 3]], [[5.5, 6.5, 7.5]]],
            )
            xyz = load_point_collection(path)
            self.assertEqual(xyz.shape, (2, 3))
            zyx = xyz_to_zyx(xyz)
            np.testing.assert_allclose(zyx[0], [3, 2, 1])
            mapped = map_scale0_voxel_centers(zyx, (100, 80, 60), (50, 40, 30))
            np.testing.assert_allclose(mapped[0], [1.25, 0.75, 0.25])

    def test_tilde_expansion_and_validation_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            _write_points(root / "valid.json", [[[1, 2, 3]]])
            with mock.patch.dict(os.environ, {"HOME": str(root)}):
                np.testing.assert_allclose(load_point_collection("~/valid.json"), [[1, 2, 3]])

            cases = {
                "unsupported.json": {"version": 2, "collections": []},
                "empty.json": {"version": 1, "collections": [{"points": []}]},
                "nonfinite.json": {
                    "version": 1,
                    "collections": [{"points": [{"p": [1, 2, float("nan")]}]}],
                },
            }
            for filename, document in cases.items():
                path = root / filename
                path.write_text(json.dumps(document), encoding="utf-8")
                with self.subTest(filename=filename), self.assertRaises(ValueError):
                    load_point_collection(path)
            malformed = root / "malformed.json"
            malformed.write_text("{", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_point_collection(malformed)
            with self.assertRaises(FileNotFoundError):
                load_point_collection(root / "missing.json")


class PointPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.zarr_path = _make_pyramid(self.root)
        self.fiber_path = _write_points(
            self.root / "fiber.json",
            [[[48, 48, 48], [45, 48, 48], [51, 48, 48]]],
        )
        self.surface_path = _write_points(
            self.root / "surface.json",
            [[[48, 45, 48], [48, 51, 48]]],
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _dataset_config(self, *, scale=0, max_points=64) -> dict:
        return {
            "epoch_length": 4,
            "vol_trim_pct": 1.0,
            "nonzero_threshold": 0.0,
            "global_crop_size": [16, 16, 16],
            "local_crop_size": [8, 8, 8],
            "source_sampling_size": [32, 32, 32],
            "global_crop_scale": [1.0, 1.0],
            "local_crop_scale": [1.0, 1.0],
            "num_local_crops": 1,
            "point_supervision": {
                "enabled": True,
                "sampling_probability": 1.0,
                "max_points_per_view": max_points,
            },
            "datasets": [{
                "volume_path": str(self.zarr_path),
                "volume_scale": scale,
                "point_collections": [
                    {"path": str(self.fiber_path), "type": "fiber"},
                    {"path": str(self.surface_path), "type": "surface"},
                ],
            }],
        }

    def test_point_centering_transform_retention_type_merge_and_cap(self) -> None:
        dataset = SSLZarrDataset(self._dataset_config(max_points=3), do_augmentations=True)
        try:
            self.assertEqual(dataset.point_type_to_id, {"fiber": 0, "surface": 1})
            sample = dataset[0]
            self.assertEqual(len(sample["global_point_coordinates"]), 2)
            for coordinates, type_ids in zip(
                sample["global_point_coordinates"], sample["global_point_type_ids"]
            ):
                self.assertGreaterEqual(coordinates.shape[0], 1)
                self.assertLessEqual(coordinates.shape[0], 3)
                self.assertTrue(torch.all(coordinates >= 0))
                self.assertTrue(torch.all(coordinates <= 15))
                self.assertEqual(coordinates.shape[0], type_ids.shape[0])
        finally:
            dataset.close()

    def test_rotation_mirroring_and_type_stratified_cap_coordinates(self) -> None:
        keypoints = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        rotation = Rot90Transform()
        rotated = rotation._apply_to_keypoints(
            keypoints,
            num_rot_per_combination=[1],
            axis_combinations=[[1, 2]],
            crop_shape=(16, 16, 16),
        )
        torch.testing.assert_close(rotated[0], torch.tensor([13.0, 1.0, 3.0]))
        mirrored = MirrorTransform((0, 1, 2))._apply_to_keypoints(
            rotated,
            axes=[2],
            crop_shape=(16, 16, 16),
        )
        torch.testing.assert_close(mirrored[0], torch.tensor([13.0, 1.0, 12.0]))

        dataset = SSLZarrDataset(self._dataset_config(max_points=3), do_augmentations=False)
        try:
            coordinates, types = dataset._filter_and_cap_view_points(
                torch.tensor([[8.0, 8.0, 8.0]] * 5),
                torch.tensor([0, 0, 0, 1, 1]),
            )
            self.assertEqual(coordinates.shape[0], 3)
            self.assertEqual(types[0].item(), 0)
            self.assertEqual(set(types.tolist()), {0, 1})
        finally:
            dataset.close()

    def test_scale_one_center_mapping_and_resize_coordinates(self) -> None:
        dataset = SSLZarrDataset(self._dataset_config(scale=1), do_augmentations=False)
        try:
            np.testing.assert_allclose(dataset.volumes[0].point_coordinates[0], [23.75] * 3)
            region = SampledCropRegion(starts=(2, 4, 6), shape=(8, 8, 8))
            mapped = dataset._map_points_to_view(np.asarray([[2.0, 7.5, 13.0]]), region, (16, 16, 16))
            torch.testing.assert_close(mapped, torch.tensor([[0.5, 7.5, 14.5]]))
        finally:
            dataset.close()

    def test_ragged_collation_unmasks_interpolation_support(self) -> None:
        empty = torch.empty((0, 3))
        samples = [
            {
                "global_views": [torch.zeros(1, 16, 16, 16), torch.zeros(1, 16, 16, 16)],
                "local_views": [],
                "global_point_coordinates": [torch.tensor([[7.5, 7.5, 7.5]]), empty],
                "global_point_type_ids": [torch.tensor([2]), torch.empty((0,), dtype=torch.long)],
            },
            {
                "global_views": [torch.zeros(1, 16, 16, 16), torch.zeros(1, 16, 16, 16)],
                "local_views": [],
                "global_point_coordinates": [empty, torch.tensor([[3.5, 3.5, 3.5], [12.5, 12.5, 12.5]])],
                "global_point_type_ids": [torch.empty((0,), dtype=torch.long), torch.tensor([0, 1])],
            },
        ]
        collate = build_dino_ibot_collate_fn({
            "global_crop_size": [16, 16, 16],
            "patch_size": [4, 4, 4],
            "mask_ratio_min_max": [1.0, 1.0],
            "mask_sample_probability": 1.0,
        })
        batch = collate(samples)
        self.assertEqual(batch["collated_point_rows"].tolist(), [0, 3, 3])
        self.assertEqual(batch["collated_point_type_ids"].tolist(), [2, 0, 1])
        from dinovol_2.ops.point_embeddings import interpolation_support_indices

        support = interpolation_support_indices(
            batch["collated_point_coordinates"], (4, 4, 4), (4, 4, 4)
        )
        self.assertFalse(bool(batch["collated_masks"][batch["collated_point_rows"][:, None], support].any()))
        self.assertEqual(batch["mask_indices_list"].numel(), int(batch["collated_masks"].sum()))

    def _trainer_config(self, *, embedding_type: str, output_name: str) -> dict:
        config = {
            "device": "cpu",
            "use_amp": False,
            "warmup_steps": 0,
            "max_iterations": 1,
            "batch_size": 1,
            "output_dir": str(self.root / output_name),
            "point_supervision": {
                "enabled": True,
                "sampling_probability": 1.0,
                "loss_weight": 0.05,
                "different_type_margin": 0.0,
                "max_points_per_view": 64,
            },
            "model": {
                "model_type": "v2",
                "input_channels": 1,
                "embedding_type": embedding_type,
                "global_crops_size": [16, 16, 16],
                "local_crops_size": [8, 8, 8],
                "patch_size": [4, 4, 4],
                "embed_dim": 48,
                "depth": 2,
                "num_heads": 4,
                "num_reg_tokens": 2,
                "dino_out_dim": 64,
                "ibot_out_dim": 64,
                "dino_head_hidden_dim": 64,
                "dino_head_bottleneck_dim": 48,
                "ibot_head_hidden_dim": 64,
                "ibot_head_bottleneck_dim": 48,
            },
            "dataset": self._dataset_config(),
        }
        config["dataset"].pop("point_supervision")
        return config

    def test_point_supervision_smoke_gradient_with_existing_losses(self) -> None:
        config = self._trainer_config(embedding_type="default", output_name="point_smoke")
        config["gram"] = {"enabled": True, "loss_weight": 2.0}
        trainer = DinoIBOTPretrainer(config)
        loader = trainer.build_dataloader()
        try:
            batch = next(iter(loader))
            self.assertGreater(batch["collated_point_coordinates"].shape[0], 0)
            self.assertTrue(torch.all(batch["collated_point_coordinates"] >= 0))
            self.assertTrue(torch.all(batch["collated_point_coordinates"] <= 15))
            report = trainer.verify_train_step(batch, 0)
            metrics = report["train_step"]["metrics"]
            self.assertTrue(report["forward"]["checks"]["all_passed"])
            self.assertGreater(report["forward"]["point_supervision"]["point_count"], 0)
            self.assertIn("point_same_type", report["forward"]["losses"])
            self.assertIn("point_different_type", report["forward"]["losses"])
            self.assertTrue(np.isfinite(metrics["point_loss"]))
            self.assertGreaterEqual(metrics["point_count"], 2)
            self.assertTrue(np.isfinite(metrics["dino_global_loss"]))
            self.assertTrue(np.isfinite(metrics["ibot_loss"]))
            self.assertTrue(np.isfinite(metrics["koleo_loss"]))
            self.assertTrue(np.isfinite(metrics["gram_loss"]))
        finally:
            trainer._close_dataloader(loader)
            trainer._close_auxiliary_datasets()

    def test_deeper_embedding_halo_alignment(self) -> None:
        config = self._dataset_config()
        config["global_view_size"] = [24, 24, 24]
        dataset = SSLZarrDataset(config, do_augmentations=False)
        try:
            self.assertGreater(dataset.global_view_size[0], 16)
            halo = torch.tensor(dataset.global_point_crop_offset)
            coordinates, types = dataset._filter_and_cap_view_points(
                torch.stack((halo + 1.25, halo + 14.0)),
                torch.tensor([0, 1]),
            )
            torch.testing.assert_close(coordinates, torch.tensor([[1.25] * 3, [14.0] * 3]))
            self.assertEqual(types.tolist(), [0, 1])
        finally:
            dataset.close()


class PointLossTests(unittest.TestCase):
    def test_type_balancing_margin_absent_groups_and_gradients(self) -> None:
        embeddings = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [0.6, 0.8]],
            requires_grad=True,
        )
        labels = torch.tensor([0, 0, 1, 1, 2])
        result = PointCosineLoss(different_type_margin=0.5)(embeddings, labels)
        self.assertEqual(result.same_pair_count, 2)
        self.assertEqual(result.different_pair_count, 8)
        self.assertAlmostEqual(float(result.same_type.detach()), 0.5, places=6)
        self.assertGreater(float(result.different_type.detach()), 0.0)
        result.loss.backward()
        self.assertTrue(torch.isfinite(embeddings.grad).all())

        singleton = PointCosineLoss()(torch.tensor([[1.0, 0.0]], requires_grad=True), torch.tensor([0]))
        self.assertEqual(float(singleton.loss.detach()), 0.0)
        self.assertEqual(singleton.same_pair_count, 0)
        self.assertEqual(singleton.different_pair_count, 0)

    def test_patch_embedding_trilinear_sampling(self) -> None:
        tokens = torch.arange(8, dtype=torch.float32).reshape(1, 8, 1).requires_grad_()
        sampled = sample_normalized_patch_embeddings(
            tokens + 1.0,
            torch.tensor([0]),
            torch.tensor([[1.5, 1.5, 1.5]]),
            (2, 2, 2),
            (2, 2, 2),
        )
        self.assertEqual(sampled.shape, (1, 1))
        sampled.sum().backward()
        self.assertTrue(torch.isfinite(tokens.grad).all())

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "Gloo is unavailable")
    def test_two_rank_uneven_autograd_gather_and_cross_rank_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            mp.spawn(
                _ddp_point_worker,
                args=(str(root / "init"), str(root / "result.pt")),
                nprocs=2,
                join=True,
            )
            result = torch.load(root / "result.pt", weights_only=True)
            self.assertEqual(result["point_count"], 3)
            self.assertEqual(result["same_pairs"], 1)
            self.assertEqual(result["different_pairs"], 2)
            self.assertTrue(result["grad_finite"])
            self.assertEqual(result["zero_rank_point_count"], 2)
            self.assertTrue(result["zero_rank_grad_finite"])
            self.assertTrue(np.isfinite(result["loss"]))


if __name__ == "__main__":
    unittest.main()
