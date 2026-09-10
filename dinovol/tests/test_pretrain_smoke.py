from __future__ import annotations

import copy
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch
import zarr

from dinovol_2.loss import GramLoss
from dinovol_2.pretrain import DinoIBOTPretrainer
from dinovol_2.verify import build_verification_report


def _make_synthetic_zarr(root: Path, *, size: int = 224) -> Path:
    zarr_path = root / "synthetic.zarr"
    group = zarr.open_group(str(zarr_path), mode="w", zarr_format=2)
    array = group.create_array("0", shape=(size, size, size), chunks=(32, 32, 32), dtype="float32")

    coords = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    z = coords[:, None, None]
    y = coords[None, :, None]
    x = coords[None, None, :]
    volume = np.exp(-(z * z + y * y + x * x) * 6.0).astype(np.float32)
    volume += 0.15 * ((z + y + x + 3.0) / 6.0).astype(np.float32)
    array[:] = np.clip(volume, 0.0, None)
    return zarr_path


def _state_dict_l1_diff(first: torch.nn.Module, second: torch.nn.Module) -> float:
    diff = 0.0
    for first_tensor, second_tensor in zip(first.state_dict().values(), second.state_dict().values()):
        diff += float(torch.sum(torch.abs(first_tensor.detach().cpu() - second_tensor.detach().cpu())).item())
    return diff


class PretrainSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tempdir = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tempdir.name)
        cls.zarr_path = _make_synthetic_zarr(cls.root)
        cls._checkpoints: dict[str, Path] = {}

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tempdir.cleanup()

    @classmethod
    def _base_config(cls, *, embedding_type: str = "default", output_name: str = "run") -> dict:
        return {
            "device": "cpu",
            "use_amp": False,
            "warmup_steps": 0,
            "max_iterations": 2,
            "batch_size": 1,
            "lr": 1e-4,
            "output_dir": str(cls.root / output_name),
            "ibot_masked_loss_chunk_size": 256,
            "model": {
                "model_type": "v2",
                "input_channels": 1,
                "embedding_type": embedding_type,
                "global_crops_size": [32, 32, 32],
                "local_crops_size": [16, 16, 16],
                "patch_size": [8, 8, 8],
                "embed_dim": 72,
                "depth": 2,
                "num_heads": 6,
                "num_reg_tokens": 2,
                "dino_out_dim": 128,
                "ibot_out_dim": 128,
                "dino_head_hidden_dim": 128,
                "dino_head_bottleneck_dim": 72,
                "ibot_head_hidden_dim": 128,
                "ibot_head_bottleneck_dim": 72,
            },
            "dataset": {
                "epoch_length": 6,
                "vol_trim_pct": 1.0,
                "global_crop_size": [32, 32, 32],
                "local_crop_size": [16, 16, 16],
                "global_crop_scale": [0.5, 1.0],
                "local_crop_scale": [0.5, 1.0],
                "num_local_crops": 2,
                "source_sampling_size": [48, 48, 48],
                "datasets": [
                    {
                        "volume_path": str(cls.zarr_path),
                        "volume_scale": 0,
                    }
                ],
            },
        }

    @classmethod
    def _checkpoint_for(cls, embedding_type: str) -> Path:
        checkpoint = cls._checkpoints.get(embedding_type)
        if checkpoint is not None:
            return checkpoint

        config = cls._base_config(embedding_type=embedding_type, output_name=f"base_ckpt_{embedding_type}")
        trainer = DinoIBOTPretrainer(config)
        try:
            checkpoint = trainer.save_checkpoint(0)
        finally:
            trainer._close_auxiliary_datasets()
            trainer._finish_wandb()
        cls._checkpoints[embedding_type] = checkpoint
        return checkpoint

    @classmethod
    def _gram_config(
        cls,
        *,
        embedding_type: str = "default",
        output_name: str = "gram",
        refresh_every: int = 1,
    ) -> dict:
        config = cls._base_config(embedding_type=embedding_type, output_name=output_name)
        checkpoint = cls._checkpoint_for(embedding_type)
        config["model"]["pretrained_weights"] = str(checkpoint)
        config["model"]["pretrained_backbone_only"] = False
        config["gram"] = {
            "enabled": True,
            "teacher_checkpoint": str(checkpoint),
            "teacher_refresh_every": refresh_every,
            "loss_weight": 2.0,
        }
        return config

    @classmethod
    def _hr_config(cls, *, embedding_type: str, output_name: str) -> dict:
        config = cls._gram_config(embedding_type=embedding_type, output_name=output_name)
        config["dataset"] = copy.deepcopy(config["dataset"])
        config["dataset"]["variants"] = [
            {
                "ratio": 0.5,
                "global_crop_size": [32, 32, 32],
                "local_crop_size": [16, 16, 16],
                "gram_teacher_crop_size": [48, 48, 48],
                "source_sampling_size": [56, 56, 56],
            },
            {
                "ratio": 0.5,
                "global_crop_size": [48, 48, 48],
                "local_crop_size": [24, 24, 24],
                "gram_teacher_crop_size": [64, 64, 64],
                "source_sampling_size": [72, 72, 72],
            },
        ]
        return config

    def test_base_pretrain_smoke_step(self) -> None:
        report = build_verification_report(self._base_config(output_name="base_smoke"), use_amp=False)
        self.assertTrue(report["forward"]["checks"]["all_passed"])
        self.assertTrue(report["train_step"]["checks"]["all_passed"])
        self.assertEqual(report["forward"]["losses"]["gram"], 0.0)

    def test_gram_refinement_smoke_step(self) -> None:
        report = build_verification_report(self._gram_config(output_name="gram_smoke"), use_amp=False)
        self.assertTrue(report["forward"]["checks"]["all_passed"])
        self.assertTrue(report["train_step"]["checks"]["all_passed"])
        self.assertGreater(report["forward"]["losses"]["gram"], 0.0)
        self.assertIsNotNone(report["forward"]["batch"]["gram_teacher_crops"])

    def test_hr_mixed_resolution_default_smoke(self) -> None:
        trainer = DinoIBOTPretrainer(self._hr_config(embedding_type="default", output_name="hr_default"))
        dataloader = trainer.build_dataloader()
        try:
            iterator = iter(dataloader)
            shapes = {tuple(next(iterator)["collated_global_crops"].shape[2:]) for _ in range(8)}
            self.assertGreaterEqual(len(shapes), 2)
            report = trainer.verify_train_step(next(iterator), step=0)
            self.assertTrue(report["train_step"]["checks"]["all_passed"])
            self.assertGreater(report["forward"]["losses"]["gram"], 0.0)
        finally:
            trainer._close_dataloader(dataloader)
            trainer._close_auxiliary_datasets()
            trainer._finish_wandb()

    def test_hr_mixed_resolution_deeper_smoke(self) -> None:
        trainer = DinoIBOTPretrainer(self._hr_config(embedding_type="deeper", output_name="hr_deeper"))
        dataloader = trainer.build_dataloader()
        try:
            iterator = iter(dataloader)
            shapes = {tuple(next(iterator)["collated_global_crops"].shape[2:]) for _ in range(8)}
            self.assertGreaterEqual(len(shapes), 2)
            report = trainer.verify_train_step(next(iterator), step=0)
            self.assertTrue(report["train_step"]["checks"]["all_passed"])
            self.assertGreater(report["forward"]["losses"]["gram"], 0.0)
        finally:
            trainer._close_dataloader(dataloader)
            trainer._close_auxiliary_datasets()
            trainer._finish_wandb()

    def test_checkpoint_roundtrip_preserves_gram_teacher(self) -> None:
        config = self._gram_config(output_name="gram_resume", refresh_every=2)
        trainer = DinoIBOTPretrainer(config)
        dataloader = trainer.build_dataloader()
        try:
            batch = next(iter(dataloader))
            trainer.train_step(batch, step=0)
            self.assertGreater(_state_dict_l1_diff(trainer.gram_teacher_backbone, trainer.model_module.teacher.backbone), 0.0)
            checkpoint_path = trainer.save_checkpoint(0)
            preserved_gram_state = {
                key: value.detach().cpu().clone()
                for key, value in trainer.gram_teacher_backbone.state_dict().items()
            }
        finally:
            trainer._close_dataloader(dataloader)
            trainer._close_auxiliary_datasets()
            trainer._finish_wandb()

        resumed = DinoIBOTPretrainer(config)
        resumed_dataloader = resumed.build_dataloader()
        try:
            loaded_step = resumed.load_checkpoint(checkpoint_path)
            self.assertEqual(loaded_step, 0)
            for key, value in resumed.gram_teacher_backbone.state_dict().items():
                self.assertTrue(torch.equal(value.detach().cpu(), preserved_gram_state[key]))

            resumed_batch = next(iter(resumed_dataloader))
            metrics = resumed.train_step(resumed_batch, step=1)
            self.assertTrue(np.isfinite(metrics["loss"]))
            self.assertLess(_state_dict_l1_diff(resumed.gram_teacher_backbone, resumed.model_module.teacher.backbone), 1e-6)
        finally:
            resumed._close_dataloader(resumed_dataloader)
            resumed._close_auxiliary_datasets()
            resumed._finish_wandb()

    def test_resume_from_base_checkpoint_refreshes_gram_teacher_from_loaded_teacher(self) -> None:
        checkpoint = self._checkpoint_for("default")
        config = self._base_config(output_name="gram_resume_from_base")
        config["gram"] = {
            "enabled": True,
            "loss_weight": 2.0,
            "teacher_refresh_every": 10000,
        }
        config["resume"] = True
        config["auto_resume"] = False
        config["resume_from"] = str(checkpoint)
        trainer = DinoIBOTPretrainer(config)
        try:
            loaded_step = trainer.load_checkpoint(checkpoint)
            self.assertEqual(loaded_step, 0)
            self.assertLess(_state_dict_l1_diff(trainer.gram_teacher_backbone, trainer.model_module.teacher.backbone), 1e-6)
        finally:
            trainer._close_auxiliary_datasets()
            trainer._finish_wandb()

    def test_gram_loss_non_img_level_is_per_sample(self) -> None:
        loss_fn = GramLoss(apply_norm=False)
        student = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[2.0, 1.0], [0.5, 1.5]],
            ],
            dtype=torch.float32,
        )
        teacher = torch.tensor(
            [
                [[1.5, 2.5], [2.5, 3.5]],
                [[1.0, 0.0], [1.0, 2.0]],
            ],
            dtype=torch.float32,
        )

        actual = loss_fn(student, teacher, img_level=False)
        student_grams = torch.matmul(student.transpose(-1, -2), student)
        teacher_grams = torch.matmul(teacher.transpose(-1, -2), teacher)
        expected = torch.nn.functional.mse_loss(student_grams, teacher_grams)
        self.assertAlmostEqual(float(actual), float(expected), places=6)


if __name__ == "__main__":
    unittest.main()
