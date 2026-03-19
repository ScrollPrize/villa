from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data.distributed import DistributedSampler
from PIL import Image

MODULE_PATH = Path(__file__).with_name("distributed_utils.py")
MODULE_SPEC = importlib.util.spec_from_file_location("dinov2_distributed_utils", MODULE_PATH)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise RuntimeError(f"Unable to load distributed utils from {MODULE_PATH}")
distributed_utils = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(distributed_utils)

build_distributed_sampler = distributed_utils.build_distributed_sampler
resolve_distributed_config = distributed_utils.resolve_distributed_config

PACKAGE_ROOT = Path(__file__).resolve().parents[6]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.append(str(PACKAGE_ROOT))

from vesuvius.models.training.trainers.self_supervised.dinov2 import pretrain as pretrain_module


class _FakeModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def get_params_groups(self, **kwargs):
        return [{"params": [self.linear.weight, self.linear.bias]}]


class _FakeLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.register_buffer("center", torch.zeros(1))


class _IdentityDDP:
    def __init__(self, module, **kwargs):
        self.module = module

    def __getattr__(self, name):
        return getattr(self.module, name)


class _FakeWandb:
    def __init__(self) -> None:
        self.init_calls: list[dict[str, object]] = []
        self.log_calls: list[tuple[dict[str, object], int | None]] = []
        self.finish_calls = 0
        self.run: object | None = None

    def init(self, **kwargs) -> None:
        self.init_calls.append(kwargs)
        self.run = object()

    def log(self, payload, step=None) -> None:
        self.log_calls.append((dict(payload), step))

    def finish(self) -> None:
        self.finish_calls += 1
        self.run = None

    def Image(self, value):
        return {"image": value}


class _FakeForwardModel:
    def eval(self):
        return self

    def __call__(self, *, student_input, **kwargs):
        batch = int(student_input.shape[0])
        return {
            "student": {
                "patch_tokens": torch.zeros((batch, 1, 3), dtype=torch.float32),
            }
        }


class _FakeDataset:
    def __init__(self, config, do_augmentations=True):
        self.config = config
        self.global_crop_size = tuple(config.get("global_crop_size", (16, 16, 16)))

    def __len__(self):
        return 8

    def __getitem__(self, index):
        return index


class PretrainDistributedHelpersTest(unittest.TestCase):
    def _minimal_config(self, **overrides):
        config = {
            "output_dir": "/tmp/dinov2_test",
            "dataset": {"crop_size": [16, 16, 16], "global_crop_size": [16, 16, 16]},
            "model": {"dino_out_dim": 16},
        }
        config.update(overrides)
        return config

    def test_resolve_distributed_config_defaults(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            config = resolve_distributed_config({})

        self.assertEqual(
            config,
            {
                "use_ddp": False,
                "world_size": 1,
                "rank": 0,
                "local_rank": 0,
            },
        )

    def test_resolve_distributed_config_from_torchrun_env(self) -> None:
        with patch.dict(
            os.environ,
            {
                "WORLD_SIZE": "4",
                "RANK": "2",
                "LOCAL_RANK": "1",
            },
            clear=True,
        ):
            config = resolve_distributed_config({})

        self.assertEqual(
            config,
            {
                "use_ddp": True,
                "world_size": 4,
                "rank": 2,
                "local_rank": 1,
            },
        )

    def test_build_distributed_sampler_disabled_returns_none(self) -> None:
        dataset = TensorDataset(torch.arange(8))

        sampler = build_distributed_sampler(
            dataset,
            is_distributed=False,
            rank=0,
            world_size=1,
            shuffle=False,
        )

        self.assertIsNone(sampler)

    def test_build_distributed_sampler_configures_partitioning(self) -> None:
        dataset = TensorDataset(torch.arange(8))

        sampler = build_distributed_sampler(
            dataset,
            is_distributed=True,
            rank=1,
            world_size=4,
            shuffle=True,
        )

        self.assertIsInstance(sampler, DistributedSampler)
        self.assertEqual(sampler.rank, 1)
        self.assertEqual(sampler.num_replicas, 4)
        self.assertTrue(sampler.shuffle)
        self.assertTrue(sampler.drop_last)

    def test_pretrainer_ddp_respects_explicit_cpu_device(self) -> None:
        config = self._minimal_config(use_ddp=True, device="cpu")

        with (
            patch.object(pretrain_module, "DinoVitStudentTeacher", _FakeModel),
            patch.object(pretrain_module, "DINOLoss", _FakeLoss),
            patch.object(pretrain_module, "iBOTPatchLoss", _FakeLoss),
            patch.object(pretrain_module, "KoLeoLoss", _FakeLoss),
            patch.object(pretrain_module, "DDP", _IdentityDDP),
            patch.object(pretrain_module.dist, "is_initialized", return_value=True),
            patch.object(pretrain_module.dist, "get_rank", return_value=1),
            patch.object(pretrain_module.dist, "get_world_size", return_value=2),
            patch.object(pretrain_module.torch.cuda, "is_available", return_value=True),
            patch.object(pretrain_module.torch.cuda, "set_device") as mock_set_device,
        ):
            trainer = pretrain_module.DinoIBOTPretrainer(config)

        self.assertEqual(trainer.device.type, "cpu")
        mock_set_device.assert_not_called()

    def test_pretrainer_initializes_wandb_from_hyphenated_config_keys(self) -> None:
        fake_wandb = _FakeWandb()
        config = self._minimal_config(
            device="cpu",
            **{
                "wandb-project": "dinov2-pretrain",
                "wandb-entity": "scroll-prize",
                "wandb-run-name": "run-001",
            },
        )

        with (
            patch.object(pretrain_module, "DinoVitStudentTeacher", _FakeModel),
            patch.object(pretrain_module, "DINOLoss", _FakeLoss),
            patch.object(pretrain_module, "iBOTPatchLoss", _FakeLoss),
            patch.object(pretrain_module, "KoLeoLoss", _FakeLoss),
            patch.dict(sys.modules, {"wandb": fake_wandb}),
        ):
            trainer = pretrain_module.DinoIBOTPretrainer(config)

        self.assertEqual(len(fake_wandb.init_calls), 1)
        self.assertEqual(fake_wandb.init_calls[0]["project"], "dinov2-pretrain")
        self.assertEqual(fake_wandb.init_calls[0]["entity"], "scroll-prize")
        self.assertEqual(fake_wandb.init_calls[0]["name"], "run-001")
        trainer._finish_wandb()
        self.assertEqual(fake_wandb.finish_calls, 1)

    def test_pretrainer_uses_dinov2_reference_optimizer_defaults(self) -> None:
        config = self._minimal_config(
            device="cpu",
            max_iterations=100,
        )

        with (
            patch.object(pretrain_module, "DinoVitStudentTeacher", _FakeModel),
            patch.object(pretrain_module, "DINOLoss", _FakeLoss),
            patch.object(pretrain_module, "iBOTPatchLoss", _FakeLoss),
            patch.object(pretrain_module, "KoLeoLoss", _FakeLoss),
        ):
            trainer = pretrain_module.DinoIBOTPretrainer(config)

        self.assertEqual(trainer.patch_embed_lr_mult, 0.2)
        self.assertEqual(trainer.freeze_last_layer_steps, 1)
        self.assertTrue(np.allclose(trainer.last_layer_lr_schedule.schedule[:1], 0.0))

    def test_pretrainer_resolves_epoch_based_freeze_last_layer_when_epoch_length_is_configured(self) -> None:
        config = self._minimal_config(
            device="cpu",
            max_iterations=100,
            official_epoch_length=12,
            freeze_last_layer_epochs=1,
        )

        with (
            patch.object(pretrain_module, "DinoVitStudentTeacher", _FakeModel),
            patch.object(pretrain_module, "DINOLoss", _FakeLoss),
            patch.object(pretrain_module, "iBOTPatchLoss", _FakeLoss),
            patch.object(pretrain_module, "KoLeoLoss", _FakeLoss),
        ):
            trainer = pretrain_module.DinoIBOTPretrainer(config)

        self.assertEqual(trainer.freeze_last_layer_steps, 12)
        self.assertTrue(np.allclose(trainer.last_layer_lr_schedule.schedule[:12], 0.0))

    def test_save_monitor_image_renders_five_samples_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._minimal_config(device="cpu", output_dir=tmpdir, monitor_batch_size=2)

            with (
                patch.object(pretrain_module, "DinoVitStudentTeacher", _FakeModel),
                patch.object(pretrain_module, "DINOLoss", _FakeLoss),
                patch.object(pretrain_module, "iBOTPatchLoss", _FakeLoss),
                patch.object(pretrain_module, "KoLeoLoss", _FakeLoss),
            ):
                trainer = pretrain_module.DinoIBOTPretrainer(config)

            trainer.model = _FakeForwardModel()
            trainer._center_slice = types.MethodType(lambda self, volume: np.full((4, 4), 64, dtype=np.uint8), trainer)
            trainer._patch_pca_slice = types.MethodType(
                lambda self, patch_tokens, sample_index, target_hw: np.full((target_hw[0], target_hw[1], 3), 192, dtype=np.uint8),
                trainer,
            )

            monitor_batch = {
                "collated_global_crops": torch.zeros((14, 1, 4, 4), dtype=torch.float32),
                "n_global_views": 2,
                "batch_size": 7,
            }
            metrics = {
                "loss": 1.0,
                "dino_global_loss": 0.2,
                "dino_local_loss": 0.3,
                "ibot_loss": 0.4,
                "koleo_loss": 0.1,
            }

            image_path = trainer.save_monitor_image(monitor_batch, step=3, metrics=metrics)

            with Image.open(image_path) as image:
                self.assertEqual(image.size, (16, 20))

    def test_dataloader_config_supports_workers_and_prefetch_factor(self) -> None:
        config = self._minimal_config(
            device="cpu",
            val_dataset={"crop_size": [16, 16, 16], "global_crop_size": [16, 16, 16]},
            **{
                "dataloader-workers": 3,
                "prefetch-factor": 4,
            },
        )
        dataloader_calls = []

        def _fake_dataloader(*args, **kwargs):
            dataloader_calls.append(kwargs)
            return kwargs

        with (
            patch.object(pretrain_module, "DinoVitStudentTeacher", _FakeModel),
            patch.object(pretrain_module, "DINOLoss", _FakeLoss),
            patch.object(pretrain_module, "iBOTPatchLoss", _FakeLoss),
            patch.object(pretrain_module, "KoLeoLoss", _FakeLoss),
            patch.object(pretrain_module, "SSLZarrDataset", _FakeDataset),
            patch.object(pretrain_module, "build_dino_ibot_collate_fn", return_value=lambda batch: batch),
            patch.object(pretrain_module, "build_distributed_sampler", return_value=None),
            patch.object(pretrain_module, "DataLoader", side_effect=_fake_dataloader),
        ):
            trainer = pretrain_module.DinoIBOTPretrainer(config)
            trainer.build_dataloader()
            trainer.build_val_dataloader()

        self.assertEqual(len(dataloader_calls), 2)
        for kwargs in dataloader_calls:
            self.assertEqual(kwargs["num_workers"], 3)
            self.assertEqual(kwargs["prefetch_factor"], 4)

    def test_build_dataloader_defaults_dataset_epoch_length_from_max_iterations(self) -> None:
        config = self._minimal_config(
            device="cpu",
            max_iterations=321,
            dataset={"crop_size": [16, 16, 16], "global_crop_size": [16, 16, 16]},
        )
        dataset_configs = []

        class _RecordingDataset:
            def __init__(self, dataset_config, do_augmentations=True):
                dataset_configs.append(dict(dataset_config))
                self.global_crop_size = (16, 16, 16)

            def __len__(self):
                return 321

        with (
            patch.object(pretrain_module, "DinoVitStudentTeacher", _FakeModel),
            patch.object(pretrain_module, "DINOLoss", _FakeLoss),
            patch.object(pretrain_module, "iBOTPatchLoss", _FakeLoss),
            patch.object(pretrain_module, "KoLeoLoss", _FakeLoss),
            patch.object(pretrain_module, "SSLZarrDataset", _RecordingDataset),
            patch.object(pretrain_module, "build_dino_ibot_collate_fn", return_value=lambda batch: batch),
            patch.object(pretrain_module, "build_distributed_sampler", return_value=None),
            patch.object(pretrain_module, "DataLoader", return_value="loader"),
        ):
            trainer = pretrain_module.DinoIBOTPretrainer(config)
            trainer.build_dataloader()

        self.assertEqual(dataset_configs[0]["epoch_length"], 321)


if __name__ == "__main__":
    unittest.main()
