from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import zarr
from torch.utils.data import DataLoader

from vesuvius.models.build.pretrained_backbones.dinovol_2_builder import build_dinovol_2_backbone
from vesuvius.models.run.inference import Inferer
from vesuvius.models.training.train import BaseTrainer


def _tiny_dinovol_model_config() -> dict:
    return {
        "model_type": "v2",
        "input_channels": 1,
        "global_crops_size": [16, 16, 16],
        "local_crops_size": [16, 16, 16],
        "patch_size": [8, 8, 8],
        "embed_dim": 48,
        "depth": 2,
        "num_heads": 4,
        "num_reg_tokens": 2,
        "mlp_ratio": 2.0,
        "drop_path_rate": 0.0,
        "qkv_fused": True,
    }


def _write_local_guide_checkpoint(path: Path) -> None:
    backbone = build_dinovol_2_backbone(_tiny_dinovol_model_config())
    teacher_state = {f"backbone.{key}": value.cpu() for key, value in backbone.state_dict().items()}
    checkpoint = {
        "config": {
            "model": _tiny_dinovol_model_config(),
            "dataset": {
                "global_crop_size": [16, 16, 16],
                "local_crop_size": [16, 16, 16],
            },
        },
        "teacher": teacher_state,
    }
    torch.save(checkpoint, path)


def _make_synthetic_dataset(root: Path) -> Path:
    data_root = root / "data"
    image_root = data_root / "images" / "volume1.zarr"
    label_root = data_root / "labels" / "volume1_ink.zarr"

    image_group = zarr.open_group(str(image_root), mode="w")
    label_group = zarr.open_group(str(label_root), mode="w")
    image_array = image_group.create_dataset("0", shape=(16, 16, 16), chunks=(16, 16, 16), dtype="float32")
    label_array = label_group.create_dataset("0", shape=(16, 16, 16), chunks=(16, 16, 16), dtype="uint8")

    coords = np.linspace(-1.0, 1.0, 16, dtype=np.float32)
    z = coords[:, None, None]
    y = coords[None, :, None]
    x = coords[None, None, :]
    image = np.exp(-(x * x + y * y + z * z) * 4.0).astype(np.float32)
    label = ((x * x + y * y + z * z) < 0.35).astype(np.uint8)

    image_array[:] = image
    label_array[:] = label
    return data_root


def _make_mgr(data_root: Path, guide_checkpoint: Path) -> SimpleNamespace:
    return SimpleNamespace(
        gpu_ids=None,
        use_ddp=False,
        targets={
            "ink": {
                "out_channels": 2,
                "activation": "none",
                "losses": [{"name": "nnUNet_DC_and_CE_loss", "weight": 1.0}],
            }
        },
        tr_configs={},
        model_name="guided_smoke",
        train_patch_size=(16, 16, 16),
        train_batch_size=1,
        in_channels=1,
        autoconfigure=False,
        enable_deep_supervision=False,
        spacing=(1.0, 1.0, 1.0),
        data_path=data_root,
        min_labeled_ratio=0.0,
        min_bbox_percent=0.0,
        skip_patch_validation=True,
        allow_unlabeled_data=False,
        normalization_scheme="zscore",
        intensity_properties={},
        skip_intensity_sampling=True,
        no_spatial_augmentation=True,
        no_scaling_augmentation=True,
        cache_valid_patches=False,
        dataset_config={},
        ome_zarr_resolution=0,
        valid_patch_find_resolution=0,
        bg_sampling_enabled=False,
        bg_to_fg_ratio=0.5,
        unlabeled_foreground_enabled=False,
        unlabeled_foreground_threshold=0.05,
        unlabeled_foreground_bbox_threshold=0.15,
        unlabeled_foreground_volumes=None,
        profile_augmentations=False,
        guide_loss_weight=0.5,
        guide_supervision_target="ink",
        train_num_dataloader_workers=0,
        model_config={
            "features_per_stage": [8, 16, 32],
            "n_stages": 3,
            "n_blocks_per_stage": [1, 1, 1],
            "n_conv_per_stage_decoder": [1, 1],
            "kernel_sizes": [[3, 3, 3], [3, 3, 3], [3, 3, 3]],
            "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2]],
            "basic_encoder_block": "ConvBlock",
            "basic_decoder_block": "ConvBlock",
            "bottleneck_block": "BasicBlockD",
            "separate_decoders": True,
            "guide_backbone": str(guide_checkpoint),
            "guide_freeze": True,
            "guide_tokenbook_sample_rate": 1.0,
            "input_shape": [16, 16, 16],
        },
    )


def test_guided_base_trainer_smoke_runs_two_training_steps(tmp_path: Path):
    data_root = _make_synthetic_dataset(tmp_path)
    guide_checkpoint = tmp_path / "guide_backbone.pt"
    _write_local_guide_checkpoint(guide_checkpoint)
    mgr = _make_mgr(data_root, guide_checkpoint)
    trainer = BaseTrainer(mgr=mgr, verbose=False)
    model = trainer._build_model().to(trainer.device)
    loss_fns = trainer._build_loss()
    dataset = trainer._configure_dataset(is_training=True)
    batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for _step in range(2):
        optimizer.zero_grad(set_to_none=True)
        _inputs, targets_dict, outputs = trainer._get_model_outputs(model, batch)
        loss, task_losses = trainer._compute_train_loss(outputs, targets_dict, loss_fns)
        assert torch.isfinite(loss)
        assert "guide_mask" in task_losses
        assert task_losses["guide_mask"] > 0.0
        loss.backward()
        optimizer.step()


def test_guided_checkpoint_roundtrip_preserves_plain_inference_forward(tmp_path: Path):
    data_root = _make_synthetic_dataset(tmp_path)
    guide_checkpoint = tmp_path / "guide_backbone.pt"
    _write_local_guide_checkpoint(guide_checkpoint)
    mgr = _make_mgr(data_root, guide_checkpoint)
    trainer = BaseTrainer(mgr=mgr, verbose=False)
    model = trainer._build_model()
    checkpoint_path = tmp_path / "guided_model.pth"
    torch.save({"model_config": model.final_config, "model": model.state_dict()}, checkpoint_path)

    output_dir = tmp_path / "inference_out"
    output_dir.mkdir()
    inferer = Inferer(
        model_path=str(checkpoint_path),
        input_dir=str(data_root / "images" / "volume1.zarr"),
        output_dir=str(output_dir),
        input_format="zarr",
        do_tta=False,
        device="cpu",
        num_dataloader_workers=0,
        model_type="train_py",
    )

    model_info = inferer._load_train_py_model(checkpoint_path)
    output = model_info["network"](torch.randn(1, 1, 16, 16, 16))

    assert isinstance(output, dict)
    assert set(output.keys()) == {"ink"}


def test_inferer_finalize_output_batch_returns_float16_numpy():
    inferer = object.__new__(Inferer)
    inferer.device = torch.device("cpu")

    output_batch = torch.randn(1, 2, 4, 4, 4, dtype=torch.float32)
    output_np = inferer._finalize_output_batch(output_batch)

    assert output_np.dtype == np.float16
    assert output_np.shape == (1, 2, 4, 4, 4)
