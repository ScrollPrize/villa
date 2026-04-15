from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import zarr
from torch.utils.data import DataLoader

from vesuvius.models.build.pretrained_backbones.dinovol_2_builder import build_dinovol_2_backbone
from vesuvius.models.run.inference import Inferer
from vesuvius.models.training.train import BaseTrainer
from vesuvius.utils.plotting import save_debug


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
    label_root_ink = data_root / "labels" / "volume1_ink.zarr"
    label_root_surface = data_root / "labels" / "volume1_surface.zarr"

    image_group = zarr.open_group(str(image_root), mode="w")
    label_group_ink = zarr.open_group(str(label_root_ink), mode="w")
    label_group_surface = zarr.open_group(str(label_root_surface), mode="w")
    image_array = image_group.create_dataset("0", shape=(16, 16, 16), chunks=(16, 16, 16), dtype="float32")
    label_array_ink = label_group_ink.create_dataset("0", shape=(16, 16, 16), chunks=(16, 16, 16), dtype="uint8")
    label_array_surface = label_group_surface.create_dataset("0", shape=(16, 16, 16), chunks=(16, 16, 16), dtype="uint8")

    coords = np.linspace(-1.0, 1.0, 16, dtype=np.float32)
    z = coords[:, None, None]
    y = coords[None, :, None]
    x = coords[None, None, :]
    image = np.exp(-(x * x + y * y + z * z) * 4.0).astype(np.float32)
    label = ((x * x + y * y + z * z) < 0.35).astype(np.uint8)

    image_array[:] = image
    label_array_ink[:] = label
    label_array_surface[:] = label
    return data_root


def _make_mgr(
    data_root: Path,
    guide_checkpoint: Path,
    *,
    guide_fusion_stage: str | None = None,
    guide_loss_weight: float = 0.5,
    guide_tokenbook_prototype_weighting: str | None = None,
    losses: list[dict] | None = None,
) -> SimpleNamespace:
    guided_model_config = {
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
    }
    if guide_fusion_stage is not None:
        guided_model_config["guide_fusion_stage"] = str(guide_fusion_stage)
    if guide_tokenbook_prototype_weighting is not None:
        guided_model_config["guide_tokenbook_prototype_weighting"] = str(guide_tokenbook_prototype_weighting)

    return SimpleNamespace(
        gpu_ids=None,
        use_ddp=False,
        targets={
            "ink": {
                "out_channels": 2,
                "activation": "none",
                "losses": losses if losses is not None else [{"name": "nnUNet_DC_and_CE_loss", "weight": 1.0}],
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
        guide_loss_weight=guide_loss_weight,
        guide_supervision_target="ink",
        train_num_dataloader_workers=0,
        model_config=guided_model_config,
    )


def _make_pretrained_backbone_mgr(
    data_root: Path,
    checkpoint_path: Path,
    *,
    pretrained_decoder_type: str = "pixelshuffle_conv",
    losses: list[dict] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        gpu_ids=None,
        use_ddp=False,
        targets={
            "surface": {
                "out_channels": 2,
                "activation": "none",
                "losses": losses if losses is not None else [{"name": "nnUNet_DC_and_CE_loss", "weight": 1.0}],
            }
        },
        tr_configs={},
        model_name="pretrained_backbone_smoke",
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
        guide_loss_weight=0.0,
        guide_supervision_target=None,
        train_num_dataloader_workers=0,
        model_config={
            "pretrained_backbone": str(checkpoint_path),
            "pretrained_decoder_type": str(pretrained_decoder_type),
            "freeze_encoder": True,
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


def test_feature_encoder_checkpoint_roundtrip_preserves_plain_inference_forward(tmp_path: Path):
    data_root = _make_synthetic_dataset(tmp_path)
    guide_checkpoint = tmp_path / "guide_backbone.pt"
    _write_local_guide_checkpoint(guide_checkpoint)
    mgr = _make_mgr(
        data_root,
        guide_checkpoint,
        guide_fusion_stage="feature_encoder",
        guide_loss_weight=0.0,
        guide_tokenbook_prototype_weighting="token_mlp",
    )
    trainer = BaseTrainer(mgr=mgr, verbose=False)
    model = trainer._build_model()
    checkpoint_path = tmp_path / "feature_encoder_model.pth"
    torch.save({"model_config": model.final_config, "model": model.state_dict()}, checkpoint_path)

    output_dir = tmp_path / "inference_out_feature_encoder"
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
    assert model_info["network"].final_config["guide_fusion_stage"] == "feature_encoder"


def test_feature_skip_concat_checkpoint_roundtrip_preserves_plain_inference_forward(tmp_path: Path):
    data_root = _make_synthetic_dataset(tmp_path)
    guide_checkpoint = tmp_path / "guide_backbone.pt"
    _write_local_guide_checkpoint(guide_checkpoint)
    mgr = _make_mgr(
        data_root,
        guide_checkpoint,
        guide_fusion_stage="feature_skip_concat",
        guide_loss_weight=0.0,
    )
    trainer = BaseTrainer(mgr=mgr, verbose=False)
    model = trainer._build_model()
    checkpoint_path = tmp_path / "feature_skip_concat_model.pth"
    torch.save({"model_config": model.final_config, "model": model.state_dict()}, checkpoint_path)

    output_dir = tmp_path / "inference_out_feature_skip_concat"
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
    assert model_info["network"].final_config["guide_fusion_stage"] == "feature_skip_concat"


def test_pretrained_backbone_pixelshuffle_checkpoint_roundtrip_preserves_plain_inference_forward(tmp_path: Path):
    data_root = _make_synthetic_dataset(tmp_path)
    guide_checkpoint = tmp_path / "guide_backbone.pt"
    _write_local_guide_checkpoint(guide_checkpoint)
    mgr = _make_pretrained_backbone_mgr(
        data_root,
        guide_checkpoint,
        pretrained_decoder_type="pixelshuffle_conv",
    )
    trainer = BaseTrainer(mgr=mgr, verbose=False)
    model = trainer._build_model()
    checkpoint_path = tmp_path / "pretrained_pixelshuffle_model.pth"
    torch.save({"model_config": model.final_config, "model": model.state_dict()}, checkpoint_path)

    output_dir = tmp_path / "inference_out_pretrained_pixelshuffle"
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
    assert set(output.keys()) == {"surface"}
    assert output["surface"].shape == (1, 2, 16, 16, 16)
    assert model_info["network"].final_config["pretrained_decoder_type"] == "pixelshuffle_conv"


def test_pretrained_backbone_pixelshuffle_convhead_big_checkpoint_roundtrip_preserves_plain_inference_forward(tmp_path: Path):
    data_root = _make_synthetic_dataset(tmp_path)
    guide_checkpoint = tmp_path / "guide_backbone.pt"
    _write_local_guide_checkpoint(guide_checkpoint)
    mgr = _make_pretrained_backbone_mgr(
        data_root,
        guide_checkpoint,
        pretrained_decoder_type="pixelshuffle_convhead_big",
    )
    trainer = BaseTrainer(mgr=mgr, verbose=False)
    model = trainer._build_model()
    checkpoint_path = tmp_path / "pretrained_pixelshuffle_convhead_big_model.pth"
    torch.save({"model_config": model.final_config, "model": model.state_dict()}, checkpoint_path)

    output_dir = tmp_path / "inference_out_pretrained_pixelshuffle_convhead_big"
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
    assert set(output.keys()) == {"surface"}
    assert output["surface"].shape == (1, 2, 16, 16, 16)
    assert model_info["network"].final_config["pretrained_decoder_type"] == "pixelshuffle_convhead_big"


def test_pretrained_backbone_explicit_volumes_dataset_loads_from_volume_specs(tmp_path: Path):
    data_root = _make_synthetic_dataset(tmp_path)
    guide_checkpoint = tmp_path / "guide_backbone.pt"
    _write_local_guide_checkpoint(guide_checkpoint)
    mgr = _make_pretrained_backbone_mgr(
        data_root,
        guide_checkpoint,
        pretrained_decoder_type="pixelshuffle_convhead_big",
    )

    runtime_cfg_dir = tmp_path / "runtime_configs"
    runtime_cfg_dir.mkdir()
    mgr.data_path = runtime_cfg_dir
    mgr.dataset_config = {
        "volumes": [
            {
                "image": str(data_root / "images" / "volume1.zarr"),
                "label": str(data_root / "labels" / "volume1_surface.zarr"),
            }
        ],
        "skip_patch_validation": True,
    }
    mgr.skip_patch_validation = True

    trainer = BaseTrainer(mgr=mgr, verbose=False)
    dataset = trainer._configure_dataset(is_training=True)

    assert len(dataset._volumes) == 1
    assert dataset._volumes[0].image_path == data_root / "images" / "volume1.zarr"
    assert dataset._volumes[0].label_paths["surface"] == data_root / "labels" / "volume1_surface.zarr"
    assert dataset._volumes[0].has_labels is True
    assert len(dataset) > 0


def test_pretrained_backbone_explicit_volumes_duplicate_image_ids_match_cache_naming(tmp_path: Path):
    data_root = _make_synthetic_dataset(tmp_path)
    guide_checkpoint = tmp_path / "guide_backbone.pt"
    _write_local_guide_checkpoint(guide_checkpoint)
    mgr = _make_pretrained_backbone_mgr(
        data_root,
        guide_checkpoint,
        pretrained_decoder_type="pixelshuffle_convhead_big",
    )

    runtime_cfg_dir = tmp_path / "runtime_configs"
    runtime_cfg_dir.mkdir()
    image_path = data_root / "images" / "volume1.zarr"
    label_path = data_root / "labels" / "volume1_surface.zarr"
    mgr.data_path = runtime_cfg_dir
    mgr.dataset_config = {
        "volumes": [
            {"image": str(image_path), "label": str(label_path)},
            {"image": str(image_path), "label": str(label_path)},
            {"image": str(image_path), "label": str(label_path)},
        ],
        "skip_patch_validation": True,
    }
    mgr.skip_patch_validation = True

    trainer = BaseTrainer(mgr=mgr, verbose=False)
    dataset = trainer._configure_dataset(is_training=True)

    assert [vol.volume_id for vol in dataset._volumes] == ["volume1", "volume1_1", "volume1_2"]


def test_direct_segmentation_checkpoint_roundtrip_preserves_plain_inference_forward(tmp_path: Path):
    data_root = _make_synthetic_dataset(tmp_path)
    guide_checkpoint = tmp_path / "guide_backbone.pt"
    _write_local_guide_checkpoint(guide_checkpoint)
    mgr = _make_mgr(
        data_root,
        guide_checkpoint,
        guide_fusion_stage="direct_segmentation",
        guide_loss_weight=0.0,
    )
    trainer = BaseTrainer(mgr=mgr, verbose=False)
    model = trainer._build_model()
    checkpoint_path = tmp_path / "direct_segmentation_model.pth"
    torch.save({"model_config": model.final_config, "model": model.state_dict()}, checkpoint_path)

    output_dir = tmp_path / "inference_out_direct_segmentation"
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
    assert output["ink"].shape == (1, 2, 16, 16, 16)
    assert model_info["network"].final_config["guide_fusion_stage"] == "direct_segmentation"


def test_inferer_finalize_output_batch_returns_float16_numpy():
    inferer = object.__new__(Inferer)
    inferer.device = torch.device("cpu")

    output_batch = torch.randn(1, 2, 4, 4, 4, dtype=torch.float32)
    output_np = inferer._finalize_output_batch(output_batch)

    assert output_np.dtype == np.float16
    assert output_np.shape == (1, 2, 4, 4, 4)


def test_save_debug_adds_aux_panel_to_preview_image():
    input_volume = torch.zeros((1, 1, 4, 8, 8), dtype=torch.float32)
    targets_dict = {"surface": torch.zeros((1, 2, 4, 8, 8), dtype=torch.float32)}
    outputs_dict = {"surface": torch.zeros((1, 2, 4, 8, 8), dtype=torch.float32)}
    aux_outputs_dict = {"guide_surface": torch.ones((1, 1, 4, 8, 8), dtype=torch.float32)}
    tasks_dict = {"surface": {"activation": "none"}}

    _, preview_without_aux = save_debug(
        input_volume=input_volume,
        targets_dict=targets_dict,
        outputs_dict=outputs_dict,
        tasks_dict=tasks_dict,
        epoch=0,
        save_media=False,
    )
    _, preview_with_aux = save_debug(
        input_volume=input_volume,
        targets_dict=targets_dict,
        outputs_dict=outputs_dict,
        aux_outputs_dict=aux_outputs_dict,
        tasks_dict=tasks_dict,
        epoch=0,
        save_media=False,
    )

    assert preview_with_aux is not None
    assert preview_without_aux is not None
    assert preview_with_aux.shape[1] > preview_without_aux.shape[1]


def test_trainer_builds_guide_preview_and_media_payload():
    mgr = _make_mgr(Path("/tmp/guide_backbone.pt"), Path("/tmp/guide_backbone.pt"))
    trainer = BaseTrainer(mgr=mgr, verbose=False)

    aux_outputs_dict = {
        "guide_surface": torch.zeros((1, 1, 8, 8, 8), dtype=torch.float32)
    }
    aux_outputs_dict["guide_surface"][:, :, 4] = 1.0

    guide_preview = trainer._make_aux_preview_image(aux_outputs_dict)
    payload = trainer._build_debug_media_payload(
        debug_preview_image=np.zeros((8, 8, 3), dtype=np.uint8),
        guide_preview_image=guide_preview,
    )

    assert guide_preview is not None
    assert guide_preview.ndim == 3
    assert guide_preview.shape[2] == 3
    assert set(payload.keys()) == {"debug_image", "debug_guide_image"}


def test_trainer_builds_only_standard_debug_payload_when_no_guide_preview():
    mgr = _make_mgr(Path("/tmp/guide_backbone.pt"), Path("/tmp/guide_backbone.pt"))
    trainer = BaseTrainer(mgr=mgr, verbose=False)

    payload = trainer._build_debug_media_payload(
        debug_preview_image=np.zeros((8, 8, 3), dtype=np.uint8),
        guide_preview_image=None,
    )

    assert set(payload.keys()) == {"debug_image"}


def test_attach_debug_media_to_wandb_metrics_includes_guide_image():
    class _FakeWandb:
        @staticmethod
        def Image(value):
            return ("image", value.shape)

    metrics = {"epoch": 0, "step": 1}
    payload = {
        "debug_image": np.zeros((8, 8, 3), dtype=np.uint8),
        "debug_guide_image": np.zeros((6, 10, 3), dtype=np.uint8),
    }

    result = BaseTrainer._attach_debug_media_to_wandb_metrics(metrics, payload, _FakeWandb)

    assert result["debug_image"] == ("image", (8, 8, 3))
    assert result["debug_guide_image"] == ("image", (6, 10, 3))


def test_guided_base_trainer_omits_guide_loss_when_weight_is_zero(tmp_path: Path):
    data_root = _make_synthetic_dataset(tmp_path)
    guide_checkpoint = tmp_path / "guide_backbone.pt"
    _write_local_guide_checkpoint(guide_checkpoint)
    mgr = _make_mgr(data_root, guide_checkpoint, guide_loss_weight=0.0)
    trainer = BaseTrainer(mgr=mgr, verbose=False)
    model = trainer._build_model().to(trainer.device)
    loss_fns = trainer._build_loss()
    dataset = trainer._configure_dataset(is_training=True)
    batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False)))

    _inputs, targets_dict, outputs = trainer._get_model_outputs(model, batch)
    loss, task_losses = trainer._compute_train_loss(outputs, targets_dict, loss_fns)

    assert torch.isfinite(loss)
    assert "guide_mask" not in task_losses


def test_feature_encoder_base_trainer_smoke_runs_two_training_steps_without_aux_guide_loss(tmp_path: Path):
    data_root = _make_synthetic_dataset(tmp_path)
    guide_checkpoint = tmp_path / "guide_backbone.pt"
    _write_local_guide_checkpoint(guide_checkpoint)
    mgr = _make_mgr(
        data_root,
        guide_checkpoint,
        guide_fusion_stage="feature_encoder",
        guide_loss_weight=0.0,
    )
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
        assert "guide_mask" not in task_losses
        assert set(trainer._current_aux_outputs.keys()) == {"enc_0", "enc_1", "enc_2"}
        loss.backward()
        optimizer.step()


def test_feature_skip_concat_base_trainer_smoke_runs_two_training_steps_without_aux_outputs(tmp_path: Path):
    data_root = _make_synthetic_dataset(tmp_path)
    guide_checkpoint = tmp_path / "guide_backbone.pt"
    _write_local_guide_checkpoint(guide_checkpoint)
    mgr = _make_mgr(
        data_root,
        guide_checkpoint,
        guide_fusion_stage="feature_skip_concat",
        guide_loss_weight=0.0,
    )
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
        assert "guide_mask" not in task_losses
        assert trainer._current_aux_outputs == {}
        loss.backward()
        optimizer.step()


def test_pretrained_backbone_pixelshuffle_base_trainer_smoke_runs_two_training_steps(tmp_path: Path):
    data_root = _make_synthetic_dataset(tmp_path)
    guide_checkpoint = tmp_path / "guide_backbone.pt"
    _write_local_guide_checkpoint(guide_checkpoint)
    mgr = _make_pretrained_backbone_mgr(
        data_root,
        guide_checkpoint,
        pretrained_decoder_type="pixelshuffle_conv",
    )
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
        assert outputs["surface"].shape == (1, 2, 16, 16, 16)
        loss.backward()
        optimizer.step()


def test_direct_segmentation_base_trainer_smoke_runs_two_training_steps_without_aux_guide_loss(tmp_path: Path):
    data_root = _make_synthetic_dataset(tmp_path)
    guide_checkpoint = tmp_path / "guide_backbone.pt"
    _write_local_guide_checkpoint(guide_checkpoint)
    mgr = _make_mgr(
        data_root,
        guide_checkpoint,
        guide_fusion_stage="direct_segmentation",
        guide_loss_weight=0.0,
    )
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
        assert outputs["ink"].shape == (1, 2, 16, 16, 16)
        assert "guide_mask" not in task_losses
        assert set(trainer._current_aux_outputs.keys()) == {"guide_mask"}
        loss.backward()
        optimizer.step()


def test_direct_segmentation_bce_style_loss_smoke_runs(tmp_path: Path):
    data_root = _make_synthetic_dataset(tmp_path)
    guide_checkpoint = tmp_path / "guide_backbone.pt"
    _write_local_guide_checkpoint(guide_checkpoint)
    mgr = _make_mgr(
        data_root,
        guide_checkpoint,
        guide_fusion_stage="direct_segmentation",
        guide_loss_weight=0.0,
        losses=[{"name": "nnUNet_DC_and_BCE_loss", "weight": 1.0}],
    )
    trainer = BaseTrainer(mgr=mgr, verbose=False)
    model = trainer._build_model().to(trainer.device)
    loss_fns = trainer._build_loss()
    dataset = trainer._configure_dataset(is_training=True)
    batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False)))

    _inputs, targets_dict, outputs = trainer._get_model_outputs(model, batch)
    loss, task_losses = trainer._compute_train_loss(outputs, targets_dict, loss_fns)

    assert torch.isfinite(loss)
    assert "guide_mask" not in task_losses


def test_direct_segmentation_plain_bce_with_logits_loss_smoke_runs(tmp_path: Path):
    data_root = _make_synthetic_dataset(tmp_path)
    guide_checkpoint = tmp_path / "guide_backbone.pt"
    _write_local_guide_checkpoint(guide_checkpoint)
    mgr = _make_mgr(
        data_root,
        guide_checkpoint,
        guide_fusion_stage="direct_segmentation",
        guide_loss_weight=0.0,
        losses=[{"name": "BCEWithLogitsLoss", "weight": 1.0}],
    )
    trainer = BaseTrainer(mgr=mgr, verbose=False)
    model = trainer._build_model().to(trainer.device)
    loss_fns = trainer._build_loss()
    dataset = trainer._configure_dataset(is_training=True)
    batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False)))

    _inputs, targets_dict, outputs = trainer._get_model_outputs(model, batch)
    loss, task_losses = trainer._compute_train_loss(outputs, targets_dict, loss_fns)

    assert torch.isfinite(loss)
    assert "guide_mask" not in task_losses


def test_direct_segmentation_plain_bce_with_logits_masks_ignored_labels(tmp_path: Path):
    data_root = _make_synthetic_dataset(tmp_path)
    guide_checkpoint = tmp_path / "guide_backbone.pt"
    _write_local_guide_checkpoint(guide_checkpoint)
    mgr = _make_mgr(
        data_root,
        guide_checkpoint,
        guide_fusion_stage="direct_segmentation",
        guide_loss_weight=0.0,
        losses=[{"name": "BCEWithLogitsLoss", "weight": 1.0}],
    )
    trainer = BaseTrainer(mgr=mgr, verbose=False)
    trainer.mgr.targets["ink"]["ignore_label"] = -1.0

    prediction = torch.zeros((1, 2, 4, 4, 4), dtype=torch.float32)
    target = torch.zeros((1, 1, 4, 4, 4), dtype=torch.float32)
    target[..., 0, 0, 0] = -1.0
    target[..., 0, 0, 1] = 1.0

    bce_loss = torch.nn.BCEWithLogitsLoss()
    adapted_prediction, adapted_target, extra_loss_kwargs = trainer._adapt_direct_segmentation_loss_inputs(
        bce_loss,
        prediction,
        target,
        target_name="ink",
    )

    assert adapted_prediction.shape == (1, 1, 4, 4, 4)
    assert adapted_target.shape == (1, 1, 4, 4, 4)
    assert torch.isfinite(adapted_target).all()
    assert adapted_target[..., 0, 0, 0].item() == pytest.approx(0.0)
    assert adapted_target[..., 0, 0, 1].item() == pytest.approx(1.0)
    assert "loss_mask" in extra_loss_kwargs
    assert extra_loss_kwargs["loss_mask"][..., 0, 0, 0].item() == pytest.approx(0.0)
    assert extra_loss_kwargs["loss_mask"][..., 0, 0, 1].item() == pytest.approx(1.0)


def test_direct_segmentation_medial_surface_recall_smoke_runs(tmp_path: Path):
    data_root = _make_synthetic_dataset(tmp_path)
    guide_checkpoint = tmp_path / "guide_backbone.pt"
    _write_local_guide_checkpoint(guide_checkpoint)
    mgr = _make_mgr(
        data_root,
        guide_checkpoint,
        guide_fusion_stage="direct_segmentation",
        guide_loss_weight=0.0,
        losses=[{"name": "MedialSurfaceRecall", "weight": 1.0}],
    )
    trainer = BaseTrainer(mgr=mgr, verbose=False)
    model = trainer._build_model().to(trainer.device)
    loss_fns = trainer._build_loss()
    dataset = trainer._configure_dataset(is_training=True)
    batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False)))

    _inputs, targets_dict, outputs = trainer._get_model_outputs(model, batch)
    loss, task_losses = trainer._compute_train_loss(outputs, targets_dict, loss_fns)

    assert torch.isfinite(loss)
    assert "guide_mask" not in task_losses


def test_feature_encoder_rejects_nonzero_guide_loss_weight(tmp_path: Path):
    data_root = _make_synthetic_dataset(tmp_path)
    guide_checkpoint = tmp_path / "guide_backbone.pt"
    _write_local_guide_checkpoint(guide_checkpoint)
    mgr = _make_mgr(
        data_root,
        guide_checkpoint,
        guide_fusion_stage="feature_encoder",
        guide_loss_weight=0.25,
        guide_tokenbook_prototype_weighting="token_mlp",
    )

    with pytest.raises(ValueError, match="guide_loss_weight"):
        BaseTrainer(mgr=mgr, verbose=False)


def test_feature_skip_concat_rejects_nonzero_guide_loss_weight(tmp_path: Path):
    data_root = _make_synthetic_dataset(tmp_path)
    guide_checkpoint = tmp_path / "guide_backbone.pt"
    _write_local_guide_checkpoint(guide_checkpoint)
    mgr = _make_mgr(
        data_root,
        guide_checkpoint,
        guide_fusion_stage="feature_skip_concat",
        guide_loss_weight=0.25,
    )

    with pytest.raises(ValueError, match="guide_loss_weight"):
        BaseTrainer(mgr=mgr, verbose=False)


def test_direct_segmentation_rejects_nonzero_guide_loss_weight(tmp_path: Path):
    data_root = _make_synthetic_dataset(tmp_path)
    guide_checkpoint = tmp_path / "guide_backbone.pt"
    _write_local_guide_checkpoint(guide_checkpoint)
    mgr = _make_mgr(
        data_root,
        guide_checkpoint,
        guide_fusion_stage="direct_segmentation",
        guide_loss_weight=0.25,
    )

    with pytest.raises(ValueError, match="guide_loss_weight"):
        BaseTrainer(mgr=mgr, verbose=False)


def test_direct_segmentation_rejects_unsupported_loss_name(tmp_path: Path):
    data_root = _make_synthetic_dataset(tmp_path)
    guide_checkpoint = tmp_path / "guide_backbone.pt"
    _write_local_guide_checkpoint(guide_checkpoint)
    mgr = _make_mgr(
        data_root,
        guide_checkpoint,
        guide_fusion_stage="direct_segmentation",
        guide_loss_weight=0.0,
        losses=[{"name": "MSELoss", "weight": 1.0}],
    )
    trainer = BaseTrainer(mgr=mgr, verbose=False)
    model = trainer._build_model().to(trainer.device)
    loss_fns = trainer._build_loss()
    dataset = trainer._configure_dataset(is_training=True)
    batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False)))

    _inputs, targets_dict, outputs = trainer._get_model_outputs(model, batch)

    with pytest.raises(ValueError, match="not supported with guide_fusion_stage='direct_segmentation'"):
        trainer._compute_train_loss(outputs, targets_dict, loss_fns)


def test_prepare_metrics_for_logging_includes_guide_loss_entries():
    mgr = _make_mgr(Path("/tmp/guide_backbone.pt"), Path("/tmp/guide_backbone.pt"))
    trainer = BaseTrainer(mgr=mgr, verbose=False)

    metrics = trainer._prepare_metrics_for_logging(
        epoch=0,
        step=3,
        epoch_losses={"ink": [0.4, 0.2], "guide_mask": [0.1, 0.3]},
        current_lr=1e-3,
        val_losses={"ink": [0.5], "guide_mask": [0.25]},
    )

    assert metrics["train_loss_ink"] == pytest.approx(0.3)
    assert metrics["train_loss_guide_mask"] == pytest.approx(0.2)
    assert metrics["val_loss_ink"] == pytest.approx(0.5)
    assert metrics["val_loss_guide_mask"] == pytest.approx(0.25)
    assert metrics["val_loss_total"] == pytest.approx(0.375)


def test_feature_encoder_aux_outputs_stay_out_of_debug_composite():
    mgr = _make_mgr(
        Path("/tmp/guide_backbone.pt"),
        Path("/tmp/guide_backbone.pt"),
        guide_fusion_stage="feature_encoder",
        guide_loss_weight=0.0,
        guide_tokenbook_prototype_weighting="token_mlp",
    )
    trainer = BaseTrainer(mgr=mgr, verbose=False)
    aux_outputs_dict = {
        "enc_0": torch.ones((1, 1, 8, 8, 8), dtype=torch.float32),
        "enc_1": torch.ones((1, 1, 4, 4, 4), dtype=torch.float32),
    }

    prepared = trainer._prepare_aux_debug_outputs(
        aux_outputs_dict,
        torch.zeros((1, 1, 8, 8, 8), dtype=torch.float32),
    )

    assert prepared == {}


def test_feature_encoder_trainer_builds_stage_montage_and_media_payload(tmp_path: Path):
    mgr = _make_mgr(
        tmp_path,
        tmp_path / "guide_backbone.pt",
        guide_fusion_stage="feature_encoder",
        guide_loss_weight=0.0,
        guide_tokenbook_prototype_weighting="token_mlp",
    )
    trainer = BaseTrainer(mgr=mgr, verbose=False)
    aux_outputs_dict = {
        "enc_0": torch.zeros((1, 1, 8, 8, 8), dtype=torch.float32),
        "enc_1": torch.zeros((1, 1, 4, 4, 4), dtype=torch.float32),
        "enc_2": torch.zeros((1, 1, 2, 2, 2), dtype=torch.float32),
    }
    aux_outputs_dict["enc_0"][:, :, 4] = 1.0
    aux_outputs_dict["enc_1"][:, :, 2] = 1.0
    aux_outputs_dict["enc_2"][:, :, 1] = 1.0

    guide_preview = trainer._make_feature_encoder_guide_preview_image(aux_outputs_dict)
    save_path = tmp_path / "guide_preview.png"
    trainer._save_preview_image(guide_preview, save_path)
    payload = trainer._build_debug_media_payload(
        debug_preview_image=np.zeros((8, 8, 3), dtype=np.uint8),
        guide_preview_image=guide_preview,
    )

    assert guide_preview is not None
    assert guide_preview.ndim == 3
    assert guide_preview.shape[2] == 3
    assert guide_preview.shape[:2] == (38, 24)
    assert save_path.exists()
    assert set(payload.keys()) == {"debug_image", "debug_guide_image"}


def test_feature_encoder_trainer_builds_train_and_val_stage_montage(tmp_path: Path):
    mgr = _make_mgr(
        tmp_path,
        tmp_path / "guide_backbone.pt",
        guide_fusion_stage="feature_encoder",
        guide_loss_weight=0.0,
        guide_tokenbook_prototype_weighting="token_mlp",
    )
    trainer = BaseTrainer(mgr=mgr, verbose=False)
    val_aux_outputs_dict = {
        "enc_0": torch.zeros((1, 1, 8, 8, 8), dtype=torch.float32),
        "enc_1": torch.zeros((1, 1, 4, 4, 4), dtype=torch.float32),
        "enc_2": torch.zeros((1, 1, 2, 2, 2), dtype=torch.float32),
    }
    train_aux_outputs_dict = {
        "enc_0": torch.ones((1, 1, 8, 8, 8), dtype=torch.float32),
        "enc_1": torch.ones((1, 1, 4, 4, 4), dtype=torch.float32),
        "enc_2": torch.ones((1, 1, 2, 2, 2), dtype=torch.float32),
    }

    guide_preview = trainer._make_feature_encoder_guide_preview_image(
        val_aux_outputs_dict,
        train_aux_outputs_dict=train_aux_outputs_dict,
    )

    assert guide_preview is not None
    assert guide_preview.ndim == 3
    assert guide_preview.shape[2] == 3
    assert guide_preview.shape[:2] == (76, 24)


def test_direct_segmentation_aux_outputs_render_only_standard_debug_payload():
    mgr = _make_mgr(Path("/tmp/guide_backbone.pt"), Path("/tmp/guide_backbone.pt"), guide_fusion_stage="direct_segmentation", guide_loss_weight=0.0)
    trainer = BaseTrainer(mgr=mgr, verbose=False)
    aux_outputs_dict = {
        "guide_mask": torch.ones((1, 1, 4, 4, 4), dtype=torch.float32),
    }

    prepared = trainer._prepare_aux_debug_outputs(
        aux_outputs_dict,
        torch.zeros((1, 1, 8, 8, 8), dtype=torch.float32),
    )
    guide_preview = trainer._make_feature_encoder_guide_preview_image(aux_outputs_dict)
    payload = trainer._build_debug_media_payload(
        debug_preview_image=np.zeros((8, 8, 3), dtype=np.uint8),
        guide_preview_image=guide_preview,
    )

    assert set(prepared.keys()) == {"guide_ink"}
    assert guide_preview is None
    assert set(payload.keys()) == {"debug_image"}


def test_select_debug_sample_index_from_targets_prefers_first_nonzero_sample():
    targets_dict = {
        "surface": torch.tensor(
            [
                [[[[0.0, 0.0], [0.0, 0.0]]]],
                [[[[0.0, 1.0], [0.0, 0.0]]]],
                [[[[1.0, 1.0], [0.0, 0.0]]]],
            ],
            dtype=torch.float32,
        )
    }

    index, found = BaseTrainer._select_debug_sample_index_from_targets(targets_dict)

    assert found is True
    assert index == 1


class _DummyDataset:
    def __len__(self):
        return 1

    def __getitem__(self, index):
        return {"image": torch.zeros((1, 4, 4, 4), dtype=torch.float32)}


class _DummyModel(torch.nn.Module):
    def __init__(self, *, guide_enabled: bool):
        super().__init__()
        self.guide_enabled = guide_enabled
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * self.weight


def _make_compile_mgr(tmp_path: Path, *, compile_policy: str) -> SimpleNamespace:
    return SimpleNamespace(
        gpu_ids=None,
        use_ddp=False,
        targets={"ink": {"out_channels": 1, "activation": "none"}},
        tr_configs={},
        model_name="compile_policy_test",
        train_patch_size=(4, 4, 4),
        train_batch_size=1,
        in_channels=1,
        autoconfigure=False,
        enable_deep_supervision=False,
        spacing=(1.0, 1.0, 1.0),
        data_path=tmp_path,
        ckpt_out_base=tmp_path / "ckpts",
        checkpoint_path=None,
        load_weights_only=False,
        guide_loss_weight=0.0,
        guide_supervision_target=None,
        train_num_dataloader_workers=0,
        compile_policy=compile_policy,
        startup_timing=False,
        ddp_find_unused_parameters="auto",
        ddp_static_graph="auto",
        ddp_gradient_as_bucket_view=False,
        no_amp=True,
        amp_dtype="float16",
        verbose=False,
        auto_detect_channels=lambda dataset, sample: None,
    )


def _make_pretrained_compile_mgr(tmp_path: Path, checkpoint_path: Path, *, compile_policy: str) -> SimpleNamespace:
    mgr = _make_compile_mgr(tmp_path, compile_policy=compile_policy)
    mgr.targets = {"surface": {"out_channels": 2, "activation": "none"}}
    mgr.model_name = "compile_policy_pretrained_pixelshuffle"
    mgr.train_patch_size = (16, 16, 16)
    mgr.train_batch_size = 1
    mgr.model_config = {
        "pretrained_backbone": str(checkpoint_path),
        "pretrained_decoder_type": "pixelshuffle_conv",
        "freeze_encoder": True,
        "input_shape": [16, 16, 16],
    }
    return mgr


def _configure_compile_test_trainer(
    trainer: BaseTrainer,
    monkeypatch,
    *,
    guide_enabled: bool,
    call_order: list[str],
    compile_guidance: bool = False,
):
    dummy_dataset = _DummyDataset()
    dummy_model = _DummyModel(guide_enabled=guide_enabled)
    if compile_guidance:
        dummy_model._compile_guidance_submodules = lambda device_type: call_order.append("guide_submodule_compile") or ["guide_backbone"]

    monkeypatch.setattr(trainer, "_configure_dataset", lambda is_training: dummy_dataset)
    monkeypatch.setattr(trainer, "_build_dataset_for_mgr", lambda mgr, is_training: dummy_dataset)
    monkeypatch.setattr(trainer, "_prepare_sample", lambda sample, is_training: sample)
    monkeypatch.setattr(trainer, "_build_model", lambda: dummy_model)
    monkeypatch.setattr(trainer, "_get_optimizer", lambda model: torch.optim.SGD(model.parameters(), lr=0.1))
    monkeypatch.setattr(
        trainer,
        "_get_scheduler",
        lambda optimizer: (torch.optim.lr_scheduler.StepLR(optimizer, step_size=1), False),
    )
    monkeypatch.setattr(trainer, "_get_scaler", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        trainer,
        "_configure_dataloaders",
        lambda train_dataset, val_dataset: ([], [], [0], [0]),
    )
    monkeypatch.setattr(trainer, "_build_loss", lambda: {})
    monkeypatch.setattr(trainer, "_initialize_ema_model", lambda model: None)
    monkeypatch.setattr(
        trainer,
        "_wrap_model_for_distributed_training",
        lambda model: call_order.append("wrap") or model,
    )
    monkeypatch.setattr(
        trainer,
        "_maybe_compile_model",
        lambda model: call_order.append("compile") or model,
    )


def test_initialize_training_compiles_guided_model_before_ddp_wrap(tmp_path: Path, monkeypatch):
    mgr = _make_compile_mgr(tmp_path, compile_policy="auto")
    trainer = BaseTrainer(mgr=mgr, verbose=False)
    call_order: list[str] = []
    _configure_compile_test_trainer(trainer, monkeypatch, guide_enabled=True, call_order=call_order)

    trainer._initialize_training()

    assert call_order == ["compile", "wrap"]


def test_initialize_training_compiles_guidance_submodule_before_model_compile(tmp_path: Path, monkeypatch):
    mgr = _make_compile_mgr(tmp_path, compile_policy="auto")
    trainer = BaseTrainer(mgr=mgr, verbose=False)
    call_order: list[str] = []
    _configure_compile_test_trainer(
        trainer,
        monkeypatch,
        guide_enabled=True,
        call_order=call_order,
        compile_guidance=True,
    )

    trainer._initialize_training()

    assert call_order == ["guide_submodule_compile", "compile", "wrap"]


def test_initialize_training_compiles_unguided_ddp_wrapper_by_default(tmp_path: Path, monkeypatch):
    mgr = _make_compile_mgr(tmp_path, compile_policy="auto")
    trainer = BaseTrainer(mgr=mgr, verbose=False)
    call_order: list[str] = []
    _configure_compile_test_trainer(trainer, monkeypatch, guide_enabled=False, call_order=call_order)

    trainer._initialize_training()

    assert call_order == ["wrap", "compile"]


def test_initialize_training_skips_compile_when_policy_is_off(tmp_path: Path, monkeypatch):
    mgr = _make_compile_mgr(tmp_path, compile_policy="off")
    trainer = BaseTrainer(mgr=mgr, verbose=False)
    call_order: list[str] = []
    _configure_compile_test_trainer(trainer, monkeypatch, guide_enabled=True, call_order=call_order)

    trainer._initialize_training()

    assert call_order == ["wrap"]


def test_initialize_training_places_run_checkpoint_dir_under_ckpt_out_base(tmp_path: Path, monkeypatch):
    mgr = _make_compile_mgr(tmp_path, compile_policy="off")
    trainer = BaseTrainer(mgr=mgr, verbose=False)
    call_order: list[str] = []
    _configure_compile_test_trainer(trainer, monkeypatch, guide_enabled=True, call_order=call_order)

    state = trainer._initialize_training()

    assert str(state["ckpt_dir"]).startswith(str(mgr.ckpt_out_base))
    assert str(state["model_ckpt_dir"]).startswith(str(mgr.ckpt_out_base))


def test_initialize_training_compiles_pretrained_pixelshuffle_model_before_ddp_wrap(tmp_path: Path, monkeypatch):
    checkpoint_path = tmp_path / "guide_backbone.pt"
    _write_local_guide_checkpoint(checkpoint_path)
    mgr = _make_pretrained_compile_mgr(tmp_path, checkpoint_path, compile_policy="module")
    trainer = BaseTrainer(mgr=mgr, verbose=False)
    call_order: list[str] = []
    dummy_dataset = _DummyDataset()

    monkeypatch.setattr(trainer, "_configure_dataset", lambda is_training: dummy_dataset)
    monkeypatch.setattr(trainer, "_build_dataset_for_mgr", lambda mgr, is_training: dummy_dataset)
    monkeypatch.setattr(trainer, "_prepare_sample", lambda sample, is_training: sample)
    monkeypatch.setattr(trainer, "_get_optimizer", lambda model: torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1))
    monkeypatch.setattr(
        trainer,
        "_get_scheduler",
        lambda optimizer: (torch.optim.lr_scheduler.StepLR(optimizer, step_size=1), False),
    )
    monkeypatch.setattr(trainer, "_get_scaler", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        trainer,
        "_configure_dataloaders",
        lambda train_dataset, val_dataset: ([], [], [0], [0]),
    )
    monkeypatch.setattr(trainer, "_build_loss", lambda: {})
    monkeypatch.setattr(trainer, "_initialize_ema_model", lambda model: None)
    monkeypatch.setattr(
        trainer,
        "_wrap_model_for_distributed_training",
        lambda model: call_order.append("wrap") or model,
    )
    monkeypatch.setattr(
        trainer,
        "_maybe_compile_model",
        lambda model: call_order.append("compile") or model,
    )

    trainer._initialize_training()

    assert call_order == ["compile", "wrap"]


def test_wrap_model_uses_guided_ddp_defaults(tmp_path: Path):
    mgr = _make_compile_mgr(tmp_path, compile_policy="module")
    trainer = BaseTrainer(mgr=mgr, verbose=False)
    trainer.is_distributed = True
    trainer.device = torch.device("cpu")
    trainer.rank = 0
    captured = {}

    class _FakeDDP:
        def __init__(self, model, **kwargs):
            captured["model"] = model
            captured["kwargs"] = kwargs

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("vesuvius.models.training.train.DDP", _FakeDDP)
    try:
        trainer._wrap_model_for_distributed_training(_DummyModel(guide_enabled=True))
    finally:
        monkeypatch.undo()

    assert captured["kwargs"]["find_unused_parameters"] is False
    assert captured["kwargs"]["static_graph"] is True


def test_wrap_model_uses_unguided_ddp_defaults(tmp_path: Path):
    mgr = _make_compile_mgr(tmp_path, compile_policy="module")
    trainer = BaseTrainer(mgr=mgr, verbose=False)
    trainer.is_distributed = True
    trainer.device = torch.device("cpu")
    trainer.rank = 0
    captured = {}

    class _FakeDDP:
        def __init__(self, model, **kwargs):
            captured["model"] = model
            captured["kwargs"] = kwargs

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("vesuvius.models.training.train.DDP", _FakeDDP)
    try:
        trainer._wrap_model_for_distributed_training(_DummyModel(guide_enabled=False))
    finally:
        monkeypatch.undo()

    assert captured["kwargs"]["find_unused_parameters"] is True
    assert captured["kwargs"]["static_graph"] is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for autocast safety test")
def test_guide_alignment_loss_is_amp_safe_on_cuda():
    mgr = _make_mgr(Path("/tmp/guide_backbone.pt"), Path("/tmp/guide_backbone.pt"))
    trainer = BaseTrainer(mgr=mgr, verbose=False)
    trainer.device = torch.device("cuda")
    trainer._current_aux_outputs = {
        "guide_mask": torch.full((1, 1, 2, 2, 2), 0.5, device=trainer.device, dtype=torch.float16)
    }
    targets_dict = {
        "ink": torch.ones((1, 1, 2, 2, 2), device=trainer.device, dtype=torch.float16)
    }

    with torch.amp.autocast("cuda"):
        loss = trainer._compute_guide_alignment_loss(targets_dict)

    assert loss is not None
    assert torch.isfinite(loss)
