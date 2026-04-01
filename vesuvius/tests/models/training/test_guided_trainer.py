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
        no_amp=True,
        amp_dtype="float16",
        verbose=False,
        auto_detect_channels=lambda dataset, sample: None,
    )


def _configure_compile_test_trainer(
    trainer: BaseTrainer,
    monkeypatch,
    *,
    guide_enabled: bool,
    call_order: list[str],
):
    dummy_dataset = _DummyDataset()
    dummy_model = _DummyModel(guide_enabled=guide_enabled)

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
