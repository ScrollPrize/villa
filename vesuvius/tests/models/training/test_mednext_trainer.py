from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import zarr
import numpy as np
from torch.utils.data import DataLoader

from vesuvius.models.build.build_network_from_config import NetworkFromConfig
from vesuvius.models.run.inference import Inferer
from vesuvius.models.training.train import BaseTrainer


def _make_synthetic_dataset(root: Path) -> Path:
    data_root = root / "data"
    image_root = data_root / "images" / "volume1.zarr"
    label_root = data_root / "labels" / "volume1_surface.zarr"

    image_group = zarr.open_group(str(image_root), mode="w")
    label_group = zarr.open_group(str(label_root), mode="w")
    image_array = image_group.create_dataset("0", shape=(32, 32, 32), chunks=(32, 32, 32), dtype="float32")
    label_array = label_group.create_dataset("0", shape=(32, 32, 32), chunks=(32, 32, 32), dtype="uint8")

    coords = np.linspace(-1.0, 1.0, 32, dtype=np.float32)
    z = coords[:, None, None]
    y = coords[None, :, None]
    x = coords[None, None, :]
    image = np.exp(-(x * x + y * y + z * z) * 4.0).astype(np.float32)
    label = ((x * x + y * y + z * z) < 0.35).astype(np.uint8)

    image_array[:] = image
    label_array[:] = label
    return data_root


def _make_mgr(
    data_root: Path,
    *,
    architecture_type: str = "mednext_v1",
    mednext_model_id: str = "S",
    compile_policy: str = "off",
    targets: dict | None = None,
    enable_deep_supervision: bool = False,
    separate_decoders: bool | None = None,
) -> SimpleNamespace:
    model_config = {
        "architecture_type": architecture_type,
        "mednext_model_id": mednext_model_id,
        "mednext_kernel_size": 3,
    }
    if separate_decoders is not None:
        model_config["separate_decoders"] = separate_decoders

    return SimpleNamespace(
        gpu_ids=None,
        use_ddp=False,
        targets=targets or {
            "surface": {
                "out_channels": 2,
                "activation": "none",
                "losses": [{"name": "nnUNet_DC_and_CE_loss", "weight": 1.0}],
            }
        },
        tr_configs={},
        model_name="mednext_smoke",
        train_patch_size=(32, 32, 32),
        train_batch_size=1,
        in_channels=1,
        autoconfigure=False,
        enable_deep_supervision=enable_deep_supervision,
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
        compile_policy=compile_policy,
        startup_timing=False,
        ddp_find_unused_parameters="auto",
        ddp_static_graph="auto",
        ddp_gradient_as_bucket_view=False,
        no_amp=True,
        amp_dtype="float16",
        verbose=False,
        auto_detect_channels=lambda dataset, sample: None,
        ckpt_out_base=data_root / "ckpts",
        checkpoint_path=None,
        load_weights_only=False,
        model_config=model_config,
    )


def test_mednext_v1_base_trainer_smoke_runs_two_training_steps(tmp_path: Path):
    data_root = _make_synthetic_dataset(tmp_path)
    mgr = _make_mgr(data_root)
    trainer = BaseTrainer(mgr=mgr, verbose=False)
    model = trainer._build_model().to(trainer.device)
    loss_fns = trainer._build_loss()
    dataset = trainer._configure_dataset(is_training=True)
    batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        _inputs, targets_dict, outputs = trainer._get_model_outputs(model, batch)
        loss, task_losses = trainer._compute_train_loss(outputs, targets_dict, loss_fns)
        assert torch.isfinite(loss)
        assert "surface" in task_losses
        loss.backward()
        optimizer.step()


def test_mednext_v1_checkpoint_roundtrip_preserves_plain_inference_forward(tmp_path: Path):
    data_root = _make_synthetic_dataset(tmp_path)
    mgr = _make_mgr(data_root)
    trainer = BaseTrainer(mgr=mgr, verbose=False)
    model = trainer._build_model()
    checkpoint_path = tmp_path / "mednext_model.pth"
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
    output = model_info["network"](torch.randn(1, 1, 32, 32, 32))

    assert isinstance(output, dict)
    assert set(output.keys()) == {"surface"}
    assert output["surface"].shape == (1, 2, 32, 32, 32)
    assert model_info["network"].final_config["architecture_type"] == "mednext_v1"
    assert model_info["network"].final_config["mednext_model_id"] == "S"


class _DummyDataset:
    def __len__(self):
        return 1

    def __getitem__(self, index):
        return {"image": torch.zeros((1, 4, 4, 4), dtype=torch.float32)}


def test_initialize_training_compiles_mednext_model_before_ddp_wrap(tmp_path: Path, monkeypatch):
    mgr = _make_mgr(tmp_path, compile_policy="module")
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
    monkeypatch.setattr(trainer, "_configure_dataloaders", lambda train_dataset, val_dataset: ([], [], [0], [0]))
    monkeypatch.setattr(trainer, "_build_loss", lambda: {})
    monkeypatch.setattr(trainer, "_initialize_ema_model", lambda model: None)
    monkeypatch.setattr(trainer, "_wrap_model_for_distributed_training", lambda model: call_order.append("wrap") or model)
    monkeypatch.setattr(trainer, "_maybe_compile_model", lambda model: call_order.append("compile") or model)

    trainer._initialize_training()

    assert call_order == ["compile", "wrap"]


def test_mednext_v2_checkpoint_roundtrip_preserves_plain_inference_forward(tmp_path: Path):
    data_root = _make_synthetic_dataset(tmp_path)
    mgr = _make_mgr(data_root, architecture_type="mednext_v2", mednext_model_id="L")
    trainer = BaseTrainer(mgr=mgr, verbose=False)
    model = trainer._build_model()
    checkpoint_path = tmp_path / "mednext_v2_model.pth"
    torch.save({"model_config": model.final_config, "model": model.state_dict()}, checkpoint_path)

    output_dir = tmp_path / "inference_out_v2"
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
    output = model_info["network"](torch.randn(1, 1, 32, 32, 32))

    assert isinstance(output, dict)
    assert set(output.keys()) == {"surface"}
    assert output["surface"].shape == (1, 2, 32, 32, 32)
    assert model_info["network"].final_config["architecture_type"] == "mednext_v2"
    assert model_info["network"].final_config["mednext_model_id"] == "L"


def test_mednext_v1_deep_supervision_checkpoint_roundtrip_loads_plain_inference_output(tmp_path: Path):
    data_root = _make_synthetic_dataset(tmp_path)
    mgr = _make_mgr(data_root, enable_deep_supervision=True)
    trainer = BaseTrainer(mgr=mgr, verbose=False)
    model = trainer._build_model()
    assert model.final_config["pool_op_kernel_sizes"] == [[2, 2, 2]] * 4
    checkpoint_path = tmp_path / "mednext_ds_model.pth"
    torch.save({"model_config": model.final_config, "model": model.state_dict()}, checkpoint_path)

    output_dir = tmp_path / "inference_out_ds"
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
    output = model_info["network"](torch.randn(1, 1, 32, 32, 32))

    assert isinstance(output, dict)
    assert isinstance(output["surface"], torch.Tensor)
    assert output["surface"].shape == (1, 2, 32, 32, 32)
    assert model_info["network"].final_config["enable_deep_supervision"] is True
    assert model_info["network"].final_config["pool_op_kernel_sizes"] == [[2, 2, 2]] * 4
    ds_scales = trainer._get_deep_supervision_scales(model_info["network"])
    assert ds_scales == [
        [1.0, 1.0, 1.0],
        [0.5, 0.5, 0.5],
        [0.25, 0.25, 0.25],
        [0.125, 0.125, 0.125],
        [0.0625, 0.0625, 0.0625],
    ]


def test_mednext_mixed_decoder_checkpoint_roundtrip_preserves_layout(tmp_path: Path):
    data_root = _make_synthetic_dataset(tmp_path)
    targets = {
        "surface": {
            "out_channels": 2,
            "activation": "none",
            "losses": [{"name": "nnUNet_DC_and_CE_loss", "weight": 1.0}],
            "separate_decoder": True,
        },
        "tissue": {
            "out_channels": 1,
            "activation": "none",
            "losses": [{"name": "nnUNet_DC_and_CE_loss", "weight": 1.0}],
        },
    }
    mgr = _make_mgr(
        data_root,
        targets=targets,
        separate_decoders=False,
    )
    model = NetworkFromConfig(mgr)
    assert model.shared_decoder is not None
    assert sorted(model.task_decoders.keys()) == ["surface"]
    assert sorted(model.task_heads.keys()) == ["tissue"]

    checkpoint_path = tmp_path / "mednext_mixed_model.pth"
    torch.save({"model_config": model.final_config, "model": model.state_dict()}, checkpoint_path)

    output_dir = tmp_path / "inference_out_mixed"
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
    loaded = model_info["network"]
    output = loaded(torch.randn(1, 1, 32, 32, 32))

    assert isinstance(output, dict)
    assert set(output.keys()) == {"surface", "tissue"}
    assert loaded.shared_decoder is not None
    assert sorted(loaded.task_decoders.keys()) == ["surface"]
    assert sorted(loaded.task_heads.keys()) == ["tissue"]
    assert loaded.final_config["targets"]["surface"]["separate_decoder"] is True
    assert loaded.final_config["targets"]["tissue"]["separate_decoder"] is False


def test_initialize_training_compiles_mednext_v2_model_before_ddp_wrap(tmp_path: Path, monkeypatch):
    mgr = _make_mgr(tmp_path, architecture_type="mednext_v2", mednext_model_id="B", compile_policy="module")
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
    monkeypatch.setattr(trainer, "_configure_dataloaders", lambda train_dataset, val_dataset: ([], [], [0], [0]))
    monkeypatch.setattr(trainer, "_build_loss", lambda: {})
    monkeypatch.setattr(trainer, "_initialize_ema_model", lambda model: None)
    monkeypatch.setattr(trainer, "_wrap_model_for_distributed_training", lambda model: call_order.append("wrap") or model)
    monkeypatch.setattr(trainer, "_maybe_compile_model", lambda model: call_order.append("compile") or model)

    trainer._initialize_training()

    assert call_order == ["compile", "wrap"]
