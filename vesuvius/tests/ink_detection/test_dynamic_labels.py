"""Regression witnesses for the staged-training pseudo-label engines."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vesuvius.ink_detection.training import dynamic_labels as labels


class _ZeroInkModel(nn.Module):
    def __init__(self, logit: float = 0.0) -> None:
        super().__init__()
        self.logit = float(logit)
        self.forward_batch_sizes: list[int] = []

    def forward(self, image: torch.Tensor):
        self.forward_batch_sizes.append(int(image.shape[0]))
        return {"ink": torch.full_like(image, self.logit)}


class _ImageInkModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.forward_batch_sizes: list[int] = []

    def forward(self, image: torch.Tensor):
        self.forward_batch_sizes.append(int(image.shape[0]))
        return {"ink": image}


class _VoxelDino(nn.Module):
    """Emit one two-dimensional token per voxel for the tiny test grid."""

    def __init__(self) -> None:
        super().__init__()
        self.seen_first_voxels: list[float] = []

    def forward_features(self, windows: torch.Tensor):
        self.seen_first_voxels.extend(
            float(value) for value in windows[:, 0, 0, 0, 0]
        )
        values = windows[:, 0].flatten(start_dim=1)
        tokens = torch.stack((values, torch.ones_like(values)), dim=-1)
        return {"x_norm_patchtokens": tokens}


class _ConstantDino(nn.Module):
    def __init__(self, token_count: int = 8) -> None:
        super().__init__()
        self.token_count = token_count

    def forward_features(self, windows: torch.Tensor):
        tokens = torch.ones(
            (int(windows.shape[0]), self.token_count, 2),
            device=windows.device,
            dtype=windows.dtype,
        )
        return {"x_norm_patchtokens": tokens}


def _tiny_dino_generator(
    *,
    unet: nn.Module | None = None,
    dino: nn.Module | None = None,
    threshold: float = 0.5,
) -> labels.DinoGuidedLabelGenerator:
    return labels.DinoGuidedLabelGenerator(
        unet=_ZeroInkModel() if unet is None else unet,
        dino=_VoxelDino() if dino is None else dino,
        reference_embedding=torch.tensor([1.0, 0.0]),
        device="cpu",
        dtype=torch.float32,
        dino_stride=2,
        dino_minibatch=3,
        dino_blend_sigma=1.0,
        threshold=threshold,
        grid=labels.DinoGridSpec(
            chunk_size=4,
            window_size=2,
            patch_size=1,
            embedding_dim=2,
        ),
    )


def test_dino_windows_and_gaussian_preserve_historical_lattice_rules():
    assert labels.dino_window_starts(
        chunk_size=256,
        window_size=128,
        stride=128,
    ) == [0, 128]
    assert labels.dino_window_starts(
        chunk_size=256,
        window_size=128,
        stride=96,
    ) == [0, 96, 128]

    weight = labels.gaussian_window_3d(5, 1.5)
    assert weight.shape == (5, 5, 5)
    assert weight[2, 2, 2] == 1.0
    assert weight[0, 2, 2] == weight[4, 2, 2]
    assert weight[2, 0, 2] == weight[2, 4, 2]


def test_dino_similarity_is_window_major_batch_minor_and_per_sample_minmax():
    dino = _VoxelDino()
    generator = _tiny_dino_generator(dino=dino)
    first = torch.arange(64, dtype=torch.float32).reshape(4, 4, 4)
    second = first + 100.0
    image = torch.stack((first, second), dim=0).unsqueeze(1)

    similarity = generator._dino_similarity(image)

    expected_first_voxels = []
    for z in (0, 2):
        for y in (0, 2):
            for x in (0, 2):
                expected_first_voxels.extend(
                    [float(first[z, y, x]), float(second[z, y, x])]
                )
    assert dino.seen_first_voxels == expected_first_voxels

    raw = image / torch.sqrt(image.square() + 1.0)
    minimum = raw.amin(dim=(1, 2, 3, 4), keepdim=True)
    maximum = raw.amax(dim=(1, 2, 3, 4), keepdim=True)
    expected = (raw - minimum) / (maximum - minimum)
    torch.testing.assert_close(similarity, expected)
    assert torch.equal(
        similarity.amin(dim=(1, 2, 3, 4)), torch.zeros(2)
    )
    assert torch.equal(
        similarity.amax(dim=(1, 2, 3, 4)), torch.ones(2)
    )


def test_dino_constant_similarity_clamps_to_zero_without_nan():
    generator = _tiny_dino_generator(dino=_ConstantDino())
    image = torch.ones((1, 1, 4, 4, 4))

    similarity = generator._dino_similarity(image)

    assert torch.isfinite(similarity).all()
    assert torch.count_nonzero(similarity) == 0


def test_dino_uses_strict_product_threshold_and_optional_foreground_mask():
    generator = _tiny_dino_generator(threshold=0.5)
    image = torch.arange(64, dtype=torch.float32).reshape(1, 1, 4, 4, 4)

    # The maximum normalized cosine is exactly one and sigmoid(0) is exactly
    # one half, so a strict product > 0.5 must reject even that maximum.
    assert torch.count_nonzero(generator.generate(image)) == 0

    generator.threshold = 0.499
    without_mask = generator.generate(image)
    assert without_mask[0, 0, -1, -1, -1] == 1.0

    mask = torch.ones_like(image)
    mask[0, 0, -1, -1, -1] = 0.0
    with_mask = generator.generate(image, mask_b1zyx=mask)
    assert with_mask[0, 0, -1, -1, -1] == 0.0


def test_dino_rejects_bad_reference_input_shape_and_token_contract():
    grid = labels.DinoGridSpec(
        chunk_size=4,
        window_size=2,
        patch_size=1,
        embedding_dim=2,
    )
    with pytest.raises(ValueError, match="reference embedding"):
        labels.DinoGuidedLabelGenerator(
            unet=_ZeroInkModel(),
            dino=_VoxelDino(),
            reference_embedding=torch.ones(3),
            device="cpu",
            dtype=torch.float32,
            dino_stride=2,
            grid=grid,
        )

    generator = _tiny_dino_generator(dino=_ConstantDino(token_count=7))
    with pytest.raises(ValueError, match="patch tokens"):
        generator.generate(torch.zeros((1, 1, 4, 4, 4)))
    with pytest.raises(ValueError, match=r"shape \[B,1,Z,Y,X\]"):
        generator.generate(torch.zeros((1, 4, 4, 4)))


def test_dino_checkpoint_schema_is_validated_before_model_construction(tmp_path):
    checkpoint = tmp_path / "malformed-dino.pth"
    torch.save({}, checkpoint)

    with pytest.raises(ValueError, match="config.model"):
        labels.load_frozen_dino_backbone(
            checkpoint,
            device="cpu",
            dtype=torch.float32,
        )


def _self_distill_generator(
    *,
    primary: nn.Module,
    ensemble: nn.Module,
    primary_threshold: float = 0.5,
    ensemble_threshold: float = 0.5,
    tta: bool = True,
) -> labels.SelfDistillLabelGenerator:
    return labels.SelfDistillLabelGenerator(
        primary=primary,
        ensemble=ensemble,
        primary_threshold=primary_threshold,
        ensemble_threshold=ensemble_threshold,
        mean_hi=105.0,
        std_lo=30.0,
        tta=tta,
        tta_batch_size=2,
        device="cpu",
        dtype=torch.float32,
        patch_size_zyx=(2, 2, 2),
    )


def test_self_distill_runs_exactly_eight_restored_mirror_predictions():
    primary = _ImageInkModel()
    generator = _self_distill_generator(
        primary=primary,
        ensemble=_ZeroInkModel(),
    )
    image = torch.tensor(
        [-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0]
    ).reshape(1, 1, 2, 2, 2)

    actual = generator.generate(
        image,
        raw_mean=torch.tensor([0.0]),
        raw_std=torch.tensor([100.0]),
    )

    assert generator.variants == [
        (),
        (0,),
        (1,),
        (2,),
        (0, 1),
        (0, 2),
        (1, 2),
        (0, 1, 2),
    ]
    assert primary.forward_batch_sizes == [2, 2, 2, 2]
    assert sum(primary.forward_batch_sizes) == 8
    assert torch.equal(actual, (image > 0.0).float())


def test_self_distill_ensemble_condition_has_strict_mean_and_std_boundaries():
    primary_probability = 0.6
    ensemble_probability = 0.2
    primary = _ZeroInkModel(torch.logit(torch.tensor(primary_probability)).item())
    ensemble = _ZeroInkModel(torch.logit(torch.tensor(ensemble_probability)).item())
    generator = _self_distill_generator(
        primary=primary,
        ensemble=ensemble,
        primary_threshold=0.65,
        ensemble_threshold=0.35,
        tta=False,
    )
    image = torch.zeros((3, 1, 2, 2, 2))

    actual = generator.generate(
        image,
        raw_mean=torch.tensor([106.0, 105.0, 106.0]),
        raw_std=torch.tensor([29.0, 29.0, 30.0]),
    )

    assert torch.equal(actual[0], torch.ones_like(actual[0]))
    assert torch.count_nonzero(actual[1:]) == 0
    assert primary.forward_batch_sizes == [1, 1, 1]
    assert ensemble.forward_batch_sizes == [1]


def test_self_distill_applies_mask_before_a_strict_probability_threshold():
    primary = _ZeroInkModel(0.0)
    generator = _self_distill_generator(
        primary=primary,
        ensemble=_ZeroInkModel(0.0),
        primary_threshold=0.5,
        tta=False,
    )
    image = torch.zeros((1, 1, 2, 2, 2))

    # sigmoid(0) == threshold and the comparison is strict.
    strict = generator.generate(
        image,
        raw_mean=torch.tensor([0.0]),
        raw_std=torch.tensor([100.0]),
    )
    assert torch.count_nonzero(strict) == 0

    generator.primary_threshold = 0.49
    mask = torch.ones_like(image)
    mask[..., 0, 0, 0] = 0.0
    masked = generator.generate(
        image,
        mask_b1zyx=mask,
        raw_mean=torch.tensor([0.0]),
        raw_std=torch.tensor([100.0]),
    )
    assert masked[..., 0, 0, 0] == 0.0
    assert torch.count_nonzero(masked) == 7


def test_dynamic_label_factory_accepts_typed_config_objects(monkeypatch):
    sentinel = _ZeroInkModel()
    monkeypatch.setattr(
        labels, "load_frozen_ink_model", lambda *args, **kwargs: sentinel
    )
    monkeypatch.setattr(
        labels,
        "load_frozen_dino_backbone",
        lambda *args, **kwargs: _VoxelDino(),
    )
    monkeypatch.setattr(
        labels,
        "load_reference_embedding",
        lambda *args, **kwargs: torch.ones(864),
    )
    config = SimpleNamespace(
        kind="dino_guided",
        unet_checkpoint="teacher.pth",
        dino_checkpoint="dino.pth",
        reference_embedding="ink.npy",
        dino_stride=128,
    )

    generator = labels.build_dynamic_label_generator(
        config,
        device="cpu",
        dtype=torch.float32,
    )

    assert isinstance(generator, labels.DinoGuidedLabelGenerator)
    with pytest.raises(ValueError, match="unsupported dynamic-label kind"):
        labels.build_dynamic_label_generator(
            {"kind": "unknown"},
            device="cpu",
            dtype=torch.float32,
        )


def test_real_ink_teacher_strict_loads_when_configured():
    checkpoint = os.environ.get("INK_TEACHER_CHECKPOINT")
    if not checkpoint:
        pytest.skip("set INK_TEACHER_CHECKPOINT for the real-artifact smoke")

    model = labels.load_frozen_ink_model(
        checkpoint,
        device="cpu",
        dtype=torch.float32,
    )

    assert sum(parameter.numel() for parameter in model.parameters()) > 0
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_real_dinovol_teacher_strict_loads_when_configured():
    checkpoint = os.environ.get("DINOVOL_TEACHER_CHECKPOINT")
    if not checkpoint:
        pytest.skip(
            "set DINOVOL_TEACHER_CHECKPOINT for the real-artifact smoke"
        )

    model = labels.load_frozen_dino_backbone(
        checkpoint,
        device="cpu",
        dtype=torch.float32,
    )

    assert sum(parameter.numel() for parameter in model.parameters()) > 0
    assert all(not parameter.requires_grad for parameter in model.parameters())
