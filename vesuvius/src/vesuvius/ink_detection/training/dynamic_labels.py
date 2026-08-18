"""Frozen-teacher pseudo-label engines for staged 3D ink training.

The generators in this module run in the trainer process, after data loading.
They deliberately accept already-normalized teacher images and optional binary
foreground masks so dataset workers do not own CUDA models.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from vesuvius.ink_detection.inference.infer_full3d_tifxyz import (
    predict_batch,
    tta_variants,
)
from vesuvius.ink_detection.inference.inference_runtime import TargetModel
from vesuvius.ink_detection.models.checkpoint import (
    config_from_checkpoint,
    load_checkpoint,
    load_model_state,
    select_inference_weights,
)
from vesuvius.ink_detection.models.model import make_model


@dataclass(frozen=True)
class DinoGridSpec:
    """Spatial contract of the historical Dinovol pseudo-label recipe."""

    chunk_size: int = 256
    window_size: int = 128
    patch_size: int = 8
    embedding_dim: int = 864

    def __post_init__(self) -> None:
        values = {
            "chunk_size": self.chunk_size,
            "window_size": self.window_size,
            "patch_size": self.patch_size,
            "embedding_dim": self.embedding_dim,
        }
        for name, value in values.items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive, got {value!r}")
        if self.window_size > self.chunk_size:
            raise ValueError("window_size must not exceed chunk_size")
        if self.window_size % self.patch_size:
            raise ValueError("window_size must be divisible by patch_size")
        if self.chunk_size % self.patch_size:
            raise ValueError("chunk_size must be divisible by patch_size")

    @property
    def tokens_per_window_axis(self) -> int:
        return self.window_size // self.patch_size

    @property
    def tokens_per_chunk_axis(self) -> int:
        return self.chunk_size // self.patch_size


def _freeze_for_inference(
    model: nn.Module,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> nn.Module:
    model.eval().to(device=device, dtype=dtype)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def load_frozen_ink_model(
    checkpoint_path: str | Path,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> nn.Module:
    """Strictly reconstruct one frozen ink model, preferring EMA weights."""

    source = Path(checkpoint_path)
    payload = load_checkpoint(source)
    config = config_from_checkpoint(payload, source=source)
    model = make_model(config)
    _, state = select_inference_weights(payload, source=source)
    load_model_state(model, state)
    return _freeze_for_inference(
        model,
        device=torch.device(device),
        dtype=dtype,
    )


def load_frozen_dino_backbone(
    checkpoint_path: str | Path,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> nn.Module:
    """Reconstruct the Dinovol teacher backbone from its training checkpoint."""

    source = Path(checkpoint_path)
    payload = load_checkpoint(source)
    if not isinstance(payload, Mapping):
        raise ValueError(f"DINO checkpoint {source} must contain a mapping")
    config = payload.get("config")
    if not isinstance(config, Mapping) or not isinstance(config.get("model"), Mapping):
        raise ValueError(f"DINO checkpoint {source} is missing config.model")
    teacher = payload.get("teacher")
    if not isinstance(teacher, Mapping):
        raise ValueError(f"DINO checkpoint {source} is missing teacher weights")
    state = {
        str(key).removeprefix("backbone."): value
        for key, value in teacher.items()
        if str(key).startswith("backbone.")
    }
    if not state:
        raise ValueError(f"DINO checkpoint {source} has no teacher.backbone weights")
    from vesuvius.models.build.pretrained_backbones.dinovol_2_builder import (
        build_dinovol_2_backbone,
    )

    backbone = build_dinovol_2_backbone(config["model"])
    backbone.load_pretrained_weights(state)
    return _freeze_for_inference(
        backbone,
        device=torch.device(device),
        dtype=dtype,
    )


def load_reference_embedding(
    path: str | Path,
    *,
    embedding_dim: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Load and L2-normalize the precomputed one-dimensional ink embedding."""

    array = np.load(str(path)).astype(np.float32)
    if array.shape != (int(embedding_dim),):
        raise ValueError(
            f"reference embedding must have shape ({embedding_dim},), got {array.shape}"
        )
    embedding = torch.from_numpy(array).to(device=device, dtype=torch.float32)
    embedding = embedding / embedding.norm().clamp_min(1e-12)
    return embedding.to(dtype=dtype)


def gaussian_window_3d(size: int, sigma: float) -> torch.Tensor:
    """Return the historical separable, peak-normalized 3D blend window."""

    size = int(size)
    sigma = float(sigma)
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    if sigma <= 0.0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    coordinates = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
    weights_1d = torch.exp(-(coordinates**2) / (2.0 * sigma**2))
    weights_1d /= weights_1d.max()
    return (
        weights_1d[:, None, None]
        * weights_1d[None, :, None]
        * weights_1d[None, None, :]
    )


def dino_window_starts(
    *,
    chunk_size: int,
    window_size: int,
    stride: int,
) -> list[int]:
    """Return starts snapped to include both the first and final valid window."""

    stride = int(stride)
    if stride <= 0:
        raise ValueError(f"DINO stride must be positive, got {stride}")
    last = int(chunk_size) - int(window_size)
    if last < 0:
        raise ValueError("DINO window_size must not exceed chunk_size")
    starts = list(range(0, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)
    return starts


_MISSING = object()


def _config_value(config: Any, key: str, default: Any = _MISSING) -> Any:
    if isinstance(config, Mapping) and key in config:
        return config[key]
    if hasattr(config, key):
        return getattr(config, key)
    if default is _MISSING:
        raise KeyError(f"dynamic-label configuration is missing {key!r}")
    return default


def _config_alias(config: Any, *keys: str) -> Any:
    for key in keys:
        try:
            return _config_value(config, key)
        except KeyError:
            continue
    raise KeyError(
        "dynamic-label configuration is missing one of "
        + ", ".join(repr(key) for key in keys)
    )


def _validate_image(image: torch.Tensor, *, chunk_size: int) -> None:
    expected = (chunk_size, chunk_size, chunk_size)
    if image.ndim != 5 or int(image.shape[1]) != 1:
        raise ValueError(
            "pseudo-label input must have shape [B,1,Z,Y,X], got "
            f"{tuple(image.shape)}"
        )
    if tuple(image.shape[-3:]) != expected:
        raise ValueError(
            f"expected pseudo-label chunk size {chunk_size}^3, got {tuple(image.shape)}"
        )


def _validate_mask(mask: torch.Tensor | None, image: torch.Tensor) -> None:
    if mask is not None and tuple(mask.shape) != tuple(image.shape):
        raise ValueError(
            f"foreground mask must match image shape {tuple(image.shape)}, "
            f"got {tuple(mask.shape)}"
        )


class DinoGuidedLabelGenerator:
    """Intersect a frozen U-Net probability with Dinovol cosine similarity."""

    def __init__(
        self,
        *,
        unet: nn.Module,
        dino: nn.Module,
        reference_embedding: torch.Tensor,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        dino_stride: int = 64,
        dino_minibatch: int = 8,
        dino_blend_sigma: float = 4.0,
        threshold: float = 0.5,
        grid: DinoGridSpec = DinoGridSpec(),
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        self.grid = grid
        self.dino_minibatch = int(dino_minibatch)
        if self.dino_minibatch <= 0:
            raise ValueError("dino_minibatch must be positive")
        self.threshold = float(threshold)
        self.unet = TargetModel(
            _freeze_for_inference(unet, device=self.device, dtype=dtype)
        )
        self.dino = _freeze_for_inference(dino, device=self.device, dtype=dtype)

        reference_embedding = torch.as_tensor(reference_embedding)
        if reference_embedding.shape != (grid.embedding_dim,):
            raise ValueError(
                "reference embedding must have shape "
                f"({grid.embedding_dim},), got {tuple(reference_embedding.shape)}"
            )
        reference_embedding = reference_embedding.to(
            device=self.device, dtype=torch.float32
        )
        self.reference_embedding = (
            reference_embedding / reference_embedding.norm().clamp_min(1e-12)
        )

        self.starts = dino_window_starts(
            chunk_size=grid.chunk_size,
            window_size=grid.window_size,
            stride=dino_stride,
        )
        if any(start % grid.patch_size for start in self.starts):
            raise ValueError(
                "all DINO window starts must align to the DINO patch lattice; "
                f"starts={self.starts}, patch_size={grid.patch_size}"
            )
        self.weight = gaussian_window_3d(
            grid.tokens_per_window_axis, dino_blend_sigma
        ).to(device=self.device)

    @classmethod
    def from_mapping(
        cls,
        config: Any,
        *,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> DinoGuidedLabelGenerator:
        """Construct from the path-oriented recipe mapping used by training."""

        grid = DinoGridSpec()
        unet = load_frozen_ink_model(
            _config_alias(config, "unet_ckpt", "unet_checkpoint"),
            device=device,
            dtype=dtype,
        )
        dino = load_frozen_dino_backbone(
            _config_alias(config, "dino_ckpt", "dino_checkpoint"),
            device=device,
            dtype=dtype,
        )
        reference = load_reference_embedding(
            _config_alias(config, "ref_embedding", "reference_embedding"),
            embedding_dim=grid.embedding_dim,
            device=device,
            dtype=dtype,
        )
        return cls(
            unet=unet,
            dino=dino,
            reference_embedding=reference,
            device=device,
            dtype=dtype,
            dino_stride=int(_config_value(config, "dino_stride", 64)),
            dino_minibatch=int(_config_value(config, "dino_minibatch", 8)),
            dino_blend_sigma=float(
                _config_value(config, "dino_blend_sigma", 4.0)
            ),
            threshold=float(_config_value(config, "threshold", 0.5)),
            grid=grid,
        )

    @torch.inference_mode()
    def generate(
        self,
        image_b1zyx: torch.Tensor,
        mask_b1zyx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _validate_image(image_b1zyx, chunk_size=self.grid.chunk_size)
        _validate_mask(mask_b1zyx, image_b1zyx)
        image = image_b1zyx.to(
            device=self.device, dtype=self.dtype, non_blocking=True
        )
        logits = self.unet(image)
        if logits.ndim != 5 or int(logits.shape[1]) != 1:
            raise ValueError(
                "frozen U-Net must return one-channel BCZYX logits, got "
                f"{tuple(logits.shape)}"
            )
        probability = logits.float().sigmoid()
        if mask_b1zyx is not None:
            probability *= mask_b1zyx.to(
                device=self.device, dtype=probability.dtype, non_blocking=True
            )
        similarity = self._dino_similarity(image)
        return (probability * similarity > self.threshold).float()

    @torch.inference_mode()
    def _dino_similarity(self, image_b1zyx: torch.Tensor) -> torch.Tensor:
        batch_size = int(image_b1zyx.shape[0])
        grid = self.grid
        coordinates = [
            (z, y, x)
            for z in self.starts
            for y in self.starts
            for x in self.starts
        ]
        windows = torch.empty(
            (
                len(coordinates) * batch_size,
                1,
                grid.window_size,
                grid.window_size,
                grid.window_size,
            ),
            device=self.device,
            dtype=self.dtype,
        )
        for window_index, (z0, y0, x0) in enumerate(coordinates):
            windows[
                window_index * batch_size : (window_index + 1) * batch_size
            ] = image_b1zyx[
                :,
                :,
                z0 : z0 + grid.window_size,
                y0 : y0 + grid.window_size,
                x0 : x0 + grid.window_size,
            ]

        similarities = []
        expected_tokens = grid.tokens_per_window_axis**3
        for start in range(0, int(windows.shape[0]), self.dino_minibatch):
            features = self.dino.forward_features(
                windows[start : start + self.dino_minibatch]
            )
            if not isinstance(features, Mapping) or not isinstance(
                features.get("x_norm_patchtokens"), torch.Tensor
            ):
                raise ValueError(
                    "DINO forward_features must return tensor x_norm_patchtokens"
                )
            tokens = features["x_norm_patchtokens"]
            expected_shape = (expected_tokens, grid.embedding_dim)
            if tuple(tokens.shape[1:]) != expected_shape:
                raise ValueError(
                    "DINO patch tokens must have trailing shape "
                    f"{expected_shape}, got {tuple(tokens.shape[1:])}"
                )
            similarity = F.normalize(tokens.float(), dim=-1) @ self.reference_embedding
            similarities.append(
                similarity.reshape(
                    int(similarity.shape[0]),
                    grid.tokens_per_window_axis,
                    grid.tokens_per_window_axis,
                    grid.tokens_per_window_axis,
                )
            )
        similarity_blocks = torch.cat(similarities, dim=0)

        accumulator = torch.zeros(
            (
                batch_size,
                grid.tokens_per_chunk_axis,
                grid.tokens_per_chunk_axis,
                grid.tokens_per_chunk_axis,
            ),
            device=self.device,
            dtype=torch.float32,
        )
        weight_accumulator = torch.zeros_like(accumulator)
        width = grid.tokens_per_window_axis
        for window_index, (z0, y0, x0) in enumerate(coordinates):
            block = similarity_blocks[
                window_index * batch_size : (window_index + 1) * batch_size
            ]
            oz, oy, ox = (
                z0 // grid.patch_size,
                y0 // grid.patch_size,
                x0 // grid.patch_size,
            )
            slices = (
                slice(None),
                slice(oz, oz + width),
                slice(oy, oy + width),
                slice(ox, ox + width),
            )
            accumulator[slices] += block * self.weight
            weight_accumulator[slices] += self.weight

        similarity_grid = accumulator / weight_accumulator.clamp_min(1e-6)
        similarity_full = F.interpolate(
            similarity_grid.unsqueeze(1),
            size=(grid.chunk_size,) * 3,
            mode="trilinear",
            align_corners=False,
        )
        minimum = similarity_full.amin(dim=(1, 2, 3, 4), keepdim=True)
        maximum = similarity_full.amax(dim=(1, 2, 3, 4), keepdim=True)
        return (similarity_full - minimum) / (maximum - minimum).clamp_min(1e-6)


class SelfDistillLabelGenerator:
    """Generate labels from a primary U-Net and a conditionally used ensemble."""

    def __init__(
        self,
        *,
        primary: nn.Module,
        ensemble: nn.Module,
        primary_threshold: float,
        ensemble_threshold: float,
        mean_hi: float,
        std_lo: float,
        tta: bool = True,
        tta_batch_size: int = 2,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        patch_size_zyx: tuple[int, int, int] = (256, 256, 256),
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        self.primary_threshold = float(primary_threshold)
        self.ensemble_threshold = float(ensemble_threshold)
        self.mean_hi = float(mean_hi)
        self.std_lo = float(std_lo)
        self.patch_size_zyx = tuple(int(value) for value in patch_size_zyx)
        if len(self.patch_size_zyx) != 3 or len(set(self.patch_size_zyx)) != 1:
            raise ValueError("self-distillation requires one cubic 3D patch size")
        self.primary = TargetModel(
            _freeze_for_inference(primary, device=self.device, dtype=dtype)
        )
        self.ensemble = TargetModel(
            _freeze_for_inference(ensemble, device=self.device, dtype=dtype)
        )
        self.variants = tta_variants(bool(tta))
        if int(tta_batch_size) <= 0:
            raise ValueError("tta_batch_size must be positive")
        self.tta_batch_size = max(
            1, min(int(tta_batch_size), len(self.variants))
        )

    @classmethod
    def from_mapping(
        cls,
        config: Any,
        *,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> SelfDistillLabelGenerator:
        """Construct from the path-oriented recipe mapping used by training."""

        primary = load_frozen_ink_model(
            _config_alias(config, "primary_ckpt", "primary_checkpoint"),
            device=device,
            dtype=dtype,
        )
        ensemble = load_frozen_ink_model(
            _config_alias(config, "ensemble_ckpt", "ensemble_checkpoint"),
            device=device,
            dtype=dtype,
        )
        return cls(
            primary=primary,
            ensemble=ensemble,
            primary_threshold=float(
                _config_value(config, "primary_threshold")
            ),
            ensemble_threshold=float(
                _config_value(config, "ensemble_threshold")
            ),
            mean_hi=float(_config_value(config, "mean_hi")),
            std_lo=float(_config_value(config, "std_lo")),
            tta=bool(_config_value(config, "tta", True)),
            tta_batch_size=int(_config_value(config, "tta_batch_size", 2)),
            device=device,
            dtype=dtype,
        )

    @torch.inference_mode()
    def generate(
        self,
        image_b1zyx: torch.Tensor,
        mask_b1zyx: torch.Tensor | None = None,
        *,
        raw_mean: torch.Tensor,
        raw_std: torch.Tensor,
    ) -> torch.Tensor:
        chunk_size = self.patch_size_zyx[0]
        _validate_image(image_b1zyx, chunk_size=chunk_size)
        _validate_mask(mask_b1zyx, image_b1zyx)
        if raw_mean.ndim != 1 or raw_std.ndim != 1:
            raise ValueError("raw_mean and raw_std must be one-dimensional [B] tensors")
        if int(raw_mean.shape[0]) != int(image_b1zyx.shape[0]) or int(
            raw_std.shape[0]
        ) != int(image_b1zyx.shape[0]):
            raise ValueError(
                "raw_mean/raw_std batch size must match the image batch; got "
                f"{tuple(raw_mean.shape)} / {tuple(raw_std.shape)} for "
                f"B={image_b1zyx.shape[0]}"
            )

        image = image_b1zyx.to(
            device=self.device, dtype=self.dtype, non_blocking=True
        )
        output = torch.empty_like(image, dtype=torch.float32, device=self.device)
        for batch_index in range(int(image.shape[0])):
            sample = image[batch_index : batch_index + 1]
            use_ensemble = (
                float(raw_mean[batch_index].item()) > self.mean_hi
                and float(raw_std[batch_index].item()) < self.std_lo
            )
            primary_probability = predict_batch(
                self.primary,
                sample,
                variants=self.variants,
                tta_batch_size=self.tta_batch_size,
                patch_size_zyx=self.patch_size_zyx,
            )
            if use_ensemble:
                ensemble_probability = predict_batch(
                    self.ensemble,
                    sample,
                    variants=self.variants,
                    tta_batch_size=self.tta_batch_size,
                    patch_size_zyx=self.patch_size_zyx,
                )
                probability = 0.5 * (
                    primary_probability + ensemble_probability
                )
                threshold = self.ensemble_threshold
            else:
                probability = primary_probability
                threshold = self.primary_threshold
            if mask_b1zyx is not None:
                probability *= mask_b1zyx[batch_index : batch_index + 1].to(
                    device=self.device,
                    dtype=probability.dtype,
                    non_blocking=True,
                )
            output[batch_index : batch_index + 1] = (
                probability > threshold
            ).float()
        return output


def build_dynamic_label_generator(
    config: Any,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> DinoGuidedLabelGenerator | SelfDistillLabelGenerator:
    """Build either opt-in engine from a mapping or typed config object."""

    kind = _config_value(config, "kind")
    kind = getattr(kind, "value", kind)
    normalized = str(kind).strip().lower()
    if normalized == "dino_guided":
        return DinoGuidedLabelGenerator.from_mapping(
            config,
            device=device,
            dtype=dtype,
        )
    if normalized == "self_distill":
        return SelfDistillLabelGenerator.from_mapping(
            config,
            device=device,
            dtype=dtype,
        )
    raise ValueError(
        f"unsupported dynamic-label kind {kind!r}; expected "
        "'dino_guided' or 'self_distill'"
    )
