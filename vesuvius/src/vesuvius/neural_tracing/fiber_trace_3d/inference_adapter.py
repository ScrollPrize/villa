from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

try:
    from lasagna.tiled_predict3d import (
        OutputChannelSpec,
        OutputProductSpec,
        ProductTileOutput,
        PYRAMID_POLICY_NONE,
        _atomic_zarr_write,
        _omezarr_chunk_group_complete,
    )
except ImportError:  # pragma: no cover - supports PYTHONPATH=lasagna style runs.
    from tiled_predict3d import (
        OutputChannelSpec,
        OutputProductSpec,
        ProductTileOutput,
        PYRAMID_POLICY_NONE,
        _atomic_zarr_write,
        _omezarr_chunk_group_complete,
    )

from vesuvius.neural_tracing.fiber_trace_3d.direction import LASAGNA_3X2_CHANNELS
from vesuvius.neural_tracing.fiber_trace_3d.model import build_fiber_trace_3d_model


FIBER_TRACE_3D_OPTION_CHANNELS: tuple[str, ...] = (
    *LASAGNA_3X2_CHANNELS,
    "presence",
)


def _sanitize_output_prefix(value: str) -> str:
    prefix = str(value).strip().strip("/")
    if not prefix:
        raise ValueError("output_prefix must be non-empty")
    return prefix


def _model_cfg_from_raw(config: Mapping[str, Any]) -> dict[str, Any]:
    return dict(config.get("model_3d", config.get("model", {})))


def _conditioned_decoder_enabled(config: Mapping[str, Any]) -> bool:
    return bool(_model_cfg_from_raw(config).get("conditioned_decoder_enabled", False))


def _configured_branch_count(config: Mapping[str, Any]) -> int:
    model_cfg = _model_cfg_from_raw(config)
    if "output_channels" in model_cfg:
        output_channels = int(model_cfg["output_channels"])
        if output_channels <= 0 or output_channels % 7 != 0:
            raise ValueError("model_3d.output_channels must be a positive multiple of 7")
        if "direction_branch_count" in model_cfg:
            branch_count = int(model_cfg["direction_branch_count"])
            if output_channels != branch_count * 7:
                raise ValueError(
                    "model_3d.output_channels must equal direction_branch_count * 7"
                )
            return branch_count
        return output_channels // 7
    branch_count = int(model_cfg.get("direction_branch_count", 1))
    if branch_count <= 0:
        raise ValueError("model_3d.direction_branch_count must be > 0")
    return branch_count


def _recurrent_steps_from_config(
    config: Mapping[str, Any],
    explicit_steps: int | None,
) -> int:
    if explicit_steps is not None:
        steps = int(explicit_steps)
    else:
        inference_cfg = dict(config.get("inference", {}))
        steps = int(inference_cfg.get("recurrent_steps", 1))
    if steps <= 0:
        raise ValueError("recurrent_steps must be > 0")
    return steps


def _option_count_from_config(
    config: Mapping[str, Any],
    *,
    recurrent_steps: int | None,
) -> int:
    steps = _recurrent_steps_from_config(config, recurrent_steps)
    if _conditioned_decoder_enabled(config):
        return steps
    if steps != 1:
        raise ValueError(
            "recurrent_steps > 1 requires model_3d.conditioned_decoder_enabled=true"
        )
    return _configured_branch_count(config)


class FiberTrace3DPredictAdapter:
    """Model adapter for 3D fiber inference.

    Each emitted option is a coherent seven-channel product:
    Lasagna 3x2 direction channels followed by scalar fiber presence.
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        checkpoint: str | Path | None = None,
        level: int = 0,
        scaledown: int = 1,
        chunk_size: int = 64,
        output_prefix: str = "fiber",
        recurrent_steps: int | None = None,
    ) -> None:
        self.config = dict(config)
        self.checkpoint = None if checkpoint is None else Path(checkpoint)
        self.recurrent_steps = _recurrent_steps_from_config(
            self.config,
            recurrent_steps,
        )
        self.option_count = _option_count_from_config(
            self.config,
            recurrent_steps=self.recurrent_steps,
        )
        self.output_prefix = _sanitize_output_prefix(output_prefix)
        self._products = tuple(
            self._make_option_product(
                option_index,
                level=int(level),
                scaledown=int(scaledown),
                chunk_size=int(chunk_size),
            )
            for option_index in range(self.option_count)
        )

    @classmethod
    def from_config_file(
        cls,
        config_path: str | Path,
        **kwargs: Any,
    ) -> "FiberTrace3DPredictAdapter":
        path = Path(config_path)
        with path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        if not isinstance(config, dict):
            raise ValueError(f"{path} must contain a JSON object")
        config.setdefault("_config_dir", str(path.parent))
        return cls(config, **kwargs)

    @property
    def output_products(self) -> tuple[OutputProductSpec, ...]:
        return self._products

    def preprocess_tile(
        self,
        tile: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        from vesuvius.neural_tracing.fiber_trace_3d.loader import _normalize_image

        normalization = str(self.config.get("image_normalization", "zscore"))
        if tile.ndim != 5 or int(tile.shape[0]) != 1 or int(tile.shape[1]) != 1:
            raise ValueError("fiber inference tile must have shape 1,1,D,H,W")
        image = tile[0, 0]
        valid = valid_mask[0, 0].to(dtype=torch.bool)
        return _normalize_image(image, valid, normalization).view_as(tile)

    def product_by_name(self, name: str) -> OutputProductSpec:
        for product in self._products:
            if product.name == name:
                return product
        raise KeyError(name)

    def _make_option_product(
        self,
        option_index: int,
        *,
        level: int,
        scaledown: int,
        chunk_size: int,
    ) -> OutputProductSpec:
        option_name = f"option_{int(option_index):03d}"
        return OutputProductSpec(
            name=f"{self.output_prefix}_{option_name}",
            level=level,
            scaledown=scaledown,
            channels=tuple(
                OutputChannelSpec(
                    channel,
                    relative_path=f"{self.output_prefix}/{option_name}/{channel}",
                )
                for channel in FIBER_TRACE_3D_OPTION_CHANNELS
            ),
            chunk_size=chunk_size,
            dtype=np.uint8,
            value_range=(0.0, 255.0),
            pyramid_policy=PYRAMID_POLICY_NONE,
        )

    def load_model(self, *, device: torch.device) -> torch.nn.Module:
        model = build_fiber_trace_3d_model(self.config).to(device)
        if self.checkpoint is not None:
            from vesuvius.neural_tracing.fiber_trace_3d.train import _load_snapshot

            _load_snapshot(
                self.checkpoint,
                model=model,
                optimizer=None,
                map_location=device,
            )
        model.eval()
        return model

    @torch.no_grad()
    def run_tile_inference(
        self,
        model: torch.nn.Module,
        tile: torch.Tensor,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        model.eval()
        tile_device = tile.to(device=device, non_blocking=True)
        training_cfg = dict(self.config.get("training", {}))
        from vesuvius.neural_tracing.fiber_trace_3d.train import (
            _autocast_context,
            _mixed_precision_config_from_training,
        )

        precision = _mixed_precision_config_from_training(training_cfg, device)
        with _autocast_context(precision):
            if (
                _conditioned_decoder_enabled(self.config)
                and self.recurrent_steps > 1
            ):
                if not hasattr(model, "forward_recurrent_grouped"):
                    raise ValueError(
                        "conditioned recurrent inference requires "
                        "forward_recurrent_grouped on the model"
                    )
                output = model.forward_recurrent_grouped(
                    tile_device,
                    steps=int(self.recurrent_steps),
                )
            else:
                output = model(tile_device)
        return output.to(dtype=torch.float32).clamp(0.0, 1.0)

    def product_tensors_from_output(
        self,
        raw_output: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        output = raw_output.to(dtype=torch.float32)
        if output.ndim == 6:
            batch, options, channels, depth, height, width = (int(v) for v in output.shape)
            if channels != 7:
                raise ValueError("recurrent raw output must have shape B,S,7,D,H,W")
            output = output.reshape(batch, options * channels, depth, height, width)
        if output.ndim != 5:
            raise ValueError("fiber model output must have shape B,C,D,H,W")
        channels = int(output.shape[1])
        expected_channels = int(self.option_count) * 7
        if channels != expected_channels:
            raise ValueError(
                "fiber model output channel count does not match configured options: "
                f"got {channels}, expected {expected_channels}"
            )
        return {
            product.name: output[:, option_index * 7 : option_index * 7 + 7]
            for option_index, product in enumerate(self.output_products)
        }

    def product_channel_arrays_from_output(
        self,
        raw_output: torch.Tensor,
        *,
        batch_index: int = 0,
    ) -> dict[str, dict[str, np.ndarray]]:
        products = self.product_tensors_from_output(raw_output)
        out: dict[str, dict[str, np.ndarray]] = {}
        for product in self.output_products:
            tensor = products[product.name]
            batch = int(batch_index)
            if batch < 0 or batch >= int(tensor.shape[0]):
                raise IndexError(
                    f"batch_index={batch} is outside output batch size {int(tensor.shape[0])}"
                )
            bundle = tensor[batch].detach().cpu().clamp(0.0, 1.0)
            channels: dict[str, np.ndarray] = {}
            for channel_index, channel in enumerate(product.channels):
                channels[channel.name] = (
                    torch.round(bundle[channel_index] * 255.0)
                    .to(dtype=torch.uint8)
                    .numpy()
                )
            out[product.name] = channels
        return out

    def accumulate_tile_output(
        self,
        raw_output: Any,
        *,
        tile_origin_zyx: tuple[int, int, int],
        tile_weight: torch.Tensor | np.ndarray,
        accumulators: Mapping[str, Any],
    ) -> None:
        del tile_origin_zyx
        products = self.product_tensors_from_output(raw_output)
        for product_name, tensor in products.items():
            target = accumulators.get(product_name)
            if target is None:
                continue
            weight = tile_weight
            if isinstance(target, torch.Tensor):
                weight_t = torch.as_tensor(
                    weight,
                    dtype=target.dtype,
                    device=target.device,
                )
                target += tensor.to(dtype=target.dtype, device=target.device) * weight_t
            elif isinstance(target, np.ndarray):
                weight_np = np.asarray(weight, dtype=target.dtype)
                target += tensor.detach().cpu().numpy().astype(target.dtype) * weight_np
            else:
                raise TypeError(
                    f"unsupported accumulator type for product {product_name!r}: "
                    f"{type(target).__name__}"
                )


class FiberTrace3DOmeZarrOutputAdapter:
    """Chunk completeness and atomic write adapter for fiber option bundles."""

    def __init__(
        self,
        *,
        output_root: str | Path,
        products: Sequence[OutputProductSpec],
        n_levels: int = 0,
    ) -> None:
        self.output_root = Path(output_root)
        self.products = tuple(products)
        self.n_levels = int(n_levels)

    def _channel_path(
        self,
        product: OutputProductSpec,
        channel: OutputChannelSpec,
    ) -> str:
        rel = channel.relative_path or channel.name
        path = Path(rel)
        return str(path if path.is_absolute() else self.output_root / path)

    def channel_path(
        self,
        product: OutputProductSpec,
        channel: OutputChannelSpec,
    ) -> str:
        return self._channel_path(product, channel)

    def _channel_paths(self, product: OutputProductSpec) -> tuple[str, ...]:
        return tuple(self._channel_path(product, channel) for channel in product.channels)

    def product_chunk_complete(
        self,
        product: OutputProductSpec,
        *,
        chunk_origin_zyx: tuple[int, int, int],
    ) -> bool:
        z, y, x = (int(v) for v in chunk_origin_zyx)
        return _omezarr_chunk_group_complete(
            self._channel_paths(product),
            int(product.level),
            z,
            y,
            x,
            int(product.chunk_size),
        )

    def write_product_chunk(
        self,
        product: OutputProductSpec,
        *,
        chunk_origin_zyx: tuple[int, int, int],
        data: ProductTileOutput,
    ) -> None:
        missing = [channel.name for channel in product.channels if channel.name not in data]
        if missing:
            raise ValueError(
                f"product {product.name!r} write is missing channels: {missing}"
            )
        z0, y0, x0 = (int(v) for v in chunk_origin_zyx)
        for channel in product.channels:
            chunk = np.asarray(data[channel.name])
            if chunk.ndim != 3:
                raise ValueError(
                    f"fiber output channel {channel.name!r} chunk must be 3D"
                )
            z1 = z0 + int(chunk.shape[0])
            y1 = y0 + int(chunk.shape[1])
            x1 = x0 + int(chunk.shape[2])
            _atomic_zarr_write(
                self._channel_path(product, channel),
                int(product.level),
                z0,
                y0,
                x0,
                z1,
                y1,
                x1,
                chunk.astype(product.dtype, copy=False),
                int(product.chunk_size),
                n_levels=self.n_levels,
            )

    def update_metadata(self, products: Sequence[OutputProductSpec]) -> None:
        del products


__all__ = [
    "FIBER_TRACE_3D_OPTION_CHANNELS",
    "FiberTrace3DPredictAdapter",
    "FiberTrace3DOmeZarrOutputAdapter",
]
