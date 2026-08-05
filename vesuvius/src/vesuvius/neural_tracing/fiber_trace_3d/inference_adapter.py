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
        PYRAMID_POLICY_CUSTOM,
    )
except ImportError:  # pragma: no cover - supports PYTHONPATH=lasagna style runs.
    from tiled_predict3d import (
        OutputChannelSpec,
        OutputProductSpec,
        ProductTileOutput,
        PYRAMID_POLICY_CUSTOM,
    )

try:
    from lasagna.normal_encoding import encode_normal_nxny_u8
except ImportError:  # pragma: no cover - supports PYTHONPATH=lasagna style runs.
    from normal_encoding import encode_normal_nxny_u8

from vesuvius.neural_tracing.fiber_trace_3d.direction import LASAGNA_3X2_CHANNELS
from vesuvius.neural_tracing.fiber_trace_3d.model import build_fiber_trace_3d_model


FIBER_TRACE_3D_INTERNAL_CHANNELS: tuple[str, ...] = (
    *LASAGNA_3X2_CHANNELS,
    "presence",
)
FIBER_TRACE_3D_PERSISTED_CHANNELS: tuple[str, ...] = ("presence", "nx", "ny")


def _sanitize_product_prefix(value: str) -> str:
    prefix = str(value).strip()
    if not prefix:
        raise ValueError("product_prefix must be non-empty")
    if "/" in prefix or "\\" in prefix:
        raise ValueError("product_prefix must be a simple product-name prefix")
    return prefix


def _model_cfg_from_raw(config: Mapping[str, Any]) -> dict[str, Any]:
    return dict(config.get("model_3d", config.get("model", {})))


def _checkpoint_payload(path: str | Path) -> Any:
    return torch.load(path, map_location="cpu")


def _checkpoint_state(payload: Any) -> Mapping[str, Any] | None:
    state = payload.get("model", payload) if isinstance(payload, Mapping) else payload
    return state if isinstance(state, Mapping) else None


def _checkpoint_config(payload: Any) -> dict[str, Any] | None:
    if isinstance(payload, Mapping) and isinstance(payload.get("config"), Mapping):
        return dict(payload["config"])
    return None


def _config_from_checkpoint(
    runtime_config: Mapping[str, Any],
    checkpoint: str | Path | None,
) -> dict[str, Any]:
    config = dict(runtime_config)
    if checkpoint is None:
        return config
    payload = _checkpoint_payload(checkpoint)
    checkpoint_config = _checkpoint_config(payload)
    if checkpoint_config is not None:
        effective = dict(config)
        effective.update(checkpoint_config)
        effective.setdefault("_config_dir", config.get("_config_dir"))
        if not _conditioned_decoder_enabled(effective):
            inference_cfg = dict(effective.get("inference", {}))
            inference_cfg["recurrent_steps"] = 1
            effective["inference"] = inference_cfg
        return effective
    state = _checkpoint_state(payload)
    if state is None:
        return config
    weight = state.get("net.decoder.final_seg_layer.weight")
    if not isinstance(weight, torch.Tensor):
        return config
    output_channels = int(weight.shape[0])
    if output_channels <= 0 or output_channels % 7 != 0:
        return config
    has_conditioned_decoder = any(str(key).startswith("conditioned_decoder.") for key in state)
    if has_conditioned_decoder:
        return config
    effective = dict(config)
    model_cfg = _model_cfg_from_raw(effective)
    model_cfg["conditioned_decoder_enabled"] = False
    model_cfg["output_channels"] = output_channels
    model_cfg["direction_branch_count"] = output_channels // 7
    effective["model_3d"] = model_cfg
    inference_cfg = dict(effective.get("inference", {}))
    inference_cfg["recurrent_steps"] = 1
    effective["inference"] = inference_cfg
    return effective


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

    Each emitted option accumulates seven raw model channels internally and
    persists a Lasagna-style ``presence/nx/ny`` product.
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        checkpoint: str | Path | None = None,
        level: int = 0,
        scaledown: int = 1,
        inference_scaledown: int = 1,
        chunk_size: int = 64,
        product_prefix: str = "fiber",
        zarr_path_prefix: str | Path | None = None,
        recurrent_steps: int | None = None,
    ) -> None:
        self.checkpoint = None if checkpoint is None else Path(checkpoint)
        self.config = _config_from_checkpoint(config, self.checkpoint)
        self.recurrent_steps = _recurrent_steps_from_config(
            self.config,
            recurrent_steps,
        )
        self.option_count = _option_count_from_config(
            self.config,
            recurrent_steps=self.recurrent_steps,
        )
        self.product_prefix = _sanitize_product_prefix(product_prefix)
        self.zarr_path_prefix = (
            self.product_prefix
            if zarr_path_prefix is None
            else str(zarr_path_prefix)
        )
        self._products = tuple(
            self._make_option_product(
                option_index,
                level=int(level),
                scaledown=int(scaledown),
                inference_scaledown=int(inference_scaledown),
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
        if tile.ndim != 5 or int(tile.shape[1]) != 1:
            raise ValueError("fiber inference tile must have shape B,1,D,H,W")
        if valid_mask.shape != tile.shape:
            raise ValueError("fiber inference valid_mask must match tile shape")
        normalized = [
            _normalize_image(tile[index, 0], valid_mask[index, 0].to(dtype=torch.bool), normalization)
            for index in range(int(tile.shape[0]))
        ]
        return torch.stack(normalized, dim=0).view_as(tile)

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
        inference_scaledown: int,
        chunk_size: int,
    ) -> OutputProductSpec:
        option_name = f"option_{int(option_index):03d}"
        if self.option_count == 1:
            persisted_names = FIBER_TRACE_3D_PERSISTED_CHANNELS
        else:
            persisted_names = tuple(
                f"{option_name}_{channel}"
                for channel in FIBER_TRACE_3D_PERSISTED_CHANNELS
            )
        return OutputProductSpec(
            name=f"{self.product_prefix}_{option_name}",
            level=level,
            scaledown=scaledown,
            channels=tuple(
                OutputChannelSpec(
                    channel,
                    relative_path=f"{self.zarr_path_prefix}_{channel}.ome.zarr",
                )
                for channel in persisted_names
            ),
            chunk_size=chunk_size,
            dtype=np.uint8,
            value_range=(0.0, 255.0),
            pyramid_policy=PYRAMID_POLICY_CUSTOM,
            accumulator_channel_count=len(FIBER_TRACE_3D_INTERNAL_CHANNELS),
            inference_scaledown=inference_scaledown,
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

    def finalize_product_slab(
        self,
        product: OutputProductSpec,
        raw_slab: np.ndarray,
    ) -> ProductTileOutput:
        slab = np.asarray(raw_slab, dtype=np.float32)
        if slab.ndim != 4 or int(slab.shape[0]) != len(FIBER_TRACE_3D_INTERNAL_CHANNELS):
            raise ValueError(
                "fiber raw product slab must have shape 7,D,H,W; "
                f"got {slab.shape}"
            )
        presence_u8 = np.clip(
            np.round(np.clip(slab[6], 0.0, 1.0) * 255.0),
            0.0,
            255.0,
        ).astype(np.uint8)
        nx_u8, ny_u8 = encode_normal_nxny_u8(
            slab[0],
            slab[1],
            slab[2],
            slab[3],
            slab[4],
            slab[5],
        )
        arrays_by_base = {
            "presence": presence_u8,
            "nx": nx_u8,
            "ny": ny_u8,
        }
        out: dict[str, np.ndarray] = {}
        for channel in product.channels:
            for base_name, array in arrays_by_base.items():
                if channel.name == base_name or channel.name.endswith(f"_{base_name}"):
                    out[channel.name] = array
                    break
        return out



__all__ = [
    "FIBER_TRACE_3D_INTERNAL_CHANNELS",
    "FIBER_TRACE_3D_PERSISTED_CHANNELS",
    "FiberTrace3DPredictAdapter",
]
