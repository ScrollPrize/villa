"""Build a vesuvius NetworkFromConfig UNet for the 4-class fiber/ink task.

Standalone helper (no dependency on sibling script directories) so this
training package can be reviewed and run on its own. Mirrors the lightweight
ConfigManager shim that vesuvius' NetworkFromConfig expects.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Mapping

from vesuvius.models.build.build_network_from_config import NetworkFromConfig


def _config_dict_to_mgr(
    *,
    crop_size: tuple[int, int, int],
    in_channels: int,
    targets: dict[str, Any],
    model_config: Mapping[str, Any] | None = None,
) -> SimpleNamespace:
    """Reconstruct just enough of a vesuvius ConfigManager for NetworkFromConfig."""
    mgr = SimpleNamespace()
    mgr.model_config = dict(model_config or {})
    mgr.train_patch_size = tuple(crop_size)
    mgr.train_batch_size = 1
    mgr.in_channels = int(in_channels)
    mgr.model_name = "fiber_unet"
    mgr.autoconfigure = bool(mgr.model_config.get("autoconfigure", True))
    mgr.spacing = mgr.model_config.get("spacing", [1, 1, 1])
    mgr.targets = targets
    mgr.enable_deep_supervision = False
    mgr.op_dims = 3
    return mgr


def build_fiber_unet(
    *,
    crop_size: tuple[int, int, int] = (256, 256, 256),
    target_name: str = "labels",
    out_channels: int = 4,
    activation: str = "none",
    in_channels: int = 1,
) -> NetworkFromConfig:
    """Build a NetworkFromConfig 3D UNet with a single segmentation head."""
    targets = {target_name: {"out_channels": out_channels, "activation": activation}}
    mgr = _config_dict_to_mgr(
        crop_size=crop_size,
        in_channels=in_channels,
        targets=targets,
    )
    return NetworkFromConfig(mgr)
