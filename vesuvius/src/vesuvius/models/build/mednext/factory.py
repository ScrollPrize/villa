from __future__ import annotations

from copy import deepcopy

from .v1 import MedNeXtV1


_MEDNEXT_V1_PRESETS: dict[str, dict] = {
    "S": {
        "n_channels": 32,
        "exp_r": 2,
        "block_counts": [2, 2, 2, 2, 2, 2, 2, 2, 2],
        "checkpoint_style": None,
    },
    "B": {
        "n_channels": 32,
        "exp_r": [2, 3, 4, 4, 4, 4, 4, 3, 2],
        "block_counts": [2, 2, 2, 2, 2, 2, 2, 2, 2],
        "checkpoint_style": None,
    },
    "M": {
        "n_channels": 32,
        "exp_r": [2, 3, 4, 4, 4, 4, 4, 3, 2],
        "block_counts": [3, 4, 4, 4, 4, 4, 4, 4, 3],
        "checkpoint_style": "outside_block",
    },
    "L": {
        "n_channels": 32,
        "exp_r": [3, 4, 8, 8, 8, 8, 8, 4, 3],
        "block_counts": [3, 4, 8, 8, 8, 8, 8, 4, 3],
        "checkpoint_style": "outside_block",
    },
}


def get_mednext_v1_config(model_id: str) -> dict:
    key = str(model_id).strip().upper()
    if key not in _MEDNEXT_V1_PRESETS:
        raise ValueError(
            f"Unknown mednext_model_id {model_id!r}. Expected one of {sorted(_MEDNEXT_V1_PRESETS)}"
        )
    cfg = deepcopy(_MEDNEXT_V1_PRESETS[key])
    cfg["model_id"] = key
    return cfg


def create_mednext_v1(
    num_input_channels: int,
    num_classes: int,
    model_id: str,
    *,
    kernel_size: int = 3,
    deep_supervision: bool = False,
    checkpoint_style: str | None = None,
) -> MedNeXtV1:
    cfg = get_mednext_v1_config(model_id)
    if checkpoint_style is None:
        checkpoint_style = cfg["checkpoint_style"]
    return MedNeXtV1(
        in_channels=num_input_channels,
        n_channels=cfg["n_channels"],
        n_classes=num_classes,
        exp_r=cfg["exp_r"],
        kernel_size=int(kernel_size),
        deep_supervision=bool(deep_supervision),
        do_res=True,
        do_res_up_down=True,
        checkpoint_style=checkpoint_style,
        block_counts=cfg["block_counts"],
        norm_type="group",
        grn=False,
    )
