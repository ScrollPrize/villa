import numpy as np
import torch


GROWTH_DIRECTION_ORDER = ("left", "right", "up", "down")
_GROWTH_DIRECTION_TO_INDEX = {
    direction: idx for idx, direction in enumerate(GROWTH_DIRECTION_ORDER)
}


def growth_direction_channel_count() -> int:
    return len(GROWTH_DIRECTION_ORDER)


def growth_direction_one_hot(direction: str) -> np.ndarray:
    direction_key = str(direction).lower()
    if direction_key not in _GROWTH_DIRECTION_TO_INDEX:
        allowed = "', '".join(GROWTH_DIRECTION_ORDER)
        raise ValueError(f"growth direction must be one of '{allowed}', got {direction!r}")
    out = np.zeros((len(GROWTH_DIRECTION_ORDER),), dtype=np.float32)
    out[_GROWTH_DIRECTION_TO_INDEX[direction_key]] = 1.0
    return out


def make_growth_direction_tensor(
    directions,
    spatial_shape,
    *,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    spatial_shape = tuple(int(v) for v in spatial_shape)
    if len(spatial_shape) != 3:
        raise ValueError(f"spatial_shape must have length 3, got {spatial_shape!r}")

    if isinstance(directions, str):
        directions = [directions]
    dirs = list(directions)
    one_hot = np.stack([growth_direction_one_hot(direction) for direction in dirs], axis=0)
    tensor = torch.as_tensor(one_hot, device=device, dtype=dtype)
    return tensor[:, :, None, None, None].expand(-1, -1, *spatial_shape)


def make_growth_direction_array(direction: str, spatial_shape) -> np.ndarray:
    spatial_shape = tuple(int(v) for v in spatial_shape)
    if len(spatial_shape) != 3:
        raise ValueError(f"spatial_shape must have length 3, got {spatial_shape!r}")
    one_hot = growth_direction_one_hot(direction)
    return np.broadcast_to(one_hot[:, None, None, None], (one_hot.shape[0], *spatial_shape)).copy()
