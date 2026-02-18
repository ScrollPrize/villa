from dataclasses import dataclass

import numpy as np


_UV_STRUCT_DTYPE = np.dtype([("r", np.int64), ("c", np.int64)])


def _empty_uv():
    return np.zeros((0, 2), dtype=np.int64)


def _empty_world():
    return np.zeros((0, 3), dtype=np.float32)


def _uv_struct_view(uv):
    uv_int = np.ascontiguousarray(np.asarray(uv, dtype=np.int64))
    if uv_int.ndim != 2 or uv_int.shape[1] != 2:
        return np.zeros((0,), dtype=_UV_STRUCT_DTYPE)
    return uv_int.view(_UV_STRUCT_DTYPE).reshape(-1)


@dataclass(frozen=True)
class ExtrapLookupArrays:
    uv: np.ndarray
    world: np.ndarray
    lookup_sort_idx: np.ndarray
    lookup_uv_sorted: np.ndarray

    def __post_init__(self):
        uv = np.asarray(self.uv, dtype=np.int64)
        world = np.asarray(self.world, dtype=np.float32)
        if uv.ndim != 2 or uv.shape[1] != 2:
            raise ValueError(f"uv must have shape (N, 2); got {tuple(uv.shape)}")
        if world.ndim != 2 or world.shape[1] != 3:
            raise ValueError(f"world must have shape (N, 3); got {tuple(world.shape)}")
        if uv.shape[0] != world.shape[0]:
            raise ValueError(
                f"uv/world row count mismatch: {uv.shape[0]} vs {world.shape[0]}"
            )

        n_rows = int(uv.shape[0])
        lookup_sort_idx = np.asarray(self.lookup_sort_idx, dtype=np.int64)
        if lookup_sort_idx.ndim != 1:
            raise ValueError(
                "lookup_sort_idx must be a 1D array; "
                f"got {tuple(lookup_sort_idx.shape)}"
            )
        if lookup_sort_idx.shape[0] != n_rows:
            raise ValueError(
                "lookup_sort_idx length must match uv rows: "
                f"{lookup_sort_idx.shape[0]} vs {n_rows}"
            )
        if n_rows > 0:
            sort_idx_sorted = np.sort(lookup_sort_idx)
            expected_idx = np.arange(n_rows, dtype=np.int64)
            if not np.array_equal(sort_idx_sorted, expected_idx):
                raise ValueError("lookup_sort_idx must be a permutation of [0, N).")

        lookup_uv_sorted = np.asarray(self.lookup_uv_sorted)
        if lookup_uv_sorted.ndim != 1:
            raise ValueError(
                "lookup_uv_sorted must be 1D; "
                f"got {tuple(lookup_uv_sorted.shape)}"
            )
        if lookup_uv_sorted.shape[0] != n_rows:
            raise ValueError(
                "lookup_uv_sorted length must match uv rows: "
                f"{lookup_uv_sorted.shape[0]} vs {n_rows}"
            )
        if lookup_uv_sorted.dtype != _UV_STRUCT_DTYPE:
            raise ValueError(
                "lookup_uv_sorted must use UV struct dtype "
                f"{_UV_STRUCT_DTYPE}; got {lookup_uv_sorted.dtype}"
            )

        expected_sorted = _uv_struct_view(uv[lookup_sort_idx])
        if not np.array_equal(lookup_uv_sorted, expected_sorted):
            raise ValueError(
                "lookup_uv_sorted must match uv rows indexed by lookup_sort_idx."
            )

        object.__setattr__(self, "uv", uv)
        object.__setattr__(self, "world", world)
        object.__setattr__(self, "lookup_sort_idx", lookup_sort_idx)
        object.__setattr__(self, "lookup_uv_sorted", lookup_uv_sorted)
