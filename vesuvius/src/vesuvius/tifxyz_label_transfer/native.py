"""Optional compiled per-pixel rasterizer for TIFXYZ label transfer."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import sys
from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray


ABI_VERSION = 1


class _RasterRequest(ctypes.Structure):
    _fields_ = [
        ("target_x", ctypes.c_void_p),
        ("target_y", ctypes.c_void_p),
        ("target_z", ctypes.c_void_p),
        ("target_valid", ctypes.c_void_p),
        ("uv_rows", ctypes.c_void_p),
        ("uv_cols", ctypes.c_void_p),
        ("uv_valid", ctypes.c_void_p),
        ("source_x", ctypes.c_void_p),
        ("source_y", ctypes.c_void_p),
        ("source_z", ctypes.c_void_p),
        ("source_valid", ctypes.c_void_p),
        ("filled_uv_rows", ctypes.c_void_p),
        ("filled_uv_cols", ctypes.c_void_p),
        ("source_validity", ctypes.c_void_p),
        ("output_source_indices", ctypes.c_void_p),
        ("output_validity", ctypes.c_void_p),
        ("output_distances", ctypes.c_void_p),
        ("target_height", ctypes.c_int64),
        ("target_width", ctypes.c_int64),
        ("source_height", ctypes.c_int64),
        ("source_width", ctypes.c_int64),
        ("label_height", ctypes.c_int64),
        ("label_width", ctypes.c_int64),
        ("output_height", ctypes.c_int64),
        ("output_width", ctypes.c_int64),
        ("row_start", ctypes.c_int64),
        ("row_end", ctypes.c_int64),
        ("col_start", ctypes.c_int64),
        ("col_end", ctypes.c_int64),
        ("label_offset_y", ctypes.c_double),
        ("label_offset_x", ctypes.c_double),
        ("max_distance", ctypes.c_double),
        ("fill_seams", ctypes.c_uint32),
        ("has_source_validity", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
    ]


class _RasterResult(ctypes.Structure):
    _fields_ = [
        ("target_surface_valid", ctypes.c_int64),
        ("measured_pixels", ctypes.c_int64),
        ("seam_filled_pixels", ctypes.c_int64),
        ("inherited_filled_pixels", ctypes.c_int64),
    ]


@dataclass(frozen=True)
class NativeTileResult:
    source_indices: NDArray[np.int64]
    validity: NDArray[np.uint8]
    distances: NDArray[np.float64]
    target_surface_valid: int
    measured_pixels: int
    seam_filled_pixels: int
    inherited_filled_pixels: int


_loaded_path: Optional[Path] = None
_loaded_library: Optional[ctypes.CDLL] = None
_load_error: Optional[str] = None


def native_library_name() -> str:
    return "_native_rasterizer.so"


def default_native_library_path() -> Path:
    return Path(__file__).with_name(native_library_name())


def native_source_fingerprint() -> str:
    source = Path(__file__).with_name("native_rasterizer.cpp")
    return hashlib.sha256(source.read_bytes()).hexdigest()[:16]


def _candidate_library_path() -> Path:
    override = os.environ.get("TIFXYZ_LABEL_TRANSFER_NATIVE")
    return Path(override) if override else default_native_library_path()


def reset_native_library_cache() -> None:
    """Forget the loaded library; intended for build tools and tests."""

    global _loaded_path, _loaded_library, _load_error
    _loaded_path = None
    _loaded_library = None
    _load_error = None


def load_native_library() -> Optional[ctypes.CDLL]:
    global _loaded_path, _loaded_library, _load_error
    candidate = _candidate_library_path().resolve()
    if _loaded_path == candidate:
        return _loaded_library
    _loaded_path = candidate
    _loaded_library = None
    _load_error = None
    if not candidate.is_file():
        _load_error = f"native rasterizer was not built at {candidate}"
        return None
    try:
        library = ctypes.CDLL(str(candidate))
        library.vc_tifxyz_rasterizer_abi_version.argtypes = []
        library.vc_tifxyz_rasterizer_abi_version.restype = ctypes.c_uint32
        actual_abi = int(library.vc_tifxyz_rasterizer_abi_version())
        if actual_abi != ABI_VERSION:
            _load_error = (
                f"native rasterizer ABI {actual_abi} does not match "
                f"Python ABI {ABI_VERSION}; rebuild it"
            )
            return None
        library.vc_tifxyz_rasterizer_source_fingerprint.argtypes = []
        library.vc_tifxyz_rasterizer_source_fingerprint.restype = ctypes.c_char_p
        actual_fingerprint = (
            library.vc_tifxyz_rasterizer_source_fingerprint().decode("ascii")
        )
        expected_fingerprint = native_source_fingerprint()
        if actual_fingerprint != expected_fingerprint:
            _load_error = (
                f"native rasterizer source {actual_fingerprint} does not "
                f"match current source {expected_fingerprint}; rebuild it"
            )
            return None
        library.vc_tifxyz_rasterizer_request_size.argtypes = []
        library.vc_tifxyz_rasterizer_request_size.restype = ctypes.c_size_t
        library.vc_tifxyz_rasterizer_result_size.argtypes = []
        library.vc_tifxyz_rasterizer_result_size.restype = ctypes.c_size_t
        request_size = int(library.vc_tifxyz_rasterizer_request_size())
        result_size = int(library.vc_tifxyz_rasterizer_result_size())
        if request_size != ctypes.sizeof(_RasterRequest):
            _load_error = (
                f"native request size {request_size} does not match "
                f"Python size {ctypes.sizeof(_RasterRequest)}; rebuild it"
            )
            return None
        if result_size != ctypes.sizeof(_RasterResult):
            _load_error = (
                f"native result size {result_size} does not match "
                f"Python size {ctypes.sizeof(_RasterResult)}; rebuild it"
            )
            return None
        library.vc_tifxyz_rasterize.argtypes = [
            ctypes.POINTER(_RasterRequest),
            ctypes.POINTER(_RasterResult),
        ]
        library.vc_tifxyz_rasterize.restype = ctypes.c_int
    except (AttributeError, OSError) as error:
        _load_error = f"could not load native rasterizer {candidate}: {error}"
        return None
    _loaded_library = library
    return library


def native_unavailable_reason() -> Optional[str]:
    load_native_library()
    return _load_error


def resolve_rasterizer(requested: str) -> str:
    if requested not in {"auto", "native", "python"}:
        raise ValueError(
            "rasterizer must be 'auto', 'native', or 'python'; "
            f"got {requested!r}"
        )
    if requested == "python":
        return "python"
    if load_native_library() is not None:
        return "native"
    if requested == "native":
        command = (
            f"{sys.executable} -m "
            "vesuvius.tifxyz_label_transfer.build_native"
        )
        raise RuntimeError(
            f"{native_unavailable_reason()}; build it with: {command}"
        )
    return "python"


def _double_array(array: NDArray) -> NDArray[np.float64]:
    return np.ascontiguousarray(array, dtype=np.float64)


def _byte_array(array: NDArray) -> NDArray[np.uint8]:
    return np.ascontiguousarray(array, dtype=np.uint8)


def _pointer(array: Optional[NDArray]) -> Optional[int]:
    return None if array is None else int(array.ctypes.data)


class NativeRasterizer:
    """Prepared native context shared by every output tile."""

    def __init__(
        self,
        *,
        target_fields: Sequence[NDArray],
        target_valid: NDArray,
        uv_rows: NDArray,
        uv_cols: NDArray,
        uv_valid: NDArray,
        source_fields: Sequence[NDArray],
        source_valid: NDArray,
        label_shape: Sequence[int],
        output_shape: Sequence[int],
        label_offset_yx: Sequence[float],
        max_distance: float,
        filled_uv_rows: Optional[NDArray] = None,
        filled_uv_cols: Optional[NDArray] = None,
        source_validity: Optional[NDArray] = None,
    ) -> None:
        library = load_native_library()
        if library is None:
            raise RuntimeError(native_unavailable_reason())
        self.library = library
        self.target_fields = tuple(_double_array(item) for item in target_fields)
        self.target_valid = _byte_array(target_valid)
        self.uv_rows = _double_array(uv_rows)
        self.uv_cols = _double_array(uv_cols)
        self.uv_valid = _byte_array(uv_valid)
        self.source_fields = tuple(_double_array(item) for item in source_fields)
        self.source_valid = _byte_array(source_valid)
        self.filled_uv_rows = (
            None if filled_uv_rows is None else _double_array(filled_uv_rows)
        )
        self.filled_uv_cols = (
            None if filled_uv_cols is None else _double_array(filled_uv_cols)
        )
        self.source_validity = (
            None if source_validity is None else _byte_array(source_validity)
        )
        if len(self.target_fields) != 3 or len(self.source_fields) != 3:
            raise ValueError("native rasterizer requires three XYZ fields")
        if any(item.shape != self.target_valid.shape for item in self.target_fields):
            raise ValueError("target XYZ and validity shapes differ")
        if self.uv_rows.shape != self.target_valid.shape:
            raise ValueError("UV and target shapes differ")
        if self.uv_cols.shape != self.target_valid.shape:
            raise ValueError("UV and target shapes differ")
        if self.uv_valid.shape != self.target_valid.shape:
            raise ValueError("UV validity and target shapes differ")
        if any(item.shape != self.source_valid.shape for item in self.source_fields):
            raise ValueError("source XYZ and validity shapes differ")
        if (self.filled_uv_rows is None) != (self.filled_uv_cols is None):
            raise ValueError("both filled UV fields must be supplied together")
        if self.filled_uv_rows is not None:
            if self.filled_uv_rows.shape != self.target_valid.shape:
                raise ValueError("filled UV and target shapes differ")
            if self.filled_uv_cols.shape != self.target_valid.shape:
                raise ValueError("filled UV and target shapes differ")
        self.label_shape = int(label_shape[0]), int(label_shape[1])
        self.output_shape = int(output_shape[0]), int(output_shape[1])
        self.label_offset_yx = (
            float(label_offset_yx[0]),
            float(label_offset_yx[1]),
        )
        self.max_distance = float(max_distance)

    def rasterize(
        self, bounds: tuple[int, int, int, int]
    ) -> NativeTileResult:
        row_start, row_end, col_start, col_end = bounds
        tile_shape = row_end - row_start, col_end - col_start
        source_indices = np.empty(tile_shape, dtype=np.int64)
        validity = np.empty(tile_shape, dtype=np.uint8)
        distances = np.empty(tile_shape, dtype=np.float64)
        request = _RasterRequest(
            *[_pointer(item) for item in self.target_fields],
            _pointer(self.target_valid),
            _pointer(self.uv_rows),
            _pointer(self.uv_cols),
            _pointer(self.uv_valid),
            *[_pointer(item) for item in self.source_fields],
            _pointer(self.source_valid),
            _pointer(self.filled_uv_rows),
            _pointer(self.filled_uv_cols),
            _pointer(self.source_validity),
            _pointer(source_indices),
            _pointer(validity),
            _pointer(distances),
            self.target_valid.shape[0],
            self.target_valid.shape[1],
            self.source_valid.shape[0],
            self.source_valid.shape[1],
            self.label_shape[0],
            self.label_shape[1],
            self.output_shape[0],
            self.output_shape[1],
            row_start,
            row_end,
            col_start,
            col_end,
            self.label_offset_yx[0],
            self.label_offset_yx[1],
            self.max_distance,
            int(self.filled_uv_rows is not None),
            int(self.source_validity is not None),
            ABI_VERSION,
        )
        result = _RasterResult()
        status = int(
            self.library.vc_tifxyz_rasterize(
                ctypes.byref(request), ctypes.byref(result)
            )
        )
        if status != 0:
            raise RuntimeError(f"native rasterizer failed with status {status}")
        return NativeTileResult(
            source_indices=source_indices,
            validity=validity,
            distances=distances,
            target_surface_valid=int(result.target_surface_valid),
            measured_pixels=int(result.measured_pixels),
            seam_filled_pixels=int(result.seam_filled_pixels),
            inherited_filled_pixels=int(result.inherited_filled_pixels),
        )
