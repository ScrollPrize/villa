"""File I/O helpers for the TIFXYZ label-transfer CLI."""

from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import secrets
import tempfile
import threading
from typing import Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from PIL import Image
import tifffile

from .core import Surface


def _read_tiff(path: Path) -> NDArray:
    try:
        return tifffile.memmap(path, mode="r")
    except ValueError:
        return tifffile.imread(path)


def load_tifxyz_mask(
    path: Path | str,
    xyz_shape: Sequence[int],
) -> NDArray[np.bool_]:
    """Load a TIFXYZ mask using the canonical stored-grid semantics."""
    mask_path = Path(path)
    target_shape = int(xyz_shape[0]), int(xyz_shape[1])
    mask = np.asarray(_read_tiff(mask_path))

    # mask.tif may be multipage; the first page is the validity mask.
    if (
        mask.ndim > 2
        and mask.shape[-2] % target_shape[0] == 0
        and mask.shape[-1] % target_shape[1] == 0
    ):
        mask = mask.reshape((-1, mask.shape[-2], mask.shape[-1]))[0]
    elif (
        mask.ndim > 2
        and mask.shape[0] % target_shape[0] == 0
        and mask.shape[1] % target_shape[1] == 0
    ):
        mask = mask.reshape((mask.shape[0], mask.shape[1], -1))[:, :, 0]

    if mask.shape == target_shape:
        return np.asarray(mask >= 255)
    if (
        mask.ndim == 2
        and mask.shape[0] % target_shape[0] == 0
        and mask.shape[1] % target_shape[1] == 0
    ):
        # QuadSurface accepts a higher-resolution mask when each dimension is
        # an integer multiple of the XYZ grid. A stored-grid point is retained
        # only when its complete mask block is valid.
        factor_y = mask.shape[0] // target_shape[0]
        factor_x = mask.shape[1] // target_shape[1]
        return np.all(
            mask.reshape(
                target_shape[0], factor_y, target_shape[1], factor_x
            )
            >= 255,
            axis=(1, 3),
        )
    raise ValueError(
        f"mask shape {mask.shape} is incompatible with XYZ shape "
        f"{target_shape}"
    )


def _open_zarr_label(path: Path):
    import zarr

    group = zarr.open_group(str(path), mode="r")
    if "0" not in group:
        raise ValueError(f"label Zarr {path} has no level-0 array")
    return group["0"]


def _zarr_label_shape(array, path: Path) -> Tuple[int, int]:
    shape = tuple(int(value) for value in array.shape)
    if len(shape) == 2:
        return shape
    if len(shape) == 3 and shape[0] > 0:
        return shape[1], shape[2]
    raise ValueError(
        f"label Zarr must contain a 2D image or z/y/x volume; "
        f"{path} level 0 has shape {shape}"
    )


def read_image(path: Path | str) -> NDArray:
    image_path = Path(path)
    if image_path.is_dir() and image_path.suffix.lower() == ".zarr":
        array = _open_zarr_label(image_path)
        _zarr_label_shape(array, image_path)
        image = np.asarray(
            array if array.ndim == 2 else array[array.shape[0] // 2]
        )
    elif image_path.suffix.lower() in {".tif", ".tiff"}:
        image = np.asarray(_read_tiff(image_path))
    else:
        with Image.open(image_path) as handle:
            image = np.asarray(handle)
    image = np.squeeze(image)
    if image.ndim != 2:
        raise ValueError(
            f"label/reference image must be 2D grayscale; "
            f"{image_path} has shape {image.shape}"
        )
    return image


def read_image_shape(path: Path | str) -> Tuple[int, int]:
    image_path = Path(path)
    if image_path.is_dir() and image_path.suffix.lower() == ".zarr":
        return _zarr_label_shape(_open_zarr_label(image_path), image_path)
    if image_path.suffix.lower() in {".tif", ".tiff"}:
        with tifffile.TiffFile(image_path) as handle:
            shape = tuple(int(value) for value in handle.series[0].shape)
        while len(shape) > 2 and shape[0] == 1:
            shape = shape[1:]
        if len(shape) != 2:
            raise ValueError(
                f"reference image must be 2D grayscale; "
                f"{image_path} has shape {shape}"
            )
        return shape[0], shape[1]
    with Image.open(image_path) as handle:
        width, height = handle.size
        if len(handle.getbands()) != 1:
            raise ValueError(
                f"reference image must be grayscale; "
                f"{image_path} has bands {handle.getbands()}"
            )
    return int(height), int(width)


def load_surface(path: Path | str, use_mask: bool = True) -> Surface:
    surface_path = Path(path)
    required = [
        surface_path / "x.tif",
        surface_path / "y.tif",
        surface_path / "z.tif",
        surface_path / "meta.json",
    ]
    missing = [str(item) for item in required if not item.is_file()]
    if missing:
        raise FileNotFoundError(
            f"not a TIFXYZ directory; missing: {', '.join(missing)}"
        )

    with (surface_path / "meta.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    scale = metadata.get("scale", [1.0, 1.0])
    if not isinstance(scale, list) or len(scale) < 2:
        raise ValueError(f"invalid scale in {surface_path / 'meta.json'}")
    # C++ metadata stores [x_scale, y_scale]; array indexing uses (y, x).
    scale_yx = float(scale[1]), float(scale[0])

    x = np.asarray(_read_tiff(surface_path / "x.tif"), dtype=np.float32)
    y = np.asarray(_read_tiff(surface_path / "y.tif"), dtype=np.float32)
    z = np.asarray(_read_tiff(surface_path / "z.tif"), dtype=np.float32)
    valid = None
    mask_path = surface_path / "mask.tif"
    if use_mask and mask_path.is_file():
        try:
            valid = load_tifxyz_mask(mask_path, x.shape)
        except ValueError as exc:
            raise ValueError(
                f"{exc}; pass --ignore-tifxyz-mask only if this mask should "
                "intentionally be ignored"
            ) from exc
    return Surface(
        x=x,
        y=y,
        z=z,
        scale_yx=scale_yx,
        valid=valid,
        name=surface_path.name,
    )


def write_image(path: Path | str, image: NDArray) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        estimated_bytes = int(np.prod(image.shape)) * int(image.dtype.itemsize)
        tifffile.imwrite(
            output_path,
            image,
            bigtiff=estimated_bytes >= (4 * 1024**3),
            compression="zlib",
            metadata=None,
        )
        return
    if suffix != ".png":
        raise ValueError("output image extension must be .tif, .tiff, or .png")
    Image.fromarray(np.asarray(image)).save(output_path)


class TemporaryRaster:
    """Disk-backed raster used to avoid holding large outputs in RAM."""

    def __init__(
        self,
        directory: Path,
        shape: Sequence[int],
        dtype: np.dtype,
        fill_value: int | float,
        prefix: str,
    ) -> None:
        self._temporary = tempfile.NamedTemporaryFile(
            prefix=prefix,
            suffix=".dat",
            dir=directory,
            delete=False,
        )
        self.path = Path(self._temporary.name)
        self._temporary.close()
        self.array = np.memmap(
            self.path,
            mode="w+",
            dtype=dtype,
            shape=(int(shape[0]), int(shape[1])),
        )
        self.array[:] = fill_value

    def close(self) -> None:
        self.array.flush()
        del self.array
        self.path.unlink(missing_ok=True)

    def __enter__(self) -> "TemporaryRaster":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


class StreamingTiffOutputs:
    """Write matching raster tiles to TIFFs without full-canvas buffers.

    A bounded queue and one encoder thread per output overlap compression with
    geometry. Files are written to sibling temporary paths and published only
    after every encoder succeeds, so a failed transfer leaves no partial
    declared output.
    """

    _SENTINEL = object()

    def __init__(
        self,
        outputs: Sequence[tuple[Path, np.dtype, int | float]],
        shape: Sequence[int],
        tile_size: int,
        queue_depth: int = 2,
    ) -> None:
        if not outputs:
            raise ValueError("at least one streaming TIFF output is required")
        if tile_size <= 0:
            raise ValueError("tile_size must be positive")
        if tile_size % 16:
            raise ValueError("streaming TIFF tile_size must be a multiple of 16")
        self.outputs = [
            (Path(path), np.dtype(dtype), fill_value)
            for path, dtype, fill_value in outputs
        ]
        if any(
            path.suffix.lower() not in {".tif", ".tiff"}
            for path, _, _ in self.outputs
        ):
            raise ValueError("streaming output supports TIFF files only")
        self.shape = int(shape[0]), int(shape[1])
        self.tile_size = int(tile_size)
        self.queues = [
            queue.Queue(maxsize=max(1, int(queue_depth))) for _ in self.outputs
        ]
        self.temporary_paths: list[Path] = []
        self.threads: list[threading.Thread] = []
        self.errors: list[BaseException] = []
        self._error_lock = threading.Lock()
        self._finished = False

    def _record_error(self, error: BaseException) -> None:
        with self._error_lock:
            self.errors.append(error)

    def _check_errors(self) -> None:
        with self._error_lock:
            if self.errors:
                raise RuntimeError("streaming TIFF encoder failed") from self.errors[0]

    def _tile_iterator(
        self, output_queue: queue.Queue, ended: threading.Event
    ):
        while True:
            item = output_queue.get()
            if item is self._SENTINEL:
                ended.set()
                return
            yield item

    def _write_one(
        self,
        temporary: Path,
        dtype: np.dtype,
        output_queue: queue.Queue,
    ) -> None:
        ended = threading.Event()
        try:
            tifffile.imwrite(
                temporary,
                self._tile_iterator(output_queue, ended),
                shape=self.shape,
                dtype=dtype,
                bigtiff=(
                    int(np.prod(self.shape)) * dtype.itemsize >= 4 * 1024**3
                ),
                tile=(self.tile_size, self.tile_size),
                compression="zlib",
                photometric="minisblack",
                metadata=None,
                maxworkers=1,
            )
        except BaseException as error:
            self._record_error(error)
            # Unblock a producer waiting on this bounded queue.
            if not ended.is_set():
                while output_queue.get() is not self._SENTINEL:
                    pass

    def __enter__(self) -> "StreamingTiffOutputs":
        try:
            for index, (path, dtype, _) in enumerate(self.outputs):
                path.parent.mkdir(parents=True, exist_ok=True)
                for _attempt in range(100):
                    temporary = path.parent / (
                        f".{path.name}.stream-{os.getpid()}-{index}-"
                        f"{secrets.token_hex(8)}.tif"
                    )
                    try:
                        descriptor = os.open(
                            temporary,
                            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                            0o666,
                        )
                    except FileExistsError:
                        continue
                    os.close(descriptor)
                    break
                else:
                    raise FileExistsError(
                        f"could not create a unique temporary beside {path}"
                    )
                self.temporary_paths.append(temporary)
                thread = threading.Thread(
                    target=self._write_one,
                    args=(temporary, dtype, self.queues[index]),
                    name=f"tiff-encoder-{index}",
                    daemon=True,
                )
                thread.start()
                self.threads.append(thread)
        except BaseException:
            self.abort()
            raise
        return self

    def _put(self, output_queue: queue.Queue, value: object) -> None:
        while True:
            self._check_errors()
            try:
                output_queue.put(value, timeout=0.1)
                return
            except queue.Full:
                continue

    def write_tile(
        self,
        bounds: tuple[int, int, int, int],
        arrays: Sequence[NDArray],
    ) -> None:
        if self._finished:
            raise RuntimeError("cannot write after streaming outputs finished")
        if len(arrays) != len(self.outputs):
            raise ValueError(
                f"expected {len(self.outputs)} tile arrays, got {len(arrays)}"
            )
        row_start, row_end, col_start, col_end = bounds
        expected_shape = row_end - row_start, col_end - col_start
        for output_queue, array, (_, dtype, fill_value) in zip(
            self.queues, arrays, self.outputs
        ):
            tile = np.asarray(array)
            if tile.shape != expected_shape or tile.dtype != dtype:
                raise ValueError(
                    f"streamed tile must have shape {expected_shape} and "
                    f"dtype {dtype}; got {tile.shape}, {tile.dtype}"
                )
            if tile.shape != (self.tile_size, self.tile_size):
                padded = np.full(
                    (self.tile_size, self.tile_size), fill_value, dtype=dtype
                )
                padded[: tile.shape[0], : tile.shape[1]] = tile
                tile = padded
            elif not tile.flags.c_contiguous:
                tile = np.ascontiguousarray(tile)
            self._put(output_queue, tile)

    def finish(self) -> None:
        if self._finished:
            return
        for output_queue in self.queues:
            self._put(output_queue, self._SENTINEL)
        for thread in self.threads:
            thread.join()
        self._check_errors()
        for temporary, (path, _, _) in zip(self.temporary_paths, self.outputs):
            os.replace(temporary, path)
        self._finished = True

    def abort(self) -> None:
        if not self._finished:
            self._finished = True
            for output_queue, thread in zip(self.queues, self.threads):
                while thread.is_alive():
                    try:
                        output_queue.put(self._SENTINEL, timeout=0.1)
                        break
                    except queue.Full:
                        continue
            for thread in self.threads:
                thread.join()
        for temporary in self.temporary_paths:
            temporary.unlink(missing_ok=True)

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is None:
            try:
                self.finish()
            except BaseException:
                self.abort()
                raise
        else:
            self.abort()


def sidecar_path(output: Path, kind: str, extension: Optional[str] = None) -> Path:
    suffix = extension if extension is not None else output.suffix
    return output.with_name(f"{output.stem}.{kind}{suffix}")
