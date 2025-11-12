"""Simple workflow orchestration helpers for image processing scripts.

The goal is to keep a tiny amount of reusable plumbing so individual tools only
need to define a worker function and pass in a set of inputs.
"""

from __future__ import annotations

import multiprocessing
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Sequence, Union

from tqdm import tqdm

PathLike = Union[str, Path]

DEFAULT_IMAGE_EXTENSIONS: tuple[str, ...] = (".tif", ".tiff", ".png", ".jpg", ".jpeg")

__all__ = ("DEFAULT_IMAGE_EXTENSIONS", "gather_inputs", "run_workflow")


def _normalize_sources(sources: Union[PathLike, Sequence[PathLike]]) -> List[Path]:
    if isinstance(sources, (str, Path)):
        sources = [sources]
    paths = [Path(s) for s in sources]
    if not paths:
        raise ValueError("No input sources provided.")
    return paths


def _normalize_extensions(extensions: Optional[Sequence[str]]) -> set[str]:
    if not extensions:
        extensions = DEFAULT_IMAGE_EXTENSIONS
    normalized: set[str] = set()
    for ext in extensions:
        if not ext:
            continue
        ext = ext.lower()
        if not ext.startswith("."):
            ext = f".{ext}"
        normalized.add(ext)
    return normalized


def _collect_from_directory(
    root: Path,
    *,
    image_extensions: set[str],
    include_zarr: bool,
    recursive: bool,
) -> List[Path]:
    items: List[Path] = []
    stack: List[Path] = [root]

    while stack:
        current = stack.pop()
        if not current.is_dir():
            # `current` can only be a directory here, but keep the guard for safety.
            if current.is_file() and current.suffix.lower() in image_extensions:
                items.append(current)
            continue

        # Skip descending into .zarr directories once they are registered as inputs.
        if include_zarr and current.suffix.lower() == ".zarr" and current != root:
            items.append(current)
            continue

        for child in current.iterdir():
            if child.is_file():
                if child.suffix.lower() in image_extensions:
                    items.append(child)
                continue

            if child.is_dir():
                if include_zarr and child.suffix.lower() == ".zarr":
                    items.append(child)
                    continue
                if recursive:
                    stack.append(child)

    return items


def gather_inputs(
    sources: Union[PathLike, Sequence[PathLike]],
    *,
    image_extensions: Optional[Sequence[str]] = None,
    include_zarr: bool = True,
    recursive: bool = True,
    sort_paths: bool = True,
) -> List[Path]:
    """Collect image files or OME-Zarr directories from the provided sources.

    Args:
        sources: One or more files/directories to inspect.
        image_extensions: Acceptable image file suffixes (defaults to common TIFF/PNG/JPG).
        include_zarr: When True, `.zarr` directories are included as single inputs.
        recursive: If True, walk subdirectories when scanning folders.
        sort_paths: Sort the returned paths alphabetically for consistent ordering.

    Returns:
        A list of Path objects ready to feed into a worker function. Paths point to either
        individual image files or directories ending with `.zarr`.
    """
    paths = _normalize_sources(sources)
    extensions = _normalize_extensions(image_extensions)

    collected: List[Path] = []

    for src in paths:
        if not src.exists():
            raise FileNotFoundError(f"Input path does not exist: {src}")

        if src.is_file():
            if src.suffix.lower() not in extensions:
                raise ValueError(f"Unsupported file extension for {src}")
            collected.append(src)
            continue

        if include_zarr and src.is_dir() and src.suffix.lower() == ".zarr":
            collected.append(src)
            continue

        collected.extend(
            _collect_from_directory(
                src,
                image_extensions=extensions,
                include_zarr=include_zarr,
                recursive=recursive,
            )
        )

    if sort_paths:
        collected.sort()

    return collected


def run_workflow(
    worker_fn: Callable[[Any], Any],
    inputs: Iterable[Any],
    *,
    num_workers: Optional[int] = 1,
    show_progress: bool = True,
    progress_desc: str = "Processing",
    chunksize: int = 1,
    initializer: Optional[Callable[..., None]] = None,
    initargs: Optional[Sequence[Any]] = None,
    use_process_pool: bool = True,
    start_method: Optional[str] = None,
) -> List[Any]:
    """Execute a worker function over a collection of inputs with optional parallelism.

    Args:
        worker_fn: Callable that processes a single item from `inputs`.
        inputs: Iterable of items to process. Converted to a list so tqdm can display progress.
        num_workers: Number of processes to spawn. Values <= 1 force sequential execution.
        show_progress: Display a tqdm progress bar when True.
        progress_desc: Text shown alongside the progress bar.
        chunksize: Batch size passed to `Pool.imap`.
        initializer: Optional initializer for each worker process.
        initargs: Arguments forwarded to the initializer.
        use_process_pool: When False, always run sequentially.
        start_method: Multiprocessing start method (e.g. "spawn"). Defaults to platform default.

    Returns:
        A list of worker results in the same order as the inputs.
    """
    initargs = tuple(initargs or ())
    items = list(inputs)

    if not items:
        return []

    worker_count = max(1, int(num_workers or 1))

    if not use_process_pool or worker_count == 1:
        if initializer is not None:
            initializer(*initargs)
        iterable = tqdm(items, desc=progress_desc) if show_progress else items
        return [worker_fn(item) for item in iterable]

    chunksize = max(1, int(chunksize))
    context = multiprocessing.get_context(start_method) if start_method else multiprocessing.get_context()

    with context.Pool(
        processes=worker_count,
        initializer=initializer,
        initargs=initargs,
    ) as pool:
        iterator = pool.imap(worker_fn, items, chunksize)
        if show_progress:
            iterator = tqdm(iterator, total=len(items), desc=progress_desc)
        return list(iterator)
