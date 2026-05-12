"""Build a fiber-cache ``.npz`` from a single WK annotation URL.

The streaming tracer's CLI takes a ``.npz`` as its prompt source. When the
user does not yet have one (e.g. the training-time fiber-cache manifest is
on another machine), this helper downloads the annotation, picks a tree,
and writes an ``.npz`` ready to feed the tracer.

Assumes the WK dataset is in the same voxel frame as the inference zarr —
no affine is applied. ``source_points_xyz == swap(points_zyx)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from vesuvius.neural_tracing.autoreg_fiber.fiber_geometry import (
    FiberPath,
    densify_polyline,
    order_tree_path_xyz,
    write_fiber_cache,
    xyz_to_zyx,
)
from vesuvius.neural_tracing.autoreg_fiber.streaming.wk_io import (
    DEFAULT_WK_SERVER_URL,
    load_wk_token,
)


def _flatten_trees(annotation: Any) -> list[Any]:
    """Return every tree in the annotation (across all skeleton groups)."""

    skeleton = getattr(annotation, "skeleton", None)
    if skeleton is None:
        return []
    if hasattr(skeleton, "flattened_trees"):
        return list(skeleton.flattened_trees())
    if hasattr(skeleton, "trees"):
        return list(skeleton.trees)
    return []


def _select_tree(trees: list[Any], *, tree_id: str | int | None, tree_name: str | None) -> Any:
    if tree_id is not None:
        target = str(tree_id)
        for tree in trees:
            if str(getattr(tree, "id", "")) == target:
                return tree
        raise ValueError(f"no tree with id={tree_id!r} in annotation; have {len(trees)} trees")
    if tree_name is not None:
        target = str(tree_name)
        for tree in trees:
            if str(getattr(tree, "name", "")) == target:
                return tree
        raise ValueError(f"no tree named {tree_name!r} in annotation; have {len(trees)} trees")
    if len(trees) == 1:
        return trees[0]
    raise ValueError(
        f"annotation has {len(trees)} trees; specify --tree-id or --tree-name to pick one"
    )


def build_npz_from_wk_url(
    *,
    annotation_url: str,
    out_dir: str | Path,
    server_url: str = DEFAULT_WK_SERVER_URL,
    token: str | None = None,
    tree_id: str | int | None = None,
    tree_name: str | None = None,
    target_volume: str = "PHercParis4",
    marker: str = "fibers_s1a",
    densify_step: float | None = 1.0,
) -> Path:
    """Download the WK annotation, convert one tree to a fiber-cache ``.npz``.

    ``densify_step=1.0`` matches the training pipeline's unit-stride sampling
    along the polyline. Returns the path to the written ``.npz``.
    """

    import webknossos as wk

    resolved_token = token if token is not None else load_wk_token()
    with wk.webknossos_context(url=str(server_url), token=resolved_token):
        annotation = wk.Annotation.download(str(annotation_url), skip_volume_data=True)

    trees = _flatten_trees(annotation)
    if not trees:
        raise ValueError("annotation has no skeleton trees")
    tree = _select_tree(trees, tree_id=tree_id, tree_name=tree_name)

    ordered = order_tree_path_xyz(tree)
    source_points_xyz = densify_polyline(ordered.points_xyz, max_step=densify_step)
    points_zyx = xyz_to_zyx(source_points_xyz)

    annotation_id = str(getattr(annotation, "annotation_id", "") or getattr(annotation, "name", "wk-annotation"))
    fiber = FiberPath(
        annotation_id=annotation_id,
        tree_id=str(getattr(tree, "id", "")),
        target_volume=str(target_volume),
        marker=str(marker),
        source_points_xyz=source_points_xyz.astype(np.float32, copy=False),
        points_zyx=points_zyx.astype(np.float32, copy=False),
        transform_checksum="identity",
        densify_step=None if densify_step is None else float(densify_step),
    )
    return write_fiber_cache(fiber, out_dir)


__all__ = ["build_npz_from_wk_url"]
