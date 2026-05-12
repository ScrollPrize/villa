"""WebKnossos I/O for the streaming fiber tracer.

This module assumes the WebKnossos dataset and the inference Zarr share the
same voxel frame (i.e. no affine transform between WK display coords and
the zarr's native voxel coords); the conversion between WK ``xyz`` and the
zarr's ``zyx`` is just the axis-order swap. The ``.npz`` prompt loader
verifies this assumption by round-tripping the prompt's stored
``source_points_xyz`` against ``swap(points_zyx)`` and aborts with a clear
error if they disagree by more than a sub-voxel tolerance.

Public surface
--------------

* :func:`load_prompt_npz`     — parse a ``build_fiber_cache.py`` output.
* :func:`build_skeleton`      — pure: polyline -> :class:`webknossos.Skeleton`.
* :func:`save_annotation`     — writes a ``.zip`` (WK's native skeleton
  bundle) and a sibling ``.nml`` for quick visual inspection.
* :func:`upload_annotation`   — opens a token-scoped ``webknossos_context``
  and uploads, returning the new annotation URL.
* :func:`upload_polyline_via_update_actions` — alternative upload path that
  creates an empty annotation server-side and pushes the polyline as update
  actions to the tracing store, bypassing servers whose
  ``mergedFromContents`` endpoint is unavailable.
* :func:`load_wk_token`       — locates and reads the project's
  ``webknossos-api-token.txt`` (or the ``WK_TOKEN`` env override). Never
  logged.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vesuvius.neural_tracing.autoreg_fiber.webknossos_annotations import (
    read_token_file,
    resolve_token_path,
)


DEFAULT_WK_SERVER_URL = "https://wk.aws.ash2txt.org"


@dataclass
class PromptPayload:
    """Parsed contents of a fiber-cache ``.npz`` prompt."""

    points_zyx: np.ndarray
    source_points_xyz: np.ndarray | None
    metadata: dict[str, Any]


def load_prompt_npz(path: str | Path) -> PromptPayload:
    """Load a ``build_fiber_cache.py``-style ``.npz`` and verify zyx<->xyz.

    The fiber cache stores both ``points_zyx`` (the inference frame) and
    ``source_points_xyz`` (the original WK frame from before any affine
    transform was applied at ingest). When the two volumes share a voxel
    frame, the relation collapses to ``points_zyx == swap(source_points_xyz)``
    and we verify that to catch silent frame mismatches.

    Raises ``ValueError`` if the .npz is malformed or if the swap-only
    invariant does not hold to within ``1e-3`` voxels.
    """

    blob = np.load(Path(path), allow_pickle=False)
    if "points_zyx" not in blob.files:
        raise ValueError(f"{path}: missing 'points_zyx' array in .npz")
    points_zyx = np.asarray(blob["points_zyx"], dtype=np.float32)
    if points_zyx.ndim != 2 or points_zyx.shape[1] != 3:
        raise ValueError(f"{path}: points_zyx must have shape (N, 3); got {tuple(points_zyx.shape)}")

    source_points_xyz: np.ndarray | None = None
    if "source_points_xyz" in blob.files:
        source_points_xyz = np.asarray(blob["source_points_xyz"], dtype=np.float32)
        if source_points_xyz.shape != points_zyx.shape:
            raise ValueError(
                f"{path}: source_points_xyz shape {tuple(source_points_xyz.shape)} != "
                f"points_zyx shape {tuple(points_zyx.shape)}"
            )

    metadata: dict[str, Any] = {}
    if "metadata" in blob.files:
        import json

        metadata = json.loads(str(blob["metadata"].item()))

    if source_points_xyz is not None and points_zyx.shape[0] > 0:
        swapped = points_zyx[:, ::-1]
        delta = np.abs(swapped - source_points_xyz)
        max_delta = float(delta.max())
        if max_delta > 1e-3:
            raise ValueError(
                f"{path}: the swap-only invariant is violated by {max_delta:.4f} voxels — "
                "this fiber cache was built with a non-identity affine and cannot be uploaded "
                "to the same WK dataset without applying its inverse. Re-run the prompt build "
                "with the matching affine, or use a WK dataset that lives in the zarr frame."
            )
    return PromptPayload(points_zyx=points_zyx, source_points_xyz=source_points_xyz, metadata=metadata)


def load_wk_token(*, env_var: str = "WK_TOKEN", start_dir: str | Path | None = None) -> str:
    """Return the WebKnossos API token.

    Resolution order: ``$WK_TOKEN`` env var, then ``webknossos-api-token.txt``
    walked from ``start_dir`` up to the filesystem root. The token is never
    logged or returned in any error message.
    """

    token = os.environ.get(env_var)
    if token:
        token = token.strip()
        if not token:
            raise ValueError(f"environment variable {env_var} is set but empty")
        return token
    return read_token_file(resolve_token_path(start_dir=start_dir or Path(__file__).parent))


# --- Skeleton construction ----------------------------------------------- #

def _polyline_zyx_to_node_positions(polyline_world_zyx: np.ndarray) -> np.ndarray:
    arr = np.asarray(polyline_world_zyx, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"polyline must have shape (N, 3); got {tuple(arr.shape)}")
    # WK skeleton nodes are positioned in XYZ voxel order.
    swapped = arr[:, ::-1]
    return np.rint(swapped).astype(np.int64)


def build_skeleton(
    polyline_world_zyx: np.ndarray,
    *,
    dataset_name: str,
    voxel_size: tuple[float, float, float] = (1.0, 1.0, 1.0),
    tree_name: str = "autoreg_fiber_prediction",
    organization_id: str | None = None,
    description: str | None = None,
):
    """Pure constructor: polyline (volume zyx) -> :class:`webknossos.Skeleton`.

    Network-free. The webknossos package is imported lazily so this module
    is importable without the optional `[webknossos]` extra.
    """

    import webknossos as wk

    skeleton = wk.Skeleton(
        voxel_size=voxel_size,
        dataset_name=str(dataset_name),
        organization_id=organization_id,
        description=description,
    )
    tree = skeleton.add_tree(str(tree_name))
    positions = _polyline_zyx_to_node_positions(polyline_world_zyx)
    nodes = []
    for position in positions:
        node = tree.add_node(position=(int(position[0]), int(position[1]), int(position[2])), radius=1.0)
        nodes.append(node)
    for prev_node, next_node in zip(nodes[:-1], nodes[1:], strict=True):
        tree.add_edge(prev_node, next_node)
    return skeleton


def build_annotation(
    skeleton,
    *,
    name: str = "autoreg_fiber_prediction",
    description: str | None = None,
):
    """Pure constructor: :class:`webknossos.Skeleton` -> :class:`webknossos.Annotation`.

    The skeleton already carries dataset / voxel_size / organization / description;
    the WK API forbids re-specifying them when a skeleton is passed.
    """

    import webknossos as wk

    annotation = wk.Annotation(name=str(name), skeleton=skeleton)
    if description is not None and getattr(annotation, "description", None) != description:
        annotation.description = description
    return annotation


# --- I/O ------------------------------------------------------------------ #

def save_annotation(annotation, out_dir: str | Path, *, basename: str = "trace") -> dict[str, str]:
    """Write the annotation locally. Returns ``{"zip": ..., "nml": ...}``.

    The ``.zip`` is WK's native skeleton bundle (suitable for drag-and-drop
    upload via the browser); the ``.nml`` is a single XML for quick visual
    inspection. Both contain the same skeleton.
    """

    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"{basename}.zip"
    nml_path = out_dir / f"{basename}.nml"
    annotation.save(zip_path)
    annotation.skeleton.save(nml_path)
    return {"zip": str(zip_path), "nml": str(nml_path)}


def upload_annotation(
    annotation,
    *,
    server_url: str = DEFAULT_WK_SERVER_URL,
    token: str | None = None,
    timeout: int | None = None,
) -> str:
    """Upload the annotation to ``server_url`` and return the new URL.

    The token is loaded via :func:`load_wk_token` if not provided. The
    network call is wrapped in a ``webknossos_context`` so the token only
    lives in process memory for the duration of the upload.
    """

    import webknossos as wk

    resolved_token = token if token is not None else load_wk_token()
    if not resolved_token:
        raise RuntimeError("no WK token available (env WK_TOKEN unset and token file missing)")
    with wk.webknossos_context(url=str(server_url), token=resolved_token, timeout=timeout):
        url = annotation.upload()
    return str(url)


def upload_polyline_via_update_actions(
    polyline_world_zyx: "np.ndarray",
    *,
    dataset_slug: str,
    tree_name: str = "autoreg_fiber_prediction",
    server_url: str = DEFAULT_WK_SERVER_URL,
    token: str | None = None,
    batch_size: int = 2000,
    timeout: int = 120,
) -> str:
    """Upload a polyline as a new annotation via the tracing-store update-actions API.

    Workaround for WK servers whose ``/annotations/upload`` endpoint fails inside
    ``mergedFromContents``. Flow:

      1. ``POST /api/datasets/{dataset_slug}/createExplorational`` creates an empty
         skeleton annotation server-side.
      2. ``POST /tracings/annotation/{annotationId}/update`` pushes a single tree
         with one node per polyline point and edges chaining consecutive nodes,
         using the same update-action format the WK frontend uses for interactive
         edits. The payload is split into transaction *groups* of ``batch_size``
         nodes so very long traces do not exceed the JSON / version budget per
         request.

    ``polyline_world_zyx`` is expected in the zarr's native ``(z, y, x)`` voxel
    frame; it is swapped to WK's ``(x, y, z)`` order on the way out (same
    convention as :func:`build_skeleton`). Returns the new annotation URL.
    """

    import time
    import uuid

    import numpy as np
    import requests

    arr = np.asarray(polyline_world_zyx, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"polyline must have shape (N, 3); got {tuple(arr.shape)}")
    if arr.shape[0] < 2:
        raise ValueError("polyline must contain at least 2 nodes")
    positions_xyz = np.rint(arr[:, ::-1]).astype(np.int64)  # (N, 3) in xyz voxel order

    resolved_token = token if token is not None else load_wk_token()
    if not resolved_token:
        raise RuntimeError("no WK token available (env WK_TOKEN unset and token file missing)")
    base = str(server_url).rstrip("/")
    headers = {"X-Auth-Token": resolved_token, "Content-Type": "application/json"}

    # 1. Create empty annotation.
    create_resp = requests.post(
        f"{base}/api/datasets/{dataset_slug}/createExplorational",
        headers=headers,
        json=[{"typ": "Skeleton"}],
        timeout=timeout,
    )
    create_resp.raise_for_status()
    ann = create_resp.json()
    annotation_id = str(ann["id"])
    skeleton_layer = ann["annotationLayers"][0]
    tracing_id = str(skeleton_layer["tracingId"])
    typ = str(ann.get("typ", "Explorational"))

    # 2. Build update-action groups. Node IDs are 1-based; edges chain consecutive ids.
    now_ms = int(time.time() * 1000)
    transaction_id = str(uuid.uuid4())
    total_nodes = int(positions_xyz.shape[0])
    if int(batch_size) <= 0:
        raise ValueError(f"batch_size must be positive; got {batch_size!r}")
    chunks: list[tuple[int, int]] = []  # (start_idx, end_idx_exclusive)
    for start in range(0, total_nodes, int(batch_size)):
        chunks.append((start, min(start + int(batch_size), total_nodes)))
    transaction_group_count = len(chunks)

    # All groups of one transaction share the same target version (the new state
    # after the transaction lands) and the same transactionId; transactionGroupIndex
    # orders them. The server reassembles them into one logical version increment.
    # We post them as one JSON array so the server treats them as one transaction.
    next_version = 1
    groups: list[dict[str, object]] = []
    for group_idx, (start, end) in enumerate(chunks):
        actions: list[dict[str, object]] = []
        if group_idx == 0:
            actions.append(
                {
                    "name": "createTree",
                    "value": {
                        "id": 1,
                        "name": str(tree_name),
                        "branchPoints": [],
                        "comments": [],
                        "groupId": None,
                        "isVisible": True,
                        "edgesAreVisible": True,
                        "timestamp": now_ms,
                        "actionTracingId": tracing_id,
                    },
                }
            )
        for i in range(start, end):
            x, y, z = (int(v) for v in positions_xyz[i])
            actions.append(
                {
                    "name": "createNode",
                    "value": {
                        "id": i + 1,
                        "position": [x, y, z],
                        "rotation": [0.0, 0.0, 0.0],
                        "radius": 1.0,
                        "viewport": 0,
                        "resolution": 0,
                        "bitDepth": 8,
                        "interpolation": False,
                        "treeId": 1,
                        "timestamp": now_ms,
                        "actionTracingId": tracing_id,
                    },
                }
            )
            if i > 0:
                actions.append(
                    {
                        "name": "createEdge",
                        "value": {
                            "source": i,
                            "target": i + 1,
                            "treeId": 1,
                            "actionTracingId": tracing_id,
                        },
                    }
                )
        groups.append(
            {
                "version": next_version,
                "timestamp": now_ms,
                "authorId": None,
                "actions": actions,
                "stats": None,
                "info": f"autoreg_fiber trace ({total_nodes} nodes)",
                "transactionId": transaction_id,
                "transactionGroupCount": transaction_group_count,
                "transactionGroupIndex": group_idx,
            }
        )

    update_resp = requests.post(
        f"{base}/tracings/annotation/{annotation_id}/update",
        headers=headers,
        json=groups,
        timeout=timeout,
    )
    if update_resp.status_code != 200:
        body = update_resp.text[:500].replace("\n", " ")
        raise RuntimeError(
            f"tracing-store update failed ({transaction_group_count} groups, "
            f"{total_nodes} nodes): {update_resp.status_code} {body}"
        )
    return f"{base}/annotations/{typ}/{annotation_id}"


__all__ = [
    "DEFAULT_WK_SERVER_URL",
    "PromptPayload",
    "build_annotation",
    "build_skeleton",
    "load_prompt_npz",
    "load_wk_token",
    "save_annotation",
    "upload_annotation",
    "upload_polyline_via_update_actions",
]
