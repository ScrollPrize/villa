#!/usr/bin/env python3
"""Sample dataset patches containing meshes and export them as 3D TIFF stacks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import tifffile
import torch

from vesuvius.models.configuration.config_manager import ConfigManager
from vesuvius.models.datasets import DatasetOrchestrator
from vesuvius.models.datasets.mesh import mesh_to_binary_voxels


def _determine_adapter(mgr) -> str:
    data_format = getattr(mgr, "data_format", "zarr").lower()
    lookup = {
        "zarr": "zarr",
        "image": "image",
        "napari": "napari",
    }
    try:
        return lookup[data_format]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(
            f"Unsupported data_format '{data_format}'. Supported: {sorted(lookup)}"
        ) from exc


def _build_dataset(mgr, *, is_training: bool, mesh_override: Dict[str, object] | None) -> DatasetOrchestrator:
    adapter = _determine_adapter(mgr)
    mesh_config = dict(getattr(mgr, "dataset_config", {}).get("meshes", {}))
    if mesh_override:
        mesh_config.update(mesh_override)
    dataset = DatasetOrchestrator(
        mgr=mgr,
        adapter=adapter,
        adapter_kwargs={},
        is_training=is_training,
        mesh_config=mesh_config,
    )
    return dataset


def _parse_triplet(values: Iterable[float] | None, *, default: Tuple[float, float, float]) -> np.ndarray:
    if values is None:
        return np.asarray(default, dtype=np.float32)
    vals = list(values)
    if len(vals) != 3:
        raise ValueError("Expected three values (z y x)")
    return np.asarray(vals, dtype=np.float32)

def _build_volume_mesh_boxes(
    dataset,
    *,
    voxel_size: float,
    grid_origin: np.ndarray,
) -> Dict[int, List[Tuple[str | None, np.ndarray, np.ndarray]]]:
    if not getattr(dataset, "target_volumes", None):
        return {}

    first_target = next(iter(dataset.target_volumes), None)
    if first_target is None:
        return {}

    volumes = dataset.target_volumes[first_target]
    boxes: Dict[int, List[Tuple[str | None, np.ndarray, np.ndarray]]] = {}

    for vol_idx, entry in enumerate(volumes):
        meshes = entry.get("meshes") or {}
        volume_boxes: List[Tuple[str | None, np.ndarray, np.ndarray]] = []
        for mesh_id, handle in meshes.items():
            payload = handle.read()
            vertices = np.asarray(payload.vertices, dtype=np.float32)
            if vertices.size == 0:
                continue
            if vertices.shape[1] != 3:
                raise ValueError(
                    f"Mesh '{mesh_id}' vertices must be Nx3; got shape {vertices.shape}"
                )

            # Convert to (z, y, x) ordering so indices align with dataset grid axes.
            vertices_zyx = vertices[:, [2, 1, 0]]
            min_world = vertices_zyx.min(axis=0)
            max_world = vertices_zyx.max(axis=0)

            margin = voxel_size * 0.5
            minimum = np.floor((min_world - margin - grid_origin) / voxel_size).astype(int)
            maximum = np.ceil((max_world + margin - grid_origin) / voxel_size).astype(int)
            volume_boxes.append((mesh_id, minimum, maximum))
        boxes[vol_idx] = volume_boxes

    return boxes


def _collect_candidate_patch_indices(
    dataset,
    *,
    volume_boxes: Dict[int, List[Tuple[str | None, np.ndarray, np.ndarray]]],
) -> List[int]:
    patch_infos = getattr(dataset, "valid_patches", []) or []
    if not patch_infos:
        return []

    candidates: set[int] = set()
    for idx, info in enumerate(patch_infos):
        volume_index = int(info.get("volume_index", 0))
        boxes = volume_boxes.get(volume_index)
        if not boxes:
            continue

        position = info.get("position")
        patch_size = info.get("patch_size")
        if position is None or patch_size is None:
            continue

        start = np.asarray(position, dtype=np.int64)
        size = np.asarray(patch_size, dtype=np.int64)
        if start.size != size.size or start.size == 0:
            continue

        end = start + size
        for _, box_min, box_max in boxes:
            if np.all(box_min < end) and np.all(start < box_max):
                candidates.add(idx)
                break

    return sorted(candidates)


def _compute_local_mask(
    payload_info: Dict[str, object],
    patch_start: np.ndarray,
    patch_size: Tuple[int, int, int],
    *,
    voxel_size: float,
    grid_origin: np.ndarray,
    fill_solid: bool,
) -> np.ndarray:
    payload = payload_info["payload"]
    vox = mesh_to_binary_voxels(payload, voxel_size=voxel_size, fill_solid=fill_solid)

    origin = np.asarray(vox.origin, dtype=np.float32)
    coords = np.argwhere(vox.voxels)
    if coords.size == 0:
        return np.zeros(patch_size, dtype=np.uint8)

    origin_indices = np.rint((origin - grid_origin) / voxel_size).astype(int)
    global_coords = coords + origin_indices
    local = global_coords - patch_start

    mask = np.zeros(patch_size, dtype=np.uint8)
    valid = (
        (local[:, 0] >= 0)
        & (local[:, 1] >= 0)
        & (local[:, 2] >= 0)
        & (local[:, 0] < patch_size[0])
        & (local[:, 1] < patch_size[1])
        & (local[:, 2] < patch_size[2])
    )
    if not np.any(valid):
        return mask
    local = local[valid]
    mask[local[:, 0], local[:, 1], local[:, 2]] = 255
    return mask


def _tensor_to_volume(image: torch.Tensor) -> np.ndarray:
    arr = image.detach().cpu().numpy()
    if arr.ndim == 4:
        arr = arr[0]  # assume single channel
    elif arr.ndim == 3:
        pass
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported image tensor shape {arr.shape}")
    return np.asarray(arr, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export dataset patches that contain mesh geometry.")
    parser.add_argument("--config", required=True, help="Path to dataset YAML config")
    parser.add_argument("--output-dir", required=True, help="Directory to write sample TIFF stacks")
    parser.add_argument("--max-samples", type=int, default=8, help="Maximum number of patches to export")
    parser.add_argument("--mesh-id", action="append", help="Filter to specific mesh IDs (repeatable)")
    parser.add_argument("--voxel-size", type=float, default=1.0, help="Voxel size for mesh voxelisation")
    parser.add_argument("--grid-origin", nargs=3, type=float, help="Grid origin (z y x) for index alignment")
    parser.add_argument("--threshold", type=float, default=0.001, help="Minimum mesh occupancy ratio in a patch")
    parser.add_argument("--shell-only", action="store_true", help="Keep only mesh shell voxels (no solid fill)")
    parser.add_argument("--validation", action="store_true", help="Sample validation dataset instead of training")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = ConfigManager(verbose=False)
    cfg.load_config(args.config)

    dataset = _build_dataset(cfg, is_training=not args.validation, mesh_override=None)

    selected_mesh_ids = set(args.mesh_id or [])
    grid_origin = _parse_triplet(args.grid_origin, default=(0.0, 0.0, 0.0)).astype(np.float32)
    voxel_size = float(args.voxel_size)
    fill_solid = not args.shell_only

    volume_boxes = _build_volume_mesh_boxes(
        dataset,
        voxel_size=voxel_size,
        grid_origin=grid_origin,
    )
    candidate_indices = _collect_candidate_patch_indices(
        dataset,
        volume_boxes=volume_boxes,
    )

    if candidate_indices:
        print(
            f"Found {len(candidate_indices)} candidate patch(es) overlapping mesh bounds (total patches: {len(dataset)}).",
            flush=True,
        )
        candidate_order: List[int] = candidate_indices
    else:
        candidate_order = list(range(len(dataset)))
        if volume_boxes:
            print(
                "Mesh bounds produced no direct candidates; falling back to scanning all patches.",
                flush=True,
            )
        else:
            print(
                "No mesh bounds detected for volumes; scanning all patches.",
                flush=True,
            )

    if not candidate_order:
        print("Dataset contains no patches to scan.")
        return

    total_candidates = len(candidate_order)
    progress_interval = max(1, total_candidates // 20) if total_candidates > 0 else None

    print(
        f"Scanning up to {total_candidates} candidate patches; sampling at most {args.max_samples}.",
        flush=True,
    )

    exported = 0
    scanned = 0

    for dataset_idx in candidate_order:
        sample = dataset[dataset_idx]
        scanned += 1
        meshes = sample.get("meshes")
        if not meshes:
            continue
        if selected_mesh_ids:
            available = set(meshes.keys()) & selected_mesh_ids
            if not available:
                continue
            mesh_items = {key: meshes[key] for key in available}
        else:
            mesh_items = dict(meshes)

        patch_info = sample.get("patch_info", {})
        position = (
            np.asarray(patch_info.get("position"), dtype=np.int64)
            if patch_info.get("position") is not None
            else None
        )
        patch_size = tuple(int(v) for v in patch_info.get("patch_size", []))
        if position is None or len(patch_size) != 3:
            continue  # skip non-chunk patches or insufficient metadata

        volume = _tensor_to_volume(sample["image"])
        if volume.shape != patch_size:
            continue

        combined_mask = np.zeros(patch_size, dtype=np.uint8)
        for mesh_id, info in mesh_items.items():
            mask = _compute_local_mask(
                info,
                patch_start=position,
                patch_size=patch_size,
                voxel_size=voxel_size,
                grid_origin=grid_origin,
                fill_solid=fill_solid,
            )
            combined_mask = np.maximum(combined_mask, mask)

        occupancy_ratio = combined_mask.mean() / 255.0
        if occupancy_ratio < args.threshold:
            if progress_interval and (scanned % progress_interval == 0):
                pct = (scanned / total_candidates) * 100.0 if total_candidates else 0.0
                print(
                    f"Scanned {scanned}/{total_candidates} candidate patches ({pct:.1f}%). "
                    f"Selected {exported} so far.",
                    flush=True,
                )
            continue

        sample_idx = exported + 1
        image_path = output_dir / f"sample_{sample_idx:03d}.tif"
        mask_path = output_dir / f"sample_{sample_idx:03d}_mask.tif"
        meta_path = output_dir / f"sample_{sample_idx:03d}.json"

        tifffile.imwrite(image_path, volume)
        tifffile.imwrite(mask_path, combined_mask)

        metadata = {
            "dataset_index": dataset_idx,
            "patch_info": patch_info,
            "meshes": [mesh_id for mesh_id in mesh_items],
            "mesh_fraction": occupancy_ratio,
            "voxel_size": args.voxel_size,
        }
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        exported += 1
        if exported >= args.max_samples:
            break

        if progress_interval:
            pct = (scanned / total_candidates) * 100.0 if total_candidates else 0.0
            print(
                f"Captured {exported}/{args.max_samples} patches after scanning {scanned} candidates ({pct:.1f}%).",
                flush=True,
            )

    if exported == 0:
        print("No mesh-containing patches matched the criteria.")
    else:
        print(f"Exported {exported} patch(es) to {output_dir}")


if __name__ == "__main__":
    main()
