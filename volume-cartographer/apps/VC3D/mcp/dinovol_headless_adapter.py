#!/usr/bin/env python3
"""Pinned DinoVol MPS exemplar search over registered raw-CT artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

EXPECTED_CHECKPOINT_SHA256 = "e041ca870dd2570f8a44d1dd26db1197b3f74121f62023bc774fbc9d40e51a59"
MAX_VOXELS = 16_777_216


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normals(xyz: np.ndarray, valid: np.ndarray) -> np.ndarray:
    du = np.gradient(xyz, axis=1)
    dv = np.gradient(xyz, axis=0)
    value = np.cross(du, dv).astype(np.float32)
    length = np.linalg.norm(value, axis=-1)
    supported = valid & np.isfinite(length) & (length > 1e-8)
    value[~supported] = 0
    value[supported] /= length[supported, None]
    return value


def project_nearest(volume: np.ndarray, xyz: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    index = np.rint(xyz).astype(np.int64)
    inside = (
        valid & (index[..., 0] >= 0) & (index[..., 1] >= 0) & (index[..., 2] >= 0)
        & (index[..., 0] < volume.shape[2]) & (index[..., 1] < volume.shape[1])
        & (index[..., 2] < volume.shape[0])
    )
    result = np.zeros(valid.shape, np.float32)
    result[inside] = volume[index[..., 2][inside], index[..., 1][inside], index[..., 0][inside]]
    return result, inside


def preview(path: Path, values: np.ndarray, valid: np.ndarray) -> None:
    from PIL import Image
    image = np.zeros(values.shape, np.uint8)
    selected = values[valid & np.isfinite(values)]
    if selected.size:
        low, high = np.percentile(selected, (1, 99))
        if high > low:
            image[valid] = np.clip((values[valid] - low) * 255 / (high - low), 0, 255).astype(np.uint8)
    Image.fromarray(image).save(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--surface", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("mps",), required=True)
    args = parser.parse_args()

    request = json.loads(args.request.read_text())
    if sha256(args.checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("DinoVol checkpoint SHA-256 mismatch")
    commit = subprocess.check_output(["git", "-C", str(args.repository), "rev-parse", "HEAD"], text=True).strip()
    if commit != request["repository_commit"]:
        raise ValueError(f"DinoVol repository commit mismatch: {commit}")

    sys.path.insert(0, str(args.repository))
    import torch
    import zarr
    from dinovol_2.eval.napari_visualizer import (
        compute_patch_embedding_grid,
        load_backbone_from_checkpoint,
        point_to_patch_index,
        upsample_patch_grid_to_volume,
    )

    if not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable")
    stage_manifest = json.loads((args.surface / "staging" / "stage-manifest.json").read_text())
    raw = np.load(args.surface / "staging" / "staged-volume.npy", allow_pickle=False)
    if raw.ndim != 3 or raw.size > MAX_VOXELS:
        raise ValueError(f"expected a bounded raw CT ZYX volume, got {raw.shape}")
    root = zarr.open_group(str(args.surface / "surface.zarr"), mode="r")
    surface_xyz = np.asarray(root["geometry/xyz"], np.float32)
    valid = np.asarray(root["geometry/valid"]) != 0
    region = stage_manifest["submitted_region_xyz"]
    local_xyz = surface_xyz.copy()
    local_xyz[..., 0] -= int(region["x"])
    local_xyz[..., 1] -= int(region["y"])
    local_xyz[..., 2] -= int(region["z"])
    surface_normals = normals(local_xyz, valid)

    loaded = load_backbone_from_checkpoint(args.checkpoint, device=torch.device("mps"), preferred_branch="teacher")
    embeddings, source_shape, patch_grid = compute_patch_embedding_grid(raw, loaded)

    def exemplar_mean(examples: list[dict]) -> np.ndarray:
        maps = []
        for example in examples:
            u, v = int(round(example["u"])), int(round(example["v"]))
            if v < 0 or u < 0 or v >= valid.shape[0] or u >= valid.shape[1] or not valid[v, u]:
                raise ValueError(f"invalid surface exemplar ({u},{v})")
            point_xyz = local_xyz[v, u] + float(example.get("offset", 0)) * surface_normals[v, u]
            point_zyx = (float(point_xyz[2]), float(point_xyz[1]), float(point_xyz[0]))
            index = point_to_patch_index(point_zyx, source_shape=source_shape,
                                         patch_size=loaded.patch_size,
                                         patch_grid_shape=embeddings.shape[:3])
            maps.append(np.tensordot(embeddings, embeddings[index], axes=([-1], [0])))
        return np.mean(maps, axis=0, dtype=np.float32) if maps else np.zeros(embeddings.shape[:3], np.float32)

    positive_grid = exemplar_mean(request["positive_examples"])
    negative_grid = exemplar_mean(request.get("negative_examples", []))
    combined_grid = positive_grid - negative_grid
    args.output.mkdir(parents=True, exist_ok=True)
    surface_maps = {}
    support = None
    for name, grid in (("positive-similarity", positive_grid), ("negative-similarity", negative_grid),
                       ("exemplar-similarity", combined_grid)):
        dense = upsample_patch_grid_to_volume(grid, patch_size=loaded.patch_size, output_shape=source_shape)
        np.save(args.output / f"{name}-volume.npy", dense, allow_pickle=False)
        projected, current_support = project_nearest(dense, local_xyz, valid)
        support = current_support if support is None else support & current_support
        np.save(args.output / f"{name}-surface.npy", projected, allow_pickle=False)
        preview(args.output / f"{name}-surface.png", projected, current_support)
        surface_maps[name] = {
            "npy": f"{name}-surface.npy", "preview": f"{name}-surface.png",
            "minimum": float(projected[current_support].min()) if current_support.any() else 0.0,
            "maximum": float(projected[current_support].max()) if current_support.any() else 0.0,
        }
    np.save(args.output / "surface-support.npy", support.astype(np.uint8), allow_pickle=False)

    manifest = {
        "kind": "dinovol_registered_exemplar_v1",
        "backend": "mps",
        "checkpoint": args.checkpoint.name,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "repository_commit": commit,
        "model_repository_commit": "6a8cccbafef191a966da815e22ff5c6eae075aae",
        "source_surface_artifact": request["surface"],
        "patch_size": list(loaded.patch_size),
        "embedding_dimension": int(embeddings.shape[-1]),
        "raw_ct_shape_zyx": list(source_shape),
        "embedding_grid_shape": list(embeddings.shape),
        "surface_shape_vu": list(valid.shape),
        "projected_support_fraction": float(support.sum() / max(1, valid.sum())),
        "positive_examples": request["positive_examples"],
        "negative_examples": request.get("negative_examples", []),
        "surface_maps": surface_maps,
        "surface_support": "surface-support.npy",
        "score_semantics": "positive_minus_negative_cosine_similarity_not_ink_probability",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
