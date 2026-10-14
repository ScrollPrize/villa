"""Pre-cache metadata.json segment meshes and ink-labels for all volumes in a pretrain config.

Usage:
    python -m dinovol_2.cache_metadata_meshes --config dinovol_2/configs/pretrain_8_9um.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import urllib.request

from dinovol_2.dataset.object_labels import load_or_build_volume_meshes
from dinovol_2.dataset.zarr_io import open_zarr_array


DEFAULT_METADATA_URL = "https://vesuvius-challenge-open-data.s3.amazonaws.com/metadata.json"


def ensure_metadata(metadata_path: Path, url: str = DEFAULT_METADATA_URL) -> dict:
    if not metadata_path.is_file():
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {url} -> {metadata_path} ...")
        urllib.request.urlretrieve(url, metadata_path)
    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("dinovol_2/configs/pretrain_8_9um.json"),
        help="Path to the pretrain JSON configuration file.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("/tmp/vesuvius_metadata.json"),
        help="Local path to cache metadata.json.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Override directory for cached mesh .npz files (defaults to config dataset.object_points.mesh_cache_dir).",
    )
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as f:
        config = json.load(f)

    dataset_cfg = dict(config.get("dataset") or {})
    obj_cfg = dict(dataset_cfg.get("object_points") or {})
    cache_dir = args.cache_dir or Path(obj_cfg.get("mesh_cache_dir", "./mesh_cache_8_9um"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    metadata = ensure_metadata(args.metadata)
    s3_opts = dict(dataset_cfg.get("s3_storage_options") or {"anon": True})
    datasets = list(dataset_cfg.get("datasets") or [])

    total_segments = 0
    total_ink_segments = 0
    volumes_with_meshes = 0
    for idx, entry in enumerate(datasets):
        vol_path = str(entry["volume_path"])
        vol_scale = str(entry.get("volume_scale", 0))
        scale0_arr = open_zarr_array(vol_path, "0", storage_options=s3_opts)
        sel_arr = open_zarr_array(vol_path, vol_scale, storage_options=s3_opts)
        scale0_shape = tuple(int(d) for d in scale0_arr.shape[:3])
        sel_shape = tuple(int(d) for d in sel_arr.shape[:3])
        meshes = load_or_build_volume_meshes(
            vol_path,
            scale0_shape,
            sel_shape,
            metadata=metadata,
            cache_dir=cache_dir,
            s3_storage_options=s3_opts,
        )
        n_ink = sum(1 for m in meshes if m.has_ink)
        if meshes:
            volumes_with_meshes += 1
        total_segments += len(meshes)
        total_ink_segments += n_ink
        print(
            f"[{idx + 1}/{len(datasets)}] {entry.get('sample_id', vol_path)}: "
            f"{len(meshes)} meshes ({n_ink} with ink-labels)"
        )

    print(
        f"Done: cached {total_segments} meshes ({total_ink_segments} with ink-labels) "
        f"across {volumes_with_meshes}/{len(datasets)} volumes in {cache_dir}"
    )


if __name__ == "__main__":
    main()
