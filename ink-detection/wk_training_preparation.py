import os
import sys
import shutil
import glob
from typing import List, Optional
from pathlib import Path

import numpy as np
from PIL import Image

try:
    # WEBKNOSSOS Python API
    from webknossos import webknossos_context
    from webknossos.dataset import Dataset
    # Geometry helpers (names based on public API docs)
    from webknossos.geometry import Mag, Vec3Int, NDBoundingBox  # type: ignore
except Exception as exc:  # pragma: no cover
    print("ERROR: webknossos package is required. Install with: pip install webknossos", file=sys.stderr)
    raise


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def center_indices(total: int, take: int) -> List[int]:
    if take >= total:
        return list(range(total))
    start = (total - take) // 2
    return list(range(start, start + take))


def try_get_highres_mag_view(layer):
    # Prefer explicit Mag(1,1,1) if present. Fall back to first available.
    try:
        return layer.get_mag(Mag(1, 1, 1))
    except Exception:
        pass
    try:
        mags = list(getattr(layer, "mags", []))
        if mags:
            # Some APIs expose MagViews directly in layer.mags
            return sorted(mags, key=lambda m: (getattr(m, "mag", getattr(m, "downsampling", (1, 1, 1))).x,
                                               getattr(m, "mag", getattr(m, "downsampling", (1, 1, 1))).y,
                                               getattr(m, "mag", getattr(m, "downsampling", (1, 1, 1))).z))[0]
    except Exception:
        pass
    # Fallback: some versions expose get_mags()
    try:
        mags = list(layer.get_mags())
        if mags:
            return mags[0]
    except Exception:
        pass
    raise RuntimeError("Could not obtain a MagView for the layer")


def try_view_shape(view) -> tuple:
    for attr in ("shape", "size", "volume_size", "data_shape"):
        shp = getattr(view, attr, None)
        if shp is not None:
            if callable(shp):
                try:
                    shp = shp()
                except Exception:
                    shp = None
            if shp is not None:
                return tuple(shp)
    # As a last resort, read a small bbox to infer Y,X dims, and probe Z by binary search is too heavy.
    # Fall back to a large Z guess and clamp by exceptions while reading.
    return (256, 256, 256)


def try_read_z_slice(view, z: int) -> np.ndarray:
    # Try various APIs that might exist across versions
    # 1) direct get_slice
    try:
        arr = view.get_slice(z=z)
        return np.asarray(arr)
    except Exception:
        pass
    # 2) read via bounding box
    try:
        # Determine X,Y extents from view or try a common pattern
        shp = try_view_shape(view)
        y = int(shp[-2])
        x = int(shp[-1])
        # Bounding boxes typically: begin Vec3Int(x, y, z), size Vec3Int(x, y, z)
        # NDBoundingBox(start, size) with order (x, y, z)
        # Many APIs use (x, y, z). We choose entire slice at z with size z=1
        bbox = NDBoundingBox(Vec3Int(0, 0, z), Vec3Int(x, y, 1))
        arr = view.read(bbox)
        # Expected shape (1, y, x) or (y, x)
        return np.asarray(arr)
    except Exception:
        pass
    # 3) some APIs use read with slices or indices
    try:
        arr = view.read(z)
        return np.asarray(arr)
    except Exception:
        pass
    raise RuntimeError("Could not read z slice from MagView")


def export_layer_as_tiff_slices(layer, out_layers_dir: Path, take_slices: int = 26, name_start_idx: int = 17) -> List[Path]:
    """
    Export central 'take_slices' z-slices of the highest-resolution mag as 2D TIFF images
    into out_layers_dir with names 00.tif, 01.tif, ... matching train_timesformer_og.py.
    """
    # Pick the highest available mag (smallest voxel size). Fall back to the first one.
    view = try_get_highres_mag_view(layer)

    vol_shape = try_view_shape(view)  # prefer (z, y, x)
    # Try to deduce Z as the first dim; if not plausible, assume first is Z
    total_slices = int(vol_shape[0])
    slice_idxs = center_indices(total_slices, take_slices)

    ensure_dir(out_layers_dir)
    written = []
    for local_i, z in enumerate(slice_idxs):
        # read a single z slice
        arr = try_read_z_slice(view, z=z)  # expected to return (Y, X) or (1, Y, X)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim == 3 and arr.shape[-1] in (3, 4):
            # convert RGB(A) to grayscale expected by training code
            arr = np.mean(arr[..., :3], axis=-1)
        arr = np.asarray(arr)

        # clip to [0, 200] like training does
        arr = np.clip(arr, 0, 200).astype(np.uint16)

        name_i = name_start_idx + local_i
        out_path = out_layers_dir / f"{name_i:02}.tif"
        Image.fromarray(arr).save(out_path)
        written.append(out_path)
    return written


def copy_labels_if_available(fragment_dir: Path, fragment_id: str, labels_root: Path) -> None:
    """Copy label and mask files from all_labels/ if present into the fragment_dir."""
    # Labels come in png or tiff per project structure
    png_label = labels_root / f"{fragment_id}_inklabels.png"
    tiff_label = labels_root / f"{fragment_id}_inklabels.tiff"
    if png_label.exists():
        shutil.copy2(png_label, fragment_dir / f"{fragment_id}_inklabels.png")
    if tiff_label.exists():
        shutil.copy2(tiff_label, fragment_dir / f"{fragment_id}_inklabels.tiff")

    # If mask not present but can be inferred from any slice, create a full-ones mask
    mask_png = fragment_dir / f"{fragment_id}_mask.png"
    if not mask_png.exists():
        any_slice = next(iter(glob.glob(str(fragment_dir / "layers" / "*.tif"))), None)
        if any_slice is not None:
            arr = np.array(Image.open(any_slice))
            ones = np.ones_like(arr, dtype=np.uint8) * 255
            Image.fromarray(ones).save(mask_png)


def list_remote_datasets() -> List[Dataset]:
    datasets = Dataset.get_remote_datasets()
    # Some instances return datasets across orgs; keep all for now
    return datasets


def download_dataset_to_training_layout(dataset: Dataset, out_root: Path, labels_root: Path, take_slices: int = 26) -> Optional[Path]:
    """
    Download a remote dataset (WKW) and export to train_timesformer_og.py layout:
    train_scrolls/{fragment_id}/layers/{i:02}.tif + optional inklabels/mask.
    """
    fragment_id = dataset.name
    fragment_dir = out_root / "train_scrolls" / fragment_id
    layers_dir = fragment_dir / "layers"

    if layers_dir.exists() and len(list(layers_dir.glob("*.tif"))) >= take_slices:
        # Already prepared
        copy_labels_if_available(fragment_dir, fragment_id, labels_root)
        return fragment_dir

    print(f"Preparing dataset '{dataset.name}' -> {fragment_dir}")

    # Download dataset locally (as WKW) to a temp/cache directory under fragment_dir
    local_store = fragment_dir / "_wkw"
    ensure_dir(local_store)
    local_ds = dataset.download(local_store)

    # Heuristic: pick the first image layer
    image_layers = [ly for ly in local_ds.layers if ly.category == "image"] or list(local_ds.layers)
    if not image_layers:
        print(f"WARNING: No layers found for dataset {dataset.name}. Skipping.")
        return None
    layer = image_layers[0]

    # Use naming 17..42 to match train_timesformer_og.py defaults
    export_layer_as_tiff_slices(layer, layers_dir, take_slices=take_slices, name_start_idx=17)

    # Copy labels if present in all_labels/
    copy_labels_if_available(fragment_dir, fragment_id, labels_root)

    return fragment_dir


def main() -> None:
    url = os.getenv("WEBKNOSSOS_URL")
    token = os.getenv("WEBKNOSSOS_TOKEN")
    if not url or not token:
        print("ERROR: Please set WEBKNOSSOS_URL and WEBKNOSSOS_TOKEN environment variables.", file=sys.stderr)
        print("See docs: https://docs.webknossos.org/webknossos-py/", file=sys.stderr)
        sys.exit(1)

    # Optional: filter datasets by names via CLI env or arg
    only_datasets_env = os.getenv("WEBKNOSSOS_DATASETS", "").strip()
    only_datasets = [s for s in (d.strip() for d in only_datasets_env.split(",")) if s]

    project_root = Path(__file__).resolve().parent
    out_root = project_root
    labels_root = project_root / "all_labels"

    # Initialize context using env vars
    with webknossos_context(url=url, token=token):
        datasets = list_remote_datasets()
        print(datasets)
        if only_datasets:
            datasets = [d for d in datasets if d.name in only_datasets]

        if not datasets:
            print("No datasets available from WEBKNOSSOS with current credentials.")
            return

        prepared = []
        for ds in datasets:
            try:
                fragment_path = download_dataset_to_training_layout(ds, out_root, labels_root, take_slices=26)
                if fragment_path is not None:
                    prepared.append(fragment_path)
            except Exception as e:  # keep going on failures
                print(f"ERROR while preparing {ds.name}: {e}")

        if prepared:
            print("Prepared fragments:")
            for p in prepared:
                print(f"- {p}")
        else:
            print("No fragments prepared.")


if __name__ == "__main__":
    main()


