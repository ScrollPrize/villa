#!/usr/bin/env python3
"""
Compute holes (Betti-1) and cavities (Betti-2) per connected component
for foreground voxels with a given value (default == 1).

Example:
  python component_topology.py \
      --in-dir /path/to/tifs \
      --out-csv ./component_topology.csv \
      --workers 6
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import importlib.util
try:
    import cupy as cp
    from cucim.skimage.measure import euler_number as cucim_euler_number
    from cucim.skimage.measure import label as cucim_label
    _HAS_CUPY = True
except Exception:
    cp = None
    cucim_euler_number = None
    cucim_label = None
    _HAS_CUPY = False

try:
    from skimage.measure import euler_number as sk_euler_number
    _HAS_SKIMAGE = True
except Exception:
    sk_euler_number = None
    _HAS_SKIMAGE = False
import numpy as np
import tifffile as tiff
from tqdm import tqdm

from scipy import ndimage as ndi

try:
    import cripser
    _HAS_CRIPSER = True
except ImportError:
    _HAS_CRIPSER = False

def _load_betti_matching_from_build() -> object:
    """Load betti_matching extension from the Vesuvius build directories."""
    repo_root = Path(__file__).resolve().parents[3]
    candidate_build_dirs = [
        repo_root / "external" / "Betti-Matching-3D" / "build",
        repo_root / "src" / "external" / "Betti-Matching-3D" / "build",
        repo_root / "scratch" / "Betti-Matching-3D" / "build",
    ]
    build_dir = next((p for p in candidate_build_dirs if p.exists()), None)
    if build_dir is None:
        raise ImportError(
            "Betti-Matching-3D build directory not found. Checked:\n"
            + "\n".join(f"  - {p}" for p in candidate_build_dirs)
        )
    candidates = []
    for pattern in ("betti_matching*.so", "betti_matching*.pyd", "betti_matching*.dll"):
        candidates.extend(build_dir.rglob(pattern))
    if not candidates:
        raise ImportError(f"betti_matching extension not found under {build_dir}")

    module_path = candidates[0]
    spec = importlib.util.spec_from_file_location("betti_matching", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load betti_matching from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


try:
    import betti_matching  # type: ignore
    _HAS_BETTI_MATCHING = True
except ImportError:
    try:
        betti_matching = _load_betti_matching_from_build()  # type: ignore
        _HAS_BETTI_MATCHING = True
    except Exception:
        _HAS_BETTI_MATCHING = False


TIFF_EXTS = {".tif", ".tiff"}


def list_tifs(folder: Path) -> List[Path]:
    return sorted([p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in TIFF_EXTS])


def list_subfolders(folder: Path) -> List[Path]:
    return sorted([p for p in folder.iterdir() if p.is_dir()])


def _suffix_from_glob(pattern: str) -> str:
    """
    Derive a reasonable suffix from a filename glob.
    Examples:
      "*_2d_hole_fill.tif" -> "2d_hole_fill"
      "*_surf.tif" -> "surf"
      "label.tif" -> "label"
    """
    stem = Path(pattern).stem  # e.g., "*_2d_hole_fill"
    # remove wildcards and leading separators
    s = stem.replace("*", "").replace("?", "")
    s = s.lstrip("_-.")
    s = s.rstrip("_-.")
    return s if s else "label"


def load_tif_3d(path: Path, allow_2d: bool = False) -> np.ndarray:
    arr = tiff.imread(str(path))
    arr = np.asarray(arr)
    if arr.ndim == 2:
        if not allow_2d:
            raise ValueError(f"{path.name}: expected 3D, got 2D. Use --allow-2d to auto-expand.")
        arr = arr[None, ...]  # (1, H, W)
    if arr.ndim != 3:
        raise ValueError(f"{path.name}: expected 3D array, got shape {arr.shape}")
    return arr


def _conn_to_skimage(conn: int, ndim: int) -> int:
    if ndim == 3:
        mapping = {6: 1, 18: 2, 26: 3}
    elif ndim == 2:
        mapping = {4: 1, 8: 2}
    else:
        raise ValueError(f"Unsupported ndim={ndim}")
    if conn not in mapping:
        raise ValueError(f"Invalid connectivity {conn} for ndim={ndim}")
    return mapping[conn]


def _border_mask(shape: Tuple[int, ...]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if len(shape) == 3:
        mask[0, :, :] = True
        mask[-1, :, :] = True
        mask[:, 0, :] = True
        mask[:, -1, :] = True
        mask[:, :, 0] = True
        mask[:, :, -1] = True
    elif len(shape) == 2:
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, 0] = True
        mask[:, -1] = True
    else:
        raise ValueError(f"Unsupported ndim={len(shape)}")
    return mask


def _bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[slice, ...]]:
    if mask.ndim == 0:
        return None
    axes = list(range(mask.ndim))
    slices: List[slice] = []
    for ax in axes:
        reduce_axes = tuple(a for a in axes if a != ax)
        proj = np.any(mask, axis=reduce_axes)
        if not proj.any():
            return None
        idx = np.where(proj)[0]
        slices.append(slice(int(idx[0]), int(idx[-1]) + 1))
    return tuple(slices)


def compute_component_topology(
    comp: np.ndarray,
    *,
    fg_connectivity: int,
    bg_connectivity: int,
    use_cpu: bool,
) -> Tuple[int, int, int]:
    """
    Returns (holes_b1, cavities_b2, euler_characteristic) for a single component mask.
    For 3D, uses b1 = b0 + b2 - chi, with b0=1.
    """
    comp = comp.astype(bool)
    comp_pad = np.pad(comp, pad_width=1, mode="constant", constant_values=False)

    # Cavities: background components not connected to the padded border
    bg = ~comp_pad
    if use_cpu:
        struct_bg = ndi.generate_binary_structure(comp_pad.ndim, _conn_to_skimage(bg_connectivity, comp_pad.ndim))
        bg_lab, num_bg = ndi.label(bg.astype(np.uint8), structure=struct_bg)
    else:
        if not _HAS_CUPY:
            raise ImportError("CuPy/cuCIM not available; use --cpu to run on CPU.")
        bg_gpu = cp.asarray(bg.astype(np.uint8))
        bg_lab = cp.asnumpy(cucim_label(bg_gpu, connectivity=_conn_to_skimage(bg_connectivity, comp_pad.ndim)))
        num_bg = int(bg_lab.max())
    border = _border_mask(bg_lab.shape)
    outside_ids = np.unique(bg_lab[border])
    outside_ids = outside_ids[outside_ids > 0]
    cavities = int(num_bg - outside_ids.size)

    # Euler characteristic (digital topology)
    if use_cpu:
        if not _HAS_SKIMAGE:
            raise ImportError("scikit-image is required for --cpu (Euler number).")
        chi = int(sk_euler_number(comp_pad, connectivity=_conn_to_skimage(fg_connectivity, comp_pad.ndim)))
    else:
        comp_pad_gpu = cp.asarray(comp_pad)
        chi = int(cucim_euler_number(comp_pad_gpu, connectivity=_conn_to_skimage(fg_connectivity, comp_pad.ndim)))

    # Betti numbers for a single component (b0 = 1)
    if comp_pad.ndim == 3:
        b2 = cavities
        b1 = int(1 + b2 - chi)
    else:  # 2D fallback
        b2 = 0
        b1 = int(1 - chi)

    return b1, b2, chi


def localize_holes(
    component_mask: np.ndarray,
    dilation_radius: int = 3,
    expected_b1: Optional[int] = None,
    backend: str = "cripser",
) -> Tuple[np.ndarray, List[Tuple[int, ...]]]:
    """Localize topological holes via persistent homology.

    Builds a sublevel-set filtration (foreground=0, background=1) and extracts
    H1 persistence pairs.  The *death cell* of each pair corresponds to the
    background voxel that would fill the hole — giving a precise spatial
    location for each topological tunnel.

    Parameters
    ----------
    component_mask : np.ndarray (bool-like)
        Binary mask of a single connected component.
    dilation_radius : int
        Radius for dilating each seed voxel into a localization region.
    expected_b1 : int or None
        If provided, warn when the cripser H1 count disagrees with the Euler-
        based Betti-1 number.

    Returns
    -------
    hole_mask : np.ndarray[bool]
        Same shape as *component_mask*; True where holes are localized.
    hole_coords : list of tuple
        One (z, y, x) coordinate per detected H1 feature (in the
        *component_mask* frame, i.e. before any global offset).
    """
    backend = backend.lower().strip()
    if backend == "cripser":
        if not _HAS_CRIPSER:
            raise ImportError(
                "cripser is required for --backend cripser. "
                "Install it with:  pip install cripser"
            )
    elif backend == "betti_matching":
        if not _HAS_BETTI_MATCHING:
            raise ImportError(
                "betti_matching is required for --backend betti_matching. "
                "Build or install it from the Betti-Matching-3D repo."
            )
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    base_mask = np.asarray(component_mask, dtype=bool)
    if not base_mask.any():
        return np.zeros(base_mask.shape, dtype=bool), []

    work_mask = base_mask
    offset = np.zeros(base_mask.ndim, dtype=int)
    bbox: Optional[Tuple[slice, ...]] = None
    if backend == "betti_matching":
        bbox = _bbox_from_mask(base_mask)
        if bbox is not None:
            work_mask = base_mask[bbox]
            offset = np.array([s.start for s in bbox], dtype=int)

    # Pad with 1-voxel background so boundary cycles close properly.
    padded = np.pad(work_mask, pad_width=1, mode="constant", constant_values=False)

    # Sublevel filtration: foreground = 0, background = 1.
    filtration = np.where(padded, 0.0, 1.0).astype(np.float64)

    hole_coords: List[Tuple[int, ...]] = []
    work_hole_mask = np.zeros(work_mask.shape, dtype=bool)

    if backend == "cripser":
        ph = cripser.compute_ph(filtration, maxdim=1)

        # Filter to H1 features (dim == 1)
        h1 = ph[ph[:, 0] == 1]

        for row in h1:
            # Cripser returns coordinates in the same order as array axes.
            # Even though the columns are labeled x/y/z, they correspond to
            # axis 0/1/2 of the input array. Do NOT reverse.
            coord = tuple(int(row[6 + i]) for i in range(padded.ndim))
            # Adjust for the 1-voxel padding
            orig = tuple(c - 1 for c in coord)
            if all(0 <= o < s for o, s in zip(orig, work_mask.shape)):
                hole_coords.append(tuple(int(o + d) for o, d in zip(orig, offset)))

        # Build hole mask by dilating around each seed coordinate.
        if hole_coords:
            seed = np.zeros(work_mask.shape, dtype=bool)
            for c in hole_coords:
                local = tuple(int(v - d) for v, d in zip(c, offset))
                seed[local] = True
            struct = ndi.generate_binary_structure(work_mask.ndim, 1)
            if dilation_radius > 0:
                work_hole_mask = ndi.binary_dilation(seed, structure=struct, iterations=dilation_radius)
            else:
                work_hole_mask = seed
    else:  # betti_matching
        bm = betti_matching.BettiMatching(filtration, filtration)
        bm.compute_matching()
        cycles = bm.compute_representative_cycles(
            input="input1",
            dim=1,
            matched="all",
            include_death_voxels=True,
            deduplicate_voxels=False,
        )

        if cycles.matched_cycles:
            struct = ndi.generate_binary_structure(work_mask.ndim, 1)
            shape = np.array(work_mask.shape, dtype=int)
            for cyc in cycles.matched_cycles:
                if cyc.size == 0:
                    continue
                coords = np.asarray(cyc, dtype=int)
                # Last coordinate is the death voxel when include_death_voxels=True.
                death = coords[-1]
                orig_death = death - 1
                if np.all((orig_death >= 0) & (orig_death < shape)):
                    hole_coords.append(tuple(int(v + d) for v, d in zip(orig_death, offset)))

                # Remove padding and clip to the component frame.
                valid = np.all((coords >= 1) & (coords <= shape), axis=1)
                coords = coords[valid] - 1
                if coords.size:
                    work_hole_mask[tuple(coords.T)] = True

            if work_hole_mask.any() and dilation_radius > 0:
                work_hole_mask = ndi.binary_dilation(work_hole_mask, structure=struct, iterations=dilation_radius)
        else:
            print("[INFO] betti_matching found 0 H1 features")

    if expected_b1 is not None and len(hole_coords) != expected_b1:
        import warnings
        warnings.warn(
            f"{backend} found {len(hole_coords)} H1 features but Euler-based b1 = {expected_b1}"
        )

    if bbox is not None:
        hole_mask = np.zeros(base_mask.shape, dtype=bool)
        hole_mask[bbox] = work_hole_mask
    else:
        hole_mask = work_hole_mask

    return hole_mask, hole_coords


def compute_file(
    path: Path,
    *,
    fg_value: int,
    ignore_value: Optional[int],
    fg_connectivity: int,
    bg_connectivity: int,
    allow_2d: bool,
    backend: str,
    use_cpu: bool,
    skip_euler: bool,
    whole_volume: bool,
) -> Tuple[List[Dict[str, int | float | str]], Optional[np.ndarray]]:
    arr = load_tif_3d(path, allow_2d=allow_2d)
    if ignore_value is not None and fg_value == ignore_value:
        raise ValueError("--fg-value must be different from --ignore-value")
    mask = (arr == fg_value)
    if ignore_value is not None:
        mask &= (arr != ignore_value)

    if whole_volume:
        rows: List[Dict[str, int | float | str]] = []
        hole_mask = np.zeros(arr.shape, dtype=bool)

        if skip_euler:
            holes, cavities, chi = 0, 0, 0
        else:
            holes, cavities, chi = compute_component_topology(
                mask,
                fg_connectivity=fg_connectivity,
                bg_connectivity=bg_connectivity,
                use_cpu=use_cpu,
            )

        hole_locations: List[str] = []
        comp_coords: List[Tuple[int, ...]] = []
        comp_holes: Optional[np.ndarray] = None

        if backend == "betti_matching":
            comp_holes, comp_coords = localize_holes(mask, expected_b1=holes, backend=backend)
        else:
            if holes > 0:
                comp_holes, comp_coords = localize_holes(mask, expected_b1=holes, backend=backend)

        holes_out = len(comp_coords) if backend == "betti_matching" else holes
        if comp_holes is not None and comp_holes.any():
            hole_mask |= comp_holes
            for coord in comp_coords:
                hole_locations.append(":".join(str(int(v)) for v in coord))

        rows.append({
            "case": path.stem,
            "component_id": 1,
            "voxels": int(mask.sum()),
            "holes": holes_out,
            "cavities": cavities,
            "euler": chi,
            "hole_locations": ";".join(hole_locations),
        })

        if not hole_mask.any():
            hole_mask = None
        return rows, hole_mask

    if use_cpu:
        struct_fg = ndi.generate_binary_structure(mask.ndim, _conn_to_skimage(fg_connectivity, mask.ndim))
        labels, num_labels = ndi.label(mask.astype(np.uint8), structure=struct_fg)
    else:
        if not _HAS_CUPY:
            raise ImportError("CuPy/cuCIM not available; use --cpu to run on CPU.")
        labels_gpu = cucim_label(cp.asarray(mask.astype(np.uint8)), connectivity=_conn_to_skimage(fg_connectivity, mask.ndim))
        labels = cp.asnumpy(labels_gpu)
        num_labels = int(labels.max())
    if num_labels == 0:
        return [], None

    slices = ndi.find_objects(labels)
    rows: List[Dict[str, int | float | str]] = []
    hole_mask = np.zeros(arr.shape, dtype=bool)

    for idx, slc in enumerate(slices, start=1):
        if slc is None:
            continue
        comp = (labels[slc] == idx)
        voxels = int(comp.sum())
        if voxels == 0:
            continue
        if skip_euler:
            holes, cavities, chi = 0, 0, 0
        else:
            holes, cavities, chi = compute_component_topology(
                comp,
                fg_connectivity=fg_connectivity,
                bg_connectivity=bg_connectivity,
                use_cpu=use_cpu,
            )
        hole_locations: List[str] = []
        comp_coords: List[Tuple[int, ...]] = []
        comp_holes: Optional[np.ndarray] = None

        if backend == "betti_matching":
            # Always compute per-component cycles when using betti_matching.
            comp_holes, comp_coords = localize_holes(comp, expected_b1=holes, backend=backend)
        else:
            if holes > 0:
                comp_holes, comp_coords = localize_holes(comp, expected_b1=holes, backend=backend)

        holes_out = len(comp_coords) if backend == "betti_matching" else holes
        if comp_holes is not None and comp_holes.any():
            hole_mask[slc] |= comp_holes
            # Convert component-local coords to global volume coords.
            for local_coord in comp_coords:
                global_coord = tuple(int(s.start + c) for s, c in zip(slc, local_coord))
                hole_locations.append(":".join(str(v) for v in global_coord))
        rows.append({
            "case": path.stem,
            "component_id": idx,
            "voxels": voxels,
            "holes": holes_out,
            "cavities": cavities,
            "euler": chi,
            "hole_locations": ";".join(hole_locations),
        })

    if not hole_mask.any():
        hole_mask = None
    return rows, hole_mask


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Compute holes (b1) and cavities (b2) per connected component (value==fg_value)."
    )
    ap.add_argument("--in-dir", type=Path, required=True, help="Folder with .tif/.tiff volumes")
    ap.add_argument("--fg-value", type=int, default=1, help="Foreground value to analyze (default: 1)")
    ap.add_argument("--ignore-value", type=int, default=None,
                    help="Label value to ignore (treated as background).")
    ap.add_argument("--fg-connectivity", type=int, default=6, choices=[6, 18, 26],
                    help="Connectivity for FG components (default: 6)")
    ap.add_argument("--bg-connectivity", type=int, default=6, choices=[6, 18, 26],
                    help="Connectivity for BG cavities (default: 6)")
    ap.add_argument("--workers", type=int, default=0, help="Number of parallel workers (0=sequential)")
    ap.add_argument("--allow-2d", action="store_true", help="Allow 2D TIFFs; auto-add singleton Z.")
    ap.add_argument("--cpu", action="store_true",
                    help="Force CPU-only labeling/Euler (recommended with --workers).")
    ap.add_argument("--backend", type=str, default="cripser",
                    choices=["cripser", "betti_matching"],
                    help="Backend for hole localization (default: cripser).")
    ap.add_argument("--subfolders", action="store_true",
                    help="Treat --in-dir as a root of per-sample subfolders.")
    ap.add_argument("--label-glob", type=str, default="*_surf.tif",
                    help="Glob for label file inside each subfolder (used with --subfolders).")
    ap.add_argument("--hole-mask-name", type=str, default="hole_mask.tif",
                    help="Output hole mask filename (used with --subfolders).")
    ap.add_argument("--hole-mask-suffix", type=str, default=None,
                    help="If set, output hole mask as hole_mask_<suffix>.tif (used with --subfolders).")
    ap.add_argument("--overwrite-hole-mask", action="store_true",
                    help="Overwrite existing hole mask (used with --subfolders).")
    ap.add_argument("--out-csv", type=Path, default=None, help="Output CSV path")
    ap.add_argument("--skip-euler", action="store_true",
                    help="Skip Euler/cavity computation (useful with --backend betti_matching).")
    ap.add_argument("--whole-volume", action="store_true",
                    help="Compute holes once on the full label volume (ignoring components).")
    ap.add_argument("--images-dir", type=Path, default=None,
                    help="Folder containing corresponding image TIFs (e.g. sample_00807.tif)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Output directory for per-sample folders (required when --images-dir given)")
    ap.add_argument("--stats-csv", type=Path, default=None,
                    help="Optional output CSV for hole statistics (per-case + overall).")
    ap.add_argument("--hist-png", type=Path, default=None,
                    help="Optional output PNG for histogram of holes per sample.")
    args = ap.parse_args()
    if args.images_dir is not None and args.out_dir is None:
        ap.error("--out-dir is required when --images-dir is given")
    return args


def main() -> None:
    args = parse_args()

    if args.workers and args.workers > 0 and not args.cpu:
        print("[WARN] --workers > 0 detected; forcing --cpu to avoid CUDA init errors.")
        args.cpu = True

    rows: List[Dict[str, int | float | str]] = []

    def _process_result_default(p: Path, result: Tuple[List[Dict], Optional[np.ndarray]]) -> None:
        file_rows, hole_mask = result
        rows.extend(file_rows)
        if hole_mask is not None and args.out_dir is not None:
            sample_dir = args.out_dir / p.stem
            sample_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, sample_dir / p.name)
            if args.images_dir is not None:
                image_stem = p.stem.replace("_surf", "")
                image_path = args.images_dir / (image_stem + p.suffix)
                if image_path.exists():
                    shutil.copy2(image_path, sample_dir / image_path.name)
                else:
                    print(f"[WARN] Image not found: {image_path}")
            tiff.imwrite(str(sample_dir / "hole_mask.tif"), hole_mask.astype(np.uint8))
            print(f"[OK] Saved hole output to {sample_dir}")

    def _process_result_subfolder(sample_dir: Path, label_path: Path, result: Tuple[List[Dict], Optional[np.ndarray]]) -> None:
        file_rows, hole_mask = result
        rows.extend(file_rows)
        if hole_mask is None:
            return
        out_dir = args.out_dir / sample_dir.name if args.out_dir is not None else sample_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.hole_mask_suffix:
            out_name = f"hole_mask_{args.hole_mask_suffix}.tif"
        else:
            out_name = args.hole_mask_name
        out_path = out_dir / out_name
        if out_path.exists() and not args.overwrite_hole_mask:
            print(f"[WARN] Hole mask exists, skipping write: {out_path}")
            return
        tiff.imwrite(str(out_path), hole_mask.astype(np.uint8))
        print(f"[OK] Saved hole output to {out_path}")

    if args.subfolders:
        if args.hole_mask_suffix is None:
            args.hole_mask_suffix = _suffix_from_glob(str(args.label_glob))
            print(f"[INFO] Using hole mask suffix from label glob: {args.hole_mask_suffix}")
        sample_dirs = list_subfolders(args.in_dir)
        if not sample_dirs:
            print("No subfolders found.")
            return
        # Build label list
        label_paths: List[Tuple[Path, Path]] = []
        for d in sample_dirs:
            matches = sorted(d.glob(args.label_glob))
            if not matches:
                print(f"[WARN] No label matches in {d} (glob={args.label_glob})")
                continue
            if len(matches) > 1:
                print(f"[WARN] Multiple label matches in {d}: {matches}. Using {matches[0].name}")
            label_paths.append((d, matches[0]))

        if args.workers and args.workers > 0:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futures = [
                    ex.submit(
                        compute_file,
                        label_path,
                        fg_value=int(args.fg_value),
                        ignore_value=None if args.ignore_value is None else int(args.ignore_value),
                        fg_connectivity=int(args.fg_connectivity),
                        bg_connectivity=int(args.bg_connectivity),
                        allow_2d=bool(args.allow_2d),
                        backend=str(args.backend),
                        use_cpu=bool(args.cpu),
                        skip_euler=bool(args.skip_euler),
                        whole_volume=bool(args.whole_volume),
                    )
                    for _, label_path in label_paths
                ]
                for fut, (sample_dir, label_path) in tqdm(zip(futures, label_paths), total=len(label_paths), desc="Computing"):
                    _process_result_subfolder(sample_dir, label_path, fut.result())
        else:
            for sample_dir, label_path in tqdm(label_paths, desc="Computing"):
                _process_result_subfolder(
                    sample_dir,
                    label_path,
                    compute_file(
                        label_path,
                        fg_value=int(args.fg_value),
                        ignore_value=None if args.ignore_value is None else int(args.ignore_value),
                        fg_connectivity=int(args.fg_connectivity),
                        bg_connectivity=int(args.bg_connectivity),
                        allow_2d=bool(args.allow_2d),
                        backend=str(args.backend),
                        use_cpu=bool(args.cpu),
                        skip_euler=bool(args.skip_euler),
                        whole_volume=bool(args.whole_volume),
                    ),
                )
    else:
        paths = list_tifs(args.in_dir)
        if not paths:
            print("No .tif/.tiff files found.")
            return

        if args.workers and args.workers > 0:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futures = [
                    ex.submit(
                        compute_file,
                        p,
                        fg_value=int(args.fg_value),
                        ignore_value=None if args.ignore_value is None else int(args.ignore_value),
                        fg_connectivity=int(args.fg_connectivity),
                        bg_connectivity=int(args.bg_connectivity),
                        allow_2d=bool(args.allow_2d),
                        backend=str(args.backend),
                        use_cpu=bool(args.cpu),
                        skip_euler=bool(args.skip_euler),
                        whole_volume=bool(args.whole_volume),
                    )
                    for p in paths
                ]
                for fut, p in tqdm(zip(futures, paths), total=len(paths), desc="Computing"):
                    _process_result_default(p, fut.result())
        else:
            for p in tqdm(paths, desc="Computing"):
                _process_result_default(
                    p,
                    compute_file(
                        p,
                        fg_value=int(args.fg_value),
                        ignore_value=None if args.ignore_value is None else int(args.ignore_value),
                        fg_connectivity=int(args.fg_connectivity),
                        bg_connectivity=int(args.bg_connectivity),
                        allow_2d=bool(args.allow_2d),
                        backend=str(args.backend),
                        use_cpu=bool(args.cpu),
                        skip_euler=bool(args.skip_euler),
                        whole_volume=bool(args.whole_volume),
                    ),
                )

    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        fields = ["case", "component_id", "voxels", "holes", "cavities", "euler", "hole_locations"]
        with args.out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for row in rows:
                w.writerow(row)
        print(f"[OK] Wrote {args.out_csv}")

    if rows:
        stats_path = args.stats_csv
        if stats_path is None and args.out_csv is not None:
            stats_path = args.out_csv.with_name(args.out_csv.stem + "_hole_stats.csv")

        by_case: Dict[str, List[int]] = {}
        for r in rows:
            by_case.setdefault(str(r["case"]), []).append(int(r["holes"]))

        def _summarize(vals: List[int]) -> Dict[str, float | int]:
            arr = np.asarray(vals, dtype=float)
            return {
                "components": int(arr.size),
                "holes_sum": int(arr.sum()),
                "holes_mean": float(arr.mean()) if arr.size else float("nan"),
                "holes_median": float(np.median(arr)) if arr.size else float("nan"),
                "holes_min": int(arr.min()) if arr.size else 0,
                "holes_max": int(arr.max()) if arr.size else 0,
                "holes_std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            }

        if stats_path is not None:
            stat_fields = [
                "case",
                "components",
                "holes_sum",
                "holes_mean",
                "holes_median",
                "holes_min",
                "holes_max",
                "holes_std",
            ]
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            with stats_path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=stat_fields)
                w.writeheader()
                for case in sorted(by_case.keys()):
                    s = _summarize(by_case[case])
                    w.writerow({"case": case, **s})
                all_vals = [int(r["holes"]) for r in rows]
                w.writerow({"case": "__all__", **_summarize(all_vals)})

            print(f"[OK] Wrote {stats_path}")

        if args.hist_png is not None:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
            except Exception as e:
                print(f"[WARN] Could not import matplotlib; histogram not written: {e}")
            else:
                holes_per_sample = [int(sum(v)) for v in by_case.values()]
                if holes_per_sample:
                    plt.figure()
                    plt.hist(holes_per_sample, bins=30)
                    plt.title("Holes per sample")
                    plt.xlabel("holes")
                    plt.ylabel("count")
                    plt.tight_layout()
                    args.hist_png.parent.mkdir(parents=True, exist_ok=True)
                    plt.savefig(args.hist_png)
                    plt.close()
                    print(f"[OK] Wrote {args.hist_png}")


if __name__ == "__main__":
    main()
