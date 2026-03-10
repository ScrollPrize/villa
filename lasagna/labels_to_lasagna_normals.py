"""Pipeline: binary label TIFF → binary zarr → vc_gen_normalgrids →
vc_ngrids --fit-normals --lasagna-format → lasagna-format normals zarr.

The lasagna-format zarr is (C, Z, Y, X) uint8 with channels
[cos, grad_mag, nx, ny] and preprocess_params metadata,
ready to be consumed by fit_data.load_3d().

Usage
-----
  python labels_to_lasagna_normals.py \
      --input labels.tif \
      --work-dir ./work \
      --output normals_lasagna.zarr \
      --step 4
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np
import tifffile
import zarr


TAG = "[labels_to_lasagna_normals]"


def _run(cmd: list[str], label: str) -> None:
    """Run a subprocess, printing its label and streaming output."""
    print(f"{TAG} running: {label}", flush=True)
    print(f"{TAG}   cmd: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"{label} failed with rc={proc.returncode}")


def _write_binary_zarr(vol: np.ndarray, out_path: Path, chunk_size: int) -> None:
    """Write binary uint8 volume as zarr group with dataset '0'."""
    store = zarr.DirectoryStore(str(out_path), dimension_separator="/")
    root = zarr.group(store, overwrite=True)
    root.create_dataset(
        "0",
        data=vol,
        chunks=(chunk_size, chunk_size, chunk_size),
        dtype=np.uint8,
        overwrite=True,
    )
    print(f"{TAG} binary zarr written: {out_path}  shape={vol.shape}", flush=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Label TIFF → binary zarr → vc normal grids → lasagna normals zarr",
    )
    p.add_argument("--input", required=True,
                    help="Multi-layer label TIFF (ZYX): 0=bg, 1=pred, 2=ignore")
    p.add_argument("--work-dir", required=True,
                    help="Working directory for intermediate files")
    p.add_argument("--output", required=True,
                    help="Output lasagna-format normals zarr (C,Z,Y,X)")
    p.add_argument("--step", type=int, default=4,
                    help="Sample step for vc_ngrids (also becomes scaledown in lasagna zarr, default: 4)")
    p.add_argument("--sparse-volume", type=int, default=1,
                    help="vc_gen_normalgrids --sparse-volume (default: 1)")
    p.add_argument("--chunk-size", type=int, default=64,
                    help="Zarr chunk size per axis")
    p.add_argument("--skip-gen-normalgrids", action="store_true",
                    help="Skip vc_gen_normalgrids (reuse existing grids in work-dir)")
    p.add_argument("--skip-fit-normals", action="store_true",
                    help="Skip vc_ngrids --fit-normals (reuse existing ngrids zarr)")
    args = p.parse_args(argv)

    input_path = Path(args.input)
    work_dir = Path(args.work_dir)
    output_path = Path(args.output)
    work_dir.mkdir(parents=True, exist_ok=True)

    binary_zarr_path = work_dir / "binary.zarr"
    grids_path = work_dir / "normalgrids"
    ngrids_zarr_path = work_dir / "ngrids_normals.zarr"

    # -- Step 1: Read TIFF, write binary zarr --------------------------------
    print(f"{TAG} step 1: reading {input_path}", flush=True)
    vol = tifffile.imread(str(input_path))
    if vol.ndim == 2:
        vol = vol[np.newaxis]
    assert vol.ndim == 3, f"expected 3D volume, got shape {vol.shape}"

    n_bg = int((vol == 0).sum())
    n_fg = int((vol == 1).sum())
    n_ign = int((vol == 2).sum())
    print(f"{TAG} volume shape={vol.shape}  bg={n_bg}  pred={n_fg}  ignore={n_ign}", flush=True)

    binary = (vol == 1).astype(np.uint8)

    if not args.skip_gen_normalgrids:
        _write_binary_zarr(binary, binary_zarr_path, args.chunk_size)

    # -- Step 2: vc_gen_normalgrids ------------------------------------------
    if not args.skip_gen_normalgrids:
        grids_path.mkdir(parents=True, exist_ok=True)
        cmd = [
            "vc_gen_normalgrids",
            "-i", str(binary_zarr_path),
            "-o", str(grids_path),
        ]
        if args.sparse_volume > 1:
            cmd += ["--sparse-volume", str(args.sparse_volume)]
        _run(cmd, "vc_gen_normalgrids")
    else:
        print(f"{TAG} step 2: skipping vc_gen_normalgrids (--skip-gen-normalgrids)", flush=True)

    # -- Step 3: vc_ngrids --fit-normals --lasagna-format --------------------
    if not args.skip_fit_normals:
        if ngrids_zarr_path.exists():
            shutil.rmtree(ngrids_zarr_path)
        cmd = [
            "vc_ngrids",
            "-i", str(grids_path),
            "--fit-normals",
            "--output-zarr", str(ngrids_zarr_path),
            "--step", str(args.step),
            "--lasagna-format",
        ]
        _run(cmd, "vc_ngrids --fit-normals")
    else:
        print(f"{TAG} step 3: skipping vc_ngrids --fit-normals (--skip-fit-normals)", flush=True)

    # -- Step 4: Move lasagna zarr to output ---------------------------------
    las_path = Path(str(ngrids_zarr_path) + ".lasagna.zarr")
    if output_path != las_path:
        if output_path.exists():
            shutil.rmtree(output_path)
        shutil.move(str(las_path), str(output_path))
    print(f"{TAG} lasagna zarr: {output_path}", flush=True)

    print(f"{TAG} pipeline complete: {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
