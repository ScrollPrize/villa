"""vesuvius.compute_st on a scroll volume given as its OME-Zarr root, and the
eigenanalysis DataLoader with worker processes started by spawn.

compute_st opened --input_dir as an array, so an OME-Zarr root failed with
"'Group' object has no attribute 'shape'". The eigenanalysis dataset was a class
defined inside a function with a lambda collate, which spawn and forkserver
workers (macOS, Python 3.14 on Linux) cannot pickle.
"""
import subprocess
import sys
import textwrap

import numpy as np
import zarr

from vesuvius.data.utils import create_zarr_array, open_zarr, open_zarr_group
from vesuvius.structure_tensor.run_create_st import _array_path


def test_array_path_uses_level_0_of_an_ome_group(tmp_path):
    path = str(tmp_path / "vol.zarr")
    open_zarr_group(path, mode="w")
    open_zarr(f"{path}/0", mode="w", shape=(4, 4, 4), dtype=np.uint8)
    assert _array_path(path) == f"{path}/0"
    assert _array_path(path + "/") == f"{path}/0"


def test_array_path_keeps_a_bare_array(tmp_path):
    path = str(tmp_path / "vol.zarr")
    open_zarr(path, mode="w", shape=(4, 4, 4), dtype=np.uint8)
    assert _array_path(path) == path


def test_eigenanalysis_runs_with_spawned_workers(tmp_path):
    Z, Y, X = 4, 4, 4
    st = np.zeros((6, Z, Y, X), dtype=np.float32)
    st[0] = 3.0  # Jzz
    st[3] = 2.0  # Jyy
    st[5] = 1.0  # Jxx
    zarr_path = str(tmp_path / "st.zarr")
    root = open_zarr_group(zarr_path, mode="w")
    create_zarr_array(root, "structure_tensor", data=st, chunks=(6, 2, Y, X), dtype="f4")

    script = textwrap.dedent(f"""
        import multiprocessing
        from vesuvius.structure_tensor.create_st import _finalize_structure_tensor_torch
        if __name__ == "__main__":
            multiprocessing.set_start_method("spawn", force=True)
            _finalize_structure_tensor_torch(
                zarr_path={zarr_path!r}, chunk_size=(2, {Y}, {X}), num_workers=2,
                compressor=None, verbose=False, device="cpu")
    """)
    result = subprocess.run([sys.executable, "-c", script], capture_output=True,
                            text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    assert tuple(zarr.open_group(zarr_path, mode="r")["confidence/0"].shape) == (Z, Y, X)
