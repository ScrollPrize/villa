"""``python -m vesuvius.structure_tensor.create_st --mode structure-tensor`` must honour
the eigen/OME options it advertises.

The structure-tensor branch called the eigen step without --keep-eigen, --ome-downsample,
--ome-scale, --confidence-metric or --ome-out, so they were silently replaced by the
defaults while the closing summary still listed eigenvectors and the requested scale.
(The ``vesuvius.compute_st`` runner and ``--mode eigenanalysis`` already passed them.)
"""

from __future__ import annotations

import sys

import numpy as np
import zarr

from vesuvius.structure_tensor import create_st


def test_structure_tensor_mode_passes_eigen_options(tmp_path, monkeypatch):
    src = tmp_path / "in.zarr"
    g = zarr.open_group(str(src), mode="w")
    rng = np.random.default_rng(0)
    g.create_array("0", data=(rng.random((9, 16, 16)) * 255).astype(np.uint8), chunks=(9, 16, 16))
    out = tmp_path / "st"
    monkeypatch.setattr(sys, "argv", [
        "create_st", "--mode", "structure-tensor", "--input_dir", str(src / "0"),
        "--output_dir", str(out), "--device", "cpu", "--keep-eigen",
        "--ome-downsample", "2", "--ome-scale", "1", "--chunk-size", "8,8,8",
        "--patch_size", "9,16,16",
    ])
    monkeypatch.setenv("VESUVIUS_DISABLE_TORCH_COMPILE", "1")
    create_st.main()
    root = zarr.open_group(str(out) + ".zarr", mode="r")
    assert "eigenvectors" in root and "eigenvalues" in root
    assert list(root["normal/z"].keys()) == ["1"]
    assert root["normal/z/1"].shape == (5, 8, 8)
