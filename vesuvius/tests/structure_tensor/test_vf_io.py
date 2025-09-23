import os
import numpy as np
import torch
import zarr

from vesuvius.structure_tensor.vf_io import load_density_mask, load_eigenvector_field


def test_load_density_mask_with_and_without_channel(tmp_path):
    Z, Y, X = 4, 3, 5
    vol = np.zeros((Z, Y, X), dtype=np.uint8)
    vol[1:3, 1:2, 2:4] = 7
    path_nochan = os.path.join(tmp_path, "raw_nochan.zarr")
    zarr.save_array(path_nochan, vol, chunks=(Z, Y, X))

    mask = load_density_mask(path_nochan, volume_id=7, bounds=(0, Z, 0, Y, 0, X))
    assert mask.shape == (Z, Y, X)
    assert mask.dtype == torch.float32
    assert mask.sum().item() == float((vol == 7).sum())

    # with a leading channel axis
    volc = vol[None]  # shape (1,Z,Y,X)
    path_chan = os.path.join(tmp_path, "raw_chan.zarr")
    zarr.save_array(path_chan, volc, chunks=(1, Z, Y, X))
    mask2 = load_density_mask(path_chan, volume_id=7, bounds=(0, Z, 0, Y, 0, X))
    assert torch.allclose(mask2, mask)


def test_load_eigenvector_field(tmp_path):
    Z, Y, X = 3, 2, 2
    # create eigenvectors dataset with shape (9,Z,Y,X)
    ev = np.zeros((9, Z, Y, X), dtype=np.float32)
    # fill v0 (channels 0..2) with [1,0,0], v1 with [0,1,0]
    ev[0] = 1.0  # v0.x
    ev[4] = 1.0  # v1.y  (since layout is [v0x,v0y,v0z, v1x,v1y,v1z, v2x,v2y,v2z])
    root = zarr.open_group(os.path.join(tmp_path, "ev.zarr"), mode="w")
    root.create_dataset("eigenvectors", data=ev, chunks=(1, Z, Y, X), dtype="f4")

    # principal (index 0)
    block0 = load_eigenvector_field(str(root.store.path), eig_index=0, bounds=(0, Z, 0, Y, 0, X))
    assert block0.shape == (3, Z, Y, X)
    # x-component is 1, others 0
    assert torch.allclose(block0[0], torch.ones_like(block0[0]))
    assert torch.allclose(block0[1], torch.zeros_like(block0[1]))
    assert torch.allclose(block0[2], torch.zeros_like(block0[2]))

    # secondary (index 1)
    block1 = load_eigenvector_field(str(root.store.path), eig_index=1, bounds=(0, Z, 0, Y, 0, X))
    assert torch.allclose(block1[1], torch.ones_like(block1[1]))  # y-component is 1
