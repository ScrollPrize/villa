# Denoising vc_tomography's reconstructions (Noise2Inverse)

A 3D U-Net that removes noise from reconstructed slices, trained on pairs of reconstructions with independent
noise. It needs PyTorch: install vc_tomography with `uv sync --extra denoise`.

| File | What it does |
|---|---|
| `reconstruct_block.py` | A block of slices as `.npz`: the training pairs, and blocks to try a network on |
| `noise2inverse.py` | The network; `train`, `apply` (a block), `slices` (pictures of a block) |
| `masked_volume.py` | A whole scan: reconstructed, denoised and written as a uint8 OME-Zarr under a mask, in one process |

How to make training pairs, train, and denoise a volume, and the trained networks: [docs/denoising.md](../../docs/denoising.md).
