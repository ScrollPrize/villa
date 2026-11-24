# Denoising

A 3D U-Net can remove noise from reconstructed slices. It runs after the reconstruction, on the float slices, before
they are converted to uint8. The back projection itself does not change.

The network is trained without clean data. It learns from pairs of reconstructions of the same slices whose noise is
independent (Hendriksen, Pelt & Batenburg, "Noise2Inverse", 2020). A network only works on the scan it was trained
on, so each scan needs its own.

The scripts are in `networks/denoising/` and are run as files, not as part of the `vc_tomography` package. The
trained weights are not in the repository.

| Script | What it does |
|---|---|
| `reconstruct_block.py` | Reconstructs a block of slices to `.npz`: training pairs, and blocks to try a network on |
| `noise2inverse.py` | The network: `train`, `apply` (to a block), `slices` (pictures of a block) |
| `masked_volume.py` | Reconstructs, denoises and writes a whole scan as a uint8 OME-Zarr under a mask |

## Setup

The network needs PyTorch, which is an optional extra of the package:

```bash
uv sync --python 3.12 --extra test --extra denoise
```

Every script runs in this one environment.

## 1. Make training pairs

A pair is two reconstructions of the same slices with independent noise. There are two ways to make one.

- **Noise splitting** (used for the trained networks). Two runs of `noise_split` on the same projections, with the
  scales `a` and `-1 / a` (see `noise_split` in [Reconstructing a scan](reconstruction.md)). Every projection is
  used. It needs the detector's noise model (`gain`, `floor`, and `smooth` and `flats` where measured), taken from
  the scan's own flats.
- **Even and odd projections** (Noise2Inverse as published). `reconstruct_block.py` writes `even` and `odd` by
  default, each using every second angle. On a scan with too few angles the network learns to remove the artefacts
  this leaves, and loses detail with them (seen on PHerc. 172).

```bash
# one slab of a noise-split pair (48 slices of 2560 x 2560 px), scale 0.5; repeat with scale=-2 for the other half
python networks/denoising/reconstruct_block.py SCANS --cache CACHE --geometry geometry.json \
    --filters "paganin(delta_beta=1000)+unsharp(sigma=1.2,coeff=4)" --set sharpness_weighting=auto \
    --heights -4000 -3952 --roi 1200 3760 1200 3760 --subsets full=0/1 \
    --set "noise_split=scale=0.5,seed=1,gain=1.55,floor=0,smooth=0.0407:3.06+0.1151:1.38,flats=401" \
    --out slab_h-4000_x1.npz
```

Use the same recipe (filters, geometry, weighting) that the final volume will use. Take slabs at several heights and
places, and keep one pair out of training to rate the network on.

## 2. Train

```bash
python networks/denoising/noise2inverse.py train --slabs "a_x1.npz>a_x2.npz" "b_x1.npz>b_x2.npz" \
    --out MODEL_DIR --width 48 --scales 4 --host --lr 2e-4 --amp bf16 --minutes 60 \
    [--init OTHER/model.pt] [--watch "t_x1.npz>t_x2.npz"]
```

- `A>B`: A is always the input. For a noise-split pair, A is the run with scale 0.5, whose noise is closest to the
  data's. `A+B`: either can be the input (two independent recordings). A single file: its `even` and `odd`.
- The default network is 3 scales of 32 channels (1.4 million parameters) trained for 10 minutes. The networks in
  use are 4 scales of 48 (13.1 million). On PHerc. 172 both rated the same.
- Training runs for a fixed number of minutes. The learning rate warms up, then falls along a cosine. Seeds are
  fixed.
- `--init` starts from another network's weights, which is faster than starting from scratch.
- `MODEL_DIR/model.pt` holds the weights, the network's size and its precision.

## 3. Try it on a block

```bash
python networks/denoising/noise2inverse.py apply --model MODEL_DIR/model.pt --block block.npz --out denoised.npz
python networks/denoising/noise2inverse.py slices --model MODEL_DIR/model.pt --block block.npz \
    --out PICTURES --name denoised --slices 16 32 --window -0.06 0.29
```

## 4. Denoise a whole volume

```bash
python networks/denoising/masked_volume.py SCANS --cache CACHE --geometry geometry.json \
    --filters "paganin(delta_beta=1000)+unsharp(sigma=1.2,coeff=4)" \
    --set sharpness_weighting=auto --set luminosity_repartition=100 \
    --window -0.06 0.29 --voxel_um 8.64 --mask mask_level2.npy \
    --model MODEL_DIR/model.pt --zarr denoised.zarr --zarr_not_denoised not_denoised.zarr
```

- The volume covers every height the scan can reconstruct. It is written as uint8 within `--window` (attenuation
  per cm), in chunks of 128, with 6 levels and `vc-delta3d` compression.
- `--mask` is a `.npy` (z, y, x) at a quarter of the volume's resolution. Voxels where it is 0 are left out and
  stored as nothing; only the mask's bounding box is reconstructed in each segment.
- `--zarr_not_denoised` also writes the same slices without the network, for comparison.
- `--slices J0 J1` makes only part of the volume.
- It all runs in one process, one segment of 480 slices at a time. A segment is reconstructed into memory, then its
  reconstructor is destroyed (giving the GPU back), then the network denoises it. The two never use the GPU at the
  same time. The host holds one segment's float slices: (segment + 16) x the mask's box x 4 bytes, at most about
  50 GB for PHerc0343P's full grid; `--segment` makes it smaller.
- If a reconstruction runs out of GPU memory, it starts again from that slice with half as many slices per call.
- A run that stops starts over; `--slices` makes the rest as a separate part.
- The network sees slabs of 64 slices and keeps the middle 48, in tiles of 256 px with a 48 px overlap.

## Trained networks

None are in the repository (a `model.pt` is about 52 MB). Train one per scan: a network trained on PHerc. 172
rated 0.90 on PHerc0343P (1: the input unchanged, 0: perfect), against 0.30 for one trained on PHerc0343P itself.

## Tests

```bash
uv run pytest tests/test_denoising_network.py
```

These check that tiling matches running the network on the whole volume, that a slab's uint8 output matches,
that a run takes each slice from the right slab, and that a saved network loads again. They are skipped when PyTorch is not installed.
