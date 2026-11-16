# Denoising vc_tomography's reconstructions (Noise2Inverse)

A 3D U-Net that takes the noise off reconstructed slices. It runs **after** the reconstruction, on the float
slices, before they are windowed to uint8; nothing in the back projection changes. It is trained without clean
data, on pairs of reconstructions of the same slices whose noise is independent (Hendriksen, Pelt & Batenburg,
"Noise2Inverse", 2020).

It is not part of the `vc_tomography` Python package: it needs PyTorch, which vc_tomography's environment does not have. The
scripts are run as files, each with the environment it names. The trained weights are **not in the repository**.

| File | Environment | What it does |
|---|---|---|
| `reconstruct_block.py` | vc_tomography | A block of slices as `.npz`: the training pairs, and blocks to try a network on |
| `noise2inverse.py` | PyTorch | The network; `train`, `apply` (a block), `slices` (pictures of a block) |
| `denoise_spool.py` | PyTorch | The network on a run of slices another process reconstructs, through a spool folder, to uint8 |
| `masked_volume.py` | vc_tomography | A whole scan: reconstructed, denoised (it starts `denoise_spool.py`) and written as a uint8 OME-Zarr under a mask |

In this repository the PyTorch environment is `vesuvius/.venv`.

## 1. Training pairs

A pair is two reconstructions of the same slices with independent noise. Two ways:

- **Noise splitting** (what the trained networks used): two runs of vc_tomography's `noise_split` on the same projections,
  with the scales `a` and `-1 / a` (README of vc_tomography, "Slices and volumes"). No projection is left out. It needs the
  detector's noise model (`gain`, `floor`, and the `smooth` fields and `flats` where measured), from the scan's own
  flats.
- **Even and odd projections** (Noise2Inverse as published): `reconstruct_block.py` writes `even` and `odd` by
  default. Each leaves out every second angle; on a scan with too few angles the network then learns to remove what
  that leaves in a slice, and loses detail with it (PHerc. 172).

```bash
# one slab of a noise-split pair (48 slices of 2560 x 2560 px), scale 0.5 and then -2
python reconstruct_block.py SCANS --cache CACHE --geometry geometry.json \
    --filters "paganin(delta_beta=1000)+unsharp(sigma=1.2,coeff=4)" --set sharpness_weighting=auto \
    --heights -4000 -3952 --roi 1200 3760 1200 3760 --subsets full=0/1 \
    --set "noise_split=scale=0.5,seed=1,gain=1.55,floor=0,smooth=0.0407:3.06+0.1151:1.38,flats=401" \
    --out slab_h-4000_x1.npz
# the same with scale=-2 -> slab_h-4000_x2.npz
```

Use the recipe (filters, geometry, weighting) the volume will be made with, and slabs at several heights and places.
Keep one pair out of the training to rate the network on.

## 2. Training

```bash
python noise2inverse.py train --slabs "a_x1.npz>a_x2.npz" "b_x1.npz>b_x2.npz" .. --out MODEL_DIR \
    --width 48 --scales 4 --host --lr 2e-4 --amp bf16 --minutes 60 [--init OTHER/model.pt] [--watch "t_x1.npz>t_x2.npz"]
```

- `A>B`: A is always the input (a noise-split pair: A is the run with the scale 0.5, whose noise is close to the
  data's). `A+B`: either is the input (two independent recordings). A single file: its `even` and `odd`.
- Defaults: 3 scales of 32 channels (1.41 million parameters), 10 minutes. The networks in use are 4 scales of 48
  (13.1 million). On PHerc. 172 the two rated the same.
- The training time is fixed in minutes; the learning rate warms up and falls as a cosine. Seeds are fixed.
- `MODEL_DIR/model.pt` holds the weights and the network's size and precision.

## 3. Trying it on a block

```bash
python noise2inverse.py apply --model MODEL_DIR/model.pt --block block.npz --out denoised.npz
python noise2inverse.py slices --model MODEL_DIR/model.pt --block block.npz --out PICTURES --name denoised \
    --slices 16 32 --window -0.06 0.29
```

## 4. A whole volume

```bash
python masked_volume.py SCANS --cache CACHE --geometry geometry.json \
    --filters "paganin(delta_beta=1000)+unsharp(sigma=1.2,coeff=4)" \
    --set sharpness_weighting=auto --set luminosity_repartition=100 \
    --window -0.06 0.29 --voxel_um 8.64 --mask mask_level2.npy \
    --model MODEL_DIR/model.pt --torch_python /path/to/vesuvius/.venv/bin/python \
    --zarr denoised.zarr --zarr_not_denoised not_denoised.zarr --spool SPOOL_DIR
```

- The volume is vc_tomography's slice grid over every height the scan can give: slice `j` is the first doable view
  height + `j`. uint8 in `--window` (per cm), chunks of 128, 6 levels, `vc-delta3d`.
- `--mask`: a `.npy` (z, y, x) at a quarter of the volume's resolution, not 0 where the volume is kept; outside it
  the volume is 0 and nothing is stored. The reconstruction is only made inside the mask's box of each segment.
- `--zarr_not_denoised` writes the same slices without the network, for comparison.
- The reconstruction and the network do not fit on one device together, so each segment of 480 slices is
  reconstructed by one process and denoised by another, through `--spool`. A run that stops is taken up from the
  spool's finished segments. `--slices J0 J1` makes a part.
- The network sees slabs of 64 slices and keeps the middle 48, in tiles of 256 px with a halo of 48.

A volume in another frame (PHerc. 172's is written in the published volume's, reversed along the axis) needs its own
reconstruction side; it shares the spool with `denoise_spool.py` (`save`, `spool_files`, `BLOCK`).

## Trained networks (not in the repository)

Each is one `model.pt` of about 52 MB, kept with the scan's reconstruction outputs.

| Name | Scan | Trained on | Rated error (1: the input as it is, 0: the sample) |
|---|---|---|---|
| `model_long_paganin175_60min` | PHerc. 172, Diamond I12, 7.91 µm | 23 slabs: noise-split pairs and the two rows of scans as pairs; 60 min | 0.13, against the other row of scans |
| `model_343p_60min` | PHerc0343P, ESRF BM18, 8.64 µm | 6 noise-split pairs, from the PHerc. 172 weights; 60 min | 0.30, on a held-out split pair |
| `model_343p_auto_finetune_30min` | the same, `sharpness_weighting=auto` | the network above, 30 min more | 0.30 |

A network does not carry over between scans: the PHerc. 172 network rates 0.90 on PHerc0343P.

## Checks

- `tests/test_denoising_network.py` (needs PyTorch; skipped in vc_tomography's environment): a tile's kept part against the
  whole volume, `denoise_spool`'s uint8 against `run_tiled`, a saved network loads again, the spool's bookkeeping.

  ```bash
  /path/to/vesuvius/.venv/bin/python -m pytest tests/test_denoising_network.py
  ```
