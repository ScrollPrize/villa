# Vesuvius Grandprize Winning Solution
![Vesuvius Challenge GP Solution](pictures/logo.png)

The Repository contains the First Place Vesuvius Grand Prize solution. 
This repository is part of the First Place Grand Prize Submission to the Vesuvius Challenge 2023 from Youssef Nader, Luke Farritor and Julian Schilliger.

<!-- <img align="center" width="60" height="60" src="pictures/ThaumatoAnakalyptor.png">  -->
## Automatic Segmentation <img align="center" width="60" height="60" src="pictures/ThaumatoAnakalyptor.png"> 
Check out the automatic segmentation pipeline ThaumatoAnakalyptor of our winning Grand Prize submission by Julian Schilliger. 
[ThaumatoAnakalyptor](https://github.com/schillij95/ThaumatoAnakalyptor/tree/main) performs in full 3D and is also capable of segmenting in very mushy and twisted scroll regions.

# Ink Detection Overview<img align="center" width="60" height="60" src="pictures/logo.png"> :
Our final canonical model was a timesformer small architecture with divided space-time attention. 
The dataset underwent expansion and cleaning rounds to increase accuracy of the labels and become as accurate as possible, approximately 15 rounds were performed between the first letters and final solution. 
Our solution also consisted of 2 other architectures, Resnet3D-101 with pretrained weights, I3D with non-local block and maxpooling. 

Our implementation uses `torch`, `torch-lightning`,the [`timesformer-pytorch`](https://github.com/lucidrains/TimeSformer-pytorch) and [`3D-ResNets-PyTorch`](https://github.com/kenshohara/3D-ResNets-PyTorch/blob/master/models/resnet.py). 


# 🚀 Get Started

EASY: build the docker image: 

```bash
# Local/manual use (no agent flags required)
docker build -t youssef_gp .

# Agent mode with explicit opt-in
AGENTS_AGENT_MODE=1 AGENTS_ALLOW_INSTALL=1 docker build -t youssef_gp .
docker run --gpus all --shm-size=150g -it -v </your-path-to-train-scrolls>:/workspace/train_scrolls youssef_gp
```

Then to train:

```bash
python train_timesformer_og.py
```

Or to run inference with the already trained model:

```bash
python inference_timesformer.py --model_path timesformer_weights.ckpt --segment_path train_scrolls --segment_id 20231005123336
```

Important note: to install the ink labels and training data inside the docker image, run:

```bash
#to download the segments from the server
./download.sh
#propagates the inklabels into the respective segment folders for training
python prepare.py
```
You can find the weights of the canonical timesformer uploaded [here](https://drive.google.com/drive/folders/1rn3GMOvtJRMBHOxVhWFVSY6IVI6xUnYp?usp=sharing)
# Example Inference

To run inference of timesformer:

```bash
python inference_timesformer.py --segment_id 20231210121321 20231221180251 --segment_path $(pwd)/train_scrolls --model_path timesformer_weights.ckpt
```

The optional parameter ```--out_path``` can be used to specify the output path of the predictions.

## ResNet3D training (`train_resnet3d.py`)

Zarr setup

- `dataset_root/<segment_id>.zarr`
- Train files:
  - `dataset_root/<segment_id>/<segment_id>_inklabels.png` (or `.tif`/`.tiff`)
  - `dataset_root/<segment_id>/<segment_id>_mask.png` (or `.tif`/`.tiff`)
- Val files:
  - `dataset_root/<segment_id>/<segment_id>_inklabels_val.png` (or `.tif`/`.tiff`)
  - `dataset_root/<segment_id>/<segment_id>_mask_val.png` (or `.tif`/`.tiff`)
- Set `training.data_backend: zarr` and `training.dataset_root` (default: `train_scrolls`).

Tiff setup

- `dataset_root/<segment_id>/layers/` with layer files (example: `00.tif`, `01.tif`, ...)
- Train files:
  - `dataset_root/<segment_id>/<segment_id>_inklabels.png` (or `.tif`/`.tiff`)
  - `dataset_root/<segment_id>/<segment_id>_mask.png` (or `.tif`/`.tiff`)
- Val files:
  - `dataset_root/<segment_id>/<segment_id>_inklabels_val.png` (or `.tif`/`.tiff`)
  - `dataset_root/<segment_id>/<segment_id>_mask_val.png` (or `.tif`/`.tiff`)
- Set `training.data_backend: tiff` and `training.dataset_root` (default: `train_scrolls`).

Segment metadata (required per segment):

- `layer_range: [start_idx, end_idx]` (end is exclusive; must include at least `in_chans` layers; if larger, centered to `in_chans`).
- `reverse_layers: true|false`.

Metadata keys to set (edit the existing `metadata.json` template):

- `segments.<segment_id>.base_path`, `segments.<segment_id>.layer_range`, `segments.<segment_id>.reverse_layers`.
- `training.train_segments`, `training.val_segments`, optional `training.cv_fold`.
- `training.objective`, `training.sampler`, `training.loss_mode`, `training.save_every_epoch`, `training.stitching_schedule`.
- `training_hyperparameters.model.backbone_pretrained_path`.

Fold/suffix behavior:

- If `training.cv_fold` is set and suffixes are not explicitly set, defaults are:
  - train suffixes: `_{cv_fold}`
  - val suffixes: `_val_{cv_fold}`

Pretrained 3D-ResNet checkpoint:

- Download from `3D-ResNets-PyTorch` pretrained models and place it in this folder (or any path).
- Set `training_hyperparameters.model.backbone_pretrained_path`, e.g. `r3d50_KM_200ep.pth`, `r3d101_KM_200ep.pth`, or `r3d152_KM_200ep.pth`.

Run:

```bash
cd villa/ink-detection
python train_resnet3d.py --metadata_json metadata.json --outputs_path ./outputs
```

W&B sweep example

```bash
cd villa/ink-detection
wandb login
wandb sweep train_resnet3d_lib/sweeps/sweep_erm_hparams_shuffle_accum4.yaml
wandb agent <entity>/<project>/<sweep_id>
```

If needed, edit the sweep YAML `command` block (`--metadata_json`, `--outputs_path`) before creating the sweep.
