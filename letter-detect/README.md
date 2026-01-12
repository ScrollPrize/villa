# Letter Detect (ALPUB_v2 + RF-DETR)

This folder contains scripts and configs to prepare the AL-PUB_v2 dataset and fine-tune RF-DETR for
single-character detection (one character per image) on ancient Greek papyri.

Important: the dataset, processed data, logs, checkpoints, and weights are intentionally NOT tracked
in git. See the data download section below and the local paths in .gitignore.

## Data source (required)

Dataset: https://www.kaggle.com/datasets/miswindall/al-pub-v2

Download via Kaggle CLI (recommended):

```bash
kaggle datasets download -d miswindall/al-pub-v2
unzip al-pub-v2.zip -d dataset
```

Expected layout after unzip:

```
dataset/ALPUB_v2/
  images/
    Alpha/...
    Beta/...
    ...
```

Notes:
- You must accept the Kaggle dataset terms on first use.
- All crops in AL-PUB_v2 are 70x70 pixels.

## Quick start (human)

1) Install dependencies for data prep:

```bash
python -m pip install -r requirements.txt
```

2) Prepare COCO annotations and export RF-DETR dataset folders:

```bash
python scripts/prepare_alpub_v2_coco.py \
  --images-root dataset/ALPUB_v2/images \
  --output-dir data/alpub_v2_coco \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --seed 1337 \
  --write-splits-csv \
  --export-rfdetr-dir data/alpub_v2_rfdetr \
  --link-mode symlink \
  --rfdetr-zero-index
```

Tips:
- If you want to skip reading every image, add `--fixed-size 70 70`.
- If symlinks are not supported, use `--link-mode copy`.

3) Install RF-DETR locally (editable):

```bash
python -m pip install -e vendor/rf-detr
```

4) Train or fine-tune:

```bash
python scripts/train_rfdetr_alpub_v2.py \
  --dataset-dir data/alpub_v2_rfdetr \
  --output-dir runs/rfdetr_alpub_v2 \
  --model-size nano
```

RF-DETR will download COCO-pretrained weights the first time you run training (network access required).
If you already have a stronger checkpoint (example: `rf-detr-base-2.pth`), place it in this folder and use
`--pretrain-weights rf-detr-base-2.pth`.

5) Visualize predictions:

```bash
python scripts/visualize_rfdetr_predictions.py \
  --checkpoint runs/rfdetr_alpub_v2/checkpoint_best_total.pth \
  --dataset-dir data/alpub_v2_rfdetr \
  --split test \
  --model-size nano \
  --num-samples 16 \
  --threshold 0.3 \
  --scale 4 \
  --font-size 18 \
  --output-dir outputs/rfdetr_predictions
```

6) Run a checkpoint on a single image:

```bash
python scripts/predict_rfdetr_image.py \
  --checkpoint runs/rfdetr_alpub_v2/checkpoint_best_total.pth \
  --image path/to/image.png \
  --dataset-dir data/alpub_v2_rfdetr \
  --model-size nano \
  --threshold 0.3 \
  --output outputs/example_pred.png
```

## W&B logging example

```bash
python scripts/train_rfdetr_alpub_v2.py \
  --dataset-dir data/alpub_v2_rfdetr \
  --output-dir runs/rfdetr_alpub_v2_base2 \
  --model-size base \
  --pretrain-weights rf-detr-base-2.pth \
  --batch-size 16 \
  --grad-accum-steps 1 \
  --num-workers 8 \
  --checkpoint-interval 1 \
  --checkpoint-steps 2000 \
  --print-freq 1 \
  --wandb-log-steps 200 \
  --lr-drop 4 \
  --wandb \
  --wandb-entity vesuvius-challenge \
  --wandb-project character-detection \
  --wandb-run base2-b16 \
  --wandb-log-images \
  --wandb-image-count 12 \
  --wandb-image-every 5 \
  --wandb-image-steps 1000
```

## Repo layout

- `configs/alpub_v2_coco.yaml`: reference label mapping for ALPUB_v2.
- `scripts/prepare_alpub_v2_coco.py`: builds COCO annotations and RF-DETR split folders.
- `scripts/train_rfdetr_alpub_v2.py`: training wrapper with logging/checkpoint controls.
- `scripts/visualize_rfdetr_predictions.py`: batch visualization on a dataset split.
- `scripts/predict_rfdetr_image.py`: single-image inference and overlay.
- `vendor/rf-detr/`: RF-DETR source (editable install).

## Modeling notes

- This is a single-object-per-image detection setup. Full-text detection requires new supervision or
  synthetic composites.
- Class imbalance is significant (e.g., Xi and Zeta are much smaller). Consider reweighting or sampling.
- Augmentations that help: mild rotation (+/- 10 deg), contrast/gamma jitter, slight blur, erode/dilate.

## Bootstrap notes for future agents

- Data is expected at `dataset/ALPUB_v2/images`. It is not committed.
- Processed data goes to `data/alpub_v2_coco` and `data/alpub_v2_rfdetr`. These are not committed.
- Training outputs go to `runs/` and visualizations to `outputs/`. These are not committed.
- The RF-DETR code is vendored in `vendor/rf-detr` and installed with `pip install -e`.
- Local changes in this folder include RF-DETR class-count handling for ALPUB_v2 and extra logging
  options in the training/visualization scripts.
- If you add new data or weights, update `.gitignore` and this README accordingly.
