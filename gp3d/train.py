import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from pytorch_lightning.loggers import WandbLogger

from torch.utils.data import DataLoader, Dataset
import zarr
import cv2

import torch
import skimage
import glob

import numpy as np
import random

from augment import apply_all_augmentations
from model import InkDetectionModel
from config import *

torch._dynamo.config.recompile_limit = 32


def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('medium')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True


class ZarrDataset(Dataset):
    def __init__(self, fragment_ids, mode):
        self.fragment_ids = fragment_ids
        self.mode = mode

        self.zarr_cache = {}
        self.ink_mask_cache_3d = {}

        self.chunks = []
        for frag_id in fragment_ids:
            self._build_chunks_for_fragment(frag_id)
        print(f"Loaded {len(self.chunks)} chunks for {mode} dataset")

        if mode == 'train' and AUGMENT:
            self.use_augmentation = True
        else:
            self.use_augmentation = False

    def _get_zarr_array(self, frag_id):
        if frag_id not in self.zarr_cache:
            zarr_path = os.path.join(ZARRS_PATH, f"{frag_id}.zarr")
            if not os.path.exists(zarr_path):
                print(f"Warning: Zarr file not found for {frag_id}: {zarr_path}")
                return None
            self.zarr_cache[frag_id] = zarr.open_array(zarr_path, mode='r')
        return self.zarr_cache[frag_id]

    def _load_and_cache_ink_mask(self, frag_id):
        if frag_id in self.ink_mask_cache_3d:
            return True

        zarr_array = self._get_zarr_array(frag_id)
        if zarr_array is None:
            return False
        volume_shape = zarr_array.shape  # (d, h, w)

        ink_zarr_path = f"{INK_LABELS_PATH}/{frag_id}.zarr"
        if not os.path.exists(ink_zarr_path):
            print(f"No 3D ink mask zarr found for {frag_id}")
            return False

        ink_zarr = zarr.open_array(ink_zarr_path, mode='r')
        ink_shape = ink_zarr.shape

        if len(ink_shape) != 3:
            raise ValueError(f"Expected 3D ink mask, got shape {ink_shape} for fragment {frag_id}")

        if ink_shape != volume_shape:
            raise ValueError(
                f"3D ink mask shape {ink_shape} doesn't match volume shape {volume_shape} for fragment {frag_id}")

        self.ink_mask_cache_3d[frag_id] = ink_zarr
        print(f"Loaded 3D ink mask from zarr for {frag_id}, shape: {ink_shape}")
        return True


    def _get_2d_ink_mask_for_building(self, frag_id):
        if not self._load_and_cache_ink_mask(frag_id):
            return None

        ink_zarr = self.ink_mask_cache_3d[frag_id]
        ink_mask_3d = np.array(ink_zarr[:])
        ink_mask_2d = np.mean(ink_mask_3d, axis=0)
        ink_mask_2d = (ink_mask_2d > 0).astype(np.uint8) * 255
        return ink_mask_2d

    def _build_chunks_for_fragment(self, frag_id):
        zarr_array = self._get_zarr_array(frag_id)
        if zarr_array is None:
            print(f"Skipping fragment {frag_id} - no zarr array found")
            return

        d, h, w = zarr_array.shape

        ink_mask_2d = self._get_2d_ink_mask_for_building(frag_id)
        if ink_mask_2d is None:
            print(f"No ink mask found for {frag_id}, skipping")
            return

        frag_mask = cv2.imread(f"{FRAGMENT_MASKS_PATH}/{frag_id}/{frag_id}_mask.png", 0)
        if frag_mask is None:
            print(f"No fragment mask found for {frag_id}, using full image as valid")
            frag_mask = np.ones((h, w), dtype=np.uint8) * 255  # All pixels valid
        else:
            frag_mask = cv2.resize(frag_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        counter = 0
        for y in range(0, h - CHUNK_SIZE, STRIDE):
            for x in range(0, w - CHUNK_SIZE, STRIDE):
                chunk_frag_mask = frag_mask[y:y + CHUNK_SIZE, x:x + CHUNK_SIZE]
                if np.all(chunk_frag_mask == 0):
                    continue
                counter += 1
                chunk_ink_2d = ink_mask_2d[y:y + CHUNK_SIZE, x:x + CHUNK_SIZE]
                has_ink = np.mean(chunk_ink_2d) > INKDETECT_MEAN
                if has_ink or self.mode == 'valid' or counter % 100 == 0:
                    self.chunks.append([frag_id, x, y])

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        frag_id, x, y = self.chunks[idx]

        zarr_array = self._get_zarr_array(frag_id)

        if (self.mode == 'train' and
                STRIDE < x < zarr_array.shape[2] - STRIDE and
                STRIDE < y < zarr_array.shape[1] - STRIDE):
            yoff, xoff = random.randint(-STRIDE, STRIDE), random.randint(-STRIDE, STRIDE)
            ystart = yoff + y
            xstart = xoff + x
        else:
            ystart = y
            xstart = x

        chunk_3d = zarr_array[:, ystart:ystart + CHUNK_SIZE, xstart:xstart + CHUNK_SIZE]

        ink_zarr = self.ink_mask_cache_3d[frag_id]
        ink_mask_3d = ink_zarr[:, ystart:ystart + CHUNK_SIZE, xstart:xstart + CHUNK_SIZE]

        if self.use_augmentation:
            chunk_3d, ink_mask_3d = apply_all_augmentations(chunk_3d, ink_mask_3d)

        chunk_3d = chunk_3d.astype(np.float32) / 255.0
        ink_mask_3d = ink_mask_3d.astype(np.float32) / 255.0
        chunk_3d = np.clip(chunk_3d, 0, 1)
        ink_mask_3d = np.clip(ink_mask_3d, 0, 1)

        chunk_tensor = torch.from_numpy(chunk_3d).float()
        mask_tensor = torch.from_numpy(ink_mask_3d).float()

        if self.mode == 'valid':
            return chunk_tensor, mask_tensor, (xstart, ystart, xstart + CHUNK_SIZE, ystart + CHUNK_SIZE)
        return chunk_tensor, mask_tensor


def get_available_fragments():
    if not os.path.exists(ZARRS_PATH):
        raise

    zarr_files = glob.glob(os.path.join(ZARRS_PATH, "*.zarr"))
    fragment_ids = []

    for zarr_file in zarr_files:
        basename = os.path.basename(zarr_file)
        if basename.endswith('.zarr'):
            frag_id = basename[:-5]
            fragment_ids.append(frag_id)

    return fragment_ids


def main():
    set_seed(SEED)
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    checkpoint_path = None
    checkpoints = glob.glob(os.path.join(OUTPUT_PATH, f'resnet3d{RESNET_DEPTH}_*.ckpt'))
    if checkpoints:
        checkpoint_path = max(checkpoints, key=lambda x: int(x.split('epoch=')[1].split('.')[0]))
        print(f"Resuming from checkpoint: {checkpoint_path}")
    else:
        print("Starting fresh training - no checkpoint found")

    all_fragments = get_available_fragments()

    if not all_fragments:
        print(f"No zarr files found in {ZARRS_PATH}")
        return

    random.shuffle(all_fragments)
    n_valid = int(len(all_fragments) * VALIDATION_SPLIT)
    valid_fragments = []
    train_fragments = all_fragments[n_valid:]

    if '20231005123336' in train_fragments:
        train_fragments.remove('20231005123336')
        # valid_fragments.append('20231005123336')

    print(f"Total fragments: {len(all_fragments)}")
    print(f"Train fragments: {len(train_fragments)}")
    print(f"Valid fragments: {len(valid_fragments)}")

    train_dataset = ZarrDataset(train_fragments, mode='train')
    valid_dataset = ZarrDataset(valid_fragments, mode='valid')

    print(f"Train chunks: {len(train_dataset)}")
    print(f"Valid chunks: {len(valid_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
        prefetch_factor=2
    )

    model = InkDetectionModel()
    model = torch.compile(model)

    print(f"Using Volumetric ResNet{RESNET_DEPTH} model for 3D ink detection")
    print(f"Input: {CHUNK_SIZE}³ voxel chunks (float32, normalized 0-1)")
    print(f"Output: {OUTPUT_SIZE}³ predictions (logits for ink probability)")
    print(f"All spatial dimensions (Z, Y, X) are processed equally")
    print(f"Batch size per GPU: {BATCH_SIZE} (effective batch size scales with number of GPUs)")

    callbacks = [
        ModelCheckpoint(
            filename=f'resnet3d{RESNET_DEPTH}_{{epoch}}',
            dirpath=OUTPUT_PATH,
            save_top_k=-1
        ),
    ]

    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        accelerator="gpu",
        devices="auto",
        precision="transformer-engine",
        gradient_clip_val=1.0,
        accumulate_grad_batches=GRADIENT_ACCUMULATE,
        # logger=WandbLogger(project="vesuvius", name=f"volumetric_resnet{RESNET_DEPTH}_ink_detection"),
        default_root_dir=OUTPUT_PATH,
        callbacks=callbacks
    )
    trainer.fit(model, train_loader, ckpt_path=checkpoint_path)


if __name__ == "__main__":
    main()