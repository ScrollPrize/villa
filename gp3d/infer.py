import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import zarr
import cv2
from tqdm import tqdm
import skimage

from model import InkDetectionModel
from config import *


def setup_ddp(rank, world_size):
    """Initialize DDP"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Clean up DDP"""
    dist.destroy_process_group()


class InferenceDataset(Dataset):
    """Lazy-loading dataset for inference chunks"""

    def __init__(self, zarr_array, fragment_id, xyxys):
        self.zarr_array = zarr_array
        self.fragment_id = fragment_id
        self.xyxys = xyxys

    def __len__(self):
        return len(self.xyxys)

    def __getitem__(self, idx):
        x1, y1, x2, y2 = self.xyxys[idx]
        chunk_3d = self.zarr_array[:, y1:y2, x1:x2]
        chunk_3d = np.array(chunk_3d).astype(np.float32)
        chunk_3d = np.nan_to_num(chunk_3d)
        chunk_3d = (chunk_3d / 256)
        chunk_tensor = torch.from_numpy(chunk_3d).float()
        return chunk_tensor, torch.tensor([x1, y1, x2, y2])


def get_valid_chunk_coords(fragment_id):
    """Get coordinates of valid chunks without loading data"""
    # Open individual fragment zarr
    zarr_path = os.path.join(ZARRS_PATH, f"{fragment_id}.zarr")
    if not os.path.exists(zarr_path):
        print(f"Zarr file not found for {fragment_id}: {zarr_path}")
        return None, None, None

    zarr_array = zarr.open_array(zarr_path, mode='r')
    d, h, w = zarr_array.shape

    # Load fragment mask
    frag_mask_path = os.path.join(FRAGMENT_MASKS_PATH, fragment_id, f"{fragment_id}_mask.png")
    if not os.path.exists(frag_mask_path):
        print(f"No fragment mask found for {fragment_id}, using full image")
        frag_mask = np.ones((h, w), dtype=np.uint8) * 255
    else:
        frag_mask = cv2.imread(frag_mask_path, 0)
        frag_mask = cv2.resize(frag_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # Generate valid chunk coordinates
    xyxys = []
    for y in range(0, h - CHUNK_SIZE + 1, STRIDE):
        for x in range(0, w - CHUNK_SIZE + 1, STRIDE):
            chunk_mask = frag_mask[y:y + CHUNK_SIZE, x:x + CHUNK_SIZE]
            # Include chunk if it has any valid pixels (not just all valid)
            if np.any(chunk_mask > 0):
                xyxys.append([x, y, x + CHUNK_SIZE, y + CHUNK_SIZE])

    return zarr_array, xyxys, (h, w)


@torch.no_grad()
def run_inference_distributed(model, dataloader, output_shape, device, rank, world_size):
    """Run distributed inference with overlapping region averaging"""
    scale_factor = CHUNK_SIZE // OUTPUT_SIZE
    h_out = output_shape[0] // scale_factor
    w_out = output_shape[1] // scale_factor

    # Local predictions for this GPU
    pred_sum = np.zeros((h_out, w_out), dtype=np.float32)
    pred_count = np.zeros((h_out, w_out), dtype=np.float32)

    model.eval()

    # Progress bar only on rank 0
    iterator = tqdm(dataloader, desc=f"GPU {rank} inference") if rank == 0 else dataloader

    for chunks, xyxys in iterator:
        chunks = chunks.to(device)
        outputs = model(chunks)
        probs = torch.sigmoid(outputs)
        # Average across depth dimension to get 2D prediction
        probs_2d = probs.mean(dim=2)
        probs_2d = probs_2d.cpu().numpy()

        for i in range(len(xyxys)):
            x1, y1, x2, y2 = xyxys[i].tolist()
            x1_out = x1 // scale_factor
            y1_out = y1 // scale_factor
            x2_out = x2 // scale_factor
            y2_out = y2 // scale_factor
            pred_sum[y1_out:y2_out, x1_out:x2_out] += probs_2d[i, 0]
            pred_count[y1_out:y2_out, x1_out:x2_out] += 1

    # Gather results on rank 0
    pred_sum_tensor = torch.from_numpy(pred_sum).cuda()
    pred_count_tensor = torch.from_numpy(pred_count).cuda()

    if rank == 0:
        gathered_sums = [torch.zeros_like(pred_sum_tensor) for _ in range(world_size)]
        gathered_counts = [torch.zeros_like(pred_count_tensor) for _ in range(world_size)]
    else:
        gathered_sums = None
        gathered_counts = None

    dist.gather(pred_sum_tensor, gathered_sums, dst=0)
    dist.gather(pred_count_tensor, gathered_counts, dst=0)

    if rank == 0:
        total_sum = sum(t.cpu().numpy() for t in gathered_sums)
        total_count = sum(t.cpu().numpy() for t in gathered_counts)
        mask_pred = np.divide(total_sum, total_count,
                              out=np.zeros_like(total_sum),
                              where=total_count > 0)

        # Print overlap statistics
        overlap_pixels = np.sum(total_count > 1)
        total_pixels = np.sum(total_count > 0)
        if total_pixels > 0:
            overlap_percentage = (overlap_pixels / total_pixels) * 100
            print(f"\nOverlap statistics:")
            print(f"  Total pixels: {total_pixels:,}")
            print(f"  Overlapping pixels: {overlap_pixels:,} ({overlap_percentage:.1f}%)")
            print(f"  Max overlap count: {int(total_count.max())}")
            print(f"  Mean overlap count: {total_count[total_count > 0].mean():.2f}")

        return mask_pred

    return None


def inference_worker(rank, world_size, checkpoint_path, fragment_id):
    """Worker function for distributed inference"""
    setup_ddp(rank, world_size)

    try:
        device = torch.device(f'cuda:{rank}')

        # Load model
        model = InkDetectionModel.load_from_checkpoint(checkpoint_path, strict=False)
        model.to(device)
        model = DDP(model, device_ids=[rank])
        model.eval()

        # Get chunk coordinates on rank 0
        if rank == 0:
            print(f"Processing fragment {fragment_id}...")
            print(f"Chunk size: {CHUNK_SIZE}, Stride: {STRIDE}")
            print(f"Overlap: {CHUNK_SIZE - STRIDE} pixels per dimension")
            zarr_array, xyxys, output_shape = get_valid_chunk_coords(fragment_id)
            if zarr_array is None:
                print(f"Failed to load fragment {fragment_id}")
                return
            print(f"Fragment shape: {zarr_array.shape}")
            print(f"Found {len(xyxys)} valid chunks")
        else:
            zarr_array, xyxys, output_shape = None, None, None

        # Broadcast data to all ranks
        if rank == 0:
            broadcast_data = (xyxys, output_shape)
        else:
            broadcast_data = None

        broadcast_list = [broadcast_data]
        dist.broadcast_object_list(broadcast_list, src=0)
        xyxys, output_shape = broadcast_list[0]

        # Each rank opens its own zarr array
        if zarr_array is None:
            zarr_path = os.path.join(ZARRS_PATH, f"{fragment_id}.zarr")
            zarr_array = zarr.open_array(zarr_path, mode='r')

        # Create dataset and dataloader
        dataset = InferenceDataset(zarr_array, fragment_id, xyxys)
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)

        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            sampler=sampler,
            num_workers=NUM_WORKERS // world_size,
            pin_memory=True,
            prefetch_factor=2
        )

        # Run inference
        mask_pred = run_inference_distributed(model, dataloader, output_shape, device, rank, world_size)

        # Save results on rank 0
        if rank == 0 and mask_pred is not None:
            os.makedirs(OUTPUT_PATH, exist_ok=True)
            output_path = os.path.join(OUTPUT_PATH, f"{fragment_id}_ink_prediction.png")
            mask_pred_uint8 = (np.clip(mask_pred, 0, 1) * 255).astype(np.uint8)
            cv2.imwrite(output_path, mask_pred_uint8)
            print(f"Saved prediction to {output_path}")

            # Also save as zarr for easier processing
            output_zarr_path = os.path.join(OUTPUT_PATH, f"{fragment_id}_ink_prediction.zarr")
            zarr_pred = zarr.open_array(
                output_zarr_path,
                mode='w',
                shape=mask_pred.shape,
                chunks=(256, 256),
                dtype='float32'
            )
            zarr_pred[:] = mask_pred
            print(f"Saved zarr prediction to {output_zarr_path}")

    finally:
        cleanup_ddp()


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run ink detection inference on a fragment')
    parser.add_argument('checkpoint_path', type=str,
                        help='Path to the model checkpoint (.ckpt file)')
    parser.add_argument('fragment_id', type=str,
                        help='Fragment ID to process (e.g., 20231005123336)')
    parser.add_argument('--gpus', type=int, default=None,
                        help='Number of GPUs to use (default: all available)')

    args = parser.parse_args()

    # Determine number of GPUs to use
    available_gpus = torch.cuda.device_count()
    if available_gpus == 0:
        print("No CUDA devices found!")
        return

    if args.gpus is None:
        world_size = available_gpus
    else:
        world_size = min(args.gpus, available_gpus)

    print(f"Found {available_gpus} CUDA devices, using {world_size}")

    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint_path):
        print(f"Checkpoint not found: {args.checkpoint_path}")
        return

    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Fragment ID: {args.fragment_id}")

    # Run distributed inference
    mp.spawn(
        inference_worker,
        args=(world_size, args.checkpoint_path, args.fragment_id),
        nprocs=world_size,
        join=True
    )


if __name__ == "__main__":
    main()