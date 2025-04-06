import os
import argparse
import numpy as np
from tqdm import tqdm
import zarr
from numcodecs import Blosc
import cv2
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from builders.build_network_from_config import NetworkFromConfig
from configuration.config_manager import ConfigManager
from dataloading.inference_dataset import InferenceDataset
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import threading
from typing import Dict, Tuple, List
from collections import defaultdict
import torch.distributed as dist

def remap_state_dict(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        # Example mapping: if the nnUNet checkpoint uses "encoder." and your model expects "shared_encoder."
        if key.startswith("encoder."):
            new_key = "shared_encoder." + key[len("encoder."):]
        # If the checkpoint uses "decoder." and your model expects a ModuleDict under "task_decoders" (say, for key "sheet")
        elif key.startswith("decoder."):
            new_key = "task_decoders.sheet." + key[len("decoder."):]
        # You might also need to check for differences in naming for stem or other submodules.
        new_state_dict[new_key] = value
    return new_state_dict


class ZarrInferenceHandler:
    def __init__(self, config_file: str, write_layers: bool, postprocess_only: bool = False,
                 num_write_workers: int = 4):
        self.mgr = ConfigManager(config_file)
        self.postprocess_only = postprocess_only
        self.write_layers = write_layers
        # Buffer to accumulate patches before writing
        self.patch_buffer = defaultdict(list)
        self.buffer_positions = []
        self.buffer_size = 32  # Number of patches to accumulate before writing

        # Initialize blend weights as None - will be created when needed
        self.blend_weights = None
        self.blend_weights_4d = None

        # Determine rank for DDP
        self.rank = 0
        if dist.is_initialized():
            self.rank = dist.get_rank()

        # Create a ThreadPoolExecutor for asynchronous writing (if not in postprocess-only mode)
        self.executor = None
        self.write_futures = []
        if not self.postprocess_only:
            self.executor = ThreadPoolExecutor(max_workers=num_write_workers)

        # --- NEW: Limit the number of pending writes so that memory does not grow unbounded ---
        self.max_pending_writes = num_write_workers * 4  # adjust as needed

        # --- NEW: A lock to protect the read–modify–write update used for blending ---
        self.write_lock = threading.Lock()

    def _build_model(self):
        model = NetworkFromConfig(self.mgr)
        return model

    def _create_blend_weights(self):
        """
        Create a 3D Gaussian window to be used as blending weights.

        If no overlap is desired (mgr.infer_overlap == 0), then simply return an array of ones.
        Otherwise, for each dimension a 1D Gaussian window is created and the full 3D weight is
        computed as the outer product over all dimensions.

        Adjust the sigma parameter as needed; here we use sigma = (patch_size / 4).
        """
        if self.mgr.infer_overlap == 0:
            return np.ones(self.mgr.infer_patch_size, dtype=np.float32)

        patch_size = self.mgr.infer_patch_size
        weights = np.ones(patch_size, dtype=np.float32)
        # Create a Gaussian window for each axis and multiply them.
        for axis, size in enumerate(patch_size):
            # Center coordinate along this axis
            center = (size - 1) / 2.0
            # Standard deviation: adjust this factor (here, patch_size/4) as needed for your data.
            sigma = size / 4.0
            x = np.arange(size, dtype=np.float32)
            gaussian_1d = np.exp(-0.5 * ((x - center) / sigma) ** 2)
            # Reshape so that it broadcasts along the proper axis.
            shape = [1] * len(patch_size)
            shape[axis] = size
            gaussian_1d = gaussian_1d.reshape(shape)
            weights *= gaussian_1d

        return weights

    def _initialize_blend_weights(self):
        """Initialize blend weights for the current patch size"""
        if len(self.mgr.infer_patch_size) == 3:  # 3D patches
            self.blend_weights = self._create_blend_weights()
            # Create 4D version for multi-channel data
            self.blend_weights_4d = np.expand_dims(self.blend_weights, axis=0)

    def _process_buffer(self, output_arrays: Dict, count_arrays: Dict,
                        patch_buffer: Dict = None, positions: List[Tuple] = None):
        """
        Write accumulated patches to the Zarr arrays, one patch at a time,
        instead of grouping patches into large bounding boxes.
        """
        # Use the provided buffers if any; otherwise, use the instance buffers.
        if patch_buffer is None or positions is None:
            patch_buffer = self.patch_buffer
            positions = self.buffer_positions

        if not patch_buffer or not positions:
            return

        # Loop over each patch and write it directly.
        for i, pos in enumerate(positions):
            for tgt_name in self.mgr.infer_output_targets:
                patch_data = patch_buffer[tgt_name][i]
                # We call _write_region_patches() with just a single patch in the list.
                self._write_region_patches([patch_data], [pos],
                                           output_arrays[tgt_name],
                                           count_arrays[tgt_name])

        # If these are the "live" buffers on the instance, clear them.
        if patch_buffer is self.patch_buffer:
            self.patch_buffer.clear()
            self.buffer_positions.clear()

    def _write_region_patches(self, patches: List[np.ndarray], positions: List[Tuple],
                              output_array: zarr.Array, count_array: zarr.Array):
        """Write multiple patches with zero–padding handled.

        For patches that have been zero–padded to reach the full patch size,
        we compute the valid region (i.e. the part that actually overlaps the image)
        and only blend that region back into the output arrays.
        """
        with self.write_lock:
            # Ensure blending weights are initialized.
            if self.blend_weights is None:
                self._initialize_blend_weights()

            # Get the full image shape from the count array.
            image_z, image_y, image_x = count_array.shape
            full_z, full_y, full_x = self.mgr.infer_patch_size

            # Compute valid sizes for each patch (i.e. how much of the patch actually falls inside the image).
            valid_sizes = []
            for pos in positions:
                z0, y0, x0 = pos
                valid_z = full_z if (z0 + full_z) <= image_z else image_z - z0
                valid_y = full_y if (y0 + full_y) <= image_y else image_y - y0
                valid_x = full_x if (x0 + full_x) <= image_x else image_x - x0
                valid_sizes.append((valid_z, valid_y, valid_x))

            # Compute the union region bounds for all patches.
            min_z = min(pos[0] for pos in positions)
            min_y = min(pos[1] for pos in positions)
            min_x = min(pos[2] for pos in positions)
            max_z = max(pos[0] + valid[0] for pos, valid in zip(positions, valid_sizes))
            max_y = max(pos[1] + valid[1] for pos, valid in zip(positions, valid_sizes))
            max_x = max(pos[2] + valid[2] for pos, valid in zip(positions, valid_sizes))

            # Depending on whether we have a channel dimension, choose the appropriate base weights.
            if len(patches[0].shape) == 4:
                region_sum = output_array[:, min_z:max_z, min_y:max_y, min_x:max_x]
                base_weights = self.blend_weights_4d
            else:
                region_sum = output_array[min_z:max_z, min_y:max_y, min_x:max_x]
                base_weights = self.blend_weights

            region_count = count_array[min_z:max_z, min_y:max_y, min_x:max_x]

            # Loop over patches and add only the valid region.
            for patch, pos, valid in zip(patches, positions, valid_sizes):
                valid_z, valid_y, valid_x = valid
                z0, y0, x0 = pos
                z_rel = z0 - min_z
                y_rel = y0 - min_y
                x_rel = x0 - min_x

                # Crop the patch and blending weights to the valid region.
                if len(patch.shape) == 4:
                    patch_valid = patch[:, :valid_z, :valid_y, :valid_x]
                    local_weights = base_weights[:, :valid_z, :valid_y, :valid_x]
                else:
                    patch_valid = patch[:valid_z, :valid_y, :valid_x]
                    local_weights = base_weights[:valid_z, :valid_y, :valid_x]

                weighted_patch = patch_valid * local_weights

                if len(patch.shape) == 4:
                    region_sum[:, z_rel:z_rel + valid_z, y_rel:y_rel + valid_y, x_rel:x_rel + valid_x] += weighted_patch
                    # For count, if using multi–channel weights, just use one channel.
                    if local_weights.ndim == 4:
                        region_count[z_rel:z_rel + valid_z, y_rel:y_rel + valid_y, x_rel:x_rel + valid_x] += \
                        local_weights[0]
                    else:
                        region_count[z_rel:z_rel + valid_z, y_rel:y_rel + valid_y,
                        x_rel:x_rel + valid_x] += local_weights
                else:
                    region_sum[z_rel:z_rel + valid_z, y_rel:y_rel + valid_y, x_rel:x_rel + valid_x] += weighted_patch
                    region_count[z_rel:z_rel + valid_z, y_rel:y_rel + valid_y, x_rel:x_rel + valid_x] += local_weights

            # Write back the updated regions.
            if len(patches[0].shape) == 4:
                output_array[:, min_z:max_z, min_y:max_y, min_x:max_x] = region_sum
            else:
                output_array[min_z:max_z, min_y:max_y, min_x:max_x] = region_sum
            count_array[min_z:max_z, min_y:max_y, min_x:max_x] = region_count

    def _process_model_outputs(self, outputs: Dict, positions: List[Tuple],
                               output_arrays: Dict, count_arrays: Dict):
        """Process model outputs with buffering and offload writing asynchronously when full"""
        for i, pos in enumerate(positions):
            for tgt_name in self.mgr.infer_output_targets:
                pred = outputs[tgt_name][i].cpu().numpy()
                if self.mgr.infer_output_targets[tgt_name]["channels"] == 1 and pred.shape[0] == 1:
                    pred = np.squeeze(pred, axis=0)
                self.patch_buffer[tgt_name].append(pred)
            self.buffer_positions.append(pos)

        if len(self.buffer_positions) >= self.buffer_size:
            # Make local copies of the current buffers and clear the instance buffers.
            local_patch_buffer = {k: v[:] for k, v in self.patch_buffer.items()}
            local_positions = self.buffer_positions[:]
            self.patch_buffer.clear()
            self.buffer_positions.clear()
            # Offload asynchronous writing.
            future = self.executor.submit(self._process_buffer, output_arrays, count_arrays,
                                          local_patch_buffer, local_positions)
            self.write_futures.append(future)

            # --- NEW: Wait (block) if too many pending write futures to avoid unbounded memory usage ---
            if len(self.write_futures) >= self.max_pending_writes:
                done, _ = wait(self.write_futures, return_when=FIRST_COMPLETED)
                for future in done:
                    self.write_futures.remove(future)

    def infer(self):
        # Create a synchronizer for concurrent writes
        sync_path = os.path.join(self.mgr.infer_output_path, ".zarr_sync")
        synchronizer = zarr.ProcessSynchronizer(sync_path)
        store_path = os.path.join(self.mgr.infer_output_path, "predictions.zarr")

        if not self.postprocess_only:
            # Only rank 0 creates the Zarr store and datasets.
            if self.rank == 0:
                if os.path.isdir(store_path):
                    raise FileExistsError(f"Zarr store '{store_path}' already exists.")
                zarr_store = zarr.open(store_path, mode='w', synchronizer=synchronizer)
                output_arrays = {}
                count_arrays = {}

                # Create a temporary dataset to determine the full output shape.
                dataset_temp = InferenceDataset(
                    input_path=self.mgr.infer_input_path,
                    targets=self.mgr.infer_output_targets,
                    patch_size=self.mgr.infer_patch_size,
                    input_format=self.mgr.infer_input_format,
                    overlap=self.mgr.infer_overlap,
                    load_all=self.mgr.infer_load_all
                )
                z_max, y_max, x_max = dataset_temp.input_shape
                chunk_z, chunk_y, chunk_x = self.mgr.infer_patch_size

                compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)

                for tgt_name, tgt_conf in self.mgr.infer_output_targets.items():
                    c = tgt_conf["channels"]
                    if c == 1:
                        out_shape = (z_max, y_max, x_max)
                        chunks = (chunk_z, chunk_y, chunk_x)
                    else:
                        out_shape = (c, z_max, y_max, x_max)
                        chunks = (c, chunk_z, chunk_y, chunk_x)

                    sum_ds = zarr_store.create_dataset(
                        name=f"{tgt_name}_sum",
                        shape=out_shape,
                        chunks=chunks,
                        dtype='float32',
                        compressor=compressor,
                        fill_value=0,
                        synchronizer=synchronizer
                    )
                    cnt_ds = zarr_store.create_dataset(
                        name=f"{tgt_name}_count",
                        shape=(z_max, y_max, x_max),
                        chunks=(chunk_z, chunk_y, chunk_x),
                        dtype='float32',
                        compressor=compressor,
                        fill_value=0,
                        synchronizer=synchronizer
                    )

                    output_arrays[tgt_name] = sum_ds
                    count_arrays[tgt_name] = cnt_ds

            # Wait for rank 0 to create the store.
            if dist.is_initialized():
                dist.barrier(device_ids=[torch.cuda.current_device()])

            if self.rank != 0:
                zarr_store = zarr.open(store_path, mode='r+', synchronizer=synchronizer)
                output_arrays = {
                    tgt_name: zarr_store[f"{tgt_name}_sum"]
                    for tgt_name in self.mgr.infer_output_targets
                }
                count_arrays = {
                    tgt_name: zarr_store[f"{tgt_name}_count"]
                    for tgt_name in self.mgr.infer_output_targets
                }

            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            device = torch.device(f'cuda:{local_rank}')

            # Initialize the model and move it to the correct device.
            model = self._build_model()
            torch.set_float32_matmul_precision('high')
            #model = torch.compile(model)
            model = model.to(device)

            # Load the checkpoint from file
            checkpoint = torch.load(self.mgr.infer_checkpoint_path, map_location=device)

            # Extract the state dict from the checkpoint
            if "network_weights" in checkpoint:
                state_dict = checkpoint["network_weights"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint  # assume the checkpoint is the state dict

            # if self.mgr.inference_config.get("load_from_nnunet", False):
            #     state_dict = remap_state_dict(state_dict)

            # Optionally remove DataParallel/DPD prefix if present
            def remove_module_prefix(state_dict):
                new_state_dict = {}
                for k, v in state_dict.items():
                    new_key = k.replace("module.", "")
                    new_state_dict[new_key] = v
                return new_state_dict

            state_dict = remove_module_prefix(state_dict)




            # Load the state dict into the model (using strict=False for flexibility)
            model.load_state_dict(state_dict, strict=True)

            # Now wrap with DDP (if needed)
            if dist.is_initialized():
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index])

            dataset = InferenceDataset(
                input_path=self.mgr.infer_input_path,
                targets=self.mgr.infer_output_targets,
                patch_size=self.mgr.infer_patch_size,
                input_format=self.mgr.infer_input_format,
                overlap=self.mgr.infer_overlap,
                load_all=self.mgr.infer_load_all
            )

            sampler = None
            if dist.is_initialized():
                sampler = DistributedSampler(dataset, shuffle=False)
                sampler.set_epoch(0)

            loader = DataLoader(
                dataset,
                batch_size=self.mgr.infer_batch_size,
                shuffle=(sampler is None),
                sampler=sampler,
                num_workers=self.mgr.infer_num_dataloader_workers,
                prefetch_factor=8,
                pin_memory=False,
                persistent_workers=False
            )

            model.eval()
            counter = 0  # For tracking indices from the sampler.
            if self.rank == 0:
                iterator = tqdm(enumerate(loader), total=len(loader),
                                desc="Running inference on patches...")
            else:
                iterator = enumerate(loader)

            with torch.no_grad(), torch.amp.autocast("cuda"):
                for batch_idx, data in iterator:
                    patches = data["image"].to(device)
                    indices = data["index"]  # these are the indices from the dataset
                    positions = [dataset.all_positions[i] for i in indices]

                    raw_outputs = model(patches)
                    outputs = {}
                    for t_name in self.mgr.infer_output_targets:
                        t_conf = self.mgr.infer_output_targets[t_name]
                        activation_str = t_conf.get("activation", "none").lower()
                        if activation_str == "sigmoid":
                            outputs[t_name] = torch.sigmoid(raw_outputs[t_name])
                        elif activation_str == "softmax":
                            outputs[t_name] = torch.softmax(raw_outputs[t_name], dim=1)
                        else:
                            outputs[t_name] = raw_outputs[t_name]
                    self._process_model_outputs(outputs, positions, output_arrays, count_arrays)

            # Process any remaining patches in the buffer.
            if self.buffer_positions:
                self._process_buffer(output_arrays, count_arrays)
            for future in self.write_futures:
                future.result()

            # Shut down the executor.
            if self.executor is not None:
                self.executor.shutdown(wait=True)

        else:
            # Postprocess-only: open the existing store.
            zarr_store = zarr.open(store_path, mode='r+', synchronizer=synchronizer)

        # Barrier: ensure all processes finish inference before postprocessing.
        if dist.is_initialized():
            dist.barrier(device_ids=[torch.cuda.current_device()])

        # Only rank 0 performs postprocessing.
        if self.rank == 0:
            self._optimized_postprocessing(zarr_store)

        # Final barrier to ensure all processes complete postprocessing.
        if dist.is_initialized():
            dist.barrier(device_ids=[torch.cuda.current_device()])

    def _optimized_postprocessing(self, zarr_store):
        """Optimized post-processing with improved vector handling"""
        for tgt_name in self.mgr.infer_output_targets:
            sum_ds = zarr_store[f"{tgt_name}_sum"]
            cnt_ds = zarr_store[f"{tgt_name}_count"]
            is_normals = (tgt_name.lower() == "normals")
            chunk_size = sum_ds.chunks[-3]

            final_dtype = "uint16" if is_normals else "uint8"
            compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)

            final_ds = zarr_store.create_dataset(
                name=f"{tgt_name}_final",
                shape=sum_ds.shape,
                chunks=sum_ds.chunks,
                dtype=final_dtype,
                compressor=compressor,
                fill_value=0
            )

            for z0 in tqdm(range(0, sum_ds.shape[-3], chunk_size),
                           desc=f"Processing {tgt_name}"):
                z1 = min(z0 + chunk_size, sum_ds.shape[-3])
                if len(sum_ds.shape) == 4:
                    sum_chunk = sum_ds[:, z0:z1].copy()
                    count_chunk = cnt_ds[z0:z1].copy()
                    count_chunk = np.expand_dims(count_chunk, axis=0)
                    count_chunk = np.broadcast_to(count_chunk, sum_chunk.shape)
                else:
                    sum_chunk = sum_ds[z0:z1].copy()
                    count_chunk = cnt_ds[z0:z1].copy()

                mask = count_chunk > 0

                if is_normals and len(sum_chunk.shape) == 4:
                    eps = 1e-8
                    mag = np.sqrt(np.sum(sum_chunk ** 2, axis=0)) + eps
                    sum_chunk = sum_chunk / np.expand_dims(mag, axis=0)
                else:
                    sum_chunk[mask] /= count_chunk[mask]

                if is_normals:
                    sum_chunk = ((sum_chunk + 1.0) / 2.0 * 65535.0).clip(0, 65535).astype(np.uint16)
                else:
                    sum_chunk = (sum_chunk * 255.0).clip(0, 255).astype(np.uint8)

                if len(sum_ds.shape) == 4:
                    final_ds[:, z0:z1] = sum_chunk
                else:
                    final_ds[z0:z1] = sum_chunk

        # if self.write_layers:
        #     self._write_jpeg_slices(zarr_store)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inference script for MultiTaskResidualUNetSE3D with DDP and asynchronous disk writes."
    )
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to your config file. Use the same one you used for training!")
    parser.add_argument("--write_layers", action="store_true",
                        help="Write the sliced z layers to disk")
    parser.add_argument("--postprocess_only", action="store_true",
                        help="Skip the inference pass and only do final averaging + casting on existing sums/counts.")
    parser.add_argument("--num_write_workers", type=int, default=4,
                        help="Number of worker threads for asynchronous disk writes.")

    args = parser.parse_args()

    # Set the CUDA device before initializing the process group.
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)  # Set the device first!
        dist.init_process_group(backend='nccl', init_method='env://')
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    inference_handler = ZarrInferenceHandler(
        config_file=args.config_path,
        write_layers=args.write_layers,
        postprocess_only=args.postprocess_only,
        num_write_workers=args.num_write_workers
    )
    inference_handler.infer()  # Ensure infer() calls barriers as needed.

    if dist.is_initialized():
        dist.destroy_process_group()
