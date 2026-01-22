
import json
import time
import queue
import threading
import zarr
import cupy as cp
from cucim.skimage.measure import label as cucim_label, regionprops_table
import torch
import accelerate
from accelerate.utils import TorchDynamoPlugin
import numpy as np
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor
from cachetools import LRUCache

from vesuvius.neural_tracing.dataset import get_crop_from_volume, build_localiser, make_heatmaps, mark_context_point
from vesuvius.image_proc.intensity.normalization import normalize_zscore
from vesuvius.models.run.tta import infer_with_tta


class CropCache:
    """
    Multi-worker prefetching LRU cache for volume crops.

    Features:
    - Background workers fetch crops while GPU runs inference
    - LRU eviction policy with memory budget
    - FIFO prefetch queue
    - Thread-safe via cachetools.LRUCache

    Supports both isotropic (cubic) and anisotropic crop sizes.
    """

    def __init__(
        self,
        volume,
        crop_size,  # int or tuple/list of 3 ints [D, H, W]
        max_memory_bytes: int = 4 * 1024**3,
        num_workers: int = 4,
        prefetch_queue_size: int = 64,
        normalize: bool = True,
    ):
        self.volume = volume

        # Normalize crop_size to numpy array [D, H, W]
        if isinstance(crop_size, (list, tuple)):
            self.crop_size = np.array(crop_size)
        else:
            self.crop_size = np.array([crop_size, crop_size, crop_size])

        # Supercrop is cubic with size = 2 * max dimension (for spatial reuse)
        self.supercrop_size = int(np.max(self.crop_size) * 2)
        self.num_workers = num_workers
        self.normalize = normalize

        # Estimate max items based on memory budget
        # Each supercrop is supercrop_size^3 * 4 bytes (float32) + min_corner (3 * 8 bytes)
        supercrop_bytes = self.supercrop_size ** 3 * 4 + 24
        max_items = max(1, max_memory_bytes // supercrop_bytes)

        # Thread-safe LRU cache
        self._cache = LRUCache(maxsize=max_items)
        self._cache_lock = threading.Lock()

        # Prefetch queue and tracking
        self._prefetch_queue: queue.Queue = queue.Queue(maxsize=prefetch_queue_size)
        self._in_flight: set = set()
        self._in_flight_lock = threading.Lock()

        # Stats for profiling
        self._stats = {'hits': 0, 'misses': 0, 'prefetch_queued': 0}

        # Worker pool
        self._shutdown = threading.Event()
        self._executor = ThreadPoolExecutor(max_workers=num_workers, thread_name_prefix='CropFetcher')
        for _ in range(num_workers):
            self._executor.submit(self._prefetch_worker)

    def _compute_supercrop_key(self, center_zyx: torch.Tensor) -> tuple:
        """
        Compute the cache key for the supercrop that contains this center.
        Supercrops are aligned to a grid with spacing = max(crop_size).
        """
        # Use max dimension for grid spacing (supercrops are cubic)
        grid_spacing = int(np.max(self.crop_size))
        grid_idx = (center_zyx / grid_spacing).int()
        supercrop_center = (grid_idx.float() + 0.5) * grid_spacing
        supercrop_min = (supercrop_center - self.supercrop_size // 2).int()
        return (int(supercrop_min[0]), int(supercrop_min[1]), int(supercrop_min[2]), self.supercrop_size)

    def _key_to_center(self, key: tuple) -> torch.Tensor:
        """Reconstruct center coordinates from cache key."""
        z_min, y_min, x_min, size = key
        return torch.tensor([z_min + size // 2, y_min + size // 2, x_min + size // 2])

    def _extract_subcrop(
        self,
        supercrop: torch.Tensor,
        supercrop_min_corner: torch.Tensor,
        center_zyx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract the requested crop_size region from a cached super-crop."""
        # crop_size is [D, H, W] numpy array
        crop_size_tensor = torch.from_numpy(self.crop_size)
        crop_min = (center_zyx - crop_size_tensor // 2).int()
        offset = crop_min - supercrop_min_corner.int()

        z0, y0, x0 = int(offset[0]), int(offset[1]), int(offset[2])

        # Safety clamp (should rarely trigger if grid alignment is correct)
        # supercrop is cubic, crop_size may be anisotropic
        supercrop_dims = supercrop.shape
        d, h, w = int(self.crop_size[0]), int(self.crop_size[1]), int(self.crop_size[2])
        z0 = max(0, min(z0, supercrop_dims[0] - d))
        y0 = max(0, min(y0, supercrop_dims[1] - h))
        x0 = max(0, min(x0, supercrop_dims[2] - w))

        crop = supercrop[z0:z0 + d, y0:y0 + h, x0:x0 + w]
        actual_min_corner = supercrop_min_corner + torch.tensor([z0, y0, x0])
        return crop, actual_min_corner

    def _prefetch_worker(self):
        """Background worker that fetches supercrops from the prefetch queue."""
        while not self._shutdown.is_set():
            try:
                key = self._prefetch_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                # Check if already cached
                with self._cache_lock:
                    if key in self._cache:
                        continue

                # Reconstruct center from key and fetch supercrop
                supercrop_center = self._key_to_center(key)
                supercrop_size = key[3]

                # Fetch the supercrop (I/O releases GIL)
                crop, min_corner = get_crop_from_volume(self.volume, supercrop_center, supercrop_size, normalize=self.normalize)

                # Store in cache
                with self._cache_lock:
                    self._cache[key] = (crop, min_corner)

            finally:
                with self._in_flight_lock:
                    self._in_flight.discard(key)

    def get(self, center_zyx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get crop for given center coordinates.
        Fetches a larger supercrop if not cached, then extracts the requested region.
        """
        key = self._compute_supercrop_key(center_zyx)

        # Check cache for supercrop
        with self._cache_lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._stats['hits'] += 1
                supercrop, supercrop_min_corner = cached
                return self._extract_subcrop(supercrop, supercrop_min_corner, center_zyx)

        # Check if in-flight from prefetch
        with self._in_flight_lock:
            is_in_flight = key in self._in_flight

        if is_in_flight:
            # Wait for prefetch to complete (spin with backoff)
            for _ in range(100):  # Max ~1 second wait
                time.sleep(0.01)
                with self._cache_lock:
                    cached = self._cache.get(key)
                    if cached is not None:
                        self._stats['hits'] += 1
                        supercrop, supercrop_min_corner = cached
                        return self._extract_subcrop(supercrop, supercrop_min_corner, center_zyx)

        # Cache miss - fetch supercrop synchronously
        self._stats['misses'] += 1
        supercrop_center = self._key_to_center(key)
        supercrop, supercrop_min_corner = get_crop_from_volume(self.volume, supercrop_center, self.supercrop_size, normalize=self.normalize)

        with self._cache_lock:
            self._cache[key] = (supercrop, supercrop_min_corner)

        return self._extract_subcrop(supercrop, supercrop_min_corner, center_zyx)

    def prefetch(self, center_zyx_list: List[torch.Tensor]):
        """
        Submit list of coordinates for background prefetching.
        Caller should call this with upcoming positions before they're needed.
        Deduplicates by supercrop key since multiple requests may share the same supercrop.
        """
        seen_keys = set()
        for center_zyx in center_zyx_list:
            key = self._compute_supercrop_key(center_zyx)

            # Skip duplicates within this batch
            if key in seen_keys:
                continue
            seen_keys.add(key)

            # Skip if already cached
            with self._cache_lock:
                if key in self._cache:
                    continue

            # Skip if already in-flight
            with self._in_flight_lock:
                if key in self._in_flight:
                    continue
                self._in_flight.add(key)

            # Queue for prefetch (non-blocking, drop if full)
            try:
                self._prefetch_queue.put_nowait(key)
                self._stats['prefetch_queued'] += 1
            except queue.Full:
                with self._in_flight_lock:
                    self._in_flight.discard(key)

    def get_stats(self) -> dict:
        """Return cache statistics."""
        stats = dict(self._stats)
        total = stats['hits'] + stats['misses']
        stats['hit_rate'] = stats['hits'] / total if total > 0 else 0.0
        stats['cache_size'] = len(self._cache)
        return stats

    def print_stats(self):
        """Print cache statistics."""
        stats = self.get_stats()
        print(f"[CropCache] hits={stats['hits']} misses={stats['misses']} "
              f"hit_rate={stats['hit_rate']:.1%} prefetch_queued={stats['prefetch_queued']} "
              f"cache_size={stats['cache_size']}")

    def shutdown(self):
        """Shutdown worker threads."""
        self._shutdown.set()
        self._executor.shutdown(wait=True)
        self.print_stats()


class Inference:

    def __init__(self, model, config, volume_zarr, volume_scale):

        dynamo_plugin = TorchDynamoPlugin(
            backend="inductor",  # Options: "inductor", "aot_eager", "aot_nvfuser", etc.
            mode="default",      # Options: "default", "reduce-overhead", "max-autotune"
            fullgraph=False,
            dynamic=False,
            use_regional_compilation=False
        )

        self.accelerator = accelerate.Accelerator(
            mixed_precision=config['mixed_precision'],
            dynamo_plugin=dynamo_plugin
        )

        self.model = self.accelerator.prepare(model)
        self.device = self.accelerator.device
        self.config = config
        self.do_tta = config.get('do_tta', False)
        self.tta_type = config.get('tta_type', "rotation")
        self.tta_batched = bool(config.get('tta_batched', True))

        self.model.eval()

        print(f"loading volume zarr {volume_zarr}...")
        ome_zarr = zarr.open_group(volume_zarr, mode='r')
        self.volume = ome_zarr[str(volume_scale)]
        with open(f'{volume_zarr}/meta.json', 'rt') as meta_fp:
            self.voxel_size_um = json.load(meta_fp)['voxelsize']
        print(f"volume shape: {self.volume.shape}, dtype: {self.volume.dtype}, voxel-size: {self.voxel_size_um * 2 ** volume_scale}um")

        # cache for prefetching
        self.use_crop_cache = config.get('use_crop_cache', True)
        if self.use_crop_cache:
            cache_gb = config.get('crop_cache_gb', 4)
            num_workers = config.get('prefetch_workers', 4)
            self.crop_cache = CropCache(
                volume=self.volume,
                crop_size=config['crop_size'],
                max_memory_bytes=int(cache_gb * 1024**3),
                num_workers=num_workers,
            )
        else:
            self.crop_cache = None

        # cache base localiser (only depends on crop_size)
        if bool(config.get('use_localiser', True)):
            crop_size = config['crop_size']
            # Normalize crop_size to tuple of 3 ints [D, H, W]
            if isinstance(crop_size, (list, tuple)):
                crop_size_dhw = tuple(crop_size)
            else:
                crop_size_dhw = (crop_size, crop_size, crop_size)
            base_localiser = torch.linalg.norm(
                torch.stack(torch.meshgrid(
                    torch.arange(crop_size_dhw[0]),
                    torch.arange(crop_size_dhw[1]),
                    torch.arange(crop_size_dhw[2]),
                    indexing='ij'
                ), dim=-1).to(torch.float32) - torch.tensor(crop_size_dhw).float() / 2,
                dim=-1
            )
            self._base_localiser = base_localiser / base_localiser.amax() * 2 - 1
        else:
            self._base_localiser = None

    def get_heatmaps_at(self, zyx, prev_u, prev_v, prev_diag, return_debug: bool = False):
        if isinstance(zyx, torch.Tensor) and zyx.ndim == 1:
            zyx = zyx[None]
            prev_u = [prev_u]
            prev_v = [prev_v]
            prev_diag = [prev_diag]
            not_originally_batched = True
        else:
            not_originally_batched = False
        crop_size = self.config['crop_size']
        # Normalize crop_size to tuple of 3 ints [D, H, W]
        if isinstance(crop_size, (list, tuple)):
            crop_size_dhw = tuple(crop_size)
        else:
            crop_size_dhw = (crop_size, crop_size, crop_size)
        use_localiser = bool(self.config.get('use_localiser', True))
        zeros = torch.zeros([1, *crop_size_dhw])

        # pre-allocate input tensor (i tested w/o pin memory and it was slower)
        batch_size = len(zyx)
        num_channels = 5 if use_localiser else 4  # volume, [localiser], u, v, diag
        inputs_cpu = torch.empty(
            (batch_size, num_channels, *crop_size_dhw),
            dtype=torch.float32,
            pin_memory=True,
        )
        # Channel indices
        CH_VOLUME = 0
        CH_LOCALISER = 1 if use_localiser else None
        CH_U = 2 if use_localiser else 1
        CH_V = 3 if use_localiser else 2
        CH_DIAG = 4 if use_localiser else 3

        # For debug output (only first sample if batched)
        debug_volume_crop = None
        debug_localiser = None
        debug_prev_u_hm = None
        debug_prev_v_hm = None
        debug_prev_diag_hm = None

        min_corner_zyxs = []
        for idx in range(len(zyx)):
            if self.crop_cache is not None:
                volume_crop, min_corner_zyx = self.crop_cache.get(zyx[idx])
            else:
                volume_crop, min_corner_zyx = get_crop_from_volume(self.volume, zyx[idx], crop_size)
            min_corner_zyxs.append(min_corner_zyx)
            inputs_cpu[idx, CH_VOLUME] = volume_crop

            if use_localiser:
                localiser = self._base_localiser.clone()
                mark_context_point(localiser, zyx[idx] - min_corner_zyx, value=0.)
                inputs_cpu[idx, CH_LOCALISER] = localiser

            prev_u_hm = make_heatmaps([prev_u[idx][None]], min_corner_zyx, crop_size) if prev_u[idx] is not None else zeros
            prev_v_hm = make_heatmaps([prev_v[idx][None]], min_corner_zyx, crop_size) if prev_v[idx] is not None else zeros
            prev_diag_hm = make_heatmaps([prev_diag[idx][None]], min_corner_zyx, crop_size) if prev_diag[idx] is not None else zeros
            inputs_cpu[idx, CH_U] = prev_u_hm[0]
            inputs_cpu[idx, CH_V] = prev_v_hm[0]
            inputs_cpu[idx, CH_DIAG] = prev_diag_hm[0]

            # Capture debug data from first sample only
            if return_debug and idx == 0:
                debug_volume_crop = volume_crop.clone()
                debug_localiser = localiser.clone() if use_localiser else None
                debug_prev_u_hm = prev_u_hm[0].clone()
                debug_prev_v_hm = prev_v_hm[0].clone()
                debug_prev_diag_hm = prev_diag_hm[0].clone()

        inputs = inputs_cpu.to(self.device, non_blocking=True)
        min_corner_zyxs = torch.stack(min_corner_zyxs)

        def forward(model_inputs):
            outputs = self.model(model_inputs)
            logits = outputs['uv_heatmaps'] if isinstance(outputs, dict) else outputs
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            return logits

        def run_with_tta(model_inputs):
            if self.do_tta:
                return infer_with_tta(forward, model_inputs, self.tta_type, batched=self.tta_batched)
            return forward(model_inputs)

        with torch.no_grad(), torch.autocast('cuda'):
            logits = run_with_tta(inputs)
            logits = logits.reshape(len(zyx), 2, self.config['step_count'], *crop_size_dhw)  # u/v, step, z, y, x
            probs = torch.sigmoid(logits)

        if not_originally_batched:
            probs = probs.squeeze(0)
            min_corner_zyxs = min_corner_zyxs.squeeze(0)

        if return_debug:
            debug_data = {
                'volume_crop': debug_volume_crop,
                'localiser': debug_localiser,
                'prev_u_hm': debug_prev_u_hm,
                'prev_v_hm': debug_prev_v_hm,
                'prev_diag_hm': debug_prev_diag_hm,
                'model_input': inputs_cpu[0].clone(),  # First sample's full input
            }
            return probs, min_corner_zyxs, debug_data

        return probs, min_corner_zyxs

    def get_blob_coordinates(self, heatmap, min_corner_zyx, threshold=0.5, min_size=8, connectivity=26):

        binary_gpu = cp.asarray(heatmap > threshold)

        # connectivity 3 for cucim label is 26-connected, 2 would be 18
        labels_gpu = cucim_label(binary_gpu, connectivity=3)

        # area + centroid
        props = regionprops_table(labels_gpu, properties=['label', 'area', 'centroid'])
        areas = props['area'].get() 

        # find largest component and extract centroid
        if len(areas) > 0:
            largest_idx = np.argmax(areas)
            largest_size = areas[largest_idx]

            if largest_size >= min_size:
                centroid = np.array([
                    props['centroid-0'][largest_idx].get(),
                    props['centroid-1'][largest_idx].get(),
                    props['centroid-2'][largest_idx].get()
                ]) + min_corner_zyx.numpy()
                centroid_zyxs = centroid.reshape(1, 3).astype(np.float32)
            else:
                centroid_zyxs = np.empty((0, 3), dtype=np.float32)
        else:
            centroid_zyxs = np.empty((0, 3), dtype=np.float32)

        return torch.from_numpy(centroid_zyxs)


    def get_distance_transform_at(
        self,
        centers_zyx: List[torch.Tensor],
        conditioning_masks: List[torch.Tensor],
    ) -> List[Tuple[torch.Tensor, torch.Tensor, bool, str]]:
        """
        Run batched EDT inference for multiple samples.

        Args:
            centers_zyx: List of center coordinates in volume space (scaled), each shape [3]
            conditioning_masks: List of conditioning masks, each shape [D, H, W] or None if failed

        Returns:
            List of tuples: (distance_transform, min_corner_zyx, success, error_message)
            - distance_transform: [D, H, W] float32 (empty if failed)
            - min_corner_zyx: [3] coordinates of crop origin
            - success: bool indicating if this sample succeeded
            - error_message: string describing failure (empty if success)
        """
        crop_size = self.config['crop_size']
        batch_size = len(centers_zyx)

        # Initialize results with failures
        results: List[Tuple[torch.Tensor, torch.Tensor, bool, str]] = [
            (torch.empty(0), torch.zeros(3), False, "Not processed")
            for _ in range(batch_size)
        ]

        # Track which samples are valid
        valid_indices = []

        # Pre-validate samples
        if isinstance(crop_size, (list, tuple)):
            expected_shape = tuple(crop_size)
        else:
            expected_shape = (crop_size, crop_size, crop_size)
        for i, (center_zyx, mask) in enumerate(zip(centers_zyx, conditioning_masks)):
            if mask is None:
                results[i] = (torch.empty(0), torch.zeros(3), False, "Conditioning mask is None")
                continue

            if mask.shape != expected_shape:
                results[i] = (torch.empty(0), torch.zeros(3), False,
                             f"Mask shape {mask.shape} != expected {expected_shape}")
                continue

            valid_indices.append(i)

        if not valid_indices:
            return results

        # Gather volume crops and build batched input
        volume_crops = []
        min_corner_zyxs = []
        valid_masks = []

        for i in valid_indices:
            center_zyx = centers_zyx[i]
            mask = conditioning_masks[i]

            # Get volume crop (raw, without [-1,1] normalization)
            if self.crop_cache is not None:
                volume_crop_raw, min_corner_zyx = self.crop_cache.get(center_zyx)
            else:
                volume_crop_raw, min_corner_zyx = get_crop_from_volume(
                    self.volume, center_zyx, crop_size, normalize=False
                )

            # Apply z-score normalization
            volume_crop_zscore = torch.from_numpy(normalize_zscore(volume_crop_raw.numpy()))

            volume_crops.append(volume_crop_zscore)
            min_corner_zyxs.append(min_corner_zyx)
            valid_masks.append(mask)

        # Stack into batch tensors
        # inputs shape: [N_valid, 2, D, H, W]
        inputs = torch.stack([
            torch.stack([vol, mask], dim=0)
            for vol, mask in zip(volume_crops, valid_masks)
        ], dim=0).to(self.device)

        print(f"[EDT_BATCH_INFER] Input tensor shape: {inputs.shape}", flush=True)

        # Run batched inference
        with torch.no_grad(), torch.autocast('cuda'):
            outputs = self.model(inputs)
            dt_batch = outputs['dt']  # [N_valid, 1, D, H, W]
            if isinstance(dt_batch, (list, tuple)):
                dt_batch = dt_batch[0]  # Handle deep supervision

        print(f"[EDT_BATCH_INFER] Output tensor shape: {dt_batch.shape}", flush=True)

        # Unpack results back to original indices
        dt_batch_cpu = dt_batch[:, 0].cpu()  # [N_valid, D, H, W]

        for batch_idx, orig_idx in enumerate(valid_indices):
            dt = dt_batch_cpu[batch_idx]
            min_corner = min_corner_zyxs[batch_idx]
            results[orig_idx] = (dt, min_corner, True, "")

        return results
