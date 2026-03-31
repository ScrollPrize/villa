import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Sequence

import torch
from skimage.morphology import skeletonize, dilation, opening, closing
import numpy as np

from vesuvius.models.augmentation.transforms.base.basic_transform import BasicTransform


class MedialSurfaceTransform(BasicTransform):
    def __init__(self,
                 do_tube: bool = True,
                 do_open: bool = False,
                 do_close: bool = True,
                 target_keys: Optional[Sequence[str]] = None,
                 ignore_values: Optional[dict] = None,
                 cache_dir: Optional[str] = None,
                 enable_disk_cache: bool = False,
                 memory_cache_size: int = 128):
        """
        Calculates the medial surface skeleton of the segmentation (plus an optional 2 px tube around it)
        and adds it to the dict with the key "skel"
        """
        super().__init__()
        self.do_tube = do_tube
        self.do_open = do_open
        self.do_close = do_close
        self.target_keys = tuple(target_keys) if target_keys else None
        self.ignore_values = dict(ignore_values or {})
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.enable_disk_cache = bool(enable_disk_cache and self.cache_dir is not None)
        self.memory_cache_size = max(1, int(memory_cache_size))
        self._cache: OrderedDict[str, tuple[tuple[tuple[int, int], ...], np.ndarray]] = OrderedDict()

    @staticmethod
    def _bbox_slices(mask: np.ndarray, margin: int = 0):
        if not np.any(mask):
            return None
        coords = np.where(mask)
        slices = []
        for axis, axis_coords in enumerate(coords):
            start = max(int(axis_coords.min()) - margin, 0)
            stop = min(int(axis_coords.max()) + margin + 1, mask.shape[axis])
            slices.append(slice(start, stop))
        return tuple(slices)

    @staticmethod
    def _roi_tuple_from_slices(roi_slices):
        return tuple((int(slc.start), int(slc.stop)) for slc in roi_slices)

    @staticmethod
    def _roi_slices_from_tuple(roi_tuple):
        return tuple(slice(start, stop) for start, stop in roi_tuple)

    def _cache_key(self, patch_info, target_key: str, ignore_value):
        if not patch_info:
            return None
        volume_name = patch_info.get("volume_name")
        position = patch_info.get("position")
        patch_size = patch_info.get("patch_size")
        scale = patch_info.get("scale")
        if volume_name is None or position is None or patch_size is None:
            return None
        payload = {
            "version": "v1",
            "target": target_key,
            "volume_name": volume_name,
            "position": list(position),
            "patch_size": list(patch_size),
            "scale": scale,
            "ignore_value": repr(ignore_value),
            "do_tube": self.do_tube,
            "do_open": self.do_open,
            "do_close": self.do_close,
        }
        return hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _cache_get(self, cache_key: Optional[str]):
        if cache_key is None:
            return None
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._cache.move_to_end(cache_key)
            return cached
        if not self.enable_disk_cache or self.cache_dir is None:
            return None

        cache_path = self.cache_dir / cache_key[:2] / f"{cache_key}.npz"
        if not cache_path.exists():
            return None
        try:
            with np.load(cache_path, allow_pickle=False) as payload:
                roi_tuple = tuple(tuple(int(v) for v in pair) for pair in payload["roi"].tolist())
                roi_values = np.ascontiguousarray(payload["values"])
        except Exception:
            return None

        self._cache_put(cache_key, roi_tuple, roi_values)
        return roi_tuple, roi_values

    def _cache_put(self, cache_key: Optional[str], roi_tuple, roi_values: np.ndarray):
        if cache_key is None or roi_tuple is None:
            return
        self._cache[cache_key] = (roi_tuple, np.ascontiguousarray(roi_values))
        self._cache.move_to_end(cache_key)
        while len(self._cache) > self.memory_cache_size:
            self._cache.popitem(last=False)

        if not self.enable_disk_cache or self.cache_dir is None:
            return

        cache_path = self.cache_dir / cache_key[:2] / f"{cache_key}.npz"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            return
        np.savez(cache_path, roi=np.asarray(roi_tuple, dtype=np.int64), values=np.ascontiguousarray(roi_values))

    def _compute_skeleton(self, seg_processed: np.ndarray) -> np.ndarray:
        bin_seg = seg_processed > 0
        seg_all_skel = np.zeros_like(seg_processed, dtype=np.float32)
        margin = 2 if (self.do_tube or self.do_open or self.do_close) else 0

        for c in range(bin_seg.shape[0]):
            seg_c = bin_seg[c]
            if seg_c.sum() == 0:
                continue

            roi_slices = self._bbox_slices(seg_c, margin=margin)
            if roi_slices is None:
                continue
            seg_roi = seg_c[roi_slices]

            if seg_roi.ndim == 3:
                skel = np.zeros_like(seg_roi, dtype=bool)
                for z in range(seg_roi.shape[0]):
                    skel[z] |= skeletonize(seg_roi[z])
            elif seg_roi.ndim == 2:
                skel = skeletonize(seg_roi)
            else:
                raise ValueError(f"Unsupported segmentation dimensionality {seg_roi.ndim} for skeletonization")

            if self.do_tube:
                skel = dilation(dilation(skel))
            if self.do_open:
                skel = opening(skel)
            if self.do_close:
                skel = closing(skel)

            seg_all_skel[(c, *roi_slices)] = (
                skel.astype(np.float32) * seg_processed[(c, *roi_slices)].astype(np.float32)
            )

        return seg_all_skel

    def apply(self, data_dict, **params):
        # Collect regression keys to avoid processing continuous aux targets
        regression_keys = set(data_dict.get('regression_keys', []) or [])
        # Find eligible target keys: tensor-valued, not image/meta, not regression aux
        candidate_keys = [
            k for k, v in data_dict.items()
            if k not in ['image', 'is_unlabeled', 'regression_keys']
            and isinstance(v, torch.Tensor)
            and k not in regression_keys
        ]

        if self.target_keys is not None:
            target_keys = [k for k in candidate_keys if k in self.target_keys]
        else:
            target_keys = candidate_keys

        # Process each target
        patch_info = data_dict.get("patch_info", {}) or {}
        for target_key in target_keys:
            t = data_dict[target_key]
            orig_device = t.device
            seg_all = t.detach().cpu().numpy()

            ignore_value = self.ignore_values.get(target_key)
            if ignore_value is not None:
                seg_processed = np.where(seg_all == ignore_value, 0, seg_all)
            else:
                seg_processed = seg_all

            cache_key = self._cache_key(patch_info, target_key, ignore_value)
            cached = self._cache_get(cache_key)
            if cached is not None:
                roi_tuple, roi_values = cached
                seg_all_skel = np.zeros_like(seg_processed, dtype=np.float32)
                seg_all_skel[(slice(None), *self._roi_slices_from_tuple(roi_tuple))] = roi_values
            else:
                seg_all_skel = self._compute_skeleton(seg_processed)
                roi_slices = self._bbox_slices(np.any(seg_all_skel != 0, axis=0), margin=0)
                if roi_slices is not None:
                    self._cache_put(
                        cache_key,
                        self._roi_tuple_from_slices(roi_slices),
                        seg_all_skel[(slice(None), *roi_slices)],
                    )

            data_dict[f"{target_key}_skel"] = torch.from_numpy(seg_all_skel).to(orig_device)

        return data_dict
