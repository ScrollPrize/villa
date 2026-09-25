"""Shared training/rollout observation builder for the direct follower."""
from dataclasses import replace

import numpy as np
import torch

from vesuvius.neural_tracing.fiber_follow.data import collate_targets, read_blocks
from vesuvius.neural_tracing.fiber_follow.geometry import crop_local_grid
from vesuvius.neural_tracing.fiber_follow.fast_sample import sample_crop
from vesuvius.neural_tracing.fiber_follow.volume import FiberVolume
from vesuvius.neural_tracing.fiber_follow.trace import ModelTracer


def image_crop(items, vol, crop, pool=None):
    """CT and presence only. Reuse the existing physical-coordinate sampler.

    The scalar sampler normalizes both channels to [0,1]. An empty history
    skips rendering. Sampling is identical in training
    and tracing, including the independently resolved presence grid.
    """
    grid = crop_local_grid(crop).reshape(-1, 3).astype(np.float64)
    empty = np.empty((0, 3), np.float32)
    mask = np.empty(0, np.float32)
    result = np.empty((len(items), 2, crop.depth, crop.width, crop.width), np.float32)
    for presence in (False, True):
        raw, starts = read_blocks(items, vol, crop, pool, presence=presence)
        scale = 1. if presence else vol.input_scale
        for j, item in enumerate(items):
            sampled = sample_crop(raw[j], starts[j], item['pos']*scale, item['frame']*scale,
                                  grid, False, empty, mask, 2, 1., 'points')[0]
            result[j, int(presence)] = sampled.reshape(crop.depth, crop.width, crop.width)
    return torch.from_numpy(result)


class ObservationBuilder:
    """Lazy worker-local coarse-volume reader; fine CT uses the supplied volume."""
    def __init__(self, cfg, coarse_level=1, coarse_grid_scale=8.):
        self.cfg, self.coarse_level, self.coarse_grid_scale = cfg, coarse_level, coarse_grid_scale
        self._coarse = None

    def __getstate__(self):
        return dict(self.__dict__, _coarse=None)

    def images(self, items, vol, pool=None):
        if self._coarse is None:
            spec = replace(vol.spec, ct_level=self.coarse_level, ct_grid_scale=self.coarse_grid_scale)
            self._coarse = FiberVolume(spec, cache_bytes=vol.ct.cache_bytes)
        return dict(fine=image_crop(items, vol, self.cfg.fine, pool),
                    coarse=image_crop(items, self._coarse, self.cfg.coarse, pool))

    def __call__(self, items, vol):
        return dict(x=self.images(items, vol),
                    hist=torch.as_tensor(np.stack([i['hist_local'] for i in items]), dtype=torch.float32),
                    hmask=torch.as_tensor(np.stack([i['hmask'] for i in items]), dtype=torch.float32),
                    **collate_targets(items))


class DirectTracer(ModelTracer):
    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.additional_crops = (model.cfg.coarse,)
        self.observations = ObservationBuilder(model.cfg)

    def build_inputs(self, pos, frames, hist, hmask):
        items = [dict(pos=p, frame=f) for p, f in zip(pos, frames)]
        return {k: v.to(self.device) for k, v in self.observations.images(items, self.vol, self.pool).items()}
