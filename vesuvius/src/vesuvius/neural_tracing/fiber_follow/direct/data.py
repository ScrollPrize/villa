"""Shared training/rollout observation builder for the direct follower."""
from dataclasses import replace

import numpy as np
import torch

from vesuvius.neural_tracing.fiber_follow.data import collate_targets
from vesuvius.neural_tracing.fiber_follow.crop_sampling import scalar_crops
from vesuvius.neural_tracing.fiber_follow.volume import FiberVolume
from vesuvius.neural_tracing.fiber_follow.trace import ModelTracer


def image_crop(items, vol, crop, pool=None):
    """CT and presence only. Reuse the existing physical-coordinate sampler.

    The scalar sampler normalizes both channels to [0,1]. An empty history
    skips rendering. Sampling is identical in training
    and tracing, including the independently resolved presence grid.
    Each item reads only the axis-aligned block its own oriented crop needs.
    """
    return torch.stack([scalar_crops(items, vol, crop, pool, presence=presence)[:, 0]
                        for presence in (False, True)], 1)



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
            # Both scales sample the same presence array. One reader with the
            # combined budget lets the wider coarse footprint serve the fine crop.
            vol.presence.cache_bytes += self._coarse.presence.cache_bytes
            self._coarse.presence = vol.presence
        return dict(fine=image_crop(items, vol, self.cfg.fine, pool),
                    coarse=image_crop(items, self._coarse, self.cfg.coarse, pool))

    def __call__(self, items, vol):
        return dict(x=self.images(items, vol),
                    hist=torch.as_tensor(np.stack([i['hist_local'] for i in items]), dtype=torch.float32),
                    hmask=torch.as_tensor(np.stack([i['hmask'] for i in items]), dtype=torch.float32),
                    **collate_targets(items))


class DirectTracer(ModelTracer):
    def __init__(self, model, *args, judge=None, judge_slices=None, judge_policy=None, judge_explore_calls=0, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.additional_crops = (model.cfg.coarse,)
        self.observations = ObservationBuilder(model.cfg)
        self.judge, self.judge_slices, self.judge_policy = judge, judge_slices, judge_policy
        self.judge_reader = None
        self.judge_explore_calls = judge_explore_calls

    def begin_observed(self, frames):
        if self.judge is None:
            return super().begin_observed(frames)
        from .judge_slices import SliceStream
        from .judge_policy import JudgePolicy
        from .judge_model import FeatureCache
        if self.judge_reader is None:
            self.judge_reader = self.judge_slices.open()
        self.judge.eval()
        return [dict(stream=SliceStream(self.judge_reader, self.judge_slices, f),
                     policy=JudgePolicy(self.judge_policy, path_step=self.judge_slices.path_step), cache=FeatureCache()) for f in frames]

    def inspect_observed(self, state, path, final=False):
        if state is None:
            return None
        from .judge_model import sequence_tensors
        import time
        began = time.perf_counter()
        policy = state['policy']
        if policy.alarm is not None:
            if not final and state.get('explored', 0) < self.judge_explore_calls:
                state['explored'] = state.get('explored', 0)+1
                return None
            return policy.reason
        records = state['stream'].update(path, policy.accepted)
        sampled = time.perf_counter()
        batch = sequence_tensors(records, self.device, policy.anchor(records) or 0.)
        with torch.no_grad():
            encoded_before = state['cache'].encoded_views
            tokens = state['cache'].tokens(self.judge, records, self.device)
            if self.device.startswith('cuda'):
                torch.cuda.synchronize()
            encoded = time.perf_counter()
            logits = self.judge.decode(tokens, batch['metadata'], batch['valid'], batch['queries'])
            scores = logits.sigmoid()[0].cpu().numpy()
        state['records'] = records
        result = policy.decide(records, scores, final)
        state.setdefault('timings', []).append(dict(sample_seconds=sampled-began, encode_seconds=encoded-sampled,
                    decoder_seconds=time.perf_counter()-encoded, total_seconds=time.perf_counter()-began,
                    encoded_views=state['cache'].encoded_views-encoded_before,
                    cached_token_bytes=sum(v.numel()*v.element_size() for v in state['cache'].entries.values())))
        if result and self.judge_explore_calls and not final and state.get('explored', 0) < self.judge_explore_calls:
            state['explored'] = state.get('explored', 0)+1
            return None
        return result

    def export_observed(self, state, path, reason):
        if state is not None:
            state['observed_path'] = path.copy()
        return super().export_observed(state, path, reason) if state is None else state['policy'].export(path, reason)

    def observed_decision(self, state):
        if state is None:
            return {}
        return dict(judge_revision=len(state['policy'].audit)-1,
                    judge_accepted=state['policy'].accepted,
                    judge_alarm=state['policy'].alarm is not None)

    def build_inputs(self, pos, frames, hist, hmask):
        items = [dict(pos=p, frame=f) for p, f in zip(pos, frames)]
        return {k: v.to(self.device) for k, v in self.observations.images(items, self.vol, self.pool).items()}
