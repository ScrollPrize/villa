"""Replace native step costs before any proposal pruning."""
import numpy as np
import torch
import torch.nn.functional as F

from vesuvius.neural_tracing.fiber_follow.beam.data import collate_beam
from vesuvius.neural_tracing.fiber_follow.beam.states import pool_state, supported_states
from vesuvius.neural_tracing.fiber_follow.geometry import normalize

# Inference scores this many proposals per call (a full cone pool fits easily);
# chunking never changes scores, and fewer small launches are much faster.
INFERENCE_CHUNK = 4096
REUSE_MIN_HEADING_COS = 0.7  # ~45 degrees; catches reversals, not per-step cone jitter


class ModelBeamHook:
    """Learned step costs for every proposal of every native beam generation.

    ``reencode_distance`` (trace-grid voxels) > 0 reuses the last encoded crop
    while the best parent stays within that distance of its centre with a
    similar heading, within the same trace and model weights, and every
    proposal endpoint lies inside it. The crop's CT is exact; its rendered
    history and history context lag by up to that distance (each proposal
    still supplies its own recent path). 0 re-encodes at every generation.
    """
    def __init__(self, model, vol, cfg, device='cuda', stop_threshold=None, compile=False, reencode_distance=0.):
        self.model, self.vol, self.cfg = model, vol, cfg
        self.device, self.stop_threshold = torch.device(device), stop_threshold
        self.calls, self.last = 0, None
        self.reencode_distance = float(reencode_distance)
        self.encodes = self.reuses = 0
        self._scene = None
        # The encoder always sees one fixed-size crop, so compiling it is
        # shape-stable; proposal scoring has variable sizes and stays eager.
        self.encode_scene = getattr(model, 'encode_scene', None)
        if compile and model is not None and self.device.type == 'cuda':
            self.encode_scene = torch.compile(model.encode_scene)

    def _score(self, features, context, candidates, point_mask):
        candidates = torch.as_tensor(candidates, device=self.device)[None]
        point_mask = torch.as_tensor(point_mask, device=self.device)[None]
        chunk = max(self.model.cfg.score_chunk, INFERENCE_CHUNK)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.device.type == 'cuda'):
            # Shared encoder once per crop; score every proposal in chunks.
            return torch.cat([self.model.score_paths(features, context, candidates[:, a:a+chunk],
                                                     point_mask[:, a:a+chunk])
                              for a in range(0, candidates.shape[1], chunk)], 1)[0].float().cpu()

    def _reusable_scene(self, pool, key):
        scene = self._scene
        if scene is None or scene['key'] != key:
            return None
        anchor = int(np.argmin(pool.parent_losses))
        start, end = pool.paths.parent_bounds(anchor)
        reference = pool.paths.parent_points[start:end]
        if np.linalg.norm(reference[-1]-scene['pos']) > self.reencode_distance:
            return None
        # Heading over the recent path (the single last step jitters with the cone).
        back = np.flatnonzero(np.linalg.norm(reference-reference[-1], axis=1) >= self.reencode_distance)
        tail = reference[back[-1]] if len(back) else reference[0]
        heading = reference[-1]-tail if np.linalg.norm(reference[-1]-tail) > 0 else pool.step_directions[anchor]
        if float(normalize(heading) @ scene['frame'][:, 2]) < REUSE_MIN_HEADING_COS:
            return None
        item = pool_state(pool, self.cfg, crop_pos=scene['pos'], crop_frame=scene['frame'])
        return item if item['supported'].all() else None

    @torch.no_grad()
    def scores(self, pool):
        result = torch.empty(len(pool), dtype=torch.float32)
        self.model.eval()
        key = None
        if self.reencode_distance > 0:
            # Weights change in place during training: their version invalidates the scene.
            key = (pool.phase, tuple(pool.start), tuple(pool.target),
                   next(self.model.parameters())._version, id(self.model))
            item = self._reusable_scene(pool, key)
            if item is not None:
                self.reuses += 1
                result[:] = self._score(self._scene['features'], self._scene['context'],
                                        item['candidates'], item['point_mask'])
                if not torch.isfinite(result).all():
                    raise ValueError('Nonfinite learned beam scores')
                return result
        self._scene = None
        for ids, item in supported_states(pool, self.cfg):
            batch = {k: v.to(self.device) for k, v in collate_beam([item], self.vol, self.cfg.crop, parallel=True).items()}
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.device.type == 'cuda'):
                features, context = self.encode_scene(batch['x'].float(), batch['hist'], batch['hmask'])
            self.encodes += 1
            if key is not None and self._scene is None:
                self._scene = dict(key=key, pos=item['pos'], frame=item['frame'], features=features, context=context)
            result[ids] = self._score(features, context, batch['candidates'][0], batch['point_mask'][0])
        if not torch.isfinite(result).all():
            raise ValueError('Nonfinite learned beam scores')
        return result

    def __call__(self, pool):
        logits = self.scores(pool)
        self.calls += 1
        probabilities = logits.sigmoid().numpy()
        costs = F.softplus(-logits).numpy()*pool.step_lengths
        # Parent already includes earlier learned increments. The hand-scored
        # proposed step is replaced, never added back or double-counted.
        losses = pool.parent_losses.astype(np.float64) + costs
        stop = self.stop_threshold is not None and float(probabilities.max()) < self.stop_threshold
        self.last = dict(ranks=logits.numpy(), onfiber=probabilities, losses=losses, stop=stop)
        return losses.astype(np.float32), bool(stop)
