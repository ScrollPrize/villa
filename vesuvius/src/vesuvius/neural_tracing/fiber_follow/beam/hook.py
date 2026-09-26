"""Replace native step costs before any proposal pruning."""
import numpy as np
import torch
import torch.nn.functional as F

from vesuvius.neural_tracing.fiber_follow.beam.data import collate_beam
from vesuvius.neural_tracing.fiber_follow.beam.states import supported_states


class ModelBeamHook:
    def __init__(self, model, vol, cfg, device='cuda', stop_threshold=None):
        self.model, self.vol, self.cfg = model, vol, cfg
        self.device, self.stop_threshold = torch.device(device), stop_threshold
        self.calls, self.last = 0, None

    @torch.no_grad()
    def scores(self, pool):
        result = torch.empty(len(pool), dtype=torch.float32)
        self.model.eval()
        for ids, item in supported_states(pool, self.cfg):
            batch = {k: v.to(self.device) for k, v in collate_beam([item], self.vol, self.cfg.crop).items()}
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.device.type == 'cuda'):
                # Shared encoder once per crop; score every proposal in chunks.
                features, context = self.model.encode_scene(batch['x'].float(), batch['hist'], batch['hmask'])
                logits = torch.cat([self.model.score_paths(features, context,
                                    batch['candidates'][:, a:a+self.model.cfg.score_chunk],
                                    batch['point_mask'][:, a:a+self.model.cfg.score_chunk])
                                    for a in range(0, len(ids), self.model.cfg.score_chunk)], 1)[0]
            result[ids] = logits.float().cpu()
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
