"""A trained re-ranker as a beam hook."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from vesuvius.neural_tracing.fiber_follow.beam.data import collate_beam
from vesuvius.neural_tracing.fiber_follow.beam.states import BeamStateConfig, pool_state
from vesuvius.neural_tracing.fiber_follow.geometry import crop_local_grid

HOOK_MODES = ('additive', 'replace')


class ModelBeamHook:
    """Re-scores each prune-time pool with one model forward.

    ``additive``: loss = hand loss + weight * (-log p_onfiber). Keeps the hand
    loss scale so later hand-scored rounds stay comparable.
    ``replace``: loss = -rank logit (pure learned order within the pool).
    In open-ended tracing the hook stops the search when no candidate reaches
    ``stop_threshold`` on-fiber probability.
    """

    def __init__(self, model, vol, cfg: BeamStateConfig, device='cuda', mode='additive', weight=1.0,
                 stop_threshold: float | None = None):
        if mode not in HOOK_MODES:
            raise ValueError(f'hook mode must be one of {HOOK_MODES}')
        self.model, self.vol, self.cfg, self.device = model, vol, cfg, torch.device(device)
        self.mode, self.weight, self.stop_threshold = mode, float(weight), stop_threshold
        self.grid = torch.from_numpy(crop_local_grid(cfg.crop)).float()
        self.calls = 0
        self.last = None

    @torch.no_grad()
    def scores(self, pool):
        item = pool_state(pool, self.cfg)
        batch = {k: v.to(self.device) for k, v in collate_beam([item], self.vol, self.cfg.crop, self.grid).items()}
        self.model.eval()
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.device.type == 'cuda'):
            out = self.model(batch['x'].float(), batch['hist'], batch['hmask'], batch['candidates'],
                             batch['point_mask'], batch['hand_rel'])
        n = len(pool)
        return (out['ranks'][0, :n].float().cpu().numpy(),
                out['onfiber_logits'][0, :n].float().cpu().numpy())

    def __call__(self, pool):
        ranks, onfiber = self.scores(pool)
        self.calls += 1
        p = 1 / (1 + np.exp(-onfiber.astype(np.float64)))
        if self.mode == 'additive':
            losses = pool.losses.astype(np.float64) + self.weight * (-F.logsigmoid(torch.from_numpy(onfiber)).numpy())
        else:
            losses = -ranks.astype(np.float64)
        stop = self.stop_threshold is not None and float(p.max()) < self.stop_threshold
        self.last = dict(ranks=ranks, onfiber=p, losses=losses, stop=stop)
        return losses.astype(np.float32), bool(stop)
