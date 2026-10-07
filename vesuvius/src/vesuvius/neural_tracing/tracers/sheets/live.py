"""Live chains: training fronts continue across iterations from where the model left them.

Like the fiber tracer's live continuation (``tracers/fibers/train/live_continuation.py``), but the chain state stays in the trainer.
After a rollout, the fronts of the chains that go on (positions, curve, confidences and grid memory, detached, on
the training device) wait in a pool; once a batch of them has gathered (``train.batch``; once a rollout has filled
their memory, chains of any depth share a batch) they are kept under a job id, and a job for the loader workers
lists each chain's episode (patch, orientation, anchor), how far along the advance its front got, and the front's
extent. A worker reads
the chains' next GT windows and blocks as one batch (``EpisodeStream``, ``EpisodeSampler.continuation``) and the
trainer resumes the served chains on it. Fronts are never corrected towards GT between rollouts, so the model
trains on the states, and memory, it produces when tracing.

A chain ends after a random number of steps in ``train.chain_steps``, when under ``chain_on_sheet`` of its
supervised points end a rollout on the seed's sheet, when a rollout row has no confident point connected to the
surface traced so far (the inference stop, ``infer._reach_row``), when its patch has no next window, or when it
waits in the pool or for its window more than ``chain_max_age`` iterations after its rollout.
"""
from __future__ import annotations

import itertools
import queue

import torch

from vesuvius.neural_tracing.tracers.sheets.losses import landing
from vesuvius.neural_tracing.tracers.sheets.rollout import Tracer, concat_fronts, select_front


def select_batch(b: dict, index: list[int]) -> dict:
    """Episodes ``index`` of a device batch."""
    rows = torch.as_tensor(index, device=b['gt'].device)
    pick = lambda x: x[rows.to(x.device)]
    out = dict(b, gt=pick(b['gt']), valid=pick(b['valid']), block=pick(b['block']), origin=pick(b['origin']),
               meta=[b['meta'][i] for i in index])
    if 'chains' in b:
        out['chains'] = [b['chains'][i] for i in index]
    return out


class LiveChains:
    def __init__(self, cfg, spec, tracer: Tracer, jobs, generator):
        """``spec``: GT grid of the training episodes; ``jobs``: queue to the loader workers."""
        self.cfg, self.spec, self.tracer, self.jobs, self.generator = cfg, spec, tracer, jobs, generator
        self.columns = tracer.columns
        self.pending = {}  # job id -> chain state, waiting for the job's batch
        self.pool = []  # chain states (one rollout's survivors each) waiting for a full job
        self.ids = itertools.count()
        self.stats = {}

    def begin(self, b: dict, it: int) -> tuple[dict, dict | None]:
        """Batch ``b`` and its chain state: new chains (``front`` None: start from GT seed rows), or the resumed
        chains of a continuation batch, without episodes whose chains were dropped meanwhile (state None if none
        is left: skip the batch)."""
        tc = self.cfg.train
        device = b['gt'].device
        if 'chains' not in b:
            B = b['gt'].shape[0]
            lo, hi = (int(v) for v in tc.chain_steps)
            chain = dict(front=None, steps=torch.zeros(B, dtype=torch.long, device=device),
                         limit=torch.randint(lo, hi+1, (B,), device=device, generator=self.generator),
                         reach=torch.ones(B, self.columns, dtype=torch.bool, device=device))
        else:
            keep, parts = [], []
            for job in dict.fromkeys(j for j, _ in b['chains']):
                state = self.pending.pop(job, None)
                if state is None or it-state['issued'] > tc.chain_max_age:
                    continue
                rows = [k for k, (j, _) in enumerate(b['chains']) if j == job]
                index = torch.as_tensor([b['chains'][k][1] for k in rows], device=device)
                keep += rows
                parts.append((select_front(state['front'], index), state['steps'][index], state['limit'][index],
                              state['reach'][index]))
            if not parts:
                return b, None
            if len(keep) < len(b['chains']):
                b = select_batch(b, keep)
            fronts, steps, limits, reach = zip(*parts)
            chain = dict(front=concat_fronts(list(fronts)), steps=torch.cat(steps), limit=torch.cat(limits),
                         reach=torch.cat(reach))
        self.stats = dict(chain_continued=torch.tensor(float('chains' in b)),
                          chain_depth=chain['steps'].float().mean(), chain_batch=torch.tensor(float(len(b['meta']))))
        return b, chain

    def _column_conf(self, conf: torch.Tensor) -> torch.Tensor:
        """Point confidences (B, N) -> output columns', as written by inference (``Tracer.output_columns``)."""
        gap = self.tracer.col_gap
        return torch.minimum(conf[:, gap], conf[:, gap+1])

    def _reach(self, reach: torch.Tensor, kept: torch.Tensor) -> torch.Tensor:
        """Columns connected to the surface after rows ``kept`` (B, T, C), from ``reach`` (B, C) before them."""
        from vesuvius.neural_tracing.tracers.sheets.infer import _reach_row
        r, k = reach.cpu().numpy(), kept.cpu().numpy()
        for i in range(len(r)):
            for t in range(k.shape[1]):
                r[i] = _reach_row(k[i, t], r[i])
        return torch.from_numpy(r).to(reach.device)

    @torch.no_grad()
    def finish(self, b: dict, chain: dict, front, confs: list[torch.Tensor], it: int) -> None:
        """After a rollout of ``chain`` on ``b`` ending at ``front`` (point confidences ``confs`` per step): end
        chains or queue their next windows."""
        tc, d = self.cfg.train, self.cfg.data
        steps = chain['steps']+d.steps
        uv, _, same, ok = landing(front.pos, b['gt'], b['valid'], self.cfg.loss.sheet_tolerance)
        on_sheet = same.sum(1)/ok.sum(1).clamp_min(1)
        kept = torch.stack([self._column_conf(c) for c in confs], 1) > self.cfg.trace.confidence_threshold
        reach = self._reach(chain['reach'], kept)
        ended = dict(length=steps >= chain['limit'], sheet=on_sheet < tc.chain_on_sheet, stop=~reach.any(1))
        alive = ~(ended['length'] | ended['sheet'] | ended['stop'])
        self.stats.update({f'chain_end_{k}': v.float().mean() for k, v in ended.items()})
        for job in [j for j, state in self.pending.items() if it-state['issued'] > tc.chain_max_age]:
            del self.pending[job]
        self.pool = [state for state in self.pool if it-state['issued'] <= tc.chain_max_age]
        index = alive.nonzero()[:, 0]
        if len(index):
            u = self.spec.u_lo+uv[..., 0]*self.spec.res  # along the advance, from this window's seed row
            shift = (u*same).sum(1)/same.sum(1).clamp_min(1)
            extent = front.pos if front.curve is None else torch.cat([front.pos, front.curve], 1)
            lo, hi = extent.amin(1).cpu().numpy(), extent.amax(1).cpu().numpy()
            self.pool.append(dict(front=select_front(front, index), steps=steps[index], limit=chain['limit'][index],
                                  reach=reach[index], issued=it,
                                  chains=[dict(meta=b['meta'][i], shift=float(shift[i]), front_lo=lo[i], front_hi=hi[i])
                                          for i in index.tolist()]))
        while sum(len(state['chains']) for state in self.pool) >= tc.batch:
            self._publish(tc.batch)
        self.stats['chain_pool'] = torch.tensor(float(sum(len(state['chains']) for state in self.pool)))

    def _publish(self, size: int) -> None:
        """Queue a job for the oldest ``size`` pooled chains."""
        take, count = [], 0
        while count < size:
            state = self.pool.pop(0)
            n = min(len(state['chains']), size-count)
            if n < len(state['chains']):  # split: the rest stays pooled
                rest = torch.arange(n, len(state['chains']), device=state['steps'].device)
                self.pool.insert(0, self._part(state, rest))
                state = self._part(state, torch.arange(n, device=state['steps'].device))
            take.append(state)
            count += n
        job = next(self.ids)
        try:
            self.jobs.put_nowait(dict(id=job, min_valid=self.cfg.train.chain_min_valid,
                                      chains=[c for state in take for c in state['chains']]))
        except queue.Full:
            return
        self.pending[job] = dict(front=concat_fronts([s['front'] for s in take]),
                                 steps=torch.cat([s['steps'] for s in take]), limit=torch.cat([s['limit'] for s in take]),
                                 reach=torch.cat([s['reach'] for s in take]), issued=min(s['issued'] for s in take))

    @staticmethod
    def _part(state: dict, index: torch.Tensor) -> dict:
        return dict(front=select_front(state['front'], index), steps=state['steps'][index],
                    limit=state['limit'][index], reach=state['reach'][index], issued=state['issued'],
                    chains=[state['chains'][i] for i in index.tolist()])
