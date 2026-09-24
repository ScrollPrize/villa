"""Held-out decision audit using the same progress-bounded GT matcher as DAgger."""
from collections import defaultdict

import numpy as np
import torch

from vesuvius.neural_tracing.fiber_follow.collect import DecisionCollector
from vesuvius.neural_tracing.fiber_follow.data import SampleConfig, label_state
from vesuvius.neural_tracing.fiber_follow.history_metrics import TANGENT_POINTS, cleaning_measurements
from vesuvius.neural_tracing.fiber_follow.supervision import candidate_labels


class HistoryAudit:
    """Observes decisions without changing the trace or supplying GT to the model.

    Censors unknown annotation ends and stops auditing after the collector's
    short departure suffix. Off-track cleaning has no GT position supervision;
    those states report correction sizes and rejection outcomes only.
    """
    def __init__(self, tracer, tolerance=1.5):
        cfg = tracer.model.cfg
        self.cfg = SampleConfig(crop=tracer.crop, n_history=tracer.n_history, clean_points=cfg.clean_points,
                                n_future=cfg.n_future, future_step=cfg.future_step, n_candidates=cfg.n_candidates)
        self.tangent_points, self.tolerance = TANGENT_POINTS, tolerance
        self.threshold = tracer.p.confidence
        self.counts = defaultdict(int)
        self.sums = defaultdict(lambda: defaultdict(float))
        self.valid = defaultdict(lambda: defaultdict(int))
        self.censored_traces = 0

    def start_batch(self, fibers, seeds):
        self.collectors = [DecisionCollector(fibers[s['fiber']], s['fiber'], s['t'], s['sign'], self.cfg)
                           for s in seeds]
        self.active = [True] * len(seeds)

    def __call__(self, index, state):
        if not self.active[index]:
            return
        collector = self.collectors[index]
        if collector(state) is False:
            self.active[index] = False
            self.censored_traces += 1
            return  # auditing never stops the actual rollout
        row = collector.rows[-1]
        item = label_state(collector.fiber, state['pos'], state['frame'], state['hist'], state['hmask'],
                           self.cfg, t=collector.t, reverse=collector.sign < 0, offtrack=row['offtrack'])
        tensor = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32)[None]
        batch = {k: tensor(item[k]) for k in ('dense_ab', 'dense_mask', 'endpoint_known', 'end_local', 'offtrack')}
        labels, masks, _, _ = candidate_labels(tensor(state['candidates']), batch, self.tolerance)
        chosen = state['chosen']
        known = bool(masks[0, chosen, 0])
        correct = bool(labels[0, chosen, 0])
        gate_open = bool(state['confidence'][chosen, 0] >= self.threshold)
        groups = ['all', 'offtrack' if row['offtrack'] else 'ontrack']
        if known:
            groups.append('correct_first' if correct else 'wrong_first')
            if correct and not gate_open:
                groups.append('false_stop_first')
            if not correct and gate_open:
                groups.append('false_go_first')
            if not row['offtrack'] and masks[0, :, 0].all() and not labels[0, :, 0].any():
                groups.append('no_correct_candidate_first')
        measurements = cleaning_measurements(tensor(state['clean_history']), tensor(item['hist_local']),
                                             tensor(item['hmask']), tensor(item['clean_local']),
                                             tensor(item['clean_mask']), self.tangent_points)
        if not row['offtrack'] and measurements['observed_current_error'][0].item() > self.tolerance:
            groups.append('recoverable_drift')
        for group in groups:
            self.counts[group] += 1
            for name, (value, valid) in measurements.items():
                if valid.item():
                    self.sums[group][name] += value.item()
                    self.valid[group][name] += 1
        # The audit does not need the collector's replay/pre-failure buffer.
        collector.rows.clear()
        collector.distances.clear()

    def summary(self):
        return dict(confidence=self.threshold, tolerance=self.tolerance, censored_traces=self.censored_traces,
                    groups={group: dict(states=count,
                                        metrics={name: dict(mean=total/self.valid[group][name],
                                                            count=self.valid[group][name])
                                                 for name, total in self.sums[group].items()})
                            for group, count in self.counts.items()})
