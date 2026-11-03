"""Opt-in statistical stationarity check; never changes optimization or LR."""
from collections import Counter
import math
import numpy as np


class ConvergenceMonitor:
    def __init__(self, config):
        self.block_steps = int(config.get('block_steps', 1000))
        self.window = int(config.get('window', 5))
        self.minimum = int(config.get('minimum_steps', 10000))
        if self.block_steps < 1 or self.window < 2 or self.minimum < 0:
            raise ValueError('Invalid convergence settings')
        self.pending = []; self.blocks = []; self.reason = None

    def observe(self, row):
        if row['iteration'] == 0:
            return False
        self.pending.append(row)
        if row['iteration'] % self.block_steps:
            return False
        records = self.pending; self.pending = []
        signatures = [tuple(r['bucket_uv_meeting_events']) for r in records]
        mode, count = Counter(signatures).most_common(1)[0]
        block = dict(step=row['iteration'], loss=float(np.median([r['loss'] for r in records])),
                     events=list(mode), membership_fraction=count/len(records), metrics={})
        keys = [('bucket_uv_local_u_p0.5_max', 1.), ('bucket_uv_local_v_p0.5_max', 1.),
                ('direction_bucket_angle_max', .1), ('direction_intermediate_angle_max', .1),
                ('direction_shared_angle_max', .1), ('shared_curve_length_relative_max', .001)]
        for key, floor in keys:
            values = [r.get(key) for r in records]
            block['metrics'][key] = (float(np.median(values)) if all(
                x is not None and math.isfinite(x) for x in values) else None, floor)
        for key in ('bucket_uv_length', 'intermediate_uv_length'):
            block['metrics'][key] = (float(np.median([r['terms'].get(key, 0.) for r in records])), 1e-5)
        self.blocks.append(block)
        if len(self.blocks) <= self.window or row['iteration'] < self.minimum:
            return False
        recent = self.blocks[-self.window:]; prior = self.blocks[:-self.window]
        if not mode or any(b['events'] != list(mode) or b['membership_fraction'] < .9 for b in recent):
            return False
        if tuple(row['bucket_uv_meeting_events']) != mode:
            return False
        losses = [b['loss'] for b in recent]; best = min(b['loss'] for b in prior)
        if min(losses) < best - max(1e-4, .01*abs(best)):
            return False
        if max(losses)-min(losses) > max(1e-4, .05*abs(float(np.median(losses)))):
            return False
        for key in block['metrics']:
            values = [b['metrics'][key][0] for b in recent]
            if any(x is None or not math.isfinite(x) for x in values):
                return False
            floor = block['metrics'][key][1]
            if max(values)-min(values) > max(floor, .05*abs(float(np.median(values)))):
                return False
        self.reason = 'stationary_loss_geometry_and_meeting_membership'
        return True
