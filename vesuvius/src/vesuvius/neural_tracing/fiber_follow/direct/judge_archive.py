"""One observed path per trace, with row cutoffs and versioned event metadata."""
import json
from pathlib import Path
import numpy as np
from ..events import EVENT_VERSION, label_path, truncate_path

ARCHIVE_VERSION = 'judge_paths_v1'


def save_archive(path, traces):
    arrays, metadata = {}, []
    for i, trace in enumerate(traces):
        arrays[f'path_{i}'] = np.asarray(trace['path'], np.float64)
        arrays[f'frame_{i}'] = np.asarray(trace['seed_frame'], np.float64)
        arrays[f'correspondence_{i}'] = np.asarray(trace.get('event_samples', np.empty((0,3))), np.float64)
        arrays[f'valid_{i}'] = np.ones(len(trace['path']), bool)
        from .judge_slices import transported_frames
        from ..geometry import arclength, interp_at
        arc = arclength(trace['path'])
        regular = np.arange(0, arc[-1]+1e-8, trace.get('path_step', 4.))
        arrays[f'arcs_{i}'] = regular
        arrays[f'centers_{i}'] = interp_at(trace['path'], arc, regular)
        arrays[f'frames_{i}'] = transported_frames(trace['path'], trace['seed_frame'], regular)
        metadata.append({k: v for k, v in trace.items() if k not in ('path', 'seed_frame', 'event_samples')})
    temporary = Path(path).with_suffix('.partial.npz')
    np.savez_compressed(temporary, metadata=json.dumps(dict(version=ARCHIVE_VERSION, event_version=EVENT_VERSION, traces=metadata)), **arrays)
    temporary.replace(path)


class PathArchive:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data['metadata']))
            if meta['version'] != ARCHIVE_VERSION or meta['event_version'] != EVENT_VERSION:
                raise ValueError('Incompatible judge path archive')
            self.traces = [dict(t, path=data[f'path_{i}'].copy(), seed_frame=data[f'frame_{i}'].copy())
                           for i, t in enumerate(meta['traces'])]

    def context(self, index, cutoff, fiber, revision=-1):
        t = self.traces[index]
        audit = None
        if revision >= 0:
            audit = t['audits'][revision]
            if audit['endpoint'] > cutoff+1e-7:
                raise ValueError('Replay ledger revision is after its observation cutoff')
        return dict(path=truncate_path(t['path'], cutoff), complete_path=t['path'], seed_frame=t['seed_frame'],
                    q0=t['q0'], annotation=fiber.points[::-1] if t['reverse'] else fiber.points,
                    physical_end=fiber.endpoint_stop[0 if t['reverse'] else 1], policy_audit=audit)
