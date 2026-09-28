"""Consecutive supervised heads sharing one causal observation sequence.

Only tracks with explicit departure/offset labels supply extra decisions. The
original sampled state (including matched candidate pairs) is always retained.
Images are views of the original batch: no extra crop reads or encoder caching
across optimizer updates. Annotation-derived correspondence is target-only.
"""
import numpy as np
import torch

from vesuvius.neural_tracing.fiber_follow.shared.data import SampleConfig, collate_targets, label_state
from vesuvius.neural_tracing.fiber_follow.shared.geometry import arclength, interp_at
from vesuvius.neural_tracing.fiber_follow.shared.reference import SEED_FIELDS
from .memory_data import memory_layout


def correspondence(fiber, point):
    """Project an already labeled original-fiber point onto its polyline."""
    edges = np.diff(fiber.points,axis=0)
    fraction = np.clip(((point-fiber.points[:-1])*edges).sum(-1)/np.maximum((edges*edges).sum(-1),1e-12),0,1)
    projected = fiber.points[:-1]+fraction[:,None]*edges
    index = int(np.square(projected-point).sum(-1).argmin())
    return float(fiber.s[index]+fraction[index]*(fiber.s[index+1]-fiber.s[index]))


def prepare_window(builder, item, fiber):
    cfg = builder.cfg
    if cfg.trajectory_window == 1 or item.get('_window_child') or item.get('source') == 5:
        return
    track = item.get('memory_track')
    if track is None:
        return
    layout,_ = memory_layout(item,cfg)
    sample = SampleConfig(crop=cfg.fine,n_history=cfg.n_history,recent_history_points=cfg.n_history,
                          n_future=cfg.n_future,future_step=cfg.future_step)
    children = []
    for j in range(len(layout)-2,max(-1,len(layout)-cfg.trajectory_window-1),-1):
        obs = layout[j]
        index = obs.get('track')
        if index is None:
            break
        off = float(track['offtrack'][index])
        offset = np.asarray(track['offset'][index])
        if not np.isfinite(off) or (off < .5 and not np.isfinite(offset).all()):
            break
        path = np.asarray(track['pos'][:index+1],dtype=np.float64)
        if not np.isfinite(path).all():
            break
        arc = arclength(path)
        back = arc[-1]-np.arange(1,cfg.n_history+1)
        history = interp_at(path,arc,np.clip(back,0,arc[-1]))
        fi,_,reverse = item['fiber_ref']
        target = obs['pos']+offset if off < .5 else obs['pos']
        t = correspondence(fiber,target)
        child = label_state(fiber,obs['pos'],obs['frame'],history,back>=0,sample,
                            t=t,reverse=reverse,offtrack=off>=.5)
        child.update({key:item[key] for key in SEED_FIELDS if key in item})
        remaining = arclength(np.concatenate((np.asarray(track['pos'][index:]),np.asarray(item['pos'])[None])))[-1]
        child['seed_age'] = max(0.,float(item.get('seed_age',0.))-float(remaining))
        child.update(fiber_ref=(fi,fiber.length-t if reverse else t,reverse),
                     source=item.get('source',0),source_step=item.get('source_step',-1),stratum=item.get('stratum',-1),
                     location_source=item.get('location_source',0),_window_child=True,
                     _window_index=cfg.memory_steps+1-len(layout)+j,
                     memory_track={key:value[:index] for key,value in track.items()})
        # Prepare supervision using a separate deterministic stream. Appearance
        # augmentation is inherited from the parent's shared crop buffer below.
        builder.prepare(child,fiber,np.random.default_rng(item['identity_seed']+j+1))
        children.append(child)
    if children:
        item['_trajectory_children'] = children[::-1]


def slice_batch(value, row):
    if isinstance(value,dict):
        return {key:slice_batch(v,row) for key,v in value.items() if key != 'trajectory_windows'}
    return value[row:row+1]


def build_windows(builder, items, batch):
    windows = []
    for row,item in enumerate(items):
        parent = slice_batch(batch,row)
        x = parent['x']
        children = item.get('_trajectory_children',())
        window = []
        for child in children:
            index = child['_window_index']
            stack = lambda key: torch.as_tensor(np.asarray(child[key],np.float32)[None])
            inputs = dict(fine=x['history_crops'][:,index],seed=stack('visible_seed'),
                seed_mask=stack('visible_seed_mask'),seed_age=stack('visible_seed_age'),seed_tangent=stack('visible_seed_tangent'),
                seed_crop=x['seed_crop'],memory_seed_valid=x['memory_seed_valid'],
                memory_seed_position=x['memory_seed_position'],memory_seed_frame=x['memory_seed_frame'])
            start = index if window else 0
            for key in ('memory_mask','memory_positions','memory_frames'):
                inputs[key] = x[key][:,start:index+1]
            inputs['history_crops'] = x['history_crops'][:,start:index]
            decision = dict(x=inputs,hist=stack('hist_local'),hmask=stack('hmask'),**collate_targets([child]))
            decision = builder.supervise([child],decision)
            decision['presence_dropped'] = parent['presence_dropped']
            # Probe targets must use the shortened streamed input schedule.
            for key in ('memory_target_identity','memory_target_identity_mask','memory_target_offset','memory_target_offset_mask'):
                if key in decision:
                    decision[key] = decision[key][:,-inputs['memory_mask'].shape[1]:]
            window.append(decision)
        if window:
            parent['x'] = dict(x,history_crops=x['history_crops'][:,:0])
            for key in ('memory_mask','memory_positions','memory_frames'):
                parent['x'][key] = x[key][:,-1:]
            for key in ('memory_target_identity','memory_target_identity_mask','memory_target_offset','memory_target_offset_mask'):
                if key in parent:
                    parent[key] = parent[key][:,-1:]
        window.append(parent)
        windows.append(window)
    return windows
