"""Explicit, narrow checkpoint transition; ordinary restoration remains strict."""
import math
from copy import deepcopy

import torch

from vesuvius.neural_tracing.nets.vesuvius_unet3d import Trilinear3dUpsample


def migrate_output_stride(payload, model, optimizer):
    """Explicit decoder truncation; keep canonical named Adam state, not IDs."""
    from .model import build_fiber_trace_3d_model
    source_config = payload.get('config', {})
    old_stride = int(source_config.get('model_3d', {}).get('output_stride', 1))
    new_stride = int(model.output_stride)
    if old_stride == new_stride:
        return payload
    if new_stride <= old_stride or new_stride % old_stride:
        raise ValueError('output_stride migration only supports coarsening by an integer factor')
    with torch.random.fork_rng(devices=[]):
        source = build_fiber_trace_3d_model(source_config)
    old, current = payload['model'], model.state_dict()
    source_state = source.state_dict()
    if set(old) != set(source_state) or any(old[k].shape != source_state[k].shape for k in old):
        raise ValueError('checkpoint does not match its saved model config')
    retained = [k for k in current if k in old and old[k].shape == current[k].shape]
    reset = [k for k in current if k not in retained]
    removed = [k for k in old if k not in current]
    if any(not k.startswith('net.decoder.') for k in reset + removed):
        raise ValueError('output_stride migration cannot change non-decoder tensors')
    # Even same-shaped encoder tensors must describe the same architecture.
    for name, module in source.named_modules():
        other = dict(model.named_modules()).get(name)
        if not name.startswith('net.decoder') and (other is None or type(other) is not type(module)):
            raise ValueError(f'non-decoder architecture changed: {name}')
    if optimizer is not None and 'optimizer' in payload:
        old_names = list(dict(source.named_parameters()))
        named = dict(model.named_parameters())
        groups = payload['optimizer']['param_groups']
        if (len(groups) != 1 or len(optimizer.param_groups) != 1
                or len(groups[0]['params']) != len(old_names)
                or [id(p) for p in optimizer.param_groups[0]['params']] != [id(p) for p in named.values()]):
            raise ValueError('stride migration requires canonical single-group optimizer ordering')
        old_ids = dict(zip(old_names, groups[0]['params'], strict=True))
        target_ids = optimizer.state_dict()['param_groups'][0]['params']
        state = {}
        for name, new_id in zip(named, target_ids, strict=True):
            if name in retained and old_ids[name] in payload['optimizer']['state']:
                state[new_id] = payload['optimizer']['state'][old_ids[name]]
        group = deepcopy(groups[0])
        group['params'] = target_ids
        payload['optimizer'] = {'state': state, 'param_groups': [group]}
    payload['model'] = {k: old[k] if k in retained else current[k].detach().clone() for k in current}
    for track in payload.get('ema', {}).get('tracks', {}).values():
        shadow = track['model']
        if set(shadow) != set(old) or any(shadow[k].shape != old[k].shape for k in old):
            raise ValueError('EMA/source mismatch during stride migration')
        track['model'] = {k: shadow[k] if k in retained else current[k].detach().clone() for k in current}
        track['best_metric'] = math.inf
    payload['decoder_migration'] = {'source_stride': old_stride, 'target_stride': new_stride,
                                    'retained': retained, 'reinitialized': reset, 'removed': removed}
    print(f'fiber stride migration {old_stride}->{new_stride}: retained={len(retained)} '
          f'reinitialized={reset} removed={removed}', flush=True)
    return payload


def migrate_pixelshuffle(payload, model, optimizer):
    source_config = payload.get('config', {})
    source_mode = source_config.get('model_3d', source_config.get('model', {})).get('decoder_upsample_mode')
    modules = [(name, module) for name, module in model.named_modules()
               if isinstance(module, Trilinear3dUpsample)]
    if not modules:
        raise ValueError('decoder migration requires a trilinear target model')
    if source_mode == 'trilinear':
        return payload  # Already migrated; ordinary strict restore follows.
    if source_mode != 'pixelshuffle':
        raise ValueError('decoder migration requires an explicit pixelshuffle source config')
    saved, current = payload['model'], model.state_dict()
    if set(saved) != set(current):
        raise ValueError('decoder migration cannot change model state keys')
    changed = {}
    for name, module in modules:
        factor = math.prod(module.scale)
        if factor == 1:
            continue
        for suffix in module.proj.state_dict():
            key = f'{name}.proj.{suffix}'
            expected = (current[key].shape[0] * factor, *current[key].shape[1:])
            if tuple(saved[key].shape) != expected:
                raise ValueError(f'unexpected pixelshuffle projection shape: {key}')
            changed[key] = current[key].detach().clone()
    for key in saved:
        if key not in changed and saved[key].shape != current[key].shape:
            raise ValueError(f'non-upsampler mismatch during decoder migration: {key}')
    # The supported trainer uses one AdamW group in named-parameter order.
    # Validate both state-dict order and actual optimizer identity before mapping
    # serialized optimizer IDs; do not silently accept custom grouping/order.
    if optimizer is not None and 'optimizer' in payload:
        named = dict(model.named_parameters())
        order = list(named)
        old_order = [key for key in saved if key in named]
        groups = payload['optimizer']['param_groups']
        if (old_order != order or len(groups) != 1 or len(optimizer.param_groups) != 1
                or len(groups[0]['params']) != len(order)
                or [id(p) for p in optimizer.param_groups[0]['params']] != [id(named[k]) for k in order]):
            raise ValueError('decoder migration requires matching canonical optimizer parameter order')
        for key, parameter_id in zip(order, groups[0]['params'], strict=True):
            if key in changed:
                payload['optimizer']['state'].pop(parameter_id, None)
    saved.update(changed)
    for track in payload.get('ema', {}).get('tracks', {}).values():
        if set(track['model']) != set(current):
            raise ValueError('EMA keys differ during decoder migration')
        for key, value in changed.items():
            track['model'][key] = value.to(dtype=track['model'][key].dtype).clone()
        track['best_metric'] = math.inf
    payload['decoder_migration'] = {'source': source_mode, 'target': 'trilinear',
                                    'reinitialized': list(changed)}
    print('fiber decoder migration: retained compatible model/optimizer/EMA; '
          f'reinitialized {len(changed)} tensors: {list(changed)}', flush=True)
    return payload
