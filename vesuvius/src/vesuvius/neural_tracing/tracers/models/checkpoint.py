"""Model checkpoints: reading and writing them (recording the model type), building a model from one, and
initializing a model from another checkpoint's weights."""
from __future__ import annotations

import copy
import dataclasses
import itertools
import operator
import re

import torch

from vesuvius.neural_tracing.tracers.models.model import MODEL_TYPES, RETIRED_FIELDS, build_model, config_class, config_from_checkpoint
from vesuvius.neural_tracing.tracers.shared.crop_sampling.volume import CTVolumeSpec
from vesuvius.neural_tracing.tracers.shared.io import atomic_torch_save
from vesuvius.neural_tracing.tracers.shared.paths import relocate_checkpoint
from vesuvius.neural_tracing.tracers.models.weights import conv_memory_format, load_matching_weights


def write_checkpoint(path, model, vol_spec, crop, n_history, model_type, extra=None):
    atomic_torch_save(dict(model_type=model_type, model=model.state_dict(),
                           model_cfg=model.cfg.to_dict(), crop=dataclasses.asdict(crop),
                           n_history=n_history, vol_spec=vol_spec.to_dict(), **(extra or {})), path)


def read_checkpoint(path, device='cuda'):
    ck = torch.load(path, map_location=device, weights_only=False)
    model_type = ck.get('model_type')
    # A pre-cleanup 'sequence' checkpoint keeps its type name but records model fields that no longer exist.
    unknown = (sorted(set(ck.get('model_cfg', {}))-{f.name for f in dataclasses.fields(config_class(model_type))}-set(RETIRED_FIELDS))
               if model_type in MODEL_TYPES else None)
    if model_type not in MODEL_TYPES or unknown:
        detail = f'model fields {", ".join(unknown)}' if unknown else f'model type {model_type!r}'
        raise ValueError(f'Unsupported checkpoint {path} ({detail}); checkpoints from before the model cleanup need '
                         'scripts/convert_checkpoint.py')
    return ck


def cnn_stage_map(old, new):
    """Crop-CNN stage index of configuration ``old`` -> the stage of ``new`` at the same stride and width (a stem or
    full-resolution stage added in front shifts the indices); stages without a counterpart are absent."""
    def stages(cfg):
        strides = getattr(cfg, 'cnn_stage_strides', None)
        if strides is None:
            return []
        return list(zip(itertools.accumulate(strides, operator.mul), cfg.cnn_channels))
    target = {stage: k for k, stage in enumerate(stages(new))}
    return {k: target[stage] for k, stage in enumerate(stages(old)) if stage in target}


def remap_cnn_stages(state, mapping):
    """``state`` with ``cnn.stages.K.`` tensors renamed to their ``mapping[K]`` stage; unmapped stages are dropped."""
    out = {}
    for key, value in state.items():
        match = re.match(r'cnn\.stages\.(\d+)\.', key)
        if match is None:
            out[key] = value
        elif int(match.group(1)) in mapping:
            out[f'cnn.stages.{mapping[int(match.group(1))]}.{key[match.end():]}'] = value
    return out


def quiet_fresh_blocks(model, fresh):
    """Residual blocks added to a loaded crop-CNN stage (entirely at initialization per ``fresh`` tensor names, in a
    stage with loaded weights) whose skip is the identity start with a zero residual branch (last norm's scale and
    shift), so they first pass their input on (up to the block's final activation)."""
    if not hasattr(model, 'cnn'):
        return
    fresh = set(fresh)
    for s, stage in enumerate(model.cnn.stages):
        if all(f'cnn.stages.{s}.{n}' in fresh for n, _ in stage.named_parameters()):
            continue  # a stage without loaded weights keeps the usual initialization
        for b, block in enumerate(stage.blocks):
            names = [f'cnn.stages.{s}.blocks.{b}.{n}' for n, _ in block.named_parameters()]
            if isinstance(block.skip, torch.nn.Module) or not all(n in fresh for n in names):
                continue
            with torch.no_grad():
                block.conv2.norm.weight.zero_()
                block.conv2.norm.bias.zero_()


def initialize_model_weights(cfg, device, initial=None, exclude=()):
    """Weight-only initialization (model and EMA tensors matched by name and shape); never inherits optimizer,
    replay or run settings. A checkpoint with another crop-CNN layout gives each stage to the stage at its stride and
    width (``cnn_stage_map``); its residual blocks beyond the checkpoint's start as pass-throughs (``quiet_fresh_blocks``)."""
    model = build_model(cfg).to(device, memory_format=conv_memory_format(device))
    report = None
    states = None
    if initial is not None:
        states = [initial['model'], initial.get('ema', initial['model'])]
        if 'model_cfg' in initial:
            mapping = cnn_stage_map(config_from_checkpoint(initial), cfg)
            if any(k != v for k, v in mapping.items()) or len(mapping) != len(getattr(cfg, 'cnn_channels', ())):
                states = [remap_cnn_stages(state, mapping) for state in states]
        report = load_matching_weights(model, states[0], exclude)
        quiet_fresh_blocks(model, report['fresh'])
    ema = copy.deepcopy(model).requires_grad_(False).eval()
    if initial is not None:
        load_matching_weights(ema, states[1], exclude)
    model.initialization_report = report
    return model, ema


def load_checkpoint(path,device='cuda'):
    # Collection shares a GPU with training. Keep duplicate weights and any
    # optimizer state on the CPU; only the EMA inference model needs VRAM.
    ck = relocate_checkpoint(read_checkpoint(path,'cpu'))
    cfg = config_from_checkpoint(ck)
    model = build_model(cfg).to(device,memory_format=conv_memory_format(device))
    model.load_state_dict(ck['ema'])
    model.eval()
    return model,cfg.fine,cfg.n_history,CTVolumeSpec.from_dict(ck['vol_spec']),ck
