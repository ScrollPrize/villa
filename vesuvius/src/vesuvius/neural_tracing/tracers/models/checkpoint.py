"""Model checkpoints: reading and writing them (recording the model type), building a model from one, and
initializing a model from another checkpoint's weights."""
from __future__ import annotations

import copy
import dataclasses

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


def initialize_model_weights(cfg, device, initial=None, exclude=()):
    """Weight-only initialization (model and EMA tensors matched by name and shape); never inherits optimizer,
    replay or run settings."""
    model = build_model(cfg).to(device, memory_format=conv_memory_format(device))
    report = None
    if initial is not None:
        report = load_matching_weights(model, initial['model'], exclude)
    ema = copy.deepcopy(model).requires_grad_(False).eval()
    if initial is not None:
        load_matching_weights(ema, initial.get('ema', initial['model']), exclude)
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
