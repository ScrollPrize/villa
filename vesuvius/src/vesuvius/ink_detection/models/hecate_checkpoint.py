"""The checkpoint boundary shared by Hecate training and the Hub runtime.

Architecture code stays in canonical_projection; exported weights load directly
with https://huggingface.co/scrollprize/hecate/blob/main/hecate.py.
"""
from pathlib import Path
import os

import torch

from .canonical_projection import CanonicalLogitProjection
from .checkpoint import load_model_state


def model_config(sampling_um, *, training=False):
    if sampling_um not in (2.4, 9.6):
        raise ValueError("Hecate sampling must be 2.4 or 9.6 um")
    fine = sampling_um == 2.4
    return {
        'model_type': 'multiteacher_3d_projection', 'mode': 'flat', 'in_channels': 1,
        'targets': {'ink': {'activation': 'none', 'out_channels': 1}},
        'projection': {'kind': 'canonical_logits'},
        'model_config': {'projection': {'kind': 'canonical_logits'}, 'canonical': {
            'with_norm': True, 'freeze_batchnorm_stats': not training,
            'input_depth': 64 if fine else 16, 'depth_margin': 1 if fine else 0,
            'native_xy_stride': 4 if fine else 1,
            'native_refinement': False, 'activation_checkpointing': False}},
        'patch_size': [64, 256, 256] if fine else [16, 64, 64],
        'image_normalization': {'mode': 'divide', 'divisor': 200. if fine else 255.},
        'patch_overlap': .5, 'patch_min_labeled_coverage': .05,
        'task': 'hecate_inference', 'architecture_family': 'hecate',
        'datasets': [{'segments_path': '.', 'volume_scale': '0'}],
    }


def sampling_from_config(config):
    shape = tuple(config['patch_size'])
    if shape not in ((64, 256, 256), (16, 64, 64)):
        raise ValueError('Unsupported Hecate patch size')
    sampling = 2.4 if shape[0] == 64 else 9.6
    expected = model_config(sampling)
    if config['image_normalization'] != expected['image_normalization']:
        raise ValueError('Normalization does not match the released Hecate runtime')
    canonical = config['model_config']['canonical']
    defaults = model_config(2.4)['model_config']['canonical']
    for key in ('with_norm', 'input_depth', 'depth_margin', 'native_xy_stride', 'native_refinement'):
        if canonical.get(key, defaults[key]) != expected['model_config']['canonical'][key]:
            raise ValueError(f'Incompatible Hecate architecture: {key}')
    if config.get('projection', {}).get('kind') != 'canonical_logits':
        raise ValueError('Require the shared canonical logit projection')
    return sampling


def load_hecate(checkpoint, device='cpu', *, training=False):
    """Load released or trusted historical EMA weights with strict tensor matching.

    Full optimizer checkpoints contain Python RNG objects: load them only from
    trusted sources. Hub releases use a tensor-only payload.
    """
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    sampling = sampling_from_config(payload['config'])
    if payload.get('architecture_family') == 'hecate':
        if payload.get('weight_source') != 'ema_model' or payload.get('sampling_um') != sampling:
            raise ValueError('Invalid Hecate release metadata')
    elif payload['config'].get('task') == 'resolution_distillation':
        from ..training.ema_batchnorm import verify_calibrated_ema
        verify_calibrated_ema(payload)
    config = model_config(sampling, training=training)
    model = CanonicalLogitProjection(**config['model_config']['canonical'])
    load_model_state(model, payload['ema_model'])
    model.requires_grad_(training)
    # These legacy tensors exist for strict loading but never enter the loss.
    for name, parameter in model.named_parameters():
        if name.startswith(('canonical.backbone.fc.', 'canonical.decoder.aux_head_')):
            parameter.requires_grad_(False)
    model.train(training).to(device)
    return model, config


def export_ema(checkpoint, output):
    """Write only EMA tensors and a whitelist of public architecture metadata."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    model, config = load_hecate(checkpoint)
    sampling = sampling_from_config(config)
    payload = {'format_version': 1, 'architecture_family': 'hecate',
               'sampling_um': sampling, 'weight_source': 'ema_model',
               'config': config, 'ema_model': model.state_dict()}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.partial')
    torch.save(payload, temporary)
    os.replace(temporary, output)
    return output
