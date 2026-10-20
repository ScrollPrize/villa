"""Normalize legacy recipes once, before constructing data, methods or runtime."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from dinovol_2.model.backbone import materialize_backbone_config

DATA_KEYS = {
    'dataset', 'val_dataset', 'monitor_dataset', 'batch_size', 'num_workers',
    'dataloader-workers', 'dataloader_workers', 'prefetch_factor', 'prefetch-factor',
    'persistent_workers', 'persistent-workers', 'shuffle', 'seed', 'epoch_length',
    'official_epoch_length',
}
OPTIMIZER_KEYS = {
    'lr', 'min_lr', 'lr_schedule', 'weight_decay', 'weight_decay_end', 'betas',
    'clip_grad', 'layer_decay', 'layerwise_decay', 'patch_embed_lr_mult',
    'warmup_steps', 'warmup_ratio', 'fused_optimizer',
}
DINO_KEYS = {
    'dino_loss_weight', 'ibot_loss_weight', 'koleo_loss_weight', 'koleo_distributed',
    'koleo_loss_group_size', 'centering', 'teacher_temp', 'warmup_teacher_temp',
    'warmup_teacher_temp_steps', 'momentum_teacher', 'final_momentum_teacher',
    'freeze_last_layer_steps', 'freeze_last_layer_epochs', 'freeze_last_layer_ratio',
    'mask_ratio_min_max', 'mask_sample_probability', 'ibot_fused_last_layer',
    'ibot_masked_loss_chunk_size', 'ibot_kernel_backend', 'ibot_targets',
    'gram', 'point_supervision', 'object_losses',
}
SECTIONS = ('data', 'backbone', 'method', 'optimizer', 'runtime')


def normalize_config(config: Mapping[str, Any], *, legacy_checkpoint=False) -> dict[str, Any]:
    """Accept flat recipes or the five sections. Explicit section values win.

    New recipes default to current DINO/v3 behavior. Older checkpoint inference
    can request the historical unnamed architecture; explicit v2 presets are
    expanded at this boundary so architecture never controls the trainer.
    """
    source = deepcopy(dict(config))
    result = {name: {} for name in SECTIONS}
    legacy_model = dict(source.get('model') or {})
    for key, value in source.items():
        if key in SECTIONS or key in {'model', 'recipes'}:
            continue
        section = 'data' if key in DATA_KEYS else 'optimizer' if key in OPTIMIZER_KEYS else 'method' if key in DINO_KEYS else 'runtime'
        result[section][key] = value
    legacy_heads = {}
    for key, value in legacy_model.items():
        if key.startswith(('dino_', 'ibot_')) or key == 'dense_head':
            legacy_heads[key] = value
        elif key.startswith('pretrained_'):
            result['method'][key] = value
        else:
            result['backbone'][key] = value
    if legacy_heads:
        result['method']['heads'] = legacy_heads
    for section in SECTIONS:
        value = source.get(section, {})
        if not isinstance(value, Mapping):
            raise ValueError(f'{section} must be a mapping')
        result[section].update(value)
    result['method'].setdefault('name', 'dino')

    backbone = result['backbone']
    preset = str(backbone.get('model_type', '')).strip().lower()
    historical = preset == 'v2' or (legacy_checkpoint and not preset and 'backbone' not in source)
    if legacy_checkpoint and not preset and 'backbone' not in source:
        # Saved before the current defaults, with no named architecture preset.
        from dinovol_2.model.backbone import _LEGACY_BACKBONE_DEFAULTS
        backbone = {**_LEGACY_BACKBONE_DEFAULTS, **backbone}
    if historical:
        result['optimizer'].setdefault('lr_schedule', 'cosine')
        result['optimizer'].setdefault('weight_decay_end', 0.4)
        result['method'].setdefault('momentum_teacher', 0.992)
        result['method'].setdefault('final_momentum_teacher', 1.0)
        result['method'].setdefault('koleo_loss_group_size', None)
    dataset = result['data'].get('dataset') or {}
    for model_key, data_key in [('global_crops_size', 'global_crop_size'), ('local_crops_size', 'local_crop_size')]:
        if data_key in dataset:
            backbone.setdefault(model_key, dataset[data_key])
    result['backbone'] = materialize_backbone_config(backbone)
    result['backbone'].pop('model_type', None)
    # Resolve recipe defaults here, independently of the architecture.
    optimizer = result['optimizer']
    optimizer.setdefault('lr_schedule', 'constant')
    optimizer.setdefault('weight_decay_end', optimizer.get('weight_decay', 0.04))
    if result['method']['name'] == 'dino':
        method = result['method']
        method.setdefault('momentum_teacher', 0.994)
        method.setdefault('final_momentum_teacher', method['momentum_teacher'])
        method.setdefault('koleo_loss_group_size', 16)
    return result


def dino_model_config(config: Mapping[str, Any]) -> dict[str, Any]:
    method = config['method']
    return {**config['backbone'], **method.get('heads', {}),
            **{key: value for key, value in method.items() if key.startswith('pretrained_')}}


def as_legacy_config(config: Mapping[str, Any], *, legacy_checkpoint=False) -> dict[str, Any]:
    """Adapter for standalone data/kernel tools that still consume flat recipes."""
    resolved = normalize_config(config, legacy_checkpoint=legacy_checkpoint)
    return {**resolved['runtime'], **resolved['optimizer'], **resolved['data'],
            **{k: v for k, v in resolved['method'].items() if k not in {'name', 'heads'}},
            'model': dino_model_config(resolved)}
