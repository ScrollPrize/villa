"""One-off v17 -> coordinate regression weight conversion (no runtime legacy loader).

Keeps model/EMA tensors and inference/data provenance; drops optimizer, RNG and
run progress. The result supports inference and fresh --init-weights training,
not --resume. The source checkpoint is never modified.
"""
import argparse
from dataclasses import fields
from pathlib import Path

import torch

from vesuvius.neural_tracing.fiber_follow.models.model import CoordinateRegressionConfig, build_model
from vesuvius.neural_tracing.fiber_follow.data.ct_normalization import ZSCORE_METHOD


def convert(source, destination):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    ck = torch.load(source, map_location='cpu', weights_only=False)
    if ck.get('architecture') != 'axial_patch4_residual_stem_tokens_fiber_slabs_v17':
        raise ValueError('This one-off converter accepts only the v17 residual-stem checkpoint')
    old = dict(ck['model_cfg'])
    expected = dict(direction_inputs=False, input_mode='ct', encoder='patch4', token_only=True,
                    history_encoder='fine', history_path_tokens=True, path_geometry_tokens=True)
    for key, value in expected.items():
        if old.pop(key, None) != value:
            raise ValueError(f'Unsupported source configuration: {key}')
    old.pop('channels')  # Unused by the v17 residual stem.
    if set(old)-{field.name for field in fields(CoordinateRegressionConfig)}:
        raise ValueError('Unrecognized source configuration fields')
    old['fine'] = {key: value for key, value in old['fine'].items()
                   if key not in ('gate_direction', 'history_render', 'history_sigma')}
    cfg = CoordinateRegressionConfig(**old)
    model = build_model(cfg)
    for name in ('model', 'ema'):
        model.load_state_dict(ck[name], strict=True)
    if ck['ct_normalization']['method'] != ZSCORE_METHOD:
        raise ValueError('The source must already use crop z-score normalization')
    # Preserve inference settings and data identities without inheriting training state.
    keep = ('model', 'ema', 'step', 'vol_spec', 'crop', 'n_history', 'n_commit', 'sample_cfg',
            'ct_normalization', 'data_policy', 'frame_policy', 'history_sampling_revision',
            'fiber_manifest', 'dataset_config', 'training_options', 'tolerance', 'identity_sampling')
    converted = {key: ck[key] for key in keep if key in ck}
    converted['crop'] = cfg.to_dict()['fine']
    if 'sample_cfg' in converted:
        converted['sample_cfg'] = dict(converted['sample_cfg'], crop=converted['crop'])
    converted.update(kind='weights', model_type='coordinate_regression', model_cfg=cfg.to_dict(),
                     conversion=dict(source=str(source.resolve()), source_step=ck.get('step')))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as stream:
        torch.save(converted, stream)
    return converted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = convert(args.source, args.destination)
    print(f'Converted step {result["conversion"]["source_step"]:,} to {args.destination}')


if __name__ == '__main__':
    main()
