"""The model registry: each model type's configuration class and model, and configurations recorded in checkpoints.

Model types: 'regression' and 'flow' (models/crop_transformer.py) and 'sequence' (models/sequence.py); their shared
contract is models/follower.py.
"""
from dataclasses import fields

from vesuvius.neural_tracing.tracers.models.crop_transformer import FlowConfig, FlowFollower, RegressionConfig, RegressionFollower
from vesuvius.neural_tracing.tracers.models.sequence import SequenceConfig, SequenceFollower
from vesuvius.neural_tracing.tracers.shared.retired import retire

MODELS = dict(regression=(RegressionConfig, RegressionFollower), flow=(FlowConfig, FlowFollower),
              sequence=(SequenceConfig, SequenceFollower))
MODEL_TYPES = tuple(MODELS)


def config_class(model_type):
    if model_type not in MODELS:
        raise ValueError(f'Unsupported model type: {model_type!r} (supported: {", ".join(MODEL_TYPES)})')
    return MODELS[model_type][0]


# Removed configuration fields and their only supported value: configurations recorded before the removal (checkpoints,
# run files) still hold them and load when they hold that value.
RETIRED_FIELDS = dict(activation_checkpointing=False, query_scale=None, flow_time_conditioning='adaln_zero',
                      flow_unknown_planes='own_path', flow_selection='retry', flow_zero_start=True,
                      flow_sample_threshold=0., flow_loss='pseudo_huber', flow_geometry_weight=0.)
# Model fields added after checkpoints were recorded without them, with the value those checkpoints were trained with.
RECORDED_DEFAULTS = dict(qk_norm=False, flow_modulation_bound=0.)


def recorded_config(model_type, recorded):
    """A configuration recorded without newer fields (RECORDED_DEFAULTS) or with retired ones (RETIRED_FIELDS), as it
    was trained."""
    cls = config_class(model_type)
    names = {f.name for f in fields(cls)}
    recorded = retire(recorded, RETIRED_FIELDS, 'model')
    return cls(**{**{k: v for k, v in RECORDED_DEFAULTS.items() if k in names}, **recorded})


def config_from_checkpoint(ck):
    """The model configuration recorded in a checkpoint."""
    return recorded_config(ck.get('model_type'), ck['model_cfg'])


def build_model(cfg):
    if cfg.model_type not in MODELS:
        raise ValueError(f'Unsupported model type: {cfg.model_type!r}')
    return MODELS[cfg.model_type][1](cfg)
