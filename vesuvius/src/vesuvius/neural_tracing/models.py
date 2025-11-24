import torch
from types import SimpleNamespace

from vesuvius.models.build.build_network_from_config import NetworkFromConfig
from youssef_mae import Vesuvius3dViTModel


def _config_dict_to_mgr(config_dict):
    """Create a minimal ConfigManager-like object from a plain config dict."""
    model_config = dict(config_dict.get('model_config', {}) or {})

    # Allow overriding targets; default to a single uv_heatmaps head
    targets = config_dict.get('targets')
    if not targets:
        targets = {
            'uv_heatmaps': {
                'out_channels': config_dict['step_count'] * 2,
                'activation': 'none',
            }
        }

    mgr = SimpleNamespace()
    mgr.model_config = model_config
    mgr.train_patch_size = tuple([config_dict['crop_size']] * 3)
    mgr.train_batch_size = int(config_dict.get('batch_size', 1))
    mgr.in_channels = 5  # volume + localiser + 3 conditioning heatmaps
    mgr.model_name = config_dict.get('model_name', 'neural_tracing')
    mgr.autoconfigure = True  # explicit per request
    mgr.spacing = model_config.get('spacing', [1, 1, 1])
    mgr.targets = targets
    mgr.enable_deep_supervision = bool(config_dict.get('enable_deep_supervision', True))
    # Explicitly mark dimensionality so NetworkFromConfig skips guessing
    mgr.op_dims = 3
    return mgr


def build_network_from_config_dict(config_dict):
    mgr = _config_dict_to_mgr(config_dict)
    model = NetworkFromConfig(mgr)

    if getattr(mgr, 'enable_deep_supervision', False) and hasattr(model, 'task_decoders'):
        for dec in model.task_decoders.values():
            if hasattr(dec, 'deep_supervision'):
                dec.deep_supervision = True
    return model


def make_model(config):
    if config['model_type'] == 'unet':
        return build_network_from_config_dict(config)
    elif config['model_type'] == 'vit':
        return Vesuvius3dViTModel(
            mae_ckpt_path=config['model_config'].get('mae_ckpt_path', None),
            in_channels=5,
            out_channels=config['step_count'] * 2,
            input_size=config['crop_size'],
            patch_size=8,  # TODO: infer automatically from volume_scale and pretraining crop size
        )
    else:
        raise RuntimeError('unexpected model_type, should be unet or vit')


def load_checkpoint(checkpoint_path, model):
    print(f'loading checkpoint {checkpoint_path}... ')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])
