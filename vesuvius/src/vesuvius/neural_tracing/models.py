import torch
from types import SimpleNamespace
from pathlib import Path

from vesuvius.models.build.build_network_from_config import NetworkFromConfig
from vesuvius.neural_tracing.youssef_mae import Vesuvius3dViTModel


def _config_dict_to_mgr(config_dict):
    """Create a minimal ConfigManager-like object from a plain config dict."""
    model_config = dict(config_dict.get('model_config', {}) or {})

    # Allow overriding targets; default to a single uv_heatmaps head
    conditioning_channels = int(config_dict.get('conditioning_channels', 3))
    use_localiser = bool(config_dict.get('use_localiser', True))
    default_out_channels = int(config_dict.get('out_channels', config_dict['step_count'] * 2))

    targets = config_dict.get('targets')
    if not targets:
        targets = {
            'uv_heatmaps': {
                'out_channels': default_out_channels,
                'activation': 'none',
            }
        }
    # If auxiliary segmentation is requested, ensure a seg head is present
    if config_dict.get('aux_segmentation', False) and 'seg' not in targets:
        targets = dict(targets)
        targets['seg'] = {
            'out_channels': 1,
            'activation': 'none',
        }
    if config_dict.get('aux_normals', False) and 'normals' not in targets:
        targets = dict(targets)
        targets['normals'] = {
            'out_channels': 3,
            'activation': 'none',
        }

    mgr = SimpleNamespace()
    mgr.model_config = model_config
    mgr.train_patch_size = tuple([config_dict['crop_size']] * 3)
    mgr.train_batch_size = int(config_dict.get('batch_size', 1))
    mgr.in_channels = 1 + conditioning_channels + (1 if use_localiser else 0)  # volume + optional localiser + conditioning
    mgr.model_name = config_dict.get('model_name', 'neural_tracing')
    mgr.autoconfigure = True  # explicit per request
    mgr.spacing = model_config.get('spacing', [1, 1, 1])
    mgr.targets = targets
    mgr.enable_deep_supervision = bool(config_dict.get('enable_deep_supervision', False))
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
    conditioning_channels = int(config.get('conditioning_channels', 3))
    use_localiser = bool(config.get('use_localiser', True))
    default_out_channels = int(config.get('out_channels', config['step_count'] * 2))

    if config['model_type'] == 'unet':
        return build_network_from_config_dict(config)
    elif config['model_type'] == 'vit':
        return Vesuvius3dViTModel(
            mae_ckpt_path=config['model_config'].get('mae_ckpt_path', None),
            in_channels=1 + conditioning_channels + (1 if use_localiser else 0),
            out_channels=default_out_channels,
            input_size=config['crop_size'],
            patch_size=8,  # TODO: infer automatically from volume_scale and pretraining crop size
        )
    else:
        raise RuntimeError('unexpected model_type, should be unet or vit')


def resolve_checkpoint_path(checkpoint_path):
    path = Path(checkpoint_path)
    if path.is_dir():
        candidates = list(path.glob("ckpt_*.pth"))
        if not candidates:
            raise FileNotFoundError(f"No checkpoints matching 'ckpt_*.pth' found in {path}")

        def iteration(p):
            stem = p.stem  # e.g. ckpt_000123
            try:
                return int(stem.split("_")[-1])
            except ValueError:
                return -1

        candidates.sort(key=iteration)
        return candidates[-1]

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    return path


def load_checkpoint(checkpoint_path):
    checkpoint_path = resolve_checkpoint_path(checkpoint_path)
    print(f'loading checkpoint {checkpoint_path}... ')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    state = checkpoint['model']
    config = checkpoint['config']

    model = make_model(config)

    state = strip_state(state)

    model.load_state_dict(state)
    return model, config


def strip_state(state):

    # Checkpoints saved from torch.compile / DDP may prepend wrapper prefixes.
    prefixes = ('module.', '_orig_mod.')

    def strip_prefixes(key: str) -> str:
        # Remove all known prefixes, even if nested (e.g., module._orig_mod.)
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if key.startswith(p):
                    key = key[len(p):]
                    changed = True
        return key

    new_state = {}
    for k, v in state.items():
        new_key = strip_prefixes(k)
        # Skip duplicate encoder keys nested inside decoder (from old checkpoints).
        # These were created when Decoder registered encoder as a submodule.
        if '.encoder.' in new_key and new_key.split('.encoder.')[0].endswith('decoder'):
            continue
        new_state[new_key] = v
    return new_state

