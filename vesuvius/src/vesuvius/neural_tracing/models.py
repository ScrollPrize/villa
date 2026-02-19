import torch
import torch.nn as nn
from types import SimpleNamespace
from pathlib import Path

from vesuvius.models.build.build_network_from_config import NetworkFromConfig
from vesuvius.neural_tracing.youssef_mae import Vesuvius3dViTModel

try:
    from huggingface_hub import snapshot_download
except ImportError:  # pragma: no cover - optional dependency in some environments
    snapshot_download = None


_CHECKPOINT_SENTINELS = {
    "extrap_displacement_latest": {
        "repo_id": "scrollprize/extrap_displacement_latest",
        "checkpoint_filename": "extrap_displacement_latest.pth",
    },
    "copy_displacement_latest": {
        "repo_id": "scrollprize/copy_displacement_latest",
        "checkpoint_filename": "copy_displacement_latest.pth",
    },
}
_HF_CHECKPOINT_CACHE_ROOT = Path("/tmp/vesuvius_hf_models")


class SlotConditionedHead(nn.Module):
    """
    Task head with slot-specific FiLM modulation for neural tracing.

    Takes shared decoder features and applies per-slot FiLM modulation before
    projecting to each slot's output. Each slot has its own projection head.

    Used as a task_head, receiving features from the shared_decoder.
    """

    def __init__(self, feature_dim, num_slots, embed_dim=64, conv_op=nn.Conv3d):
        super().__init__()
        self.num_slots = num_slots
        self.feature_dim = feature_dim

        # Learnable slot embeddings
        self.slot_embed = nn.Embedding(num_slots, embed_dim)

        # Project embedding to scale/shift for FiLM
        self.film_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, feature_dim * 2)
        )

        # Per-slot 1x1 conv heads (instead of shared)
        self.slot_heads = nn.ModuleList([
            conv_op(feature_dim, 1, kernel_size=1, bias=True)
            for _ in range(num_slots)
        ])

        # Initialize FiLM with small random values to break symmetry
        nn.init.normal_(self.film_proj[-1].weight, std=0.02)
        nn.init.normal_(self.film_proj[-1].bias, std=0.02)

    def forward(self, shared_features):
        # shared_features: [B, feat_dim, Z, Y, X]

        # Compute all FiLM parameters at once (batched)
        all_film_params = self.film_proj(self.slot_embed.weight)  # [num_slots, feat_dim*2]
        scales, shifts = all_film_params.chunk(2, dim=1)  # [num_slots, feat_dim] each

        # Reshape for broadcasting: [num_slots, feat_dim, 1, 1, 1]
        scales = scales.view(self.num_slots, self.feature_dim, 1, 1, 1)
        shifts = shifts.view(self.num_slots, self.feature_dim, 1, 1, 1)

        # Apply FiLM and per-slot heads
        outputs = []
        for i in range(self.num_slots):
            # FiLM: feature * (1 + scale) + shift
            modulated = shared_features * (1 + scales[i]) + shifts[i]
            outputs.append(self.slot_heads[i](modulated))

        return torch.cat(outputs, dim=1)  # [B, num_slots, Z, Y, X]


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
    if config_dict.get('aux_srf_overlap', False) and 'srf_overlap' not in targets:
        targets = dict(targets)
        targets['srf_overlap'] = {
            'out_channels': 1,
            'activation': 'none',
        }

    # Direction field only mode - replaces default uv_heatmaps output
    if config_dict.get('direction_field_only', False):
        targets = {
            'direction_field': {
                'out_channels': 6,  # U (3) + V (3) displacement vectors
                'activation': 'none',
            }
        }
        # Direction field doesn't use conditioning channels
        conditioning_channels = 0

    mgr = SimpleNamespace()
    mgr.model_config = model_config
    # Handle crop_size as int (cubic) or list [D, H, W]
    crop_size = config_dict['crop_size']
    if isinstance(crop_size, (list, tuple)):
        mgr.train_patch_size = tuple(crop_size)
    else:
        mgr.train_patch_size = (crop_size, crop_size, crop_size)
    mgr.train_batch_size = int(config_dict.get('batch_size', 1))
    if 'in_channels' in config_dict:
        mgr.in_channels = int(config_dict['in_channels'])
    else:
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
        model = build_network_from_config_dict(config)

        # If slot conditioning enabled, replace uv_heatmaps head with SlotConditionedHead
        if config.get('slot_conditioning', False):
            num_slots = int(config.get('out_channels', 5))
            embed_dim = int(config.get('slot_embed_dim', 64))
            feature_dim = model.shared_encoder.output_channels[0]
            conv_op = model.shared_encoder.conv_op

            # Replace the simple 1x1 conv head with SlotConditionedHead
            model.task_heads['uv_heatmaps'] = SlotConditionedHead(
                feature_dim, num_slots, embed_dim, conv_op
            )
            # Remove from task_decoders if it exists (we want to use shared_decoder + our head)
            if 'uv_heatmaps' in model.task_decoders:
                del model.task_decoders['uv_heatmaps']
            print(f"Using SlotConditionedHead for uv_heatmaps: {num_slots} slots, embed_dim={embed_dim}, feature_dim={feature_dim}")

        return model
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


def _resolve_checkpoint_sentinel(checkpoint_path):
    sentinel = str(checkpoint_path).strip()
    sentinel_config = _CHECKPOINT_SENTINELS.get(sentinel)
    if sentinel_config is None:
        return None
    repo_id = sentinel_config["repo_id"]
    checkpoint_filename = sentinel_config["checkpoint_filename"]

    if snapshot_download is None:
        raise ImportError(
            "The huggingface_hub package is required to resolve "
            f"checkpoint sentinel '{sentinel}'. Install it with: pip install huggingface_hub"
        )

    cache_dir = _HF_CHECKPOINT_CACHE_ROOT / repo_id.replace("/", "__")
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        downloaded = snapshot_download(
            repo_id=repo_id,
            local_dir=str(cache_dir),
            local_dir_use_symlinks=False,
        )
    except TypeError:
        # Compatibility with huggingface_hub versions that do not support local_dir_use_symlinks.
        downloaded = snapshot_download(
            repo_id=repo_id,
            local_dir=str(cache_dir),
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download checkpoint sentinel '{sentinel}' from Hugging Face repo "
            f"'{repo_id}' into '{cache_dir}': {exc}"
        ) from exc

    downloaded_path = Path(downloaded)
    if checkpoint_filename:
        resolved_checkpoint = downloaded_path / checkpoint_filename
        if resolved_checkpoint.exists():
            return resolved_checkpoint

    candidates = sorted(
        p for p in downloaded_path.rglob("*")
        if p.is_file() and p.suffix.lower() in {".pth", ".pt"}
    )
    if len(candidates) == 1:
        return candidates[0]

    if checkpoint_filename:
        raise FileNotFoundError(
            f"Checkpoint sentinel '{sentinel}' expected file '{checkpoint_filename}' in "
            f"'{downloaded}', but it was not found. "
            f"Found {len(candidates)} .pt/.pth candidate(s)."
        )
    if len(candidates) == 0:
        raise FileNotFoundError(
            f"Checkpoint sentinel '{sentinel}' in '{downloaded}' did not contain any .pt/.pth files."
        )
    raise RuntimeError(
        f"Checkpoint sentinel '{sentinel}' in '{downloaded}' is ambiguous: "
        f"found multiple .pt/.pth files ({', '.join(str(p.relative_to(downloaded_path)) for p in candidates)})."
    )


def resolve_checkpoint_path(checkpoint_path):
    checkpoint_text = str(checkpoint_path).strip() if checkpoint_path is not None else ""
    if not checkpoint_text:
        raise FileNotFoundError("Checkpoint path is empty.")

    resolved_sentinel_path = _resolve_checkpoint_sentinel(checkpoint_text)
    if resolved_sentinel_path is not None:
        path = resolved_sentinel_path
    else:
        path = Path(checkpoint_text)
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
