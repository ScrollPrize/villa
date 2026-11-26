import os
import numpy as np
import torch
import vc_reader.transforms as A
import napari
from pathlib import Path
import torch.nn as nn
from tqdm.auto import tqdm
import contextlib
from typing import Optional, List, Tuple, Dict, Any, Union, Callable
import itertools
from copy import deepcopy
from types import SimpleNamespace

from vesuvius.models.run.blending import generate_gaussian_map
from vesuvius.utils.models.helpers import compute_steps_for_sliding_window

def _checkpoint_model_config(checkpoint_data):
    """The network config a train.py checkpoint was saved with: its 'model_config', or the
    nested legacy 'config' unwrapped (as vesuvius.predict reads it). Primus saves its patch
    as 'input_shape'; it is also returned as 'patch_size', which inference tiles with."""
    if checkpoint_data.get('model_config'):
        model_config = deepcopy(checkpoint_data['model_config'])
    else:
        legacy = checkpoint_data['config']
        model_config = deepcopy(legacy.get('model_config') or {})
        if 'patch_size' in legacy:
            model_config.setdefault('patch_size', list(legacy['patch_size']))
        for key in ('batch_size', 'in_channels', 'targets'):
            if key in legacy:
                model_config.setdefault(key, deepcopy(legacy[key]))
    if 'patch_size' not in model_config and 'train_patch_size' not in model_config and 'input_shape' in model_config:
        model_config['patch_size'] = list(model_config['input_shape'])
    return model_config


def _apply_saved_decoder_layout(model_config, state_dict):
    """Rebuild each target with the decoder its weights were saved with. The saved
    'separate_decoders' is True as soon as one target has its own decoder, so a target
    that used the shared decoder would otherwise get a separate, untrained one."""
    for name, info in (model_config.get('targets') or {}).items():
        if any(k.startswith(f"task_decoders.{name}.") for k in state_dict):
            info['separate_decoder'] = True
        elif any(k.startswith(f"task_heads.{name}.") for k in state_dict):
            info['separate_decoder'] = False


def _training_normalizer(scheme, intensity_properties):
    """The intensity normalization the trainer's ZarrDataset applied to every training patch
    for a checkpoint saved with `scheme` and `intensity_properties` (save_checkpoint stores
    both), built by ZarrDataset's own code. Returns a function of one window."""
    from vesuvius.models.datasets.zarr_dataset import ZarrDataset

    dataset = ZarrDataset.__new__(ZarrDataset)  # only its normalization setup is used
    dataset.mgr = SimpleNamespace(skip_intensity_sampling=True)
    dataset.normalization_scheme = scheme
    dataset.intensity_properties = dict(intensity_properties or {})
    dataset._initialize_normalization()
    normalizer = dataset.normalizer

    def normalize(window: np.ndarray) -> np.ndarray:
        # as ZarrDataset.__getitem__ normalizes a patch before padding it
        window = np.asarray(window, dtype=np.float32)
        if normalizer is None or window.size == 0:
            return window
        normalized = normalizer({"image": window.reshape(1, 1, -1)}, seed=0)["image"]
        return np.asarray(normalized, dtype=np.float32).reshape(window.shape)

    return normalize


def _source_transform(layer, ndim: int) -> Dict[str, Any]:
    """scale, translate, rotate, shear and affine that put an ndim-D prediction on the grid of
    the napari layer it was computed from: the source's transform on the trailing (spatial)
    axes, identity on the leading (batch, channel) axes."""
    if not hasattr(layer, "affine"):  # not a napari layer
        return {}
    from napari.utils.transforms import CompositeAffine

    extra = ndim - layer.ndim
    if extra < 0:
        raise ValueError(f"A {ndim}D prediction cannot carry the transform of the {layer.ndim}D layer '{layer.name}'")
    physical = CompositeAffine(scale=layer.scale, translate=layer.translate, rotate=layer.rotate,
                               shear=layer.shear, ndim=layer.ndim)
    affine = layer.affine
    if extra:
        physical, affine = physical.expand_dims(list(range(extra))), affine.expand_dims(list(range(extra)))
    return {"scale": physical.scale, "translate": physical.translate, "rotate": physical.rotate,
            "shear": physical.shear, "affine": affine.affine_matrix}


def _strip_wrapper_prefixes(state_dict):
    """Drop DDP ('module.') and torch.compile ('_orig_mod.') key prefixes."""
    def strip(key):
        changed = True
        while changed:
            changed = False
            for prefix in ('module.', '_orig_mod.'):
                if key.startswith(prefix):
                    key, changed = key[len(prefix):], True
        return key
    return {strip(k): v for k, v in state_dict.items()}


# Model loader class
class ModelLoader:
    """
    Class for loading models from checkpoints with optional config manager integration.
    """
    
    def __init__(self, checkpoint_path, config_manager=None):
        self.checkpoint_path = Path(checkpoint_path)
        self.config_manager = config_manager
        # The training preprocessing saved with the checkpoint (None: not saved).
        self.normalization_scheme = None
        self.intensity_properties = None
        
    def load(self):
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
            
        checkpoint_data = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)
        if isinstance(checkpoint_data, dict) and checkpoint_data.get('normalization_scheme') is not None:
            self.normalization_scheme = checkpoint_data['normalization_scheme']
            self.intensity_properties = checkpoint_data.get('intensity_properties') or {}

        if 'model' in checkpoint_data:
            print("Loading weights from 'model' key in checkpoint")
            model_state_dict = checkpoint_data['model']
        elif 'model_state_dict' in checkpoint_data:
            model_state_dict = checkpoint_data['model_state_dict']
        elif 'state_dict' in checkpoint_data:
            model_state_dict = checkpoint_data['state_dict']
        else:
            model_state_dict = checkpoint_data
        model_state_dict = _strip_wrapper_prefixes(model_state_dict)

        from vesuvius.models.build.build_network_from_config import NetworkFromConfig
        
        class MinimalConfigManager:
            def __init__(self, model_config):
                self.model_config = model_config
                self.inference_config = model_config
                self.targets = model_config.get('targets', {})
                self.train_patch_size = model_config.get('patch_size', (128, 128, 128))
                self.train_batch_size = model_config.get('batch_size', 1)
                self.in_channels = model_config.get('in_channels', 1)
                self.autoconfigure = model_config.get('autoconfigure', False)
                
                self.spacing = [1] * len(self.train_patch_size)
                self.model_name = "Loaded_Model"

        # The checkpoint's own config describes the network its weights belong to;
        # the GUI's config manager is only a fallback for checkpoints without one.
        if checkpoint_data.get('model_config') or isinstance(checkpoint_data.get('config'), dict):
            model_config = _checkpoint_model_config(checkpoint_data)
            _apply_saved_decoder_layout(model_config, model_state_dict)
            mgr = MinimalConfigManager(model_config)
        elif self.config_manager is not None:
            # Use config from config manager
            model_config = self.config_manager.model_config
            mgr = self.config_manager
            print(f"Using model configuration from config manager")
        else:
            print(f"Warning: No model configuration found in checkpoint. Using empty config.")
            model_config = {}
            mgr = MinimalConfigManager(model_config)
            
        model = NetworkFromConfig(mgr)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        # Every tensor of the network must come from the checkpoint; checkpoint tensors the
        # network does not use (e.g. deep-supervision or old shared-decoder heads) are skipped.
        try:
            missing, unexpected = model.load_state_dict(model_state_dict, strict=False)
        except RuntimeError as e:
            raise RuntimeError(f"The weights in {self.checkpoint_path} do not fit the network built "
                               f"from its config: {e}") from e
        if missing:
            raise RuntimeError(
                f"{self.checkpoint_path} has no weights for {len(missing)} of the network's "
                f"{len(model.state_dict())} tensors ({', '.join(missing[:5])}{', ...' if len(missing) > 5 else ''}); "
                "not running inference with untrained weights.")
        if unexpected:
            print(f"Skipped {len(unexpected)} checkpoint tensors the inference network does not use "
                  f"({', '.join(unexpected[:5])}{', ...' if len(unexpected) > 5 else ''})")
        print("Model weights loaded successfully")
                
        return model, model_config

def _as_outputs(outputs) -> Dict[str, torch.Tensor]:
    if isinstance(outputs, torch.Tensor):
        return {"output": outputs}
    if isinstance(outputs, tuple) and all(isinstance(o, torch.Tensor) for o in outputs):
        return {f"output_{i}": tensor for i, tensor in enumerate(outputs)}
    if isinstance(outputs, dict):
        return outputs
    raise ValueError(f"Unexpected output format: {type(outputs)}")


def sliding_window_inference(model: nn.Module,
                             image: np.ndarray,
                             patch_size: Tuple[int, ...],
                             overlap: float = 0.25,
                             batch_size: int = 2,
                             sigma_scale: float = 8.0,
                             post_transform: Optional[Callable] = None,
                             normalize_window: Optional[Callable[[np.ndarray], np.ndarray]] = None
                             ) -> Dict[str, np.ndarray]:
    """Gaussian-blended sliding-window prediction of an in-memory image.

    Patch positions and blend weights are those of ``vesuvius.predict`` /
    ``vesuvius.blend_logits``: per-axis steps from
    ``compute_steps_for_sliding_window`` with step fraction ``1 - overlap``,
    weights from ``generate_gaussian_map`` (sigma = patch / sigma_scale).
    ``image`` is spatial ``(Y, X)`` / ``(Z, Y, X)`` or channel-first
    ``(C, ...)``; spatial axes smaller than the patch are zero-padded.
    ``normalize_window`` (the checkpoint's training normalization) is applied to
    each window before padding, as the trainer did to each patch; without it the
    whole image is min-max scaled. Returns ``(1, C_out, ...)`` arrays per model output.
    """
    patch_size = tuple(int(p) for p in patch_size)
    dims = len(patch_size)
    if image.ndim == dims:
        image = image[None]
    elif image.ndim != dims + 1:
        raise ValueError(f"{dims}D patch needs a {dims}D or channel-first {dims + 1}D image, got shape {image.shape}")

    device = next(model.parameters()).device
    model.eval()
    if normalize_window is None:
        inputs = A.Normalize("minmax", eps=1e-5)(torch.as_tensor(np.asarray(image), dtype=torch.float32, device=device))[None]
    else:
        raw = np.asarray(image, dtype=np.float32)  # each window is normalized on its own below
        inputs = torch.as_tensor(raw)[None]

    spatial = tuple(inputs.shape[2:])
    padded = tuple(max(s, p) for s, p in zip(spatial, patch_size))
    if padded != spatial:
        pad = []
        for s, p in reversed(list(zip(spatial, padded))):
            pad += [0, p - s]
        inputs = torch.nn.functional.pad(inputs, pad, mode="constant", value=0.0)

    step = 1.0 - overlap
    positions = list(itertools.product(*(compute_steps_for_sliding_window(s, p, step)
                                          for s, p in zip(padded, patch_size))))
    weight = torch.as_tensor(generate_gaussian_map(patch_size, sigma_scale=sigma_scale)[0], device=device)
    weight_sum = torch.zeros(padded, dtype=torch.float32, device=device)
    sums: Dict[str, torch.Tensor] = {}

    with torch.no_grad():
        for start in tqdm(range(0, len(positions), batch_size), desc="Sliding window"):
            batch_pos = positions[start:start + batch_size]
            windows = [tuple(slice(o, o + p) for o, p in zip(pos, patch_size)) for pos in batch_pos]
            if normalize_window is None:
                batch = torch.cat([inputs[(slice(None), slice(None)) + w] for w in windows])
            else:
                batch = torch.cat([_normalized_window(raw, w, spatial, normalize_window) for w in windows]).to(device)
            batch = batch.contiguous()
            for name, pred in _as_outputs(model(batch)).items():
                acc = sums.get(name)
                if acc is None:
                    acc = sums[name] = torch.zeros((pred.shape[1],) + padded, dtype=torch.float32, device=device)
                for i, w in enumerate(windows):
                    acc[(slice(None),) + w] += pred[i].float() * weight
            for w in windows:
                weight_sum[w] += weight

    crop = tuple(slice(0, s) for s in spatial)
    weight_sum = weight_sum[crop]
    result = {}
    for name, acc in sums.items():
        acc = acc[(slice(None),) + crop]
        blended = torch.where(weight_sum > 0, acc / weight_sum, acc)[None]
        if post_transform is not None:
            blended = post_transform(blended)
        result[name] = blended.cpu().numpy()
    return result


def _normalized_window(raw, window, spatial, normalize_window):
    """The window of the (C, ...) image: its part inside the image normalized, then zero-padded
    to the patch size (ZarrDataset pads a patch at the volume edge the same way)."""
    inside = tuple(slice(w.start, min(w.stop, n)) for w, n in zip(window, spatial))
    out = np.zeros((raw.shape[0],) + tuple(w.stop - w.start for w in window), dtype=np.float32)
    out[(slice(None),) + tuple(slice(0, s.stop - s.start) for s in inside)] = \
        normalize_window(raw[(slice(None),) + inside])
    return torch.from_numpy(out)[None]


def run_inference(viewer: napari.Viewer,
                  layer: napari.layers.Layer,
                  checkpoint_path: str,
                  roi_size: Optional[Tuple[int, ...]] = None,
                  overlap: float = 0.25,
                  batch_size: int = 2,
                  sigma_scale: float = 8.0,
                  activation_type: str = "none") -> List[napari.layers.Layer]:

    config_manager = None
    try:
        from .main_window import _config_manager
        if _config_manager is not None:
            config_manager = _config_manager
            print("Using global config manager from main_window")
    except (ImportError, AttributeError):
        print("No global config manager available")

    loader = ModelLoader(checkpoint_path, config_manager=config_manager)
    model, model_config = loader.load()
    model_name = Path(checkpoint_path).stem
    image_data = layer.data

    if roi_size is None:
        for key in ('patch_size', 'train_patch_size'):
            if key in model_config:
                roi_size = tuple(model_config[key])
                print(f"Using {key} from model_config: {roi_size}")
                break
        else:
            raise ValueError("No patch_size or train_patch_size found in model config. Cannot proceed with inference.")

    # Dimensionality comes from the model patch, not the raw data: napari
    # images are often (C, H, W), which is 2D inference with a 2D patch.
    if len(roi_size) not in (2, 3):
        raise ValueError(f"roi_size must be 2D or 3D, got {roi_size}")
    print(f"Final roi_size for inference: {roi_size}")
    print(f"Data shape: {image_data.shape}, Using {len(roi_size)}D inference")

    post_transform = None
    activation = activation_type.lower()
    if activation == "sigmoid":
        post_transform = torch.sigmoid
    elif activation == "softmax":
        post_transform = lambda x: torch.softmax(x, dim=1)
    elif activation == "argmax":
        post_transform = lambda x: torch.argmax(x, dim=1, keepdim=True).float()

    normalize_window = None
    if loader.normalization_scheme is not None:
        normalize_window = _training_normalizer(loader.normalization_scheme, loader.intensity_properties)
        print(f"Normalizing each window with the checkpoint's '{loader.normalization_scheme}' scheme")
    else:
        print("The checkpoint has no saved normalization; min-max scaling the image")

    results = sliding_window_inference(model, np.asarray(image_data), roi_size, overlap=overlap,
                                       batch_size=batch_size, sigma_scale=sigma_scale,
                                       post_transform=post_transform, normalize_window=normalize_window)
    new_layers = []
    for output_name, result in results.items():
        new_layers.append(viewer.add_image(result, name=f"{layer.name}_{model_name}_{output_name}",
                                           colormap='gray', blending='translucent',
                                           **_source_transform(layer, result.ndim)))

    print(f"Inference completed. Added {len(new_layers)} new layers to the viewer.")
    return new_layers
