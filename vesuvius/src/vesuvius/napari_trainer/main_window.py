import napari
from magicgui import magicgui, widgets
import napari.viewer
import scipy.ndimage

from .inference_widget import inference_widget
from vesuvius.models.configuration.config_manager import ConfigManager
from PIL import Image
import numpy as np
from pathlib import Path
from collections.abc import Sequence
from copy import deepcopy
import napari.layers
import json
import yaml
from pathlib import Path
import re
import shutil
import tempfile
import torch.nn as nn


Image.MAX_IMAGE_PIXELS = None

_config_manager = None

# The shipped configs are in <package>/models/configuration; this file is <package>/napari_trainer/main_window.py.
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / 'models' / 'configuration' / 'single_task' / 'ps128_dicece.yaml'

@magicgui(filenames={"label": "select config file", "filter": "*.yaml"},
          auto_call=True)
def filespicker(filenames: Sequence[Path] = str(DEFAULT_CONFIG_PATH)) -> Sequence[Path]:
    print("selected config : ", filenames)
    if filenames and _config_manager is not None:
        # Load the first selected file into the config manager
        _config_manager.load_config(filenames[0])
        print(f"Config loaded from {filenames[0]}")
    return filenames

@magicgui(
    call_button='run training',
    patch_size_z={'widget_type': 'SpinBox', 'label': 'Patch Size Z', 'min': 0, 'max': 4096, 'value': 0},
    patch_size_x={'widget_type': 'SpinBox', 'label': 'Patch Size X', 'min': 0, 'max': 4096, 'value': 128},
    patch_size_y={'widget_type': 'SpinBox', 'label': 'Patch Size Y', 'min': 0, 'max': 4096, 'value': 128},
    min_labeled_percentage={'widget_type': 'FloatSpinBox', 'label': 'Min Labeled Percentage', 'min': 0.0, 'max': 100.0, 'step': 1.0, 'value': 10.0},
    max_epochs={'widget_type': 'SpinBox', 'label': 'Max Epochs', 'min': 1, 'max': 1000, 'value': 5},
    # The loss for targets the config gives no loss: two output channels, one label channel, ignore label kept.
    loss_function={'widget_type': 'ComboBox', 'choices': ["nnUNet_DC_and_CE_loss"], 'value': "nnUNet_DC_and_CE_loss"}
)
def run_training(patch_size_z: int = 0, patch_size_x: int = 128, patch_size_y: int = 128,
                min_labeled_percentage: float = 10.0,
                max_epochs: int = 5,
                loss_function: str = "nnUNet_DC_and_CE_loss"):
    if _config_manager is None:
        print("Error: No configuration loaded. Please load a config file first.")
        return
    
    print("Starting training process...")
    print("Using images and labels from current viewer")

    # Check if viewer has valid image-label pairs
    viewer = napari.current_viewer()
    if viewer is None:
        print("Error: No active napari viewer found.")
        return
    
    # Quick validation of layer naming
    all_layers = list(viewer.layers)
    image_layers = [l for l in all_layers if isinstance(l, napari.layers.Image)]
    label_layers = [l for l in all_layers if isinstance(l, napari.layers.Labels)]
    
    if not image_layers:
        print("Error: No image layers found in the viewer.")
        return
    
    if not label_layers:
        print("Error: No label layers found in the viewer.")
        return

    pairs = _pair_layers(image_layers, label_layers)
    if not pairs:
        print("Error: No valid image-label pairs found.")
        print("Label layers should be named as: {image_name}_{target_name}")
        print("For example: if image is 'image1', label should be 'image1_ink'")
        return

    # Training changes the config (data, targets, patch size, ops); work on a copy so the
    # selected config stays as loaded for the next run and for inference's fallback.
    mgr = deepcopy(_config_manager)
    _check_trains_on_viewer_layers(mgr, pairs)

    # Determine dimensionality from the paired layers
    all_ndims = [np.ndim(l.data) for img, labels in pairs for l in [img] + [lbl for _, lbl in labels]]
    is_2d_view = all(n == 2 for n in all_ndims)
    if not is_2d_view and patch_size_z == 0:
        raise ValueError(
            f"The paired layers are {max(all_ndims)}D, but Patch Size Z is 0, which selects 2D patches. "
            "Set Patch Size Z for 3D training.")

    # Set data format before updating other configurations
    mgr.data_format = 'napari'

    # Handle 2D case when the viewer layers are 2D
    if is_2d_view:
        new_patch_size = [patch_size_y, patch_size_x]  # 2D: [height, width]
        # Force 2D ops in manager to avoid accidental 3D config
        mgr.op_dims = 2
        # Hint datasets about dimensionality
        if hasattr(mgr, 'dataset_config'):
            mgr.dataset_config['force_2d'] = True
            mgr.dataset_config.pop('force_3d', None)
        if hasattr(mgr, 'model_config'):
            mgr.model_config["conv_op"] = "nn.Conv2d"
            mgr.model_config["norm_op"] = "nn.InstanceNorm2d"
        print(f"Detected 2D setup; using patch size {new_patch_size} and 2D ops")
    else:
        new_patch_size = [patch_size_z, patch_size_y, patch_size_x]  # 3D: [depth, height, width]
        mgr.op_dims = 3
        if hasattr(mgr, 'dataset_config'):
            mgr.dataset_config['force_3d'] = True
            mgr.dataset_config.pop('force_2d', None)
        if hasattr(mgr, 'model_config'):
            mgr.model_config["conv_op"] = "nn.Conv3d"
            mgr.model_config["norm_op"] = "nn.InstanceNorm3d"
        print(f"Detected 3D setup; using patch size {new_patch_size} and 3D ops")

    min_labeled_ratio = min_labeled_percentage / 100.0

    # Update configuration with new parameters. Patches are selected by the widget's
    # labeled percentage alone; the label bounding-box criterion is off.
    mgr.update_config(
        patch_size=new_patch_size,
        min_labeled_ratio=min_labeled_ratio,
        max_epochs=max_epochs,
        skip_patch_validation=False,
        min_bbox_percent=0.0
    )

    targets = _run_targets(mgr, pairs, loss_function)
    _check_patch_selection_rule(targets)
    _check_losses_fit_the_labels(targets, pairs)

    run_dir = Path(tempfile.mkdtemp(prefix="napari_trainer_"))
    try:
        _export_viewer_layers(mgr, pairs, run_dir / "data", targets)
        selected = _write_patch_cache(mgr, min_labeled_percentage)

        from vesuvius.models.training.train import BaseTrainer

        trainer = _viewer_layer_trainer(BaseTrainer, selected)(mgr=mgr, verbose=True)
        print("\nStarting training...")
        trainer.train()

    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
        print(f"Removed the exported layers ({run_dir})")


def _viewer_layer_trainer(base, selected):
    """The trainer class for a run on the viewer's layers: `base` (BaseTrainer), stopping
    before the model directory or any checkpoint is written when its datasets would not
    read exactly the `selected` patches [(volume id, position), ...], or when the
    train/validation split leaves no training patch (a run without an optimizer step
    would still save a final checkpoint and could overwrite an earlier one)."""

    class ViewerLayerTrainer(base):
        def _build_dataset_for_mgr(self, mgr, *, is_training):
            dataset = super()._build_dataset_for_mgr(mgr, is_training=is_training)
            patches = [(p.volume_name, tuple(p.position)) for p in dataset.valid_patches]
            if patches != selected:
                raise RuntimeError(
                    f"The trainer's {'training' if is_training else 'validation'} dataset has "
                    f"{len(patches)} patches, not the {len(selected)} selected ones; not training on "
                    "patches the labeled-percentage check did not select.")
            return dataset

        def _configure_dataloaders(self, train_dataset, val_dataset=None):
            loaders = super()._configure_dataloaders(train_dataset, val_dataset)
            train_indices, val_indices = loaders[2], loaders[3]
            if len(train_indices) == 0:
                raise ValueError(
                    f"The train/validation split (tr_val_split {self.mgr.tr_val_split}) puts all "
                    f"{len(val_indices)} selected patches into validation, so there is nothing to train on; "
                    "nothing was saved. Label more, or lower Min Labeled Percentage or the patch size, "
                    "so that more patches qualify.")
            return loaders

    return ViewerLayerTrainer


def _pair_layers(image_layers, label_layers):
    """Pair each '<image>_<target>' label layer with its image layer. A label whose name
    starts with several image names ('vol_crop_ink' with images 'vol' and 'vol_crop')
    belongs to the longest one. Returns [(image_layer, [(target, label_layer), ...]), ...]."""
    pairs = {}
    for lbl_layer in label_layers:
        if lbl_layer.name.endswith("_mask"):
            continue
        owners = [img for img in image_layers if lbl_layer.name.startswith(f"{img.name}_")]
        if not owners:
            continue
        img_layer = max(owners, key=lambda l: len(l.name))
        target = lbl_layer.name[len(img_layer.name) + 1:]
        pairs.setdefault(img_layer.name, (img_layer, []))[1].append((target, lbl_layer))
    return list(pairs.values())


def _check_trains_on_viewer_layers(mgr, pairs):
    """Refuse configs that would train on other data or on unlabeled patches, images without
    a label for every target, targets that are not plain names, and labels that do not
    cover their image voxel for voxel."""
    dataset_config = getattr(mgr, 'dataset_config', None) or {}
    if dataset_config.get("volumes"):
        raise ValueError(
            "The selected config lists dataset_config.volumes; the trainer would read those "
            "volumes instead of the viewer's layers. Choose a config without 'volumes'.")
    dataset_type = str(dataset_config.get("dataset_type", "zarr")).strip().lower()
    if dataset_type != "zarr":
        raise ValueError(
            f"The selected config uses dataset_type '{dataset_type}', which reads its own image and "
            "label data instead of the viewer's layers. Choose a config with the default 'zarr' dataset.")

    if getattr(mgr, 'unlabeled_foreground_enabled', False):
        raise ValueError(
            "The selected config sets unlabeled_foreground_enabled; the napari trainer trains only on "
            "labeled patches of the viewer's layers. Choose a config without it.")
    all_targets = list(dict.fromkeys(target for _, labels in pairs for target, _ in labels))
    for img_layer, labels in pairs:
        missing = [t for t in all_targets if t not in {target for target, _ in labels}]
        if missing:
            raise ValueError(
                f"Image layer '{img_layer.name}' has no label layer for target(s) {', '.join(missing)} "
                f"(e.g. '{img_layer.name}_{missing[0]}'). Every paired image needs a label layer for each "
                "target, since a missing label would be trained as background; add the label layers or "
                "train the targets separately.")

    for img_layer, labels in pairs:
        for target, lbl_layer in labels:
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", target):
                raise ValueError(
                    f"Label layer '{lbl_layer.name}' names the target '{target}'; target names may only "
                    "contain letters, digits, '_' and '-'.")
            if np.shape(lbl_layer.data) != np.shape(img_layer.data):
                raise ValueError(
                    f"Label layer '{lbl_layer.name}' has shape {np.shape(lbl_layer.data)}, but image layer "
                    f"'{img_layer.name}' has shape {np.shape(img_layer.data)}; a label layer must cover "
                    "its image voxel for voxel.")
            for attr in ("scale", "translate", "rotate", "shear", "affine"):
                img_value, lbl_value = getattr(img_layer, attr, None), getattr(lbl_layer, attr, None)
                img_value = getattr(img_value, "affine_matrix", img_value)
                lbl_value = getattr(lbl_value, "affine_matrix", lbl_value)
                if img_value is None and lbl_value is None:
                    continue
                if (img_value is None or lbl_value is None or np.shape(img_value) != np.shape(lbl_value)
                        or not np.allclose(img_value, lbl_value)):
                    raise ValueError(
                        f"Label layer '{lbl_layer.name}' is not on the grid of image layer '{img_layer.name}': "
                        f"its {attr} is {lbl_value}, the image's is {img_value}. Training reads both by "
                        "array index, so they need the same scale, translate, rotate, shear and affine.")


def _write_layer_zarr(path, data):
    from vc_reader.reader import ZarrWriter

    data = np.asarray(data)
    chunks = tuple(min(int(n), 128) for n in data.shape)
    ZarrWriter.create(path, shape=data.shape, chunks=chunks, dtype=data.dtype, overwrite=True)[...] = data


def _run_targets(mgr, pairs, loss_function):
    """The run's targets, one per label-layer suffix. A target the loaded config defines
    keeps its settings (losses, ignore label, ...); the widget's loss is used only for
    targets without configured losses."""
    targets = {}
    for _, labels in pairs:
        for target, _ in labels:
            targets.setdefault(target, deepcopy(mgr.targets.get(target) or {}))
    for name, info in targets.items():
        if info.get("losses"):
            print(f"  - Target '{name}' keeps its configured losses: "
                  f"{', '.join(str(loss.get('name')) for loss in info['losses'])}")
        else:
            info["losses"] = [{"name": loss_function, "weight": 1.0, "kwargs": {}}]
            print(f"  - Target '{name}' has no configured loss; using {loss_function}")
    return targets


# Losses whose label requirements are checked before training: a class-index label for
# cross-entropy over out_channels >= 2 channels, a 0/1 label for one-channel BCE. Other
# configured losses are kept as configured, without a check.
_CLASS_INDEX_LOSSES = {"nnUNet_DC_and_CE_loss"}
_ONE_CHANNEL_BINARY_LOSSES = {"BCEWithLogitsLoss"}


def _check_losses_fit_the_labels(targets, pairs):
    """Refuse a target whose losses cannot take its labels (e.g. cross-entropy on one output
    channel, or label values a loss cannot index), which would otherwise fail on the first batch."""
    layers = {}
    for _, labels in pairs:
        for target, lbl_layer in labels:
            layers.setdefault(target, []).append(lbl_layer)
    for name, info in targets.items():
        if name not in layers:
            continue
        out_channels = info.get("out_channels") or info.get("channels") or 2  # set_targets_and_data's default
        ignore = _ignore_value(info)
        values = set()
        for lbl_layer in layers[name]:
            values.update(np.unique(np.asarray(lbl_layer.data)).tolist())
        values.discard(ignore)
        for loss in info["losses"]:
            loss_name = loss.get("name")
            if loss_name in _CLASS_INDEX_LOSSES:
                allowed, need = set(range(int(out_channels))), "out_channels of at least 2"
                fits = int(out_channels) >= 2
            elif loss_name in _ONE_CHANNEL_BINARY_LOSSES:
                allowed, need = {0, 1}, "out_channels 1"
                fits = int(out_channels) == 1
            else:
                continue
            if not fits:
                raise ValueError(
                    f"Target '{name}' has out_channels {out_channels}, but its loss {loss_name} needs {need}. "
                    "Fix the target's out_channels or losses in the config.")
            bad = sorted(v for v in values if v not in allowed)
            if bad:
                raise ValueError(
                    f"The label layers of target '{name}' contain the values {bad}, but its loss {loss_name} "
                    f"with out_channels {out_channels} accepts only {sorted(allowed)}"
                    + (f" and the ignore label {ignore}" if ignore is not None else "")
                    + ". Repaint those voxels or change the target's settings.")


def _ignore_value(target_info):
    """The target's ignore label, under any of the names the trainer reads."""
    for alias in ("ignore_index", "ignore_label", "ignore_value"):
        if target_info.get(alias) is not None:
            return target_info[alias]
    return None


def _check_patch_selection_rule(targets):
    """Patches are selected, and the patch cache is keyed, with one valid_patch_value; the
    trainer's dataset would look the cache up with another value than the one it was written
    with if the targets set different ones, and would then train on every patch."""
    values = {name: info.get("valid_patch_value") for name, info in targets.items()
              if not info.get("auxiliary_task", False)}
    if len(set(values.values())) > 1:
        raise ValueError(
            "The targets set different valid_patch_value ("
            + ", ".join(f"{name}: {value}" for name, value in values.items())
            + "); patches are selected with one value for all targets. Give every target the same "
            "valid_patch_value (or none), or train the targets separately.")


def _export_viewer_layers(mgr, pairs, data_path, targets):
    """Write the paired layers to data_path in the images/ + labels/ layout the trainer's
    ZarrDataset reads (file names are indexes, not layer names), and set the run's
    data_path and targets to them."""
    for index, (img_layer, labels) in enumerate(pairs):
        volume_id = f"image{index}"
        _write_layer_zarr(data_path / "images" / f"{volume_id}.zarr", img_layer.data)
        for target, lbl_layer in labels:
            _write_layer_zarr(data_path / "labels" / f"{volume_id}_{target}.zarr", lbl_layer.data)
            print(f"Exported '{img_layer.name}' + '{lbl_layer.name}' as {volume_id}, target '{target}'")

    mgr.data_path = data_path
    mgr.set_targets_and_data(targets, {})
    print(f"Wrote the viewer's layers to {data_path} (targets: {', '.join(targets)})")


def _write_patch_cache(mgr, min_labeled_percentage):
    """Select the training patches and write them as the patch cache ZarrDataset loads.

    As in vesuvius.find_patches, the first target's labels decide: a patch qualifies when at
    least the widget's percentage of its voxels equal valid_patch_value (any non-zero value
    when none is set). Voxels with the target's ignore label never count. The cache key uses
    the parameters ZarrDataset looks the cache up with. Returns the selected patches as
    [(volume id, position), ...] in the order the dataset reads them."""
    from vc_reader.reader import open_node
    from vesuvius.models.datasets.find_valid_patches import find_valid_patches
    from vesuvius.models.preprocessing.patches.cache import (
        PatchCacheData, PatchEntry, build_cache_params, save_patch_cache)

    data_path = Path(mgr.data_path)
    patch_size = tuple(int(v) for v in mgr.train_patch_size)
    target = next(name for name, info in mgr.targets.items() if not info.get("auxiliary_task", False))
    info = mgr.targets[target]
    valid_patch_value = info.get("valid_patch_value")  # the same on every target (_check_patch_selection_rule)
    if valid_patch_value is None:
        valid_patch_value = (getattr(mgr, "dataset_config", None) or {}).get("valid_patch_value")
    ignore = _ignore_value(info)
    volume_ids = sorted(p.stem for p in (data_path / "images").iterdir() if p.suffix == ".zarr")

    params = build_cache_params(
        data_path=data_path,
        volume_ids=volume_ids,
        patch_size=patch_size,
        min_labeled_ratio=float(getattr(mgr, "min_labeled_ratio", 0.10)),
        bbox_threshold=float(getattr(mgr, "min_bbox_percent", 0.95)),
        valid_patch_find_resolution=int(getattr(mgr, "valid_patch_find_resolution", 1)),
        valid_patch_value=valid_patch_value,
        unlabeled_fg_enabled=bool(getattr(mgr, "unlabeled_foreground_enabled", False)),
        unlabeled_fg_threshold=float(getattr(mgr, "unlabeled_foreground_threshold", 0.05)),
        unlabeled_fg_bbox_threshold=float(getattr(mgr, "unlabeled_foreground_bbox_threshold", 0.15)),
    )
    found = find_valid_patches(
        label_arrays=[open_node(data_path / "labels" / f"{v}_{target}.zarr") for v in volume_ids],
        label_names=volume_ids,
        patch_size=patch_size,
        bbox_threshold=params.bbox_threshold,
        label_threshold=params.min_labeled_ratio,
        valid_patch_find_resolution=params.valid_patch_find_resolution,
        ignore_labels=[ignore] * len(volume_ids),
        valid_patch_values=[valid_patch_value] * len(volume_ids) if valid_patch_value is not None else None,
        collect_unlabeled_fg=False,
    )
    fg = [PatchEntry(volume_idx=p["volume_idx"], volume_name=p["volume_name"], position=tuple(p["start_pos"]))
          for p in found["fg_patches"]]
    if not fg:
        raise ValueError(
            f"No {'x'.join(map(str, patch_size))} patch of the paired layers has at least "
            f"{min_labeled_percentage}% voxels labeled for target '{target}'"
            + (f" (ignore label {ignore} not counted)" if ignore is not None else "")
            + "; lower Min Labeled Percentage, the patch size, or label more.")
    save_patch_cache(data_path / ".patches_cache", params, PatchCacheData(fg_patches=fg, bg_patches=[],
                                                                           unlabeled_fg_patches=[]))
    print(f"{len(fg)} patches have at least {min_labeled_percentage}% voxels labeled for target '{target}'"
          + (f" (ignore label {ignore} not counted)" if ignore is not None else ""))
    return [(entry.volume_name, entry.position) for entry in fg]


def main():
    viewer = napari.Viewer()
    global _config_manager
    _config_manager = ConfigManager(verbose=True)
    default_config_path = DEFAULT_CONFIG_PATH
    _config_manager.load_config(default_config_path)
    print(f"Default config loaded from {default_config_path}")

    file_picker_widget = filespicker
    viewer.window.add_dock_widget(file_picker_widget, area='right', name="config file")
    viewer.window.add_dock_widget(run_training, area='right', name="training")
    viewer.window.add_dock_widget(inference_widget, area='right', name="inference")

    napari.run()

if __name__ == "__main__":
    main()
