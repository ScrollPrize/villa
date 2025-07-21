from pathlib import Path
from vesuvius.models.training.auxiliary_tasks import (
    preserve_auxiliary_targets,
    restore_auxiliary_targets,
    apply_auxiliary_tasks_from_config
)


def configure_targets(mgr, loss_list=None):
    """
    Detect available targets from the data directory and apply optional loss_list.
    """
    # Save existing auxiliary tasks before detection
    auxiliary_targets = preserve_auxiliary_targets(mgr.targets if hasattr(mgr, 'targets') and mgr.targets else {})

    # Detect data-based targets if not yet configured
    if not getattr(mgr, 'targets', None):
        data_path = Path(mgr.data_path)
        images_dir = data_path / "images"
        if not images_dir.exists():
            raise ValueError(f"Images directory not found: {images_dir}")

        targets = set()
        if mgr.data_format == "zarr":
            for d in images_dir.iterdir():
                if d.is_dir() and d.suffix == '.zarr' and '_' in d.stem:
                    targets.add(d.stem.rsplit('_', 1)[1])
        elif mgr.data_format.lower() == "image":
            for ext in ['*.tif', '*.tiff', '*.png', '*.jpg', '*.jpeg']:
                for f in images_dir.glob(ext):
                    if '_' in f.stem:
                        targets.add(f.stem.rsplit('_', 1)[1])
        elif mgr.data_format == "napari":
            print("Warning: target detection not implemented for napari format.")
        if targets:
            mgr.targets = {}
            for t in sorted(targets):
                mgr.targets[t] = {
                    "out_channels": 2,
                    "activation": "softmax",
                    "loss_fn": "CrossEntropyLoss"
                }
            print(f"Detected targets from data: {sorted(targets)}")
        else:
            print("No targets detected from data. Please configure targets in config file.")

    # Re-add auxiliary targets
    if auxiliary_targets:
        mgr.targets = restore_auxiliary_targets(mgr.targets, auxiliary_targets)
        print(f"Re-added auxiliary targets: {list(auxiliary_targets.keys())}")

    # Re-apply auxiliary tasks from config
    apply_auxiliary_tasks_from_config(mgr)

    # Apply loss_list to configured targets, if provided
    if loss_list:
        names = list(mgr.targets.keys())
        for i, tname in enumerate(names):
            fn = loss_list[i] if i < len(loss_list) else loss_list[-1]
            mgr.targets[tname]["loss_fn"] = fn
            print(f"Applied {fn} to target '{tname}'")