import os
import datetime

def save_train_val_filenames(self, train_dataset, val_dataset, train_indices, val_indices):
    """
    Save the filenames of the volumes used in training and validation sets along with patch locations.

    Parameters
    ----------
    train_dataset : Dataset
        Training dataset
    val_dataset : Dataset
        Validation dataset
    train_indices : list
        Indices used for training patches
    val_indices : list
        Indices used for validation patches
    """
    import json

    # Extract volume information and patch locations from datasets
    train_patches = []
    val_patches = []
    train_volumes = set()
    val_volumes = set()

    # Get volume IDs and patch locations for training set
    for idx in train_indices:
        patch_info = train_dataset.valid_patches[idx]
        vol_idx = patch_info["volume_index"]
        position = patch_info["position"]  # [z, y, x] coordinates

        # Get volume ID if available
        volume_id = f"volume_{vol_idx}"  # Default if no volume_ids
        if hasattr(train_dataset, 'volume_ids'):
            first_target = list(train_dataset.volume_ids.keys())[0]
            if vol_idx < len(train_dataset.volume_ids[first_target]):
                volume_id = train_dataset.volume_ids[first_target][vol_idx]

        train_volumes.add(volume_id)
        train_patches.append({
            "patch_index": idx,
            "volume_id": volume_id,
            "volume_index": vol_idx,
            "position": position  # [z, y, x] coordinates
        })

    # Get volume IDs and patch locations for validation set
    for idx in val_indices:
        patch_info = val_dataset.valid_patches[idx]
        vol_idx = patch_info["volume_index"]
        position = patch_info["position"]  # [z, y, x] coordinates

        # Get volume ID if available
        volume_id = f"volume_{vol_idx}"  # Default if no volume_ids
        if hasattr(val_dataset, 'volume_ids'):
            first_target = list(val_dataset.volume_ids.keys())[0]
            if vol_idx < len(val_dataset.volume_ids[first_target]):
                volume_id = val_dataset.volume_ids[first_target][vol_idx]

        val_volumes.add(volume_id)
        val_patches.append({
            "patch_index": idx,
            "volume_id": volume_id,
            "volume_index": vol_idx,
            "position": position  # [z, y, x] coordinates
        })

    # Save split information with patch details
    split_info = {
        "metadata": {
            "train_patch_count": len(train_indices),
            "val_patch_count": len(val_indices),
            "train_volume_count": len(train_volumes),
            "val_volume_count": len(val_volumes),
            "train_volumes": sorted(list(train_volumes)),
            "val_volumes": sorted(list(val_volumes)),
            "train_val_split": self.mgr.tr_val_split,
            "patch_size": self.mgr.train_patch_size,
            "timestamp": datetime.datetime.now().isoformat()
        },
        "train_patches": train_patches,
        "val_patches": val_patches
    }

    return split_info



