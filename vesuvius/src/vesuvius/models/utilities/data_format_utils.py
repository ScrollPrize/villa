from pathlib import Path


def detect_data_format(data_path):
    """
    Automatically detect the data format based on file extensions in the input directory.

    Parameters
    ----------
    data_path : Path
        Path to the data directory containing images/ and labels/ subdirectories

    Returns
    -------
    str or None
        Detected format ('zarr' or 'image') or None if cannot be determined
    """
    data_path = Path(data_path)
    images_dir = data_path / "images"
    labels_dir = data_path / "labels"

    if not images_dir.exists():
        return None

    # Check for zarr directories and image files
    zarr_count = 0
    image_count = 0

    # Check images directory
    for item in images_dir.iterdir():
        if item.is_dir() and item.suffix == '.zarr':
            zarr_count += 1
        elif item.is_file() and item.suffix.lower() in ['.tif', '.tiff', '.png', '.jpg', '.jpeg']:
            image_count += 1

    # Also check labels directory if it exists
    if labels_dir.exists():
        for item in labels_dir.iterdir():
            if item.is_dir() and item.suffix == '.zarr':
                zarr_count += 1
            elif item.is_file() and item.suffix.lower() in ['.tif', '.tiff', '.png', '.jpg', '.jpeg']:
                image_count += 1

    # Determine format based on what was found
    if zarr_count > 0 and image_count == 0:
        # Only zarr files found
        return 'zarr'
    elif image_count > 0:
        # If there are any image files, it's image format
        # (even if there are zarr files too, as they may have been created during training)
        return 'image'
    else:
        # No recognized files found
        return None