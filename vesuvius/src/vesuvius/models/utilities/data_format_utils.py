from pathlib import Path


def detect_data_format(data_path):

    data_path = Path(data_path)
    images_dir = data_path / "images"
    labels_dir = data_path / "labels"

    if not images_dir.exists():
        return None

    zarr_count = 0
    image_count = 0

    for item in images_dir.iterdir():
        if item.is_dir() and item.suffix == '.zarr':
            zarr_count += 1
        elif item.is_file() and item.suffix.lower() in ['.tif', '.tiff', '.png', '.jpg', '.jpeg']:
            image_count += 1

    if labels_dir.exists():
        for item in labels_dir.iterdir():
            if item.is_dir() and item.suffix == '.zarr':
                zarr_count += 1
            elif item.is_file() and item.suffix.lower() in ['.tif', '.tiff', '.png', '.jpg', '.jpeg']:
                image_count += 1

    if zarr_count > 0 and image_count == 0:
        return 'zarr'
    elif image_count > 0:
        # If there are any image files, it's image format
        # (even if there are zarr files too, as they may have been created during training)
        return 'image'
    else:
        return None