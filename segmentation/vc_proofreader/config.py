# config.py

config = {
    # Zarr paths can be local file system paths or HTTP URLs.
    "image_zarr": "/mnt/raid_hdd/scrolls/s5/s5.zarr/",  # Replace with your image zarr path
    "label_zarr": "/mnt/raid_nvme/s5_voxelize.zarr/",    # Replace with your label zarr path

    # Output directory where subfolders "imagesTr" and "labelsTr" will be created.
    "dataset_out_path": "/mnt/raid_nvme/s5_320",

    # Patch extraction settings (patch_size is specified in display resolution units).
    "patch_size": 320,
    "sampling": "sequence",  # Options: "sequence" or "random"

    # Progress saving options.
    "save_progress": True,                                    # Set to True to enable saving progress.
    "progress_file": "/mnt/raid_nvme/s5_320/progress.txt",                # Path to the file where progress is saved.
}
