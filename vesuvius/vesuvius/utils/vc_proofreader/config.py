# config.py

scroll = "1451"
config = {
    # Zarr paths can be local file system paths or HTTP URLs.
    "image_zarr": "/mnt/raid_nvme/volpkgs/PHerc1451.volpkg/volumes/4.320um_1.0m_116keV_binmean_2_PHerc1451.zarr",  # Replace with your image zarr path
    "label_zarr": "/mnt/raid_nvme/volpkgs/PHerc1451.volpkg/volumes/1451_nnunetpredictor_059_ome.zarr",    # Replace with your label zarr path

    #/mnt/raid_nvme/ced_volumes/s1_paths_vx_ced_raw_ome.zarr

    # Output directory where subfolders "imagesTr" and "labelsTr" will be created.
    "dataset_out_path": f"/mnt/raid_nvme/new_train_labels/{scroll}/",

    # Patch extraction settings (patch_size is specified in display resolution units).
    "patch_size": 256,
    "sampling": "sequence",  # Options: "sequence" or "random"

    # Progress saving options.
    "save_progress": True,                                    # Set to True to enable saving progress.
    "progress_file": f"/mnt/raid_nvme/esrf_labels/{scroll}/progress.json",                # Path to the file where progress is saved.
    
    # Output zarr for approved labels
    "output_label_zarr": f"/mnt/raid_nvme/esrf_labels/{scroll}/labels.zarr",      # Path for the output label zarr (binary uint8)
}
