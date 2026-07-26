### Julian Schilliger - ThaumatoAnakalyptor - Vesuvius Challenge 2023

import os
import tifffile
import numpy as np
import shutil
import tqdm
import numpy as np

def sort_tif_by_mean(src_folder, dest_folder, threshold):
    """
    Moves .tif files from the src_folder to dest_folder if their mean value is 
    smaller than the given threshold.
    
    :param src_folder: Source directory path.
    :param dest_folder: Destination directory path.
    :param threshold: Mean value threshold.
    """
    
    # Ensure destination folder exists
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    files = os.listdir(src_folder)
    # Shuffle
    np.random.shuffle(files)

    # Loop through all files in the source folder
    for filename in tqdm.tqdm(files):
        if filename.endswith('.tif'):
            file_path = os.path.join(src_folder, filename)
            
            with tifffile.TiffFile(file_path) as tif:
                volume = tif.asarray()
            
                # Compute mean of the volume
                mean_value = np.mean(volume)
                
                # If the mean is below the threshold, move the file
                if mean_value < threshold:
                    print(f"Moving {filename} with mean value {mean_value} to {dest_folder}")
                    shutil.move(file_path, os.path.join(dest_folder, filename))
                    print("Done.")

def compute_mean_of_tif_volume(tif_path):
    """
    Compute and print the mean value of a .tif volume.
    
    :param tif_path: Path to the .tif file.
    """
    with tifffile.TiffFile(tif_path) as tif:
        volume = tif.asarray()
        
        # Ensure the array is 3D (assuming single channel data)
        if len(volume.shape) != 3:
            raise ValueError("The provided TIF does not seem to be a 3D volume.")
        
        mean_value = np.mean(volume)
        
        print(f"Mean value of the .tif volume: {mean_value}")

'''# Example usage
tif_path = "../scroll1_grids/cell_yxz_002_003_026.tif"
compute_mean_of_tif_volume(tif_path)

# second path
tif_path = "../scroll1_grids/cell_yxz_003_003_027.tif"
compute_mean_of_tif_volume(tif_path)

# third path, filled volume
tif_path = "../scroll1_grids/cell_yxz_003_008_015.tif"
compute_mean_of_tif_volume(tif_path)'''

# filled mean value: 33774.121804304
# empirical empty threshold value: 25259.996749536

src = "scroll3_grids"
dst = "scroll3_grids_empty"
thresh = 26000  # Empty tif volumes have less bright spots -> lower mean value, empirical threshold
sort_tif_by_mean(src, dst, thresh)