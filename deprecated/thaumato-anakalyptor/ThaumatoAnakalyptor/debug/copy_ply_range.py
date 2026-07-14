import os
import shutil
import argparse

def copy_files_in_range(src_folder, dest_folder, x_range, y_range, z_range):
    """
    Copy files that match the given ranges from src_folder to dest_folder.

    :param src_folder: Source folder where the files are located.
    :param dest_folder: Destination folder where the files will be copied.
    :param x_range: A tuple (start, end) for the x values.
    :param y_range: A tuple (start, end) for the y values.
    :param z_range: A tuple (start, end) for the z values.
    """
    # Ensure the destination folder exists
    os.makedirs(dest_folder, exist_ok=True)

    # Loop through each file in the source directory
    for filename in os.listdir(src_folder):
        # Check if the file name matches the pattern "cell_yxz_*_*_*.ply"
        if filename.startswith("cell_yxz_") and filename.endswith(".ply"):
            # Extract the numbers from the filename
            _, _, x, y, z = filename.split('_')[:5]
            x, y, z = int(x), int(y), int(z[:-4])  # z[:-4] to exclude .ply extension

            # Check if the values are within the given ranges
            if x_range[0] <= x <= x_range[1] and y_range[0] <= y <= y_range[1] and z_range[0] <= z <= z_range[1]:
                # Copy the file to the destination directory
                shutil.copy(os.path.join(src_folder, filename), dest_folder)


if __name__ == "__main__":
    # Usage
    src_folder = "point_cloud"
    dest_folder = "point_cloud_debug"

    # Args
    x_range = (13, 14)
    y_range = (21, 23)
    z_range = (25, 27)

    parser = argparse.ArgumentParser(description='Copy .ply files in subdirectories based on specified ranges.')
    parser.add_argument('--src_folder', type=str, help='Source folder where the files are located.', default=src_folder)
    parser.add_argument('--dest_folder', type=str, help='Destination folder where the files will be copied.', default=dest_folder)

    parser.add_argument('--x_range', type=int, nargs=2, help='A tuple (start, end) for the x values.', default=x_range)
    parser.add_argument('--y_range', type=int, nargs=2, help='A tuple (start, end) for the y values.', default=y_range)
    parser.add_argument('--z_range', type=int, nargs=2, help='A tuple (start, end) for the z values.', default=z_range)

    args = parser.parse_args()

    src_folder = args.src_folder
    dest_folder = args.dest_folder
    x_range = args.x_range
    y_range = args.y_range
    z_range = args.z_range

    print(f"Args: {src_folder}, {dest_folder}, {x_range}, {y_range}, {z_range}")


    copy_files_in_range(src_folder, dest_folder, x_range, y_range, z_range)
