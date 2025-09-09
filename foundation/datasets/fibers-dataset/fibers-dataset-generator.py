# Giorgio Angelotti - 2024
import argparse
import os
import webknossos as wk
from webknossos import webknossos_context
from webknossos import Annotation, webknossos_context
import numpy as np
import tifffile
from scipy.ndimage import distance_transform_edt
from tools import detect_vesselness
from tqdm import tqdm
# import vesuvius
# from vesuvius import Volume

def interpolate_adaptive(start_pos, end_pos, curvature_threshold=0.1, max_recursion=100):
    """Adaptive interpolation between two nodes based on curvature and resolution."""
    segment_vector = end_pos - start_pos
    segment_length = np.linalg.norm(segment_vector)

    # Base case: Stop recursion if max recursion is reached or segment length is below threshold
    if max_recursion == 0 or segment_length < curvature_threshold:
        return [start_pos, end_pos]

    # Midpoint sampling
    mid_pos = (start_pos + end_pos) / 2

    # Recursive subdivision
    left_segment = interpolate_adaptive(start_pos, mid_pos, curvature_threshold, max_recursion - 1)
    right_segment = interpolate_adaptive(mid_pos, end_pos, curvature_threshold, max_recursion - 1)

    # Combine results, removing the duplicated midpoint
    return left_segment[:-1] + right_segment

def voxelize_skeleton(annotation, output_shape=(100, 100, 100), origins=(0, 0, 0)):    
    # Create an empty voxel grid
    voxel_grid = np.zeros(output_shape, dtype=np.uint8)
    origins = np.array(origins)
    for group in tqdm(annotation.skeleton.groups, desc="Groups"):
        for tree in tqdm(group.trees, desc="Trees in groups"):
            temp_fiber = np.zeros_like(voxel_grid)
            for node1, node2 in tree.edges:
                # Perform adaptive interpolation
                node1_pos = np.array([node1.position.x, node1.position.y, node1.position.z])
                node2_pos = np.array([node2.position.x, node2.position.y, node2.position.z])
                interpolated_points = interpolate_adaptive(node1_pos, node2_pos)
                # Convert points to voxel space and filter bounds
                for point in interpolated_points:
                    voxel_coords = (point - origins).astype(int)
                    if np.all((0 <= voxel_coords) & (voxel_coords < np.asarray(output_shape))):
                        temp_fiber[tuple(voxel_coords)] = 1
            voxel_grid = np.maximum(voxel_grid, temp_fiber)
    for tree in tqdm(annotation.skeleton.trees, desc="Trees in root"):
        temp_fiber = np.zeros_like(voxel_grid)
        for node1, node2 in tree.edges:
            # Perform adaptive interpolation
            node1_pos = np.array([node1.position.x, node1.position.y, node1.position.z])
            node2_pos = np.array([node2.position.x, node2.position.y, node2.position.z])
            interpolated_points = interpolate_adaptive(node1_pos, node2_pos)
            # Convert points to voxel space and filter bounds
            for point in interpolated_points:
                voxel_coords = (point - origins).astype(int)
                if np.all((0 <= voxel_coords) & (voxel_coords < np.asarray(output_shape))):
                    temp_fiber[tuple(voxel_coords)] = 1
        voxel_grid = np.maximum(voxel_grid, temp_fiber)
    binary_inverted = 1 - voxel_grid
    # Compute the Euclidean Distance Transform
    edt = distance_transform_edt(binary_inverted)
    expanded_structure = edt <= 3

    # Detect vesselness
    vessel = detect_vesselness(expanded_structure.astype(np.float32))
    vessel = np.maximum(voxel_grid, vessel)

    # Binning
    bin_edges = np.histogram_bin_edges(vessel, bins=2)
    binned_data = np.digitize(vessel, bin_edges[1:]).astype(np.uint8)
    
    # *** Set all non-zero values to 1 ***
    binned_data[binned_data > 0] = 1

    return np.transpose(binned_data, (2,1,0))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Voxelize a skeleton .nml and save the annotation + volume chunk."
    )
    parser.add_argument("--nml_path", required=True, help="Path to the .nml file.")
    # parser.add_argument("--size", type=int, required=True, help="Chunk size.")
    parser.add_argument("--output_folder", required=True, help="Output folder path.")
    args = parser.parse_args()

    # Create folder structure if it does not exist
    labels_folder = os.path.join(args.output_folder, "labelsTr")
    images_folder = os.path.join(args.output_folder, "imagesTr")
    os.makedirs(labels_folder, exist_ok=True)
    os.makedirs(images_folder, exist_ok=True)

    # Load the annotation
    annotation = Annotation.load(args.nml_path)
    # size = args.size

    # Inspect the loaded annotation
    print(f"Annotation Name: {annotation.name}")

    # Parse the annotation name to extract coordinates
    parts = annotation.name.split('_')
    # Extract and convert the values
    scroll_id = parts[1]

    if scroll_id == "s1a": scroll_id = "s1"
    
    z_start = int(parts[2][:-1])
    y_start = int(parts[3][:-1])
    x_start = int(parts[4][:-1])
    size = int(parts[5])
    print(f"Scroll: {scroll_id}")
    print(f"Starting coordinates: {z_start},{y_start},{x_start}. Chunk size {size}")

    print("Voxelizing...")
    # Voxelize skeleton
    voxel_grid = voxelize_skeleton(
        annotation, 
        output_shape=(size, size, size), 
        origins=(x_start, y_start, z_start)
    )

    # Write annotation labels
    labels_filename = os.path.join(labels_folder, f"{scroll_id}_{z_start:05d}_{y_start:05d}_{x_start:05d}_{size}.tif")
    print(f"Writing annotation to {labels_filename}...")
    tifffile.imwrite(labels_filename, voxel_grid)
    print("Annotation wrote.")

    # Load and write image chunk
    images_filename = os.path.join(images_folder, f"{scroll_id}_{z_start:05d}_{y_start:05d}_{x_start:05d}_{size}_0000.tif")
    print(f"Writing image chunk to {images_filename}...")

    print("Loading volume (using webknossos)...")

    WK_URL = "http://dl.ash2txt.org:8080"

    # Open and read the token file
    with open("token.txt", "r") as file:
        TOKEN = file.read().strip()

    print(f"Loaded TOKEN: {TOKEN}")

    ORGANIZATION_ID = "Scroll_Prize"

    # define bounding box
    bb = wk.NDBoundingBox(topleft=(x_start, y_start, z_start), size=(size, size, size), index=(0,1,2), axes=('x', 'y', 'z'))


    if int(scroll_id[1:]) == 1:
        dataset_name = "scroll1a"
    elif int(scroll_id[1:]) == 5:
        dataset_name = "scroll5-full"

    with webknossos_context(url=WK_URL, token=TOKEN):
        ds = wk.Dataset.open_remote(dataset_name, ORGANIZATION_ID)

        volume = ds.get_layer("volume")
        view = volume.get_mag("1").get_view(absolute_bounding_box=bb)
        data = np.clip(view.read()[0].astype(np.float64)/257,0,255).astype(np.uint8)
        data = np.transpose(data, (2, 1, 0))

        tifffile.imwrite(images_filename, data)

    print("Done")
