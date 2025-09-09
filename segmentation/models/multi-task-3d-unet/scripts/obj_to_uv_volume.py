import numpy as np
import zarr
import os
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

###############################################################################
# Global in-memory cache for OBJ data
###############################################################################
OBJ_CACHE = {}

def load_objs_into_memory(obj_paths):
    """
    Parse all OBJ files exactly once, storing vertices & uvs in a global OBJ_CACHE
    so we don't repeatedly read them for each chunk.
    """
    for obj_path in obj_paths:
        if not os.path.exists(obj_path):
            # Skip if file not found
            OBJ_CACHE[obj_path] = None
            continue

        vertices = []
        uvs = []

        with open(obj_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('v '):
                    try:
                        _, vx, vy, vz = line.split(maxsplit=3)
                        vertices.append([float(vx), float(vy), float(vz)])
                    except (ValueError, IndexError):
                        continue
                elif line.startswith('vt '):
                    try:
                        _, u, v = line.split(maxsplit=2)
                        uvs.append([float(u), float(v)])
                    except (ValueError, IndexError):
                        continue

        # Convert lists to numpy arrays
        if len(vertices) > 0 and len(uvs) > 0:
            vertices = np.array(vertices, dtype=np.float32)
            uvs = np.array(uvs, dtype=np.float32)
            OBJ_CACHE[obj_path] = {
                "vertices": vertices,
                "uvs": uvs
            }
        else:
            # If no valid data found
            OBJ_CACHE[obj_path] = None


def process_obj_chunk(obj_path, chunk_bounds, merge_strategy="last"):
    """
    Process a single OBJ file for a specific chunk of the volume,
    now using the global OBJ_CACHE instead of re-reading the file.
    """
    z_start, z_end, y_start, y_end, x_start, x_end = chunk_bounds
    chunk_shape = (z_end - z_start, y_end - y_start, x_end - x_start, 2)
    local_uv = np.zeros(chunk_shape, dtype=np.float32)

    local_count = None
    if merge_strategy == "average":
        local_count = np.zeros(chunk_shape[:-1], dtype=np.int32)

    # If we never loaded data for this path or file doesn't exist
    if obj_path not in OBJ_CACHE or OBJ_CACHE[obj_path] is None:
        # Just return the empty arrays
        return local_uv, local_count

    # Retrieve pre-loaded data
    vertices = OBJ_CACHE[obj_path]["vertices"]
    uvs = OBJ_CACHE[obj_path]["uvs"]
    if vertices is None or uvs is None:
        return local_uv, local_count

    # Convert to voxel indices
    voxel_indices = np.round(vertices).astype(int)

    # Filter vertices within chunk bounds
    chunk_mask = (
        (voxel_indices[:, 0] >= z_start) & (voxel_indices[:, 0] < z_end) &
        (voxel_indices[:, 1] >= y_start) & (voxel_indices[:, 1] < y_end) &
        (voxel_indices[:, 2] >= x_start) & (voxel_indices[:, 2] < x_end)
    )

    chunk_vertices = voxel_indices[chunk_mask]
    chunk_uvs = uvs[chunk_mask]

    # Adjust indices to chunk coordinates
    chunk_vertices[:, 0] -= z_start
    chunk_vertices[:, 1] -= y_start
    chunk_vertices[:, 2] -= x_start

    for (z_i, y_i, x_i), (u, v) in zip(chunk_vertices, chunk_uvs):
        if merge_strategy == "last":
            local_uv[z_i, y_i, x_i] = [u, v]
        elif merge_strategy in ["sum", "average"]:
            local_uv[z_i, y_i, x_i] += [u, v]
            if merge_strategy == "average":
                local_count[z_i, y_i, x_i] += 1

    return local_uv, local_count


def process_chunk(chunk_info):
    """
    Process all OBJ files for a single chunk.
    chunk_info = (obj_paths, chunk_bounds, merge_strategy)
    """
    obj_paths, chunk_bounds, merge_strategy = chunk_info
    z_start, z_end, y_start, y_end, x_start, x_end = chunk_bounds

    chunk_shape = (
        z_end - z_start,
        y_end - y_start,
        x_end - x_start,
        2
    )

    combined_uv = np.zeros(chunk_shape, dtype=np.float32)
    combined_count = None
    if merge_strategy == "average":
        combined_count = np.zeros(chunk_shape[:-1], dtype=np.int32)

    for obj_path in obj_paths:
        uv_chunk, count_chunk = process_obj_chunk(obj_path, chunk_bounds, merge_strategy)
        if merge_strategy == "last":
            mask = np.any(uv_chunk != 0, axis=-1)
            combined_uv[mask] = uv_chunk[mask]
        else:
            combined_uv += uv_chunk
            if merge_strategy == "average" and count_chunk is not None:
                combined_count += count_chunk

    if merge_strategy == "average" and combined_count is not None:
        nonzero_mask = (combined_count > 0)
        combined_uv[nonzero_mask] /= combined_count[nonzero_mask, np.newaxis]

    return combined_uv, chunk_bounds


def mesh_to_uv_volume_multi(obj_paths, reference_volume_shape, merge_strategy="last",
                            chunk_size=(128, 128, 128), num_processes=8):
    """
    Convert multiple OBJ files to a UV volume, processing in chunks,
    but now loading all OBJ data in memory first to avoid re-reading each file repeatedly.
    """
    if num_processes is None:
        num_processes = mp.cpu_count()

    Z, Y, X = reference_volume_shape

    # -------------------------------------------------------------------------
    # 1) Parse all OBJ files once, store them in global OBJ_CACHE
    # -------------------------------------------------------------------------
    load_objs_into_memory(obj_paths)

    # -------------------------------------------------------------------------
    # 2) Create zarr array for output
    # -------------------------------------------------------------------------
    store = zarr.open(
        '/mnt/raid_hdd/combined_uv_volume.zarr',
        mode='w',
        shape=(Z, Y, X, 2),
        chunks=chunk_size + (2,),
        dtype=np.float32
    )

    # -------------------------------------------------------------------------
    # 3) Generate chunk bounds
    # -------------------------------------------------------------------------
    chunk_bounds_list = []
    for z in range(0, Z, chunk_size[0]):
        for y in range(0, Y, chunk_size[1]):
            for x in range(0, X, chunk_size[2]):
                chunk_bounds = (
                    z, min(z + chunk_size[0], Z),
                    y, min(y + chunk_size[1], Y),
                    x, min(x + chunk_size[2], X)
                )
                chunk_bounds_list.append((obj_paths, chunk_bounds, merge_strategy))

    # -------------------------------------------------------------------------
    # 4) Process chunks in parallel
    # -------------------------------------------------------------------------
    with mp.Pool(num_processes) as pool:
        for uv_chunk, bounds in tqdm(
            pool.imap(process_chunk, chunk_bounds_list),
            total=len(chunk_bounds_list),
            desc="Processing chunks"
        ):
            z_start, z_end, y_start, y_end, x_start, x_end = bounds
            store[z_start:z_end, y_start:y_end, x_start:x_end] = uv_chunk

    return store


if __name__ == "__main__":
    obj_files = [
        "20230702185753.obj",
        "20230929220926.obj",
        "20231005123336.obj",
        "20231007101619.obj",
        "20231012184424.obj",
        "20231016151002.obj",
        "20231022170901.obj",
        "20231031143852.obj",
        "20231106155351.obj",
        "20231210121321.obj",
        "20231221180251.obj",
        "20240218140920.obj",
        "20240221073650.obj",
        "20240222111510.obj",
        "20240223130140.obj",
        "20240227085920.obj",
        "20240301161650.obj"
    ]

    volume_shape = (14376, 7888, 8096)
    chunk_size = (128, 128, 128)  # Adjust based on available RAM

    combined_uv = mesh_to_uv_volume_multi(
        obj_paths=obj_files,
        reference_volume_shape=volume_shape,
        merge_strategy="last",
        chunk_size=chunk_size
    )
    print("Successfully processed UV volume")
