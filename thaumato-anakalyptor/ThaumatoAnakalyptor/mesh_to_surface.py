### Julian Schilliger - ThaumatoAnakalyptor - 2024

import multiprocessing
import open3d as o3d
import argparse
import os
import tempfile
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore
# Set max_split_size_mb to a smaller value
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'  # or another value you deem appropriate
import torch.distributed
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import normalize
import pytorch_lightning as pl
from pytorch_lightning.callbacks import BasePredictionWriter
from .rendering_utils.interpolate_image_3d import extract_from_image_4d
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import tifffile
import cv2
import zarr
from multiprocessing import cpu_count, shared_memory
import time
import warnings

class MyPredictionWriter(BasePredictionWriter):
    def __init__(self, save_path, image_size, r, r_steps=None, max_queue_size=10, max_workers=1, display=True):
        super().__init__(write_interval="batch")  # or "epoch" for end of an epoch
        self.save_path = save_path
        self.image_size = image_size
        self.display = display
        self.image = None
        # This composite will be updated after each segment.
        self.composite = np.zeros(self.image_size, dtype=np.uint16)
        self.r = r
        self.r_steps = r_steps  # new argument: number of layers to process at a time
        self.surface_volume_np = None
        self.max_queue_size = max(max_queue_size, max_workers)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)  # Adjust number of workers as needed
        self.semaphore = Semaphore(self.max_queue_size)
        self.futures = []
        self.num_workers = cpu_count()
        self.trainer_rank = None
        # For segmented rendering, track the current global layer offset
        self.current_step_offset = 0
    
    def write_on_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, prediction, batch_indices, batch, batch_idx: int, dataloader_idx: int) -> None:
        # if self.trainer_rank is None: # Only set the rank once
            # self.trainer_rank = trainer.global_rank if trainer.world_size > 1 else 0

        if self.surface_volume_np is None:
            if trainer.global_rank == 0:
                if self.r_steps is not None:
                    seg_layers = min(self.r_steps, (2*self.r+1 - self.current_step_offset))
                    self.surface_volume_np, self.shm, success = self.create_shared_array((seg_layers, self.image_size[0], self.image_size[1]), np.uint16, name=f"surface_volume_{self.current_step_offset}")
                else:
                    self.surface_volume_np, self.shm, success = self.create_shared_array((2*self.r+1, self.image_size[0], self.image_size[1]), np.uint16, name="surface_volume")
                self.trainer_rank = 0 if success else None
                # Gather the shared memory name
                torch.distributed.barrier()
            else:
                if self.trainer_rank is None:
                    self.trainer_rank = 1
                torch.distributed.barrier()
                if self.r_steps is not None:
                    seg_layers = min(self.r_steps, (2*self.r+1 - self.current_step_offset))
                    self.surface_volume_np, self.shm = self.attach_shared_array((seg_layers, self.image_size[0], self.image_size[1]), np.uint16, name=f"surface_volume_{self.current_step_offset}")
                else:
                    self.surface_volume_np, self.shm = self.attach_shared_array((2*self.r+1, self.image_size[0], self.image_size[1]), np.uint16, name="surface_volume")
        self.semaphore.acquire()  # Wait for a slot to become available if necessary
        future = self.executor.submit(self.process_and_write_data, prediction)
        future.add_done_callback(lambda _future: self.semaphore.release())
        self.futures.append(future)
        # display progress
        self.display_progress()

    def process_display_progress(self):
        if not self.display:
            return
        
        try:
            # display progress cv2.imshow
            image = (self.surface_volume_np[self.r if self.r_steps is None else self.r_steps//2].astype(np.float32) / 65535)
            screen_y = 2560
            screen_x = 1440

            if (screen_y * image.shape[1])//image.shape[0] > screen_x:
                screen_y = (screen_x * image.shape[0])//image.shape[1]
            else:
                screen_x = (screen_y * image.shape[1])//image.shape[0]

            image = cv2.resize(image, (screen_x, screen_y))
            image = image.T
            image = image[::-1, :]
            self.image = image
        except Exception as e:
            print(e)
            pass

    def display_progress(self):
        if not self.display:
            return
        
        try:
            if self.image is None:
                return
            cv2.imshow("Surface Volume", self.image)
            cv2.waitKey(1)
        except Exception as e:
            print(e)
            pass

    def create_shared_array(self, shape, dtype, name="shared_array"):
        array_size = np.prod(shape) * np.dtype(dtype).itemsize
        success = True
        try:
            # Create a shared array
            shm = shared_memory.SharedMemory(create=True, size=array_size, name=name)
        except FileExistsError:
            print(f"Shared memory with name {name} already exists.")
            # Clean up the shared memory if it already exists
            shm = shared_memory.SharedMemory(create=False, size=array_size, name=name)
            success = False

        arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        arr.fill(0)  # Initialize the array with zeros
        return arr, shm, success
    
    def attach_shared_array(self, shape, dtype, name="shared_array"):
        while True:
            try:
                # Attach to an existing shared array
                shm = shared_memory.SharedMemory(name=name)
                arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
                assert arr.shape == shape, f"Expected shape {shape} but got {arr.shape}"
                assert arr.dtype == dtype, f"Expected dtype {dtype} but got {arr.dtype}"
                print("Attached to shared memory")
                return arr, shm
            except FileNotFoundError:
                time.sleep(0.2)

    def process_and_write_data(self, prediction):
        try:
            # print("Writing to Numpy")
            if prediction is None:
                return
            if len(prediction) == 0:
                return

            values, indexes_3d = prediction
            
            if indexes_3d.shape[0] == 0:
                return

            # If segmented rendering is enabled, only write points in the current segment
            if self.r_steps is not None:
                seg_start = self.current_step_offset
                seg_end = seg_start + self.surface_volume_np.shape[0]
                mask = (indexes_3d[:, 0] >= seg_start) & (indexes_3d[:, 0] < seg_end)
                if not np.any(mask):
                    return
                # Adjust the layer indices to be local to the segment
                indexes_local = indexes_3d[mask].copy()
                indexes_local[:, 0] -= seg_start
                self.surface_volume_np[indexes_local[:, 0], indexes_local[:, 1], indexes_local[:, 2]] = values[mask]
            else:
                # save into surface_volume_np
                self.surface_volume_np[indexes_3d[:, 0], indexes_3d[:, 1], indexes_3d[:, 2]] = values

            # display progress
            self.process_display_progress()
        except Exception as e:
            print(e)
            

    def wait_for_all_writes_to_complete(self):
        # Wait for all queued tasks to complete
        for future in tqdm(self.futures, desc="Finalizing writes"):
            future.result()
        # Wait for all GPU writes to complete
        torch.distributed.barrier()

    def write_to_disk(self, flag='jpg'):
        # Make sure this method is called after the prediction loop is complete
        print("Waiting for all writes to complete")
        self.wait_for_all_writes_to_complete()
        if self.trainer_rank != 0: # Only rank 0 should write to disk
            self.shm.close()
            self.shm = None
            return
        print("Writing Segment to disk")
        # Make folder if it does not exist
        os.makedirs(self.save_path, exist_ok=True)

        if flag == 'tif' or flag == 'tiff':
            self.write_tif()
        elif flag == 'jpg':
            self.write_jpg()
        elif flag == 'memmap':
            self.write_memmap()
        elif flag == 'npz':
            self.write_npz()
        elif flag == 'zarr':
            self.write_zarr()
        else:
            print("Invalid flag. Choose between 'tif', 'jpg', 'memmap', 'npz', 'zarr'")
            return
        
        # Close the shared memory
        if self.trainer_rank == 0:
            try:
                self.shm.close()
                # No warnings verbose
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.shm.unlink()
                    self.shm = None
            except Exception as e:
                print(e)
        print("Segment written to disk")
        # Reset surface_volume_np to allow reinitialization for the next segment
        self.surface_volume_np = None

    def compute_composite(self):
        # Compute composite (max) image from the current segment (surface_volume_np)
        for i in range(self.surface_volume_np.shape[0]):
            self.composite = np.maximum(self.composite, self.surface_volume_np[i])
        return self.composite

    def write_tif(self):
        def save_tif(i, filename):
            image = self.surface_volume_np[i]
            image = image.T
            image = image[::-1, :]
            tifffile.imsave(filename, image)

        with ThreadPoolExecutor(self.num_workers) as executor:
            futures = []
            seg_offset = self.current_step_offset
            for i in range(self.surface_volume_np.shape[0]):
                global_layer = seg_offset + i
                i_str = str(global_layer).zfill(len(str(2*self.r+1)))
                filename = os.path.join(self.save_path, f"{i_str}.tif")
                futures.append(executor.submit(save_tif, i, filename))

            # Wait for all futures to complete if needed
            for future in tqdm(as_completed(futures), desc="Writing TIF"):
                future.result()

        # Create Composite max image from all tifs in the current segment
        composite_image = self.compute_composite()

        composite_image = composite_image.astype(np.uint16)
        composite_image = composite_image.T
        composite_image = composite_image[::-1, :]
        tifffile.imsave(os.path.join(os.path.dirname(self.save_path), "composite.tif"), composite_image)


    def write_jpg(self, quality=60):  # Adjust the default quality value as needed
        def save_jpg(i, filename):
            image = self.surface_volume_np[i]
            # Normalize to 0-1, then scale to 0-255
            image = (image / 256).astype(np.uint8)
            image = image.T
            image = image[::-1, :]
            cv2.imwrite(filename, image, [cv2.IMWRITE_JPEG_QUALITY, quality])

        with ThreadPoolExecutor(self.num_workers) as executor:
            futures = []
            seg_offset = self.current_step_offset
            for i in range(self.surface_volume_np.shape[0]):
                global_layer = seg_offset + i
                i_str = str(global_layer).zfill(len(str(2*self.r+1)))
                filename = os.path.join(self.save_path, f"{i_str}.jpg")
                futures.append(executor.submit(save_jpg, i, filename))

            # Wait for all futures to complete if needed
            for future in tqdm(as_completed(futures), desc="Writing JPG"):
                future.result()

        # Create Composite max image from all jpgs in the current segment
        composite_image = self.compute_composite()

        composite_image = composite_image.astype(np.uint16)
        composite_image = (composite_image / 256).astype(np.uint8)
        composite_image = composite_image.T
        composite_image = composite_image[::-1, :]
        cv2.imwrite(os.path.join(os.path.dirname(self.save_path), f"composite.jpg"), composite_image, [cv2.IMWRITE_JPEG_QUALITY, quality])

    def write_memmap(self):
        # save to disk as memmap
        memmap_path = os.path.join(self.save_path, "surface_volume")
        memmap = np.memmap(memmap_path, dtype='uint16', mode='w+', shape=self.surface_volume_np.shape)
        memmap[:] = self.surface_volume_np[:]
        del memmap

    def write_npz(self):
        # save to disk as npz
        npz_path = os.path.join(self.save_path, "surface_volume.npz")
        np.savez_compressed(npz_path, surface_volume=self.surface_volume_np)

    def write_zarr(self):
        # Define the chunk size
        chunk_size = (16, 16, 16)  # Example: Modify according to your needs

        # Define compression options
        compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=zarr.Blosc.SHUFFLE)  # Example compression

        # save to disk as zarr with chunks and compression
        zarr_path = os.path.join(self.save_path, "surface_volume.zarr")
        z = zarr.open(zarr_path, mode='w', shape=self.surface_volume_np.shape, dtype='uint16', chunks=chunk_size, compressor=compressor)
        z[:] = self.surface_volume_np
        
class MeshDataset(Dataset):
    """Dataset class for rendering a mesh."""
    def __init__(self, path, scroll, scroll_format="grid cells", output_path=None, grid_size=500, r=32, max_side_triangle=10, max_workers=1, display=False, r_steps=None):
        """Initialize the dataset."""
        self.grid_size = grid_size
        self.path = path
        self.scroll_format = scroll_format
        if self.scroll_format == "grid cells":
            self.grid_template = scroll
        if self.scroll_format == "zarr":
            self.zarr = self.load_zarr(scroll)
            # chunk_size = self.zarr.chunks
            # print(f"Chunk size: {chunk_size}")
            # self.grid_size = chunk_size[0]
        self.r = r+1
        self.max_side_triangle = max_side_triangle
        self.load_mesh(path)
        output_path = os.path.dirname(path) if output_path is None else output_path
        write_path = os.path.join(output_path, "layers")
        self.writer = MyPredictionWriter(write_path, self.image_size, r, r_steps=r_steps, max_workers=max_workers, display=display)

        self.grids_to_process = self.init_grids_to_process()

        self.adjust_triangle_sizes()

    def load_zarr(self, dirname):
        stack_array = zarr.open(dirname, mode="r")
        print("Contents of the group:", list(stack_array.groups()))
        print("Arrays in the group:", list(stack_array.arrays()))
        stack_array = stack_array[0]
        print(f"zarr shape: {stack_array.shape}")
        return stack_array

    def parse_mtl_for_texture_filenames(self, mtl_filepath):
        texture_filenames = []
        with open(mtl_filepath, 'r') as file:
            for line in file:
                if line.strip().startswith('map_Kd'):  # This line indicates a texture file in MTL
                    parts = line.split()
                    if len(parts) > 1:
                        texture_filenames.append(parts[1])  # The second part is the filename
        return texture_filenames
    
    def generate_mask_png(self):
        mask = np.zeros(self.image_size[::-1], dtype=np.uint8)
        for triangle in self.uv:
            triangle = triangle.astype(np.int32)
            try:
                cv2.fillPoly(mask, [triangle], 255)
            except:
                pass
        mask = mask[::-1, :]
        cv2.imwrite(os.path.join(os.path.dirname(self.path), os.path.basename(self.path).split(".")[0] + "_mask.png"), mask)
        
    def load_mesh(self, path):
        """Load the mesh from the given path and extract the vertices, normals, triangles, and UV coordinates."""
        # Get the working path and base name
        working_path = os.path.dirname(path)
        base_name = os.path.splitext(os.path.basename(path))[0]

        # Construct the potential paths for the .tif and .png files
        tif_path = os.path.join(working_path, f"{base_name}.tif")
        png_path = os.path.join(working_path, f"{base_name}.png")
        mtl_path = os.path.join(working_path, f"{base_name}.mtl")
        print(f"Names: {tif_path}, {png_path}, {mtl_path}", end="\n")

        # Check if the .tif version exists
        if os.path.exists(tif_path):
            image_path = tif_path
            print(f"Found TIF image at: {image_path}", end="\n")
        # If not, check if the .png version exists
        elif os.path.exists(png_path):
            image_path = png_path
            print(f"Found PNG image at: {image_path}", end="\n")
        elif os.path.exists(mtl_path):
            texture_filenames = self.parse_mtl_for_texture_filenames(mtl_path)
            if len(texture_filenames) > 0:
                image_path = os.path.join(working_path, texture_filenames[0])
                print(f"Found material texture image at: {image_path}", end="\n")
            else:
                image_path = None
                print("No corresponding TIF, PNG, or MTL image found.", end="\n")
        # If neither exists, handle the case (e.g., error message)
        else:
            image_path = None
            print("No corresponding TIF or PNG image found.", end="\n")
        
        print("Texture Image Name:", image_path)
        if image_path:
            # Open the image file
            with Image.open(image_path) as img:
                # Get dimensions
                y_size, x_size = img.size
                if y_size < 10 or x_size < 10:
                    print("Texture image too small. Exit.")
                    exit(1)
        else:
            print("No texture image found. Exit.")
            exit(1)
        print(f"Y-size: {y_size}, X-size: {x_size}", end="\n")

        print(f"Loading mesh from {path}", end="\n")
        # copy mesh to tempfile
        with tempfile.NamedTemporaryFile(suffix=".obj") as temp_file:
            # copy mesh to tempfile
            temp_path = temp_file.name
            # os copy
            os.system(f"cp {path} {temp_path}")
            # load mesh
            mesh = o3d.io.read_triangle_mesh(temp_path)
        self.mesh = mesh
        print(f"Loaded mesh from {path}", end="\n")

        self.vertices = np.asarray(self.mesh.vertices)
        self.normals = np.asarray(self.mesh.vertex_normals)
        self.triangles = np.asarray(self.mesh.triangles)
        uv = np.asarray(self.mesh.triangle_uvs).reshape(-1, 3, 2)
        # scale numpy UV coordinates to the image size
        self.uv = uv * np.array([y_size, x_size])
        self.image_size = (y_size, x_size)

        # Generate the mask image for foreground/background separation
        self.generate_mask_png()

        # vertices of triangles
        self.triangles_vertices = self.vertices[self.triangles]
        self.triangles_normals = self.normals[self.triangles]

    def adjust_triangle_sizes(self):
        """
        Adjusts the sizes of the triangles in the mesh based on the `max_side_triangle` parameter.

        This method splits the triangles that are too big into smaller triangles. It computes the size of each triangle
        in the x and y directions and compares it with the `max_side_triangle` parameter. If a triangle's side length
        exceeds the `max_side_triangle`, it is split into two smaller triangles.

        Returns:
            None
        """
        triangles_vertices = self.triangles_vertices
        triangles_normals = self.triangles_normals
        uv = self.uv

        # Print the original number of triangles
        print(f"Original triangles: {triangles_vertices.shape[0]}, {triangles_normals.shape[0]}, {uv.shape[0]}", end="\n")

        # Split the triangles that are too big as per the max_side_triangle parameter
        # Compute the size of each triangle in x and y
        # Side lengths for all triangles: T x 3
        uv_good = []
        triangles_vertices_good = []
        triangles_normals_good = []

        # Function to compute the current progress
        def current_progress(maxS):
            return np.log(2) - np.log(self.max_side_triangle / maxS)
        # Iterate over the triangles and adjust the sizes, show progress with tqdm
        with tqdm(total=100, desc="Adjusting triangle sizes") as pbar:
            start = None
            while True:
                # Compute the minimum and maximum UV coordinates for each triangle
                triangle_min_uv = np.min(uv, axis=1)
                triangle_max_uv = np.max(uv, axis=1)
                # Compute the side lengths for each triangle, ceil the max and floor the min to get the integer side lengths
                side_lengths = np.ceil(triangle_max_uv) - np.floor(triangle_min_uv)
                max_side_lengths = np.max(side_lengths, axis=0)

                # Update the progress bar
                if start is None:
                    # Compute the starting progress value
                    start = current_progress(max_side_lengths[0]) + current_progress(max_side_lengths[1])
                    progress = 0
                else:
                    # Compute the current progress and update the progress bar
                    now = current_progress(max_side_lengths[0]) + current_progress(max_side_lengths[1])
                    progress = max(1 - now / start, 0)
                pbar.n = int(progress * 100)
                pbar.refresh()

                # Check if any triangle is too large
                mask_large_side = np.any(side_lengths > self.max_side_triangle, axis=1)

                # Break without adding the last good triangles if mask_large_side has shape 0
                if mask_large_side.shape[0] == 0:
                    break

                # Keep the triangles that are not too large
                uv_good_ = uv[np.logical_not(mask_large_side)]
                triangles_vertices_good_ = triangles_vertices[np.logical_not(mask_large_side)]
                triangles_normals_good_ = triangles_normals[np.logical_not(mask_large_side)]

                # Hold on to the triangles that are good
                uv_good.append(uv_good_)
                triangles_vertices_good.append(triangles_vertices_good_)
                triangles_normals_good.append(triangles_normals_good_)
                
                # If no triangle is too large, we are done
                if not np.any(mask_large_side):
                    break

                # Get the triangles that are too large
                uv_large = uv[mask_large_side]
                side_lengths = side_lengths[mask_large_side]
                triangle_min_uv = np.expand_dims(triangle_min_uv[mask_large_side], axis=1)
                triangle_max_uv = np.expand_dims(triangle_max_uv[mask_large_side], axis=1)
                triangles_vertices_large = triangles_vertices[mask_large_side]
                triangles_normals_large = triangles_normals[mask_large_side]

                # Determine the larger side (x or y) for each triangle
                mask_larger_side_x = side_lengths[:, 0] >= side_lengths[:, 1]
                mask_larger_side_y = np.logical_not(mask_larger_side_x)

                # Create masks for the minimum and maximum UV coordinates
                mask_uv_x_min = uv_large[:, :, 0] == triangle_min_uv[:, :, 0]
                mask_uv_x_max = uv_large[:, :, 0] == triangle_max_uv[:, :, 0]
                mask_uv_y_min = uv_large[:, :, 1] == triangle_min_uv[:, :, 1]
                mask_uv_y_max = uv_large[:, :, 1] == triangle_max_uv[:, :, 1]

                # Create masks for the x and y directions
                mask_x_min = np.logical_and(mask_larger_side_x[:, None], mask_uv_x_min)
                mask_x_max = np.logical_and(mask_larger_side_x[:, None], mask_uv_x_max)
                mask_y_min = np.logical_and(mask_larger_side_y[:, None], mask_uv_y_min)
                mask_y_max = np.logical_and(mask_larger_side_y[:, None], mask_uv_y_max)

                # Create masks for the minimum and maximum sides
                mask_x_min_ = np.zeros_like(mask_x_min)
                mask_x_max_ = np.zeros_like(mask_x_max)
                mask_y_min_ = np.zeros_like(mask_y_min)
                mask_y_max_ = np.zeros_like(mask_y_max)

                # Set the maximum value to True for each triangle
                idx_x_min = np.argmax(mask_x_min, axis=1)
                idx_x_max = np.argmax(mask_x_max, axis=1)
                idx_y_min = np.argmax(mask_y_min, axis=1)
                idx_y_max = np.argmax(mask_y_max, axis=1)

                # Create an index array
                ix = np.arange(mask_x_min.shape[0])

                mask_x_min_[ix, idx_x_min] = True
                mask_x_max_[ix, idx_x_max] = True
                mask_y_min_[ix, idx_y_min] = True
                mask_y_max_[ix, idx_y_max] = True

                # Combine the masks for the minimum and maximum sides
                mask_x_min__ = np.logical_and(mask_x_min, mask_x_min_)
                mask_x_max__ = np.logical_and(mask_x_max, mask_x_max_)
                mask_y_min__ = np.logical_and(mask_y_min, mask_y_min_)
                mask_y_max__ = np.logical_and(mask_y_max, mask_y_max_)

                # Combine the masks for the x and y directions
                mask_min = np.logical_or(mask_x_min__, mask_y_min__)
                mask_max = np.logical_or(mask_x_max__, mask_y_max__)

                # Create new vertices, normals, and UV coordinates
                new_vertices = (triangles_vertices_large[mask_min] + triangles_vertices_large[mask_max]) / 2
                new_normals = (triangles_normals_large[mask_min] + triangles_normals_large[mask_max]) / 2
                new_uv = (uv_large[mask_min] + uv_large[mask_max]) / 2

                new_triangles_vertices_0 = np.copy(triangles_vertices_large)
                new_triangles_vertices_0[mask_min] = new_vertices
                new_triangles_normals_0 = np.copy(triangles_normals_large)
                new_triangles_normals_0[mask_min] = new_normals

                new_triangles_vertices_1 = np.copy(triangles_vertices_large)
                new_triangles_vertices_1[mask_max] = new_vertices
                new_triangles_normals_1 = np.copy(triangles_normals_large)
                new_triangles_normals_1[mask_max] = new_normals

                new_uv_0 = np.copy(uv_large)
                new_uv_0[mask_min] = new_uv
                new_uv_1 = np.copy(uv_large)
                new_uv_1[mask_max] = new_uv

                # Set up for the next iteration
                triangles_vertices = np.concatenate((new_triangles_vertices_0, new_triangles_vertices_1), axis=0)
                triangles_normals = np.concatenate((new_triangles_normals_0, new_triangles_normals_1), axis=0)
                uv = np.concatenate((new_uv_0, new_uv_1), axis=0)

            # Show the final progress
            pbar.n = 100
            pbar.refresh()

        # Update the triangles, normals, and UV coordinates with the adjusted values
        self.triangles_vertices = np.concatenate(triangles_vertices_good, axis=0)
        self.triangles_normals = np.concatenate(triangles_normals_good, axis=0)
        self.uv = np.concatenate(uv_good, axis=0)

        # Print the adjusted number of triangles
        print(f"Adjusted triangles: {self.triangles_vertices.shape[0]}, {self.triangles_normals.shape[0]}, {self.uv.shape[0]}", end="\n")
        
    def init_grids_to_process(self):
        triangles_vertices = self.triangles_vertices.reshape(-1, 3)
        grids_to_process = set(map(tuple, (triangles_vertices / self.grid_size).astype(int)))

        mask_changing_vertices = np.floor(((triangles_vertices - self.r) / self.grid_size).astype(int)) != np.floor(((triangles_vertices + self.r) / self.grid_size).astype(int))
        mask_changing_vertices = np.any(mask_changing_vertices, axis=1)
        changing_vertices = triangles_vertices[mask_changing_vertices]

        for x in range(-1, 2):
            for y in range(-1, 2):
                for z in range(-1, 2):
                    grids_to_process.update(set(map(tuple, np.floor((changing_vertices + np.array([x*self.r, y*self.r, z*self.r])) / self.grid_size).astype(int).reshape(-1, 3))))
        
        grids_to_process = sorted(list(grids_to_process)) # Sort the blocks to process for deterministic behavior
        print(f"Number of grids to process: {len(grids_to_process)}")

        # grids_to_process = grids_to_process[:30] # Debugging

        return grids_to_process
    
    def get_writer(self):
        return self.writer
    
    def extract_triangles_mask(self, grid_index):
        # select all triangles that have at least one vertice in the grid with a r-padded bounding box
        # grid_size * grid_index - r <= vertice <= grid_size * grid_index + r
        selected_triangles_mask = np.any(np.all(np.logical_and(self.triangles_vertices >= np.array(grid_index) * self.grid_size - 2*self.r, self.triangles_vertices <= (np.array(grid_index) + 1) * self.grid_size + 2*self.r), axis=2), axis=1)

        return selected_triangles_mask
    
    def load_grid(self, grid_index, uint8=False):
        if self.scroll_format == "grid cells":
            return self.load_grid_cell(grid_index, uint8)
        elif self.scroll_format == "zarr":
            return self.load_grid_zarr(grid_index, uint8)
        else:
            raise NotImplementedError(f"Scroll format {self.scroll_format} not yet implemented.")
    
    def load_grid_cell(self, grid_index, uint8=False):
        grid_index_ = np.asarray(grid_index)[[1, 0, 2]] # swap axis for 'special' grid cell naming ...
        path = self.grid_template.format(grid_index_[0]+1, grid_index_[1]+1, grid_index_[2]+1)

        # Check if the file exists
        if not os.path.exists(path):
            # print(f"File {path} does not exist.")
            return None

        # Read the image
        with tifffile.TiffFile(path) as tif:
            grid_cell = tif.asarray()

        if uint8:
            grid_cell = np.uint8(grid_cell//256)
        return grid_cell
    
    def load_grid_zarr(self, grid_index, uint8=False):
        # Calculate indices for zarr indexing
        grid_index_ = np.asarray(grid_index)[[2, 1, 0]] # swap axis for zar indexing. z x y
        grid_start = grid_index_ * self.grid_size
        grid_end = grid_start + self.grid_size
        grid_end = [min(grid_end[i], self.zarr.shape[i]) for i in range(3)]
        # copy data into grid size block
        grid_cell = np.zeros((self.grid_size, self.grid_size, self.grid_size), dtype=np.uint16)
        zarr_grid = self.zarr[grid_start[0]:grid_end[0], grid_start[1]:grid_end[1], grid_start[2]:grid_end[2]]
        grid_cell[:zarr_grid.shape[0], :zarr_grid.shape[1], :zarr_grid.shape[2]] = zarr_grid

        if uint8:
            grid_cell = np.uint8(grid_cell//256)
        return grid_cell
    
    def __len__(self):
        return len(self.grids_to_process)

    def __getitem__(self, idx):
        grid_index = self.grids_to_process[idx]
        triangles_mask = self.extract_triangles_mask(grid_index)
        # Vertices and normals in triangles that have at least one vertice in the grid with a r-padded bounding box
        vertices = self.triangles_vertices[triangles_mask]
        normals = self.triangles_normals[triangles_mask]
        # UV of vertices in triangles that have at least one vertice in the grid with a r-padded bounding box
        uv = self.uv[triangles_mask]

        # load grid cell from disk
        grid_cell = self.load_grid(grid_index)
        if grid_cell is None:
            return None, None, None, None, None
        grid_cell = grid_cell.astype(np.float32)

        # Convert NumPy arrays to PyTorch tensors
        vertices_tensor = torch.tensor(vertices, dtype=torch.float32)
        normals_tensor = torch.tensor(normals, dtype=torch.float32)
        uv_tensor = torch.tensor(uv, dtype=torch.float32)
        grid_cell_tensor = torch.tensor(grid_cell, dtype=torch.float32)
        grid_coord = torch.tensor(np.array(grid_index) * self.grid_size, dtype=torch.int32)

        return grid_coord, grid_cell_tensor, vertices_tensor, normals_tensor, uv_tensor

class PPMAndTextureModel(pl.LightningModule):
    def __init__(self, r: int = 32, max_side_triangle: int = 10):
        print("instantiating model")
        self.r = r
        self.max_side_triangle = max_side_triangle
        self.new_order = [2,1,0] # [2,1,0], [2,0,1], [0,2,1], [0,1,2], [1,2,0], [1,0,2]
        super().__init__()

    def ppm(self, pts, tri):
        # pts T x W*H x 2
        # tri_pts T x 3 x 2
        # triangles 3
        v0 = tri[:, 2, :].unsqueeze(1) - tri[:, 0, :].unsqueeze(1)
        v1 = tri[:, 1, :].unsqueeze(1) - tri[:, 0, :].unsqueeze(1)
        v2 = pts - tri[:, 0, :].unsqueeze(1)

        dot00 = v0.pow(2).sum(dim=2)
        dot01 = (v0 * v1).sum(dim=2)
        dot11 = v1.pow(2).sum(dim=2)
        dot02 = (v2 * v0).sum(dim=2)
        dot12 = (v2 * v1).sum(dim=2)

        invDenom = 1 / (dot00 * dot11 - dot01.pow(2))
        u = (dot11 * dot02 - dot01 * dot12) * invDenom
        v = (dot00 * dot12 - dot01 * dot02) * invDenom

        is_inside = (u >= 0) & (v >= 0) & ((u + v) <= 1 )

        w = 1 - u - v

        bary_coords = torch.stack([u, v, w], dim=2)
        bary_coords = normalize(bary_coords, p=1, dim=2)

        return bary_coords, is_inside
    
    def create_grid_points_tensor(self, starting_points, w, h):
        device = starting_points.device
        n = starting_points.shape[0]
        
        # Generate all combinations of offsets in the grid
        dx = torch.arange(w, device=device)  # Shape (w,)
        dy = torch.arange(h, device=device)  # Shape (h,)
        
        # Create a meshgrid from dx and dy
        mesh_dx, mesh_dy = torch.meshgrid(dx, dy, indexing='xy')  # Shapes (h, w)
        
        # Stack and reshape to create the complete offset grid
        offset_grid = torch.stack((mesh_dx, mesh_dy), dim=2).view(-1, 2)  # Shape (w*h, 2)
        
        # Expand starting points for broadcasting
        starting_points_expanded = starting_points.view(n, 1, 2)  # Shape (n, 1, 2)
        
        # Add starting points to offset grid (broadcasting in action)
        grid_points = starting_points_expanded + offset_grid  # Shape (n, w*h, 2)
        
        return grid_points

    def forward(self, x):
        # grid_coords: T x 3, grid_cell: B x W x W x W, vertices: T x 3 x 3, normals: T x 3 x 3, uv_coords_triangles: T x 3 x 2, grid_index: T
        grid_coords, grid_cells, vertices, normals, uv_coords_triangles, grid_index = x
        
        # Handle the case where the grid cells are empty
        if grid_cells is None:
            return None
                
        # Step 1: Compute AABBs for each triangle (only starting points of AABB rectangles)
        min_uv, _ = torch.min(uv_coords_triangles, dim=1)
        # Floor and ceil the UV coordinates
        min_uv = torch.floor(min_uv)

        nr_triangles = vertices.shape[0]
        max_triangles_per_loop = 5000
        values_list = []
        grid_points_list = []
        for i in range(0, nr_triangles, max_triangles_per_loop):
            min_uv_ = min_uv[i:i+max_triangles_per_loop]
            grid_coords_ = grid_coords[i:i+max_triangles_per_loop]
            vertices_ = vertices[i:i+max_triangles_per_loop]
            normals_ = normals[i:i+max_triangles_per_loop]
            uv_coords_triangles_ = uv_coords_triangles[i:i+max_triangles_per_loop]
            grid_index_ = grid_index[i:i+max_triangles_per_loop]

            # Step 2: Generate Meshgrids for All Triangles
            # create grid points tensor: T x W*H x 2
            grid_points = self.create_grid_points_tensor(min_uv_, self.max_side_triangle, self.max_side_triangle)
            del min_uv_

            # Step 3: Compute Barycentric Coordinates for all Triangles grid_points
            # baryicentric_coords: T x W*H x 3, is_inside: T x W*H
            baryicentric_coords, is_inside = self.ppm(grid_points, uv_coords_triangles_)
            grid_points = grid_points[is_inside] # S x 2

            # adjust to new_order
            vertices_ = vertices_[:, self.new_order, :]
            normals_ = normals_[:, self.new_order, :]

            # vertices: T x 3 x 3, normals: T x 3 x 3, baryicentric_coords: T x W*H x 3
            coords = torch.einsum('ijk,isj->isk', vertices_, baryicentric_coords).squeeze()
            norms = torch.einsum('ijk,isj->isk', normals_, baryicentric_coords).squeeze()
            # Handle case where T == 1 by ensuring dimensions are not squeezed away
            if coords.dim() == 2:
                coords = coords.unsqueeze(0)
            if norms.dim() == 2:
                norms = norms.unsqueeze(0)
            norms = normalize(norms,dim=2)
            del vertices_, normals_, uv_coords_triangles_

            # broadcast grid index to T x W*H -> S
            grid_index_ = grid_index_.unsqueeze(-1).expand(-1, baryicentric_coords.shape[1])
            grid_index_ = grid_index_[is_inside]
            # broadcast grid_coords to T x W*H x 3 -> S x 3
            grid_coords_ = grid_coords_.unsqueeze(-2).expand(-1, baryicentric_coords.shape[1], -1)
            coords = coords - grid_coords_ # Reorient coordinate system origin to 0 for extraction on grid_cells
            del baryicentric_coords, grid_coords_

            # coords: S x 3, norms: S x 3
            coords = coords[is_inside]
            norms = norms[is_inside]

            # Poper axis order
            coords = coords[:, self.new_order]
            norms = norms[:, self.new_order]

            # Step 4: Compute the 3D coordinates for every r slice
            r_arange = torch.arange(-self.r, self.r+1, device=coords.device).reshape(1, -1, 1)

            # coords: S x 2*r+1 x 3, grid_index: S x 2*r+1 x 1
            coords = coords.unsqueeze(-2).expand(-1, 2*self.r+1, -1) + r_arange * norms.unsqueeze(-2).expand(-1, 2*self.r+1, -1)
            grid_index_ = grid_index_.unsqueeze(-1).unsqueeze(-1).expand(-1, 2*self.r+1, -1)

            # Expand and add 3rd dimension to grid points
            r_arange = r_arange.expand(grid_points.shape[0], -1, -1) + self.r # [0 to 2*r]
            grid_points = grid_points.unsqueeze(-2).expand(-1, 2*self.r+1, -1)
            grid_points = torch.cat((grid_points, r_arange), dim=-1)
            del r_arange, is_inside

            # Step 5: Filter out the points that are outside the grid_cells
            mask_coords = (coords[:, :, 0] >= 0) & (coords[:, :, 0] < grid_cells.shape[1]) & (coords[:, :, 1] >= 0) & (coords[:, :, 1] < grid_cells.shape[2]) & (coords[:, :, 2] >= 0) & (coords[:, :, 2] < grid_cells.shape[3])

            # coords: S' x 3, norms: S' x 3
            coords = coords[mask_coords]
            grid_points = grid_points[mask_coords] # S' x 2
            grid_index_ = grid_index_[mask_coords] # S'

            # Step 5: Extract the values from the grid cells
            # grid_cells: T x W x H x D, coords: S' x 3, grid_index: S' x 1
            values = extract_from_image_4d(grid_cells, grid_index_, coords)
            del coords, mask_coords, grid_index_

            # Step 6: Return the 3D Surface Volume coordinates and the values
            values = values.reshape(-1)
            grid_points = grid_points.reshape(-1, 3) # grid_points: S' x 3
            
            # Empty the cache to free up memory
            torch.cuda.empty_cache()

            # reorder grid_points
            grid_points = grid_points[:, [2, 0, 1]]

            values_list.append(values)
            grid_points_list.append(grid_points)

        
        del grid_cells, grid_index, min_uv, vertices, normals, uv_coords_triangles
        # Empty the cache to free up memory
        torch.cuda.empty_cache()

        if len(values_list) == 0:
            return None, None
        
        values = torch.cat(values_list, dim=0)
        grid_points = torch.cat(grid_points_list, dim=0)

        values = values.cpu().numpy().astype(np.uint16)
        grid_points = grid_points.cpu().numpy().astype(np.int32)

        # Return the 3D Surface Volume coordinates and the values
        return values, grid_points
    
# Custom collation function
def custom_collate_fn(batch):
    try:
        # Initialize containers for the aggregated items
        grid_cells = []
        vertices = []
        normals = []
        uv_coords_triangles = []
        grid_index = []
        grid_coords = []
    
        # Loop through each batch and aggregate its items
        for i, items in enumerate(batch):
            if items is None:
                continue
            grid_coord, grid_cell, vertice, normal, uv_coords_triangle = items
            if grid_cell is None:
                continue
            if len(grid_cell) == 0:
                continue
            if grid_cell.size()[0] == 0:
                continue
            grid_cells.append(grid_cell)
            vertices.append(vertice)
            normals.append(normal)
            uv_coords_triangles.append(uv_coords_triangle)
            grid_index.extend([i]*vertice.shape[0])
            grid_coord = grid_coord.unsqueeze(0).expand(vertice.shape[0], -1)
            grid_coords.extend(grid_coord)
            
        if len(grid_cells) == 0:
            return None, None, None, None, None, None
            
        # Turn the lists into tensors
        grid_cells = torch.stack(grid_cells, dim=0)
        vertices = torch.concat(vertices, dim=0)
        normals = torch.concat(normals, dim=0)
        uv_coords_triangles = torch.concat(uv_coords_triangles, dim=0)
        grid_index = torch.tensor(grid_index, dtype=torch.int32)
        grid_coords = torch.stack(grid_coords, dim=0)
    
        # Return a single batch containing all aggregated items
        return grid_coords, grid_cells, vertices, normals, uv_coords_triangles, grid_index
    except:
        return None, None, None, None, None, None
    
def ppm_and_texture(obj_path, scroll, output_path=None, grid_size=500, gpus=1, batch_size=1, r=32, format='jpg', max_side_triangle: int = 10, display=False, nr_workers=None, prefetch_factor=2, r_steps=None):
    # Number of workers
    num_threads = multiprocessing.cpu_count() // int(1.5 * int(gpus))
    num_treads_for_gpus = 12
    num_workers = min(num_threads, num_treads_for_gpus)
    num_workers = max(num_workers, 1)
    # Number of workers for the Writer
    max_workers = max(1, min(multiprocessing.cpu_count()//2, 20))
    max_workers = min(max_workers, 20)

    if nr_workers is not None:
        num_workers = nr_workers
        max_workers = nr_workers

    # get scroll format
    is_zarr = scroll.endswith(".zarr")
    if is_zarr:
        scroll_format = "zarr"
    else:
        scroll_format = "grid cells"
        # Template for the grid cell files
        scroll = os.path.join(scroll, "cell_yxz_{:03}_{:03}_{:03}.tif")

    # Initialize the dataset and dataloader
    dataset = MeshDataset(obj_path, scroll, scroll_format=scroll_format, output_path=output_path, grid_size=grid_size, r=r, max_side_triangle=max_side_triangle, max_workers=max_workers, display=display, r_steps=r_steps)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=False, num_workers=num_workers, prefetch_factor=prefetch_factor)
    model = PPMAndTextureModel(r=r, max_side_triangle=max_side_triangle)
    
    writer = dataset.get_writer()
    trainer = pl.Trainer(callbacks=[writer], accelerator='gpu', devices=int(gpus), strategy="ddp")
    
    print("Start Rendering")
    total_layers = 2*r+1
    if r_steps is not None:
        # Process in segments to reduce memory usage
        for seg_start in range(0, total_layers, r_steps):
            seg_end = min(seg_start + r_steps, total_layers)
            writer.current_step_offset = seg_start
            # reset the surface volume
            writer.surface_volume_np = None
            print(f"Rendering segment: layers {seg_start} to {seg_end-1}")
            trainer.predict(model, dataloaders=dataloader, return_predictions=False)
            writer.write_to_disk(format)
    else:
        # Render full volume at once
        trainer.predict(model, dataloaders=dataloader, return_predictions=False)
        writer.write_to_disk(format)
    print("Rendering done")
    # Sync up before exiting
    torch.distributed.barrier()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('obj', type=str)
    parser.add_argument('scroll', type=str, help="Path to the grid cells or zarr file.")
    parser.add_argument('--output_path', type=str, default=None, help="Output folder path that shall contain the layers folder.")
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--r', type=int, default=32)
    parser.add_argument('--format', type=str, default='jpg')
    parser.add_argument('--max_side_triangle', type=int, default=10)
    parser.add_argument('--display', action='store_true')
    parser.add_argument('--nr_workers', type=int, default=None)
    parser.add_argument('--prefetch_factor', type=int, default=2)
    parser.add_argument('--r_steps', type=int, default=None, help="Number of layers to render at a time (segment size).")
    args = parser.parse_known_args()[0]

    print(f"Rendering args: {args}")
    if args.display:
        print("[INFO]: Displaying the rendering image slows down the rendering process by about 20%.")

    ppm_and_texture(args.obj, gpus=args.gpus, scroll=args.scroll, output_path=args.output_path, r=args.r, format=args.format, max_side_triangle=args.max_side_triangle, display=args.display, nr_workers=args.nr_workers, prefetch_factor=args.prefetch_factor, r_steps=args.r_steps)
