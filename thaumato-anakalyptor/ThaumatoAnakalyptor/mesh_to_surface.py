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
    def __init__(self, save_path, image_size, r, max_queue_size=10, max_workers=1, display=True, use_mmemmap=False):
        super().__init__(write_interval="batch")  # or "epoch" for end of an epoch
        self.save_path = save_path
        self.image_size = image_size
        self.display = display
        self.image = None
        self.r = r
        self.surface_volume_np = None
        self.use_mmemmap = use_mmemmap
        self.max_queue_size = max(max_queue_size, max_workers)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)  # Adjust number of workers as needed
        self.semaphore = Semaphore(self.max_queue_size)
        self.futures = []
        self.num_workers = cpu_count()
        self.trainer_rank = None
        # Only used if not using mmemmap
        self.shm = None
    
    def write_on_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, prediction, batch_indices, batch, batch_idx: int, dataloader_idx: int) -> None:
        if self.trainer_rank is None:  # Only set the rank once
            self.trainer_rank = trainer.global_rank if trainer.world_size > 1 else 0

        if self.surface_volume_np is None:
            if self.use_mmemmap:
                if trainer.global_rank == 0:
                    self.surface_volume_np = self.create_mmemmap_array(
                        (2*self.r+1, self.image_size[0], self.image_size[1]), np.uint16, filename="surface_volume_mmemmap.dat")
                    torch.distributed.barrier()
                else:
                    torch.distributed.barrier()
                    self.surface_volume_np = self.attach_mmemmap_array(
                        (2*self.r+1, self.image_size[0], self.image_size[1]), np.uint16, filename="surface_volume_mmemmap.dat")
            else:
                if trainer.global_rank == 0:
                    self.surface_volume_np, self.shm = self.create_shared_array(
                        (2*self.r+1, self.image_size[0], self.image_size[1]), np.uint16, name="surface_volume")
                    torch.distributed.barrier()
                else:
                    torch.distributed.barrier()
                    self.surface_volume_np, self.shm = self.attach_shared_array(
                        (2*self.r+1, self.image_size[0], self.image_size[1]), np.uint16, name="surface_volume")
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
            image = (self.surface_volume_np[self.r].astype(np.float32) / 65535)
            screen_y = 2560
            screen_x = 1440

            if (screen_y * image.shape[1]) // image.shape[0] > screen_x:
                screen_y = (screen_x * image.shape[0]) // image.shape[1]
            else:
                screen_x = (screen_y * image.shape[1]) // image.shape[0]

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
        try:
            # Create a shared array
            shm = shared_memory.SharedMemory(create=True, size=array_size, name=name)
        except FileExistsError:
            print(f"Shared memory with name {name} already exists.")
            shm = shared_memory.SharedMemory(create=False, size=array_size, name=name)

        arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        arr.fill(0)
        return arr, shm

    def attach_shared_array(self, shape, dtype, name="shared_array"):
        while True:
            try:
                shm = shared_memory.SharedMemory(name=name)
                arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
                assert arr.shape == shape, f"Expected shape {shape} but got {arr.shape}"
                assert arr.dtype == dtype, f"Expected dtype {dtype} but got {arr.dtype}"
                print("Attached to shared memory")
                return arr, shm
            except FileNotFoundError:
                time.sleep(0.2)

    def create_mmemmap_array(self, shape, dtype, filename="surface_volume_mmemmap.dat"):
        filepath = os.path.join(self.save_path, filename)
        # Create (or overwrite) the memmap file
        memmap_array = np.memmap(filepath, dtype=dtype, mode='w+', shape=shape)
        memmap_array[:] = 0  # initialize with zeros
        return memmap_array

    def attach_mmemmap_array(self, shape, dtype, filename="surface_volume_mmemmap.dat"):
        filepath = os.path.join(self.save_path, filename)
        while not os.path.exists(filepath):
            time.sleep(0.2)
        memmap_array = np.memmap(filepath, dtype=dtype, mode='r+', shape=shape)
        return memmap_array

    def process_and_write_data(self, prediction):
        try:
            if prediction is None:
                return
            if len(prediction) == 0:
                return

            values, indexes_3d = prediction

            if indexes_3d.shape[0] == 0:
                return

            self.surface_volume_np[indexes_3d[:, 0], indexes_3d[:, 1], indexes_3d[:, 2]] = values

            self.process_display_progress()
        except Exception as e:
            print(e)

    def wait_for_all_writes_to_complete(self):
        for future in tqdm(self.futures, desc="Finalizing writes"):
            future.result()
        torch.distributed.barrier()

    def write_to_disk(self, flag='jpg'):
        print("Waiting for all writes to complete")
        self.wait_for_all_writes_to_complete()
        if self.trainer_rank != 0:
            if not self.use_mmemmap:
                self.shm.close()
                self.shm = None
            return
        print("Writing Segment to disk")
        os.makedirs(self.save_path, exist_ok=True)

        if flag == 'tif':
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

        if not self.use_mmemmap:
            try:
                self.shm.close()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.shm.unlink()
                    self.shm = None
            except Exception as e:
                print(e)
        print("Segment written to disk")

    def write_tif(self):
        def save_tif(i, filename):
            image = self.surface_volume_np[i]
            image = image.T
            image = image[::-1, :]
            tifffile.imsave(filename, image)

        with ThreadPoolExecutor(self.num_workers) as executor:
            futures = []
            for i in range(self.surface_volume_np.shape[0]):
                i_str = str(i).zfill(len(str(self.surface_volume_np.shape[0])))
                filename = os.path.join(self.save_path, f"{i_str}.tif")
                futures.append(executor.submit(save_tif, i, filename))

            for future in tqdm(as_completed(futures), desc="Writing TIF"):
                future.result()

        composite_image = np.zeros((self.surface_volume_np.shape[1], self.surface_volume_np.shape[2]), dtype=np.float32)
        for i in range(self.surface_volume_np.shape[0]):
            composite_image = np.maximum(composite_image, self.surface_volume_np[i])

        composite_image = composite_image.astype(np.uint16)
        composite_image = composite_image.T
        composite_image = composite_image[::-1, :]
        tifffile.imsave(os.path.join(os.path.dirname(self.save_path), "composite.tif"), composite_image)

    def write_jpg(self, quality=60):
        def save_jpg(i, filename):
            image = self.surface_volume_np[i]
            image = (image / 256).astype(np.uint8)
            image = image.T
            image = image[::-1, :]
            cv2.imwrite(filename, image, [cv2.IMWRITE_JPEG_QUALITY, quality])

        with ThreadPoolExecutor(self.num_workers) as executor:
            futures = []
            for i in range(self.surface_volume_np.shape[0]):
                i_str = str(i).zfill(len(str(self.surface_volume_np.shape[0])))
                filename = os.path.join(self.save_path, f"{i_str}.jpg")
                futures.append(executor.submit(save_jpg, i, filename))

            for future in tqdm(as_completed(futures), desc="Writing JPG"):
                future.result()

        composite_image = np.zeros((self.surface_volume_np.shape[1], self.surface_volume_np.shape[2]), dtype=np.float32)
        for i in range(self.surface_volume_np.shape[0]):
            composite_image = np.maximum(composite_image, self.surface_volume_np[i])

        composite_image = composite_image.astype(np.uint16)
        composite_image = (composite_image / 256).astype(np.uint8)
        composite_image = composite_image.T
        composite_image = composite_image[::-1, :]
        cv2.imwrite(os.path.join(os.path.dirname(self.save_path), f"composite.jpg"), composite_image, [cv2.IMWRITE_JPEG_QUALITY, quality])

    def write_memmap(self):
        memmap_path = os.path.join(self.save_path, "surface_volume")
        memmap = np.memmap(memmap_path, dtype='uint16', mode='w+', shape=self.surface_volume_np.shape)
        memmap[:] = self.surface_volume_np[:]
        del memmap

    def write_npz(self):
        npz_path = os.path.join(self.save_path, "surface_volume.npz")
        np.savez_compressed(npz_path, surface_volume=self.surface_volume_np)

    def write_zarr(self):
        chunk_size = (16, 16, 16)
        compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=zarr.Blosc.SHUFFLE)
        zarr_path = os.path.join(self.save_path, "surface_volume.zarr")
        z = zarr.open(zarr_path, mode='w', shape=self.surface_volume_np.shape, dtype='uint16', chunks=chunk_size, compressor=compressor)
        z[:] = self.surface_volume_np

class MeshDataset(Dataset):
    def __init__(self, path, scroll, scroll_format="grid cells", output_path=None, grid_size=500, r=32, max_side_triangle=10, max_workers=1, display=False, use_mmemmap=False):
        self.grid_size = grid_size
        self.path = path
        self.scroll_format = scroll_format
        if self.scroll_format == "grid cells":
            self.grid_template = scroll
        if self.scroll_format == "zarr":
            self.zarr = self.load_zarr(scroll)
        self.r = r+1
        self.max_side_triangle = max_side_triangle
        self.load_mesh(path)
        output_path = os.path.dirname(path) if output_path is None else output_path
        write_path = os.path.join(output_path, "layers")
        self.writer = MyPredictionWriter(write_path, self.image_size, r, max_workers=max_workers, display=display, use_mmemmap=use_mmemmap)

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
                if line.strip().startswith('map_Kd'):
                    parts = line.split()
                    if len(parts) > 1:
                        texture_filenames.append(parts[1])
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
        working_path = os.path.dirname(path)
        base_name = os.path.splitext(os.path.basename(path))[0]
        tif_path = os.path.join(working_path, f"{base_name}.tif")
        png_path = os.path.join(working_path, f"{base_name}.png")
        mtl_path = os.path.join(working_path, f"{base_name}.mtl")
        print(f"Names: {tif_path}, {png_path}, {mtl_path}", end="\n")

        if os.path.exists(tif_path):
            image_path = tif_path
            print(f"Found TIF image at: {image_path}", end="\n")
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
        else:
            image_path = None
            print("No corresponding TIF or PNG image found.", end="\n")
        
        print("Texture Image Name:", image_path)
        if image_path:
            with Image.open(image_path) as img:
                y_size, x_size = img.size
        print(f"Y-size: {y_size}, X-size: {x_size}", end="\n")

        print(f"Loading mesh from {path}", end="\n")
        with tempfile.NamedTemporaryFile(suffix=".obj") as temp_file:
            temp_path = temp_file.name
            os.system(f"cp {path} {temp_path}")
            mesh = o3d.io.read_triangle_mesh(temp_path)
        self.mesh = mesh
        print(f"Loaded mesh from {path}", end="\n")

        self.vertices = np.asarray(self.mesh.vertices)
        self.normals = np.asarray(self.mesh.vertex_normals)
        self.triangles = np.asarray(self.mesh.triangles)
        uv = np.asarray(self.mesh.triangle_uvs).reshape(-1, 3, 2)
        self.uv = uv * np.array([y_size, x_size])
        self.image_size = (y_size, x_size)
        self.generate_mask_png()
        self.triangles_vertices = self.vertices[self.triangles]
        self.triangles_normals = self.normals[self.triangles]

    def adjust_triangle_sizes(self):
        triangles_vertices = self.triangles_vertices
        triangles_normals = self.triangles_normals
        uv = self.uv

        print(f"Original triangles: {triangles_vertices.shape[0]}, {triangles_normals.shape[0]}, {uv.shape[0]}", end="\n")

        uv_good = []
        triangles_vertices_good = []
        triangles_normals_good = []

        def current_progress(maxS):
            return np.log(2) - np.log(self.max_side_triangle / maxS)
        with tqdm(total=100, desc="Adjusting triangle sizes") as pbar:
            start = None
            while True:
                triangle_min_uv = np.min(uv, axis=1)
                triangle_max_uv = np.max(uv, axis=1)
                side_lengths = np.ceil(triangle_max_uv) - np.floor(triangle_min_uv)
                max_side_lengths = np.max(side_lengths, axis=0)

                if start is None:
                    start = current_progress(max_side_lengths[0]) + current_progress(max_side_lengths[1])
                    progress = 0
                else:
                    now = current_progress(max_side_lengths[0]) + current_progress(max_side_lengths[1])
                    progress = max(1 - now / start, 0)
                pbar.n = int(progress * 100)
                pbar.refresh()

                mask_large_side = np.any(side_lengths > self.max_side_triangle, axis=1)
                if mask_large_side.shape[0] == 0:
                    break

                uv_good_ = uv[np.logical_not(mask_large_side)]
                triangles_vertices_good_ = triangles_vertices[np.logical_not(mask_large_side)]
                triangles_normals_good_ = triangles_normals[np.logical_not(mask_large_side)]
                uv_good.append(uv_good_)
                triangles_vertices_good.append(triangles_vertices_good_)
                triangles_normals_good.append(triangles_normals_good_)
                
                if not np.any(mask_large_side):
                    break

                uv_large = uv[mask_large_side]
                side_lengths = side_lengths[mask_large_side]
                triangle_min_uv = np.expand_dims(triangle_min_uv[mask_large_side], axis=1)
                triangle_max_uv = np.expand_dims(triangle_max_uv[mask_large_side], axis=1)
                triangles_vertices_large = triangles_vertices[mask_large_side]
                triangles_normals_large = triangles_normals[mask_large_side]

                mask_larger_side_x = side_lengths[:, 0] >= side_lengths[:, 1]
                mask_larger_side_y = np.logical_not(mask_larger_side_x)

                mask_uv_x_min = uv_large[:, :, 0] == triangle_min_uv[:, :, 0]
                mask_uv_x_max = uv_large[:, :, 0] == triangle_max_uv[:, :, 0]
                mask_uv_y_min = uv_large[:, :, 1] == triangle_min_uv[:, :, 1]
                mask_uv_y_max = uv_large[:, :, 1] == triangle_max_uv[:, :, 1]

                mask_x_min = np.logical_and(mask_larger_side_x[:, None], mask_uv_x_min)
                mask_x_max = np.logical_and(mask_larger_side_x[:, None], mask_uv_x_max)
                mask_y_min = np.logical_and(mask_larger_side_y[:, None], mask_uv_y_min)
                mask_y_max = np.logical_and(mask_larger_side_y[:, None], mask_uv_y_max)

                mask_x_min_ = np.zeros_like(mask_x_min)
                mask_x_max_ = np.zeros_like(mask_x_max)
                mask_y_min_ = np.zeros_like(mask_y_min)
                mask_y_max_ = np.zeros_like(mask_y_max)

                idx_x_min = np.argmax(mask_x_min, axis=1)
                idx_x_max = np.argmax(mask_x_max, axis=1)
                idx_y_min = np.argmax(mask_y_min, axis=1)
                idx_y_max = np.argmax(mask_y_max, axis=1)

                ix = np.arange(mask_x_min.shape[0])

                mask_x_min_[ix, idx_x_min] = True
                mask_x_max_[ix, idx_x_max] = True
                mask_y_min_[ix, idx_y_min] = True
                mask_y_max_[ix, idx_y_max] = True

                mask_x_min__ = np.logical_and(mask_x_min, mask_x_min_)
                mask_x_max__ = np.logical_and(mask_x_max, mask_x_max_)
                mask_y_min__ = np.logical_and(mask_y_min, mask_y_min_)
                mask_y_max__ = np.logical_and(mask_y_max, mask_y_max_)

                mask_min = np.logical_or(mask_x_min__, mask_y_min__)
                mask_max = np.logical_or(mask_x_max__, mask_y_max__)

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

                triangles_vertices = np.concatenate((new_triangles_vertices_0, new_triangles_vertices_1), axis=0)
                triangles_normals = np.concatenate((new_triangles_normals_0, new_triangles_normals_1), axis=0)
                uv = np.concatenate((new_uv_0, new_uv_1), axis=0)

            pbar.n = 100
            pbar.refresh()

        self.triangles_vertices = np.concatenate(triangles_vertices_good, axis=0)
        self.triangles_normals = np.concatenate(triangles_normals_good, axis=0)
        self.uv = np.concatenate(uv_good, axis=0)

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
        
        grids_to_process = sorted(list(grids_to_process))
        print(f"Number of grids to process: {len(grids_to_process)}")
        return grids_to_process
    
    def get_writer(self):
        return self.writer
    
    def extract_triangles_mask(self, grid_index):
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
        grid_index_ = np.asarray(grid_index)[[1, 0, 2]]
        path = self.grid_template.format(grid_index_[0]+1, grid_index_[1]+1, grid_index_[2]+1)
        if not os.path.exists(path):
            return None

        with tifffile.TiffFile(path) as tif:
            grid_cell = tif.asarray()

        if uint8:
            grid_cell = np.uint8(grid_cell//256)
        return grid_cell
    
    def load_grid_zarr(self, grid_index, uint8=False):
        grid_index_ = np.asarray(grid_index)[[2, 1, 0]]
        grid_start = grid_index_ * self.grid_size
        grid_end = grid_start + self.grid_size
        grid_end = [min(grid_end[i], self.zarr.shape[i]) for i in range(3)]
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
        vertices = self.triangles_vertices[triangles_mask]
        normals = self.triangles_normals[triangles_mask]
        uv = self.uv[triangles_mask]

        grid_cell = self.load_grid(grid_index)
        if grid_cell is None:
            return None, None, None, None, None
        grid_cell = grid_cell.astype(np.float32)

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
        self.new_order = [2,1,0]
        super().__init__()

    def ppm(self, pts, tri):
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
        dx = torch.arange(w, device=device)
        dy = torch.arange(h, device=device)
        mesh_dx, mesh_dy = torch.meshgrid(dx, dy, indexing='xy')
        offset_grid = torch.stack((mesh_dx, mesh_dy), dim=2).view(-1, 2)
        starting_points_expanded = starting_points.view(n, 1, 2)
        grid_points = starting_points_expanded + offset_grid
        return grid_points

    def forward(self, x):
        grid_coords, grid_cells, vertices, normals, uv_coords_triangles, grid_index = x
        if grid_cells is None:
            return None

        min_uv, _ = torch.min(uv_coords_triangles, dim=1)
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

            grid_points = self.create_grid_points_tensor(min_uv_, self.max_side_triangle, self.max_side_triangle)
            del min_uv_

            baryicentric_coords, is_inside = self.ppm(grid_points, uv_coords_triangles_)
            grid_points = grid_points[is_inside]

            vertices_ = vertices_[:, self.new_order, :]
            normals_ = normals_[:, self.new_order, :]

            coords = torch.einsum('ijk,isj->isk', vertices_, baryicentric_coords).squeeze()
            norms = torch.einsum('ijk,isj->isk', normals_, baryicentric_coords).squeeze()
            if coords.dim() == 2:
                coords = coords.unsqueeze(0)
            if norms.dim() == 2:
                norms = norms.unsqueeze(0)
            norms = normalize(norms, dim=2)
            del vertices_, normals_, uv_coords_triangles_

            grid_index_ = grid_index_.unsqueeze(-1).expand(-1, baryicentric_coords.shape[1])
            grid_index_ = grid_index_[is_inside]
            grid_coords_ = grid_coords_.unsqueeze(-2).expand(-1, baryicentric_coords.shape[1], -1)
            coords = coords - grid_coords_
            del baryicentric_coords, grid_coords_

            coords = coords[is_inside]
            norms = norms[is_inside]

            coords = coords[:, self.new_order]
            norms = norms[:, self.new_order]

            r_arange = torch.arange(-self.r, self.r+1, device=coords.device).reshape(1, -1, 1)
            coords = coords.unsqueeze(-2).expand(-1, 2*self.r+1, -1) + r_arange * norms.unsqueeze(-2).expand(-1, 2*self.r+1, -1)
            grid_index_ = grid_index_.unsqueeze(-1).unsqueeze(-1).expand(-1, 2*self.r+1, -1)

            r_arange = r_arange.expand(grid_points.shape[0], -1, -1) + self.r
            grid_points = grid_points.unsqueeze(-2).expand(-1, 2*self.r+1, -1)
            grid_points = torch.cat((grid_points, r_arange), dim=-1)
            del r_arange, is_inside

            mask_coords = (coords[:, :, 0] >= 0) & (coords[:, :, 0] < grid_cells.shape[1]) & (coords[:, :, 1] >= 0) & (coords[:, :, 1] < grid_cells.shape[2]) & (coords[:, :, 2] >= 0) & (coords[:, :, 2] < grid_cells.shape[3])
            coords = coords[mask_coords]
            grid_points = grid_points[mask_coords]
            grid_index_ = grid_index_[mask_coords]

            values = extract_from_image_4d(grid_cells, grid_index_, coords)
            del coords, mask_coords, grid_index_

            values = values.reshape(-1)
            grid_points = grid_points.reshape(-1, 3)
            torch.cuda.empty_cache()
            grid_points = grid_points[:, [2, 0, 1]]
            values_list.append(values)
            grid_points_list.append(grid_points)

        del grid_cells, grid_index, min_uv, vertices, normals, uv_coords_triangles
        torch.cuda.empty_cache()

        if len(values_list) == 0:
            return None, None

        values = torch.cat(values_list, dim=0)
        grid_points = torch.cat(grid_points_list, dim=0)

        values = values.cpu().numpy().astype(np.uint16)
        grid_points = grid_points.cpu().numpy().astype(np.int32)
        return values, grid_points

def custom_collate_fn(batch):
    try:
        grid_cells = []
        vertices = []
        normals = []
        uv_coords_triangles = []
        grid_index = []
        grid_coords = []
    
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
            
        grid_cells = torch.stack(grid_cells, dim=0)
        vertices = torch.concat(vertices, dim=0)
        normals = torch.concat(normals, dim=0)
        uv_coords_triangles = torch.concat(uv_coords_triangles, dim=0)
        grid_index = torch.tensor(grid_index, dtype=torch.int32)
        grid_coords = torch.stack(grid_coords, dim=0)
    
        return grid_coords, grid_cells, vertices, normals, uv_coords_triangles, grid_index
    except:
        return None, None, None, None, None, None
    
def ppm_and_texture(obj_path, scroll, output_path=None, grid_size=500, gpus=1, batch_size=1, r=32, format='jpg', max_side_triangle: int = 10, display=False, nr_workers=None, prefetch_factor=2, use_mmemmap=False):
    num_threads = multiprocessing.cpu_count() // int(1.5 * int(gpus))
    num_treads_for_gpus = 12
    num_workers = min(num_threads, num_treads_for_gpus)
    num_workers = max(num_workers, 1)
    max_workers = max(1, min(multiprocessing.cpu_count()//2, 20))
    max_workers = min(max_workers, 20)

    if nr_workers is not None:
        num_workers = nr_workers
        max_workers = nr_workers

    is_zarr = scroll.endswith(".zarr")
    if is_zarr:
        scroll_format = "zarr"
    else:
        scroll_format = "grid cells"
        scroll = os.path.join(scroll, "cell_yxz_{:03}_{:03}_{:03}.tif")

    dataset = MeshDataset(obj_path, scroll, scroll_format=scroll_format, output_path=output_path, grid_size=grid_size, r=r, max_side_triangle=max_side_triangle, max_workers=max_workers, display=display, use_mmemmap=use_mmemmap)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=False, num_workers=num_workers, prefetch_factor=prefetch_factor)
    model = PPMAndTextureModel(r=r, max_side_triangle=max_side_triangle)
    
    writer = dataset.get_writer()
    trainer = pl.Trainer(callbacks=[writer], accelerator='gpu', devices=int(gpus), strategy="ddp")
    
    print("Start Rendering")
    trainer.predict(model, dataloaders=dataloader, return_predictions=False)
    print("Rendering done")
    writer.write_to_disk(format)
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
    parser.add_argument('--mmemmap', action='store_true', help="Use memory mapped numpy arrays for multi-GPU support")
    args = parser.parse_known_args()[0]

    print(f"Rendering args: {args}")
    if args.display:
        print("[INFO]: Displaying the rendering image slows down the rendering process by about 20%.")

    ppm_and_texture(args.obj, gpus=args.gpus, scroll=args.scroll, output_path=args.output_path, r=args.r, format=args.format, max_side_triangle=args.max_side_triangle, display=args.display, nr_workers=args.nr_workers, prefetch_factor=args.prefetch_factor, use_mmemmap=args.mmemmap)
