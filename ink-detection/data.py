import os
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = 9331200000000
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,42))
import random
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import threading
import gc
import sys
from scipy import ndimage

class Segment:
    """A class that handles loading and preprocessing of a Vesuvius segment."""
    
    def __init__(self, 
                 segment_id: str,
                 layer_range: tuple = (17, 43),
                 reverse_layers: bool = None,
                 base_path: str = "./train_scrolls",
                 tile_size: int = 64,
                 clip_limit: int = 200,
                 xy_scale: float = 1.0,
                 xyz_scale: float = 1.0,
                 z_scale: float = 1.0):
        self.segment_id = segment_id
        self.segment_id_base = segment_id.split("_")[0]
        self.start_idx, self.end_idx = layer_range
        self.base_path = base_path
        self.tile_size = tile_size
        self.clip_limit = clip_limit
        self.xy_scale = xy_scale
        self.xyz_scale = xyz_scale
        self.z_scale = z_scale
        
        self.reverse_segments = [
            '20230701020044', 'verso', '20230901184804', '20230901234823', 
            '20230531193658', '20231007101615', '20231005123333', '20231011144857', 
            '20230522215721', '20230919113918', '20230625171244', '20231022170900', 
            '20231012173610', '20231016151000'
        ]
        
        if reverse_layers is None:
            self.reverse_layers = any(id_ in self.segment_id_base for id_ in self.reverse_segments)
        else:
            self.reverse_layers = reverse_layers
            
        if not os.path.exists(f"{self.base_path}/{self.segment_id}"):
            self.segment_id = f"{self.segment_id}_superseded"
            if not os.path.exists(f"{self.base_path}/{self.segment_id}"):
                raise FileNotFoundError(f"Segment {segment_id} not found in {base_path}")
        
        self.image, self.mask, self.fragment_mask = self._load_segment()
        
    def _apply_scaling(self, images, mask, fragment_mask):
        """Apply XY or XYZ scaling to the loaded data."""
        if self.xyz_scale != 1.0:
            images = ndimage.zoom(images, (self.xyz_scale, self.xyz_scale, self.xyz_scale), order=1)
            mask = ndimage.zoom(mask, (self.xyz_scale, self.xyz_scale), order=1)
            fragment_mask = ndimage.zoom(fragment_mask, (self.xyz_scale, self.xyz_scale), order=0)
        elif self.z_scale != 1.0:
            images = ndimage.zoom(images, (1.0, 1.0, self.z_scale), order=1)
        elif self.xy_scale != 1.0:
            new_height = int(images.shape[0] * self.xy_scale)
            new_width = int(images.shape[1] * self.xy_scale)
            
            resized_channels = []
            for i in range(images.shape[2]):
                resized_channel = cv2.resize(images[:, :, i], (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                resized_channels.append(resized_channel)
            images = np.stack(resized_channels, axis=2)
            
            mask = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            fragment_mask = cv2.resize(fragment_mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        
        return images, mask, fragment_mask
        
    def _load_segment(self):
        """Load segment data from disk with improved robustness to format variations."""
        import glob
        
        images = []
        idxs = range(self.start_idx, self.end_idx)
        layers_dir = f"{self.base_path}/{self.segment_id}/layers"
        
        layer_extension = None
        for ext in ['.tif', '.jpg', '.png', '.tiff']:
            if os.path.exists(f"{layers_dir}/{self.start_idx:02d}{ext}"):
                layer_extension = ext
                break
        
        if layer_extension is None:
            layer_files = glob.glob(f"{layers_dir}/[0-9][0-9].*")
            if layer_files:
                layer_extension = os.path.splitext(layer_files[0])[1]
            # else:
            #     raise FileNotFoundError(f"No layer files found in {layers_dir}")
        if layer_extension is None:
            layer_files = glob.glob(f"{layers_dir}/[0-9][0-9][0-9].*")
            if layer_files:
                layer_extension = os.path.splitext(layer_files[0])[1]
            else:
                raise FileNotFoundError(f"No layer files found in {layers_dir}")    
        for i in idxs:
            layer_path = f"{layers_dir}/{i:02d}{layer_extension}"
            if self.segment_id=='20231122192640':
                layer_path = f"{layers_dir}/{i:03d}{layer_extension}"
            if not os.path.exists(layer_path):
                continue
                
            image = cv2.imread(layer_path,0)
            if image is None:
                continue
                
            pad0 = (self.tile_size - image.shape[0] % self.tile_size) % self.tile_size
            pad1 = (self.tile_size - image.shape[1] % self.tile_size) % self.tile_size
            image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)
            image = np.clip(image, 0, self.clip_limit)
            
            images.append(image)
            
        if not images:
            raise ValueError(f"No valid layer images found for segment {self.segment_id}")
            
        images = np.stack(images, axis=2,dtype=np.uint8)
        if self.reverse_layers:
            images = images[:, :, ::-1]
            
        inklabels_files = glob.glob(f"{self.base_path}/{self.segment_id}/*inklabels*")
        if not inklabels_files:
            mask = np.zeros((images.shape[0], images.shape[1]), dtype=np.float32)
        else:
            mask = cv2.imread(inklabels_files[0],0)
            if mask is None:
                mask = np.zeros((images.shape[0], images.shape[1]), dtype=np.float32)
            
        mask_pattern = f"{self.base_path}/{self.segment_id}/*mask*"
        mask_files = [f for f in glob.glob(mask_pattern) if 'inklabels' not in f.lower()]
        if not mask_files:
            fragment_mask = np.ones((images.shape[0], images.shape[1]), dtype=np.uint8) * 255
        else:
            fragment_mask = cv2.imread(mask_files[0],0)
            if fragment_mask is None:
                fragment_mask = np.ones((images.shape[0], images.shape[1]), dtype=np.uint8) * 255
        
        if 'keV' in self.segment_id or 'rag' in self.segment_id or self.segment_id in ['658','20250511003658','550','500p2','500P2um','500P4um','2um_44kev_0.22m','2um_62kev_0.22m','2um_77kev_0.35m','2um_43kev_0.22m','08312025_l2_0','z_dbg_gen_00668_inp_hr','z_dbg_gen_00294_inp_hr','20241108120732','20241113080880','20241108111522','20241025145341','20241025145701','20241113090990','20241030152031']:
            pad0 = (self.tile_size - mask.shape[0] % self.tile_size) % self.tile_size
            pad1 = (self.tile_size - mask.shape[1] % self.tile_size) % self.tile_size
            mask = np.pad(mask, [(0, pad0), (0, pad1)], constant_values=0)
        
        pad0 = (self.tile_size - fragment_mask.shape[0] % self.tile_size) % self.tile_size
        pad1 = (self.tile_size - fragment_mask.shape[1] % self.tile_size) % self.tile_size
        fragment_mask = np.pad(fragment_mask, [(0, pad0), (0, pad1)], constant_values=0)
        
        mask = mask.astype('float32') / 255.0
        
        if self.xy_scale != 1.0 or self.xyz_scale != 1.0 or self.z_scale != 1.0:
            images, mask, fragment_mask = self._apply_scaling(images, mask, fragment_mask)
        
        return images, mask, fragment_mask
        
    def get_data(self):
        """Get the segment data."""
        return self.image, self.mask, self.fragment_mask
        
    @property
    def shape(self):
        """Get the shape of the segment data."""
        return self.image.shape
        
    def __repr__(self):
        return f"Segment(id={self.segment_id}, shape={self.shape}, layers={self.start_idx}-{self.end_idx})"


class VesuviusAugmentation:
    """A class to handle data augmentation for Vesuvius dataset."""
    
    def __init__(self, size=64, in_chans=26, mode='train'):
        self.size = size
        self.in_chans = in_chans
        self.mode = mode
        
        if mode == 'train':
            self.transform = A.Compose([
                A.Resize(size, size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=(-.4,.3),contrast_limit=(-.4,.4),p=0.65),
                A.ShiftScaleRotate(rotate_limit=360, shift_limit=0.15, scale_limit=0.15, p=0.75),
                A.OneOf([
                    A.GaussianBlur(),
                    A.MotionBlur(),
                ], p=0.4),
                A.CoarseDropout(max_holes=2, max_width=int(size * 0.2), max_height=int(size * 0.2), 
                                mask_fill_value=0, p=0.5),
                A.Normalize(
                    mean=[0] * in_chans,
                    std=[1] * in_chans
                ),
                ToTensorV2(transpose_mask=True),
            ])
            
            self.rotate = A.Compose([A.Rotate(5, p=1)])
            
        else:
            self.transform = A.Compose([
                A.Resize(size, size),
                A.Normalize(
                    mean=[0] * in_chans,
                    std=[1] * in_chans
                ),
                ToTensorV2(transpose_mask=True),
            ])
    
    def __call__(self, **kwargs):
        """Apply the transformation."""
        return self.transform(**kwargs)


class VesuviusDataset(Dataset):
    """A PyTorch Dataset for the Vesuvius Challenge scrolls."""
    
    def __init__(self,
                 volumes_config,
                 layer_range=(17, 43),
                 base_path="./train_scrolls",
                 tile_size=256,
                 sub_tile_size=64,
                 stride=32,
                 transform=None,
                 mode='train',
                 filter_empty=True,
                 filter_threshold=0.05,
                 valid_id=None,
                 downsample_mask=16,
                 use_threading=False,
                 thread_batch_size=4):
        self.volumes_config = volumes_config
        self.layer_range = layer_range
        self.base_path = base_path
        self.tile_size = tile_size
        self.sub_tile_size = sub_tile_size
        self.stride = stride
        self.transform = transform
        self.mode = mode
        self.filter_empty = filter_empty
        self.filter_threshold = filter_threshold
        self.valid_id = valid_id
        self.downsample_mask = downsample_mask
        self.use_threading = use_threading
        self.thread_batch_size = thread_batch_size
        
        self.configs = self._standardize_configs()
        self.samples = self._preprocess_segments()
        gc.collect()
        
        print(f"Loaded {len(self.samples)} {mode} samples from {len(self.configs)} segments")
        
    def _standardize_configs(self):
        """Convert volumes_config to standardized format."""
        configs = []
        for config in self.volumes_config:
            if isinstance(config, str):
                configs.append({'segment_id': config})
            elif isinstance(config, dict):
                if 'segment_id' not in config:
                    raise ValueError("Each configuration must contain 'segment_id'")
                configs.append(config)
            else:
                raise ValueError(f"Invalid configuration type: {type(config)}")
        return configs
        
    def _process_segment(self, config):
        """Load and process a single segment into samples."""
        segment_id = config['segment_id']
        layer_range = config.get('layer_range', self.layer_range)
        reverse_layers = config.get('reverse_layers', None)
        xy_scale = config.get('xy_scale', 1.0)
        xyz_scale = config.get('xyz_scale', 1.0)
        z_scale = config.get('z_scale', 1.0)
        samples = []
        windows_dict = {}
        
        try:
            segment = Segment(
                segment_id=segment_id,
                layer_range=layer_range,
                reverse_layers=reverse_layers,
                base_path=self.base_path,
                tile_size=self.tile_size,
                xy_scale=xy_scale,
                xyz_scale=xyz_scale,
                z_scale=z_scale
            )
            
            if self.mode == 'train' and segment_id == self.valid_id:
                return []
            if self.mode == 'val' and segment_id != self.valid_id:
                return []
                
            image, mask, fragment_mask = segment.get_data()
            
            x1_list = list(range(0, image.shape[1] - self.tile_size + 1, self.stride))
            y1_list = list(range(0, image.shape[0] - self.tile_size + 1, self.stride))
            
            for y1 in y1_list:
                for x1 in x1_list:
                    y2 = y1 + self.tile_size
                    x2 = x1 + self.tile_size
                    
                    if np.any(fragment_mask[y1:y2, x1:x2] == 0):
                        continue
                    
                    if self.mode == 'train' and self.filter_empty:
                        if np.all(mask[y1:y2, x1:x2] < self.filter_threshold):
                            continue
                    
                    for sub_y in range(0, self.tile_size, self.sub_tile_size):
                        for sub_x in range(0, self.tile_size, self.sub_tile_size):
                            sub_y1 = y1 + sub_y
                            sub_x1 = x1 + sub_x
                            sub_y2 = min(sub_y1 + self.sub_tile_size, image.shape[0])
                            sub_x2 = min(sub_x1 + self.sub_tile_size, image.shape[1])
                            
                            if sub_y2 - sub_y1 != self.sub_tile_size or sub_x2 - sub_x1 != self.sub_tile_size:
                                continue
                                
                            if self.mode == 'val':
                                key = (sub_y1, sub_y2, sub_x1, sub_x2)
                                if key in windows_dict:
                                    continue
                                windows_dict[key] = True
                                
                            sub_img = image[sub_y1:sub_y2, sub_x1:sub_x2]
                            if sub_img.shape[2] != layer_range[1] - layer_range[0]:
                                print(f"Warning: {segment_id} subtile has wrong channel count: {sub_img.shape}")
                                continue
                                
                            samples.append({
                                'segment_id': segment_id,
                                'image': sub_img,
                                'mask': mask[sub_y1:sub_y2, sub_x1:sub_x2],
                                'coords': [sub_x1, sub_y1, sub_x2, sub_y2]
                            })
            print(f"Processed segment {segment_id} with shape {segment.shape}, image_shape: {image.shape}, mask_shape: {mask.shape} , n_samples: {len(samples)}")
            del segment, image, mask, fragment_mask
            return samples
            
        except Exception as e:
            print(f"Error processing segment {segment_id}: {e}")
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            return []
    
    def _thread_worker(self, idx, config, results):
        """Worker function for threaded processing."""
        results[idx] = self._process_segment(config)
    
    def _load_and_process_segments_threaded(self, batch_size=4):
        """Load and process segments in parallel batches."""
        all_samples = []
        configs = self.configs.copy()
        total_batches = (len(configs) + batch_size - 1) // batch_size
        
        for i in range(0, len(configs), batch_size):
            batch_configs = configs[i:i+batch_size]
            threads = []
            results = [None] * len(batch_configs)
            
            for batch_idx, config in enumerate(batch_configs):
                thread = threading.Thread(
                    target=self._thread_worker, 
                    args=(batch_idx, config, results)
                )
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join()
            
            for batch_samples in results:
                if batch_samples:
                    all_samples.extend(batch_samples)
            
            print(f"Completed batch {i//batch_size + 1}/{total_batches}")
        
        return all_samples
    
    def _load_and_process_segments_sequential(self):
        """Load and process segments sequentially to save memory."""
        all_samples = []
        for config in self.configs:
            samples = self._process_segment(config)
            all_samples.extend(samples)
        return all_samples
            
    def _load_segments(self):
        """This method is no longer used but kept for compatibility."""
        return []
        
    def _preprocess_segments(self):
        """Process all segments into samples, either sequentially or with threading."""
        if self.use_threading:
            return self._load_and_process_segments_threaded(self.thread_batch_size)
        else:
            return self._load_and_process_segments_sequential()
    
    def fourth_augment(self, image):
        """Applies temporal augmentation to the 3D image volume."""
        in_chans = image.shape[2]
        image_tmp = np.zeros_like(image)
        cropping_num = random.randint(18, min(26, in_chans))

        start_idx = random.randint(0, in_chans - cropping_num)
        crop_indices = np.arange(start_idx, start_idx + cropping_num)

        start_paste_idx = random.randint(0, in_chans - cropping_num)

        temporal_random_cutout_idx = []
        if cropping_num > 2:
            tmp = list(range(start_paste_idx, start_paste_idx + cropping_num))
            random.shuffle(tmp)
            cutout_idx = random.randint(0, min(2, cropping_num - 1))
            temporal_random_cutout_idx = tmp[:cutout_idx]

        image_tmp[..., start_paste_idx:start_paste_idx + cropping_num] = image[..., crop_indices]

        if random.random() > 0.4 and temporal_random_cutout_idx:
            for idx in temporal_random_cutout_idx:
                if 0 <= idx < image_tmp.shape[2]:
                    image_tmp[..., idx] = 0
            
        return image_tmp
        
    def __len__(self):
        """Return the number of samples."""
        return len(self.samples)
        
    def __getitem__(self, idx):
        """Get a sample by index."""
        sample = self.samples[idx]
        image = sample['image']
        mask = sample['mask']
        coords = sample['coords']
        
        mask_with_channel = mask[:, :, None]
        
        if self.transform:
            data = self.transform(image=image, mask=mask_with_channel)
            image = data['image'].unsqueeze(0)
            mask = data['mask']
            
            if self.downsample_mask > 1:
                mask = F.interpolate(
                    mask.unsqueeze(0),
                    (self.sub_tile_size // self.downsample_mask, self.sub_tile_size // self.downsample_mask)
                ).squeeze(0)
        
        if self.mode == 'val':
            return image, mask, coords
        else:
            return image, mask