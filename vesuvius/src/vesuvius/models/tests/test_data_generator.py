"""
Test data generator for training pipeline validation.

This module creates deterministic test datasets for all training scenarios:
- Binary (single task, single image)
- Multiclass (single task, single image)  
- Multi-task single image
- Multi-task multiple images

The data patterns are completely predictable for verification purposes.
"""

import numpy as np
import zarr
import tifffile
from pathlib import Path
import shutil
from typing import Tuple, List, Union, Optional


class TestDataGenerator:
    """
    Creates deterministic test datasets for training pipeline validation.
    
    Supports all training scenarios with predictable data patterns that can be
    verified in tests. Creates both zarr and TIF formats.
    """
    
    def __init__(self, base_path: str = "models/tests/test_data"):
        """
        Initialize the test data generator.
        
        Parameters
        ----------
        base_path : str
            Base directory where test data will be created
        """
        self.base_path = Path(base_path)
        
        # Define paths for each scenario
        self.binary_single_path = self.base_path / "binary_single"
        self.multiclass_single_path = self.base_path / "multiclass_single"
        self.multitask_single_path = self.base_path / "multitask_single"
        self.multitask_multi_path = self.base_path / "multitask_multi"
        
        # Standard test shapes - larger than patch sizes to allow multiple patches
        self.shape_2d = (128, 128)
        self.shape_3d = (32, 128, 128)
    
    def create_binary_pattern_2d(self, shape: Tuple[int, int] = None) -> np.ndarray:
        """
        Create deterministic binary pattern for 2D data.
        
        Uses checkerboard pattern: (y + x) % 2
        
        Parameters
        ----------
        shape : tuple, optional
            Shape (height, width), defaults to self.shape_2d
            
        Returns
        -------
        np.ndarray
            Binary array with values 0 and 1
        """
        if shape is None:
            shape = self.shape_2d
        
        h, w = shape
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        pattern = (y_coords + x_coords) % 2
        return pattern.astype(np.float32)
    
    def create_binary_pattern_3d(self, shape: Tuple[int, int, int] = None) -> np.ndarray:
        """
        Create deterministic binary pattern for 3D data.
        
        Uses alternating slices: z % 2
        
        Parameters
        ----------
        shape : tuple, optional
            Shape (depth, height, width), defaults to self.shape_3d
            
        Returns
        -------
        np.ndarray
            Binary array with values 0 and 1
        """
        if shape is None:
            shape = self.shape_3d
        
        d, h, w = shape
        pattern = np.zeros(shape, dtype=np.float32)
        
        for z in range(d):
            if z % 2 == 0:
                pattern[z] = self.create_binary_pattern_2d((h, w))
            else:
                pattern[z] = 1 - self.create_binary_pattern_2d((h, w))
        
        return pattern
    
    def create_multiclass_pattern_2d(self, shape: Tuple[int, int] = None, num_classes: int = 4) -> np.ndarray:
        """
        Create deterministic multiclass pattern for 2D data.
        
        Uses quadrant pattern for 4 classes.
        
        Parameters
        ----------
        shape : tuple, optional
            Shape (height, width), defaults to self.shape_2d
        num_classes : int, default=4
            Number of classes
            
        Returns
        -------
        np.ndarray
            Multiclass array with values 0 to num_classes-1
        """
        if shape is None:
            shape = self.shape_2d
        
        h, w = shape
        pattern = np.zeros(shape, dtype=np.float32)
        
        # Create quadrant pattern for 4 classes
        mid_h, mid_w = h // 2, w // 2
        
        # Top-left: class 0 (already zeros)
        # Top-right: class 1
        pattern[:mid_h, mid_w:] = 1
        # Bottom-left: class 2  
        pattern[mid_h:, :mid_w] = 2
        # Bottom-right: class 3
        pattern[mid_h:, mid_w:] = 3
        
        if num_classes < 4:
            # Reduce to fewer classes if needed
            pattern = pattern % num_classes
        
        return pattern
    
    def create_multiclass_pattern_3d(self, shape: Tuple[int, int, int] = None, num_classes: int = 4) -> np.ndarray:
        """
        Create deterministic multiclass pattern for 3D data.
        
        Uses layer-based pattern: each depth slice gets a different base pattern.
        
        Parameters
        ----------
        shape : tuple, optional
            Shape (depth, height, width), defaults to self.shape_3d
        num_classes : int, default=4
            Number of classes
            
        Returns
        -------
        np.ndarray
            Multiclass array with values 0 to num_classes-1
        """
        if shape is None:
            shape = self.shape_3d
        
        d, h, w = shape
        pattern = np.zeros(shape, dtype=np.float32)
        
        for z in range(d):
            base_pattern = self.create_multiclass_pattern_2d((h, w), num_classes)
            # Rotate pattern based on depth to create variation
            offset = z % num_classes
            pattern[z] = (base_pattern + offset) % num_classes
        
        return pattern
    
    def create_image_data_2d(self, shape: Tuple[int, int] = None) -> np.ndarray:
        """
        Create synthetic image data for 2D.
        
        Parameters
        ----------
        shape : tuple, optional
            Shape (height, width), defaults to self.shape_2d
            
        Returns
        -------
        np.ndarray
            Normalized image data
        """
        if shape is None:
            shape = self.shape_2d
        
        h, w = shape
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        
        # Create gradient patterns
        image = (0.5 * y_coords / h + 
                0.3 * x_coords / w + 
                0.2 * np.sin(y_coords * 2 * np.pi / h) * np.cos(x_coords * 2 * np.pi / w))
        
        # Normalize to [0, 1]
        image = (image - image.min()) / (image.max() - image.min())
        return image.astype(np.float32)
    
    def create_image_data_3d(self, shape: Tuple[int, int, int] = None) -> np.ndarray:
        """
        Create synthetic image data for 3D.
        
        Parameters
        ----------
        shape : tuple, optional
            Shape (depth, height, width), defaults to self.shape_3d
            
        Returns
        -------
        np.ndarray
            Normalized image data
        """
        if shape is None:
            shape = self.shape_3d
        
        d, h, w = shape
        image = np.zeros(shape, dtype=np.float32)
        
        for z in range(d):
            base_2d = self.create_image_data_2d((h, w))
            # Add depth variation
            depth_factor = 0.8 + 0.4 * np.sin(z * 2 * np.pi / d)
            image[z] = base_2d * depth_factor
        
        # Normalize to [0, 1]
        image = (image - image.min()) / (image.max() - image.min())
        return image
    
    def create_mask_data(self, shape: Tuple[int, ...], coverage: float = 0.85) -> np.ndarray:
        """
        Create mask data with deterministic excluded regions.
        
        Parameters
        ----------
        shape : tuple
            Shape of the mask
        coverage : float, default=0.85
            Fraction of area that should be valid (1.0)
            
        Returns
        -------
        np.ndarray
            Binary mask (0=exclude, 1=include)
        """
        mask = np.ones(shape, dtype=np.float32)
        
        if len(shape) == 2:
            h, w = shape
            # Create deterministic excluded regions
            # Small rectangular exclusion in top-right
            exclude_h = int(h * 0.1)
            exclude_w = int(w * 0.1)
            mask[:exclude_h, -exclude_w:] = 0
            
            # Small exclusion in bottom-left
            mask[-exclude_h:, :exclude_w] = 0
        else:
            d, h, w = shape
            # Exclude some edge slices
            exclude_d = max(1, int(d * 0.05))
            mask[:exclude_d] = 0
            mask[-exclude_d:] = 0
            
            # Exclude corner regions in middle slices
            mid_z = d // 2
            exclude_h = int(h * 0.1)
            exclude_w = int(w * 0.1)
            mask[mid_z, :exclude_h, :exclude_w] = 0
            mask[mid_z, -exclude_h:, -exclude_w:] = 0
        
        return mask
    
    def _save_data(self, data: np.ndarray, filepath: Path, format_type: str):
        """
        Save data in the specified format.
        
        Parameters
        ----------
        data : np.ndarray
            Data to save
        filepath : Path
            Output file path (without extension)
        format_type : str
            Either 'zarr' or 'tif'
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if format_type == "zarr":
            zarr_path = filepath.with_suffix('.zarr')
            zarr.save_array(str(zarr_path), data)
        elif format_type == "tif":
            tif_path = filepath.with_suffix('.tif')
            tifffile.imwrite(str(tif_path), data)
        else:
            raise ValueError(f"Unknown format: {format_type}")
    
    def create_binary_single_dataset(self, format_type: str = "both", is_3d: bool = True):
        """
        Create binary single-task, single-image dataset.
        
        Parameters
        ----------
        format_type : str
            'zarr', 'tif', or 'both'
        is_3d : bool, default=True
            Whether to create 3D or 2D data
        """
        formats = ["zarr", "tif"] if format_type == "both" else [format_type]
        
        # Create data
        if is_3d:
            image_data = self.create_image_data_3d()
            label_data = self.create_binary_pattern_3d()
            mask_data = self.create_mask_data(self.shape_3d)
        else:
            image_data = self.create_image_data_2d()
            label_data = self.create_binary_pattern_2d()
            mask_data = self.create_mask_data(self.shape_2d)
        
        for fmt in formats:
            base_path = self.binary_single_path / f"{fmt}_data"
            
            # Save files
            self._save_data(image_data, base_path / "images" / "test_image", fmt)
            self._save_data(label_data, base_path / "labels" / "test_image_ink", fmt)
            self._save_data(mask_data, base_path / "masks" / "test_image_ink", fmt)
        
        print(f"Created binary single dataset ({'3D' if is_3d else '2D'}) in format(s): {format_type}")
    
    def create_multiclass_single_dataset(self, format_type: str = "both", is_3d: bool = True):
        """
        Create multiclass single-task, single-image dataset.
        
        Parameters
        ----------
        format_type : str
            'zarr', 'tif', or 'both'
        is_3d : bool, default=True
            Whether to create 3D or 2D data
        """
        formats = ["zarr", "tif"] if format_type == "both" else [format_type]
        
        # Create data
        if is_3d:
            image_data = self.create_image_data_3d()
            label_data = self.create_multiclass_pattern_3d()
            mask_data = self.create_mask_data(self.shape_3d)
        else:
            image_data = self.create_image_data_2d()
            label_data = self.create_multiclass_pattern_2d()
            mask_data = self.create_mask_data(self.shape_2d)
        
        for fmt in formats:
            base_path = self.multiclass_single_path / f"{fmt}_data"
            
            # Save files
            self._save_data(image_data, base_path / "images" / "test_image", fmt)
            self._save_data(label_data, base_path / "labels" / "test_image_ink", fmt)
            self._save_data(mask_data, base_path / "masks" / "test_image_ink", fmt)
        
        print(f"Created multiclass single dataset ({'3D' if is_3d else '2D'}) in format(s): {format_type}")
    
    def create_multitask_single_dataset(self, format_type: str = "both", is_3d: bool = True):
        """
        Create multi-task single-image dataset.
        
        Parameters
        ----------
        format_type : str
            'zarr', 'tif', or 'both'
        is_3d : bool, default=True
            Whether to create 3D or 2D data
        """
        formats = ["zarr", "tif"] if format_type == "both" else [format_type]
        
        # Create shared image data
        if is_3d:
            image_data = self.create_image_data_3d()
            shape = self.shape_3d
        else:
            image_data = self.create_image_data_2d()
            shape = self.shape_2d
        
        # Create different label patterns for each task
        tasks = {
            "ink": self.create_binary_pattern_3d(shape) if is_3d else self.create_binary_pattern_2d(shape),
            "normals": self.create_multiclass_pattern_3d(shape) if is_3d else self.create_multiclass_pattern_2d(shape),
            "damage": (1 - (self.create_binary_pattern_3d(shape) if is_3d else self.create_binary_pattern_2d(shape)))  # Inverted binary
        }
        
        # Create masks for each task
        masks = {task: self.create_mask_data(shape) for task in tasks.keys()}
        
        for fmt in formats:
            base_path = self.multitask_single_path / f"{fmt}_data"
            
            # Save shared image
            self._save_data(image_data, base_path / "images" / "test_image", fmt)
            
            # Save task-specific labels and masks
            for task, label_data in tasks.items():
                self._save_data(label_data, base_path / "labels" / f"test_image_{task}", fmt)
                self._save_data(masks[task], base_path / "masks" / f"test_image_{task}", fmt)
        
        print(f"Created multitask single dataset ({'3D' if is_3d else '2D'}) with tasks: {list(tasks.keys())} in format(s): {format_type}")
    
    def create_multitask_multi_dataset(self, format_type: str = "both", is_3d: bool = True):
        """
        Create multi-task multiple-image dataset.
        
        Parameters
        ----------
        format_type : str
            'zarr', 'tif', or 'both'
        is_3d : bool, default=True
            Whether to create 3D or 2D data
        """
        formats = ["zarr", "tif"] if format_type == "both" else [format_type]
        
        images = ["image1", "image2"]
        tasks = ["ink", "normals", "damage"]
        
        for fmt in formats:
            base_path = self.multitask_multi_path / f"{fmt}_data"
            
            for i, image_id in enumerate(images):
                # Create slightly different image data for each image
                if is_3d:
                    image_data = self.create_image_data_3d()
                    shape = self.shape_3d
                else:
                    image_data = self.create_image_data_2d()
                    shape = self.shape_2d
                
                # Add variation based on image index
                image_data = image_data * (0.8 + 0.4 * i / len(images))
                image_data = np.clip(image_data, 0, 1)
                
                # Save image
                self._save_data(image_data, base_path / "images" / image_id, fmt)
                
                # Create task-specific data with slight variations per image
                for j, task in enumerate(tasks):
                    if task == "ink":
                        if is_3d:
                            label_data = self.create_binary_pattern_3d(shape)
                        else:
                            label_data = self.create_binary_pattern_2d(shape)
                        # Add slight variation based on image
                        if i > 0:
                            label_data = 1 - label_data
                    elif task == "normals":
                        if is_3d:
                            label_data = self.create_multiclass_pattern_3d(shape)
                        else:
                            label_data = self.create_multiclass_pattern_2d(shape)
                        # Rotate classes for variation
                        label_data = (label_data + i) % 4
                    else:  # damage
                        if is_3d:
                            label_data = self.create_binary_pattern_3d(shape)
                        else:
                            label_data = self.create_binary_pattern_2d(shape)
                        # Different pattern for damage
                        label_data = (label_data + j) % 2
                    
                    # Create mask
                    mask_data = self.create_mask_data(shape)
                    
                    # Save labels and masks
                    self._save_data(label_data, base_path / "labels" / f"{image_id}_{task}", fmt)
                    self._save_data(mask_data, base_path / "masks" / f"{image_id}_{task}", fmt)
        
        print(f"Created multitask multi dataset ({'3D' if is_3d else '2D'}) with {len(images)} images and tasks: {tasks} in format(s): {format_type}")
    
    def create_all_scenarios(self, format_type: str = "both", is_3d: bool = True):
        """
        Create all test scenarios.
        
        Parameters
        ----------
        format_type : str
            'zarr', 'tif', or 'both'
        is_3d : bool, default=True
            Whether to create 3D or 2D data
        """
        print(f"Creating all test scenarios ({'3D' if is_3d else '2D'}) in format(s): {format_type}")
        
        self.create_binary_single_dataset(format_type, is_3d)
        self.create_multiclass_single_dataset(format_type, is_3d)
        self.create_multitask_single_dataset(format_type, is_3d)
        self.create_multitask_multi_dataset(format_type, is_3d)
        
        print("All test scenarios created successfully!")
    
    def cleanup(self):
        """Remove all test data."""
        if self.base_path.exists():
            shutil.rmtree(self.base_path)
            print(f"Cleaned up test data at {self.base_path}")
    
    # Path getters for test integration
    def get_binary_single_path(self, format_type: str = "zarr") -> Path:
        """Get path for binary single dataset."""
        return self.binary_single_path / f"{format_type}_data"
    
    def get_multiclass_single_path(self, format_type: str = "zarr") -> Path:
        """Get path for multiclass single dataset."""
        return self.multiclass_single_path / f"{format_type}_data"
    
    def get_multitask_single_path(self, format_type: str = "zarr") -> Path:
        """Get path for multitask single dataset."""
        return self.multitask_single_path / f"{format_type}_data"
    
    def get_multitask_multi_path(self, format_type: str = "zarr") -> Path:
        """Get path for multitask multi dataset."""
        return self.multitask_multi_path / f"{format_type}_data"
    
    def verify_data_creation(self) -> bool:
        """
        Verify that created data has expected patterns.
        
        Returns
        -------
        bool
            True if verification passes
        """
        try:
            # Check binary pattern
            binary_2d = self.create_binary_pattern_2d((4, 4))
            expected = np.array([[0, 1, 0, 1],
                               [1, 0, 1, 0], 
                               [0, 1, 0, 1],
                               [1, 0, 1, 0]], dtype=np.float32)
            assert np.array_equal(binary_2d, expected), "Binary 2D pattern incorrect"
            
            # Check multiclass pattern
            multiclass_2d = self.create_multiclass_pattern_2d((4, 4))
            assert multiclass_2d[0, 0] == 0, "Multiclass top-left should be 0"
            assert multiclass_2d[0, 3] == 1, "Multiclass top-right should be 1"
            assert multiclass_2d[3, 0] == 2, "Multiclass bottom-left should be 2"
            assert multiclass_2d[3, 3] == 3, "Multiclass bottom-right should be 3"
            
            print("Data pattern verification passed!")
            return True
            
        except AssertionError as e:
            print(f"Data pattern verification failed: {e}")
            return False


# Convenience functions
def create_test_data(format_type: str = "both", is_3d: bool = True) -> TestDataGenerator:
    """
    Convenience function to create all test data.
    
    Parameters
    ----------
    format_type : str
        'zarr', 'tif', or 'both'
    is_3d : bool, default=True
        Whether to create 3D or 2D data
        
    Returns
    -------
    TestDataGenerator
        The generator instance (for cleanup)
    """
    generator = TestDataGenerator()
    generator.create_all_scenarios(format_type, is_3d)
    return generator


def cleanup_test_data():
    """Convenience function to cleanup test data."""
    generator = TestDataGenerator()
    generator.cleanup()


if __name__ == "__main__":
    # Create all test data when run directly
    print("Creating comprehensive test datasets...")
    
    # Verify patterns first
    generator = TestDataGenerator()
    if not generator.verify_data_creation():
        exit(1)
    
    # Create both 2D and 3D datasets
    generator.create_all_scenarios("both", is_3d=True)  # 3D datasets
    generator.create_all_scenarios("both", is_3d=False)  # 2D datasets
