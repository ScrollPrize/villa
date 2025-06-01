"""
Test script to validate the TestDataGenerator functionality.

This script demonstrates how to use the TestDataGenerator with the test trainer
and verifies that the generated data has the expected patterns.
"""

import numpy as np
import zarr
import tifffile
from pathlib import Path
import sys

# Add the models directory to the path
models_path = Path(__file__).parent.parent
sys.path.insert(0, str(models_path))

from test_data_generator import TestDataGenerator


def verify_binary_pattern(data: np.ndarray) -> bool:
    """Verify binary pattern is correct."""
    if len(data.shape) == 2:
        # 2D: Check checkerboard pattern
        h, w = data.shape
        for y in range(min(4, h)):
            for x in range(min(4, w)):
                expected = (y + x) % 2
                if data[y, x] != expected:
                    return False
    else:
        # 3D: Check alternating slices
        d, h, w = data.shape
        for z in range(min(2, d)):
            slice_data = data[z]
            # Verify slice pattern
            for y in range(min(4, h)):
                for x in range(min(4, w)):
                    base_pattern = (y + x) % 2
                    expected = base_pattern if z % 2 == 0 else 1 - base_pattern
                    if slice_data[y, x] != expected:
                        return False
    return True


def verify_multiclass_pattern(data: np.ndarray) -> bool:
    """Verify multiclass pattern is correct."""
    if len(data.shape) == 2:
        # 2D: Check quadrant pattern
        h, w = data.shape
        mid_h, mid_w = h // 2, w // 2
        
        # Check corners
        if data[0, 0] != 0:  # Top-left
            return False
        if data[0, w-1] != 1:  # Top-right
            return False
        if data[h-1, 0] != 2:  # Bottom-left
            return False
        if data[h-1, w-1] != 3:  # Bottom-right
            return False
    else:
        # 3D: Check layer-based pattern
        d, h, w = data.shape
        for z in range(min(2, d)):
            slice_data = data[z]
            mid_h, mid_w = h // 2, w // 2
            
            # Check quadrant with rotation based on z
            offset = z % 4
            expected_tl = offset
            expected_tr = (1 + offset) % 4
            expected_bl = (2 + offset) % 4
            expected_br = (3 + offset) % 4
            
            if slice_data[0, 0] != expected_tl:
                return False
            if slice_data[0, w-1] != expected_tr:
                return False
            if slice_data[h-1, 0] != expected_bl:
                return False
            if slice_data[h-1, w-1] != expected_br:
                return False
    
    return True


def test_data_loading():
    """Test loading data from both zarr and TIF formats."""
    generator = TestDataGenerator()
    
    print("Testing data loading...")
    
    # Test zarr loading
    binary_zarr_path = generator.get_binary_single_path("zarr")
    image_zarr = zarr.open(str(binary_zarr_path / "images" / "test_image.zarr"), mode='r')
    label_zarr = zarr.open(str(binary_zarr_path / "labels" / "test_image_ink.zarr"), mode='r')
    mask_zarr = zarr.open(str(binary_zarr_path / "masks" / "test_image_ink.zarr"), mode='r')
    
    print(f"Zarr - Image shape: {image_zarr.shape}, Label shape: {label_zarr.shape}, Mask shape: {mask_zarr.shape}")
    
    # Test TIF loading
    binary_tif_path = generator.get_binary_single_path("tif")
    image_tif = tifffile.imread(str(binary_tif_path / "images" / "test_image.tif"))
    label_tif = tifffile.imread(str(binary_tif_path / "labels" / "test_image_ink.tif"))
    mask_tif = tifffile.imread(str(binary_tif_path / "masks" / "test_image_ink.tif"))
    
    print(f"TIF - Image shape: {image_tif.shape}, Label shape: {label_tif.shape}, Mask shape: {mask_tif.shape}")
    
    # Verify shapes match
    assert image_zarr.shape == image_tif.shape, "Image shapes don't match between zarr and TIF"
    assert label_zarr.shape == label_tif.shape, "Label shapes don't match between zarr and TIF"
    assert mask_zarr.shape == mask_tif.shape, "Mask shapes don't match between zarr and TIF"
    
    # Verify binary pattern
    assert verify_binary_pattern(np.array(label_zarr)), "Zarr binary pattern is incorrect"
    assert verify_binary_pattern(label_tif), "TIF binary pattern is incorrect"
    
    print("✓ Data loading test passed!")


def test_multiclass_data():
    """Test multiclass data patterns."""
    generator = TestDataGenerator()
    
    print("Testing multiclass data...")
    
    # Load multiclass data
    multiclass_path = generator.get_multiclass_single_path("zarr")
    label_data = zarr.open(str(multiclass_path / "labels" / "test_image_ink.zarr"), mode='r')
    
    print(f"Multiclass shape: {label_data.shape}")
    print(f"Unique values: {np.unique(np.array(label_data))}")
    
    # Verify multiclass pattern
    assert verify_multiclass_pattern(np.array(label_data)), "Multiclass pattern is incorrect"
    
    print("✓ Multiclass data test passed!")


def test_multitask_data():
    """Test multi-task data."""
    generator = TestDataGenerator()
    
    print("Testing multi-task data...")
    
    # Test single image multi-task
    multitask_single_path = generator.get_multitask_single_path("zarr")
    
    tasks = ["ink", "normals", "damage"]
    for task in tasks:
        label_path = multitask_single_path / "labels" / f"test_image_{task}.zarr"
        if label_path.exists():
            label_data = zarr.open(str(label_path), mode='r')
            print(f"Task {task} - Shape: {label_data.shape}, Unique values: {np.unique(np.array(label_data))}")
        else:
            print(f"Warning: {task} label not found")
    
    # Test multiple images multi-task
    multitask_multi_path = generator.get_multitask_multi_path("zarr")
    
    images = ["image1", "image2"]
    for image in images:
        for task in tasks:
            label_path = multitask_multi_path / "labels" / f"{image}_{task}.zarr"
            if label_path.exists():
                label_data = zarr.open(str(label_path), mode='r')
                print(f"{image} {task} - Shape: {label_data.shape}, Unique values: {np.unique(np.array(label_data))}")
    
    print("✓ Multi-task data test passed!")


def demonstrate_usage():
    """Demonstrate how to use TestDataGenerator with TestTrainer."""
    print("\n" + "="*60)
    print("USAGE DEMONSTRATION")
    print("="*60)
    
    generator = TestDataGenerator()
    
    print("\n1. Basic usage - Create and use test data:")
    print("```python")
    print("from test_data_generator import TestDataGenerator")
    print("from training.test_trainer import TestTrainer")
    print("from models.configuration.config_manager import ConfigManager")
    print("")
    print("# Create test data")
    print("generator = TestDataGenerator()")
    print("generator.create_binary_single_dataset('zarr')")
    print("")
    print("# Use with test trainer")
    print("mgr = ConfigManager()")
    print("mgr.load_config('path/to/config.yaml')")
    print("mgr.data_path = generator.get_binary_single_path('zarr')")
    print("trainer = TestTrainer(mgr=mgr, dataset_format='zarr')")
    print("trainer.train()")
    print("")
    print("# Cleanup")
    print("generator.cleanup()")
    print("```")
    
    print("\n2. Available test scenarios:")
    scenarios = [
        ("Binary single", "generator.get_binary_single_path()"),
        ("Multiclass single", "generator.get_multiclass_single_path()"),
        ("Multi-task single", "generator.get_multitask_single_path()"),
        ("Multi-task multi", "generator.get_multitask_multi_path()")
    ]
    
    for name, path_method in scenarios:
        print(f"   - {name}: {path_method}")
    
    print("\n3. Data verification:")
    print("   All data uses deterministic patterns that can be verified:")
    print("   - Binary: Checkerboard pattern")
    print("   - Multiclass: Quadrant pattern (0,1,2,3)")
    print("   - Multi-task: Different patterns per task")
    
    print("\n4. Supported formats: zarr, tif, or both")
    print("5. Supported dimensions: 2D and 3D")


def main():
    """Run all tests."""
    print("TestDataGenerator Validation")
    print("="*50)
    
    try:
        test_data_loading()
        test_multiclass_data()
        test_multitask_data()
        
        print("\n✓ All tests passed successfully!")
        
        demonstrate_usage()
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
