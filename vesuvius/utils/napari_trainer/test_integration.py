#!/usr/bin/env python3
"""
Test script to verify napari trainer integration with new train.py
"""

import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    try:
        from models.configuration.config_manager import ConfigManager
        print("✓ ConfigManager imported successfully")
        
        from models.run.train import BaseTrainer
        print("✓ BaseTrainer imported successfully")
        
        from models.datasets import NapariDataset
        print("✓ NapariDataset imported successfully")
        
        # The napari trainer requires Qt bindings which may not be available in test env
        try:
            from utils.napari_trainer.main_window import run_training
            print("✓ napari trainer main_window imported successfully")
        except ImportError as qt_error:
            if "Qt bindings" in str(qt_error):
                print("⚠ Qt bindings not available (expected in headless environment)")
            else:
                raise
        
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_config_manager():
    """Test ConfigManager functionality"""
    print("\nTesting ConfigManager...")
    try:
        from models.configuration.config_manager import ConfigManager
        
        mgr = ConfigManager(verbose=False)  # Use verbose=False for cleaner test output
        print("✓ ConfigManager created successfully")
        
        # Initialize attributes first (needed before update_config)
        mgr.tr_info = {}
        mgr.tr_configs = {}
        mgr.model_config = {}
        mgr.dataset_config = {}
        mgr._init_attributes()
        print("✓ ConfigManager attributes initialized")
        
        # Test setting data format
        mgr.data_format = 'napari'
        print("✓ Data format set to 'napari'")
        
        # Test update_config
        mgr.update_config(
            patch_size=[128, 128],
            min_labeled_ratio=0.1,
            max_epochs=5
        )
        print("✓ Config updated successfully")
        print(f"  - Patch size: {mgr.train_patch_size}")
        print(f"  - Min labeled ratio: {mgr.min_labeled_ratio}")
        print(f"  - Max epochs: {mgr.max_epoch}")
        
        # Test that set_targets_and_data exists
        assert hasattr(mgr, 'set_targets_and_data'), "ConfigManager missing set_targets_and_data method"
        print("✓ set_targets_and_data method exists")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_napari_dataset_structure():
    """Test NapariDataset structure"""
    print("\nTesting NapariDataset structure...")
    try:
        from models.datasets.napari_dataset import NapariDataset
        from models.datasets.base_dataset import BaseDataset
        
        # Check inheritance
        assert issubclass(NapariDataset, BaseDataset), "NapariDataset should inherit from BaseDataset"
        print("✓ NapariDataset inherits from BaseDataset")
        
        # Check required methods
        assert hasattr(NapariDataset, '_initialize_volumes'), "NapariDataset missing _initialize_volumes"
        print("✓ _initialize_volumes method exists")
        
        assert hasattr(NapariDataset, '_get_images_from_napari'), "NapariDataset missing _get_images_from_napari"
        print("✓ _get_images_from_napari method exists")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def main():
    """Run all tests"""
    print("Running napari trainer integration tests...\n")
    
    tests = [
        ("Imports", test_imports),
        ("ConfigManager", test_config_manager),
        ("NapariDataset Structure", test_napari_dataset_structure)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running {test_name} test...")
        print('='*50)
        success = test_func()
        results.append((test_name, success))
    
    print(f"\n{'='*50}")
    print("Test Summary:")
    print('='*50)
    
    all_passed = True
    for test_name, success in results:
        status = "PASSED" if success else "FAILED"
        print(f"{test_name}: {status}")
        if not success:
            all_passed = False
    
    if all_passed:
        print("\n✓ All tests passed! The napari trainer should work with the new train.py")
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
