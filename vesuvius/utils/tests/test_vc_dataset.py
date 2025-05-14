#!/usr/bin/env python
"""
Test script for validating the VCDataset with multiresolution zarr stores.
Tests loading patches through the VCDataset class and test-time augmentation.
"""

import os
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

# Add the project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '../..'))
sys.path.insert(0, project_root)

from data.vc_dataset import VCDataset
from utils.models.tta import get_tta_augmented_inputs, merge_tensors

def test_dataset():
    print("\n=== Testing VCDataset with Local Zarr ===")
    
    # Path to the local zarr store
    zarr_path = os.path.join(script_dir, "s5_test.zarr")
    
    if not os.path.exists(zarr_path):
        print(f"Local zarr not found at {zarr_path}")
        return False
    
    print(f"Loading dataset from zarr path: {zarr_path}")
    
    # Define different normalization schemes to test
    normalization_schemes = [
        ("none", False),  # No normalization
        ("zscore", True),  # z-score normalization
        ("minmax", True)   # min-max normalization
    ]
    
    # Testing different patch configurations
    patch_configs = [
        {"size": (64, 64, 64), "step": 0.75, "name": "Small patches"},
        {"size": (128, 128, 128), "step": 0.5, "name": "Medium patches"},
        {"size": (192, 192, 192), "step": 0.9, "name": "Large patches (like inference)"}
    ]
    
    all_tests_passed = True
    
    # Test each normalization scheme with different configurations
    for norm_scheme, normalize in normalization_schemes:
        print(f"\n=== Testing normalization scheme: {norm_scheme} (normalize={normalize}) ===")
        
        for patch_config in patch_configs:
            # Skip large patches for faster testing unless explicitly requested
            if patch_config["size"][0] > 128 and not os.environ.get("TEST_LARGE_PATCHES"):
                print(f"Skipping {patch_config['name']} (set TEST_LARGE_PATCHES=1 to enable)")
                continue
                
            print(f"\n--- Testing {patch_config['name']} ---")
            patch_size = patch_config["size"]
            step_size = patch_config["step"]
            
            # Initialize dataset
            start_time = time.time()
            targets = [{"name": "test_output"}]
            
            try:
                dataset = VCDataset(
                    input_path=zarr_path,
                    targets=targets,
                    patch_size=patch_size,
                    input_format="zarr",
                    step_size=step_size,
                    normalize=normalize,
                    normalization_scheme=norm_scheme,
                    verbose=True
                )
                
                init_time = time.time() - start_time
                print(f"Dataset initialized in {init_time:.2f} seconds")
                print(f"Dataset length: {len(dataset)}")
                print(f"Dataset input shape: {dataset.input_shape}")
                
                # Get patches from various locations, including edges
                # For large datasets, get patches from beginning, middle, and end
                indices_to_test = []
                if len(dataset) <= 5:
                    indices_to_test = list(range(len(dataset)))
                else:
                    # Get beginning, middle, and end indices
                    indices_to_test = [0, len(dataset)//4, len(dataset)//2, 3*len(dataset)//4, len(dataset)-1]
                
                for i in indices_to_test:
                    start_time = time.time()
                    patch_data = dataset[i]
                    patch_time = time.time() - start_time
                    
                    # Get position and data
                    position = patch_data["pos"]
                    data = patch_data["data"]
                    
                    # Convert to numpy for detailed checking (even if it's already a tensor)
                    if torch.is_tensor(data):
                        np_data = data.detach().cpu().numpy()
                    else:
                        np_data = data
                    
                    # Check for NaN or infinity values
                    has_nan = torch.isnan(data).any().item() if torch.is_tensor(data) else np.isnan(np_data).any()
                    has_inf = torch.isinf(data).any().item() if torch.is_tensor(data) else np.isinf(np_data).any()
                    
                    # Print detailed info
                    print(f"Patch {i} (index {indices_to_test.index(i)+1}/{len(indices_to_test)}):")
                    print(f"  - Position: {position}")
                    print(f"  - Extraction time: {patch_time:.2f} seconds")
                    print(f"  - Shape: {data.shape}")
                    print(f"  - Dtype: {data.dtype}")
                    
                    # Check if we have NaN or inf values
                    if has_nan or has_inf:
                        print(f"  ❌ ERROR: Contains {'NaN' if has_nan else ''} {'and ' if has_nan and has_inf else ''}{'infinity' if has_inf else ''} values!")
                        all_tests_passed = False
                        
                        # Count NaN values if present
                        if has_nan:
                            nan_count = torch.isnan(data).sum().item() if torch.is_tensor(data) else np.isnan(np_data).sum()
                            print(f"     - NaN count: {nan_count}/{np.prod(data.shape)}")
                    else:
                        # Get min/max values
                        if torch.is_tensor(data):
                            min_val = torch.min(data).item()
                            max_val = torch.max(data).item()
                        else:
                            min_val = np.min(np_data)
                            max_val = np.max(np_data)
                        
                        print(f"  ✅ No NaN or infinity values")
                        print(f"  - Value range: {min_val:.4f} to {max_val:.4f}")
                        
                        # Extra validation for normalization schemes
                        if norm_scheme == "minmax" and normalize:
                            # For minmax, values should be in [0, 1]
                            if min_val < 0 or max_val > 1:
                                print(f"  ❌ WARNING: MinMax normalization produced values outside [0,1] range")
                                all_tests_passed = False

                    # Now test TTA operations if this is a large patch (similar to inference)
                    if patch_size[0] >= 128 and i == 0:
                        print(f"\n  Testing TTA with {patch_config['name']}:")
                        
                        # Create a DataLoader like in inference.py
                        batch_size = 2  # Small batch size for testing
                        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
                        print(f"  Created DataLoader with batch_size={batch_size}")
                        
                        # Get a batch from the dataloader
                        batch = next(iter(dataloader))
                        
                        # Mock model_info for TTA
                        model_info = {
                            'allowed_mirroring_axes': [0, 1, 2]  # Standard axes for 3D data
                        }
                        
                        # Use batch data which has batch dimension already
                        if torch.is_tensor(batch['data']):
                            test_data = batch['data'].clone().detach()
                            if not test_data.is_cuda:
                                test_data = test_data.to(device='cpu')
                        else:
                            test_data = torch.from_numpy(batch['data']).clone()
                            
                        print(f"  Batch tensor shape: {test_data.shape}")
                        
                        # Try different TTA configurations
                        tta_configs = [
                            {'name': 'Mirror TTA', 'use_mirroring': True, 'use_rotation_tta': False, 'max_tta': 3},
                            {'name': 'Rotation TTA', 'use_mirroring': False, 'use_rotation_tta': True, 'max_tta': 3},
                            {'name': 'Both Mirror & Rotation', 'use_mirroring': True, 'use_rotation_tta': True, 'max_tta': 6}
                        ]
                        
                        for tta_config in tta_configs:
                            print(f"    Testing {tta_config['name']}:")
                            
                            try:
                                # Generate TTA inputs
                                augmented_inputs, transform_info = get_tta_augmented_inputs(
                                    input_tensor=test_data,
                                    model_info=model_info,
                                    max_tta_combinations=tta_config['max_tta'],
                                    use_rotation_tta=tta_config['use_rotation_tta'],
                                    use_mirroring=tta_config['use_mirroring'],
                                    verbose=True,
                                    rank=0
                                )
                                
                                print(f"      Generated {len(augmented_inputs)} TTA variants")
                                
                                # Check each augmented input for NaNs
                                for idx, aug_input in enumerate(augmented_inputs):
                                    has_nan = torch.isnan(aug_input).any().item()
                                    has_inf = torch.isinf(aug_input).any().item()
                                    
                                    if has_nan or has_inf:
                                        print(f"      ❌ ERROR in TTA variant {idx}: Contains {'NaN' if has_nan else ''} {'and ' if has_nan and has_inf else ''}{'infinity' if has_inf else ''} values!")
                                        all_tests_passed = False
                                        
                                        # Count NaN values if present
                                        if has_nan:
                                            nan_count = torch.isnan(aug_input).sum().item()
                                            print(f"         - NaN count: {nan_count}/{np.prod(aug_input.shape)}")
                                    else:
                                        # Get min/max values
                                        min_val = torch.min(aug_input).item()
                                        max_val = torch.max(aug_input).item()
                                        print(f"      ✅ TTA variant {idx}: No NaN/infinity values, range: {min_val:.4f} to {max_val:.4f}")
                                
                                # Now simulate TTA merge (as done in inference)
                                # First, create mock outputs as if from a model
                                mock_outputs = []
                                for idx, aug_input in enumerate(augmented_inputs):
                                    # Simulate model output (just use the input for testing)
                                    mock_output = aug_input.clone()
                                    # Add the outputs and transform info to the list
                                    mock_outputs.append((mock_output, transform_info[idx]))
                                
                                # Merge the outputs
                                merged_output = merge_tensors(mock_outputs)
                                
                                # Check merged output for NaNs
                                has_nan = torch.isnan(merged_output).any().item()
                                has_inf = torch.isinf(merged_output).any().item()
                                
                                if has_nan or has_inf:
                                    print(f"      ❌ ERROR in merged output: Contains {'NaN' if has_nan else ''} {'and ' if has_nan and has_inf else ''}{'infinity' if has_inf else ''} values!")
                                    all_tests_passed = False
                                    
                                    # Count NaN values if present
                                    if has_nan:
                                        nan_count = torch.isnan(merged_output).sum().item()
                                        print(f"         - NaN count: {nan_count}/{np.prod(merged_output.shape)}")
                                else:
                                    # Get min/max values
                                    min_val = torch.min(merged_output).item()
                                    max_val = torch.max(merged_output).item()
                                    print(f"      ✅ Merged output: No NaN/infinity values, range: {min_val:.4f} to {max_val:.4f}")
                                
                            except Exception as e:
                                print(f"      ❌ ERROR in TTA testing: {e}")
                                import traceback
                                traceback.print_exc()
                                all_tests_passed = False
                
            except Exception as e:
                print(f"Error testing dataset with {patch_config['name']} and {norm_scheme} normalization: {e}")
                import traceback
                traceback.print_exc()
                all_tests_passed = False
    
    return all_tests_passed

if __name__ == "__main__":
    success = test_dataset()
    sys.exit(0 if success else 1)