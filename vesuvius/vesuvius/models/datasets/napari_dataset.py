import napari
import napari.layers
from copy import deepcopy
from .base_dataset import BaseDataset

class NapariDataset(BaseDataset):
    """
    A PyTorch Dataset for handling both 2D and 3D data from napari.
    
    This dataset automatically detects if the provided data is 2D or 3D and 
    handles it appropriately throughout the data loading process.
    """
    
    def _initialize_volumes(self):
        """Initialize volumes by extracting data from napari viewer."""
        targets_dict, data_dict = self._get_images_from_napari()
        
        # Update ConfigManager with target information
        self.mgr.set_targets_and_data(targets_dict, data_dict)
        
        # Store the data in the format expected by BaseDataset
        self.target_volumes = data_dict
    
    def _get_images_from_napari(self):
        """
        Extract image/label pairs from napari viewer for the dataset.
        Labels are expected to be named {image_name}_{suffix} where suffix is a unique identifier.
        For each image layer, multiple label layers can be associated.
        
        Returns
        -------
        tuple
            (targets_dict, data_dict) where:
            - targets_dict: Dictionary with target configurations 
            - data_dict: Dictionary with target names as keys and list of volume data as values
        """
        viewer = napari.current_viewer()
        if viewer is None:
            raise ValueError("No active viewer found")
        
        all_layers = list(viewer.layers)
        layer_names = [layer.name for layer in all_layers]
        image_layers = [layer for layer in all_layers if isinstance(layer, napari.layers.Image)]
        
        if not image_layers:
            raise ValueError("No image layers found in the viewer")
        
        targets_dict = {}
        data_dict = {}
        
        for image_layer in image_layers:
            image_name = image_layer.name
            
            if self.mgr.verbose:
                print(f"Processing image layer: {image_name}")
                print(f"Available layers: {layer_names}")
            
            # Find all label layers that correspond to this image layer
            # Pattern: image_name_suffix
            matching_label_layers = []
            
            for layer in all_layers:
                if (isinstance(layer, napari.layers.Labels) and 
                    layer.name.startswith(f"{image_name}_")):
                    suffix = layer.name[len(image_name) + 1:]  # Extract suffix after image_name_
                    
                    # Validate suffix is meaningful
                    if suffix.isdigit():
                        print(f"\nWarning: Label layer '{layer.name}' uses numeric suffix '{suffix}'.")
                        print(f"Consider using a descriptive name like '{image_name}_ink' or '{image_name}_labels' instead.")
                        print(f"The suffix after '{image_name}_' will be used as the target name in the model.\n")
                    
                    if self.mgr.verbose:
                        print(f"Found matching label layer: {layer.name} with suffix: {suffix}")
                    matching_label_layers.append((suffix, layer))
            
            if not matching_label_layers:
                if self.mgr.verbose:
                    print(f"No matching label layers found for image: {image_name}")
                continue
            
            mask_layer = None
            mask_layer_name = f"{image_name}_mask"
            for layer in all_layers:
                if isinstance(layer, napari.layers.Labels) and layer.name == mask_layer_name:
                    mask_layer = layer
                    print(f"Found mask layer for image {image_name}: {mask_layer_name}")
                    break
            
            # For each matching label layer, create a target
            for target_suffix, label_layer in matching_label_layers:
                if target_suffix == "mask":
                    continue

                target_name = target_suffix
                if target_name not in targets_dict:
                    # Don't set out_channels here - let auto-detection handle it
                    targets_dict[target_name] = {
                        "activation": "none"  # Default activation for auto-detected channels
                    }
                    print(f"DEBUG: Creating new target '{target_name}' - channels will be auto-detected")
                    data_dict[target_name] = []
                
                # Prepare the data dictionary
                data_entry = {
                    'data': deepcopy(image_layer.data),
                    'label': deepcopy(label_layer.data)
                }
                
                if mask_layer is not None:
                    data_entry['mask'] = deepcopy(mask_layer.data)
                    print(f"Including mask for target {target_name}")
                
                data_dict[target_name].append({
                    'data': data_entry,
                    'out_channels': targets_dict[target_name].get("out_channels", None),  # Will be auto-detected
                    'name': f"{image_name}_{target_name}"
                })
                
                if self.mgr.verbose:
                    print(f"Added target {target_name} with data from {image_name} and {label_layer.name}")
        
        if not targets_dict:
            raise ValueError("No valid image-label pairs found. Label layers should be named as image_name_suffix.")
        
        if self.mgr.verbose:
            print(f"Final targets dictionary: {targets_dict}")
            print(f"Final data dictionary keys: {list(data_dict.keys())}")
                
        return targets_dict, data_dict
