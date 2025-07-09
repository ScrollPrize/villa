import os
import json
import numpy as np
import napari
import zarr
import tifffile
from magicgui import magicgui
from skimage.filters import threshold_otsu
from skimage.morphology import binary_erosion, binary_dilation, binary_opening, binary_closing, disk, ball
from skimage.measure import label, regionprops

# Try to import config defaults; if not found, use empty defaults.
try:
    from .config import config
except ImportError:
    config = {}

state = {
    # Volumes (using highest resolution only)
    'image_volume': None,
    'label_volume': None,
    'patch_size': None,  # Patch size in actual volume coordinates
    'patch_coords': None,  # List of patch coordinates (tuples)
    'current_index': 0,  # Next patch index to consider
    'dataset_out_path': None,
    'images_out_dir': None,
    'labels_out_dir': None,
    'current_patch': None,  # Info about current patch
    'save_progress': False,  # Whether to save progress to a file
    'progress_file': "",  # Path to the progress file
    'min_label_percentage': 0,  # Minimum percentage (0-100) required in a patch
    # New progress log: a list of dicts recording every patch processed.
    'progress_log': [],  # Each entry: { "index": int, "coords": tuple, "percentage": float, "status": str }
    'output_label_zarr': None,  # Path to output zarr for labels
    'output_zarr_array': None,  # Zarr array object for appending labels
    'output_zarr_index': 0,  # Current index in the output zarr
    # Threshold management
    'original_label_patch': None,  # Store the original label patch before thresholding
    'threshold_value': 115,  # Default threshold value
    'use_otsu': False,  # Whether to use Otsu thresholding
    'label_min': 0,  # Min value in current label patch
    'label_max': 255,  # Max value in current label patch
    'min_component_size': 100  # Minimum component size for filtering
}


def generate_patch_coords(vol_shape, patch_size, sampling, min_z=0):
    """
    Generate a list of top-left (or front-top-left) coordinates for patches.
    Works for 2D (shape (H, W)) or 3D (shape (D, H, W)) volumes.

    For 3D volumes, only patches starting at a z-index >= min_z will be included.
    """
    coords = []
    if len(vol_shape) == 2:
        H, W = vol_shape
        for i in range(0, H - patch_size + 1, patch_size):
            for j in range(0, W - patch_size + 1, patch_size):
                coords.append((i, j))
    elif len(vol_shape) >= 3:
        # Assume the first three dimensions are spatial.
        D, H, W = vol_shape[:3]
        for z in range(min_z, D - patch_size + 1, patch_size):
            for y in range(0, H - patch_size + 1, patch_size):
                for x in range(0, W - patch_size + 1, patch_size):
                    coords.append((z, y, x))
    else:
        raise ValueError("Volume must be at least 2D")
    if sampling == 'random':
        np.random.shuffle(coords)
    return coords


def find_closest_coord_index(old_coord, coords):
    """
    Given an old coordinate (tuple) and a list of new coordinates, find
    the index in the new coordinate list that is closest (using Euclidean distance)
    to the old coordinate.
    """
    distances = [np.linalg.norm(np.array(coord) - np.array(old_coord)) for coord in coords]
    return int(np.argmin(distances))


def extract_patch(volume, coord, patch_size):
    """
    Extract a patch from a volume starting at the given coordinate.
    For spatial dimensions, a slice is created from coord to coord+patch_size.
    Any extra dimensions (e.g. channels) are included in full.
    """
    slices = tuple(slice(c, c + patch_size) for c in coord)
    if volume.ndim > len(coord):
        slices = slices + (slice(None),) * (volume.ndim - len(coord))
    
    try:
        return volume[slices]
    except ValueError as e:
        if "cannot reshape array of size 0" in str(e):
            # Handle empty/all-zero chunks
            # Calculate the shape of the patch
            shape = []
            for i, s in enumerate(slices):
                if s.start is not None and s.stop is not None:
                    shape.append(s.stop - s.start)
                else:
                    # For full slices, use the volume dimension
                    shape.append(volume.shape[i])
            return np.zeros(shape, dtype=volume.dtype)
        else:
            raise


def filter_small_components(binary_patch, min_size=30):
    """
    Filter out small connected components from a binary patch.
    Works layer by layer for 3D data.
    
    Args:
        binary_patch: Binary image (0 or 1)
        min_size: Minimum component size to keep
    
    Returns:
        Filtered binary patch
    """
    if binary_patch is None or min_size <= 0:
        return binary_patch
    
    result = np.zeros_like(binary_patch, dtype=np.uint8)
    
    if binary_patch.ndim == 2:
        # 2D case: process the single layer
        labeled = label(binary_patch)
        for region in regionprops(labeled):
            if region.area >= min_size:
                result[labeled == region.label] = 1
    elif binary_patch.ndim == 3:
        # 3D case: process layer by layer
        for z in range(binary_patch.shape[0]):
            labeled = label(binary_patch[z])
            for region in regionprops(labeled):
                if region.area >= min_size:
                    result[z][labeled == region.label] = 1
    
    return result


def apply_threshold(original_patch, threshold_value=None, use_otsu=False, min_component_size=0):
    """
    Apply threshold to the original label patch and optionally filter small components.
    
    Args:
        original_patch: The original label patch with raw values
        threshold_value: Manual threshold value (ignored if use_otsu is True)
        use_otsu: Whether to use Otsu thresholding
        min_component_size: Minimum component size to keep (0 = no filtering)
    
    Returns:
        Binary patch (uint8) with values 0 or 1
    """
    if original_patch is None:
        return None
    
    if use_otsu:
        # Calculate Otsu threshold
        try:
            threshold = threshold_otsu(original_patch)
        except ValueError:
            # If all values are the same, Otsu fails
            threshold = threshold_value if threshold_value is not None else 1
    else:
        threshold = threshold_value if threshold_value is not None else 1
    
    # Apply threshold
    binary_patch = (original_patch >= threshold).astype(np.uint8)
    
    # Apply morphological opening with kernel size 1 right after threshold
    binary_patch = apply_morphology(binary_patch, 'opening', 1, kernel_size=1)
    
    # Apply component filtering if requested
    if min_component_size > 0:
        binary_patch = filter_small_components(binary_patch, min_component_size)
    
    return binary_patch


def update_label_layer():
    """
    Update the label layer in napari based on current threshold settings.
    This is called when the threshold slider or Otsu checkbox changes.
    """
    global state, viewer
    
    if state.get('original_label_patch') is None:
        return
    
    # Apply threshold to get binary patch with component filtering
    binary_patch = apply_threshold(
        state['original_label_patch'],
        state.get('threshold_value', 1),
        state.get('use_otsu', False),
        state.get('min_component_size', 0)
    )
    
    # Update the napari layer
    if "patch_label" in viewer.layers:
        viewer.layers["patch_label"].data = binary_patch
    else:
        viewer.add_labels(binary_patch, name="patch_label")


def apply_morphology(binary_patch, operation, iterations, kernel_size=1):
    """
    Apply morphological operations to a binary patch.
    
    Args:
        binary_patch: Binary image (0 or 1)
        operation: Type of operation ('erosion', 'dilation', 'opening', 'closing')
        iterations: Number of iterations to apply
        kernel_size: Size of the structuring element (radius for ball/disk)
    
    Returns:
        Processed binary patch
    """
    if binary_patch is None or operation == 'none':
        return binary_patch
    
    # Determine if 2D or 3D
    is_3d = binary_patch.ndim == 3
    
    # Create structuring element
    if is_3d:
        selem = ball(kernel_size)  # 3D ball with specified radius
    else:
        selem = disk(kernel_size)  # 2D disk with specified radius
    
    # Apply operation
    result = binary_patch.copy()
    for _ in range(iterations):
        if operation == 'erosion':
            result = binary_erosion(result, selem)
        elif operation == 'dilation':
            result = binary_dilation(result, selem)
        elif operation == 'opening':
            result = binary_opening(result, selem)
        elif operation == 'closing':
            result = binary_closing(result, selem)
    
    return result.astype(np.uint8)


# Create the threshold control widget
threshold_control = None

@magicgui(
    threshold_value={"widget_type": "SpinBox", "min": 0, "max": 255, "value": 115},
    use_otsu={"widget_type": "CheckBox", "value": False},
    min_component_size={"widget_type": "SpinBox", "min": 0, "max": 1000, "value": 100},
    auto_call=True
)
def threshold_control(threshold_value: int = 115, use_otsu: bool = False, min_component_size: int = 30):
    """
    Control threshold for binarizing labels and filter small components.
    
    Args:
        threshold_value: Manual threshold value (0-255)
        use_otsu: Use Otsu's method for automatic thresholding
        min_component_size: Minimum component size to keep (0 = no filtering)
    """
    global state
    
    # Update state
    state['threshold_value'] = threshold_value
    state['use_otsu'] = use_otsu
    state['min_component_size'] = min_component_size
    
    # Update the label layer
    update_label_layer()
    
    # Update slider enable/disable based on Otsu checkbox
    threshold_control.threshold_value.enabled = not use_otsu
    
    # If using Otsu, calculate and display the threshold value
    if use_otsu and state.get('original_label_patch') is not None:
        try:
            otsu_val = threshold_otsu(state['original_label_patch'])
            print(f"Otsu threshold: {otsu_val:.2f}")
        except ValueError:
            print("Otsu threshold calculation failed (uniform image)")


# Create the morphology control widget
morphology_control = None

@magicgui(
    operation={"choices": ["none", "erosion", "dilation", "opening", "closing"], "value": "none"},
    iterations={"widget_type": "SpinBox", "min": 1, "max": 10, "value": 1},
    kernel_size={"widget_type": "SpinBox", "min": 1, "max": 10, "value": 1},
    call_button="Apply",
    auto_call=False
)
def morphology_control(operation: str = "none", iterations: int = 1, kernel_size: int = 1):
    """
    Apply morphological operations to the current label layer in napari.
    
    Args:
        operation: Type of morphological operation
        iterations: Number of iterations to apply
        kernel_size: Size of the structuring element (radius for ball/disk)
    """
    global viewer
    
    # Check if we have a label layer
    if "patch_label" not in viewer.layers:
        print("No label layer found in napari.")
        return
    
    if operation == "none":
        print("No operation selected.")
        return
    
    # Get current label data from napari
    current_label = viewer.layers["patch_label"].data
    
    # Apply morphological operation
    processed_label = apply_morphology(current_label, operation, iterations, kernel_size)
    
    # Update the napari layer with the processed result
    viewer.layers["patch_label"].data = processed_label
    
    print(f"Applied {operation} with {iterations} iteration(s) and kernel size {kernel_size}.")


def update_progress():
    """
    Write the progress log to a JSON file if progress saving is enabled.
    Saves in a format that zarr_dataset can consume.
    """
    if state.get('save_progress') and state.get('progress_file'):
        try:
            # Filter for approved patches
            approved_patches = [entry for entry in state['progress_log'] if entry['status'] == 'approved']
            
            # Create export data structure compatible with zarr_dataset
            export_data = {
                "metadata": {
                    "patch_size": state.get('patch_size'),
                    "image_zarr": init_volume.image_zarr.value if hasattr(init_volume, 'image_zarr') else '',
                    "label_zarr": init_volume.label_zarr.value if hasattr(init_volume, 'label_zarr') else '',
                    "volume_shape": list(state['image_volume'].shape) if state.get('image_volume') is not None else None,
                    "coordinate_system": "highest_resolution",
                    "export_timestamp": __import__('datetime').datetime.now().isoformat(),
                    "total_approved": len(approved_patches)
                },
                "approved_patches": [],
                "progress_log": state['progress_log']  # Keep original progress log for compatibility
            }
            
            # Add approved patch information in zarr_dataset format
            for entry in approved_patches:
                patch_info = {
                    "volume_index": 0,  # Single volume for now
                    "coords": list(entry['coords']),
                    "percentage": entry['percentage'],
                    "index": entry['index']
                }
                export_data["approved_patches"].append(patch_info)
            
            with open(state['progress_file'], "w") as f:
                json.dump(export_data, f, indent=2)
            print(f"Progress saved to {state['progress_file']} with {len(approved_patches)} approved patches.")
        except Exception as e:
            print("Error saving progress:", e)


def load_progress():
    """
    Load the progress log from a JSON file if progress saving is enabled.
    The current_index is set to the number of entries already processed.
    """
    if state.get('save_progress') and state.get('progress_file'):
        if os.path.exists(state['progress_file']):
            try:
                with open(state['progress_file'], "r") as f:
                    data = json.load(f)
                state['progress_log'] = data.get("progress_log", [])
                # This value will be overridden later if a new patch grid is computed.
                state['current_index'] = len(state['progress_log'])
                print(f"Loaded progress file with {len(state['progress_log'])} entries.")
            except Exception as e:
                print("Error loading progress:", e)


def load_next_patch():
    """
    Load the next valid patch from the volumes and show it in napari.
    A patch is only shown if its label patch has a nonzero percentage
    greater than or equal to the user-specified threshold.

    For each patch encountered:
      - If the patch does not meet the threshold, a log entry is recorded with status "auto-skipped".
      - If the patch meets the threshold, a log entry with status "pending" is recorded,
        the patch is displayed, and the function returns.
    """
    global state, viewer
    if state.get('patch_coords') is None:
        print("Volume not initialized.")
        return

    patch_size = state['patch_size']
    image_volume = state['image_volume']
    label_volume = state['label_volume']
    coords = state['patch_coords']
    min_label_percentage = state.get('min_label_percentage', 0)

    while state['current_index'] < len(coords):
        idx = state['current_index']
        coord = coords[idx]
        print(f"Loading patch {idx} at coordinate {coord}...")
        print("  Extracting image patch...")
        image_patch = extract_patch(image_volume, coord, patch_size)
        print("  Extracting label patch...")
        label_patch = extract_patch(label_volume, coord, patch_size)
        print("  Patches extracted successfully.")
        
        # Check if patches are empty (all zeros) due to empty zarr chunks
        if np.all(image_patch == 0):
            print("  Warning: Image patch is all zeros (empty zarr chunk)")
        if np.all(label_patch == 0):
            print("  Warning: Label patch is all zeros (empty zarr chunk)")
        state['current_index'] += 1

        # Calculate the percentage of labeled (nonzero) pixels.
        nonzero = np.count_nonzero(label_patch)
        total = label_patch.size
        percentage = (nonzero / total * 100) if total > 0 else 0

        if percentage >= min_label_percentage:
            # Store the original label patch
            state['original_label_patch'] = label_patch.copy()
            
            # Update min/max for threshold slider
            if label_patch.size > 0:
                state['label_min'] = int(np.min(label_patch))
                state['label_max'] = int(np.max(label_patch))
                # Update threshold slider range
                if 'threshold_control' in globals() and threshold_control is not None:
                    threshold_control.threshold_value.min = state['label_min']
                    threshold_control.threshold_value.max = state['label_max']
            
            # Apply threshold using current settings with component filtering
            binary_patch = apply_threshold(
                label_patch,
                state.get('threshold_value', 1),
                state.get('use_otsu', False),
                state.get('min_component_size', 0)
            )
            
            # Record this patch as pending (waiting for the user decision)
            entry = {"index": idx, "coords": coord, "percentage": percentage, "status": "pending"}
            state['progress_log'].append(entry)
            state['current_patch'] = {
                'coords': coord,
                'image': image_patch,
                'label': binary_patch,
                'index': idx
            }
            # Update or add napari layers.
            if "patch_image" in viewer.layers:
                viewer.layers["patch_image"].data = image_patch
            else:
                viewer.add_image(image_patch, name="patch_image", colormap='gray')
            if "patch_label" in viewer.layers:
                viewer.layers["patch_label"].data = binary_patch
            else:
                viewer.add_labels(binary_patch, name="patch_label")
            print(f"Loaded patch at {coord} with {percentage:.2f}% labeled (threshold: {min_label_percentage}%).")
            print(f"Label range: [{state['label_min']}, {state['label_max']}]")
            return
        else:
            # Record an auto-skipped patch
            entry = {"index": idx, "coords": coord, "percentage": percentage, "status": "auto-skipped"}
            state['progress_log'].append(entry)
            print(f"Skipping patch at {coord} ({percentage:.2f}% labeled, below threshold of {min_label_percentage}%).")
    print("No more patches available.")


def save_current_patch():
    """
    Save the current patch extracted from the volumes.
    File names are constructed using the zyx (or yx) coordinates:
      - Image file gets a '_0000' suffix (e.g. img_z{z}_y{y}_x{x}_0000.tif).
      - Label file does not (e.g. lbl_z{z}_y{y}_x{x}.tif).
    Also saves the edited label to the output zarr if configured.
    """
    global state, viewer
    if state.get('current_patch') is None:
        print("No patch available to save.")
        return

    patch = state['current_patch']
    coord = patch['coords']
    patch_size = state['patch_size']

    # Extract image patch from the volume
    image_patch = extract_patch(state['image_volume'], coord, patch_size)
    
    # Get the edited label from napari viewer
    if "patch_label" in viewer.layers:
        label_patch = viewer.layers["patch_label"].data
        # Ensure it's binarized
        label_patch = (label_patch > 0).astype(np.uint8)
    else:
        print("Warning: No label layer found in napari, using original label.")
        label_patch = extract_patch(state['label_volume'], coord, patch_size)
        label_patch = (label_patch > 0).astype(np.uint8)

    # Construct coordinate string.
    if len(coord) == 3:
        coord_str = f"z{coord[0]}_y{coord[1]}_x{coord[2]}"
    elif len(coord) == 2:
        coord_str = f"y{coord[0]}_x{coord[1]}"
    else:
        coord_str = "_".join(str(c) for c in coord)

    image_filename = f"{coord_str}_0000.tif"
    label_filename = f"{coord_str}.tif"
    image_path = os.path.join(state['images_out_dir'], image_filename)
    label_path = os.path.join(state['labels_out_dir'], label_filename)

    # Save tif files
    tifffile.imwrite(image_path, image_patch)
    tifffile.imwrite(label_path, label_patch)
    print(f"Saved image patch to {image_path} and label patch to {label_path}")
    
    # Save to output zarr if configured
    if state.get('output_zarr_array') is not None:
        try:
            # Resize the zarr array to accommodate the new patch
            new_shape = (state['output_zarr_index'] + 1,) + state['output_zarr_array'].shape[1:]
            state['output_zarr_array'].resize(new_shape)
            
            # Write the label patch to the zarr
            state['output_zarr_array'][state['output_zarr_index']] = label_patch
            state['output_zarr_index'] += 1
            
            print(f"Saved label patch to zarr at index {state['output_zarr_index'] - 1}")
        except Exception as e:
            print(f"Error saving to output zarr: {e}")


@magicgui(
    sampling={"choices": ["random", "sequence"]},
    min_label_percentage={"min": 0, "max": 100},
    min_z={"widget_type": "SpinBox", "min": 0, "max": 999999},
)
def init_volume(
        image_zarr: str = config.get("image_zarr", ""),
        label_zarr: str = config.get("label_zarr", ""),
        dataset_out_path: str = config.get("dataset_out_path", ""),
        output_label_zarr: str = config.get("output_label_zarr", ""),
        patch_size: int = config.get("patch_size", 64),
        sampling: str = config.get("sampling", "sequence"),
        save_progress: bool = config.get("save_progress", False),
        progress_file: str = config.get("progress_file", ""),
        min_label_percentage: int = config.get("min_label_percentage", 0),
        min_z: int = 0,  # minimum z index from which to start patching (only for 3D)
):
    """
    Load image and label volumes from Zarr using highest resolution only.
    """
    global state, viewer


    # Try to open as array first (for zarrs with array at root)
    image_volume = zarr.open(image_zarr, mode='r')
    image_volume = image_volume['0'] if isinstance(image_volume, zarr.hierarchy.Group) else image_volume
    print("Loaded image zarr as array.")

    # Try to open as array first (for zarrs with array at root)
    label_volume = zarr.open(label_zarr, mode='r')
    label_volume = label_volume['0'] if isinstance(label_volume, zarr.hierarchy.Group) else label_volume
    print("Loaded label zarr as array.")
        

    # Save the loaded volumes.
    state['image_volume'] = image_volume
    state['label_volume'] = label_volume
    state['dataset_out_path'] = dataset_out_path
    state['patch_size'] = patch_size

    # Save progress options.
    state['save_progress'] = save_progress
    state['progress_file'] = progress_file

    # Save the minimum label percentage.
    state['min_label_percentage'] = min_label_percentage
    
    # Save output zarr path
    state['output_label_zarr'] = output_label_zarr

    # Create output directories.
    images_dir = os.path.join(dataset_out_path, 'imagesTr')
    labels_dir = os.path.join(dataset_out_path, 'labelsTr')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    state['images_out_dir'] = images_dir
    state['labels_out_dir'] = labels_dir

    # Compute patch coordinates on the volume.
    num_spatial = 2 if len(image_volume.shape) == 2 else 3
    
    # Initialize output zarr if path is provided
    if output_label_zarr:
        try:
            # Determine patch shape based on dimensionality
            if num_spatial == 2:
                patch_shape = (patch_size, patch_size)
            else:
                patch_shape = (patch_size, patch_size, patch_size)
            
            # Check if zarr already exists
            if os.path.exists(output_label_zarr):
                # Open existing zarr
                state['output_zarr_array'] = zarr.open(output_label_zarr, mode='r+')
                state['output_zarr_index'] = state['output_zarr_array'].shape[0]
                print(f"Opened existing output zarr with {state['output_zarr_index']} patches.")
            else:
                # Create new zarr with appropriate shape
                # First dimension is for patches (will grow), rest are spatial dimensions
                initial_shape = (0,) + patch_shape
                chunks = (1,) + patch_shape  # One patch per chunk
                
                state['output_zarr_array'] = zarr.open(
                    output_label_zarr,
                    mode='w',
                    shape=initial_shape,
                    chunks=chunks,
                    dtype='uint8',
                    compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=zarr.Blosc.BITSHUFFLE)
                )
                state['output_zarr_index'] = 0
                print(f"Created new output zarr at {output_label_zarr}")
        except Exception as e:
            print(f"Error initializing output zarr: {e}")
            state['output_zarr_array'] = None
    vol_shape = image_volume.shape[:num_spatial]
    new_patch_coords = generate_patch_coords(vol_shape, patch_size, sampling, min_z=min_z)
    state['patch_coords'] = new_patch_coords

    # Attempt to load prior progress.
    load_progress()

    # If a progress log exists, adjust the starting patch index based on the last processed coordinate.
    if state['progress_log']:
        old_coord = state['progress_log'][-1]['coords']
        # Option 1: "Snap" the old coordinate to the new grid.
        new_start_coord = tuple((c // patch_size) * patch_size for c in old_coord)
        if new_start_coord in new_patch_coords:
            new_index = new_patch_coords.index(new_start_coord)
        else:
            # Option 2: Use nearest neighbor.
            new_index = find_closest_coord_index(old_coord, new_patch_coords)
        state['current_index'] = new_index
        print(f"Resuming from new patch index {new_index} (closest to old coordinate {old_coord}).")
    else:
        state['current_index'] = 0

    print(f"Loaded volumes with shape {vol_shape}.")
    print(f"Found {len(state['patch_coords'])} patch positions using '{sampling}' sampling.")
    load_next_patch()


@magicgui(call_button="next pair")
def iter_pair(approved: bool):
    """
    When "next pair" is pressed (or spacebar used), this function:
      - Updates the current (pending) patch’s record to "approved" (if checked) or "skipped".
      - If approved, saves the high-res patch.
      - Then loads the next patch.
      - Updates the progress file.
      - Resets the approved checkbox.
    """
    # Update the minimum label percentage from the current value in the init_volume widget.
    # This assumes that the init_volume widget is still available as a global variable.
    state['min_label_percentage'] = init_volume.min_label_percentage.value

    # Update the pending entry from load_next_patch.
    if state['progress_log'] and state['progress_log'][-1]['status'] == "pending":
        if approved:
            state['progress_log'][-1]['status'] = "approved"
            save_current_patch()
        else:
            state['progress_log'][-1]['status'] = "skipped"
    load_next_patch()
    update_progress()
    iter_pair.approved.value = False


@magicgui(call_button="previous pair")
def prev_pair():
    """
    When "previous pair" is pressed, go back to the last patch that was shown (i.e. one that wasn't auto-skipped).
    This is done by removing the most recent record (ignoring auto-skipped ones) from the progress log
    and resetting the current index. The patch is then reloaded into the viewer.
    """
    global state, viewer
    if not state['progress_log']:
        print("No previous patch available.")
        return

    # Remove any trailing auto-skipped entries.
    while state['progress_log'] and state['progress_log'][-1]['status'] == "auto-skipped":
        state['progress_log'].pop()

    if not state['progress_log']:
        print("No previous patch available.")
        return

    # Pop the last processed patch (could be approved, skipped, or pending).
    entry = state['progress_log'].pop()
    state['current_index'] = entry['index']  # Rewind the current_index.
    coord = entry['coords']
    patch_size = state['patch_size']
    image_patch = extract_patch(state['image_volume'], coord, patch_size)
    label_patch = extract_patch(state['label_volume'], coord, patch_size)
    
    # Store the original label patch
    state['original_label_patch'] = label_patch.copy()
    
    # Update min/max for threshold slider
    if label_patch.size > 0:
        state['label_min'] = int(np.min(label_patch))
        state['label_max'] = int(np.max(label_patch))
        # Update threshold slider range
        if 'threshold_control' in globals() and threshold_control is not None:
            threshold_control.threshold_value.min = state['label_min']
            threshold_control.threshold_value.max = state['label_max']
    
    # Apply threshold using current settings with component filtering
    binary_patch = apply_threshold(
        label_patch,
        state.get('threshold_value', 1),
        state.get('use_otsu', False),
        state.get('min_component_size', 0)
    )
    
    state['current_patch'] = {"coords": coord, "image": image_patch, "label": binary_patch, "index": entry['index']}

    # Update the viewer with this patch.
    if "patch_image" in viewer.layers:
        viewer.layers["patch_image"].data = image_patch
    else:
        viewer.add_image(image_patch, name="patch_image", colormap='gray')
    if "patch_label" in viewer.layers:
        viewer.layers["patch_label"].data = binary_patch
    else:
        viewer.add_labels(binary_patch, name="patch_label")
    update_progress()
    print(f"Reverted to patch at {coord}.")
    print(f"Label range: [{state['label_min']}, {state['label_max']}]")


# Create the reset label button
@magicgui(call_button="Reset Label")
def reset_label():
    """
    Reset the label layer to the original patch with current threshold settings.
    This discards all manual edits and morphological operations.
    """
    global state, viewer
    
    if state.get('original_label_patch') is None:
        print("No original label patch available.")
        return
    
    # Apply threshold to get binary patch from original with component filtering
    binary_patch = apply_threshold(
        state['original_label_patch'],
        state.get('threshold_value', 1),
        state.get('use_otsu', False),
        state.get('min_component_size', 0)
    )
    
    # Update the napari layer
    if "patch_label" in viewer.layers:
        viewer.layers["patch_label"].data = binary_patch
        print("Label reset to original with current threshold settings.")
    else:
        print("No label layer found in napari.")


# Create the jump control widget
@magicgui(
    z_jump={"widget_type": "SpinBox", "min": 50, "max": 10000, "step": 50, "value": 500},
    call_button="Jump"
)
def jump_control(z_jump: int = 500):
    """
    Jump up by the specified number of z layers and begin processing from there.
    
    Args:
        z_jump: Number of z layers to jump up
    """
    global state
    
    if state.get('patch_coords') is None or state.get('current_patch') is None:
        print("No volume initialized or current patch available.")
        return
    
    current_patch = state['current_patch']
    current_coord = current_patch['coords']
    
    # Only works for 3D volumes
    if len(current_coord) != 3:
        print("Jump function only works with 3D volumes.")
        return
    
    # Calculate new z coordinate
    current_z, current_y, current_x = current_coord
    new_z = current_z + z_jump
    
    # Check if new z is within volume bounds
    vol_shape = state['image_volume'].shape
    if new_z >= vol_shape[0]:
        print(f"Cannot jump to z={new_z}, volume only has {vol_shape[0]} layers.")
        return
    
    # Find the closest patch coordinate at the new z level
    patch_size = state['patch_size']
    target_coord = (new_z, current_y, current_x)
    
    # Snap to patch grid
    snapped_coord = tuple((c // patch_size) * patch_size for c in target_coord)
    
    # Find this coordinate in our patch list
    if snapped_coord in state['patch_coords']:
        new_index = state['patch_coords'].index(snapped_coord)
        state['current_index'] = new_index
        print(f"Jumped from z={current_z} to z={new_z} (snapped to {snapped_coord})")
        load_next_patch()
        update_progress()
    else:
        # Find the closest available coordinate
        new_index = find_closest_coord_index(snapped_coord, state['patch_coords'])
        state['current_index'] = new_index
        actual_coord = state['patch_coords'][new_index]
        print(f"Jumped from z={current_z} to closest available coordinate {actual_coord}")
        load_next_patch()
        update_progress()




def main():
    """Main entry point for the proofreader application."""
    global viewer
    viewer = napari.Viewer()
    viewer.window.add_dock_widget(init_volume, name="Initialize Volumes", area="right")
    viewer.window.add_dock_widget(prev_pair, name="Previous Patch", area="right")
    viewer.window.add_dock_widget(threshold_control, name="Threshold Control", area="right")
    viewer.window.add_dock_widget(morphology_control, name="Morphology Operations", area="right")
    viewer.window.add_dock_widget(reset_label, name="Reset Label", area="right")
    viewer.window.add_dock_widget(jump_control, name="Jump Control", area="right")
    viewer.window.add_dock_widget(iter_pair, name="Iterate Patches", area="right")

    # --- Keybindings ---
    @viewer.bind_key("Space")
    def next_pair_key(event):
        """Call the next pair function when the spacebar is pressed."""
        iter_pair()

    @viewer.bind_key("a")
    def toggle_approved_key(event):
        """Toggle the 'approved' checkbox when the 'a' key is pressed."""
        iter_pair.approved.value = not iter_pair.approved.value

    napari.run()


if __name__ == '__main__':
    main()
