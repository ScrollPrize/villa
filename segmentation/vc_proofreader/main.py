import os
import json
import glob
import numpy as np
import napari
import zarr
import tifffile
from magicgui import magicgui
from datetime import datetime
import cc3d
from typing import Optional, Tuple


# Try to import config defaults; if not found, use empty defaults.
try:
    from .config import config
except ImportError:
    try:
        from config import config
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
    'target_label_value': None,  # Target value to calculate percentage from (None means all non-zero)
    # New progress log: a list of dicts recording every patch processed.
    'progress_log': [],  # Each entry: { "index": int, "coords": tuple, "percentage": float, "status": str }
    'output_label_zarr': None,  # Path to output zarr for labels
    'output_zarr_array': None,  # Zarr array object for writing labels
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
                    "volume_selection": init_volume.volume_selection.value if hasattr(init_volume, 'volume_selection') else '',
                    "image_zarr": state.get('image_zarr', ''),
                    "label_zarr": state.get('label_zarr', ''),
                    "volume_shape": list(state['image_volume'].shape) if state.get('image_volume') is not None else None,
                    "coordinate_system": "highest_resolution",
                    "export_timestamp": datetime.now().isoformat(),
                    "session_start_timestamp": state.get('session_start_timestamp', ''),
                    "total_approved": len(approved_patches),
                    "dataset_out_path": state.get('dataset_out_path', ''),
                    "images_out_dir": state.get('images_out_dir', ''),
                    "labels_out_dir": state.get('labels_out_dir', '')
                },
                "approved_patches": [],
                "progress_log": state['progress_log']  # Keep original progress log for compatibility
            }
            
            # Add approved patch information in zarr_dataset format
            for entry in approved_patches:
                # Generate filenames based on coordinates
                coord = entry['coords']
                if len(coord) == 3:
                    coord_str = f"z{coord[0]}_y{coord[1]}_x{coord[2]}"
                elif len(coord) == 2:
                    coord_str = f"y{coord[0]}_x{coord[1]}"
                else:
                    coord_str = "_".join(str(c) for c in coord)
                
                patch_info = {
                    "volume_index": 0,  # Single volume for now
                    "coords": list(entry['coords']),
                    "percentage": entry['percentage'],
                    "index": entry['index'],
                    "date_processed": entry.get('date_processed', ''),
                    "image_filename": f"{coord_str}_0000.tif",
                    "label_filename": f"{coord_str}.tif",
                    "patch_size": state.get('patch_size')
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
                
                # Count approved patches
                approved_count = len([e for e in state['progress_log'] if e.get('status') == 'approved'])
                print(f"Loaded progress file: {state['progress_file']}")
                print(f"  Total entries: {len(state['progress_log'])}")
                print(f"  Approved patches: {approved_count}")
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
        print("  Extracting label patch...")
        label_patch = extract_patch(label_volume, coord, patch_size)
        print("  Label patch extracted successfully.")
        
        # Check if label patch is empty (all zeros) due to empty zarr chunks
        if np.all(label_patch == 0):
            print("  Warning: Label patch is all zeros (empty zarr chunk)")
        
        state['current_index'] += 1

        # Calculate the percentage of labeled pixels.
        # If target_label_value is specified, only count that value
        target_value = state.get('target_label_value')
        if target_value is not None:
            nonzero = np.sum(label_patch == target_value)
        else:
            nonzero = np.count_nonzero(label_patch)
        total = label_patch.size
        percentage = (nonzero / total * 100) if total > 0 else 0

        if percentage >= min_label_percentage:
            # Only extract image patch if label percentage meets threshold
            print("  Extracting image patch...")
            image_patch = extract_patch(image_volume, coord, patch_size)
            print("  Image patch extracted successfully.")
            
            # Check if image patch is empty (all zeros) due to empty zarr chunks
            if np.all(image_patch == 0):
                print("  Warning: Image patch is all zeros (empty zarr chunk)")
            
            # Record this patch as pending (waiting for the user decision)
            entry = {
                "index": idx, 
                "coords": coord, 
                "percentage": percentage, 
                "status": "pending",
                "date_processed": datetime.now().isoformat()
            }
            state['progress_log'].append(entry)
            state['current_patch'] = {
                'coords': coord,
                'image': image_patch,
                'label': label_patch,
                'index': idx
            }
            # Update or add napari layers.
            if "patch_image" in viewer.layers:
                viewer.layers["patch_image"].data = image_patch
            else:
                viewer.add_image(image_patch, name="patch_image", colormap='gray')
            if "patch_label" in viewer.layers:
                viewer.layers["patch_label"].data = label_patch
            else:
                viewer.add_labels(label_patch, name="patch_label")
            target_msg = f" (target value: {state.get('target_label_value')})" if state.get('target_label_value') is not None else ""
            print(f"Loaded patch at {coord} with {percentage:.2f}% labeled{target_msg} (threshold: {min_label_percentage}%).")
            return
        else:
            # Record an auto-skipped patch
            entry = {
                "index": idx, 
                "coords": coord, 
                "percentage": percentage, 
                "status": "auto-skipped",
                "date_processed": datetime.now().isoformat()
            }
            state['progress_log'].append(entry)
            target_msg = f" of target value {state.get('target_label_value')}" if state.get('target_label_value') is not None else ""
            print(f"Skipping patch at {coord} ({percentage:.2f}% labeled{target_msg}, below threshold of {min_label_percentage}%).")
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

    # Use the stored image patch instead of re-extracting
    image_patch = patch['image']
    
    # Get the edited label from napari viewer
    if "patch_label" in viewer.layers:
        label_patch = viewer.layers["patch_label"].data
        # Keep original dtype
        label_patch = label_patch.astype(np.uint8)
    else:
        print("Warning: No label layer found in napari, using original label.")
        label_patch = extract_patch(state['label_volume'], coord, patch_size)

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
            # Write the label patch to the correct location in the full volume
            slices = tuple(slice(c, c + patch_size) for c in coord)
            state['output_zarr_array'][slices] = label_patch
            
            print(f"Saved label patch to zarr at coordinates {coord}")
        except Exception as e:
            print(f"Error saving to output zarr: {e}")


@magicgui(
    volume_selection={"choices": list(config.keys()) if config else ["No volumes available"]},
    energy={"widget_type": "LineEdit", "enabled": False},
    resolution={"widget_type": "LineEdit", "enabled": False},
    sampling={"choices": ["random", "sequence"]},
    min_label_percentage={"min": 0, "max": 100},
    target_label_value={"widget_type": "LineEdit", "label": "Target Label Value (optional)"},
    min_z={"widget_type": "SpinBox", "min": 0, "max": 999999},
    call_button="Initialize Volumes"
)
def init_volume(
        volume_selection: str = list(config.keys())[0] if config else "",
        energy: str = "",
        resolution: str = "",
        dataset_out_path: str = config.get("dataset_out_path", ""),
        output_label_zarr: str = config.get("output_label_zarr", ""),
        patch_size: int = config.get("patch_size", 384),
        sampling: str = config.get("sampling", "sequence"),
        save_progress: bool = config.get("save_progress", True),
        progress_file: str = config.get("progress_file", ""),
        min_label_percentage: int = config.get("min_label_percentage", 1),
        target_label_value: str = config.get("target_label_value", ""),
        min_z: int = 2500,  # minimum z index from which to start patching (only for 3D)
):
    """
    Load image and label volumes from Zarr using highest resolution only.
    """
    global state, viewer
    
    # Generate timestamp for use in default paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # If no dataset_out_path provided, check for existing directories
    if not dataset_out_path:
        # Look for existing dataset directories for this volume
        existing_dirs = glob.glob(f"{volume_selection}_*")
        existing_dirs = [d for d in existing_dirs if os.path.isdir(d)]
        
        if existing_dirs:
            # Use the most recent directory
            dataset_out_path = max(existing_dirs, key=os.path.getmtime)
            print(f"Found existing dataset directory: {dataset_out_path}")
        else:
            # No existing directory, create new one
            dataset_out_path = f"{volume_selection}_{timestamp}"
            
        # Update the widget to show the path
        init_volume.dataset_out_path.value = dataset_out_path
    
    # Check if dataset_out_path exists and look for existing progress files
    existing_progress_file = None
    if os.path.exists(dataset_out_path):
        # Look for existing progress files in the directory
        progress_files = glob.glob(os.path.join(dataset_out_path, f"{volume_selection}_*_progress.json"))
        if progress_files:
            # Use the most recent progress file
            existing_progress_file = max(progress_files, key=os.path.getmtime)
            print(f"Found existing progress file: {existing_progress_file}")
    
    # Generate default output_label_zarr if not provided
    if not output_label_zarr:
        # Check if there's an existing zarr in the dataset path
        if os.path.exists(dataset_out_path):
            existing_zarrs = glob.glob(os.path.join(dataset_out_path, f"{volume_selection}_*_labels.zarr"))
            if existing_zarrs:
                output_label_zarr = existing_zarrs[0]  # Use the first found
            else:
                output_label_zarr = os.path.join(dataset_out_path, f"{volume_selection}_{timestamp}_labels.zarr")
        else:
            output_label_zarr = os.path.join(dataset_out_path, f"{volume_selection}_{timestamp}_labels.zarr")
        # Update the widget to show the generated path
        init_volume.output_label_zarr.value = output_label_zarr
    
    # Handle progress file path
    if save_progress:
        if not progress_file:
            # Use existing progress file if found, otherwise generate new
            if existing_progress_file:
                progress_file = existing_progress_file
            else:
                progress_filename = f"{volume_selection}_{timestamp}_progress.json"
                if dataset_out_path:
                    progress_file = os.path.join(dataset_out_path, progress_filename)
                else:
                    progress_file = progress_filename
        else:
            # Check if progress_file is just a filename (no directory path)
            if os.path.dirname(progress_file) == "":
                progress_filename = progress_file
                # If we have a filename without path, place it in dataset_out_path
                if dataset_out_path:
                    progress_file = os.path.join(dataset_out_path, progress_filename)
            # else: Full path provided, use as is
            
        # Update the widget to show the final progress file path
        init_volume.progress_file.value = progress_file

    # Get the selected volume configuration
    if not config or volume_selection not in config:
        print(f"Error: Volume '{volume_selection}' not found in config.")
        return
    
    selected_config = config[volume_selection]
    image_zarr = selected_config['volume_path']
    label_zarr = selected_config['label_path']
    energy = selected_config.get('energy', 'Unknown')
    resolution = selected_config.get('resolution', 'Unknown')
    
    print(f"Selected: {volume_selection}")
    print(f"  Energy: {energy} keV")
    print(f"  Resolution: {resolution} μm")

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
    state['image_zarr'] = image_zarr
    state['label_zarr'] = label_zarr
    state['dataset_out_path'] = dataset_out_path
    state['patch_size'] = patch_size
    # Remember sampling strategy for later grid recomputations
    state['sampling'] = sampling
    state['session_start_timestamp'] = timestamp  # Use the same timestamp from earlier

    # Save progress options.
    state['save_progress'] = save_progress
    state['progress_file'] = progress_file

    # Save the minimum label percentage and target value.
    state['min_label_percentage'] = min_label_percentage
    
    # Parse and save target label value if provided
    if target_label_value.strip():
        try:
            state['target_label_value'] = int(target_label_value)
            print(f"Target label value set to: {state['target_label_value']}")
        except ValueError:
            print(f"Warning: Invalid target label value '{target_label_value}'. Using all non-zero values.")
            state['target_label_value'] = None
    else:
        state['target_label_value'] = None

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
            # Get the shape of the input volume
            vol_shape = image_volume.shape[:num_spatial]
            
            # Check if zarr already exists
            if os.path.exists(output_label_zarr):
                # Open existing zarr
                state['output_zarr_array'] = zarr.open(output_label_zarr, mode='r+', write_empty_chunks=False)
                print(f"Opened existing output zarr at {output_label_zarr}")
                print(f"  Shape: {state['output_zarr_array'].shape}")
                print(f"  Chunks: {state['output_zarr_array'].chunks}")
            else:
                if num_spatial == 2:
                    chunks = (128,128)
                else:
                    chunks = (128,128,128)
                
                state['output_zarr_array'] = zarr.open(
                    output_label_zarr,
                    mode='w',
                    shape=vol_shape,
                    chunks=chunks,
                    dtype='uint8',
                    compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=zarr.Blosc.BITSHUFFLE),
                    write_empty_chunks=False
                )
                print(f"Created new output zarr at {output_label_zarr}")
                print(f"  Shape: {vol_shape}")
                print(f"  Chunks: {chunks}")
                print(f"  Compression: zstd with bitshuffle")
                print(f"  write_empty_chunks: False")
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
    if state['current_index'] > 0:
        print(f"Starting from patch index {state['current_index']} (resuming from previous session)")
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
    # Update the minimum label percentage and target value from the current values in the init_volume widget.
    # This assumes that the init_volume widget is still available as a global variable.
    state['min_label_percentage'] = init_volume.min_label_percentage.value

    # Update target label value if changed
    target_value_str = init_volume.target_label_value.value.strip()
    if target_value_str:
        try:
            state['target_label_value'] = int(target_value_str)
        except ValueError:
            state['target_label_value'] = None
    else:
        state['target_label_value'] = None

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
    # Extract label patch first (it's lighter and we already know it meets the threshold)
    label_patch = extract_patch(state['label_volume'], coord, patch_size)
    # Now extract image patch
    image_patch = extract_patch(state['image_volume'], coord, patch_size)
    
    state['current_patch'] = {"coords": coord, "image": image_patch, "label": label_patch, "index": entry['index']}

    # Update the viewer with this patch.
    if "patch_image" in viewer.layers:
        viewer.layers["patch_image"].data = image_patch
    else:
        viewer.add_image(image_patch, name="patch_image", colormap='gray')
    if "patch_label" in viewer.layers:
        viewer.layers["patch_label"].data = label_patch
    else:
        viewer.add_labels(label_patch, name="patch_label")
    update_progress()
    print(f"Reverted to patch at {coord}.")




# Create the jump control widget
@magicgui(
    z_jump={"widget_type": "SpinBox", "min": 1, "max": 10000, "step": 1, "value": 500},
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
    
    # Calculate requested z coordinate
    current_z, current_y, current_x = current_coord
    requested_z = current_z + z_jump

    # Gather available z positions for the current (y, x) on the patch grid
    same_xy_coords = [c for c in state['patch_coords'] if len(c) == 3 and c[1] == current_y and c[2] == current_x]
    if not same_xy_coords:
        # Fallback to previous behavior if for some reason we don't have same-(y,x) coords
        new_index = find_closest_coord_index((requested_z, current_y, current_x), state['patch_coords'])
        state['current_index'] = new_index
        actual_coord = state['patch_coords'][new_index]
        print(f"Jumped from z={current_z} to nearest available {actual_coord} (fallback).")
        load_next_patch()
        update_progress()
        return

    # Sort candidates by z and build candidate list
    same_xy_coords.sort(key=lambda c: c[0])
    candidate_zs = [c[0] for c in same_xy_coords]

    # If exact requested z is not on the current grid for this (y, x),
    # rebase the grid starting at requested_z to honor exact jump.
    if requested_z not in candidate_zs:
        print(f"Requested jump +{z_jump} to z={requested_z}; not on current grid. Recomputing from z={requested_z}...")
        goto_z(z_abs=requested_z)
        return

    # Resolve the chosen coordinate exactly
    chosen_coord = (requested_z, current_y, current_x)
    new_index = state['patch_coords'].index(chosen_coord)
    state['current_index'] = new_index
    print(f"Requested jump +{z_jump} to z={requested_z}; moved to exact {chosen_coord}.")
    load_next_patch()
    update_progress()


@magicgui(
    z_abs={"widget_type": "SpinBox", "min": 0, "max": 999999},
    call_button="Go to Z"
)
def goto_z(z_abs: int = 0):
    """
    Recompute the patch grid starting at absolute z and continue from there.
    Attempts to start at the same (y, x) as the current patch.
    """
    global state

    if state.get('image_volume') is None or state.get('patch_size') is None:
        print("No volume initialized.")
        return

    # Only for 3D volumes
    if len(state['image_volume'].shape) < 3:
        print("Go to Z only applies to 3D volumes.")
        return

    num_spatial = 3
    vol_shape = state['image_volume'].shape[:num_spatial]
    patch_size = state['patch_size']
    sampling = state.get('sampling', 'sequence')

    # Bounds check: ensure there will be at least one patch starting at z_abs
    max_start_z = vol_shape[0] - patch_size
    if z_abs > max_start_z:
        print(f"Cannot start at z={z_abs}; max valid start is {max_start_z} for patch_size={patch_size}.")
        return

    # Remember current (y, x) if we have a current patch shown
    target_xy = None
    if state.get('current_patch') and len(state['current_patch']['coords']) == 3:
        _, cy, cx = state['current_patch']['coords']
        target_xy = (cy, cx)

    # Recompute patch grid from z_abs forward
    new_patch_coords = generate_patch_coords(vol_shape, patch_size, sampling, min_z=z_abs)
    state['patch_coords'] = new_patch_coords

    # Choose starting index
    if target_xy is not None:
        desired = (z_abs, target_xy[0], target_xy[1])
        if desired in new_patch_coords:
            new_index = new_patch_coords.index(desired)
        else:
            # Fallback: nearest coordinate with same (y, x) at or after z_abs, else nearest overall
            same_xy = [c for c in new_patch_coords if c[1] == target_xy[0] and c[2] == target_xy[1]]
            if same_xy:
                # Prefer the first at/after z_abs
                same_xy.sort(key=lambda c: c[0])
                after = [c for c in same_xy if c[0] >= z_abs]
                chosen = after[0] if after else same_xy[-1]
                new_index = new_patch_coords.index(chosen)
            else:
                new_index = find_closest_coord_index(desired, new_patch_coords)
    else:
        new_index = 0

    state['current_index'] = new_index
    print(f"Recomputed grid from z={z_abs}; starting at index {new_index} with coord {state['patch_coords'][new_index]}.")
    load_next_patch()
    update_progress()



# Create the set unlabeled to ignore widget
@magicgui(
    ignore_index={"widget_type": "LineEdit", "value": ""},
    call_button="Set Unlabeled to Ignore"
)
def set_unlabeled_to_ignore(ignore_index: str = ""):
    """
    Set all unlabeled (zero) regions in the current label patch to the specified ignore index.
    If no index is provided, uses the maximum value in the current label array + 1.
    """
    global viewer
    
    if "patch_label" not in viewer.layers:
        print("No label layer found in napari.")
        return
    
    # Get the current label data
    label_data = viewer.layers["patch_label"].data
    
    # Determine the ignore index
    if ignore_index.strip():
        try:
            ignore_val = int(ignore_index)
        except ValueError:
            print(f"Invalid ignore index: {ignore_index}. Must be an integer.")
            return
    else:
        # Use max value + 1
        max_val = np.max(label_data)
        ignore_val = max_val + 1
        # Update the widget to show the computed value
        set_unlabeled_to_ignore.ignore_index.value = str(ignore_val)
    
    # Create a copy to avoid modifying the original
    new_label_data = label_data.copy()
    
    # Set all zero (unlabeled) regions to the ignore value
    new_label_data[label_data == 0] = ignore_val
    
    # Update the napari layer
    viewer.layers["patch_label"].data = new_label_data
    
    print(f"Set all unlabeled regions to ignore index: {ignore_val}")
    print(f"  Unlabeled pixels modified: {np.sum(label_data == 0)}")


# Create the remove small objects widget
@magicgui(
    min_voxel_size={"widget_type": "SpinBox", "min": 0, "max": 10000, "step": 25, "value": 50},
    call_button="Remove Small Objects"
)
def remove_small_objects(min_voxel_size: int = 50):
    """
    Remove objects smaller than the specified voxel size using connected components analysis.
    
    Args:
        min_voxel_size: Minimum number of voxels for an object to be kept
    """
    global viewer
    
    if "patch_label" not in viewer.layers:
        print("No label layer found in napari.")
        return
    
    # Get the current label data
    label_data = viewer.layers["patch_label"].data
    
    # Ensure data is integer type for cc3d
    label_data_int = label_data.astype(np.uint32)
    
    # Run connected components
    print(f"Running connected components analysis...")
    labels_out = cc3d.connected_components(label_data_int)
    
    # Get statistics for each component
    stats = cc3d.statistics(labels_out)
    
    # Create a mapping of which labels to keep
    # Note: label 0 is background, so we start from 1
    num_components = len(stats['voxel_counts']) - 1  # Exclude background
    removed_count = 0
    total_voxels_removed = 0
    
    # Create a new array for the filtered result
    filtered_labels = np.zeros_like(labels_out)
    
    for label_id in range(1, len(stats['voxel_counts'])):
        voxel_count = stats['voxel_counts'][label_id]
        if voxel_count >= min_voxel_size:
            # Keep this component
            filtered_labels[labels_out == label_id] = 1
        else:
            removed_count += 1
            total_voxels_removed += voxel_count
    
    # Update the napari layer
    viewer.layers["patch_label"].data = filtered_labels.astype(np.uint8)
    
    print(f"Removed {removed_count} objects smaller than {min_voxel_size} voxels")
    print(f"  Total voxels removed: {total_voxels_removed}")
    print(f"  Objects remaining: {num_components - removed_count}")


def _get_layer_data(layer_name: str) -> Optional[np.ndarray]:
    global viewer
    if layer_name in viewer.layers:
        return viewer.layers[layer_name].data
    print(f"Layer '{layer_name}' not found.")
    return None


def main():
    """Main entry point for the proofreader application."""
    global viewer
    viewer = napari.Viewer()
    viewer.window.add_dock_widget(init_volume, name="Initialize Volumes", area="right")
    viewer.window.add_dock_widget(jump_control, name="Jump Control", area="right")
    viewer.window.add_dock_widget(set_unlabeled_to_ignore, name="Set Unlabeled to Ignore", area="right")
    viewer.window.add_dock_widget(remove_small_objects, name="Remove Small Objects", area="right")
    viewer.window.add_dock_widget(prev_pair, name="Previous Patch", area="right")
    viewer.window.add_dock_widget(iter_pair, name="Iterate Patches", area="right")
    
    # Set up event handler to update energy and resolution when volume selection changes
    @init_volume.volume_selection.changed.connect
    def update_energy_resolution():
        selected = init_volume.volume_selection.value
        if selected in config:
            init_volume.energy.value = f"{config[selected].get('energy', 'Unknown')} keV"
            init_volume.resolution.value = f"{config[selected].get('resolution', 'Unknown')} μm"
    
    # Initialize with the default selection
    if config:
        default_selection = list(config.keys())[0]
        init_volume.energy.value = f"{config[default_selection].get('energy', 'Unknown')} keV"
        init_volume.resolution.value = f"{config[default_selection].get('resolution', 'Unknown')} μm"

    # --- Keybindings ---
    @viewer.bind_key("Space")
    def next_pair_key(event):
        """Call the next pair function when the spacebar is pressed."""
        # Call iter_pair with the current value of the approved checkbox
        iter_pair(iter_pair.approved.value)

    @viewer.bind_key("a")
    def toggle_approved_key(event):
        """Toggle the 'approved' checkbox when the 'a' key is pressed."""
        iter_pair.approved.value = not iter_pair.approved.value

    napari.run()


if __name__ == '__main__':
    main()
