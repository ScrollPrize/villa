import os
import json
import numpy as np
import napari
import zarr
import fsspec
import tifffile
from magicgui import magicgui

# Try to import config defaults; if not found, use empty defaults.
try:
    from .config import config
except ImportError:
    config = {}

# Global state – for production code you might encapsulate this in a class.
state = {
    # Volumes for display (lower res) and saving (high res)
    'display_image_volume': None,
    'display_label_volume': None,
    'highres_image_volume': None,
    'highres_label_volume': None,
    'patch_size': None,  # (in display resolution units)
    'patch_coords': None,  # List of patch coordinates (tuples) in display resolution
    'current_index': 0,  # Next patch index to consider
    'dataset_out_path': None,
    'images_out_dir': None,
    'labels_out_dir': None,
    'current_patch': None,  # Info about current patch (from display volume)
    'scale_factor': None,  # Downsampling factor: highres = display * scale_factor
    'save_progress': False,  # Whether to save progress to a file
    'progress_file': "",  # Path to the progress file
    'min_label_percentage': 0,  # Minimum percentage (0-100) required in a patch
    # New progress log: a list of dicts recording every patch processed.
    'progress_log': []  # Each entry: { "index": int, "coords": tuple, "percentage": float, "status": str }
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
    return volume[slices]


def update_progress():
    """
    Write the progress log to a JSON file if progress saving is enabled.
    """
    if state.get('save_progress') and state.get('progress_file'):
        try:
            with open(state['progress_file'], "w") as f:
                json.dump({"progress_log": state['progress_log']}, f, indent=2)
            print(f"Progress saved to {state['progress_file']}.")
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
    Load the next valid patch from the display volumes and show it in napari.
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
    display_image_volume = state['display_image_volume']
    display_label_volume = state['display_label_volume']
    coords = state['patch_coords']
    min_label_percentage = state.get('min_label_percentage', 0)

    while state['current_index'] < len(coords):
        idx = state['current_index']
        coord = coords[idx]
        image_patch = extract_patch(display_image_volume, coord, patch_size)
        label_patch = extract_patch(display_label_volume, coord, patch_size)
        state['current_index'] += 1

        # Calculate the percentage of labeled (nonzero) pixels.
        nonzero = np.count_nonzero(label_patch)
        total = label_patch.size
        percentage = (nonzero / total * 100) if total > 0 else 0

        if percentage >= min_label_percentage:
            # Record this patch as pending (waiting for the user decision)
            entry = {"index": idx, "coords": coord, "percentage": percentage, "status": "pending"}
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
            print(f"Loaded patch at {coord} with {percentage:.2f}% labeled (threshold: {min_label_percentage}%).")
            return
        else:
            # Record an auto-skipped patch
            entry = {"index": idx, "coords": coord, "percentage": percentage, "status": "auto-skipped"}
            state['progress_log'].append(entry)
            print(f"Skipping patch at {coord} ({percentage:.2f}% labeled, below threshold of {min_label_percentage}%).")
    print("No more patches available.")


def save_current_patch():
    """
    Save the current patch extracted from the high resolution volumes.
    The display patch coordinates are scaled up to map to the high resolution.
    File names are constructed using the high-res zyx (or yx) coordinates:
      - Image file gets a '_0000' suffix (e.g. img_z{z}_y{y}_x{x}_0000.tif).
      - Label file does not (e.g. lbl_z{z}_y{y}_x{x}.tif).
    """
    global state
    if state.get('current_patch') is None:
        print("No patch available to save.")
        return

    patch = state['current_patch']
    display_coord = patch['coords']
    patch_size = state['patch_size']
    scale_factor = state['scale_factor']

    # Map display coordinate to high resolution coordinate.
    hr_coord = tuple(c * scale_factor for c in display_coord)
    hr_patch_size = patch_size * scale_factor

    # Extract high-resolution patches.
    highres_image_patch = extract_patch(state['highres_image_volume'], hr_coord, hr_patch_size)
    highres_label_patch = extract_patch(state['highres_label_volume'], hr_coord, hr_patch_size)

    # Construct coordinate string.
    if len(hr_coord) == 3:
        coord_str = f"z{hr_coord[0]}_y{hr_coord[1]}_x{hr_coord[2]}"
    elif len(hr_coord) == 2:
        coord_str = f"y{hr_coord[0]}_x{hr_coord[1]}"
    else:
        coord_str = "_".join(str(c) for c in hr_coord)

    image_filename = f"img_{coord_str}_0000.tif"
    label_filename = f"lbl_{coord_str}.tif"
    image_path = os.path.join(state['images_out_dir'], image_filename)
    label_path = os.path.join(state['labels_out_dir'], label_filename)

    tifffile.imwrite(image_path, highres_image_patch)
    tifffile.imwrite(label_path, highres_label_patch)
    print(f"Saved high-res image patch to {image_path} and label patch to {label_path}")


@magicgui(
    sampling={"choices": ["random", "sequence"]},
    min_label_percentage={"min": 0, "max": 100},
    min_z={"widget_type": "SpinBox", "min": 0, "max": 999999},
)
def init_volume(
        image_zarr: str = config.get("image_zarr", ""),
        label_zarr: str = config.get("label_zarr", ""),
        dataset_out_path: str = config.get("dataset_out_path", ""),
        patch_size: int = config.get("patch_size", 64),  # patch_size is provided in full-res units
        sampling: str = config.get("sampling", "sequence"),
        save_progress: bool = config.get("save_progress", False),
        progress_file: str = config.get("progress_file", ""),
        min_label_percentage: int = config.get("min_label_percentage", 0),
        min_z: int = 0,  # <-- New parameter: minimum z index from which to start patching (only for 3D)
        use_full_resolution: bool = False  # New checkbox parameter.
):
    """
    Load image and label volumes from Zarr. If the checkbox for 'use_full_resolution' is
    checked then the full resolution (usually stored at key "0") is used for display;
    otherwise, resolution "1" (downsampled) is used for display. In either case the full
    resolution (or "0") is used for saving patches. The patch_size provided is assumed to be
    in full-resolution units.
    """
    global state, viewer

    # --- Load image volume ---
    try:
        image_mapper = fsspec.get_mapper(image_zarr)
        image_group = zarr.open_group(image_mapper, mode='r')
        if use_full_resolution:
            if "0" in image_group:
                display_image_volume = image_group["0"]
                print("Using full resolution ('0') for display of image volume.")
            else:
                display_image_volume = image_group
                print("No resolution '0' found; using provided array for display (assumed full resolution).")
        else:
            if "1" in image_group:
                display_image_volume = image_group["1"]
                print("Using resolution '1' for display of image volume.")
            else:
                display_image_volume = image_group
                print("No resolution '1' found for image volume; using provided array for display.")
        if "0" in image_group:
            highres_image_volume = image_group["0"]
            print("Using resolution '0' for saving patches of image volume.")
        else:
            highres_image_volume = image_group
            print("No resolution '0' found for image volume; using provided array for saving patches.")
    except Exception as e:
        print(f"Error loading image zarr: {e}")
        return

    # --- Load label volume ---
    try:
        label_mapper = fsspec.get_mapper(label_zarr)
        label_group = zarr.open_group(label_mapper, mode='r')
        if use_full_resolution:
            if "0" in label_group:
                display_label_volume = label_group["0"]
                print("Using full resolution ('0') for display of label volume.")
            else:
                display_label_volume = label_group
                print("No resolution '0' found; using provided array for display (assumed full resolution).")
        else:
            if "1" in label_group:
                display_label_volume = label_group["1"]
                print("Using resolution '1' for display of label volume.")
            else:
                display_label_volume = label_group
                print("No resolution '1' found for label volume; using provided array for display.")
        if "0" in label_group:
            highres_label_volume = label_group["0"]
            print("Using resolution '0' for saving patches of label volume.")
        else:
            highres_label_volume = label_group
            print("No resolution '0' found for label volume; using provided array for saving patches.")
    except Exception as e:
        print(f"Error loading label zarr: {e}")
        return

    # Save the loaded volumes.
    state['display_image_volume'] = display_image_volume
    state['highres_image_volume'] = highres_image_volume
    state['display_label_volume'] = display_label_volume
    state['highres_label_volume'] = highres_label_volume
    state['dataset_out_path'] = dataset_out_path

    # Save progress options.
    state['save_progress'] = save_progress
    state['progress_file'] = progress_file

    # Save the minimum label percentage.
    state['min_label_percentage'] = min_label_percentage

    # Create output directories.
    images_dir = os.path.join(dataset_out_path, 'imagesTr')
    labels_dir = os.path.join(dataset_out_path, 'labelsTr')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    state['images_out_dir'] = images_dir
    state['labels_out_dir'] = labels_dir

    # Compute the scale factor between high-res and display volumes.
    num_spatial = 2 if len(display_image_volume.shape) == 2 else 3
    display_shape = display_image_volume.shape[:num_spatial]
    highres_shape = highres_image_volume.shape[:num_spatial]
    scale_factors = [high / disp for disp, high in zip(display_shape, highres_shape)]
    if not all(np.isclose(sf, scale_factors[0]) for sf in scale_factors):
        print("Warning: Non-isotropic scale factors detected. Using first dimension's scale factor.")
    scale_factor = int(round(scale_factors[0]))
    state['scale_factor'] = scale_factor
    print(f"Computed scale factor: {scale_factor}")

    # Adjust patch size. If full resolution is used for display then use patch_size as-is;
    # otherwise, compute the display patch size by dividing by the scale factor.
    if use_full_resolution:
        display_patch_size = patch_size
        print(f"Using full resolution for display; patch size remains {display_patch_size}.")
    else:
        display_patch_size = int(round(patch_size / scale_factor))
        print(f"Adjusted patch size for display: {display_patch_size} (from full-res patch size {patch_size})")
    state['patch_size'] = display_patch_size

    # Compute patch coordinates on the display volume.
    vol_shape = display_shape  # 2D or 3D shape of the display volume.
    new_patch_coords = generate_patch_coords(vol_shape, display_patch_size, sampling, min_z=min_z)
    state['patch_coords'] = new_patch_coords

    # Attempt to load prior progress.
    load_progress()

    # If a progress log exists, adjust the starting patch index based on the last processed coordinate.
    if state['progress_log']:
        old_coord = state['progress_log'][-1]['coords']
        # Option 1: "Snap" the old coordinate to the new grid.
        new_start_coord = tuple((c // display_patch_size) * display_patch_size for c in old_coord)
        if new_start_coord in new_patch_coords:
            new_index = new_patch_coords.index(new_start_coord)
        else:
            # Option 2: Use nearest neighbor.
            new_index = find_closest_coord_index(old_coord, new_patch_coords)
        state['current_index'] = new_index
        print(f"Resuming from new patch index {new_index} (closest to old coordinate {old_coord}).")
    else:
        state['current_index'] = 0

    print(f"Loaded display volumes with shape {vol_shape}.")
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
    When "previous pair" is pressed, go back to the last patch that was shown (i.e. one that wasn’t auto-skipped).
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
    image_patch = extract_patch(state['display_image_volume'], coord, patch_size)
    label_patch = extract_patch(state['display_label_volume'], coord, patch_size)
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


def main():
    """Main entry point for the proofreader application."""
    global viewer
    viewer = napari.Viewer()
    viewer.window.add_dock_widget(init_volume, name="Initialize Volumes", area="right")
    viewer.window.add_dock_widget(iter_pair, name="Iterate Patches", area="right")
    viewer.window.add_dock_widget(prev_pair, name="Previous Patch", area="right")

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
