import os
import yaml
import json
from numpy.typing import NDArray
from typing import Any, Dict, Optional, Tuple, Union, List
import numpy as np
import requests
import nrrd
import tempfile
from PIL import Image
from io import BytesIO
from pathlib import Path
# Direct import to avoid circular reference issues
# Import necessary functions directly to avoid circular imports
import os
import yaml
import requests
from setup.accept_terms import get_installation_path
import zarr

# Define the functions needed here to avoid circular imports
def list_files():
    """Load and return the scrolls configuration data from a YAML file."""
    install_path = get_installation_path()
    scroll_config = os.path.join(install_path, 'vesuvius', 'configs', f'scrolls.yaml')
    with open(scroll_config, 'r') as file:
        data = yaml.safe_load(file)
    return data


def is_aws_ec2_instance():
    """Determine if the current system is an AWS EC2 instance."""
    try:
        response = requests.get("http://169.254.169.254/latest/meta-data/", timeout=2)
        if response.status_code == 200:
            return True
    except requests.RequestException:
        return False
    return False


import torch
import fsspec
from .utils import get_max_value, open_zarr

# Remove the PIL image size limit
Image.MAX_IMAGE_PIXELS = None


class Volume:
    """
    A class to represent a 3D volume in a scroll or segment.

    Attributes
    ----------
    type : Union[str, int]
        The type of volume, either a scroll or a segment.
    scroll_id : Optional[int]
        ID of the scroll.
    energy : Optional[int]
        Energy value associated with the volume.
    resolution : Optional[float]
        Resolution of the volume.
    segment_id : Optional[int]
        ID of the segment.
    normalization_scheme : str
        Specifies the normalization method:
        - 'none': No normalization.
        - 'instance_zscore': Z-score normalization computed per slice/volume instance.
        - 'global_zscore': Z-score normalization using pre-computed global mean/std.
        - 'instance_minmax': Min-max scaling to [0, 1] computed per slice/volume instance.
    global_mean : Optional[float]
        Global mean value (required for 'global_zscore').
    global_std : Optional[float]
        Global standard deviation value (required for 'global_zscore').
    return_as_type : str
        Target NumPy dtype for the returned data (e.g., 'np.float32', 'np.uint8').
        'none' keeps the dtype resulting from normalization (usually float32) or original dtype.
    return_as_tensor : bool
        If True, returns data as a PyTorch tensor.
    verbose : bool
        If True, prints detailed information during operations.
    domain : str
        Data source domain ('dl.ash2txt' or 'local').
    path : Optional[str]
        Path to local data or base URL for remote data.
    configs : str
        Path to the YAML configuration file (for non-Zarr types).
    url : str
        Resolved URL or path to the data store.
    metadata : Dict[str, Any]
        Metadata loaded from the data store (e.g., .zattrs).
    inklabel : Optional[np.ndarray]
        Ink label data (only for segments). None otherwise.
    dtype : np.dtype
        Original data type of the primary volume data.
    """

    def __init__(self, type: Union[str, int],
                 scroll_id: Optional[Union[int, str]] = None,
                 energy: Optional[int] = None,
                 resolution: Optional[float] = None,
                 segment_id: Optional[int] = None,
                 format: str = 'zarr',  # Currently only zarr 
                 normalization_scheme: str = 'none',
                 global_mean: Optional[float] = None,
                 global_std: Optional[float] = None,
                 return_as_type: str = 'none',
                 return_as_tensor: bool = False,
                 verbose: bool = False,
                 domain: Optional[str] = None,
                 path: Optional[str] = None,
                 download_only: bool = False,
                 ):

        """
        Initializes the Volume object.

        Parameters
        ----------
        type : Union[str, int]
            Volume type or identifier ('scroll', 'segment', 'zarr', scroll name, segment timestamp).
        scroll_id : Optional[Union[int, str]]
            Scroll ID (required if type is 'scroll' or 'segment' and not implicitly defined by 'type').
        energy : Optional[int]
            Energy level. Uses canonical if None.
        resolution : Optional[float]
            Resolution. Uses canonical if None.
        segment_id : Optional[int]
            Segment ID (required if type is 'segment').
        format : str, default = 'zarr'
            Data format (currently only 'zarr').
        normalization_scheme : str, default = 'none'
            Normalization method ('none', 'instance_zscore', 'global_zscore', 'instance_minmax').
        global_mean : Optional[float], default = None
            Global mean for 'global_zscore'. Must be provided if scheme is 'global_zscore'.
        global_std : Optional[float], default = None
            Global standard deviation for 'global_zscore'. Must be provided if scheme is 'global_zscore'.
        return_as_type : str, default = 'none'
            Target NumPy dtype string (e.g., 'np.float32', 'np.uint16'). 'none' means no explicit conversion after normalization.
        return_as_tensor : bool, default = False
            If True, return PyTorch tensors.
        verbose : bool, default = False
            Enable verbose logging.
        domain : Optional[str], default = Determined automatically ('dl.ash2txt' or 'local')
            Data source domain.
        path : Optional[str], default = None
            Direct path/URL to the Zarr store if type is 'zarr'.
        download_only : bool, default = False
            If True, only prepare for downloading without loading the actual data.
            Useful for segments when you only want to download the ink labels.
        """

        # Initialize basic attributes
        self.format = format
        self.normalization_scheme = normalization_scheme
        self.global_mean = global_mean
        self.global_std = global_std
        self.return_as_type = return_as_type
        self.return_as_tensor = return_as_tensor
        self.path = path
        self.verbose = verbose
        self.inklabel = None  # Initialize inklabel

        # --- Input Validation ---
        valid_schemes = ['none', 'instance_zscore', 'global_zscore', 'instance_minmax']
        if self.normalization_scheme not in valid_schemes:
            raise ValueError(
                f"Invalid normalization_scheme: '{self.normalization_scheme}'. Must be one of {valid_schemes}")

        if self.normalization_scheme == 'global_zscore' and (self.global_mean is None or self.global_std is None):
            raise ValueError("global_mean and global_std must be provided when normalization_scheme is 'global_zscore'")

        try:
            # --- Zarr Direct Path Handling ---
            if format == "zarr" and self.path is not None:
                if self.verbose:
                    print(f"Initializing Volume from direct Zarr path: {self.path}")
                self.type = "zarr"  # Explicitly set type for zarr path initialization
                self._init_from_zarr_path()
                if self.verbose:
                    self.meta()
                return  # Initialization complete for direct Zarr

            # --- Scroll/Segment Type Resolution ---
            # Determine type, scroll_id, segment_id from 'type' parameter if needed
            if isinstance(type, str):
                if type.lower().startswith("scroll") and len(type) > 6:  # e.g., "scroll1", "scroll1b"
                    self.type = "scroll"
                    scroll_part = type[6:]
                    self.scroll_id = int(scroll_part) if scroll_part.isdigit() else scroll_part
                    self.segment_id = None
                elif type.isdigit():  # Assume it's a segment timestamp
                    segment_id_str = str(type)
                    details = self.find_segment_details(segment_id_str)
                    if details[0] is None:
                        raise ValueError(f"Could not find details for segment ID: {segment_id_str}")
                    s_id, e, res, _ = details
                    self.type = "segment"
                    self.segment_id = int(segment_id_str)
                    self.scroll_id = scroll_id if scroll_id is not None else s_id
                    energy = energy if energy is not None else e
                    resolution = resolution if resolution is not None else res
                    if self.verbose:
                        print(
                            f"Resolved segment {segment_id_str} to scroll {self.scroll_id}, E={energy}, Res={resolution}")
                elif type in ["scroll", "segment"]:
                    self.type = type
                    if type == "segment":
                        assert isinstance(segment_id, int), "segment_id must be an int when type is 'segment'"
                        self.segment_id = segment_id
                        self.scroll_id = scroll_id
                    else:  # type == "scroll"
                        self.segment_id = None
                        self.scroll_id = scroll_id
                else:
                    raise ValueError(
                        f"Invalid 'type' string: {type}. Expected 'scroll', 'segment', 'ScrollX', 'zarr', or segment timestamp.")
            elif isinstance(type, int):  # Assume it's a scroll ID if just an int
                self.type = "scroll"
                self.scroll_id = type
                self.segment_id = None
            else:
                raise ValueError(f"Invalid 'type': {type}. Must be str or int.")

            # --- Domain Determination ---
            if domain is None:
                self.aws = is_aws_ec2_instance()
                self.domain = "local" if self.aws else "dl.ash2txt"
            else:
                self.aws = False  # Assume not AWS if domain is explicitly set
                assert domain in ["dl.ash2txt", "local"], "domain should be 'dl.ash2txt' or 'local'"
                self.domain = domain
            if self.verbose:
                print(f"Using domain: {self.domain}")

            # --- Config File ---
            # Use relative paths for config files instead of installation path
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            possible_paths = [
                os.path.join(base_dir, 'setup', 'configs', f'scrolls.yaml'),
                os.path.join(base_dir, 'configs', f'scrolls.yaml')
            ]
            self.configs = None
            for config_path in possible_paths:
                if os.path.exists(config_path):
                    self.configs = config_path
                    break
            if self.configs is None:
                self.configs = possible_paths[0]  # Default to first path for error message
                print(
                    f"Warning: Could not find config file at expected locations: {possible_paths}. Will try default: {self.configs}")
                # Error will be raised in get_url_from_yaml if file truly doesn't exist

            # --- Energy & Resolution ---
            self.energy = energy if energy is not None else self.grab_canonical_energy()
            self.resolution = resolution if resolution is not None else self.grab_canonical_resolution()
            if self.energy is None or self.resolution is None:
                raise ValueError(
                    f"Could not determine energy/resolution for scroll {self.scroll_id}. Please provide them explicitly.")

            # --- Get URL and Load Data ---
            self.url = self.get_url_from_yaml()  # This sets self.url based on type, scroll, energy, res
            if self.verbose:
                print(f"Resolved data URL/path: {self.url}")
            
            # Early return for download_only mode (for segments)
            if download_only:
                self.metadata = {}  # Empty metadata
                if self.type == "segment":
                    self.download_inklabel()  # Only download the ink label
                return

            self.metadata = self.load_ome_metadata()  # Loads .zattrs
            self.data = self.load_data() 
            
            # Handle different data source types
            if isinstance(self.data, zarr.Array):
                # Direct zarr array case
                self.dtype = self.data.dtype
            elif hasattr(self.data[0].dtype, 'numpy_dtype'):
                #  case
                self.dtype = self.data[0].dtype.numpy_dtype
            else:
                # Fallback for other cases
                self.dtype = self.data[0].dtype

            # --- Segment Specific ---
            if self.type == "segment":
                self.download_inklabel()  # Sets self.inklabel

            if self.verbose:
                self.meta()

        except Exception as e:
            print(f"ERROR initializing Volume: {e}")
            print("Common issues:")
            print("- Ensure Zarr path is correct and accessible if using direct path.")
            print(
                "- Ensure config file exists and contains entries for the requested scroll/segment/energy/resolution.")
            print("- Check network connection if accessing remote data.")
            # Provide example usage hints
            print("\nExample Usage:")
            print('  volume = Volume(type="scroll", scroll_id=1)')
            print('  segment = Volume(type="segment", segment_id=20230827161847)')
            print('  zarr_vol = Volume(type="zarr", path="/path/to/my/data.zarr")')
            raise

    def _init_from_zarr_path(self):
        """Helper to initialize directly from a Zarr path."""
        # Log what we're doing if verbose
        if self.verbose:
            print(f"Opening zarr store at path: {self.path}")
        
        # Use our helper function for direct zarr access
        try:
            # Open the zarr store directly
            self.data = open_zarr(
                path=self.path,
                mode='r',
                storage_options={'anon': False} if self.path.startswith('s3://') else None,
                verbose=self.verbose
            )
            
            # Get original dtype
            if isinstance(self.data.dtype, type):
                self.dtype = np.dtype(self.data.dtype)
            else:
                self.dtype = self.data.dtype
                
            if self.verbose:
                print(f"Successfully opened zarr store: {self.data}")
                
            # Set URL for consistency with other parts of the code
            self.url = self.path
                
            # Load metadata (.zattrs)
            try:
                self.metadata = self.load_ome_metadata()
            except Exception as meta_e:
                if self.verbose:
                    print(f"Warning: Could not load .zattrs metadata from {self.path}: {meta_e}")
                self.metadata = {}  # Assign empty dict if metadata loading fails
    
            # Set remaining attributes for consistency
            if not hasattr(self, 'type'):
                self.type = "zarr"
            self.scroll_id = None
            self.segment_id = None
            # Determine domain from path
            self.domain = "local" if not self.path.startswith(('http://', 'https://', 's3://')) else "dl.ash2txt"
            self.resolution = None
            self.energy = None
            
        except Exception as e:
            if self.verbose:
                print(f"Error opening zarr at {self.path}: {e}")
            raise

    def meta(self) -> None:
        """Prints shape information for loaded volume data."""
        print(f"--- Volume Metadata ({self.type}) ---")
        if self.scroll_id: print(f"Scroll ID: {self.scroll_id}")
        if self.segment_id: print(f"Segment ID: {self.segment_id}")
        if self.energy: print(f"Energy: {self.energy}")
        if self.resolution: print(f"Resolution: {self.resolution}")
        print(f"URL/Path: {self.url}")
        print(f"Original Dtype: {self.dtype}")
        print(f"Normalization Scheme: {self.normalization_scheme}")
        if self.normalization_scheme == 'global_zscore':
            print(f"  Global Mean: {self.global_mean}, Global Std: {self.global_std}")
        print(f"Return Type: {self.return_as_type}")
        print(f"Return as Tensor: {self.return_as_tensor}")
        print(f"Number of Resolution Levels: {len(self.data)}")
        for idx, store in enumerate(self.data):
            print(f"  Level {idx} Shape: {store.shape}, Dtype: {store.dtype}")
        if self.inklabel is not None:
            print(f"Ink Label Shape: {self.inklabel.shape}")
        print("-------------------------")

    def find_segment_details(self, segment_id: str) -> Tuple[
        Optional[Union[int, str]], Optional[int], Optional[float], Optional[Dict[str, Any]]]:
        """
        Find the details of a segment given its ID.

        Parameters
        ----------
        segment_id : str
            The ID of the segment to search for.

        Returns
        -------
        Tuple[Optional[int], Optional[int], Optional[float], Optional[Dict[str, Any]]]
            A tuple containing scroll_id, energy, resolution, and segment metadata.

        Raises
        ------
        ValueError
            If the segment details cannot be found.
        """

        dictionary = list_files()
        stack = [(list(dictionary.items()), [])]

        while stack:
            items, path = stack.pop()

            for key, value in items:
                if isinstance(value, dict):
                    # Check if 'segments' key is present in the current level of the dictionary
                    if 'segments' in value:
                        # Check if the segment_id is in the segments dictionary
                        if segment_id in value['segments']:
                            scroll_id, energy, resolution = path[0], path[1], key
                            return scroll_id, energy, resolution, value['segments'][segment_id]
                    # Add nested dictionary to the stack for further traversal
                    stack.append((list(value.items()), path + [key]))

        return None, None, None, None

    def get_url_from_yaml(self) -> str:
        """Retrieves the data URL/path from the YAML config file."""
        # This method is primarily for scroll/segment types, not direct Zarr paths
        if self.type == 'zarr':
            # This case should ideally be handled by _init_from_zarr_path setting self.url
            # If called unexpectedly, return the path provided.
            return self.path if self.path else ""

        if not self.configs or not os.path.exists(self.configs):
            error_msg = f"Configuration file not found at {self.configs}. "
            # ... (rest of your helpful error message) ...
            raise FileNotFoundError(error_msg)

        try:
            with open(self.configs, 'r') as file:
                config_data: Dict = yaml.safe_load(file)

            if not config_data:
                raise ValueError(f"Config file {self.configs} is empty or invalid YAML.")

            # Navigate the config structure
            scroll_data = config_data.get(str(self.scroll_id), {})
            energy_data = scroll_data.get(str(self.energy), {})
            res_data = energy_data.get(str(self.resolution), {})

            if self.type == 'scroll':
                url = res_data.get("volume")
                if url is None:
                    raise ValueError(
                        f"URL not found in config for scroll={self.scroll_id}, energy={self.energy}, resolution={self.resolution}")
            elif self.type == 'segment':
                url = res_data.get("segments", {}).get(str(self.segment_id))
                if url is None:
                    raise ValueError(
                        f"URL not found in config for segment={self.segment_id} (scroll={self.scroll_id}, energy={self.energy}, resolution={self.resolution})")
            else:
                # Should not happen if type logic is correct
                raise TypeError(f"Cannot retrieve URL from config for type: {self.type}")

            return url

        except FileNotFoundError:
            # This duplicates the check at the start, but covers the case where self.configs was None
            error_msg = f"Configuration file not found at {self.configs}. "
            # ... (rest of your helpful error message) ...
            raise FileNotFoundError(error_msg)
        except Exception as e:
            print(f"Error reading or parsing config file {self.configs}: {e}")
            raise

    def load_ome_metadata(self) -> Dict[str, Any]:
        """Loads OME-Zarr metadata (.zattrs) from zarr group attributes."""
        # Determine the base URL/path correctly, handling direct path or config-derived URL
        base_path = self.path if self.type == 'zarr' and self.path else self.url
        if not base_path:
            raise ValueError("Could not determine base path/URL for metadata loading.")

        base_path = base_path.rstrip("/")
        
        # First try to access metadata directly from the zarr store if it's already loaded
        if hasattr(self, 'data') and self.data is not None and hasattr(self.data, 'attrs'):
            try:
                attrs_dict = dict(self.data.attrs)
                if attrs_dict:  # If attributes exist and aren't empty
                    if self.verbose:
                        print(f"Successfully loaded metadata from zarr store attributes")
                    return {"zattrs": attrs_dict}
            except Exception as e:
                if self.verbose:
                    print(f"Could not access attributes from data: {e}")
        
        # If we couldn't get attributes from the already-loaded store, try opening the paths directly
        potential_zattrs_paths = [
            base_path,            # Try the base path directly (zarr.open will access attrs)
            f"{base_path}/0",     # Try the first resolution level
        ]

        for zattrs_path in potential_zattrs_paths:
            if self.verbose:
                print(f"Attempting to load metadata from: {zattrs_path}")
                
            try:
                # Use our helper function to open the zarr store
                zarr_store = open_zarr(
                    path=zattrs_path, 
                    mode='r',
                    storage_options={'anon': False} if zattrs_path.startswith('s3://') else None,
                    verbose=self.verbose
                )
                
                # Get attributes from the store
                if hasattr(zarr_store, 'attrs'):
                    attrs_dict = dict(zarr_store.attrs)
                    if self.verbose:
                        print(f"Successfully loaded metadata from {zattrs_path}")
                    return {"zattrs": attrs_dict}
                else:
                    if self.verbose:
                        print(f"No attributes found in {zattrs_path}")
            except Exception as e:
                if self.verbose:
                    print(f"Error accessing {zattrs_path}: {e}")

        # If we still haven't found metadata, try looking for .zattrs files explicitly with fsspec
        # This is a fallback for stores that don't have proper zarr attributes
        zattrs_paths = [
            f"{base_path}/.zattrs",  # Standard location at root
            f"{base_path}/0/.zattrs"  # Common location for first level in multi-resolution
        ]
        
        for zattrs_path in zattrs_paths:
            if self.verbose:
                print(f"Falling back to direct .zattrs file access: {zattrs_path}")
                
            try:
                # Use fsspec to open the file directly - works for any protocol
                with fsspec.open(zattrs_path, mode='rb') as f:
                    zattrs_content = json.load(f)
                
                if self.verbose:
                    print(f"Successfully loaded metadata from {zattrs_path}")
                
                return {"zattrs": zattrs_content}
            except (FileNotFoundError, json.JSONDecodeError) as e:
                if self.verbose:
                    print(f"Could not load metadata from {zattrs_path}: {e}")
            except Exception as e:
                if self.verbose:
                    print(f"Error accessing {zattrs_path}: {e}")

        # If loop completes without returning, metadata wasn't found
        if self.verbose:
            print(f"Warning: Could not load any metadata from base path {base_path}")
        return {}  # Return empty dict if no metadata found

    def load_data(self):
        """Load data from URL or path using direct zarr.open"""
        base_path = self.path if self.type == 'zarr' and self.path else self.url
        if not base_path: 
            raise ValueError("Base path/URL missing.")
        
        if self.verbose:
            print(f"Loading data from: {base_path}")
        
        try:
            # Use our helper function to open the zarr store
            data = open_zarr(
                path=base_path,
                mode='r',
                storage_options={'anon': False} if base_path.startswith('s3://') else None,
                verbose=self.verbose
            )
            
            if self.verbose:
                print(f"Successfully opened zarr store: {data}")
                print(f"Shape: {data.shape}, Dtype: {data.dtype}")
                
            return data
        except Exception as e:
            if self.verbose:
                print(f"Error opening zarr at {base_path}: {e}")
            raise

    def download_inklabel(self, save_path=None) -> None:
        """
        Downloads and loads the ink label image for a segment.
        
        Parameters
        ----------
        save_path : str, optional
            If provided, saves the downloaded ink label to this path.
            
        Returns
        -------
        None
            Sets self.inklabel to the loaded image as numpy array.
        """
        assert self.type == "segment", "Ink labels are only available for segments."
        if not self.url:
            print("Warning: Cannot download inklabel, URL is not set.")
            self.inklabel = np.zeros((1, 1), dtype=np.uint8)  # Placeholder
            return
            
        # Create inklabel attribute if not already created
        if not hasattr(self, 'inklabel'):
            self.inklabel = None

        # Construct inklabel URL (heuristic based on typical naming)
        base_url = self.url.rstrip('/')
        # Extract parent URL and segment ID
        parent_url = os.path.dirname(base_url)
        segment_id_str = os.path.basename(base_url)
        # Construct ink label path
        inklabel_filename = f"{segment_id_str}_inklabels.png"
        inklabel_url = os.path.join(parent_url, inklabel_filename)

        if self.verbose:
            print(f"Attempting to load ink label from: {inklabel_url}")

        try:
            # Note: We still use fsspec here because this is a PNG file, not a zarr store
            # For PNGs and other file types, fsspec.open is still the appropriate choice
            storage_options = {'anon': False} if inklabel_url.startswith('s3://') else None
            with fsspec.open(inklabel_url, mode='rb', **({} if storage_options is None else storage_options)) as f:
                img_bytes = f.read()
                img = Image.open(BytesIO(img_bytes))
                
                # Save the downloaded image if a save path is provided
                if save_path:
                    img.save(save_path)
                    print(f"Saved ink label to: {save_path}")
                    
                # Convert to grayscale if it's not already L mode
                if img.mode != 'L':
                    img = img.convert('L')
                    
                self.inklabel = np.array(img)

            if self.verbose:
                print(f"Successfully loaded ink label with shape: {self.inklabel.shape}, dtype: {self.inklabel.dtype}")

        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not load ink label from {inklabel_url}: {e}")
            
            # Create an empty/dummy ink label array based on data shape if possible
            if hasattr(self, 'data') and self.data:
                try:
                    base_shape = self.shape(0)  # Shape of highest resolution
                    # Assume inklabel matches YX dimensions of the 3D volume
                    if len(base_shape) >= 3:
                        self.inklabel = np.zeros(base_shape[-2:], dtype=np.uint8)  # (Y, X)
                        if self.verbose:
                            print(f"Created empty placeholder ink label with shape: {self.inklabel.shape}")
                    else:
                        self.inklabel = np.zeros((1, 1), dtype=np.uint8)  # Fallback
                except Exception:
                    self.inklabel = np.zeros((1, 1), dtype=np.uint8)  # Final fallback
            else:
                self.inklabel = np.zeros((1, 1), dtype=np.uint8)  # Fallback if data not loaded

    def __getitem__(self, idx: Union[Tuple[Union[int, slice], ...], int]) -> Union[NDArray, torch.Tensor]:
        """
        Gets a sub-volume or slice, applying specified normalization and type conversion.

        Indexing follows NumPy conventions. For 3D data, the order is (z, y, x).
        A 4th index can specify the resolution level (sub-volume index), default is 0.

        Parameters
        ----------
        idx : Union[Tuple[Union[int, slice], ...], int]
            Index tuple (z, y, x) or (z, y, x, subvolume_idx). Slices are allowed.

        Returns
        -------
        Union[NDArray, torch.Tensor]
            The requested data slice, processed according to instance settings.

        Raises
        ------
        IndexError
            If the index format or bounds are invalid.
        ValueError
            If normalization settings are inconsistent.
        """
        subvolume_idx = 0
        coord_idx = idx

        # --- Parse Index ---
        if isinstance(idx, tuple):
            if len(idx) == 0:
                raise IndexError("Empty index tuple provided.")

            # Check if the last element looks like a subvolume index (integer)
            # compared to the dimensionality of the data.
            # Handle the case when self.data is a zarr array directly (from _init_from_zarr_path)
            if isinstance(self.data, zarr.Array):
                data_ndim = self.data.ndim  # Dimensionality of the zarr array
            else:
                data_ndim = self.data[0].ndim  # Dimensionality of the base resolution
            if len(idx) == data_ndim + 1 and isinstance(idx[-1], int):
                # Assume last element is subvolume index
                potential_subvolume_idx = idx[-1]
                if 0 <= potential_subvolume_idx < len(self.data):
                    subvolume_idx = potential_subvolume_idx
                    coord_idx = idx[:-1]  # Use preceding elements as coordinates
                    if len(coord_idx) != data_ndim:
                        # This case shouldn't happen if logic above is sound, but safety check
                        raise IndexError(
                            f"Coordinate index length {len(coord_idx)} doesn't match data ndim {data_ndim} after extracting subvolume index.")
                else:
                    # Last element is int but out of bounds for subvolumes, treat as coordinate
                    coord_idx = idx
                    if len(coord_idx) != data_ndim:
                        raise IndexError(
                            f"Index tuple length {len(coord_idx)} does not match data dimensions ({data_ndim}).")

            elif len(idx) == data_ndim:
                # Index length matches data dimensions, use subvolume 0
                coord_idx = idx
                subvolume_idx = 0
            else:
                raise IndexError(
                    f"Index tuple length {len(idx)} does not match data dimensions ({data_ndim}) or format (coords + subvolume_idx).")

        elif isinstance(idx, (int, slice)):
            # Allow single index/slice if data is 1D (unlikely for volumes but possible)
            if self.data[subvolume_idx].ndim == 1:
                coord_idx = (idx,)  # Make it a tuple
            else:
                raise IndexError("Single index/slice provided for multi-dimensional data. Use a tuple (z, y, x, ...).")
        else:
            raise IndexError(f"Unsupported index type: {type(idx)}")

        # Validate subvolume index again just in case
        if isinstance(self.data, zarr.Array):
            # Direct zarr array doesn't have subvolumes
            if subvolume_idx != 0:
                raise IndexError(f"Invalid subvolume index: {subvolume_idx}. Direct zarr array has only index 0.")
        elif not (0 <= subvolume_idx < len(self.data)):
            raise IndexError(f"Invalid subvolume index: {subvolume_idx}. Must be between 0 and {len(self.data) - 1}.")

        # --- Read Data Slice ---
        if self.verbose:
            print(f"Accessing data level {subvolume_idx} with coordinates: {coord_idx}")
            if isinstance(self.data, zarr.Array):
                print(f"  Store shape: {self.data.shape}, Store dtype: {self.data.dtype}")
            else:
                print(f"  Store shape: {self.data[subvolume_idx].shape}, Store dtype: {self.data[subvolume_idx].dtype}")

        try:
            # Handle the case when self.data is a zarr array directly (from _init_from_zarr_path)
            if isinstance(self.data, zarr.Array):
                data_slice = self.data[coord_idx]
            else:
                data_slice = self.data[subvolume_idx][coord_idx]

            original_dtype = data_slice.dtype  

            if self.verbose:
                print(f"  Read slice shape: {data_slice.shape}, dtype: {data_slice.dtype}")

        except Exception as e:
            print(f"ERROR during zarr read operation:")
            print(f"  Subvolume: {subvolume_idx}, Index: {coord_idx}")
            if isinstance(self.data, zarr.Array):
                print(f"  Store Shape: {self.data.shape}")
            else:
                print(f"  Store Shape: {self.data[subvolume_idx].shape}")
            print(f"  Error: {e}")
            raise  # Re-raise the exception

        # --- Preprocessing Steps ---

        # 1. Convert to float32 for normalization calculations (if needed)
        if self.normalization_scheme != 'none':
            if np.issubdtype(data_slice.dtype, np.floating):
                # If already float, ensure it's float32 for consistency
                if data_slice.dtype != np.float32:
                    data_slice = data_slice.astype(np.float32)
                    if self.verbose: print(f"  Cast existing float ({original_dtype}) to np.float32 for normalization.")
            else:
                # If integer or other, convert to float32
                data_slice = data_slice.astype(np.float32)
                if self.verbose: print(f"  Cast {original_dtype} to np.float32 for normalization.")

        # 2. Apply Normalization Scheme
        # Handle potential channel dimension (assume channels are dim 0 if present)
        # Add temporary channel dim for 3D data (Z, Y, X) -> (1, Z, Y, X) for consistent logic
        original_ndim = data_slice.ndim
        has_channel_dim = original_ndim > 3  # Heuristic: assume >3D means channels exist at dim 0
        if not has_channel_dim and original_ndim == 3:  # Add channel dim for 3D volumes
            data_slice = data_slice[np.newaxis, ...]
            if self.verbose: print(f"  Added temporary channel dim for normalization: {data_slice.shape}")

        if self.normalization_scheme == 'instance_zscore':
            for c in range(data_slice.shape[0]):  # Iterate over channels (or the single pseudo-channel)
                mean = np.mean(data_slice[c])
                std = np.std(data_slice[c])
                # Epsilon prevents division by zero or near-zero std dev
                data_slice[c] = (data_slice[c] - mean) / max(std, 1e-8)
            if self.verbose: print(f"  Applied instance Z-score normalization.")

        elif self.normalization_scheme == 'global_zscore':
            # Assuming global_mean/std are single floats. Adapt if they are per-channel arrays.
            if self.global_mean is None or self.global_std is None:
                raise ValueError(
                    "Internal Error: global_mean/std missing for global_zscore.")  # Should be caught in init
            data_slice = (data_slice - self.global_mean) / max(self.global_std, 1e-8)
            if self.verbose: print(
                f"  Applied global Z-score (mean={self.global_mean:.4f}, std={self.global_std:.4f}).")

        elif self.normalization_scheme == 'instance_minmax':
            for c in range(data_slice.shape[0]):
                min_val = np.min(data_slice[c])
                max_val = np.max(data_slice[c])
                denominator = max(max_val - min_val, 1e-8)  # Epsilon for stability
                data_slice[c] = (data_slice[c] - min_val) / denominator
            if self.verbose: print(f"  Applied instance Min-Max scaling to [0, 1].")

        elif self.normalization_scheme != 'none':
            raise ValueError(f"Internal Error: Unknown normalization scheme '{self.normalization_scheme}' encountered.")

        # Remove temporary channel dimension if it was added
        if not has_channel_dim and original_ndim == 3 and data_slice.ndim == 4:
            data_slice = data_slice[0, ...]
            if self.verbose: print(f"  Removed temporary channel dim: {data_slice.shape}")

        # 3. Apply Final Type Conversion (return_as_type)
        final_dtype = data_slice.dtype  # Start with the current dtype (likely float32 if normalized)

        if self.return_as_type != 'none':
            try:
                # Convert string like 'np.float32' to actual numpy dtype
                target_dtype_str = self.return_as_type.replace('np.', '')
                target_dtype = getattr(np, target_dtype_str)

                if np.issubdtype(target_dtype, np.integer):
                    # Handle conversion to integer types
                    if self.normalization_scheme in ['instance_minmax']:  # Data is in [0, 1] range
                        max_target_val = get_max_value(target_dtype)
                        # Scale to target range, clip just in case due to float precision
                        data_slice = np.clip(data_slice * max_target_val, 0, max_target_val)
                        final_dtype = target_dtype
                        if self.verbose: print(f"  Scaled [0,1] data to target integer {target_dtype_str}.")
                    elif self.normalization_scheme in ['instance_zscore', 'global_zscore']:
                        # WARNING: Converting Z-scored data to integer is lossy and non-standard.
                        # We will NOT change the dtype here, keep it float.
                        print(f"  Warning: Requesting integer type ({target_dtype_str}) after Z-score normalization. "
                              f"Output remains {final_dtype} to avoid data loss. Adjust 'return_as_type' or normalization scheme if needed.")
                        # final_dtype remains float32
                    else:  # Normalization was 'none'
                        # Allow direct casting if no normalization occurred
                        final_dtype = target_dtype
                        if self.verbose: print(f"  Casting non-normalized data to target integer {target_dtype_str}.")

                elif np.issubdtype(target_dtype, np.floating):
                    # Cast to target float type (e.g., float16)
                    final_dtype = target_dtype
                    if self.verbose: print(f"  Casting data to target float {target_dtype_str}.")
                else:
                    # Handle other types if necessary (e.g., bool) - less common
                    final_dtype = target_dtype
                    if self.verbose: print(f"  Casting data to target type {target_dtype_str}.")

                # Perform the actual cast if the final_dtype changed or needs casting
                if data_slice.dtype != final_dtype:
                    data_slice = data_slice.astype(final_dtype)
                    if self.verbose: print(f"  Final cast to {final_dtype} performed.")

            except AttributeError:
                print(
                    f"  Warning: Invalid numpy type string in return_as_type: '{self.return_as_type}'. Skipping final type conversion.")
            except Exception as e:
                print(f"  Warning: Error during final type conversion to {self.return_as_type}: {e}. Skipping.")

        # 4. Convert to Tensor (if requested)
        if self.return_as_tensor:
            try:
                # Ensure data is contiguous for PyTorch
                data_slice = np.ascontiguousarray(data_slice)
                data_slice = torch.from_numpy(data_slice)
                if self.verbose: print(f"  Converted final NumPy array to torch.Tensor.")
            except Exception as e:
                print(f"  Error converting NumPy array to PyTorch Tensor: {e}")
                # Decide how to handle - maybe return numpy array instead?
                # For now, let the error propagate if conversion fails.
                raise

        return data_slice

    def grab_canonical_energy(self) -> Optional[int]:
        """Gets the default energy for a given scroll ID."""
        # Ensure scroll_id is comparable
        scroll_id_key = str(self.scroll_id) if self.scroll_id is not None else None

        energy_mapping = {
            "1": 54, "1b": 54, "2": 54, "2b": 54, "2c": 88,
            "3": 53, "4": 88, "5": 53
        }
        return energy_mapping.get(scroll_id_key)

    def grab_canonical_resolution(self) -> Optional[float]:
        """Gets the default resolution for a given scroll ID."""
        # Ensure scroll_id is comparable
        scroll_id_key = str(self.scroll_id) if self.scroll_id is not None else None

        resolution_mapping = {
            "1": 7.91, "1b": 7.91, "2": 7.91, "2b": 7.91, "2c": 7.91,
            "3": 3.24, "4": 3.24, "5": 7.91
        }
        return resolution_mapping.get(scroll_id_key)


    def shape(self, subvolume_idx: int = 0) -> Tuple[int, ...]:
        """Gets the shape of a specific sub-volume (resolution level)."""
        # Handle the case when self.data is a zarr array directly (from _init_from_zarr_path)
        if isinstance(self.data, zarr.Array):
            return tuple(self.data.shape)
        
        # Original behavior for when self.data is a list of resolution levels
        if not (0 <= subvolume_idx < len(self.data)):
            raise IndexError(f"Invalid subvolume index: {subvolume_idx}. Available: 0 to {len(self.data) - 1}")
        return tuple(self.data[subvolume_idx].shape)

    @property
    def ndim(self, subvolume_idx: int = 0) -> int:
        """Gets the number of dimensions of a specific sub-volume."""
        # Handle the case when self.data is a zarr array directly (from _init_from_zarr_path)
        if isinstance(self.data, zarr.Array):
            return self.data.ndim
            
        # Original behavior for when self.data is a list of resolution levels
        if not (0 <= subvolume_idx < len(self.data)):
            raise IndexError(f"Invalid subvolume index: {subvolume_idx}. Available: 0 to {len(self.data) - 1}")
        return self.data[subvolume_idx].ndim


class Cube:
    """
    A class to represent a 3D instance annotated cube within a scroll.

    Attributes
    ----------
    scroll_id : int
        ID of the scroll.
    energy : int
        Energy value associated with the cube.
    resolution : float
        Resolution of the cube.
    z : int
        Z-coordinate of the cube.
    y : int
        Y-coordinate of the cube.
    x : int
        X-coordinate of the cube.
    cache : bool
        Indicates if caching is enabled.
    cache_dir : Optional[os.PathLike]
        Directory where cached files are stored.
    normalize : bool
        Indicates if the data should be normalized.
    configs : str
        Path to the configuration file.
    volume_url : str
        URL to access the volume data.
    mask_url : str
        URL to access the mask data.
    volume : NDArray
        Loaded volume data.
    mask : NDArray
        Loaded mask data.
    max_dtype : Union[float, int]
        Maximum value of the dtype if normalization is enabled.
    """
    def __init__(self, scroll_id: int, energy: int, resolution: float, z: int, y: int, x: int, cache: bool = False, cache_dir : Optional[os.PathLike] = None, normalize: bool = False) -> None:
        """
        Initialize the Cube object.

        Parameters
        ----------
        scroll_id : int
            ID of the scroll.
        energy : int
            Energy value associated with the cube.
        resolution : float
            Resolution of the cube.
        z : int
            Z-coordinate of the cube.
        y : int
            Y-coordinate of the cube.
        x : int
            X-coordinate of the cube.
        cache : bool, default = False
            Indicates if caching is enabled.
        cache_dir : Optional[os.PathLike], default = None
            Directory where cached files are stored. If None the files will be saved in $HOME / vesuvius / annotated-instances
        normalize : bool, default = False
            Indicates if the data should be normalized.

        Raises
        ------
        ValueError
            If the URL cannot be found in the configuration.
        """
        self.scroll_id = scroll_id
        install_path = get_installation_path()
        self.configs = os.path.join(install_path, 'setup', 'configs', f'cubes.yaml')
        self.energy = energy
        self.resolution = resolution
        self.z, self.y, self.x = z, y, x
        self.volume_url, self.mask_url = self.get_url_from_yaml()
        self.aws = is_aws_ec2_instance()
        if self.aws is False:
            self.cache = cache
            if self.cache:
                if cache_dir is not None:
                    self.cache_dir = Path(cache_dir)
                else:
                    self.cache_dir = Path.home() / 'vesuvius' / 'annotated-instances'
                os.makedirs(self.cache_dir, exist_ok=True)
        self.normalize = normalize

        self.volume, self.mask = self.load_data()

        if self.normalize:
            self.max_dtype = get_max_value(self.volume.dtype)
        
    def get_url_from_yaml(self) -> str:
        """
        Retrieve the URLs for the volume and mask data from the YAML configuration file.

        Returns
        -------
        Tuple[str, str]
            The URLs for the volume and mask data.

        Raises
        ------
        ValueError
            If the URLs cannot be found in the configuration.
        """
        # Load the YAML file
        with open(self.configs, 'r') as file:
            data: Dict[str, Any] = yaml.safe_load(file)
        
        # Retrieve the URL for the given id, energy, and resolution
        base_url: str = data.get(self.scroll_id, {}).get(self.energy, {}).get(self.resolution, {}).get(f"{self.z:05d}_{self.y:05d}_{self.x:05d}")
        if base_url is None:
                raise ValueError("URL not found.")

        volume_filename = f"{self.z:05d}_{self.y:05d}_{self.x:05d}_volume.nrrd"
        mask_filename = f"{self.z:05d}_{self.y:05d}_{self.x:05d}_mask.nrrd"

        volume_url = os.path.join(base_url, volume_filename)
        mask_url = os.path.join(base_url, mask_filename)
        return volume_url, mask_url
    
    def load_data(self) -> Tuple[NDArray, NDArray]:
        """
        Load the data for the cube.

        Returns
        -------
        Tuple[NDArray, NDArray]
            A tuple containing the loaded volume and mask data.

        Raises
        ------
        requests.RequestException
            If there is an error downloading the data from the server.
        """
        output = []
        for url in [self.volume_url, self.mask_url]:
            if self.aws:
                array, _ = nrrd.read(url)
            else:
                if self.cache:
                    # Extract the relevant path after "instance-labels"
                    path_after_finished_cubes = url.split('instance-labels/')[1]
                    # Extract the directory structure and the filename
                    dir_structure, filename = os.path.split(path_after_finished_cubes)

                    # Create the full directory path in the temp_dir
                    full_temp_dir_path = os.path.join(self.cache_dir, dir_structure)

                    # Make sure the directory structure exists
                    os.makedirs(full_temp_dir_path, exist_ok=True)

                    # Create the full path for the temporary file
                    temp_file_path = os.path.join(full_temp_dir_path, filename)

                    # Check if the file already exists in the cache
                    if os.path.exists(temp_file_path):
                        # Read the NRRD file from the cache
                        array, _ = nrrd.read(temp_file_path)

                    else:
                        # Download the remote file
                        response = requests.get(url)
                        response.raise_for_status()  # Ensure we notice bad responses
                        # Write the downloaded content to the temporary file with the same directory structure and filename
                        with open(temp_file_path, 'wb') as tmp_file:
                            tmp_file.write(response.content)

                            array, _ = nrrd.read(temp_file_path)

                else:
                    response = requests.get(url)
                    response.raise_for_status()  # Ensure we notice bad responses
                    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                        tmp_file.write(response.content)
                        temp_file_path = tmp_file.name
                        # Read the NRRD file from the temporary file

                        array, _ = nrrd.read(temp_file_path)

            output.append(array)

        return output[0], output[1]


    def __getitem__(self, idx: Tuple[int, ...]) -> NDArray:
        """
        Get a slice of the cube data.

        Parameters
        ----------
        idx : Tuple[int, ...]
            A tuple representing the coordinates (z, y, x) within the cube.

        Returns
        -------
        NDArray
            The selected data slice.

        Raises
        ------
        IndexError
            If the index is invalid.
        """
        if isinstance(idx, tuple) and len(idx) == 3:
            zz, yy, xx = idx

            if self.normalize:
                return self.volume[zz, yy, xx]/self.max_dtype, self.mask[zz, yy, xx]
            
            else:
                return self.volume[zz, yy, xx], self.mask[zz, yy, xx]
            
        else:
            raise IndexError("Invalid index. Must be a tuple of three elements.")
    
    def activate_caching(self, cache_dir: Optional[os.PathLike] = None) -> None:
        """
        Activate caching for the cube data.

        Parameters
        ----------
        cache_dir : Optional[os.PathLike], default = None
            Directory where cached files are stored.
        """
        if not self.cache:
            if cache_dir is None:
                self.cache_dir = Path.home() / 'vesuvius' / 'annotated-instances'
            else:
                self.cache_dir = Path(cache_dir)
            self.cache = True
            self.volume, self.mask = self.load_data()

    def deactivate_caching(self) -> None:
        """
        Deactivate caching for the cube data.
        """
        if self.cache:
            self.cache = False
            self.volume, self.mask = self.load_data()