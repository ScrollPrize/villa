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
import tensorstore as ts
from .utils import get_max_value

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
    cache : bool
        Indicates if TensorStore caching is enabled.
    cache_pool : int
        Size of the TensorStore cache pool in bytes.
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
    data : List[ts.TensorStore]
        List of TensorStore objects representing volume data (potentially multi-resolution).
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
                 cache: bool = True, cache_pool: int = 1e10,
                 format: str = 'zarr',  # Currently only zarr via TensorStore is fully implemented here
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
        cache : bool, default = True
            Enable TensorStore caching.
        cache_pool : int, default = 1e10
            TensorStore cache size in bytes.
        format : str, default = 'zarr'
            Data format (currently only 'zarr' via TensorStore).
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
        self.cache = cache
        self.cache_pool = cache_pool
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
                if type.startswith("scroll") and len(type) > 6:  # e.g., "scroll1", "scroll1b"
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
                        f"Invalid 'type' string: {type}. Expected 'scroll', 'segment', 'scrollX', 'zarr', or segment timestamp.")
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
            self.data = self.load_data()  # Loads TensorStore objects
            self.dtype = self.data[0].dtype.numpy_dtype  # Get original dtype from TensorStore

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
        cache_pool_bytes = int(self.cache_pool) if self.cache else 0
        is_http = self.path.startswith(('http://', 'https://'))

        kvstore_driver = 'http' if is_http else 'file'
        kvstore_spec = {'driver': kvstore_driver}
        if is_http:
            kvstore_spec['base_url'] = self.path
        else:
            kvstore_spec['path'] = self.path

        ts_config_base = {
            'driver': 'zarr',
            'kvstore': kvstore_spec,
            'context': {'cache_pool': {'total_bytes_limit': cache_pool_bytes}}
        }

        if self.verbose:
            print(f"Opening zarr store with TensorStore at path: {self.path}")
            print(f"Base TensorStore config: {json.dumps(ts_config_base, indent=2)}")

        opened_stores = []
        try:
            # Try opening root level first
            if self.verbose: print("Attempting to open Zarr at root level...")
            future = ts.open(ts_config_base)
            root_store = future.result()
            opened_stores.append(root_store)
            if self.verbose: print("Successfully opened Zarr at root level.")
            self.url = self.path.rstrip("/")  # Set url attribute

        except Exception as root_e:
            if self.verbose: print(f"Failed opening root level: {root_e}. Checking for multi-resolution...")
            # Try opening as multi-resolution (NGFF spec with datasets in subgroups 0, 1, ...)
            try:
                # Try opening level '0'
                ts_config_level0 = ts_config_base.copy()
                ts_config_level0['path'] = '0'  # Specify the group path within the zarr store
                if self.verbose: print(f"Attempting to open Zarr at path '0'...")
                future0 = ts.open(ts_config_level0)
                store0 = future0.result()
                opened_stores.append(store0)
                if self.verbose: print("Successfully opened level '0'. Checking for further levels...")
                self.url = self.path.rstrip("/")  # Set url attribute

                # Try opening subsequent levels
                for level in range(1, 6):  # Check levels 1 to 5
                    ts_config_level = ts_config_base.copy()
                    ts_config_level['path'] = str(level)
                    try:
                        if self.verbose: print(f"Attempting to open Zarr at path '{level}'...")
                        future_level = ts.open(ts_config_level)
                        store_level = future_level.result()
                        opened_stores.append(store_level)
                        if self.verbose: print(f"Successfully opened level '{level}'.")
                    except Exception as level_e:
                        if self.verbose: print(
                            f"Level '{level}' not found or error opening: {level_e}. Stopping search.")
                        break  # Stop searching if a level is missing

            except Exception as multi_e:
                if self.verbose: print(f"Failed opening as multi-resolution: {multi_e}")
                # If both root and multi-resolution failed, re-raise the original root error
                raise root_e from multi_e

        if not opened_stores:
            raise RuntimeError(f"Could not open Zarr store at {self.path} either as root or multi-resolution.")

        self.data = opened_stores
        self.dtype = self.data[0].dtype.numpy_dtype

        # Load metadata (.zattrs)
        try:
            self.metadata = self.load_ome_metadata()  # Use existing method, adapted for direct path
        except Exception as meta_e:
            print(f"Warning: Could not load .zattrs metadata from {self.path}: {meta_e}")
            self.metadata = {}  # Assign empty dict if metadata loading fails

        # Set remaining attributes for consistency if not already set
        if not hasattr(self, 'type'):
            self.type = "zarr"
        self.scroll_id = None
        self.segment_id = None
        self.domain = "local" if not is_http else "dl.ash2txt"
        self.resolution = None  # Resolution might be in metadata, but not set directly here
        self.energy = None

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
        """Loads OME-Zarr metadata (.zattrs)."""
        # Determine the base URL/path correctly, handling direct path or config-derived URL
        base_path = self.path if self.type == 'zarr' and self.path else self.url
        if not base_path:
            raise ValueError("Could not determine base path/URL for metadata loading.")

        base_path = base_path.rstrip("/")
        is_http = base_path.startswith(('http://', 'https://'))

        potential_zattrs_paths = [
            ".zattrs",  # Standard location at root
            "0/.zattrs"  # Common location for first level in multi-resolution
        ]

        for relative_path in potential_zattrs_paths:
            if is_http:
                zattrs_url = f"{base_path}/{relative_path}"
                if self.verbose: print(f"Attempting to load metadata from URL: {zattrs_url}")
                try:
                    response = requests.get(zattrs_url, timeout=10)  # Add timeout
                    response.raise_for_status()
                    zattrs_content = response.json()
                    if self.verbose: print(f"Successfully loaded metadata from {zattrs_url}")
                    # OME-Zarr metadata is usually under 'multiscales', etc.
                    # Wrap it for consistency, though the direct content might be more useful.
                    return {"zattrs": zattrs_content}
                except requests.exceptions.RequestException as e:
                    if self.verbose: print(f"Failed to load {zattrs_url}: {e}")
                except json.JSONDecodeError as e:
                    if self.verbose: print(f"Failed to parse JSON from {zattrs_url}: {e}")

            else:  # Local file system
                zattrs_file_path = os.path.join(base_path, relative_path)
                if self.verbose: print(f"Attempting to load metadata from file: {zattrs_file_path}")
                if os.path.exists(zattrs_file_path):
                    try:
                        with open(zattrs_file_path, 'r') as f:
                            zattrs_content = json.load(f)
                        if self.verbose: print(f"Successfully loaded metadata from {zattrs_file_path}")
                        return {"zattrs": zattrs_content}
                    except json.JSONDecodeError as e:
                        if self.verbose: print(f"Failed to parse JSON from {zattrs_file_path}: {e}")
                    except Exception as e:
                        if self.verbose: print(f"Error reading {zattrs_file_path}: {e}")
                else:
                    if self.verbose: print(f"File not found: {zattrs_file_path}")

        # If loop completes without returning, metadata wasn't found
        print(f"Warning: Could not load .zattrs metadata from base path {base_path} at standard locations.")
        return {}  # Return empty dict if no metadata found

    def load_data(self) -> List[ts.TensorStore]:
        """Loads data using TensorStore based on metadata."""
        # This method relies on metadata having been loaded first
        if not self.metadata or 'zattrs' not in self.metadata or 'multiscales' not in self.metadata['zattrs']:
            # If standard OME metadata structure is missing, try to load from base url/path directly
            # This covers simple Zarr stores without explicit multiscale metadata
            if self.verbose:
                print("OME metadata structure missing, attempting direct TensorStore open on base path.")
            try:
                # Use the logic from _init_from_zarr_path to open potentially multi-res stores
                # This is slightly redundant but ensures data loading works even without perfect metadata
                base_path = self.path if self.type == 'zarr' and self.path else self.url
                if not base_path: raise ValueError("Base path/URL missing.")

                cache_pool_bytes = int(self.cache_pool) if self.cache else 0
                is_http = base_path.startswith(('http://', 'https://'))
                kvstore_driver = 'http' if is_http else 'file'
                kvstore_spec = {'driver': kvstore_driver}
                if is_http:
                    kvstore_spec['base_url'] = base_path
                else:
                    kvstore_spec['path'] = base_path

                ts_config_base = {
                    'driver': 'zarr', 'kvstore': kvstore_spec,
                    'context': {'cache_pool': {'total_bytes_limit': cache_pool_bytes}}
                }
                # Try opening levels 0 to 5, similar to _init_from_zarr_path
                stores = []
                # Try root first
                try:
                    future = ts.open(ts_config_base);
                    stores.append(future.result())
                    if self.verbose: print("Opened data directly from root path.")
                    return stores
                except Exception:
                    pass  # Ignore if root fails, try levels

                # Try levels
                for level in range(6):  # Check levels 0 to 5
                    ts_config_level = ts_config_base.copy()
                    ts_config_level['path'] = str(level)
                    try:
                        future_level = ts.open(ts_config_level)
                        store_level = future_level.result()
                        stores.append(store_level)
                        if self.verbose: print(f"Opened data from path '{level}'.")
                    except Exception:
                        if level == 0:  # If level 0 fails, unlikely others will succeed
                            break
                        else:  # Stop if a higher level fails after finding lower ones
                            break
                if stores:
                    if self.verbose: print(f"Found {len(stores)} resolution levels.")
                    return stores
                else:  # If nothing opened
                    raise RuntimeError(f"Could not open data store at {base_path} directly or via levels 0-5.")

            except Exception as e:
                print(f"Error loading data directly with TensorStore: {e}")
                raise RuntimeError(f"Failed to load data: OME metadata missing and direct load failed.") from e

        # --- Load based on OME multiscales metadata ---
        sub_volumes = []
        base_url = self.url.rstrip("/")  # Assumes URL was set correctly
        is_http = base_url.startswith(('http://', 'https://'))

        # Check if multiscales exist and is a list
        multiscales_data = self.metadata['zattrs'].get('multiscales')
        if not isinstance(multiscales_data, list) or not multiscales_data:
            raise ValueError("Invalid or missing 'multiscales' data in metadata.")

        # Assume first multiscale entry, standard OME-Zarr
        datasets = multiscales_data[0].get('datasets')
        if not isinstance(datasets, list):
            raise ValueError("Invalid or missing 'datasets' list within multiscales metadata.")

        for dataset_info in datasets:
            path_suffix = dataset_info.get('path')
            if path_suffix is None:
                print(f"Warning: Dataset entry missing 'path': {dataset_info}. Skipping.")
                continue

            # Construct the full path/URL to the dataset level
            # Important: For TensorStore Zarr driver, kvstore path/base_url points to the *root*
            # of the Zarr store, and the 'path' parameter in the Zarr spec points to the *group*
            # within the store (e.g., '0', '1').

            cache_pool_bytes = int(self.cache_pool) if self.cache else 0
            context = {'cache_pool': {'total_bytes_limit': cache_pool_bytes}}

            kvstore_driver = 'http' if is_http else 'file'
            kvstore_spec = {'driver': kvstore_driver}
            if is_http:
                kvstore_spec['base_url'] = base_url
            else:
                kvstore_spec['path'] = base_url  # Local file path

            ts_config = {
                'driver': 'zarr',
                'kvstore': kvstore_spec,
                'path': path_suffix,  # Specifies the group ('0', '1', etc.)
                'context': context,
            }

            if self.verbose:
                print(f"Attempting TensorStore open for dataset path: '{path_suffix}'")
                print(f"TensorStore config: {json.dumps(ts_config, indent=2)}")

            try:
                future = ts.open(ts_config)
                store = future.result()
                sub_volumes.append(store)
                if self.verbose:
                    print(
                        f"Successfully loaded data for path '{path_suffix}'. Shape: {store.shape}, Dtype: {store.dtype}")

            except Exception as e:
                print(f"ERROR loading data for path '{path_suffix}' with TensorStore: {e}")
                # Decide whether to continue or fail hard
                if not sub_volumes:  # If even the first level failed
                    raise RuntimeError(
                        f"Failed to load the base resolution level '{path_suffix}'. Cannot continue.") from e
                else:
                    print(f"Stopping data loading after failure. Using {len(sub_volumes)} successfully loaded levels.")
                    break  # Stop trying further levels

        if not sub_volumes:
            raise RuntimeError("Could not load any data levels based on the provided metadata.")

        return sub_volumes

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
        # Assuming segment URL might be like ".../20230827161847" or ".../20230827161847/"
        # We want to replace the segment ID part with "_inklabels.png" at the same level
        parent_url = os.path.dirname(base_url)
        # Extract segment ID (timestamp) as the last part of the original URL
        segment_id_str = os.path.basename(base_url)
        # Construct potential ink label filename
        inklabel_filename = f"{segment_id_str}_inklabels.png"  # Adjust if naming differs
        inklabel_url_or_path = os.path.join(parent_url, inklabel_filename)

        if self.verbose:
            print(f"Attempting to load ink label from: {inklabel_url_or_path}")

        is_http = inklabel_url_or_path.startswith(('http://', 'https://'))

        try:
            if is_http:
                response = requests.get(inklabel_url_or_path, timeout=10)
                response.raise_for_status()
                img = Image.open(BytesIO(response.content))
                # Save the downloaded image if a save path is provided
                if save_path:
                    img.save(save_path)
                    print(f"Saved ink label to: {save_path}")
                # Convert to grayscale if it's not already L mode
                if img.mode != 'L':
                    img = img.convert('L')
                self.inklabel = np.array(img)
            else:  # Local file path
                if not os.path.exists(inklabel_url_or_path):
                    raise FileNotFoundError(f"Inklabel file not found: {inklabel_url_or_path}")
                img = Image.open(inklabel_url_or_path)
                # Save the loaded image if a save path is provided
                if save_path:
                    img.save(save_path)
                    print(f"Saved ink label to: {save_path}")
                if img.mode != 'L':
                    img = img.convert('L')
                self.inklabel = np.array(img)

            if self.verbose:
                print(f"Successfully loaded ink label with shape: {self.inklabel.shape}, dtype: {self.inklabel.dtype}")

        except Exception as e:
            print(f"Warning: Could not load ink label from {inklabel_url_or_path}: {e}")
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

        Indexing follows NumPy/TensorStore conventions. For 3D data, the order is (z, y, x).
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
        if not (0 <= subvolume_idx < len(self.data)):
            raise IndexError(f"Invalid subvolume index: {subvolume_idx}. Must be between 0 and {len(self.data) - 1}.")

        # --- Read Data Slice ---
        if self.verbose:
            print(f"Accessing data level {subvolume_idx} with coordinates: {coord_idx}")
            print(f"  Store shape: {self.data[subvolume_idx].shape}, Store dtype: {self.data[subvolume_idx].dtype}")

        try:
            # TensorStore uses standard NumPy-like slicing
            future = self.data[subvolume_idx][coord_idx].read()
            data_slice = future.result()
            # Ensure it's a NumPy array for subsequent processing
            data_slice = np.array(data_slice)
            original_dtype = data_slice.dtype  # Store for potential later use

            if self.verbose:
                print(f"  Read slice shape: {data_slice.shape}, dtype: {data_slice.dtype}")

        except Exception as e:
            print(f"ERROR during TensorStore read operation:")
            print(f"  Subvolume: {subvolume_idx}, Index: {coord_idx}")
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

    def activate_caching(self) -> None:
        """Activates TensorStore caching and reloads data if necessary."""
        if not self.cache:
            if self.verbose: print("Activating caching...")
            self.cache = True
            # Reload data only if it was potentially loaded without cache active
            # Re-initializing TensorStore with cache context is needed
            try:
                if self.type == 'zarr' and self.path:
                    self._init_from_zarr_path()  # Re-init with cache active
                else:
                    # For config-based, need to re-run the loading sequence
                    self.metadata = self.load_ome_metadata()
                    self.data = self.load_data()
                    self.dtype = self.data[0].dtype.numpy_dtype
                    if self.type == "segment": self.download_inklabel()
                if self.verbose: print("Caching activated. Data handles potentially updated.")
            except Exception as e:
                print(f"Error reactivating cache and reloading data: {e}")
                self.cache = False  # Revert state if reload failed

    def deactivate_caching(self) -> None:
        """Deactivates TensorStore caching and reloads data."""
        if self.cache:
            if self.verbose: print("Deactivating caching...")
            self.cache = False
            self.cache_pool = 0  # Ensure pool size is 0
            # Re-initializing TensorStore without cache context is needed
            try:
                if self.type == 'zarr' and self.path:
                    self._init_from_zarr_path()  # Re-init with cache disabled
                else:
                    # For config-based, re-run loading sequence
                    self.metadata = self.load_ome_metadata()
                    self.data = self.load_data()
                    self.dtype = self.data[0].dtype.numpy_dtype
                    if self.type == "segment": self.download_inklabel()
                if self.verbose: print("Caching deactivated. Data handles potentially updated.")
            except Exception as e:
                print(f"Error deactivating cache and reloading data: {e}")
                self.cache = True  # Revert state if reload failed

    def shape(self, subvolume_idx: int = 0) -> Tuple[int, ...]:
        """Gets the shape of a specific sub-volume (resolution level)."""
        if not (0 <= subvolume_idx < len(self.data)):
            raise IndexError(f"Invalid subvolume index: {subvolume_idx}. Available: 0 to {len(self.data) - 1}")
        return tuple(self.data[subvolume_idx].shape)

    @property
    def ndim(self, subvolume_idx: int = 0) -> int:
        """Gets the number of dimensions of a specific sub-volume."""
        if not (0 <= subvolume_idx < len(self.data)):
            raise IndexError(f"Invalid subvolume index: {subvolume_idx}. Available: 0 to {len(self.data) - 1}")
        return self.data[subvolume_idx].ndim


# --- Cube Class (largely unchanged as requested, added placeholder get_max_value) ---

class Cube:
    """
    Represents a 3D annotated cube (volume + mask) loaded from NRRD files.
    (Note: Normalization logic here is simpler than the Volume class).
    """

    def __init__(self, scroll_id: int, energy: int, resolution: float, z: int, y: int, x: int, cache: bool = False,
                 cache_dir: Optional[os.PathLike] = None, normalize: bool = False, verbose: bool = False) -> None:
        """
        Initializes the Cube object.
        Parameters follow original structure. Added verbose flag.
        """
        self.scroll_id = scroll_id
        self.energy = energy
        self.resolution = resolution
        self.z, self.y, self.x = z, y, x
        self.cache = cache
        self.normalize = normalize  # Keep simple normalize flag for Cube
        self.verbose = verbose

        # Config file path resolution
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        possible_paths = [
            os.path.join(base_dir, 'setup', 'configs', f'cubes.yaml'),
            os.path.join(base_dir, 'configs', f'cubes.yaml')
        ]
        self.configs = None
        for config_path in possible_paths:
            if os.path.exists(config_path):
                self.configs = config_path
                break
        if self.configs is None:
            self.configs = possible_paths[0]
            if self.verbose: print(f"Warning: Cube config not found, using default path: {self.configs}")

        # Cache directory setup
        self.aws = is_aws_ec2_instance()
        self.cache_dir = None
        if not self.aws and self.cache:
            if cache_dir is not None:
                self.cache_dir = Path(cache_dir)
            else:
                self.cache_dir = Path.home() / 'vesuvius' / 'annotated-instances'
            try:
                os.makedirs(self.cache_dir, exist_ok=True)
                if self.verbose: print(f"Using cache directory: {self.cache_dir}")
            except OSError as e:
                print(f"Warning: Could not create cache directory {self.cache_dir}: {e}. Disabling cache.")
                self.cache = False

        # Get URLs and load data
        try:
            self.volume_url, self.mask_url = self.get_url_from_yaml()
            if self.verbose:
                print(f"Cube Volume URL: {self.volume_url}")
                print(f"Cube Mask URL: {self.mask_url}")

            self.volume, self.mask = self.load_data()

            self.max_dtype = None
            if self.normalize:
                # Use the actual dtype of the loaded volume
                self.max_dtype = get_max_value(self.volume.dtype)
                if self.verbose: print(f"Normalization enabled for Cube. Max dtype value: {self.max_dtype}")

        except Exception as e:
            print(f"Error initializing Cube ({scroll_id}, {z}_{y}_{x}): {e}")
            raise

    def get_url_from_yaml(self) -> Tuple[str, str]:
        """Retrieves volume and mask URLs from the cubes YAML config."""
        if not self.configs or not os.path.exists(self.configs):
            raise FileNotFoundError(f"Cube configuration file not found at {self.configs}")

        try:
            with open(self.configs, 'r') as file:
                data: Dict = yaml.safe_load(file)
            if not data:
                raise ValueError(f"Cube config file {self.configs} is empty or invalid.")

            # Navigate config: scroll -> energy -> resolution -> cube_id
            cube_id_str = f"{self.z:05d}_{self.y:05d}_{self.x:05d}"

            # Try both numeric and string keys for the hierarchical lookup
            scroll_data = data.get(self.scroll_id, data.get(str(self.scroll_id), {}))
            energy_data = scroll_data.get(self.energy, scroll_data.get(str(self.energy), {}))
            resolution_data = energy_data.get(self.resolution, energy_data.get(str(self.resolution), {}))
            base_url = resolution_data.get(cube_id_str)

            if base_url is None:
                raise ValueError(
                    f"URL not found in config for cube: scroll={self.scroll_id}, E={self.energy}, R={self.resolution}, cube={cube_id_str}")

            # Construct full URLs (assuming NRRD files are in the base_url directory)
            # Ensure no double slashes if base_url already ends with one
            base_url = base_url.rstrip('/')
            volume_filename = f"{cube_id_str}_volume.nrrd"
            mask_filename = f"{cube_id_str}_mask.nrrd"

            # Use os.path.join for robustness, even for URLs
            volume_url = f"{base_url}/{volume_filename}"
            mask_url = f"{base_url}/{mask_filename}"

            return volume_url, mask_url

        except Exception as e:
            print(f"Error processing cube config file {self.configs}: {e}")
            raise

    def load_data(self) -> Tuple[NDArray, NDArray]:
        """Loads volume and mask data (NRRD), using cache if enabled."""
        loaded_arrays = []
        for url, name in [(self.volume_url, "volume"), (self.mask_url, "mask")]:
            array = None
            cache_file_path = None

            # Determine cache file path if caching is active
            if self.cache and self.cache_dir:
                try:
                    # Create a unique-ish path based on the URL structure after a common root
                    # This might need refinement depending on URL patterns
                    relative_path = url.split('instance-labels/')[-1] if 'instance-labels/' in url else url.split('/')[
                                                                                                        -3:]  # Heuristic
                    relative_path = os.path.join(*relative_path)  # Rejoin parts in case split gave multiple
                    cache_file_path = self.cache_dir / Path(relative_path)
                    os.makedirs(cache_file_path.parent, exist_ok=True)
                    if self.verbose: print(f"Cache path for {name}: {cache_file_path}")
                except Exception as e:
                    print(f"Warning: Error creating cache path for {url}: {e}. Caching for this file might fail.")
                    cache_file_path = None  # Disable caching for this file if path creation fails

            # Try loading from cache first
            if cache_file_path and cache_file_path.exists():
                try:
                    if self.verbose: print(f"Loading {name} from cache: {cache_file_path}")
                    array, _ = nrrd.read(str(cache_file_path))
                except Exception as e:
                    print(
                        f"Warning: Failed to read {name} from cache file {cache_file_path}: {e}. Attempting download.")
                    array = None  # Ensure download happens

            # If not loaded from cache, download or read directly
            if array is None:
                is_http = url.startswith(('http://', 'https://'))
                if is_http:  # Download required
                    if self.verbose: print(f"Downloading {name} from: {url}")
                    try:
                        response = requests.get(url, timeout=30)  # Increased timeout for potentially large files
                        response.raise_for_status()
                        # Read NRRD from downloaded bytes
                        with tempfile.NamedTemporaryFile(suffix=".nrrd", delete=False) as tmp_file:
                            tmp_file.write(response.content)
                            temp_nrrd_path = tmp_file.name
                        try:
                            array, _ = nrrd.read(temp_nrrd_path)
                            # Save to cache if download successful and caching enabled
                            if cache_file_path:
                                os.rename(temp_nrrd_path, cache_file_path)  # Move temp file to cache
                                if self.verbose: print(f"Saved downloaded {name} to cache: {cache_file_path}")
                            else:
                                os.remove(temp_nrrd_path)  # Clean up temp file if not caching
                        except Exception as read_e:
                            os.remove(temp_nrrd_path)  # Clean up temp file on read error
                            raise RuntimeError(f"Failed to read NRRD data downloaded from {url}") from read_e

                    except requests.exceptions.RequestException as req_e:
                        raise RuntimeError(f"Failed to download {name} from {url}") from req_e
                else:  # Local file path (non-HTTP)
                    if self.verbose: print(f"Reading {name} from local path: {url}")
                    if not os.path.exists(url):
                        raise FileNotFoundError(f"Local NRRD file not found for {name}: {url}")
                    try:
                        array, _ = nrrd.read(url)
                        # Optionally copy to cache if needed, but less common for local files
                        # if cache_file_path and url != str(cache_file_path):
                        #    shutil.copy2(url, cache_file_path)

                    except Exception as read_e:
                        raise RuntimeError(f"Failed to read local NRRD file {url}") from read_e

            if array is None:
                # Should not happen if logic above is correct, but safety net
                raise RuntimeError(f"Failed to load data for {name} from {url}")

            loaded_arrays.append(array)

        return loaded_arrays[0], loaded_arrays[1]  # Return volume, mask

    def __getitem__(self, idx: Tuple[Union[int, slice], ...]) -> Tuple[NDArray, NDArray]:
        """Gets a slice of the cube's volume and mask."""
        if not isinstance(idx, tuple) or len(idx) != 3:
            raise IndexError("Invalid index for Cube. Must be a tuple of three elements (z, y, x).")

        # Slicing is applied directly to the loaded numpy arrays
        volume_slice = self.volume[idx]
        mask_slice = self.mask[idx]

        # Apply simple normalization if enabled
        if self.normalize and self.max_dtype is not None and self.max_dtype > 0:
            # Ensure float before division
            volume_slice = volume_slice.astype(np.float32) / self.max_dtype

        return volume_slice, mask_slice

    def activate_caching(self, cache_dir: Optional[os.PathLike] = None) -> None:
        """Activates caching and potentially reloads data."""
        if not self.aws and not self.cache:
            if self.verbose: print("Activating caching for Cube...")
            self.cache = True
            if cache_dir is not None:
                self.cache_dir = Path(cache_dir)
            elif self.cache_dir is None:  # Set default if not set previously
                self.cache_dir = Path.home() / 'vesuvius' / 'annotated-instances'

            # Ensure cache dir exists
            if self.cache_dir:
                try:
                    os.makedirs(self.cache_dir, exist_ok=True)
                    # Force reload to potentially populate cache
                    self.volume, self.mask = self.load_data()
                except OSError as e:
                    print(f"Warning: Could not create cache directory {self.cache_dir}: {e}. Disabling cache.")
                    self.cache = False
            else:
                print("Warning: Cannot activate cache, cache directory not set.")
                self.cache = False

    def deactivate_caching(self) -> None:
        """Deactivates caching (data remains loaded)."""
        if self.cache:
            if self.verbose: print("Deactivating caching for Cube.")
            self.cache = False
            # Data is already in memory, no reload needed, just stops future caching


