import os
import re
import yaml
from typing import Dict, List
from vesuvius.utils import get_installation_path


def update_local_list(base_dir: str, base_dir_cubes: str) -> None:
    install_path = get_installation_path()
    scroll_config = os.path.join(install_path, 'vesuvius', 'configs', 'scrolls.yaml')
    directory_config = os.path.join(install_path, 'vesuvius', 'configs', 'directory_structure.yaml')
    cubes_config = os.path.join(install_path, 'vesuvius', 'configs', 'cubes.yaml')

    #print(f"Starting directory traversal for: {base_dir}")
    
    tree = get_directory_structure(base_dir)
    #print(f"Directory structure: {tree}")

    zarr_files = categorize_zarr_files(tree, base_dir)
    #print(f"Zarr files found: {zarr_files}")

    cubes_folders = list_subfolders(base_dir_cubes)
    #print(f"Cubes subfolders: {cubes_folders}")

    save_yaml(directory_config, tree)
    save_yaml(scroll_config, zarr_files)
    update_cubes_config(cubes_config, cubes_folders, base_dir_cubes)


def get_directory_structure(base_dir: str):
    directory_tree = {}

    for root, dirs, files in os.walk(base_dir):
        dirs_to_remove = []
        for name in dirs:
            dir_path = os.path.relpath(os.path.join(root, name), base_dir)
            #print(f"Encountered directory: {dir_path}")
            if name.endswith('.zarr'):
                directory_tree[dir_path] = None
                #print(f"Identified zarr directory: {dir_path}")
                # Mark this directory to be removed from further traversal
                dirs_to_remove.append(name)
            else:
                directory_tree[dir_path] = {}

        # Remove directories from traversal after checking all dirs
        for dir_to_remove in dirs_to_remove:
            dirs.remove(dir_to_remove)

    #print(f"Final directory tree: {directory_tree}")
    return directory_tree

def categorize_zarr_files(tree, base_dir):
    zarr_files: Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, str]]]]] = {}

    # Regex patterns for matching volume and segment paths
    volume_pattern = re.compile(r'volumes[/\\](?P<intensity>\d+)(?=keV|um)(?P<unit>[a-zA-Z]+)_(?P<resolution>\d+\.\d{2})(?P<suffix>[a-zA-Z]*)\.zarr')
    segment_pattern = re.compile(r'segments[/\\](?P<intensity>\d+)(?=keV|um)(?P<unit>[a-zA-Z]+)_(?P<resolution>\d+\.\d{2})(?P<suffix>[a-zA-Z]*)[/\\](?P<segment_id>[^/\\]+)\.zarr')
    scroll_pattern = re.compile(r'(?P<scrollnumber>\d+)[/\\](?=volumes|segments)(volumes|segments)')

    for key in tree.keys():
        full_path = os.path.join(base_dir, key)
        # Check for scroll number
        scroll_match = scroll_pattern.search(key)
        if not scroll_match:
            continue
        scrollnumber = scroll_match.group('scrollnumber')

        # Initialize scrollnumber in the dictionary if not present
        if scrollnumber not in zarr_files:
            zarr_files[scrollnumber] = {}

        # Check for volume matches
        volume_match = volume_pattern.search(key)
        if volume_match:
            intensity = str(volume_match.group('intensity'))
            resolution = str(volume_match.group('resolution'))

            # Initialize intensity and resolution in the dictionary if not present
            if intensity not in zarr_files[scrollnumber]:
                zarr_files[scrollnumber][intensity] = {}
            if resolution not in zarr_files[scrollnumber][intensity]:
                zarr_files[scrollnumber][intensity][resolution] = {'volume': full_path, 'segments': {}}
            else:
                zarr_files[scrollnumber][intensity][resolution]['volume'] = full_path
            continue

        # Check for segment matches
        segment_match = segment_pattern.search(key)
        if segment_match:
            intensity = str(segment_match.group('intensity'))
            resolution = str(segment_match.group('resolution'))
            segment_id = str(segment_match.group('segment_id'))

            # Initialize intensity and resolution in the dictionary if not present
            if intensity not in zarr_files[scrollnumber]:
                zarr_files[scrollnumber][intensity] = {}
            if resolution not in zarr_files[scrollnumber][intensity]:
                zarr_files[scrollnumber][intensity][resolution] = {'volume': None, 'segments': {}}
            # Add the segment
            zarr_files[scrollnumber][intensity][resolution]['segments'][segment_id] = full_path

    return zarr_files






def list_subfolders(directory: str) -> List[str]:
    """
    List all subfolders in a directory.

    Parameters
    ----------
    directory : str
        The directory to list subfolders from.

    Returns
    -------
    List[str]
        A list of subfolder paths.
    """
    subfolders = []

    for root, dirs, _ in os.walk(directory):
        relative_path = os.path.relpath(root, directory)
        if relative_path == ".":
            relative_path = ""

        for name in dirs:
            dir_path = os.path.join(relative_path, name)
            subfolders.append(dir_path)
            #print(f"Subfolder found: {dir_path}")

    return subfolders


def update_cubes_config(cubes_config: str, cubes_folders: List[str], base_dir_cubes: str) -> None:
    """
    Update the cubes configuration YAML file.

    Parameters
    ----------
    cubes_config : str
        The path to the cubes configuration file.
    cubes_folders : List[str]
        A list of subfolder paths under the cubes base directory.
    base_dir_cubes : str
        The base directory for cubes.

    Returns
    -------
    None
    """
    data = {
        1: {
            54: {
                7.91: {}
            }
        }
    }

    for folder in cubes_folders:
        folder_name = os.path.basename(folder)
        data[1][54][7.91][folder_name] = os.path.join(base_dir_cubes, folder)
        #print(f"Cube folder added to config: {folder_name}")

    save_yaml(cubes_config, data)


def save_yaml(file_path: str, data: Dict) -> None:
    """
    Save data to a YAML file.

    Parameters
    ----------
    file_path : str
        The path to the YAML file.
    data : Dict
        The data to save to the file.

    Returns
    -------
    None
    """
    with open(file_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)
    #print(f"YAML saved: {file_path}")