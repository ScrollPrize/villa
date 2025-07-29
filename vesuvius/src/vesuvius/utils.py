import asyncio
import aiohttp
import yaml
import requests
from vesuvius.setup.accept_terms import get_installation_path
from typing import List, Optional, Dict, Tuple
from vesuvius.data.paths.parser import get_directory_structure, find_zarr_files, list_subfolders
import nest_asyncio
import ssl
import os

async def scrape_website(base_url: str, ignore_list: List[str]) -> Tuple[Dict[str, Optional[Dict]], Dict[str, str]]:
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context), timeout=aiohttp.ClientTimeout(total=60)) as session:
        directory_tree = await get_directory_structure(base_url, session, ignore_list)
        zarr_files = await find_zarr_files(directory_tree, base_url, session)
        return directory_tree, zarr_files

# Define the function to scrape the website and generate the YAML
async def collect_subfolders(base_url: str, ignore_list: List[str]) -> List[str]:
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context), timeout=aiohttp.ClientTimeout(total=60)) as session:
        subfolders = await list_subfolders(base_url, session, ignore_list)
        return subfolders
    
def update_list(base_url: str, base_url_cubes: str, ignore_list: Optional[List[str]] = None) -> None:
    """
    Scrape a website for directory structures and Zarr files, then update the configuration files.

    This function scrapes a given base URL for a directory structure and Zarr files, then saves this data 
    to YAML configuration files. It also updates the list of cubes in a separate configuration file.

    Parameters
    ----------
    base_url : str
        The base URL to scrape for directory structures and Zarr files.
    base_url_cubes : str
        The base URL to scrape for cubes folder structure.
    ignore_list : Optional[List[str]], default = None
        A list of regex patterns to ignore during scraping. If None, defaults to ignoring `.zarr` files.

    Returns
    -------
    None

    Notes
    -----
    - This function makes use of asyncio to scrape websites concurrently.
    - It updates the following YAML configuration files:
      - 'directory_structure.yaml'
      - 'scrolls.yaml'
      - 'cubes.yaml'
    - The part of the function that deals with cubes is currently designed to work with scroll 1 and energy 54 at resolution 7.91, but should 
      be generalized in the future.
    """
    install_path = get_installation_path()
    scroll_config = os.path.join(install_path, 'vesuvius', 'configs', f'scrolls.yaml')
    directory_config = os.path.join(install_path, 'vesuvius', 'configs', f'directory_structure.yaml')
    cubes_config = os.path.join(install_path, 'vesuvius', 'configs', f'cubes.yaml')

    if ignore_list is None:
        ignore_list = [r'\.zarr$']
    
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    if loop.is_running():
        nest_asyncio.apply()
        tree, zarr_files = loop.run_until_complete(scrape_website(base_url, ignore_list))
        cubes_folders = loop.run_until_complete(scrape_website(base_url_cubes, ignore_list))
    else:
        tree, zarr_files = loop.run_until_complete(scrape_website(base_url, ignore_list))
        cubes_folders = loop.run_until_complete(scrape_website(base_url_cubes, ignore_list))

    with open(directory_config, 'w') as file:
        yaml.dump(tree, file, default_flow_style=False)
    
    with open(scroll_config, 'w') as file:
        yaml.dump(zarr_files, file, default_flow_style=False)


    #TODO: implement not only for scroll 1
    #TODO: fix cubes path on website
    data = {
            1: {
                54: {
                    7.91: { }
                }
            }}
    for folder in cubes_folders[0].keys():
        # Extract the name of the first subfolder
        folder_name = folder[:-1]
        data[1][54][7.91][folder_name] = base_url_cubes + folder

    with open(cubes_config, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)
    
    #print("Directory structure saved to 'directory_structure.yaml'")
    #print("Scrolls paths saved to 'scrolls.yaml'")

def list_files() -> Dict:
    """
    Load and return the scrolls configuration data from a YAML file.

    This function reads the updated 'scrolls.yaml' file and returns its contents as a dictionary.

    To update the files run:
    update_list("https://dl.ash2txt.org/other/dev/", "https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/volumetric-instance-labels/instance-labels/")
    
    Returns
    -------
    Dict
        A dictionary representing the scrolls configuration data.
    """
    install_path = get_installation_path()
    scroll_config = os.path.join(install_path, 'vesuvius', 'configs', f'scrolls.yaml')
    with open(scroll_config, 'r') as file:
        data = yaml.safe_load(file)
    return data

def list_cubes() -> Dict:
    """
    Load and return the cubes configuration data from a YAML file.

    This function reads the updated 'cubes.yaml' file and returns its contents as a dictionary.

    To update the files run:
    update_list("https://dl.ash2txt.org/other/dev/", "https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/volumetric-instance-labels/instance-labels/")

    Returns
    -------
    Dict
        A dictionary representing the cubes configuration data.
    """
    install_path = get_installation_path()
    cubes_config = os.path.join(install_path, 'vesuvius', 'configs', f'cubes.yaml')
    with open(cubes_config, 'r') as file:
        data = yaml.safe_load(file)
    return data

def is_aws_ec2_instance() -> bool:
    """
    Determine if the current system is an AWS EC2 instance.

    Returns
    -------
    bool
        True if running on an AWS EC2 instance, False otherwise.
    """
    try:
        # Query EC2 instance metadata to check if running on AWS EC2
        response = requests.get("http://169.254.169.254/latest/meta-data/", timeout=2)
        if response.status_code == 200:
            return True
    except requests.RequestException:
        return False

    return False