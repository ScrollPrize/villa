from lxml import html
import re
import asyncio
import aiohttp
from typing import Dict, Optional, List
from .fetcher import fetch

async def get_directory_structure(url: str, session: aiohttp.ClientSession, ignore_list: Optional[List[str]] = None, zarr_pattern: str = r'\.zarr/$') -> Dict[str, Optional[Dict]]:    
    if ignore_list is None:
        ignore_list = []
    
    page_content = await fetch(session, url)
    if not page_content:
        return {}
    
    tree = html.fromstring(page_content)
    directory_tree: Dict[str, Optional[Dict]] = {}
    tasks = []
    hrefs = []

    for element in tree.xpath('//a'):
        href = element.get('href')
        if href and href not in ['../', './'] and '?' not in href:
            full_url = url + href
            if any(re.search(pattern, href) for pattern in ignore_list):
                continue
            if href.endswith('/') and not re.search(zarr_pattern, href):  # Normal directories
                tasks.append(get_directory_structure(full_url, session, ignore_list, zarr_pattern))
                hrefs.append(href)
            elif re.search(zarr_pattern, href):  # .zarr directories
                directory_tree[href] = None  # Mark it as a zarr directory without further traversal
                hrefs.append(href)
    
    nested_directories = await asyncio.gather(*tasks)
    for idx, href in enumerate(hrefs):
        if not re.search(zarr_pattern, href) and (idx < len(nested_directories)):
            directory_tree[href] = nested_directories[idx]

    return directory_tree

async def list_subfolders(url: str, session: aiohttp.ClientSession, ignore_list: Optional[List[str]] = None) -> List[str]:
    if ignore_list is None:
        ignore_list = []

    page_content = await fetch(session, url)
    if not page_content:
        return []

    tree = html.fromstring(page_content)
    subfolders = []

    for element in tree.xpath('//a'):
        href = element.get('href')
        if href and href not in ['../', './'] and '?' not in href:
            full_url = url + href
            if any(re.search(pattern, href) for pattern in ignore_list):
                continue
            if href.endswith('/'):  # Normal directories
                subfolders.append(full_url)

    return subfolders

async def find_zarr_files(tree: Dict[str, Optional[Dict]], url: str, session: aiohttp.ClientSession) -> Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, str]]]]]:
    zarr_files: Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, str]]]]] = {}
    volume_pattern = re.compile(r'volumes/(?P<intensity>\d+)keV_(?P<resolution>\d+\.\d{2})um\.zarr/')
    segment_pattern = re.compile(r'segments/(?P<intensity>\d+)keV_(?P<resolution>\d+\.\d{2})um/(?P<segment_id>[^/]+)\.zarr/')

    for key, value in tree.items():
        if isinstance(value, dict):
            nested_zarr = await find_zarr_files(value, url + key, session)
            for scrollnumber, intensities in nested_zarr.items():
                if scrollnumber not in zarr_files:
                    zarr_files[scrollnumber] = {}
                for intensity, resolutions in intensities.items():
                    if intensity not in zarr_files[scrollnumber]:
                        zarr_files[scrollnumber][intensity] = {}
                    for resolution, data in resolutions.items():
                        if resolution not in zarr_files[scrollnumber][intensity]:
                            zarr_files[scrollnumber][intensity][resolution] = {'volume': None, 'segments': {}}
                        if 'volume' in data:
                            zarr_files[scrollnumber][intensity][resolution]['volume'] = data['volume']
                        if 'segments' in data:
                            zarr_files[scrollnumber][intensity][resolution]['segments'].update(data['segments'])
        elif value is None:
            volume_match = volume_pattern.search(url + key)
            segment_match = segment_pattern.search(url + key)
            scroll_match = re.search(r'scrolls/(?P<scrollnumber>\d+[a-zA-Z]?)/', url)
            if volume_match and scroll_match:
                scrollnumber = scroll_match.group('scrollnumber')
                intensity = volume_match.group('intensity')
                resolution = volume_match.group('resolution')
                if scrollnumber not in zarr_files:
                    zarr_files[scrollnumber] = {}
                if intensity not in zarr_files[scrollnumber]:
                    zarr_files[scrollnumber][intensity] = {}
                if resolution not in zarr_files[scrollnumber][intensity]:
                    zarr_files[scrollnumber][intensity][resolution] = {'volume': None, 'segments': {}}
                zarr_files[scrollnumber][intensity][resolution]['volume'] = url + key
            elif segment_match and scroll_match:
                scrollnumber = scroll_match.group('scrollnumber')
                intensity = segment_match.group('intensity')
                resolution = segment_match.group('resolution')
                segment_id = segment_match.group('segment_id')
                if scrollnumber not in zarr_files:
                    zarr_files[scrollnumber] = {}
                if intensity not in zarr_files[scrollnumber]:
                    zarr_files[scrollnumber][intensity] = {}
                if resolution not in zarr_files[scrollnumber][intensity]:
                    zarr_files[scrollnumber][intensity][resolution] = {'volume': None, 'segments': {}}
                zarr_files[scrollnumber][intensity][resolution]['segments'][segment_id] = url + key
    
    return zarr_files
