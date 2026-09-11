"""Utilities for reading and refreshing packaged dataset catalog metadata."""

from __future__ import annotations

import asyncio
import importlib
import ssl
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Optional, Tuple

import requests
import yaml

from vesuvius.install.accept_terms import get_installation_path


CatalogTree = Dict[str, Optional[Dict]]
CatalogListing = Dict[str, str]


def _require(module_name: str, used_for: str) -> ModuleType:
    """Import a package only needed for refreshing the catalog from the data server.

    Reading the packaged catalog (:func:`list_files`, :func:`list_cubes`) needs neither
    of these, so they are imported at the point of use rather than at module import.
    """

    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised via tests
        raise ModuleNotFoundError(
            f"{used_for} needs the '{module_name}' package, which is not installed. "
            f"Install it with 'pip install {module_name}', or install the 'all' extra. "
            "Reading the packaged catalog with list_files() or list_cubes() does not "
            "need it."
        ) from exc


def _config_dir() -> Path:
    """Return the directory where packaged catalog YAML files live."""
    root = Path(get_installation_path()) / "vesuvius" / "install" / "configs"
    root.mkdir(parents=True, exist_ok=True)
    return root


async def scrape_website(
    base_url: str, ignore_list: List[str]
) -> Tuple[CatalogTree, CatalogListing]:
    """Collect directory metadata and Zarr links from a remote listing."""

    from vesuvius.data.paths import parser

    aiohttp = _require("aiohttp", "Refreshing the catalog from the data server")

    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    timeout = aiohttp.ClientTimeout(total=60)
    connector = aiohttp.TCPConnector(ssl=ssl_context)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        directory_tree = await parser.get_directory_structure(
            base_url, session, ignore_list
        )
        zarr_files = await parser.find_zarr_files(directory_tree, base_url, session)
        return directory_tree, zarr_files


async def collect_subfolders(base_url: str, ignore_list: List[str]) -> List[str]:
    """Return subfolder paths beneath ``base_url`` ignoring provided patterns."""

    from vesuvius.data.paths import parser

    aiohttp = _require("aiohttp", "Refreshing the catalog from the data server")

    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    timeout = aiohttp.ClientTimeout(total=60)
    connector = aiohttp.TCPConnector(ssl=ssl_context)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        subfolders = await parser.list_subfolders(base_url, session, ignore_list)
        return subfolders


def _ensure_loop() -> asyncio.AbstractEventLoop:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def update_list(
    base_url: str,
    base_url_cubes: str,
    ignore_list: Optional[List[str]] = None,
) -> None:
    """Refresh packaged scroll and cube catalog YAML files from remote sources."""

    config_dir = _config_dir()
    scroll_config = config_dir / "scrolls.yaml"
    directory_config = config_dir / "directory_structure.yaml"
    cubes_config = config_dir / "cubes.yaml"

    if ignore_list is None:
        ignore_list = [r"\.zarr$"]

    loop = _ensure_loop()
    if loop.is_running():
        _require(
            "nest_asyncio", "Refreshing the catalog from inside a running event loop"
        ).apply()

    tree, zarr_files = loop.run_until_complete(scrape_website(base_url, ignore_list))
    cubes_tree, _ = loop.run_until_complete(scrape_website(base_url_cubes, ignore_list))

    directory_config.write_text(
        yaml.dump(tree, default_flow_style=False), encoding="utf-8"
    )
    scroll_config.write_text(
        yaml.dump(zarr_files, default_flow_style=False), encoding="utf-8"
    )

    data = {1: {54: {7.91: {}}}}
    for folder in cubes_tree.keys():
        folder_name = folder.rstrip("/")
        data[1][54][7.91][folder_name] = f"{base_url_cubes}{folder}"

    cubes_config.write_text(yaml.dump(data, default_flow_style=False), encoding="utf-8")


def list_files() -> Dict:
    """Load the packaged scroll configuration YAML as a dictionary."""

    scroll_config = _config_dir() / "scrolls.yaml"
    with scroll_config.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def list_cubes() -> Dict:
    """Load the packaged cube configuration YAML as a dictionary."""

    cubes_config = _config_dir() / "cubes.yaml"
    with cubes_config.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def is_aws_ec2_instance() -> bool:
    """Best-effort detection for AWS EC2 environment."""

    try:
        response = requests.get(
            "http://169.254.169.254/latest/meta-data/", timeout=2
        )
        if response.status_code == 200:
            return True
    except requests.RequestException:
        return False

    return False


__all__ = [
    "collect_subfolders",
    "is_aws_ec2_instance",
    "list_cubes",
    "list_files",
    "scrape_website",
    "update_list",
]
