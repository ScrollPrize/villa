"""Shared location, authentication and metadata fetching for reader backends."""

from pathlib import Path

from .auth import Access, http_url, is_remote


def read_node_file(location: str, name: str, access: Access) -> bytes | None:
    """Read one storage metadata object; only a missing object returns None."""
    if not is_remote(location):
        try:
            return Path(location, name).read_bytes()
        except (FileNotFoundError, NotADirectoryError):
            return None
    from .. import _reader

    try:
        return _reader.http_get(access.url(f'{location}/{name}'), auth=access.auth_tuple())
    except PermissionError as error:
        if access.kind == 'anonymous':
            raise PermissionError(f'{error} (requested anonymously: pass auth= with credentials)') from None
        raise


def cache_url(location: str, access: Access) -> str:
    """Canonical cache identity shared with the C++ Zarr reader."""
    return location if access.kind == 's3' else http_url(location)
