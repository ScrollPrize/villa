"""Store locations: a store file path, or the endpoint of a vc_object_server (``http://`` or ``https://``)."""
from __future__ import annotations

import getpass
from contextlib import contextmanager

from ._native import ObjectStoreSession


@contextmanager
def attached(location, *, author=None, read_only=False):
    """A session attached to ``location`` for the duration of the block; yields ``(session, attachment)``.
    Endpoints are attached writable (``read_only`` applies to files)."""
    session = ObjectStoreSession()
    attachment = session.attach_location(str(location), author or getpass.getuser(), read_only)
    try:
        yield session, attachment
    finally:
        session.detach(attachment)
