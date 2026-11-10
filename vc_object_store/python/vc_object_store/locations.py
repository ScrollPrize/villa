"""Store locations: a store file path, or the endpoint of a vc_object_server (``http://``, ``https://``,
or ``ssh://[user@]host:port`` for the service on a remote host's loopback port).

``attached(location)`` yields a session attachment for either; ``open_store(location)`` yields a store handle
with the ``ObjectStore`` reading and request methods on either: the ``ObjectStore`` itself for a file, and a
``HostedStoreHandle`` (the same methods over a session's hosted attachment) for an endpoint, so one code path
reads, links and writes a store wherever it is.
"""
from __future__ import annotations

import getpass
from contextlib import contextmanager

from ._native import ObjectStore, ObjectStoreSession

ENDPOINT_SCHEMES = ("http://", "https://", "ssh://", "sftp://")


def is_endpoint(location):
    """Whether a store location names a served store rather than a file."""
    return str(location).startswith(ENDPOINT_SCHEMES)


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


def _object_query(object_kind=None, tag=None, object_uuids=(), include_retired=False, limit=None, fields=None):
    selection = {"kind": object_kind, "tag": tag, "uuids": list(object_uuids), "include_retired": bool(include_retired)}
    if limit is not None:
        selection["limit"] = int(limit)
    query = {"selection": selection}
    if fields is not None:
        query["fields"] = list(fields)
    return query


class HostedStoreHandle:
    """A store served by vc_object_server, read and written through one session attachment with the
    ``ObjectStore`` method names and argument forms this handle covers: status, submit and receipts, the change
    feed, current and numbered records, relationships at versions, geometry descriptors, arrays and point
    annotations, object and edge selections by kind, tag and UUID, and pinned overlap correspondences.

    ``submit`` takes (and ignores) ``progress``: a ``link_objects`` request's detectors run on the service,
    which reports no progress over the wire, and the reply waits ``link_timeout_s`` for them. The attachment
    keeps a persistent cache of numbered records and geometry under ``cache_directory`` when one is given,
    so repeated reads of pinned inputs cost no transfers.
    """

    def __init__(self, endpoint, *, author=None, link_timeout_s=300, cache_directory=None,
                 disk_cache_bytes=None, retries=5, connect_timeout_ms=10000):
        self.endpoint = str(endpoint)
        self.session = ObjectStoreSession()
        options = {"author": author or getpass.getuser(), "link_timeout_s": int(link_timeout_s),
                   "retries": int(retries), "connect_timeout_ms": int(connect_timeout_ms)}
        if cache_directory:
            options["cache_directory"] = str(cache_directory)
            if disk_cache_bytes is not None:
                options["disk_cache_bytes"] = int(disk_cache_bytes)
        self.attachment = self.session.attach_hosted(self.endpoint, **options)

    @property
    def location(self):
        return self.endpoint

    @property
    def uuid(self):
        return self.attachment.store_uuid

    def close(self):
        if self.attachment is not None:
            self.session.detach(self.attachment)
            self.attachment = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _require(self):
        if self.attachment is None:
            raise RuntimeError(f"the store at {self.endpoint} is closed")
        return self.attachment

    # --- status, requests and the change feed ---------------------------------------------------------

    def status(self):
        return self.session.status(self._require())

    def submit(self, request, payloads=None, progress=None):
        del progress  # detection runs on the service; no progress crosses the wire
        return self.session.submit(self._require(), request, payloads or {})

    def receipt(self, request_uuid):
        return self.session.receipt(self._require(), request_uuid)

    def changes_since(self, cursor, max_changes=4096):
        return self.session.changes_since(self._require(), cursor, max_changes)

    # --- records ----------------------------------------------------------------------------------------

    def current_object_revision(self, object_uuid):
        return self.session.current_object(self._require(), object_uuid)

    def current_version(self, object_uuid):
        revision = self.current_object_revision(object_uuid)
        return None if revision is None else revision["version"]

    def objects_at_versions(self, refs):
        return self.session.objects_at_versions(self._require(), list(refs))

    def relationships_at_versions(self, refs):
        return self.session.relationships_at_versions(self._require(), list(refs))

    def select_objects(self, object_kind=None, tag=None, object_uuids=(), fields=None, include_retired=False,
                       limit=None):
        """Objects by kind, tag and UUID (``ObjectStore.select_objects``'s arguments this handle covers);
        complete unless ``limit`` is given."""
        return self.session.select_objects(self._require(), _object_query(
            object_kind, tag, object_uuids, include_retired, limit, fields))

    def select_edges(self, edge_kind=None, edge_uuids=None, endpoint_uuids=None, include_retired=False,
                     fields=None):
        selection = {"kind": edge_kind, "include_retired": bool(include_retired)}
        if edge_uuids is not None:
            selection["uuids"] = list(edge_uuids)
        if endpoint_uuids is not None:
            selection["endpoint_uuids"] = list(endpoint_uuids)
        query = {"selection": selection}
        if fields is not None:
            query["fields"] = list(fields)
        return self.session.select_edges(self._require(), query)

    def edge(self, uuid):
        """An edge's record by UUID (retired ones included), or None."""
        edges = self.select_edges(edge_uuids=[uuid], include_retired=True)
        return edges[0] if edges else None

    # --- geometry ---------------------------------------------------------------------------------------

    def geometry_revision(self, revision_uuid):
        return self.session.geometry_revision(self._require(), revision_uuid)

    def read_geometry_array(self, revision_uuid, array_name, byte_offset=0, byte_count=None):
        return self.session.read_geometry_array(self._require(), revision_uuid, array_name, byte_offset, byte_count)

    def geometry_point_annotations(self, revision_uuid, array_name):
        return self.session.geometry_point_annotations(self._require(), revision_uuid, array_name)

    def overlap_correspondences_at(self, edges, volume_uuid, spacing=8.0, max_pairs=256):
        return self.session.overlap_correspondences_at(self._require(), list(edges), volume_uuid, spacing, max_pairs)


@contextmanager
def open_store(location, *, author=None, read_only=False, link_timeout_s=300, cache_directory=None,
               disk_cache_bytes=None, **hosted_options):
    """The store at ``location`` for the duration of the block: an ``ObjectStore`` opened on a file
    (``author`` or ``read_only``, as ``ObjectStore.open``; a file a vc_object_server holds refuses a writable
    open), or a ``HostedStoreHandle`` on an endpoint (always writable as ``author``; ``link_timeout_s``,
    ``cache_directory`` and ``disk_cache_bytes`` are its options). Both yield the same reading and request
    methods."""
    if is_endpoint(location):
        with HostedStoreHandle(location, author=author, link_timeout_s=link_timeout_s,
                               cache_directory=cache_directory, disk_cache_bytes=disk_cache_bytes,
                               **hosted_options) as store:
            yield store
        return
    options = {"read_only": True} if read_only else {"author": author or getpass.getuser()}
    with ObjectStore.open(str(location), **options) as store:
        yield store
