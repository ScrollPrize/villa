"""Surfaces in a VC object store (``vc_object_store``) for the fit service and its tools.

A stored surface is named by a reference string ``vcobj:<uuid>`` or ``vcobj:<uuid>@<geometry revision>`` in the
places a tifxyz directory path is accepted; the store file and the volume whose voxels coordinates are in travel
separately (``--store`` / ``--volume-uuid``). Geometry is a ``points`` array (rows x columns x 3, xyz, -1 for a
missing vertex), an optional ``valid`` array and per-vertex channels (``d``, ``approval``, ...), as VC3D stores
surfaces; coordinates are bound to a volume by ``meta.coordinates`` ({volume_uuid, scale_factor}).

Surfaces are created with ``vc_object_store.surfaces.create_surface``; fit results (correspondence and
atlas-control results) are attached to them with ``vc_object_store.documents.attach_document``.
"""
from __future__ import annotations

import getpass
import uuid as uuidlib
from pathlib import Path

import numpy as np

REF_PREFIX = "vcobj:"


def surface_ref(surface_uuid: str, geometry_revision: str | None = None) -> str:
	return REF_PREFIX + str(surface_uuid) + (f"@{geometry_revision}" if geometry_revision else "")


def parse_surface_ref(value: object) -> tuple[str, str | None] | None:
	"""(uuid, geometry revision or None) of a stored-surface reference; None for anything else (a path)."""
	if not isinstance(value, str) or not value.startswith(REF_PREFIX):
		return None
	uuid_text, _, revision = value[len(REF_PREFIX):].partition("@")
	return str(uuidlib.UUID(uuid_text)), (revision or None)


def open_store(store: str | Path, *, read_only: bool = True, author: str | None = None):
	from vc_object_store import ObjectStore
	if read_only:
		return ObjectStore.open(str(store), read_only=True, reader_pool_size=1)
	return ObjectStore.open(str(store), author=author or getpass.getuser())


class StoreSession:
	"""A store attached for this process: a hosted replica syncs with its hub in the background (pushing writes
	made through any connection to the file); artifact directories hold files belonging to an object (a fit's
	model checkpoint) next to the hub. Reads and writes go through ``open_store`` connections to ``path``.

	``shared``: the file is another process's store (VC3D's, for a service on its host) and that process's
	session syncs it; this one only reads and writes it, and ``sync`` has nothing to do."""

	def __init__(self, path: str | Path, *, author: str | None = None, shared: bool = False):
		from vc_object_store import ObjectStoreSession
		self.path = str(path)
		self.author = author or getpass.getuser()
		self._session = ObjectStoreSession()
		self._attachment = self._session.attach(self.path, self.author, read_only=shared)

	@property
	def store_uuid(self) -> str:
		return self._attachment.store_uuid

	def artifact_directory(self, object_uuid: str, *, create: bool = True) -> Path:
		return Path(self._session.artifact_directory(self._attachment, object_uuid, create=create))

	def sync(self, *, timeout_s: float = 120.0) -> None:
		"""Pull the hub's changes and push this replica's; returns once a sync started now has finished. A store
		without a hub, or a shared one, has nothing to sync."""
		status = self._session.sync_hub_and_wait(self._attachment, timeout_s=timeout_s)
		if status is None:
			return
		if status["conflicts"]:
			raise RuntimeError(f"store {self.path} has hub conflicts to resolve")
		if status["error"]:
			raise RuntimeError(f"store {self.path} did not sync with its hub: {status['error']}")

	def close(self) -> None:
		self._session.detach(self._attachment)


class StoredSurface:
	"""One revision of a stored surface in voxels of a volume.

	xyz: (H, W, 3) float32, -1 at missing vertices; meta: the record meta plus ``uuid``, ``name`` and the grid
	``scale``; channel(name) reads a per-vertex channel (None when absent)."""

	def __init__(self, store, surface_uuid, grid):
		self._store = store
		self.uuid = surface_uuid
		self.revision = grid.geometry_revision
		self.xyz = grid.points
		self._channels = grid.channels
		record = grid.record
		self.meta = {**(record.get("meta") or {}), "uuid": surface_uuid, "name": record.get("name") or surface_uuid,
					 "scale": list(grid.scale)}

	def channel(self, name: str) -> np.ndarray | None:
		if name not in self._channels:
			return None
		with open_store(self._store) as store:
			return np.array(store.read_geometry_arrays([self.revision], name).values)


def read_surface(store: str | Path, ref: str, *, volume: str | None) -> StoredSurface:
	"""The referenced surface in voxels of ``volume``, through the store's mapping to it."""
	from vc_object_store.surfaces import read_surface as read_grid
	parsed = parse_surface_ref(ref)
	if parsed is None:
		raise ValueError(f"not a stored surface reference: {ref!r}")
	surface_uuid, revision = parsed
	if volume is None:
		raise ValueError(f"reading stored surface {surface_uuid} needs the volume its coordinates are in")
	return StoredSurface(store, surface_uuid, read_grid(store, surface_uuid, volume=volume, geometry_revision=revision))
