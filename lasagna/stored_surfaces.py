"""Surfaces in a VC object store (``vc_object_store``) for the fit service and its tools.

A stored surface is named by a reference string ``vcobj:<uuid>`` (its current version) or
``vcobj:<uuid>@<generation>/<version>`` (that numbered version of the store's history) in the
places a tifxyz directory path is accepted; the store and the volume whose voxels coordinates are in travel
separately (``--store`` / ``--volume-uuid``). A store is a location: a store file, or the endpoint of a
``vc_object_server`` (``http://...``) for a store on another machine. Geometry is a ``points`` array (rows x columns x 3, xyz, -1 for a
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


def surface_ref(surface_uuid: str, version: tuple[str, int] | None = None) -> str:
	"""The reference to a surface's current version, or to ``version`` = (generation, number)."""
	if version is None:
		return REF_PREFIX + str(surface_uuid)
	generation, number = version
	if not generation or int(number) < 1:
		raise ValueError(f"not a numbered version: {version!r}")
	return f"{REF_PREFIX}{surface_uuid}@{generation}/{int(number)}"


def parse_surface_ref(value: object) -> tuple[str, tuple[str, int] | None] | None:
	"""(uuid, (generation, version) or None for the current version) of a stored-surface reference; None for
	anything else (a path)."""
	if not isinstance(value, str) or not value.startswith(REF_PREFIX):
		return None
	uuid_text, at, version = value[len(REF_PREFIX):].partition("@")
	if not at:
		return str(uuidlib.UUID(uuid_text)), None
	generation, slash, number = version.partition("/")
	if not generation or not slash or not number.isdigit() or int(number) < 1:
		raise ValueError(f"stored surface reference {value!r} does not name <generation>/<version>")
	return str(uuidlib.UUID(uuid_text)), (generation, int(number))


def next_free_name(store: str | Path, root: str, mode: str) -> str:
	"""The name a new surface of ``root`` takes among the store's surfaces (``store`` a location): ``version``
	gives the next ``<root>_vNNN`` after the highest stored, ``offset`` the first free ``<root>_offN``."""
	import re
	from vc_object_store.locations import attached
	with attached(store, read_only=True) as (session, attachment):
		rows = session.select_objects(attachment, {"selection": {"kind": "surface"}, "fields": ["name"]})
	taken = {row["name"] for row in rows}
	infix, width = ("_v", 3) if mode == "version" else ("_off", 1)
	number = 1
	if mode == "version":
		pattern = re.compile(re.escape(root + infix) + r"(\d+)$")
		for name in taken:
			if match := pattern.fullmatch(name):
				number = max(number, int(match.group(1)) + 1)
	while f"{root}{infix}{number:0{width}d}" in taken:
		number += 1
	return f"{root}{infix}{number:0{width}d}"


class StoreSession:
	"""A store attached for this process: a store file, or a ``vc_object_server`` endpoint for a store on another
	machine. Files belonging to an object (a fit's model checkpoint) are its artifacts: published into the store
	with the object's version and read back as cached local copies. Writes go through short-lived attachments of
	``location``; every write is committed when it returns.

	``shared``: the file is another process's store (VC3D's, for a service on its host); this session attaches it
	read-only (an endpoint is always attached writable)."""

	def __init__(self, location: str | Path, *, author: str | None = None, shared: bool = False):
		from vc_object_store import ObjectStoreSession
		self.location = str(location)
		self.author = author or getpass.getuser()
		self._session = ObjectStoreSession()
		self._attachment = self._session.attach_location(self.location, self.author, shared)

	@property
	def store_uuid(self) -> str:
		return self._attachment.store_uuid

	def artifact_file(self, object_uuid: str, name: str, cache_directory: str | Path) -> Path | None:
		"""The object's current artifact ``name`` as a local file under ``cache_directory`` (kept by content);
		None when it has none."""
		path = self._session.artifact_file(self._attachment, object_uuid, name, str(cache_directory))
		return None if path is None else Path(path)

	def publish_artifacts(self, object_uuid: str, files: dict[str, str | Path]) -> dict:
		"""Publish local files ``{name: path}`` as artifacts of the object's current version (one checked
		request; the bytes are copied into the store first)."""
		from vc_object_store.locations import attached
		with attached(self.location, author=self.author) as (session, attachment):
			current = session.current_object(attachment, object_uuid)
			if current is None:
				raise ValueError(f"no object {object_uuid} to publish artifacts of")
			return session.publish_artifacts(attachment, object_uuid, current["version"],
											 {name: str(path) for name, path in files.items()}, self.author)

	def close(self) -> None:
		self._session.detach(self._attachment)


class StoredSurface:
	"""One version of a stored surface in voxels of a volume.

	xyz: (H, W, 3) float32, -1 at missing vertices; meta: the record meta plus ``uuid``, ``name`` and the grid
	``scale``; channel(name) reads a per-vertex channel (None when absent)."""

	def __init__(self, store, surface_uuid, grid):
		self._store = store
		self.uuid = surface_uuid
		self.version = (grid.generation, grid.version)
		self._geometry = grid.geometry_revision
		self.xyz = grid.points
		self._channels = grid.channels
		record = grid.record
		self.meta = {**(record.get("meta") or {}), "uuid": surface_uuid, "name": record.get("name") or surface_uuid,
					 "scale": list(grid.scale)}

	def channel(self, name: str) -> np.ndarray | None:
		if name not in self._channels:
			return None
		from vc_object_store.locations import attached
		with attached(self._store, read_only=True) as (session, attachment):
			return np.array(session.read_geometry_arrays(attachment, [self._geometry], name).values)


def read_surface(store: str | Path, ref: str, *, volume: str | None, target_scale: float = 1.0) -> StoredSurface:
	"""The referenced surface in voxels of ``volume`` at ``target_scale`` of its own (``--voxel-scale``), through
	the store's mapping to it; ``store`` is a location."""
	from vc_object_store.surfaces import read_surface as read_grid
	parsed = parse_surface_ref(ref)
	if parsed is None:
		raise ValueError(f"not a stored surface reference: {ref!r}")
	surface_uuid, version = parsed
	if volume is None:
		raise ValueError(f"reading stored surface {surface_uuid} needs the volume its coordinates are in")
	generation, number = version or (None, None)
	grid = read_grid(store, surface_uuid, volume=volume, version=number, generation=generation, target_scale=target_scale)
	return StoredSurface(store, surface_uuid, grid)
