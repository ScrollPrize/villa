"""Surfaces in a VC object store (``vc_object_store``) for the fit service and its tools.

A stored surface carries the shared ``{store,generation,object,version}`` reference.
CLI inputs also accept ``vcobj:<uuid>`` (current discovery) or
``vcobj:<uuid>@<store>/<generation>/<version>``; native input decoding resolves these once.
The store location and the volume whose voxels coordinates are in travel
separately (``--store`` / ``--volume-uuid``). A store is a location: a store file, or the endpoint of a
``vc_object_server`` (``http://...``) for a store on another machine. Geometry is a ``points`` array (rows x columns x 3, xyz, -1 for a
missing vertex), an optional ``valid`` array and per-vertex channels (``d``, ``approval``, ...), as VC3D stores
surfaces; coordinates are bound to a volume by ``meta.coordinates`` ({volume_uuid, scale_factor}).
Legacy ``vcobj:<uuid>@<generation>/<version>`` references still select history from the current owner.

Surfaces are created with ``vc_object_store.surfaces.create_surface``; fit results (correspondence and
atlas-control results) are attached to them with ``vc_object_store.documents.attach_document``.
"""
from __future__ import annotations

import getpass
from pathlib import Path

import numpy as np

REF_PREFIX = "vcobj:"


def is_surface_ref(value: object) -> bool:
	"""Dispatch structured references and external vcobj locators, not filesystem paths."""
	return isinstance(value, dict) or isinstance(value, str) and value.startswith(REF_PREFIX)


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

	def decode_reference(self, value) -> dict:
		"""Decode an external request once through the shared store boundary."""
		if isinstance(value, dict) and "object" in value:
			return value
		return self._session.decode_object_reference(self._attachment, value)

	def artifact_file(self, reference: dict, name: str, cache_directory: str | Path) -> Path | None:
		"""The selected object's artifact ``name`` as a local file under ``cache_directory`` (kept by content);
		None when it has none."""
		path = self._session.artifact_file(self._attachment, reference["object"], name, str(cache_directory),
										 reference=reference)
		return None if path is None else Path(path)

	def publish_artifacts(self, object_uuid: str, files: dict[str, str | Path]) -> dict:
		"""Publish local files ``{name: path}`` as artifacts of the object's current version (one checked
		request; the bytes are copied into the store first)."""
		from vc_object_store.locations import attached
		with attached(self.location, author=self.author) as (session, attachment):
			current = session.current_object(attachment, object_uuid)
			if current is None:
				raise ValueError(f"no object {object_uuid} to publish artifacts of")
			from vc_object_store import object_reference
			return session.publish_artifacts(attachment, object_reference(current["record"]),
											 {name: str(path) for name, path in files.items()}, self.author)

	def close(self) -> None:
		self._session.detach(self._attachment)


class StoredSurface:
	"""One version of a stored surface in voxels of a volume.

	xyz: (H, W, 3) float32; valid: the shared source-validity mask, preserved through coordinate transforms;
	meta: the record meta plus ``uuid``, ``name`` and the grid
	``scale``; reference is the unchanged shared selected identity; channel(name) reads a per-vertex channel (None when absent)."""

	def __init__(self, store, reference, grid):
		self._store = store
		self.uuid = reference["object"]
		self.reference = reference
		self._geometry = grid.geometry_revision
		self.xyz = grid.points
		self.valid = grid.valid
		self._channels = grid.channels
		record = grid.record
		self.meta = {**(record.get("meta") or {}), "uuid": self.uuid, "name": record.get("name") or self.uuid,
					 "scale": list(grid.scale)}

	def channel(self, name: str) -> np.ndarray | None:
		if name not in self._channels:
			return None
		from vc_object_store.locations import attached
		with attached(self._store, read_only=True) as (session, attachment):
			return np.array(session.read_geometry_arrays(attachment, [self._geometry], name).values)


def read_surface(store: str | Path, ref: dict | str, *, volume: str | None, target_scale: float = 1.0) -> StoredSurface:
	"""The referenced surface in voxels of ``volume`` at ``target_scale`` of its own (``--voxel-scale``), through
	the store's mapping to it; ``store`` is a location."""
	from vc_object_store.locations import attached
	from vc_object_store.surfaces import surface_grid
	if volume is None:
		raise ValueError("reading a stored surface needs the volume its coordinates are in")
	with attached(store, read_only=True) as (session, attachment):
		# Already-selected references are carried through unchanged. Only CLI
		# locators and external legacy selectors need the shared decoder.
		reference = ref if isinstance(ref, dict) and "object" in ref else session.decode_object_reference(attachment, ref)
		grid = session.read_surface_grid(attachment, reference["object"], str(volume), float(target_scale), reference=reference)
	return StoredSurface(store, reference, surface_grid(grid))
