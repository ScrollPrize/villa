"""Surfaces in a VC object store (``vc_object_store``) for the fit service and its tools.

A stored surface is named by a reference string ``vcobj:<uuid>`` or ``vcobj:<uuid>@<geometry revision>`` in the
places a tifxyz directory path is accepted; the store file and the volume whose voxels coordinates are in travel
separately (``--store`` / ``--volume-uuid``). Geometry is a ``points`` array (rows x columns x 3, xyz, -1 for a
missing vertex), an optional ``valid`` array and per-vertex channels (``d``, ``approval``, ...), as VC3D stores
surfaces; coordinates are bound to a volume by ``meta.coordinates`` ({volume_uuid, scale_factor}).

Fit results attached to a surface (correspondence and atlas-control results) are point collections with a record
meta ``role`` hanging off the surface by an ``attachment`` relationship: data on the surface, never graph links.
"""
from __future__ import annotations

import getpass
import math
import time
import uuid as uuidlib
from pathlib import Path

import numpy as np

REF_PREFIX = "vcobj:"
INVALID = -1.0


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
		if self._session.hub_status(self._attachment) is None:
			return
		requested_us = time.time() * 1e6
		self._session.sync_hub(self._attachment)
		deadline = time.monotonic() + timeout_s
		while True:
			status = self._session.hub_status(self._attachment)
			if status["conflicts"]:
				raise RuntimeError(f"store {self.path} has hub conflicts to resolve")
			if (status["synced_at_us"] or 0) > requested_us and not status["pending"]:
				return
			if time.monotonic() > deadline:
				raise RuntimeError(f"store {self.path} did not sync with its hub: {status['error'] or 'timed out'}")
			time.sleep(0.05)

	def close(self) -> None:
		self._session.detach(self._attachment)


def _volume_factor(meta: dict, volume: str | None, surface_uuid: str) -> float:
	coordinates = (meta or {}).get("coordinates") or {}
	if volume is None:
		raise ValueError(f"reading stored surface {surface_uuid} needs the volume its coordinates are in")
	if coordinates.get("volume_uuid") != volume or not coordinates.get("scale_factor"):
		raise ValueError(f"stored surface {surface_uuid} is not bound to volume {volume}")
	return 1.0 / float(coordinates["scale_factor"])


class StoredSurface:
	"""One revision of a stored surface in voxels of a volume.

	xyz: (H, W, 3) float32, -1 at missing vertices; meta: the record meta plus ``uuid``, ``name`` and the grid
	``scale``; channel(name) reads a per-vertex channel (None when absent)."""

	def __init__(self, store, surface_uuid, revision, record, geometry, xyz, factor):
		self._store = store
		self.uuid = surface_uuid
		self.revision = revision
		self.xyz = xyz
		self._arrays = {array["name"]: array for array in geometry["arrays"]}
		scale = [float(s) / factor for s in geometry["attributes"]["scale"]]
		self.meta = {**(record.get("meta") or {}), "uuid": surface_uuid, "name": record.get("name") or surface_uuid,
					 "scale": scale}

	def channel(self, name: str) -> np.ndarray | None:
		array = self._arrays.get(name)
		if array is None:
			return None
		with open_store(self._store) as store:
			data = store.read_geometry_array(self.revision, name)
		return np.frombuffer(data, dtype=np.dtype(array["dtype"])).reshape(array["shape"]).copy()


def read_surface(store: str | Path, ref: str, *, volume: str | None) -> StoredSurface:
	parsed = parse_surface_ref(ref)
	if parsed is None:
		raise ValueError(f"not a stored surface reference: {ref!r}")
	surface_uuid, revision = parsed
	with open_store(store) as db:
		record = db.object(surface_uuid)
		if record is None or record["kind"] != "surface":
			raise ValueError(f"no surface {surface_uuid} in {store}")
		factor = _volume_factor(record.get("meta"), volume, surface_uuid)
		revision = revision or db.current_geometry(surface_uuid)["uuid"]
		geometry = db.geometry_revision(revision)
		shapes = {array["name"]: array["shape"] for array in geometry["arrays"]}
		xyz = np.frombuffer(db.read_geometry_array(revision, "points"), dtype="<f4").reshape(shapes["points"]).copy()
		# Missing: any coordinate -1 or non-finite (the store's rule), or marked invalid.
		missing = ~np.isfinite(xyz).all(axis=-1) | (xyz == INVALID).any(axis=-1)
		if "valid" in shapes:
			valid = np.frombuffer(db.read_geometry_array(revision, "valid"), dtype="u1").reshape(shapes["valid"])
			missing |= valid == 0
	xyz[missing] = INVALID
	if factor != 1.0:
		xyz[~missing] *= np.float32(factor)
	return StoredSurface(store, surface_uuid, revision, record, geometry, xyz, factor)


def create_surface(store: str | Path, xyz: np.ndarray, *, name: str, scale: float, volume: str, tool: str,
				   parameters: dict | None = None, sources: list[str] | tuple[str, ...] = (),
				   meta: dict | None = None, channels: dict[str, np.ndarray] | None = None,
				   tags: list[str] | tuple[str, ...] = (), author: str | None = None) -> str:
	"""A new surface holding ``xyz`` ((H, W, 3) in voxels of ``volume``; non-finite or -1 vertices are missing),
	bound to ``volume``, with per-vertex ``channels`` and provenance {tool, parameters, sources}. Returns its
	UUID."""
	points = np.ascontiguousarray(np.asarray(xyz, dtype="<f4"))
	if points.ndim != 3 or points.shape[2] != 3:
		raise ValueError(f"expected an (H, W, 3) grid, got {points.shape}")
	missing = ~np.isfinite(points).all(axis=2) | (points == INVALID).any(axis=2)
	points[missing] = INVALID
	arrays = [dict(name="points", dtype="<f4", shape=list(points.shape), data=points.tobytes())]
	for channel_name, values in sorted((channels or {}).items()):
		data = np.ascontiguousarray(values)
		if tuple(data.shape[:2]) != tuple(points.shape[:2]):
			raise ValueError(f"channel {channel_name!r} shape {data.shape} does not match the grid {points.shape}")
		arrays.append(dict(name=channel_name, dtype=data.dtype.str, shape=list(data.shape), data=data.tobytes()))
	surface_uuid = str(uuidlib.uuid4())
	record_meta = dict(meta or {})
	for key in ("uuid", "name", "format", "type", "scale", "bbox"):
		record_meta.pop(key, None)
	record_meta["coordinates"] = dict(volume_uuid=str(volume), scale_factor=1)
	record_meta["provenance"] = dict(tool=tool, parameters=parameters or {}, sources=list(sources))
	record = dict(uuid=surface_uuid, kind="surface", name=name, tags=list(tags), meta=record_meta)
	geometry = dict(kind="surface", attributes=dict(scale=[float(scale), float(scale)]), arrays=arrays)
	with open_store(store, read_only=False, author=author) as db:
		with db.write(expected_revisions={surface_uuid: None}) as tx:
			tx.upsert_object(record)
			tx.put_geometry(surface_uuid, geometry)
	return surface_uuid


def _position(value: object) -> list[float]:
	if isinstance(value, (list, tuple)) and len(value) >= 3:
		try:
			xyz = [float(value[0]), float(value[1]), float(value[2])]
		except (TypeError, ValueError):
			return [INVALID] * 3
		if all(math.isfinite(v) for v in xyz):
			return xyz
	return [INVALID] * 3


def attach_results(store: str | Path, surface_uuid: str, document: dict, *, role: str, list_key: str,
				   position_key: str, name: str, volume: str, author: str | None = None) -> str:
	"""Store a results document (records under ``list_key``, each positioned by ``position_key``) as a point
	collection attached to the surface: one point per record (-1 when the record has no finite position) whose
	annotation is the record, and the rest of the document in meta.source. Returns the collection's UUID."""
	records = [r for r in document.get(list_key, []) if isinstance(r, dict)]
	points = np.asarray([_position(r.get(position_key)) for r in records], dtype="<f8").reshape(len(records), 3)
	ids = np.arange(len(records), dtype="<i8")
	collection_uuid = str(uuidlib.uuid4())
	rest = {key: value for key, value in document.items() if key != list_key}
	record = dict(uuid=collection_uuid, kind="point_collection", name=name,
				  meta=dict(role=role, coordinates=dict(volume_uuid=str(volume), scale_factor=1),
							source=dict(document=rest, list_key=list_key, position_key=position_key,
										point_attributes=records)))
	geometry = dict(kind="point_collection", arrays=[
		dict(name="points", dtype="<f8", shape=[len(records), 3], data=points.tobytes()),
		dict(name="point_ids", dtype="<i8", shape=[len(records)], data=ids.tobytes())])
	with open_store(store, read_only=False, author=author) as db:
		surface_revision = db.current_object_revision(surface_uuid)
		with db.write(expected_revisions={collection_uuid: None,
										  surface_uuid: surface_revision["uuid"] if surface_revision else None}) as tx:
			tx.upsert_object(record)
			tx.put_geometry(collection_uuid, geometry)
			tx.attach(collection_uuid, surface_uuid, role)
	return collection_uuid
