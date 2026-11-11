"""Fits read input surfaces from a VC object store and export results into it (the fit service as VC3D uses it)."""
from __future__ import annotations

import os
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
	sys.path.insert(0, ROOT)

try:
	from vc_object_store import ObjectStore, Surface, object_reference
	from vc_object_store.documents import attached_document
	from vc_object_store.surfaces import create_surface
except ImportError:  # pragma: no cover - the store bindings are an optional dependency
	ObjectStore = None

import approval_inpaint
import fit2tifxyz
import fit_service
import stored_surfaces
import tifxyz_io

VOLUME = "11111111-2222-4333-8444-555555555555"


def _array(name: str, values: np.ndarray) -> dict:
	values = np.ascontiguousarray(values)
	return dict(name=name, dtype=values.dtype.str, shape=list(values.shape), data=values.tobytes())


def _source_surface(path: Path, *, scale_factor: float = 1.0, history_enabled: bool = True) -> tuple[str, np.ndarray]:
	rows, cols = np.mgrid[:5, :6]
	points = np.stack([cols * 2.0 + 10, rows * 2.0 + 20, np.full(rows.shape, 30.0)], axis=-1).astype("<f4")
	points[0, 0] = -1
	valid = np.full((5, 6), 1, "u1")
	valid[4, 5] = 0
	approval = np.zeros((5, 6), "u1")
	approval[1:4, 1:5] = 255
	approval[2, 2] = 0
	with ObjectStore.create(str(path), author="test", history_enabled=history_enabled) as db:
		uid = db.upsert_object(dict(kind="surface", name="source", meta=dict(
			coordinates=dict(volume_uuid=VOLUME, scale_factor=scale_factor))))
		db.put_geometry(uid, dict(kind="surface", attributes=dict(scale=[0.5, 0.5]), arrays=[
			_array("points", points), _array("valid", valid), _array("approval", approval),
			_array("d", np.full((5, 6), 3.0, "<f4"))]))
	return uid, points


class SurfaceReferenceDispatchTest(unittest.TestCase):
	def test_one_rule_distinguishes_vcobj_inputs_from_paths_and_artifacts(self) -> None:
		reference = dict(store=VOLUME, generation=VOLUME, object=VOLUME, version=1)
		for value in (reference, {"reference": reference}, {"type": "surface", "uuid": VOLUME},
					  {"uuid": VOLUME}, f"vcobj:{VOLUME}", f"vcobj:{VOLUME}@{VOLUME}/{VOLUME}/1",
					  {"type": "surface"}, {"object": "malformed"}):
			with self.subTest(value=value):
				self.assertTrue(stored_surfaces.is_surface_ref(value))
		for value in (None, {}, "/data/surface.tifxyz", Path("surface.tifxyz"),
					  {"path": "/data/surface.tifxyz"},
					  {"type": "lasagna_model", "name": "model.pt", "hash": "md5:" + "0" * 32},
					  {"type": "lasagna_model", "uuid": VOLUME, "name": "model.pt", "hash": "md5:" + "0" * 32},
					  {"type": "atlas-base", "name": "base", "hash": "md5:" + "0" * 32}):
			with self.subTest(value=value):
				self.assertFalse(stored_surfaces.is_surface_ref(value))


@unittest.skipIf(ObjectStore is None, "vc_object_store is not importable")
class StoredSurfaceReadTest(unittest.TestCase):
	def test_shared_reference_survives_fit_config_argument_defaults(self) -> None:
		import fit
		import cli_json
		ref = dict(store=VOLUME, generation=VOLUME, object=VOLUME, version=1)
		for option in ("tifxyz-init", "approval-inpaint-tifxyz"):
			with tempfile.TemporaryDirectory() as td:
				config = Path(td) / "fit.json"
				config.write_text(json.dumps(dict(args={option: ref})))
				parsed = cli_json.parse_args(fit._build_parser(), [str(config)])
				self.assertEqual(getattr(parsed, option.replace("-", "_")), ref)
	def test_transformed_validity_survives_all_fit_loaders_and_scaling(self) -> None:
		for history in (False, True):
			with self.subTest(history=history), tempfile.TemporaryDirectory() as td:
				store = Path(td) / "objects.vcobj"
				uid, points = _source_surface(store, history_enabled=history)
				with ObjectStore.open(str(store), author="test") as db:
					db.register_volume_transform(VOLUME, "target", [[1, 0, 0, -11], [0, 1, 0, 0], [0, 0, 1, 0]])
				current_ref = f"vcobj:{uid}"
				first = stored_surfaces.read_surface(store, current_ref, volume="target")
				refs = [current_ref, first.reference]
				if history:
					with ObjectStore.open(str(store), author="test") as db:
						record = db.object(uid)
						record["name"] = "new name"
						db.upsert_object(record)
				for ref in refs:
					with self.subTest(ref=ref):
						loaded = stored_surfaces.read_surface(store, ref, volume="target")
						self.assertTrue(loaded.valid[1, 0])
						self.assertEqual(float(loaded.xyz[1, 0, 0]), -1.0)
						xyz, valid, _ = tifxyz_io.load_tifxyz(ref, store=store, volume="target")
						np.testing.assert_array_equal(valid.numpy(), loaded.valid)
						np.testing.assert_array_equal(xyz[1, 0], loaded.xyz[1, 0])
						self.assertTrue(np.all(xyz[~valid].numpy() == 0))
						approval = approval_inpaint.load_approval_surface(ref, store=store, volume="target")
						np.testing.assert_array_equal(approval.valid, loaded.valid)
						scaled = approval.scaled(2, base_shape_zyx=(8, 8, 8))
						np.testing.assert_array_equal(scaled.valid, loaded.valid)
						np.testing.assert_array_equal(scaled.xyz[1, 0], (points[1, 0] - [11, 0, 0]) * 2)
						np.testing.assert_array_equal(scaled.xyz[~scaled.valid], approval.xyz[~approval.valid])

	def test_fit_loaders_read_stored_surfaces_in_volume_voxels(self) -> None:
		with tempfile.TemporaryDirectory() as td:
			store = Path(td) / "objects.vcobj"
			uid, points = _source_surface(store, scale_factor=2.0)
			ref = f"vcobj:{uid}"
			xyz, valid, meta = tifxyz_io.load_tifxyz(ref, store=store, volume=VOLUME)
			self.assertEqual(tuple(xyz.shape), (5, 6, 3))
			self.assertFalse(bool(valid[0, 0]) or bool(valid[4, 5]))
			self.assertTrue(bool(valid[1, 1]))
			self.assertEqual(float(xyz[0, 0, 0]), 0.0)  # invalid vertices are zeroed for the fit
			np.testing.assert_allclose(xyz[1, 1].numpy(), points[1, 1] / 2)
			self.assertEqual(meta["uuid"], uid)
			self.assertEqual(meta["scale"], [1.0, 1.0])  # 0.5 per stored unit = 1 per voxel

			surface = approval_inpaint.load_approval_surface(ref, store=store, volume=VOLUME)
			self.assertEqual(surface.approval.dtype, bool)
			self.assertTrue(bool(surface.approval[1, 1]) and not bool(surface.approval[2, 2]))
			self.assertEqual(float(surface.d[1, 1]), 3.0)
			self.assertFalse(bool(surface.valid[4, 5]))

			with self.assertRaisesRegex(ValueError, "does not map"):
				tifxyz_io.load_tifxyz(ref, store=store, volume="other")
			with self.assertRaisesRegex(ValueError, "needs a store"):
				tifxyz_io.load_tifxyz(ref)

	def test_missing_vertices_follow_the_store_rule(self) -> None:
		# Any -1 or non-finite coordinate, or z <= 0, is missing, read and written.
		with tempfile.TemporaryDirectory() as td:
			store = Path(td) / "objects.vcobj"
			ObjectStore.create(str(store), author="test").close()
			xyz = np.full((1, 3, 3), 30.0, "<f4")
			xyz[0, 0] = (5, -1, 30)
			xyz[0, 1] = (6, 7, 0)
			uid = create_surface(store, xyz, name="s", scale=1.0, volume=VOLUME, tool="test")
			surface = stored_surfaces.read_surface(store, f"vcobj:{uid}", volume=VOLUME)
			self.assertEqual(surface.xyz[0, 0].tolist(), [-1.0, -1.0, -1.0])
			self.assertEqual(surface.xyz[0, 1].tolist(), [-1.0, -1.0, -1.0])

	def test_numbered_version_refs_read_that_version(self) -> None:
		with tempfile.TemporaryDirectory() as td:
			store = Path(td) / "objects.vcobj"
			ObjectStore.create(str(store), author="test").close()
			xyz = np.full((2, 2, 3), 30.0, "<f4")
			uid = create_surface(store, xyz, name="s", scale=1.0, volume=VOLUME, tool="test")
			first = stored_surfaces.read_surface(store, f"vcobj:{uid}", volume=VOLUME)
			with ObjectStore.open(str(store), author="test") as db:
				with db.write(expected_versions={uid: first.reference}) as tx:
					tx.put_geometry(uid, Surface.from_arrays(np.ascontiguousarray(xyz + 1), [1.0, 1.0]),
									expected_version=1)
			ref = first.reference
			self.assertEqual(ref["version"], 1)
			self.assertEqual(set(ref), {"store", "generation", "object", "version"})
			legacy = f"vcobj:{uid}@{ref['generation']}/01"
			np.testing.assert_array_equal(stored_surfaces.read_surface(store, legacy, volume=VOLUME).xyz, xyz)
			older = stored_surfaces.read_surface(store, ref, volume=VOLUME)
			current = stored_surfaces.read_surface(store, f"vcobj:{uid}", volume=VOLUME)
			self.assertEqual((older.reference, current.reference["version"]), (first.reference, 2))
			np.testing.assert_array_equal(older.xyz, xyz)
			np.testing.assert_array_equal(current.xyz, xyz + 1)
			for bad in (f"vcobj:{uid}@r1", f"vcobj:{uid}@g/0", f"vcobj:{uid}@/3"):
				with self.assertRaises(ValueError):
					stored_surfaces.read_surface(store, bad, volume=VOLUME)

	def test_approval_surface_scales_in_memory(self) -> None:
		with tempfile.TemporaryDirectory() as td:
			store = Path(td) / "objects.vcobj"
			uid, points = _source_surface(store)
			surface = approval_inpaint.load_approval_surface(f"vcobj:{uid}", store=store, volume=VOLUME)
			scaled = surface.scaled(2.0, base_shape_zyx=(8, 8, 8))
			np.testing.assert_allclose(scaled.xyz[1, 1], points[1, 1] * 2)
			self.assertTrue(np.all(scaled.xyz[0, 0] == -1.0))
			self.assertEqual(scaled.meta["scale"], [0.25, 0.25])  # a grid step twice as long in voxels
			self.assertEqual(scaled.meta["base_shape_zyx"], [8, 8, 8])


def _checkpoint(reference: dict) -> dict:
	rows, cols = np.indices((3, 4), dtype=np.float32)
	mesh = np.stack([cols + 5, rows + 6, np.full((3, 4), 7.0, np.float32)], axis=0)[:, None]
	return {
		"mesh_flat": torch.from_numpy(mesh),
		"_model_params_": {
			"mesh_step": 1, "winding_step": 1, "subsample_mesh": 1, "subsample_winding": 1, "scaledown": 1.0,
			"z_step_eff": 1, "volume_extent": None, "pyramid_d": False, "depth_windings": [0],
		},
		"_job_spec_": {"input_references": [reference], "model": None,
			"config": {"external_surfaces": [{"reference": reference}]}},
		"_corr_points_results_": {
			"points_list": [
				{"collection_id": 7, "point_id": 1, "valid": True, "p": [5.0, 6.0, 7.0], "winding_err": 0.25},
				{"collection_id": 7, "point_id": 2, "valid": False, "p": None},
			],
			"collection_avgs": {"7": 0.25},
		},
		"_atlas_control_points_results_": {
			"format": "lasagna_atlas_control_points_results", "version": 1,
			"records": [{"fiber_id": "f", "object_id": "f", "target_xyz": [1.0, 2.0, 3.0], "mesh_xyz": [1.0, 2.0, 4.0],
						 "model_h": 1.0, "model_w": 1.0, "distance": 1.0, "valid": True}],
		},
	}


@unittest.skipIf(ObjectStore is None, "vc_object_store is not importable")
class StoredExportTest(unittest.TestCase):
	def test_export_creates_a_sourced_surface_with_attached_results_and_model_artifact(self) -> None:
		with tempfile.TemporaryDirectory() as td:
			root = Path(td)
			store = root / "objects.vcobj"
			source, _ = _source_surface(store)
			ckpt = root / "model.pt"
			with ObjectStore.open(str(store), read_only=True) as db:
				torch.save(_checkpoint(object_reference(db.object(source))), ckpt)
			created: list[str] = []
			fit2tifxyz.main(["--input", str(ckpt), "--store", str(store), "--volume-uuid", VOLUME,
							 "--output-name", "fitted.tifxyz", "--single-segment", "--tag", "lasagna"],
							created_surfaces=created)
			self.assertEqual(len(created), 1)
			surface_uuid = created[0]

			with ObjectStore.open(str(store), read_only=True) as db:
				record = db.object(surface_uuid)
				self.assertEqual(record["name"], "fitted")
				self.assertEqual(record["tags"], ["lasagna"])
				self.assertEqual(record["meta"]["coordinates"], {"volume_uuid": VOLUME, "scale_factor": 1})
				self.assertEqual(record["meta"]["provenance"]["sources"], [source])
				self.assertNotIn("model_source", record["meta"])
				self.assertNotIn("path", record["meta"]["atlas_control_points_results"])
				geometry = db.geometry_revision(db.current_geometry(surface_uuid)["uuid"])
				self.assertEqual({a["name"] for a in geometry["arrays"]}, {"points", "d"})

				# Results hang off the surface as attached data, never as graph links.
				self.assertEqual(db.neighbours(surface_uuid), [])
			corr = attached_document(store, surface_uuid, fit2tifxyz.CORR_RESULTS_ROLE)
			atlas = attached_document(store, surface_uuid, fit2tifxyz.ATLAS_RESULTS_ROLE)
			self.assertEqual(corr["collection_avgs"], {"7": 0.25})
			self.assertEqual([p["point_id"] for p in corr["points_list"]], [1, 2])
			self.assertEqual(corr["points_list"][0]["winding_err"], 0.25)
			self.assertEqual(atlas["records"][0]["target_xyz"], [1.0, 2.0, 3.0])

			# The model checkpoint and parameters are published as the surface's artifacts.
			session = stored_surfaces.StoreSession(store, author="test", shared=True)
			try:
				cache = root / "artifact-cache"
				ref = session.decode_reference(dict(uuid=surface_uuid))
				self.assertEqual(session.artifact_file(ref, "model.pt", cache).read_bytes(), ckpt.read_bytes())
				self.assertTrue(session.artifact_file(ref, "model_params.json", cache).is_file())
			finally:
				session.close()
			self.assertFalse(Path(str(store) + ".artifacts", surface_uuid).exists())

	def test_new_surfaces_take_the_next_free_name_of_their_root(self) -> None:
		with tempfile.TemporaryDirectory() as td:
			store = Path(td) / "objects.vcobj"
			with ObjectStore.create(str(store), author="test") as db:
				for name in ("sheet_v001", "sheet_v004", "sheet_off1", "other_v009"):
					db.upsert_object(dict(kind="surface", name=name, meta={}))
			self.assertEqual(stored_surfaces.next_free_name(store, "sheet", "version"), "sheet_v005")
			self.assertEqual(stored_surfaces.next_free_name(store, "sheet", "offset"), "sheet_off2")
			self.assertEqual(stored_surfaces.next_free_name(store, "new", "version"), "new_v001")

	def test_export_takes_exactly_one_destination(self) -> None:
		for argv in (["--input", "m.pt"], ["--input", "m.pt", "--output", "o", "--store", "s"],
					 ["--input", "m.pt", "--store", "s"]):
			with self.assertRaises(SystemExit):
				fit2tifxyz.main(argv)


@unittest.skipIf(ObjectStore is None, "vc_object_store is not importable")
class ServiceStoreRequestTest(unittest.TestCase):
	def setUp(self) -> None:
		self._td = tempfile.TemporaryDirectory()
		self.store_path = Path(self._td.name) / "objects.vcobj"
		self.source, _ = _source_surface(self.store_path)
		fit_service._store = stored_surfaces.StoreSession(self.store_path, author="service")
		self._old_object_store_dir = fit_service._object_store_dir
		fit_service._object_store_dir = Path(self._td.name) / "cache"

	def tearDown(self) -> None:
		fit_service._store.close()
		fit_service._store = None
		fit_service._object_store_dir = self._old_object_store_dir
		self._td.cleanup()

	def test_requests_name_the_service_store_and_a_volume(self) -> None:
		store_uuid = fit_service._store.store_uuid
		self.assertIsNone(fit_service._request_store_volume({}))
		self.assertEqual(fit_service._request_store_volume({"store_uuid": store_uuid, "volume_uuid": VOLUME}), VOLUME)
		with self.assertRaisesRegex(ValueError, "this service writes to"):
			fit_service._request_store_volume({"store_uuid": "other", "volume_uuid": VOLUME})
		with self.assertRaisesRegex(ValueError, "volume_uuid"):
			fit_service._request_store_volume({"store_uuid": store_uuid})

	def test_correction_collections_resolve_through_the_store(self) -> None:
		with ObjectStore.open(str(self.store_path), author="test") as db:
			uid = db.upsert_object(dict(kind="point_collection", name="seeds", meta=dict(
				coordinates=dict(volume_uuid=VOLUME, scale_factor=2),
				source=dict(color=[1, 0, 0], point_attributes=[{"wind_a": 1.0}, {"wind_a": 2.0}]))))
			db.put_geometry(uid, dict(kind="point_collection", arrays=[
				_array("points", np.array([[2, 4, 6], [8, 10, 12]], "<f8")), _array("point_ids", np.array([5, 7], "<i8"))]))
		store_uuid = fit_service._store.store_uuid
		resolved = fit_service._body_with_resolved_job_spec({
			"store_uuid": store_uuid, "volume_uuid": VOLUME,
			"job_spec": {"config": {"corr_point_collections": [uid]}}})
		config = resolved["config"]
		self.assertNotIn("corr_point_collections", config)
		collection = config["corr_points"]["collections"]["1"]
		self.assertEqual((collection["uuid"], collection["name"], collection["color"]), (uid, "seeds", [1, 0, 0]))
		self.assertEqual(collection["points"]["7"], {"wind_a": 2.0, "p": [4.0, 5.0, 6.0]})
		with ObjectStore.open(str(self.store_path), author="test") as db:
			selected = object_reference(db.object(uid))
			self.assertEqual(collection["reference"], selected)
			self.assertEqual(resolved["_job_spec_"]["input_references"], [selected])
			db.set_meta(uid, "edited_after_materialization", True)
		# Export uses the saved selections, never today's correction head.
		self.assertEqual(resolved["_job_spec_"]["input_references"], [selected])
		self.assertEqual(fit2tifxyz._stored_sources(resolved["_job_spec_"]), (uid,))

	def test_export_sources_require_captured_references(self) -> None:
		self.assertEqual(fit2tifxyz._stored_sources({"input_references": []}), ())
		with self.assertRaisesRegex(ValueError, "regenerate"):
			fit2tifxyz._stored_sources({"config": {"surfaces": [{"type": "surface", "uuid": self.source}]}})

	def test_surface_and_model_refs_resolve_through_the_store(self) -> None:
		with self.assertRaisesRegex(ValueError, "no Lasagna model"):
			fit_service._body_with_resolved_job_spec({"job_spec": {
				"model": {"type": "surface", "uuid": self.source}, "config": {}}})
		with self.assertRaisesRegex(ValueError, f"surface {self.source} has no Lasagna model"):
			fit_service._stored_model_path(f"vcobj:{self.source}")
		checkpoint = Path(self._td.name) / "checkpoint.pt"
		checkpoint.write_bytes(b"checkpoint")
		receipt = fit_service._store.publish_artifacts(self.source, {"model.pt": checkpoint})
		self.assertEqual(len(receipt["objects"]), 1)
		reference = fit_service._store.decode_reference(dict(uuid=self.source))
		resolved = fit_service._body_with_resolved_job_spec({"job_spec": {
			"model": {"type": "surface", "uuid": self.source},
			"config": {"external_surfaces": [{"type": "surface", "reference": reference}]},
		}})
		self.assertEqual(resolved["config"]["external_surfaces"][0]["path"], reference)
		with self.assertRaisesRegex(ValueError, "numbered version"):
			fit_service._body_with_resolved_job_spec({"job_spec": {
				"model": {"type": "surface", "uuid": self.source},
				"config": {"external_surfaces": [{"type": "surface", "uuid": self.source, "revision": "r1"}]}}})
		model = Path(resolved["model_input"])
		self.assertEqual(model.read_bytes(), b"checkpoint")
		self.assertNotEqual(model, checkpoint)  # a cached copy of the published bytes
		for model_ref in (reference, {"reference": reference}, {"uuid": self.source}, f"vcobj:{self.source}"):
			with self.subTest(model_ref=model_ref):
				resolved = fit_service._body_with_resolved_job_spec({"job_spec": {"model": model_ref, "config": {}}})
				self.assertEqual(Path(resolved["model_input"]).read_bytes(), b"checkpoint")

	def test_content_addressed_artifacts_and_path_configs_keep_their_own_route(self) -> None:
		payload = b"content-addressed model"
		artifact = dict(type="lasagna_model", name="model.pt", hash=fit_service._hash_bytes(payload))
		fit_service._store_uploaded_object({"object": artifact, "data": fit_service.base64.b64encode(payload).decode("ascii")})
		config = {"external_surfaces": [{"path": "/data/surface.tifxyz"}]}
		resolved = fit_service._body_with_resolved_job_spec({"job_spec": {"model": artifact, "config": config}})
		self.assertEqual(Path(resolved["model_input"]).read_bytes(), payload)
		self.assertEqual(resolved["config"]["external_surfaces"], config["external_surfaces"])

	def test_a_shared_store_is_written_through_short_lived_connections(self) -> None:
		shared = stored_surfaces.StoreSession(self.store_path, author="service", shared=True)
		try:
			self.assertEqual(shared.store_uuid, fit_service._store.store_uuid)
			checkpoint = Path(self._td.name) / "shared.pt"
			checkpoint.write_bytes(b"shared")
			shared.publish_artifacts(self.source, {"model.pt": checkpoint})  # read-only session, written anyway
			ref = fit_service._store.decode_reference(dict(uuid=self.source))
			cached = fit_service._store.artifact_file(ref, "model.pt", Path(self._td.name) / "cache")
			self.assertEqual(cached.read_bytes(), b"shared")
			created = create_surface(
				shared.location, np.zeros((2, 2, 3), np.float32), name="written", scale=1.0, volume=VOLUME, tool="test")
		finally:
			shared.close()
		with ObjectStore.open(str(self.store_path), read_only=True) as db:
			self.assertEqual(db.object(created)["name"], "written")

	def test_model_reference_reads_the_selected_checkpoint_after_replacement_and_deletion(self) -> None:
		checkpoint = Path(self._td.name) / "checkpoint.pt"
		checkpoint.write_bytes(b"original model")
		first = fit_service._store.publish_artifacts(self.source, {"model.pt": checkpoint})
		ref = dict(type="surface", uuid=self.source, store=first["store_uuid"],
				   generation=first["generation"], version=first["objects"][self.source])
		checkpoint.write_bytes(b"new model")
		fit_service._store.publish_artifacts(self.source, {"model.pt": checkpoint})
		self.assertEqual(fit_service._stored_model_path(ref).read_bytes(), b"original model")
		self.assertEqual(fit_service._stored_model_path(dict(type="surface", uuid=self.source)).read_bytes(), b"new model")
		with ObjectStore.open(str(self.store_path), author="test") as db:
			db.delete_object(self.source)
		self.assertEqual(fit_service._stored_model_path(ref).read_bytes(), b"original model")
		with self.assertRaisesRegex(ValueError, "numbered version"):
			fit_service._stored_model_path(dict(type="surface", uuid=self.source, revision="old"))



SERVER = os.environ.get("VC_OBJECT_SERVER")


@unittest.skipIf(ObjectStore is None or not SERVER, "needs vc_object_store and VC_OBJECT_SERVER (a vc_object_server binary)")
class HostedStoreTest(unittest.TestCase):
	"""A Lasagna service on another machine works on VC3D's store through its vc_object_server endpoint."""

	def setUp(self) -> None:
		import re
		import subprocess
		self._td = tempfile.TemporaryDirectory()
		self.root = Path(self._td.name)
		self.store_path = self.root / "objects.vcobj"
		self.source, self.points = _source_surface(self.store_path)
		self.server = subprocess.Popen([SERVER, str(self.store_path), "--port", "0"], stderr=subprocess.PIPE, text=True)
		line = self.server.stderr.readline()
		port = re.search(r":(\d+)\s*$", line)
		if not port:
			self.server.terminate()
			raise RuntimeError(f"vc_object_server did not start: {line}")
		self.endpoint = f"http://127.0.0.1:{port.group(1)}"

	def tearDown(self) -> None:
		self.server.terminate()
		self.server.wait(timeout=30)
		self._td.cleanup()

	def test_fit_reads_and_export_writes_through_the_endpoint(self) -> None:
		ref = f"vcobj:{self.source}"
		xyz, valid, meta = tifxyz_io.load_tifxyz(ref, store=self.endpoint, volume=VOLUME)
		np.testing.assert_allclose(xyz[1, 1].numpy(), self.points[1, 1])
		self.assertFalse(bool(valid[0, 0]))
		surface = approval_inpaint.load_approval_surface(ref, store=self.endpoint, volume=VOLUME)
		self.assertEqual(float(surface.d[1, 1]), 3.0)
		self.assertEqual(stored_surfaces.next_free_name(self.endpoint, "source", "version"), "source_v001")

		ckpt = self.root / "model.pt"
		with ObjectStore.open(str(self.store_path), read_only=True) as db:
			torch.save(_checkpoint(object_reference(db.object(self.source))), ckpt)
		created: list[str] = []
		fit2tifxyz.main(["--input", str(ckpt), "--store", self.endpoint, "--volume-uuid", VOLUME,
						 "--output-name", "fitted.tifxyz", "--single-segment"], created_surfaces=created)
		[surface_uuid] = created
		# The service's file shows the results (read-only opens are allowed while it is served).
		with ObjectStore.open(str(self.store_path), read_only=True) as db:
			record = db.object(surface_uuid)
			self.assertEqual(record["name"], "fitted")
			self.assertEqual(record["meta"]["provenance"]["sources"], [self.source])
		corr = attached_document(self.endpoint, surface_uuid, fit2tifxyz.CORR_RESULTS_ROLE)
		self.assertEqual([p["point_id"] for p in corr["points_list"]], [1, 2])

		# A service started on the endpoint resolves the published model.
		fit_service._store = stored_surfaces.StoreSession(self.endpoint, author="service")
		try:
			with ObjectStore.open(str(self.store_path), read_only=True) as db:
				self.assertEqual(fit_service._store.store_uuid, db.status()["store"])
			resolved = fit_service._body_with_resolved_job_spec({"job_spec": {
				"model": {"type": "surface", "uuid": surface_uuid}, "config": {}}})
			self.assertEqual(Path(resolved["model_input"]).read_bytes(), ckpt.read_bytes())
		finally:
			fit_service._store.close()
			fit_service._store = None


if __name__ == "__main__":
	unittest.main()
