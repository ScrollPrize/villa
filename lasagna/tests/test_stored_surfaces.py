"""Fits read input surfaces from a VC object store and export results into it (the fit service as VC3D uses it)."""
from __future__ import annotations

import os
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
	from vc_object_store import ObjectStore
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


def _source_surface(path: Path, *, scale_factor: float = 1.0) -> tuple[str, np.ndarray]:
	rows, cols = np.mgrid[:5, :6]
	points = np.stack([cols * 2.0 + 10, rows * 2.0 + 20, np.full(rows.shape, 30.0)], axis=-1).astype("<f4")
	points[0, 0] = -1
	valid = np.full((5, 6), 1, "u1")
	valid[4, 5] = 0
	approval = np.zeros((5, 6), "u1")
	approval[1:4, 1:5] = 255
	approval[2, 2] = 0
	with ObjectStore.create(str(path), author="test") as db:
		uid = db.upsert_object(dict(kind="surface", name="source", meta=dict(
			coordinates=dict(volume_uuid=VOLUME, scale_factor=scale_factor))))
		db.put_geometry(uid, dict(kind="surface", attributes=dict(scale=[0.5, 0.5]), arrays=[
			_array("points", points), _array("valid", valid), _array("approval", approval),
			_array("d", np.full((5, 6), 3.0, "<f4"))]))
	return uid, points


@unittest.skipIf(ObjectStore is None, "vc_object_store is not importable")
class StoredSurfaceReadTest(unittest.TestCase):
	def test_fit_loaders_read_stored_surfaces_in_volume_voxels(self) -> None:
		with tempfile.TemporaryDirectory() as td:
			store = Path(td) / "objects.vcobj"
			uid, points = _source_surface(store, scale_factor=2.0)
			ref = stored_surfaces.surface_ref(uid)
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
		# Any -1 or non-finite coordinate is missing, read and written; z = 0 is a slice.
		with tempfile.TemporaryDirectory() as td:
			store = Path(td) / "objects.vcobj"
			ObjectStore.create(str(store), author="test").close()
			xyz = np.full((1, 3, 3), 30.0, "<f4")
			xyz[0, 0] = (5, -1, 30)
			xyz[0, 1] = (6, 7, 0)
			uid = create_surface(store, xyz, name="s", scale=1.0, volume=VOLUME, tool="test")
			surface = stored_surfaces.read_surface(store, stored_surfaces.surface_ref(uid), volume=VOLUME)
			self.assertEqual(surface.xyz[0, 0].tolist(), [-1.0, -1.0, -1.0])
			self.assertEqual(surface.xyz[0, 1].tolist(), [6.0, 7.0, 0.0])

	def test_approval_surface_scales_in_memory(self) -> None:
		with tempfile.TemporaryDirectory() as td:
			store = Path(td) / "objects.vcobj"
			uid, points = _source_surface(store)
			surface = approval_inpaint.load_approval_surface(stored_surfaces.surface_ref(uid), store=store, volume=VOLUME)
			scaled = surface.scaled(2.0, base_shape_zyx=(8, 8, 8))
			np.testing.assert_allclose(scaled.xyz[1, 1], points[1, 1] * 2)
			self.assertTrue(np.all(scaled.xyz[0, 0] == -1.0))
			self.assertEqual(scaled.meta["scale"], [0.25, 0.25])  # a grid step twice as long in voxels
			self.assertEqual(scaled.meta["base_shape_zyx"], [8, 8, 8])


def _checkpoint(source_uuid: str) -> dict:
	rows, cols = np.indices((3, 4), dtype=np.float32)
	mesh = np.stack([cols + 5, rows + 6, np.full((3, 4), 7.0, np.float32)], axis=0)[:, None]
	return {
		"mesh_flat": torch.from_numpy(mesh),
		"_model_params_": {
			"mesh_step": 1, "winding_step": 1, "subsample_mesh": 1, "subsample_winding": 1, "scaledown": 1.0,
			"z_step_eff": 1, "volume_extent": None, "pyramid_d": False, "depth_windings": [0],
		},
		"_job_spec_": {"model": None, "config": {"external_surfaces": [{"type": "surface", "uuid": source_uuid}]}},
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
			torch.save(_checkpoint(source), ckpt)
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

			artifacts = Path(str(store) + ".artifacts") / surface_uuid
			self.assertEqual((artifacts / "model.pt").read_bytes(), ckpt.read_bytes())
			self.assertTrue((artifacts / "model_params.json").is_file())

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

	def tearDown(self) -> None:
		fit_service._store.close()
		fit_service._store = None
		self._td.cleanup()

	def test_requests_name_the_service_store_and_a_volume(self) -> None:
		store_uuid = fit_service._store.store_uuid
		self.assertIsNone(fit_service._request_store_volume({}))
		self.assertEqual(fit_service._request_store_volume({"store_uuid": store_uuid, "volume_uuid": VOLUME}), VOLUME)
		with self.assertRaisesRegex(ValueError, "this service writes to"):
			fit_service._request_store_volume({"store_uuid": "other", "volume_uuid": VOLUME})
		with self.assertRaisesRegex(ValueError, "volume_uuid"):
			fit_service._request_store_volume({"store_uuid": store_uuid})

	def test_surface_and_model_refs_resolve_through_the_store(self) -> None:
		model = fit_service._store.artifact_directory(self.source) / "model.pt"
		model.write_bytes(b"checkpoint")
		resolved = fit_service._body_with_resolved_job_spec({"job_spec": {
			"model": {"type": "surface", "uuid": self.source},
			"config": {"external_surfaces": [{"type": "surface", "uuid": self.source, "revision": "r1"}]},
		}})
		self.assertEqual(resolved["config"]["external_surfaces"][0]["path"], f"vcobj:{self.source}@r1")
		self.assertEqual(resolved["model_input"], str(model))
		model.unlink()
		with self.assertRaisesRegex(ValueError, "no Lasagna model"):
			fit_service._body_with_resolved_job_spec({"job_spec": {
				"model": {"type": "surface", "uuid": self.source}, "config": {}}})

	def test_a_shared_store_is_written_but_synced_by_its_owner(self) -> None:
		shared = stored_surfaces.StoreSession(self.store_path, author="service", shared=True)
		try:
			self.assertEqual(shared.store_uuid, fit_service._store.store_uuid)
			self.assertEqual(shared.artifact_directory(self.source), fit_service._store.artifact_directory(self.source))
			shared.sync()  # the owning session syncs; nothing to wait for here
			created = create_surface(
				shared.path, np.zeros((2, 2, 3), np.float32), name="written", scale=1.0, volume=VOLUME, tool="test")
		finally:
			shared.close()
		with ObjectStore.open(str(self.store_path), read_only=True) as db:
			self.assertEqual(db.object(created)["name"], "written")


if __name__ == "__main__":
	unittest.main()
