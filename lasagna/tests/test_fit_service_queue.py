import time
import threading
import tempfile
import unittest
from unittest import mock

import fit_service


class FitServiceQueueTest(unittest.TestCase):
	def wait_for_state(self, queue, job_id, state, timeout=1.0):
		deadline = time.time() + timeout
		while time.time() < deadline:
			snap = queue.snapshot(job_id)
			if snap and snap["state"] == state:
				return snap
			time.sleep(0.01)
		self.fail(f"job {job_id} did not reach {state}")

	def test_fifo_sequence_and_stable_job_ids(self):
		queue = fit_service._JobQueue()
		seen = []

		def fake_run(job, body):
			seen.append((job.job_id, body["name"]))
			job.set_finished("/tmp/out")

		with mock.patch.object(fit_service, "_run_optimization", side_effect=fake_run):
			j1 = queue.create_upload(source="a", config_name="one.json")
			j2 = queue.create_upload(source="b", config_name="two.json")
			queue.enqueue_body(j1, {"name": "first"})
			queue.enqueue_body(j2, {"name": "second"})
			self.wait_for_state(queue, j2.job_id, "finished")

		self.assertNotEqual(j1.job_id, j2.job_id)
		self.assertEqual(j1.sequence + 1, j2.sequence)
		self.assertEqual(seen, [(j1.job_id, "first"), (j2.job_id, "second")])

	def test_created_job_reports_upload_until_body_is_ready(self):
		queue = fit_service._JobQueue()
		j1 = queue.create_upload(source="a", config_name="one.json")
		self.assertEqual(queue.snapshot(j1.job_id)["state"], "upload")
		self.assertIn("queue_generation", queue.legacy_status())
		self.assertEqual(queue.snapshot_response()["queue_generation"], queue.generation)

	def test_enqueue_body_reports_requested_output_name(self):
		queue = fit_service._JobQueue()
		j1 = queue.create_upload(source="a", config_name="one.json")
		queue.enqueue_body(j1, {"output_name": "sheet_042"})

		self.assertEqual(queue.snapshot(j1.job_id)["output_name"], "sheet_042")
		self.assertEqual(queue.snapshot_response()["jobs"][0]["output_name"], "sheet_042")

	def test_reorder_waiting_jobs_changes_execution_order(self):
		queue = fit_service._JobQueue()
		seen = []
		release_first = threading.Event()

		def fake_run(job, body):
			seen.append(body["name"])
			if body["name"] == "first":
				release_first.wait(1.0)
			job.set_finished("/tmp/out")

		with mock.patch.object(fit_service, "_run_optimization", side_effect=fake_run):
			j1 = queue.create_upload(source="a", config_name="one.json")
			j2 = queue.create_upload(source="b", config_name="two.json")
			j3 = queue.create_upload(source="c", config_name="three.json")
			queue.enqueue_body(j1, {"name": "first"})
			queue.enqueue_body(j2, {"name": "second"})
			queue.enqueue_body(j3, {"name": "third"})
			ok, _ = queue.reorder({"job_id": j3.job_id, "before_job_id": j2.job_id})
			self.assertTrue(ok)
			release_first.set()
			self.wait_for_state(queue, j3.job_id, "finished")

		self.assertEqual(seen, ["first", "third", "second"])

	def test_cancel_waiting_job(self):
		queue = fit_service._JobQueue()
		release_first = threading.Event()

		def fake_run(job, body):
			release_first.wait(1.0)
			job.set_finished("/tmp/out")

		with mock.patch.object(fit_service, "_run_optimization", side_effect=fake_run):
			j1 = queue.create_upload(source="a", config_name="one.json")
			j2 = queue.create_upload(source="b", config_name="two.json")
			queue.enqueue_body(j1, {"name": "first"})
			queue.enqueue_body(j2, {"name": "second"})
			ok, msg = queue.cancel(j2.job_id)
			release_first.set()

		self.assertTrue(ok)
		self.assertEqual(msg, "cancelled")
		self.assertEqual(queue.snapshot(j2.job_id)["state"], "cancelled")

	def test_reject_reorder_finished_job(self):
		queue = fit_service._JobQueue()

		def fake_run(job, body):
			job.set_finished("/tmp/out")

		with mock.patch.object(fit_service, "_run_optimization", side_effect=fake_run):
			j1 = queue.create_upload(source="a", config_name="one.json")
			queue.enqueue_body(j1, {"name": "first"})
			self.wait_for_state(queue, j1.job_id, "finished")

		ok, msg = queue.reorder({"job_id": j1.job_id})
		self.assertFalse(ok)
		self.assertIn("not reorderable", msg)

	def test_reject_reorder_upload_job(self):
		queue = fit_service._JobQueue()
		j1 = queue.create_upload(source="a", config_name="one.json")
		ok, msg = queue.reorder({"job_id": j1.job_id})
		self.assertFalse(ok)
		self.assertIn("not reorderable", msg)


class FitServiceObjectStoreTest(unittest.TestCase):
	def test_model_upload_validates_hash_and_resolves_path(self):
		with tempfile.TemporaryDirectory() as td:
			old_store = fit_service._object_store_dir
			fit_service._object_store_dir = fit_service.Path(td)
			try:
				data = b"checkpoint"
				ref = {
					"type": "lasagna_model",
					"name": "sheet_v001.tifxyz/model.pt",
					"hash": fit_service._hash_bytes(data),
				}
				stored = fit_service._store_uploaded_object({
					"object": ref,
					"data": fit_service.base64.b64encode(data).decode("ascii"),
				})
				self.assertEqual(stored, ref)
				self.assertTrue(fit_service._object_present(ref))
				self.assertEqual(fit_service._resolve_object_ref(ref).read_bytes(), data)

				bad = dict(ref)
				bad["hash"] = "md5:" + "0" * 32
				with self.assertRaises(ValueError):
					fit_service._store_uploaded_object({
						"object": bad,
						"data": fit_service.base64.b64encode(data).decode("ascii"),
					})
			finally:
				fit_service._object_store_dir = old_store

	def test_segment_upload_uses_manifest_hash(self):
		with tempfile.TemporaryDirectory() as td:
			old_store = fit_service._object_store_dir
			fit_service._object_store_dir = fit_service.Path(td)
			try:
				files = {"z.tif": b"z", "meta.json": b"{}", "x.tif": b"x"}
				ref = {
					"type": "tifxyz_segment",
					"name": "reference_surface.tifxyz",
					"hash": fit_service._segment_manifest_hash(files),
				}
				fit_service._store_uploaded_object({
					"object": ref,
					"files": {
						name: fit_service.base64.b64encode(data).decode("ascii")
						for name, data in files.items()
					},
				})
				path = fit_service._resolve_object_ref(ref)
				self.assertTrue((path / "x.tif").is_file())
				self.assertEqual((path / "meta.json").read_bytes(), b"{}")
			finally:
				fit_service._object_store_dir = old_store

	def test_job_spec_resolves_model_and_one_linked_surface(self):
		with tempfile.TemporaryDirectory() as td:
			old_store = fit_service._object_store_dir
			fit_service._object_store_dir = fit_service.Path(td)
			try:
				model = b"model"
				model_ref = {
					"type": "lasagna_model",
					"name": "sheet_v001.tifxyz/model.pt",
					"hash": fit_service._hash_bytes(model),
				}
				fit_service._store_uploaded_object({
					"object": model_ref,
					"data": fit_service.base64.b64encode(model).decode("ascii"),
				})
				seg_files = {"x.tif": b"x", "y.tif": b"y", "z.tif": b"z", "meta.json": b"{}"}
				seg_ref = {
					"type": "tifxyz_segment",
					"name": "reference_surface.tifxyz",
					"hash": fit_service._segment_manifest_hash(seg_files),
				}
				fit_service._store_uploaded_object({
					"object": seg_ref,
					"files": {
						name: fit_service.base64.b64encode(data).decode("ascii")
						for name, data in seg_files.items()
					},
				})
				body = fit_service._body_with_resolved_job_spec({
					"job_spec": {
						"model": model_ref,
						"linked_surfaces": [seg_ref],
						"config": {"args": {"model-init": "model"}, "offset_value": 2.5},
					}
				})
				self.assertEqual(fit_service.Path(body["model_input"]).read_bytes(), model)
				self.assertEqual(body["config"]["external_surfaces"][0]["offset"], 2.5)
				self.assertTrue(fit_service.Path(body["config"]["external_surfaces"][0]["path"]).is_dir())
			finally:
				fit_service._object_store_dir = old_store


if __name__ == "__main__":
	unittest.main()
