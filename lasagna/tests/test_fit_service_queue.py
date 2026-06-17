import time
import threading
import tempfile
import unittest
import io
import os
import types
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

	def run_laplace_rank_job(self, request):
		job = fit_service._JobState(
			job_id=fit_service.uuid.uuid4().hex[:12],
			sequence=1,
			source="vc3d",
			config_name="laplace_rank",
		)
		self.addCleanup(job.cleanup_tmp_dirs, keep_results=False)
		fit_service._run_laplace_rank_job(job, request)
		return job, job.snapshot()

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

	def test_laplace_rank_direct_endpoint_returns_404(self):
		handler = object.__new__(fit_service._Handler)
		handler._validate_api_version = lambda: True
		handler._job_path_parts = lambda: ["laplace", "rank"]
		sent = []
		handler._send_json = lambda body, status=200: sent.append((status, body))

		fit_service._Handler.do_POST(handler)

		self.assertEqual(sent, [(404, {"error": "not found"})])

	def test_laplace_rank_rejects_malformed_queued_request_before_import(self):
		with mock.patch.object(fit_service.importlib, "import_module") as import_module:
			_job, snap = self.run_laplace_rank_job({
				"job_type": "laplace_rank",
				"manifest": "/tmp/data.json",
				"jobs": [{}],
			})

		self.assertEqual(snap["state"], "error")
		self.assertIn("side_a", snap["error"])
		import_module.assert_not_called()

	def test_laplace_rank_missing_binding_errors_queued_job(self):
		request = {
			"job_type": "laplace_rank",
			"manifest": "/tmp/data.lasagna.json",
			"jobs": [{"id": "a", "side_a": [[1, 2, 3]], "side_b": [[4, 5, 6]]}],
			"options": {},
		}
		with mock.patch.object(fit_service.importlib, "import_module", side_effect=ImportError("missing")):
			_job, snap = self.run_laplace_rank_job(request)

		self.assertEqual(snap["state"], "error")
		self.assertIn("vc_lasagna_amgx", snap["error"])

	def test_laplace_rank_queued_job_writes_binding_response_without_ratios(self):
		request = {
			"job_type": "laplace_rank",
			"manifest": "/tmp/data.lasagna.json",
			"jobs": [{"id": "cache-key", "side_a": [[1, 2, 3]], "side_b": [[4, 5, 6], [7, 8, 9]]}],
			"options": {"debug_dir": "/tmp/laplace-debug"},
		}
		binding_response = {
			"results": [{
				"id": "cache-key",
				"status": "success",
				"selected_lambda": 0.25,
				"values": [{"value": 0.5}],
				"debug_dir": "/tmp/laplace-debug/request/cache-key",
			}]
		}
		module = types.SimpleNamespace(rank_snap_pairs=mock.Mock(return_value=binding_response))
		with mock.patch.object(fit_service.importlib, "import_module", return_value=module):
			_job, snap = self.run_laplace_rank_job(request)

		self.assertEqual(snap["state"], "finished")
		sent = module.rank_snap_pairs.call_args.args[0]
		self.assertEqual(sent["manifest"], request["manifest"])
		self.assertEqual(sent["options"]["debug_dir"], request["options"]["debug_dir"])
		self.assertIn("progress_callback", module.rank_snap_pairs.call_args.kwargs)
		result_path = fit_service.Path(snap["output_dir"]) / "rank_result.json"
		body = fit_service.json.loads(result_path.read_text(encoding="utf-8"))
		self.assertEqual(body["results"], binding_response["results"])
		self.assertNotIn("ratios", body["results"][0])

	def test_laplace_rank_resolves_relative_debug_dir_against_server_cwd(self):
		request = {
			"job_type": "laplace_rank",
			"manifest": "/tmp/data.lasagna.json",
			"jobs": [{"id": "cache-key", "side_a": [[1, 2, 3]], "side_b": [[4, 5, 6]]}],
			"options": {"debug_dir": "laplace_rank_debug/atlas_snap"},
		}
		module = types.SimpleNamespace(rank_snap_pairs=mock.Mock(return_value={"results": []}))
		old_cwd = os.getcwd()
		with tempfile.TemporaryDirectory() as td:
			try:
				os.chdir(td)
				with mock.patch.object(fit_service.importlib, "import_module", return_value=module):
					_job, snap = self.run_laplace_rank_job(request)
			finally:
				os.chdir(old_cwd)

		self.assertEqual(snap["state"], "finished")
		sent = module.rank_snap_pairs.call_args.args[0]
		self.assertEqual(
			sent["options"]["debug_dir"],
			str((fit_service.Path(td) / "laplace_rank_debug" / "atlas_snap").resolve()),
		)

	def test_laplace_rank_defaults_debug_dir_to_server_cwd(self):
		request = {
			"job_type": "laplace_rank",
			"manifest": "/tmp/data.lasagna.json",
			"jobs": [{"id": "cache-key", "side_a": [[1, 2, 3]], "side_b": [[4, 5, 6]]}],
			"options": {},
		}
		module = types.SimpleNamespace(rank_snap_pairs=mock.Mock(return_value={"results": []}))
		old_cwd = os.getcwd()
		with tempfile.TemporaryDirectory() as td:
			try:
				os.chdir(td)
				with mock.patch.object(fit_service.importlib, "import_module", return_value=module):
					_job, snap = self.run_laplace_rank_job(request)
			finally:
				os.chdir(old_cwd)

		self.assertEqual(snap["state"], "finished")
		sent = module.rank_snap_pairs.call_args.args[0]
		self.assertEqual(
			sent["options"]["debug_dir"],
			str((fit_service.Path(td) / "laplace_rank_debug" / "atlas_snap").resolve()),
		)

	def test_laplace_rank_resolves_relative_manifest_against_data_dir(self):
		with tempfile.TemporaryDirectory() as td:
			old_data_dir = fit_service._data_dir
			fit_service._data_dir = td
			try:
				manifest = fit_service.Path(td) / "data.lasagna.json"
				manifest.write_text("{}", encoding="utf-8")
				request = {
					"manifest": "data.lasagna.json",
					"jobs": [{"id": "cache-key", "side_a": [[1, 2, 3]], "side_b": [[4, 5, 6]]}],
				}
				module = types.SimpleNamespace(rank_snap_pairs=mock.Mock(return_value={"results": []}))
				with mock.patch.object(fit_service.importlib, "import_module", return_value=module):
					_job, snap = self.run_laplace_rank_job(request)

				self.assertEqual(snap["state"], "finished")
				sent = module.rank_snap_pairs.call_args.args[0]
				self.assertEqual(sent["manifest"], str(manifest.resolve()))
				self.assertEqual(request["manifest"], "data.lasagna.json")
			finally:
				fit_service._data_dir = old_data_dir

	def test_laplace_rank_recovers_client_absolute_manifest_with_data_dir_basename(self):
		with tempfile.TemporaryDirectory() as td:
			old_data_dir = fit_service._data_dir
			fit_service._data_dir = td
			try:
				manifest = fit_service.Path(td) / "data.lasagna.json"
				manifest.write_text("{}", encoding="utf-8")
				request = {
					"manifest": "/client/only/path/data.lasagna.json",
					"jobs": [{"id": "cache-key", "side_a": [[1, 2, 3]], "side_b": [[4, 5, 6]]}],
				}
				module = types.SimpleNamespace(rank_snap_pairs=mock.Mock(return_value={"results": []}))
				with mock.patch.object(fit_service.importlib, "import_module", return_value=module):
					_job, snap = self.run_laplace_rank_job(request)

				self.assertEqual(snap["state"], "finished")
				sent = module.rank_snap_pairs.call_args.args[0]
				self.assertEqual(sent["manifest"], str(manifest.resolve()))
				self.assertEqual(request["manifest"], "/client/only/path/data.lasagna.json")
			finally:
				fit_service._data_dir = old_data_dir

	def test_queued_laplace_rank_records_out_of_order_events(self):
		queue = fit_service._JobQueue()
		request = {
			"job_type": "laplace_rank",
			"manifest": "/tmp/data.lasagna.json",
			"jobs": [
				{"id": "term-0", "side_a": [[1, 2, 3]], "side_b": [[4, 5, 6]]},
				{"id": "term-1", "side_a": [[7, 8, 9]], "side_b": [[10, 11, 12]]},
			],
			"options": {},
		}

		def fake_rank(body, progress_callback=None):
			results = [
				{"id": "term-0", "status": "success", "values": []},
				{"id": "term-1", "status": "success", "values": []},
			]
			progress_callback({
				"index": 1,
				"id": "term-1",
				"result": results[1],
				"completed": 1,
				"total": 2,
			})
			progress_callback({
				"index": 0,
				"id": "term-0",
				"result": results[0],
				"completed": 2,
				"total": 2,
			})
			return {"results": results}

		with mock.patch.object(fit_service, "_rank_laplace_snap_pairs", side_effect=fake_rank):
			job = queue.create_upload(source="vc3d", config_name="")
			queue.enqueue_body(job, request)
			snap = self.wait_for_state(queue, job.job_id, "finished")

		self.assertEqual(snap["config_name"], "laplace_rank")
		events = job.events_after(0)
		self.assertEqual([event["seq"] for event in events], [1, 2])
		self.assertEqual([event["index"] for event in events], [1, 0])
		self.assertEqual(events[0]["type"], "laplace_rank_result")
		result_path = fit_service.Path(snap["output_dir"]) / "rank_result.json"
		self.assertTrue(result_path.is_file())
		result = fit_service.json.loads(result_path.read_text(encoding="utf-8"))
		self.assertEqual([item["id"] for item in result["results"]], ["term-0", "term-1"])

	def test_enqueue_body_reports_requested_output_name(self):
		queue = fit_service._JobQueue()
		j1 = queue.create_upload(source="a", config_name="one.json")
		queue.enqueue_body(j1, {"output_name": "sheet_042"})

		self.assertEqual(queue.snapshot(j1.job_id)["output_name"], "sheet_042")
		self.assertEqual(queue.snapshot_response()["jobs"][0]["output_name"], "sheet_042")

	def test_single_result_archive_child_uses_requested_output_name(self):
		self.assertEqual(
			fit_service._result_archive_child_name("selected_segment.tifxyz", 1, "atlas_result_v002.tifxyz"),
			"atlas_result_v002.tifxyz",
		)

	def test_normalize_single_tifxyz_output_rewrites_path_and_meta(self):
		with tempfile.TemporaryDirectory() as td:
			out = fit_service.Path(td)
			child = out / "winding_0000.tifxyz"
			child.mkdir()
			(child / "meta.json").write_text(
				fit_service.json.dumps({"uuid": "winding_0000.tifxyz", "name": "winding_0000.tifxyz"}),
				encoding="utf-8",
			)

			final = fit_service._normalize_single_tifxyz_output(out, "atlas_v003.tifxyz")

			self.assertEqual(final, out / "atlas_v003.tifxyz")
			self.assertFalse(child.exists())
			self.assertTrue(final.is_dir())
			meta = fit_service.json.loads((final / "meta.json").read_text(encoding="utf-8"))
			self.assertEqual(meta["uuid"], "atlas_v003.tifxyz")
			self.assertEqual(meta["name"], "atlas_v003.tifxyz")

	def test_multi_result_archive_keeps_child_names(self):
		self.assertEqual(
			fit_service._result_archive_child_name("layer_0000.tifxyz", 2, "combined.tifxyz"),
			"layer_0000.tifxyz",
		)

	def test_results_archive_rejects_result_symlink(self):
		with tempfile.TemporaryDirectory() as td:
			root = fit_service.Path(td)
			model = root / "model_reopt.pt"
			model.write_bytes(b"checkpoint-bytes")
			out = root / "out"
			out.mkdir()
			seg = out / "atlas_v031.tifxyz"
			seg.mkdir()
			(seg / "meta.json").write_text("{}", encoding="utf-8")
			(seg / "model.pt").symlink_to(model)

			with self.assertRaisesRegex(ValueError, "result contains unsupported symlink: .*model\\.pt ->"):
				fit_service._pack_results_archive(out, "atlas_v031.tifxyz")

	def test_results_archive_rejects_broken_model_symlink(self):
		with tempfile.TemporaryDirectory() as td:
			root = fit_service.Path(td)
			out = root / "out"
			out.mkdir()
			seg = out / "atlas_v032.tifxyz"
			seg.mkdir()
			(seg / "meta.json").write_text("{}", encoding="utf-8")
			(seg / "model.pt").symlink_to(root / "missing_model.pt")

			with self.assertRaisesRegex(ValueError, "result contains unsupported symlink: .*model\\.pt ->"):
				fit_service._pack_results_archive(out, "atlas_v032.tifxyz")

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

	def test_same_hash_different_names_do_not_alias(self):
		with tempfile.TemporaryDirectory() as td:
			old_store = fit_service._object_store_dir
			fit_service._object_store_dir = fit_service.Path(td)
			try:
				model = b"checkpoint"
				model_hash = fit_service._hash_bytes(model)
				model_a = {
					"type": "lasagna_model",
					"name": "sheet_a.tifxyz/model.pt",
					"hash": model_hash,
				}
				model_b = {
					"type": "lasagna_model",
					"name": "sheet_b.tifxyz/model.pt",
					"hash": model_hash,
				}
				for ref in (model_a, model_b):
					fit_service._store_uploaded_object({
						"object": ref,
						"data": fit_service.base64.b64encode(model).decode("ascii"),
					})
				self.assertTrue(fit_service._object_present(model_a))
				self.assertTrue(fit_service._object_present(model_b))
				self.assertNotEqual(
					fit_service._resolve_object_ref(model_a),
					fit_service._resolve_object_ref(model_b),
				)

				files = {"x.tif": b"x", "y.tif": b"y", "z.tif": b"z", "meta.json": b"{}"}
				segment_hash = fit_service._segment_manifest_hash(files)
				segment_a = {
					"type": "tifxyz_segment",
					"name": "surface_a.tifxyz",
					"hash": segment_hash,
				}
				segment_b = {
					"type": "tifxyz_segment",
					"name": "surface_b.tifxyz",
					"hash": segment_hash,
				}
				encoded_files = {
					name: fit_service.base64.b64encode(data).decode("ascii")
					for name, data in files.items()
				}
				for ref in (segment_a, segment_b):
					fit_service._store_uploaded_object({"object": ref, "files": encoded_files})
				self.assertTrue(fit_service._object_present(segment_a))
				self.assertTrue(fit_service._object_present(segment_b))
				self.assertNotEqual(
					fit_service._resolve_object_ref(segment_a),
					fit_service._resolve_object_ref(segment_b),
				)
			finally:
				fit_service._object_store_dir = old_store

	def test_job_spec_resolves_model_and_external_surface_refs(self):
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
						"config": {
							"args": {"model-init": "model"},
							"external_surfaces": [{**seg_ref, "offset": 2.5}],
						},
					}
				})
				self.assertEqual(fit_service.Path(body["model_input"]).read_bytes(), model)
				self.assertEqual(body["config"]["external_surfaces"][0]["offset"], 2.5)
				self.assertTrue(fit_service.Path(body["config"]["external_surfaces"][0]["path"]).is_dir())
				self.assertEqual(body["_job_spec_"]["linked_surfaces"], [seg_ref])
			finally:
				fit_service._object_store_dir = old_store

	def test_job_spec_does_not_synthesize_external_surfaces_from_linked_surfaces(self):
		with tempfile.TemporaryDirectory() as td:
			old_store = fit_service._object_store_dir
			fit_service._object_store_dir = fit_service.Path(td)
			try:
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
						"linked_surfaces": [seg_ref],
						"config": {"args": {"model-init": "seed"}},
					}
				})
				self.assertNotIn("external_surfaces", body["config"])
				self.assertEqual(body["_job_spec_"]["linked_surfaces"], [seg_ref])
			finally:
				fit_service._object_store_dir = old_store

	def test_atlas_objects_upload_and_job_spec_resolution(self):
		with tempfile.TemporaryDirectory() as td:
			old_store = fit_service._object_store_dir
			fit_service._object_store_dir = fit_service.Path(td)
			try:
				line_json = b'{"type":"vc3d_fiber","version":1,"line_points":[[1,2,3]],"control_points":[]}'
				line_ref = {
					"type": "line",
					"name": "fibers/fiber_a.json",
					"hash": fit_service._hash_bytes(line_json),
					"format": "vc3d_fiber_json",
				}
				map_json = b'{"type":"vc3d_atlas_fiber_mapping","version":4,"line_anchors":[]}'
				map_ref = {
					"type": "line-map",
					"name": "atlas/mappings/fibers/fiber_a.json",
					"hash": fit_service._hash_bytes(map_json),
					"format": "vc3d_atlas_fiber_mapping_json",
				}
				base_files = {"x.tif": b"x", "y.tif": b"y", "z.tif": b"z", "meta.json": b"{}"}
				base_ref = {
					"type": "atlas-base",
					"name": "atlas/base_mesh.tifxyz",
					"hash": fit_service._segment_manifest_hash(base_files),
					"format": "tifxyz",
				}
				atlas_obj = {
					"type": "lasagna_atlas",
					"version": 1,
					"name": "atlas",
					"base": {"ref": base_ref, "path": None},
					"metadata": {"zero_winding_column": 0, "period_columns": 3},
					"objects": {"line": [{"id": "fibers/fiber_a.json", "ref": line_ref, "path": None}]},
					"maps": [{
						"object_type": "line",
						"object_id": "fibers/fiber_a.json",
						"object_ref": line_ref,
						"map_ref": map_ref,
						"map_path": None,
						"winding_offset": -2,
					}],
				}
				atlas_json = fit_service.json.dumps(atlas_obj).encode("utf-8")
				atlas_ref = {
					"type": "atlas",
					"name": "atlas/atlas.json",
					"hash": fit_service._hash_bytes(atlas_json),
					"format": "lasagna_atlas_json",
				}
				for ref, data in ((line_ref, line_json), (map_ref, map_json), (atlas_ref, atlas_json)):
					stored = fit_service._store_uploaded_object({
						"object": ref,
						"data": fit_service.base64.b64encode(data).decode("ascii"),
					})
					self.assertEqual({k: stored[k] for k in ("type", "name", "hash")}, {k: ref[k] for k in ("type", "name", "hash")})
				fit_service._store_uploaded_object({
					"object": base_ref,
					"files": {
						name: fit_service.base64.b64encode(data).decode("ascii")
						for name, data in base_files.items()
					},
				})

				body = fit_service._body_with_resolved_job_spec({
					"job_spec": {
						"atlas": atlas_ref,
						"config": {"args": {"model-init": "atlas"}},
					}
				})

				resolved = body["config"]["atlas"]
				self.assertTrue(fit_service.Path(resolved["base"]["path"]).is_dir())
				self.assertTrue(fit_service.Path(resolved["objects"]["line"][0]["path"]).is_file())
				self.assertTrue(fit_service.Path(resolved["maps"][0]["map_path"]).is_file())
				self.assertEqual(resolved["maps"][0]["winding_offset"], -2)
				self.assertEqual(body["_job_spec_"]["atlas"], atlas_ref)
			finally:
				fit_service._object_store_dir = old_store

	def test_model_saved_atlas_ref_overrides_request_atlas_ref(self):
		with tempfile.TemporaryDirectory() as td:
			old_store = fit_service._object_store_dir
			fit_service._object_store_dir = fit_service.Path(td)
			try:
				import torch

				base_files = {"x.tif": b"x", "y.tif": b"y", "z.tif": b"z", "meta.json": b"{}"}
				base_ref = {
					"type": "atlas-base",
					"name": "atlas/base_mesh.tifxyz",
					"hash": fit_service._segment_manifest_hash(base_files),
					"format": "tifxyz",
				}
				fit_service._store_uploaded_object({
					"object": base_ref,
					"files": {
						name: fit_service.base64.b64encode(data).decode("ascii")
						for name, data in base_files.items()
					},
				})
				saved_atlas_obj = {
					"type": "lasagna_atlas",
					"version": 1,
					"name": "saved_atlas",
					"base": {"ref": base_ref, "path": None},
					"metadata": {},
					"objects": {"line": []},
					"maps": [],
				}
				saved_atlas_json = fit_service.json.dumps(saved_atlas_obj).encode("utf-8")
				saved_atlas_ref = {
					"type": "atlas",
					"name": "atlas/saved_atlas.json",
					"hash": fit_service._hash_bytes(saved_atlas_json),
					"format": "lasagna_atlas_json",
				}
				fit_service._store_uploaded_object({
					"object": saved_atlas_ref,
					"data": fit_service.base64.b64encode(saved_atlas_json).decode("ascii"),
				})

				buf = io.BytesIO()
				torch.save({
					"_object_refs_": {
						"version": 1,
						"objects": [base_ref, saved_atlas_ref],
						"job_spec": {"atlas": saved_atlas_ref},
					},
				}, buf)
				model_bytes = buf.getvalue()
				model_ref = {
					"type": "lasagna_model",
					"name": "sheet_v001.tifxyz/model.pt",
					"hash": fit_service._hash_bytes(model_bytes),
				}
				fit_service._store_uploaded_object({
					"object": model_ref,
					"data": fit_service.base64.b64encode(model_bytes).decode("ascii"),
				})
				request_atlas_ref = {
					"type": "atlas",
					"name": "atlas/request_atlas.json",
					"hash": "md5:" + "1" * 32,
					"format": "lasagna_atlas_json",
				}

				body = fit_service._body_with_resolved_job_spec({
					"job_spec": {
						"model": model_ref,
						"atlas": request_atlas_ref,
						"config": {"args": {"model-init": "model"}},
					}
				})

				self.assertEqual(body["_job_spec_"]["atlas"], saved_atlas_ref)
				self.assertEqual(body["config"]["atlas"]["name"], "saved_atlas")
				self.assertNotIn(request_atlas_ref, body["_object_refs_"]["objects"])
			finally:
				fit_service._object_store_dir = old_store

	def test_atlas_pred_snap_object_upload_and_resolution(self):
		with tempfile.TemporaryDirectory() as td:
			old_store = fit_service._object_store_dir
			fit_service._object_store_dir = fit_service.Path(td)
			try:
				snap_json = b'{"type":"vc3d_atlas_pred_snap_points","version":1,"fiber_path":"fibers/fiber_a.json","entries":{}}'
				snap_ref = {
					"type": "atlas-pred-snap",
					"name": "atlas/attachments/pred_snap_points/fiber_a.json",
					"hash": fit_service._hash_bytes(snap_json),
					"format": "vc3d_atlas_pred_snap_points_json",
				}
				stored = fit_service._store_uploaded_object({
					"object": snap_ref,
					"data": fit_service.base64.b64encode(snap_json).decode("ascii"),
				})
				self.assertEqual(stored, snap_ref)
				self.assertTrue(fit_service._object_present(snap_ref))
				self.assertEqual(fit_service._resolve_object_ref(snap_ref).read_bytes(), snap_json)
			finally:
				fit_service._object_store_dir = old_store


if __name__ == "__main__":
	unittest.main()
