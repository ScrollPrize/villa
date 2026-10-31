import os
import threading
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest import mock

from omezarr_pyramid import (
	_NATIVE_THREAD_ENV,
	_run_pool,
	_single_threaded_native_runtime,
)
import tiled_predict3d


class OmezarrPyramidThreadTests(unittest.TestCase):
	def test_shared_slot_ledger_rejects_duplicate_and_foreign_release(self):
		ledger = tiled_predict3d._SlotLedger(2)
		ledger.acquire("result", 0, band=4, sequence=7, stage="gpu")
		with self.assertRaisesRegex(RuntimeError, "already owned"):
			ledger.acquire("result", 0, band=4, sequence=8, stage="gpu")
		with self.assertRaisesRegex(RuntimeError, "conflicts"):
			ledger.release("result", 0, band=4, sequence=8)
		ledger.transition("result", 0, band=4, sequence=7, stage="accumulating")
		ledger.release("result", 0, band=4, sequence=7)
		ledger.validate_all_free()

	def test_quiescent_state_without_a_producer_fails_immediately(self):
		with self.assertRaisesRegex(RuntimeError, "unresolved work and no producer"):
			tiled_predict3d._coordinator_wait_stage(
				statuses={"awaiting_read": 1}, reads=0, accumulator_tasks=0,
				slot_owners={},
			)
		self.assertEqual(
			tiled_predict3d._coordinator_wait_stage(
				statuses={"assigned": 1}, reads=0, accumulator_tasks=0,
				slot_owners={("input", 0): (4, 7, "gpu")},
			),
			"gpu",
		)

	def test_parent_native_thread_environment_restored_after_error(self):
		original = {name: os.environ.get(name) for name in _NATIVE_THREAD_ENV}
		os.environ["OPENBLAS_NUM_THREADS"] = "7"
		os.environ.pop("VECLIB_MAXIMUM_THREADS", None)
		try:
			with self.assertRaisesRegex(RuntimeError, "worker failed"):
				with _single_threaded_native_runtime():
					self.assertTrue(all(os.environ.get(name) == "1" for name in _NATIVE_THREAD_ENV))
					raise RuntimeError("worker failed")
			self.assertEqual(os.environ.get("OPENBLAS_NUM_THREADS"), "7")
			self.assertNotIn("VECLIB_MAXIMUM_THREADS", os.environ)
		finally:
			for name, value in original.items():
				if value is None:
					os.environ.pop(name, None)
				else:
					os.environ[name] = value

	def test_run_pool_restores_limits_and_progress_thread_after_error(self):
		original = os.environ.get("OPENBLAS_NUM_THREADS")

		def fail(_value):
			raise RuntimeError("worker failed")

		with redirect_stdout(StringIO()):
			with self.assertRaisesRegex(RuntimeError, "worker failed"):
				_run_pool([1], fail, workers=1, tag="unit-failure")
		self.assertEqual(os.environ.get("OPENBLAS_NUM_THREADS"), original)
		self.assertFalse(any(t.name == "pyramid-progress:unit-failure" for t in threading.enumerate()))

	def test_run_pool_preserves_requested_parallelism(self):
		created = []

		class FakePool:
			def __init__(self, *, max_workers):
				created.append(max_workers)

			def __enter__(self):
				return self

			def __exit__(self, exc_type, exc, traceback):
				return False

			def map(self, worker, work):
				return map(worker, work)

		with mock.patch("omezarr_pyramid.ThreadPoolExecutor", FakePool):
			with redirect_stdout(StringIO()):
				_run_pool([1, 2, 3], str, workers=128, tag="unit-size")
		self.assertEqual(created, [3])


if __name__ == "__main__":
	unittest.main()
