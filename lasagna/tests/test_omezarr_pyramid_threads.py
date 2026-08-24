import multiprocessing
import os
import threading
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest import mock

from threadpoolctl import threadpool_info

from omezarr_pyramid import (
	_NATIVE_THREAD_ENV,
	_limit_zarr_codec_threads,
	_pyramid_worker_init,
	_run_pool,
	_single_threaded_native_runtime,
)
import tiled_predict3d


def _worker_thread_state(_value):
	return (
		{name: os.environ.get(name) for name in _NATIVE_THREAD_ENV},
		threadpool_info(),
	)


class OmezarrPyramidThreadTests(unittest.TestCase):
	def test_shared_worker_limiter_caps_zarr_and_blosc(self):
		with (
			mock.patch.dict(os.environ, {}, clear=False),
			mock.patch("omezarr_pyramid.zarr.config.set") as zarr_set,
			mock.patch("omezarr_pyramid.numcodecs.blosc.set_nthreads") as blosc_set,
		):
			limiter = tiled_predict3d._limit_native_worker_threads()
			self.assertEqual(os.environ["BLOSC_NTHREADS"], "1")
			zarr_set.assert_called_once_with({"async.concurrency": 1, "threading.max_workers": 1})
			blosc_set.assert_called_once_with(1)
		del limiter

	def test_pyramid_worker_limiter_caps_zarr_and_blosc(self):
		with (
			mock.patch("omezarr_pyramid.zarr.config.set") as zarr_set,
			mock.patch("omezarr_pyramid.numcodecs.blosc.set_nthreads") as blosc_set,
		):
			_limit_zarr_codec_threads()
			zarr_set.assert_called_once_with({"async.concurrency": 1, "threading.max_workers": 1})
			blosc_set.assert_called_once_with(1)

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

	def test_worker_processes_use_one_native_thread(self):
		method = "fork" if "fork" in multiprocessing.get_all_start_methods() else None
		context = multiprocessing.get_context(method)
		with _single_threaded_native_runtime():
			with context.Pool(processes=2, initializer=_pyramid_worker_init) as pool:
				states = pool.map(_worker_thread_state, range(2))
		for environment, runtimes in states:
			self.assertTrue(all(value == "1" for value in environment.values()))
			for runtime in runtimes:
				threads = runtime.get("num_threads")
				if threads is not None:
					self.assertLessEqual(int(threads), 1, runtime)

	def test_run_pool_restores_limits_and_progress_thread_after_error(self):
		original = os.environ.get("OPENBLAS_NUM_THREADS")

		def fail(_value):
			raise RuntimeError("worker failed")

		with redirect_stdout(StringIO()):
			with self.assertRaisesRegex(RuntimeError, "worker failed"):
				_run_pool([1], fail, workers=1, tag="unit-failure")
		self.assertEqual(os.environ.get("OPENBLAS_NUM_THREADS"), original)
		self.assertFalse(any(t.name == "pyramid-progress:unit-failure" for t in threading.enumerate()))

	def test_run_pool_preserves_requested_process_parallelism(self):
		created = []

		class FakePool:
			def __init__(self, *, processes, initializer):
				created.append((processes, initializer))

			def __enter__(self):
				return self

			def __exit__(self, exc_type, exc, traceback):
				return False

			def imap_unordered(self, worker, work):
				return map(worker, work)

		with mock.patch("omezarr_pyramid.multiprocessing.Pool", FakePool):
			with redirect_stdout(StringIO()):
				_run_pool([1, 2, 3], str, workers=128, tag="unit-size")
		self.assertEqual(created, [(3, _pyramid_worker_init)])


if __name__ == "__main__":
	unittest.main()
