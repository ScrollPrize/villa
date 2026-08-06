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
	_pyramid_worker_init,
	_run_pool,
	_single_threaded_native_runtime,
)


def _worker_thread_state(_value):
	return (
		{name: os.environ.get(name) for name in _NATIVE_THREAD_ENV},
		threadpool_info(),
	)


class OmezarrPyramidThreadTests(unittest.TestCase):
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
