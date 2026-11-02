import importlib
import sys
import unittest
from unittest import mock


class GpuPauseWithoutFcntlTests(unittest.TestCase):
	def _import_gpu_pause(self):
		# ``fcntl`` does not exist on Windows. ``sys.modules[name] = None``
		# makes ``import fcntl`` raise ImportError, as it does there.
		sys.modules.pop("gpu_pause", None)
		with mock.patch.dict(sys.modules, {"fcntl": None}):
			return importlib.import_module("gpu_pause")

	def tearDown(self):
		sys.modules.pop("gpu_pause", None)

	def test_module_imports_and_context_is_a_noop(self):
		module = self._import_gpu_pause()
		self.assertIsNone(module.fcntl)
		with mock.patch.object(module, "_stack_push") as push, \
				mock.patch.object(module, "_find_owner_socket") as find_owner:
			with module.gpu_pause_context() as ctx:
				ctx.check()
			push.assert_not_called()
			find_owner.assert_not_called()

	def test_server_is_inert_and_creates_no_socket_or_stack_file(self):
		module = self._import_gpu_pause()
		with mock.patch.object(module, "_stack_push") as push:
			server = module.GpuPauseServer()
			calls = []
			server.check(lambda: calls.append("offload"), lambda: calls.append("reload"))
			server.close()
		push.assert_not_called()
		self.assertEqual(calls, [])


if __name__ == "__main__":
	unittest.main()
