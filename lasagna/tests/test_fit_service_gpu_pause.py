import sys
import unittest
from unittest import mock

import fit_service


class GpuPauseOrNullTest(unittest.TestCase):
	def test_disabled_does_not_import_gpu_pause(self):
		# gpu_pause needs fcntl; a None entry makes importing it raise.
		with mock.patch.object(fit_service, "_gpu_pause_enabled", False), \
				mock.patch.dict(sys.modules, {"gpu_pause": None}):
			with fit_service._gpu_pause_or_null():
				pass


if __name__ == "__main__":
	unittest.main()
