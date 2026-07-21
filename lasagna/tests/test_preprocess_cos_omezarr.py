import os
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np


common_stub = types.ModuleType("common")
common_stub.load_unet = None
common_stub.unet_infer_tiled = None
sys.modules.setdefault("common", common_stub)

train_stub = types.ModuleType("train_unet_3d")
train_stub.build_model = None
sys.modules.setdefault("train_unet_3d", train_stub)

import preprocess_cos_omezarr as preprocess
from preprocess_cos_omezarr import _grad_mag_factor_from_input_sd


class PreprocessCosOmezarrTests(unittest.TestCase):
	def test_grad_mag_factor_uses_input_scale_not_output_level(self):
		self.assertEqual(_grad_mag_factor_from_input_sd(1), 1.0)
		self.assertEqual(_grad_mag_factor_from_input_sd(4), 0.25)

	def test_release_memmap_pages_does_not_touch_other_slices_or_channels(self):
		page = os.sysconf("SC_PAGE_SIZE")
		with tempfile.NamedTemporaryFile() as backing:
			arr = np.memmap(backing.name, dtype=np.uint8, mode="w+", shape=(7, 12, 1, page))
			arr[:] = 1
			arr._lasagna_tmp_path = backing.name
			base = arr.ctypes.data
			calls = []

			class FakeLibc:
				def madvise(self, addr, length, _advice):
					calls.append((addr.value - base, length.value))
					return 0

				def fallocate(self, *_args):
					return 0

			with mock.patch.object(preprocess, "_get_libc", return_value=FakeLibc()):
				preprocess._release_memmap_pages(arr, 3, 7)
			del arr

		self.assertEqual(len(calls), 7)
		for channel, (offset, length) in enumerate(calls):
			self.assertEqual(offset, (channel * 12 + 3) * page)
			self.assertEqual(length, 4 * page)

	@unittest.skipUnless(sys.platform.startswith("linux"), "requires Linux hole punching")
	def test_release_memmap_pages_reclaims_only_requested_band(self):
		page = os.sysconf("SC_PAGE_SIZE")
		with tempfile.NamedTemporaryFile() as backing:
			shape = (3, 8, 1, page)
			arr = np.memmap(backing.name, dtype=np.uint8, mode="w+", shape=shape)
			arr[:] = 0x5A
			arr.flush()
			arr._lasagna_tmp_path = backing.name

			preprocess._release_memmap_pages(arr, 2, 5)
			arr.flush()
			view = np.memmap(backing.name, dtype=np.uint8, mode="r", shape=shape)
			try:
				self.assertTrue(np.all(view[:, 2:5] == 0))
				self.assertTrue(np.all(view[:, :2] == 0x5A))
				self.assertTrue(np.all(view[:, 5:] == 0x5A))
			finally:
				del view
				del arr

if __name__ == "__main__":
	unittest.main()
