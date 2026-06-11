import unittest

from tensorstore_omezarr import (
	_aligned_scalar_scale_job_chunk,
	_first_chunk_job,
	_iter_chunk_jobs_for_z_range,
)


class TensorStoreOmeZarrTests(unittest.TestCase):
	def test_aligned_scalar_jobs_use_source_chunk_when_it_is_larger(self):
		self.assertEqual(
			_aligned_scalar_scale_job_chunk(
				src_chunk=(256, 256, 256),
				dst_chunk=(32, 32, 32),
			),
			(256, 256, 256),
		)

	def test_aligned_scalar_jobs_use_target_chunk_in_source_coords_when_larger(self):
		self.assertEqual(
			_aligned_scalar_scale_job_chunk(
				src_chunk=(64, 64, 64),
				dst_chunk=(128, 128, 128),
			),
			(256, 256, 256),
		)

	def test_aligned_scalar_jobs_reject_non_integral_mapping(self):
		with self.assertRaises(ValueError):
			_aligned_scalar_scale_job_chunk(
				src_chunk=(192, 256, 256),
				dst_chunk=(64, 32, 32),
			)

	def test_first_chunk_job_for_debug_uses_requested_chunk_grid(self):
		self.assertEqual(
			_first_chunk_job((70, 80, 90), (32, 32, 32)),
			(0, 32, 0, 32, 0, 32),
		)

	def test_chunk_job_iterator_is_lazy(self):
		jobs = _iter_chunk_jobs_for_z_range((10_000_000, 10_000_000, 10_000_000), (32, 32, 32), 0, 10_000_000)
		self.assertEqual(next(jobs), (0, 32, 0, 32, 0, 32))


if __name__ == "__main__":
	unittest.main()
