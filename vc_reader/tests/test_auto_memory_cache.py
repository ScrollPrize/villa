"""Adaptive decoded cache configuration uses the native policy in every process."""
import pickle
import unittest

from build_support import load_build
load_build()
from vc_reader.reader.memory_cache import MemoryCache


class AutoMemoryCacheTests(unittest.TestCase):
    def test_fixed_default_is_unchanged(self):
        cache = MemoryCache(4096)
        self.assertFalse(cache.is_auto)
        self.assertEqual(cache.max_bytes, 4096)
        self.assertEqual(cache.nbytes, 0)
        cache.refresh_budget()
        self.assertEqual(cache.max_bytes, 4096)

    def test_auto_default_and_pickle(self):
        cache = MemoryCache("auto")
        self.assertTrue(cache.is_auto)
        self.assertEqual(cache.fraction, .5)
        self.assertEqual(cache.smoothing_seconds, 10)
        self.assertGreater(cache.max_bytes, 0)
        restored = pickle.loads(pickle.dumps(cache))
        self.assertTrue(restored.is_auto)
        self.assertEqual(restored.fraction, .5)
        self.assertEqual(restored.smoothing_seconds, 10)
        self.assertEqual(restored.nbytes, 0)

    def test_custom_fraction_and_shared_pickle(self):
        cache = MemoryCache("auto", fraction=.25, smoothing_seconds=3)
        first, second = pickle.loads(pickle.dumps([cache, cache]))
        self.assertIs(first, second)
        self.assertEqual(first.fraction, .25)
        self.assertEqual(first.smoothing_seconds, 3)
        self.assertTrue(first.is_auto)
        first.refresh_budget()
        self.assertGreater(first.max_bytes, 0)

    def test_invalid_fraction(self):
        for fraction in (0, 1, -1, float("nan"), float("inf")):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                MemoryCache("auto", fraction=fraction)

    def test_fixed_pickle(self):
        restored = pickle.loads(pickle.dumps(MemoryCache(1234)))
        self.assertFalse(restored.is_auto)
        self.assertEqual(restored.max_bytes, 1234)

    def test_smoothing_configuration(self):
        self.assertEqual(MemoryCache("auto", smoothing_seconds=0).smoothing_seconds, 0)
        for seconds in (-1, float("nan"), float("inf")):
            with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                MemoryCache("auto", smoothing_seconds=seconds)


if __name__ == "__main__":
    unittest.main()
