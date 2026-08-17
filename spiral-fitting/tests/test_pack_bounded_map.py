"""Packing must not hold the whole store in memory.

read_chunk decodes bricks and the caller writes them to disk one chunk at a
time. Executor.map submits every task up front and keeps each finished result
until the consumer reaches it, so a reader that outruns the writer accumulates
results without bound: packing a lasagna store that way peaked at 26.2 GB and
was killed, and the only way through was --io-threads 1, which throttles the
reads until the pile-up is small.

These tests pin the bound and the ordering, and the control shows the bound is
not something Executor.map provides on its own.
"""
import itertools
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

from pack_resident_pools import map_bounded


class ResidentResultTracker:
    """Counts results produced but not yet consumed."""

    def __init__(self):
        self.lock = threading.Lock()
        self.live = 0
        self.peak = 0

    def produce(self, item):
        with self.lock:
            self.live += 1
            self.peak = max(self.peak, self.live)
        # Long enough that several workers overlap one consumer step.
        time.sleep(0.001)
        return item

    def consume(self):
        with self.lock:
            self.live -= 1


class BoundedMapTests(unittest.TestCase):
    ITEMS = 400
    THREADS = 8
    IN_FLIGHT = 16

    def drain(self, iterator, tracker):
        seen = []
        for value in iterator:
            tracker.consume()
            seen.append(value)
            time.sleep(0.002)      # a writer is slower than a reader
        return seen

    def test_resident_results_stay_within_the_window(self):
        tracker = ResidentResultTracker()
        with ThreadPoolExecutor(self.THREADS) as executor:
            seen = self.drain(
                map_bounded(executor, tracker.produce,
                            range(self.ITEMS), self.IN_FLIGHT),
                tracker)

        self.assertEqual(seen, list(range(self.ITEMS)), 'order must be kept')
        # The window admits one extra: the replacement is submitted before the
        # result is yielded, so between those two statements W+1 tasks have run
        # while the consumer has taken W. Asserting W alone fails about once in
        # forty runs on eight threads and roughly half the time on thirty-two,
        # which would read as a flaky bound rather than as an off-by-one here.
        self.assertLessEqual(
            tracker.peak, self.IN_FLIGHT + 1,
            f'{tracker.peak} results were resident at once for a window of '
            f'{self.IN_FLIGHT}')

    def test_executor_map_does_not_bound_them(self):
        """The control: without the window the whole input piles up."""
        tracker = ResidentResultTracker()
        with ThreadPoolExecutor(self.THREADS) as executor:
            self.drain(
                executor.map(tracker.produce, range(self.ITEMS), chunksize=4),
                tracker)

        self.assertGreater(
            tracker.peak, self.IN_FLIGHT * 4,
            'Executor.map was expected to run far ahead of the consumer; if it '
            'no longer does, this regression test needs rethinking')

    def test_window_smaller_than_one_still_makes_progress(self):
        with ThreadPoolExecutor(2) as executor:
            self.assertEqual(
                list(map_bounded(executor, lambda x: x * 2, range(5), 0)),
                [0, 2, 4, 6, 8])

    def test_empty_input(self):
        with ThreadPoolExecutor(2) as executor:
            self.assertEqual(
                list(map_bounded(executor, lambda x: x, [], 4)), [])

    def test_exceptions_reach_the_caller(self):
        def boom(item):
            if item == 3:
                raise ValueError('boom')
            return item

        with ThreadPoolExecutor(2) as executor:
            with self.assertRaises(ValueError):
                list(map_bounded(executor, boom, range(10), 4))


if __name__ == '__main__':
    unittest.main()
