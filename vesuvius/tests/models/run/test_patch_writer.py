import threading

import pytest

from vesuvius.models.run.patch_writer import BoundedPatchWriter


class FailingStore:
    def __init__(self, fail_index):
        self.fail_index = fail_index
        self.written = {}
        self.lock = threading.Lock()

    def __setitem__(self, index, value):
        if index == self.fail_index:
            raise OSError(28, "No space left on device")
        with self.lock:
            self.written[index] = value


@pytest.mark.parametrize("fail_index", [0, 5, 9])
def test_a_failed_write_raises_wherever_it_falls_in_the_run(fail_index):
    store = FailingStore(fail_index)
    with pytest.raises(OSError, match="No space left on device"):
        with BoundedPatchWriter(store, max_workers=1, max_inflight=4) as writer:
            for index in range(10):
                writer.submit(index, index)
    assert fail_index not in store.written


def test_a_run_without_failed_writes_stores_every_patch():
    store = FailingStore(fail_index=None)
    with BoundedPatchWriter(store, max_workers=2) as writer:
        for index in range(10):
            writer.submit(index, index)
    assert store.written == {index: index for index in range(10)}


def test_an_error_raised_inside_the_block_is_not_replaced_by_a_failed_write():
    store = FailingStore(fail_index=0)
    with pytest.raises(KeyboardInterrupt):
        with BoundedPatchWriter(store, max_workers=1) as writer:
            writer.submit(0, 0)
            raise KeyboardInterrupt
