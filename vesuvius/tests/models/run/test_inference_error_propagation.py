"""A failed inference must surface its original error, not a follow-up symptom.

Inferer.infer() used to catch every exception, print it, and fall through to an
implicit None return. main() then failed unpacking that None ("cannot unpack
non-iterable NoneType object"), reporting a traceback pointing at the wrong
stage — and callers that checked the returned path instead of unpacking
(structure_tensor.create_st) fell through with no error at all. See issue #1360.
"""

from __future__ import annotations

import pytest

from vesuvius.models.run.inference import Inferer


class _BrokenInferer(Inferer):
    def __init__(self):
        self.output_dir = "unused"
        self.part_id = 0
        self.coords_store_path = None

    def _run_inference(self):
        raise RuntimeError("store creation failed")


def test_infer_propagates_run_inference_error() -> None:
    with pytest.raises(RuntimeError, match="store creation failed"):
        _BrokenInferer().infer()
