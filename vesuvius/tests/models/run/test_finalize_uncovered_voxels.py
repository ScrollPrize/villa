"""Voxels no patch reached are background, not p = 0.5 (#1114, #1875).

Blending leaves a voxel that no inferred patch covered at logits exactly 0 in
every channel. With two channels ``apply_finalization`` turned that into
softmax 0.5 (uint8 127), and with ``--threshold 0.2`` into foreground (255):
the halo outside the scroll. A single channel is a sigmoid logit, where 0 is a
valid value, so it keeps its old output.
"""

from __future__ import annotations

import numpy as np

from vesuvius.models.run.finalize_outputs import FinalizeConfig, apply_finalization


def _logits() -> np.ndarray:
    """(2, 1, 1, 4): voxels 0-1 uncovered (0, 0); voxels 2-3 carry real logits."""
    logits = np.zeros((2, 1, 1, 4), dtype=np.float32)
    logits[:, 0, 0, 2] = [1.0, 1.0]   # covered, equal logits: p = 0.5 stays 127
    logits[:, 0, 0, 3] = [0.0, 20.0]  # covered, foreground; a single 0 is not "uncovered"
    return logits


def _finalize(logits: np.ndarray, mode: str = "binary", threshold=None) -> np.ndarray:
    out, is_empty = apply_finalization(logits, logits.shape[0], FinalizeConfig(mode=mode, threshold=threshold))
    assert not is_empty
    return out


def test_uncovered_voxels_are_background_with_threshold():
    out = _finalize(_logits(), threshold=0.2)
    assert out[0, 0, 0].tolist() == [0, 0, 255, 255]


def test_uncovered_voxels_are_background_in_probability_map():
    out = _finalize(_logits())
    assert out[0, 0, 0].tolist() == [0, 0, 127, 255]


def test_uncovered_voxels_are_background_in_multiclass():
    probs_and_argmax = _finalize(_logits(), mode="multiclass")
    assert probs_and_argmax[:, 0, 0].tolist() == [[0, 0, 128, 0], [0, 0, 128, 255], [0, 0, 0, 1]]
    argmax = _finalize(_logits(), mode="multiclass", threshold=0.5)
    assert argmax[0, 0, 0].tolist() == [0, 0, 0, 1]


def test_single_channel_zero_logit_is_unchanged():
    logits = np.zeros((1, 1, 1, 3), dtype=np.float32)
    logits[0, 0, 0, 2] = 20.0
    assert _finalize(logits)[0, 0, 0].tolist() == [127, 127, 255]
    assert _finalize(logits, threshold=0.2)[0, 0, 0].tolist() == [255, 255, 255]
