"""Multiclass finalization must quantise on fixed, chunk-independent scales (#1432).

``apply_finalization`` writes a multiclass volume documented as
``[softmax_c0...softmax_cN, argmax]``. It used to min-max rescale that whole
array into uint8 per chunk; the maximum of the array is the largest class index
present in the chunk, so that index set the scale. Two things broke:

* the same class landed on a different byte in different chunks (class 3 alone
  in a chunk became 255, class 1 alone in another chunk also became 255), so
  class indices were not recoverable and jumped at every chunk boundary;
* a softmax probability of 1.0 was stored as 85 in a chunk whose largest class
  index was 3 and as 255 in a chunk whose largest was 1.

Now the class index is written raw and the probabilities use the same
[0, 1] -> [0, 255] scale as the binary path. Test shapes follow the stale
PR #1433 by @zkasuran, which pinned the same contract.
"""

from __future__ import annotations

import numpy as np
import pytest

from vesuvius.models.run.finalize_outputs import FinalizeConfig, apply_finalization

NUM_CLASSES = 4


def _chunk_dominated_by(class_idx: int) -> np.ndarray:
    """(C, 1, 1, 2) logits: voxel 0 is `class_idx` with ~1.0 probability, voxel 1 is class 0."""
    logits = np.zeros((NUM_CLASSES, 1, 1, 2), dtype=np.float32)
    logits[class_idx, 0, 0, 0] = 20.0
    logits[0, 0, 0, 1] = 20.0
    return logits


def _finalize(logits: np.ndarray, **cfg) -> np.ndarray:
    out, is_empty = apply_finalization(logits, NUM_CLASSES, FinalizeConfig(mode="multiclass", **cfg))
    assert not is_empty
    return out


def test_argmax_channel_holds_the_raw_class_index():
    out_a = _finalize(_chunk_dominated_by(3))  # chunk contains classes {0, 3}
    out_b = _finalize(_chunk_dominated_by(1))  # chunk contains classes {0, 1}
    assert out_a.dtype == np.uint8 and out_b.dtype == np.uint8
    assert out_a[-1].ravel().tolist() == [3, 0]
    assert out_b[-1].ravel().tolist() == [1, 0]
    assert out_a[-1, 0, 0, 0] != out_b[-1, 0, 0, 0]  # min-max stored both as 255


def test_probability_of_one_is_255_in_every_chunk():
    out_a = _finalize(_chunk_dominated_by(3))
    out_b = _finalize(_chunk_dominated_by(1))
    assert int(out_a[3, 0, 0, 0]) == 255
    assert int(out_b[1, 0, 0, 0]) == 255
    assert int(out_a[0, 0, 0, 1]) == 255  # class 0 at voxel 1, both chunks
    assert int(out_b[0, 0, 0, 1]) == 255


def test_probabilities_are_on_the_binary_path_scale():
    logits = np.zeros((NUM_CLASSES, 1, 1, 1), dtype=np.float32)
    logits[:, 0, 0, 0] = [np.log(0.5), np.log(0.25), np.log(0.125), np.log(0.125)]
    out = _finalize(logits)
    assert out[:NUM_CLASSES, 0, 0, 0].tolist() == [128, 64, 32, 32]  # rint(p * 255)
    assert out[-1, 0, 0, 0] == 0


def test_threshold_multiclass_emits_raw_argmax_only():
    out = _finalize(_chunk_dominated_by(3), threshold=0.5)
    assert out.shape[0] == 1
    assert out[0].ravel().tolist() == [3, 0]


def test_chunk_encoding_is_continuous_across_chunks():
    """Two neighbouring chunks with different class populations decode identically."""
    rng = np.random.default_rng(0)
    volume = rng.normal(size=(NUM_CLASSES, 2, 4, 8)).astype(np.float32)
    volume[3, :, :, :4] += 6.0  # left half dominated by class 3, right half mixed {0,1,2}
    volume[3, :, :, 4:] -= 6.0
    left = _finalize(volume[..., :4])
    right = _finalize(volume[..., 4:])
    whole = _finalize(volume)
    np.testing.assert_array_equal(np.concatenate([left, right], axis=-1), whole)


def test_homogeneous_chunk_is_still_skipped():
    logits = np.zeros((NUM_CLASSES, 1, 2, 2), dtype=np.float32)
    logits[2] = 30.0  # every voxel class 2 with probability 1
    out, is_empty = apply_finalization(logits, NUM_CLASSES, FinalizeConfig(mode="multiclass", threshold=0.5))
    assert is_empty and out is None


def test_too_many_classes_for_uint8_is_rejected():
    logits = np.zeros((300, 1, 1, 2), dtype=np.float32)
    logits[7, 0, 0, 0] = 1.0
    with pytest.raises(ValueError, match="uint8"):
        apply_finalization(logits, 300, FinalizeConfig(mode="multiclass"))


def test_cli_main_keeps_its_parser(monkeypatch, tmp_path):
    """`vesuvius.finalize_outputs` crashed with NameError('parser') before reaching
    finalize_logits on main 23adee0: main() built the parser inline and then
    referred to a name it never bound."""
    import sys

    from vesuvius.models.run import finalize_outputs

    calls = {}
    monkeypatch.setattr(finalize_outputs, "finalize_logits", lambda **kw: calls.update(kw))
    monkeypatch.setattr(sys, "argv", ["vesuvius.finalize_outputs", str(tmp_path / "in.zarr"),
                                      str(tmp_path / "out.zarr"), "--mode", "multiclass", "--threshold"])
    assert finalize_outputs.main() == 0
    assert calls["mode"] == "multiclass" and calls["threshold"] == 0.5

    monkeypatch.setattr(sys, "argv", ["vesuvius.finalize_outputs", "in.zarr", "out.zarr",
                                      "--num_parts", "2", "--part_id", "5"])
    with pytest.raises(SystemExit) as excinfo:  # parser.error, not NameError
        finalize_outputs.main()
    assert excinfo.value.code == 2


def test_binary_path_is_unchanged():
    logits = np.zeros((2, 1, 1, 2), dtype=np.float32)
    logits[1, 0, 0, 0] = 20.0
    logits[0, 0, 0, 1] = 20.0
    out, is_empty = apply_finalization(logits, 2, FinalizeConfig(mode="binary"))
    assert not is_empty
    assert out.shape[0] == 1 and out.dtype == np.uint8
    assert out[0].ravel().tolist() == [255, 0]
