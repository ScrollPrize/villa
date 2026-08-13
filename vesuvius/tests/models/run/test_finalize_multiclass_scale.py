"""Multiclass finalization must quantize on fixed, chunk-independent scales.

`apply_finalization` writes a multiclass volume documented as
`[softmax_c0...softmax_cN, argmax]`. It used to min-max rescale that whole array into
uint8, but the array concatenates softmax probabilities in [0, 1] with an argmax channel
holding class indices in [0, C-1]. The maximum of the array is therefore the largest class
index that happens to appear in the chunk. That value set the scale. Two things broke:

* the same class landed on a different byte in different chunks (class 3 alone in a chunk
  became 255, class 1 alone in another chunk also became 255), so class indices were not
  recoverable and were discontinuous at every chunk boundary; and
* a softmax probability of 1.0 was stored as 85 in a chunk whose largest class index was 3
  and as 255 in a chunk whose largest was 1, so the probabilities were not comparable
  either.

The fix writes the class index raw and puts the probabilities on the same [0, 1] -> [0, 255]
scale the binary path already uses, so a byte means the same thing in every chunk. These
tests pin both halves and fail on the old min-max code. Reported as #1432.
"""

from __future__ import annotations

import numpy as np

from vesuvius.models.run.finalize_outputs import apply_finalization, FinalizeConfig


# --- helpers -----------------------------------------------------------------------

NUM_CLASSES = 4


def chunk_dominated_by(class_idx):
    """A (C, 1, 1, 2) logits chunk: voxel 0 is class_idx, voxel 1 is class 0.

    A large logit makes that class the argmax and drives its softmax to ~1.0.
    """
    logits = np.zeros((NUM_CLASSES, 1, 1, 2), dtype=np.float32)
    logits[class_idx, 0, 0, 0] = 20.0
    logits[0, 0, 0, 1] = 20.0
    return logits


def finalize(logits, **cfg_kwargs):
    cfg = FinalizeConfig(mode="multiclass", **cfg_kwargs)
    out, is_empty = apply_finalization(logits, NUM_CLASSES, cfg)
    assert not is_empty
    return out


# --- the class index must survive, unchanged, in every chunk -----------------------

def test_argmax_channel_holds_the_raw_class_index():
    # Last channel is the argmax. Different chunks contain different top classes.
    out_a = finalize(chunk_dominated_by(3))  # classes {0, 3}
    out_b = finalize(chunk_dominated_by(1))  # classes {0, 1}
    assert out_a[-1].ravel().tolist() == [3, 0]
    assert out_b[-1].ravel().tolist() == [1, 0]


def test_two_classes_do_not_collapse_to_the_same_byte():
    out_a = finalize(chunk_dominated_by(3))
    out_b = finalize(chunk_dominated_by(1))
    class3_byte = out_a[-1, 0, 0, 0]
    class1_byte = out_b[-1, 0, 0, 0]
    assert class3_byte != class1_byte  # min-max stored both as 255


# --- a probability means the same thing in every chunk -----------------------------

def test_probability_of_one_is_255_regardless_of_the_chunk():
    out_a = finalize(chunk_dominated_by(3))
    out_b = finalize(chunk_dominated_by(1))
    # Softmax for the dominant class at voxel 0 is ~1.0 in both chunks.
    assert int(out_a[3, 0, 0, 0]) == 255
    assert int(out_b[1, 0, 0, 0]) == 255


# --- the threshold (argmax-only) path keeps class indices too ----------------------

def test_threshold_multiclass_emits_raw_argmax():
    # A multiclass threshold arrives as 0.5 and emits argmax only (one channel).
    out = finalize(chunk_dominated_by(3), threshold=0.5)
    assert out.shape[0] == 1
    assert out[0].ravel().tolist() == [3, 0]


# --- the refactor must not disturb the binary path ---------------------------------

def test_binary_probabilities_use_the_fixed_scale():
    logits = np.zeros((2, 1, 1, 2), dtype=np.float32)
    logits[1, 0, 0, 0] = 20.0   # foreground prob ~1.0
    logits[0, 0, 0, 1] = 20.0   # foreground prob ~0.0
    out, is_empty = apply_finalization(logits, 2, FinalizeConfig(mode="binary"))
    assert not is_empty
    assert int(out[0, 0, 0, 0]) == 255
    assert int(out[0, 0, 0, 1]) == 0
